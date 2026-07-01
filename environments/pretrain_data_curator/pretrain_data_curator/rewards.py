"""The curation scoring pass.

``CuratorScorer`` computes

    R(M) = alpha_perf * Perf(M) - lambda_cost*Cost(M) - lambda_leakage*Leakage(M)

where Perf(M) defaults to the bounded relative val-loss reduction over a neutral
baseline (``baseline_relative_perf=True``), or falls back to ``exp(-loss)`` for
toy models when ``baseline_relative_perf=False``.

Performance, cost, and leakage derive from one prepared scoring pass over the
finalized manifest. The expensive corpus build and proxy-student training run
happen exactly once per rollout: the taskset wraps this scorer in a per-rollout
lock and cache so concurrent reward/metric evaluation shares the result.
"""

from __future__ import annotations

import asyncio
import logging
import math
from typing import Any

import verifiers.v1 as vf

from .corpus import CorpusBuilder, CuratedCorpus
from .leakage import LeakageDetector
from .models import CuratorConfig
from .rollout_state import CuratorState, RolloutStore
from .trainer import ProxyStudentTrainer, TrainResult

logger = logging.getLogger(__name__)


class CuratorScorer:
    """Heavy curation scoring over a :class:`CuratorState` (framework-agnostic)."""

    def __init__(
        self,
        config: CuratorConfig,
        corpus_builder: CorpusBuilder,
        trainer: ProxyStudentTrainer,
        leakage_detector: LeakageDetector,
    ) -> None:
        self.config = config
        self.corpus_builder = corpus_builder
        self.trainer = trainer
        self.leakage_detector = leakage_detector

    async def compute_scoring(
        self, state: CuratorState, runtime: vf.Runtime | None = None
    ) -> dict[str, Any]:
        manifest = RolloutStore.manifest(state)
        finalized = RolloutStore.is_finalized(state)
        if not finalized or not manifest.sources:
            return self._empty_scoring(state)

        corpus = await self.corpus_builder.materialize(manifest, state)
        train_result = await self._train(corpus, state, runtime)

        ledger = RolloutStore.ledger(state)
        ledger.train_flops += train_result.flops
        # Token cost was already charged once per unique fetch in materialize.
        RolloutStore.set_ledger(state, ledger)

        # Heavy CPU MinHash leakage work stays off the event loop.
        leakage = await asyncio.to_thread(self.leakage_detector.score, corpus.documents)

        return {
            "perf": self._perf(train_result),
            "cost": ledger.total(self.config.prices),
            "leakage": leakage.as_dict(),
            "loss": train_result.loss if math.isfinite(train_result.loss) else 0.0,
            "accuracy": float(train_result.accuracy or 0.0),
            "flops": train_result.flops,
            "tokens": corpus.total_tokens,
            "num_sources": len([s for s in corpus.sources if s.documents]),
            # Always-on baseline-relative diagnostics (zero-weight; never summed
            # into the reward). ``perf_vs_baseline`` is the relative val-loss
            # reduction over ``perf_baseline_loss`` -- a sharper, scale-anchored
            # read on curated-data quality than the raw loss.
            "perf_vs_baseline": self._relative_improvement(train_result),
            "perf_baseline_loss": self.config.perf_baseline_loss,
        }

    async def _train(
        self,
        corpus: CuratedCorpus,
        state: CuratorState,
        runtime: vf.Runtime | None = None,
    ) -> TrainResult:
        """Train the proxy student, degrading external failures to a sentinel.

        A trainer/sandbox failure records typed telemetry and yields a defined
        sentinel (infinite loss -> zero perf, viability not met) so the rollout
        completes and scoring stays deterministic rather than crashing.
        """
        try:
            if self.config.proxy_student.trainer_backend == "docker":
                return await self.trainer.train_and_eval(
                    corpus, self.config.proxy_student, runtime=runtime
                )
            if self.config.proxy_student.trainer_backend == "modal":
                return await self.trainer.train_and_eval(
                    corpus, self.config.proxy_student, runtime=runtime
                )
            return await self.trainer.train_and_eval(corpus, self.config.proxy_student)
        except Exception as exc:  # noqa: BLE001 - surfaced as telemetry + sentinel
            stderr_tail = getattr(exc, "stderr_tail", "")
            message = f"{type(exc).__name__}: {exc}"
            if stderr_tail:
                message = f"{message} | stderr tail: {stderr_tail}"
            logger.warning("[curator] training failed: %s", message)
            RolloutStore.record_tool_error(state, "train")
            RolloutStore.set_external_failure(state, True)
            RolloutStore.set_trainer_error(state, message)
            return TrainResult(
                loss=float("inf"),
                accuracy=0.0,
                flops=0.0,
                tokens_trained=0,
                backend="error",
            )

    def _empty_scoring(self, state: CuratorState) -> dict[str, Any]:
        ledger = RolloutStore.ledger(state)
        return {
            "perf": 0.0,
            "cost": ledger.total(self.config.prices),
            "leakage": {"exact": 0.0, "fuzzy": 0.0, "semantic": 0.0, "overall": 0.0},
            "loss": 0.0,
            "accuracy": 0.0,
            "flops": 0.0,
            "tokens": 0,
            "num_sources": 0,
            "perf_vs_baseline": 0.0,
            "perf_baseline_loss": self.config.perf_baseline_loss,
        }

    def _perf(self, result: TrainResult) -> float:
        """The Perf REWARD term.

        Baseline-relative (bounded relative loss reduction) when
        ``config.baseline_relative_perf`` is set, otherwise ``exp(-loss)``.
        Both variants depend only on held-out cross-entropy loss.
        """
        if self.config.baseline_relative_perf:
            return max(0.0, min(1.0, self._relative_improvement(result)))
        return self._perf_from_result(result)

    @staticmethod
    def _perf_from_result(result: TrainResult) -> float:
        if not math.isfinite(result.loss):
            return 0.0
        return max(0.0, min(1.0, math.exp(-result.loss)))

    def _relative_improvement(self, result: TrainResult) -> float:
        """Relative val-loss reduction over the neutral baseline: ``(baseline -
        loss) / baseline``.

        ``loss == baseline`` -> 0, ``loss -> 0`` -> 1, worse-than-baseline -> < 0,
        and the infinite-loss sentinel (or a non-positive baseline) -> 0.0. Surfaced
        as the always-on diagnostic and, clamped to [0, 1], used as the optional
        baseline-relative reward.
        """
        baseline = self.config.perf_baseline_loss
        if not math.isfinite(result.loss) or baseline <= 0:
            return 0.0
        return (baseline - result.loss) / baseline
