"""The curation scoring pass.

``CuratorScorer`` computes

    R(M) = alpha_perf * Perf(M) - lambda_cost*Cost(M) - lambda_leakage*Leakage(M)

where Leakage(M) is a token-weighted contamination scalar from the decon
Rust n-gram detector run against PUBLIC BENCHMARK eval sets (never the
held-out validation set).

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
from .leakage import DeconError, DeconLeakageDetector, LeakageScores
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
        decon_detector: DeconLeakageDetector | None = None,
    ) -> None:
        self.config = config
        self.corpus_builder = corpus_builder
        self.trainer = trainer
        self.decon_detector = decon_detector

    async def compute_scoring(
        self, state: CuratorState, runtime: vf.Runtime | None = None
    ) -> dict[str, Any]:
        manifest = RolloutStore.manifest(state)
        finalized = RolloutStore.is_finalized(state)
        if not finalized or not manifest.sources:
            return self._empty_scoring(state)

        corpus = await self.corpus_builder.materialize(
            manifest, state, runtime=runtime
        )
        train_result = await self._train(corpus, state, runtime)

        ledger = RolloutStore.ledger(state)
        ledger.train_flops += train_result.flops
        RolloutStore.set_ledger(state, ledger)

        # Decon runs off the event loop via subprocess.
        decon_error = False
        if self.decon_detector is not None:
            try:
                leakage = await asyncio.to_thread(
                    self.decon_detector.score, corpus.iter_documents()
                )
            except DeconError as exc:
                logger.warning("[curator] decon detection failed: %s", exc)
                RolloutStore.record_tool_error(state, "decon")
                RolloutStore.set_external_failure(state, True)
                leakage = LeakageScores(0.0, 0, ())
                decon_error = True
        else:
            leakage = LeakageScores(0.0, 0, ())

        return {
            "perf": self._perf(train_result),
            "cost": ledger.total(self.config.prices),
            "leakage": leakage.as_dict(),
            "decon_error": float(decon_error),
            "loss": train_result.loss if math.isfinite(train_result.loss) else 0.0,
            "accuracy": float(train_result.accuracy or 0.0),
            "flops": train_result.flops,
            "tokens": corpus.total_tokens,
            "num_sources": len([s for s in corpus.sources if s.doc_count]),
            "budget_fill_ratio": state.budget_fill_ratio,
            "perf_vs_baseline": self._relative_improvement(train_result),
            "perf_baseline_loss": self.config.perf_baseline_loss,
        }

    async def _train(
        self,
        corpus: CuratedCorpus,
        state: CuratorState,
        runtime: vf.Runtime | None = None,
    ) -> TrainResult:
        try:
            if self.config.use_real_trainer:
                return await self.trainer.train_and_eval(
                    corpus, self.config.proxy_student, runtime=runtime
                )
            return await self.trainer.train_and_eval(corpus, self.config.proxy_student)
        except Exception as exc:
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
            "leakage": {"leakage_score": 0.0, "num_contaminated_matches": 0},
            "decon_error": 0.0,
            "loss": 0.0,
            "accuracy": 0.0,
            "flops": 0.0,
            "tokens": 0,
            "num_sources": 0,
            "budget_fill_ratio": 0.0,
            "perf_vs_baseline": 0.0,
            "perf_baseline_loss": self.config.perf_baseline_loss,
        }

    def _perf(self, result: TrainResult) -> float:
        if self.config.baseline_relative_perf:
            return max(0.0, min(1.0, self._relative_improvement(result)))
        return self._perf_from_result(result)

    @staticmethod
    def _perf_from_result(result: TrainResult) -> float:
        if not math.isfinite(result.loss):
            return 0.0
        return max(0.0, min(1.0, math.exp(-result.loss)))

    def _relative_improvement(self, result: TrainResult) -> float:
        baseline = self.config.perf_baseline_loss
        if not math.isfinite(result.loss) or baseline <= 0:
            return 0.0
        return (baseline - result.loss) / baseline
