"""The curation scoring pass.

``CuratorScorer`` computes

    R(M) = alpha_perf * Perf_scaled(M) - lambda_leakage*Leakage(M)

where, by default,

    Perf_scaled(M) = (baseline_loss - loss(M)) / (baseline_loss - target_loss)

so the neutral baseline maps to 0.0, the nanoGPT speedrun target maps to 1.0,
and worse-than-baseline training can make the performance term negative.

where Leakage(M) is a token-weighted contamination scalar from the decon
Rust n-gram detector run against PUBLIC BENCHMARK eval sets AND, optionally,
the held-out validation set (detokenised from GPT-2-BPE token IDs back to
text via tiktoken at scoring time only; the val set is NEVER exposed to the
agent).

Performance, cost (telemetry-only), and leakage derive from one prepared scoring pass over the
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
from .hf_access import DatasetAccessError
from .leakage import DeconError, DeconLeakageDetector, LeakageScores
from .models import CuratorConfig
from .rollout_state import CuratorState, RolloutStore
from .trainer import ProxyStudentTrainer, TrainResult
from .val_set import ValTokenLoader

logger = logging.getLogger(__name__)


class CuratorScorer:
    """Heavy curation scoring over a :class:`CuratorState` (framework-agnostic)."""

    def __init__(
        self,
        config: CuratorConfig,
        corpus_builder: CorpusBuilder,
        trainer: ProxyStudentTrainer,
        decon_detector: DeconLeakageDetector | None = None,
        *,
        val_loader: ValTokenLoader | None = None,
        screen_val_set: bool = True,
    ) -> None:
        self.config = config
        self.corpus_builder = corpus_builder
        self.trainer = trainer
        self.decon_detector = decon_detector
        self.val_loader = val_loader
        self.screen_val_set = screen_val_set

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
        val_screen_skipped = False
        if self.decon_detector is not None:
            # Load the held-out val set for decon screening if configured.
            val_set = None
            if self.screen_val_set and self.val_loader is not None:
                try:
                    val_set = await self.val_loader.load()
                except DatasetAccessError:
                    logger.warning(
                        "[curator] val set load failed, skipping val leakage screen"
                    )
                    val_screen_skipped = True
            try:
                leakage = await asyncio.to_thread(
                    self.decon_detector.score,
                    corpus.iter_documents(),
                    val_set,
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
            "val_screen_skipped": float(val_screen_skipped),
            "loss": train_result.loss if math.isfinite(train_result.loss) else 0.0,
            "accuracy": float(train_result.accuracy or 0.0),
            "flops": train_result.flops,
            "tokens": corpus.total_tokens,
            "num_sources": len([s for s in corpus.sources if s.doc_count]),
            "budget_fill_ratio": state.budget_fill_ratio,
            "perf_vs_baseline": self._relative_improvement(train_result),
            "perf_baseline_loss": self.config.perf_baseline_loss,
            "perf_target_loss": self.config.perf_target_loss,
            "perf_scaling_exponent": self.config.perf_scaling_exponent,
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
                message = f"{message} | training output: {stderr_tail}"
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
            "val_screen_skipped": 0.0,
            "loss": 0.0,
            "accuracy": 0.0,
            "flops": 0.0,
            "tokens": 0,
            "num_sources": 0,
            "budget_fill_ratio": 0.0,
            "perf_vs_baseline": 0.0,
            "perf_baseline_loss": self.config.perf_baseline_loss,
            "perf_target_loss": self.config.perf_target_loss,
            "perf_scaling_exponent": self.config.perf_scaling_exponent,
        }

    def _perf(self, result: TrainResult) -> float:
        if self.config.baseline_relative_perf:
            return self._target_scaled_perf(result)
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

    def _target_scaled_perf(self, result: TrainResult) -> float:
        if not math.isfinite(result.loss):
            return 0.0
        baseline = self.config.perf_baseline_loss
        target = self.config.perf_target_loss
        gamma = self.config.perf_scaling_exponent
        denom = baseline - target
        if denom <= 0:
            raise ValueError(
                "perf_baseline_loss must be greater than perf_target_loss "
                f"(got baseline={baseline}, target={target})"
            )
        p = (baseline - result.loss) / denom
        if p >= 0:
            return p ** gamma
        return p
