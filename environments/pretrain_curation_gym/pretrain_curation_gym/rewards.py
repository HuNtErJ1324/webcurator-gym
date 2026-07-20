"""The curation scoring pass."""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass

import verifiers.v1 as vf

from .utils.corpus import CorpusBuilder, CuratedCorpus
from .utils.leakage import DeconError, DeconLeakageDetector, LeakageScores
from .utils.models import CuratorConfig, ScoringResult
from .state import CuratorState
from .utils.trainer import ProxyStudentTrainer, TrainResult
from .utils.async_utils import decon_semaphore, run_blocking_drained
from .utils.hf_access import DatasetAccessError
from .utils.val_set import ValTokenLoader

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _LeakageResult:
    scores: LeakageScores
    decon_error: bool = False
    val_screen_skipped: bool = False


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
    ) -> ScoringResult:
        manifest = state.parsed_manifest
        if not state.manifest_finalized or not manifest.sources:
            return self._empty_scoring()

        corpus = await self.corpus_builder.materialize(manifest, state, runtime=runtime)
        train_outcome, leakage_outcome = await asyncio.gather(
            self._train(corpus, state, runtime),
            self._score_leakage(corpus, state),
            return_exceptions=True,
        )
        failures = [
            outcome
            for outcome in (train_outcome, leakage_outcome)
            if isinstance(outcome, BaseException)
        ]
        cancellations = [
            failure
            for failure in failures
            if isinstance(failure, asyncio.CancelledError)
        ]
        if cancellations:
            others = [failure for failure in failures if failure not in cancellations]
            if others:
                raise cancellations[0] from BaseExceptionGroup(
                    "scoring also failed during cancellation", others
                )
            raise cancellations[0]
        if len(failures) == 2:
            raise BaseExceptionGroup("training and leakage scoring failed", failures)
        if failures:
            raise failures[0]
        assert isinstance(leakage_outcome, _LeakageResult)
        assert isinstance(train_outcome, TrainResult)
        leakage_result = leakage_outcome
        train_result = train_outcome

        scores = leakage_result.scores
        return ScoringResult(
            perf=self._perf(train_result),
            leakage_score=round(scores.leakage_score, 6),
            num_contaminated_matches=scores.num_contaminated_matches,
            decon_error=leakage_result.decon_error,
            val_screen_skipped=leakage_result.val_screen_skipped,
            loss=train_result.loss if math.isfinite(train_result.loss) else 0.0,
            accuracy=float(train_result.accuracy or 0.0),
            flops=train_result.flops,
            tokens=corpus.total_tokens,
            num_sources=len([s for s in corpus.sources if s.doc_count]),
            budget_fill_ratio=state.budget_fill_ratio,
            perf_vs_baseline=self._relative_improvement(train_result),
            perf_baseline_loss=self.config.perf_baseline_loss,
            perf_target_loss=self.config.perf_target_loss,
            perf_scaling_exponent=self.config.perf_scaling_exponent,
        )

    async def _score_leakage(
        self,
        corpus: CuratedCorpus,
        state: CuratorState,
    ) -> _LeakageResult:
        if self.decon_detector is None:
            return _LeakageResult(LeakageScores(0.0, 0))

        val_set = None
        val_screen_skipped = False
        if self.screen_val_set and self.val_loader is not None:
            try:
                val_set = await self.val_loader.load()
            except DatasetAccessError:
                logger.warning(
                    "[curator] val set load failed, skipping val leakage screen"
                )
                val_screen_skipped = True

        try:
            async with decon_semaphore(self.config.max_concurrent_training):
                scores = await run_blocking_drained(
                    self.decon_detector.score,
                    corpus.iter_documents(),
                    val_set,
                )
        except DeconError as exc:
            logger.warning("[curator] decon detection failed: %s", exc)
            state.record_error("decon")
            return _LeakageResult(
                LeakageScores(0.0, 0),
                decon_error=True,
                val_screen_skipped=val_screen_skipped,
            )

        return _LeakageResult(
            scores,
            val_screen_skipped=val_screen_skipped,
        )

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
            state.record_error("train")
            state.trainer_error = message
            return TrainResult(
                loss=float("inf"),
                accuracy=0.0,
                flops=0.0,
                tokens_trained=0,
                backend="error",
            )

    def _empty_scoring(self) -> ScoringResult:
        """Zero-signal sentinel; every field default IS the empty outcome."""
        return ScoringResult(
            perf_baseline_loss=self.config.perf_baseline_loss,
            perf_target_loss=self.config.perf_target_loss,
            perf_scaling_exponent=self.config.perf_scaling_exponent,
        )

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
            return p**gamma
        return p
