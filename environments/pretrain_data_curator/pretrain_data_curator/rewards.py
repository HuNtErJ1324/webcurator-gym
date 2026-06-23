"""The composite curation reward.

`CuratorRubric` implements

    R(M, H) = a1*Perf(M) + a2*Quality(M, H) + a3*Diversity(M)
              - l1*Cost(M) - l2*Leakage(M)

Perf, Quality, Diversity, Cost, and Leakage all derive from a single prepared
scoring pass over the finalized manifest, cached in state so the (expensive)
corpus build and proxy-student training run happen exactly once per rollout. The
prepare region is guarded by a per-rollout lock with double-checked locking, so
even if verifiers evaluates the rubric functions concurrently the heavy work
runs only once.

The quality and diversity *bonuses* are gated on minimum viability (finalized,
nonempty corpus, no severe leakage, training succeeded); penalties and the base
performance term are always applied.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any

import verifiers as vf

from .corpus import CorpusBuilder, CuratedCorpus
from .leakage import LeakageDetector
from .models import CuratorConfig
from .rollout_state import RolloutStore
from .trainer import ProxyStudentTrainer, TrainResult


class CuratorRubric(vf.Rubric):
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
        # Per-rollout locks guard the (expensive) prepare region; keyed by the
        # rollout's trajectory id so each rollout serializes its own scoring.
        self._scoring_locks: dict[str, asyncio.Lock] = {}
        super().__init__(
            funcs=[
                self.perf_reward,
                self.quality_reward,
                self.diversity_reward,
                self.cost_penalty,
                self.leakage_penalty,
            ],
            weights=[
                config.alpha_perf,
                config.alpha_quality,
                config.alpha_diversity,
                -config.lambda_cost,
                -config.lambda_leakage,
            ],
        )
        for metric in (
            self.perf_loss,
            self.perf_accuracy,
            self.train_flops,
            self.corpus_tokens,
            self.num_sources,
            self.leakage_exact,
            self.leakage_fuzzy,
            self.leakage_semantic,
            self.cost_total,
            self.finalized,
            self.viable,
            # Zero-weight diagnostics that separate "bad curation" from
            # "external/HF/sandbox failure" without affecting reward.
            self.tool_error_count,
            self.external_failure,
        ):
            self.add_metric(metric)

    async def perf_reward(self, state: vf.State) -> float:
        return (await self._prepared(state))["perf"]

    async def quality_reward(self, state: vf.State) -> float:
        return (await self._prepared(state))["quality"]

    async def diversity_reward(self, state: vf.State) -> float:
        return (await self._prepared(state))["diversity"]

    async def cost_penalty(self, state: vf.State) -> float:
        return (await self._prepared(state))["cost"]

    async def leakage_penalty(self, state: vf.State) -> float:
        return (await self._prepared(state))["leakage"]["overall"]

    async def perf_loss(self, state: vf.State) -> float:
        return (await self._prepared(state))["loss"]

    async def perf_accuracy(self, state: vf.State) -> float:
        return (await self._prepared(state))["accuracy"]

    async def train_flops(self, state: vf.State) -> float:
        return (await self._prepared(state))["flops"]

    async def corpus_tokens(self, state: vf.State) -> float:
        return float((await self._prepared(state))["tokens"])

    async def num_sources(self, state: vf.State) -> float:
        return float((await self._prepared(state))["num_sources"])

    async def leakage_exact(self, state: vf.State) -> float:
        return (await self._prepared(state))["leakage"]["exact"]

    async def leakage_fuzzy(self, state: vf.State) -> float:
        return (await self._prepared(state))["leakage"]["fuzzy"]

    async def leakage_semantic(self, state: vf.State) -> float:
        return (await self._prepared(state))["leakage"]["semantic"]

    async def cost_total(self, state: vf.State) -> float:
        return (await self._prepared(state))["cost"]

    async def finalized(self, state: vf.State) -> float:
        return 1.0 if RolloutStore.is_finalized(state) else 0.0

    async def viable(self, state: vf.State) -> float:
        return float((await self._prepared(state))["viable"])

    async def tool_error_count(self, state: vf.State) -> float:
        return float(RolloutStore.tool_error_count(state))

    async def external_failure(self, state: vf.State) -> float:
        return 1.0 if RolloutStore.has_external_failure(state) else 0.0

    def _lock_for(self, state: vf.State) -> asyncio.Lock:
        token = str(state.get("trajectory_id") or id(state))
        lock = self._scoring_locks.get(token)
        if lock is None:
            lock = asyncio.Lock()
            self._scoring_locks[token] = lock
        return lock

    async def _prepared(self, state: vf.State) -> dict[str, Any]:
        # Double-checked locking: the cache is populated exactly once even when
        # many reward/metric funcs await this concurrently for the same rollout.
        cached = state.get(RolloutStore.SCORING)
        if cached is not None:
            return cached
        lock = self._lock_for(state)
        async with lock:
            cached = state.get(RolloutStore.SCORING)
            if cached is not None:
                return cached
            scoring = await self._compute_scoring(state)
            state[RolloutStore.SCORING] = scoring
        # Safe to drop: later callers short-circuit on the populated cache above.
        self._scoring_locks.pop(str(state.get("trajectory_id") or id(state)), None)
        return scoring

    async def _compute_scoring(self, state: vf.State) -> dict[str, Any]:
        manifest = RolloutStore.manifest(state)
        finalized = RolloutStore.is_finalized(state)
        if not finalized or not manifest.sources:
            return self._empty_scoring(state)

        corpus = await self.corpus_builder.materialize(manifest, state)
        train_result = await self._train(corpus, state)

        ledger = RolloutStore.ledger(state)
        ledger.train_flops += train_result.flops
        # Token cost was already charged once per unique fetch in materialize.
        RolloutStore.set_ledger(state, ledger)

        # Heavy CPU work (MinHash leakage + per-doc quality) stays off the loop.
        leakage = await asyncio.to_thread(self.leakage_detector.score, corpus.documents)

        docs = corpus.documents
        training_ok = math.isfinite(train_result.loss) and train_result.backend != "error"
        severe_leak = leakage.overall >= self.config.leakage_severe_threshold
        viable = finalized and bool(docs) and not severe_leak and training_ok

        # Gate the quality/diversity BONUSES on minimum viability; penalties and
        # the base performance term are unaffected.
        if viable:
            quality = await asyncio.to_thread(self._quality, corpus, dict(state))
            diversity = self._diversity(corpus, state)
        else:
            quality = 0.0
            diversity = 0.0

        return {
            "perf": self._perf_from_result(train_result),
            "quality": quality,
            "diversity": diversity,
            "cost": ledger.total(self.config.prices),
            "leakage": leakage.as_dict(),
            "loss": train_result.loss if math.isfinite(train_result.loss) else 0.0,
            "accuracy": float(train_result.accuracy or 0.0),
            "flops": train_result.flops,
            "tokens": corpus.total_tokens,
            "num_sources": len([s for s in corpus.sources if s.documents]),
            "viable": viable,
        }

    async def _train(self, corpus: CuratedCorpus, state: vf.State) -> TrainResult:
        """Train the proxy student, degrading external failures to a sentinel.

        A trainer/sandbox failure records typed telemetry and yields a defined
        sentinel (infinite loss -> zero perf, viability not met) so the rollout
        completes and scoring stays deterministic rather than crashing.
        """
        try:
            return await self.trainer.train_and_eval(corpus, self.config.proxy_student)
        except Exception as exc:  # noqa: BLE001 - surfaced as telemetry + sentinel
            stderr_tail = getattr(exc, "stderr_tail", "")
            message = f"{type(exc).__name__}: {exc}"
            if stderr_tail:
                message = f"{message} | stderr tail: {stderr_tail}"
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

    def _empty_scoring(self, state: vf.State) -> dict[str, Any]:
        ledger = RolloutStore.ledger(state)
        return {
            "perf": 0.0,
            "quality": 0.0,
            "diversity": 0.0,
            "cost": ledger.total(self.config.prices),
            "leakage": {"exact": 0.0, "fuzzy": 0.0, "semantic": 0.0, "overall": 0.0},
            "loss": 0.0,
            "accuracy": 0.0,
            "flops": 0.0,
            "tokens": 0,
            "num_sources": 0,
            "viable": False,
        }

    @staticmethod
    def _perf_from_result(result: TrainResult) -> float:
        if not math.isfinite(result.loss):
            return 0.0
        loss_term = math.exp(-result.loss)
        acc_term = float(result.accuracy or 0.0)
        return max(0.0, min(1.0, 0.5 * loss_term + 0.5 * acc_term))

    def _quality(self, corpus: CuratedCorpus, state: vf.State) -> float:
        docs = corpus.documents
        if not docs:
            return 0.0
        doc_quality = sum(self._doc_quality(d) for d in docs) / len(docs)
        provenance = self._provenance_coverage(corpus, state)
        return max(0.0, min(1.0, 0.7 * doc_quality + 0.3 * provenance))

    @staticmethod
    def _doc_quality(doc: str) -> float:
        if not doc:
            return 0.0
        alpha = sum(1 for c in doc if c.isalpha() or c.isspace()) / len(doc)
        length_term = min(len(doc) / 500.0, 1.0)
        words = doc.split()
        repetition = (len(set(words)) / len(words)) if words else 0.0
        return 0.4 * alpha + 0.3 * length_term + 0.3 * repetition

    @staticmethod
    def _provenance_coverage(corpus: CuratedCorpus, state: vf.State) -> float:
        candidates = RolloutStore.candidates(state)
        if not corpus.sources:
            return 0.0
        known = sum(1 for s in corpus.sources if s.dataset_id in candidates)
        return known / len(corpus.sources)

    def _diversity(self, corpus: CuratedCorpus, state: vf.State) -> float:
        non_empty = [s for s in corpus.sources if s.documents]
        if not non_empty:
            return 0.0
        count_term = min(len(non_empty), 5) / 5.0
        weight_entropy = RolloutStore.manifest(state).weight_entropy()
        tag_term = self._tag_diversity(non_empty, state)
        return max(0.0, min(1.0, 0.5 * count_term + 0.3 * weight_entropy + 0.2 * tag_term))

    @staticmethod
    def _tag_diversity(sources: Any, state: vf.State) -> float:
        candidates = RolloutStore.candidates(state)
        tags: set[str] = set()
        for source in sources:
            entry = candidates.get(source.dataset_id, {})
            for tag in entry.get("tags", []) or []:
                tags.add(str(tag))
        return min(len(tags), 10) / 10.0
