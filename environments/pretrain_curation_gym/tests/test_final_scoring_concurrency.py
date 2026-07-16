from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any

import pytest

from pretrain_curation_gym.corpus import CuratedCorpus, SourceCorpus
from pretrain_curation_gym.leakage import DeconError, LeakageScores
from pretrain_curation_gym.models import CuratorConfig, Manifest, Source
from pretrain_curation_gym.rewards import CuratorScorer
from pretrain_curation_gym.state import CuratorState
from pretrain_curation_gym.trainer import TrainResult
from pretrain_curation_gym.util.hf_access import DatasetAccessError


_TRAIN_RESULT = TrainResult(
    loss=3.5,
    accuracy=0.625,
    flops=1234.0,
    tokens_trained=16,
    backend="test",
)
_LEAKAGE_RESULT = LeakageScores(0.31234567, 2, ())


class _CorpusBuilder:
    def __init__(self, corpus: CuratedCorpus) -> None:
        self.corpus = corpus
        self.calls = 0

    async def materialize(
        self,
        manifest: Manifest,
        state: CuratorState,
        *,
        runtime: Any = None,
    ) -> CuratedCorpus:
        del manifest, state, runtime
        self.calls += 1
        return self.corpus


class _BarrierTrainer:
    def __init__(
        self,
        barrier: threading.Barrier,
        *,
        failure: BaseException | None = None,
    ) -> None:
        self.barrier = barrier
        self.failure = failure
        self.started = threading.Event()
        self.finished = threading.Event()

    async def train_and_eval(self, corpus, config, *, runtime=None) -> TrainResult:
        del corpus, config, runtime
        self.started.set()
        await asyncio.to_thread(self.barrier.wait, 2.0)
        try:
            if self.failure is not None:
                raise self.failure
            return _TRAIN_RESULT
        finally:
            self.finished.set()


class _BarrierDetector:
    def __init__(
        self,
        barrier: threading.Barrier,
        *,
        failure: BaseException | None = None,
    ) -> None:
        self.barrier = barrier
        self.failure = failure
        self.started = threading.Event()
        self.finished = threading.Event()
        self.documents: list[str] = []
        self.val_set: object | None = None

    def score(self, docs, val_set=None) -> LeakageScores:
        self.started.set()
        self.barrier.wait(timeout=2.0)
        try:
            self.documents = list(docs)
            self.val_set = val_set
            if self.failure is not None:
                raise self.failure
            return _LEAKAGE_RESULT
        finally:
            self.finished.set()


class _ValLoader:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls = 0

    async def load(self) -> object:
        self.calls += 1
        await asyncio.sleep(0)
        return self.result


class _FailingValLoader:
    async def load(self) -> object:
        raise DatasetAccessError(
            "held-out validation unavailable",
            kind="network",
            dataset_id="test/validation",
        )


def _scoring_case(
    tmp_path: Path,
    trainer: Any,
    detector: Any,
    *,
    val_loader: object | None = None,
) -> tuple[CuratorScorer, CuratorState, CuratedCorpus, _CorpusBuilder]:
    source = SourceCorpus.from_docs(
        "test/source",
        None,
        1.0,
        ["alpha document", "beta document"],
        dest_dir=tmp_path,
    )
    corpus = CuratedCorpus([source])
    builder = _CorpusBuilder(corpus)
    scorer = CuratorScorer(
        CuratorConfig(),
        builder,
        trainer,
        detector,
        val_loader=val_loader,  # type: ignore[arg-type]
        screen_val_set=True,
    )
    state = CuratorState()
    state.set_manifest(
        Manifest(
            token_budget=32,
            sources=[Source(dataset_id="test/source")],
        ),
        finalized=True,
    )
    state.set_materialization_stats(
        budget_fill_ratio=0.75,
        source_doc_counts=[source.doc_count],
        source_token_counts=[source.tokens],
    )
    return scorer, state, corpus, builder


@pytest.mark.asyncio
async def test_final_training_and_decon_actually_overlap(tmp_path: Path) -> None:
    # Neither branch can cross this barrier unless both have started. The old
    # sequential implementation deterministically breaks the barrier instead.
    barrier = threading.Barrier(2)
    trainer = _BarrierTrainer(barrier)
    detector = _BarrierDetector(barrier)
    held_out = object()
    val_loader = _ValLoader(held_out)
    scorer, state, corpus, builder = _scoring_case(
        tmp_path,
        trainer,
        detector,
        val_loader=val_loader,
    )

    scoring = await asyncio.wait_for(scorer.compute_scoring(state), timeout=3.0)

    assert trainer.started.is_set() and detector.started.is_set()
    assert trainer.finished.is_set() and detector.finished.is_set()
    assert builder.calls == 1
    assert val_loader.calls == 1
    assert detector.val_set is held_out
    assert detector.documents == ["alpha document", "beta document"]
    assert scoring["loss"] == _TRAIN_RESULT.loss
    assert scoring["accuracy"] == _TRAIN_RESULT.accuracy
    assert scoring["flops"] == _TRAIN_RESULT.flops
    assert scoring["leakage"] == {
        "leakage_score": 0.312346,
        "num_contaminated_matches": 2,
    }
    assert scoring["tokens"] == corpus.total_tokens
    assert scoring["num_sources"] == 1
    assert scoring["budget_fill_ratio"] == 0.75
    assert scoring["decon_error"] == 0.0
    assert scoring["val_screen_skipped"] == 0.0
    assert state.tool_errors == {}


@pytest.mark.asyncio
async def test_training_failure_keeps_concurrent_decon_result(tmp_path: Path) -> None:
    barrier = threading.Barrier(2)
    trainer = _BarrierTrainer(barrier, failure=RuntimeError("gpu failed"))
    detector = _BarrierDetector(barrier)
    scorer, state, _, _ = _scoring_case(tmp_path, trainer, detector)

    scoring = await asyncio.wait_for(scorer.compute_scoring(state), timeout=3.0)

    assert trainer.finished.is_set() and detector.finished.is_set()
    assert scoring["perf"] == 0.0
    assert scoring["loss"] == 0.0
    assert scoring["accuracy"] == 0.0
    assert scoring["flops"] == 0.0
    assert scoring["leakage"] == {
        "leakage_score": 0.312346,
        "num_contaminated_matches": 2,
    }
    assert scoring["decon_error"] == 0.0
    assert state.tool_errors == {"train": 1}
    assert state.external_failure is True
    assert state.trainer_error == "RuntimeError: gpu failed"


@pytest.mark.asyncio
async def test_decon_failure_keeps_concurrent_training_result(tmp_path: Path) -> None:
    barrier = threading.Barrier(2)
    trainer = _BarrierTrainer(barrier)
    detector = _BarrierDetector(barrier, failure=DeconError("decon failed"))
    scorer, state, _, _ = _scoring_case(tmp_path, trainer, detector)

    scoring = await asyncio.wait_for(scorer.compute_scoring(state), timeout=3.0)

    assert trainer.finished.is_set() and detector.finished.is_set()
    assert scoring["loss"] == _TRAIN_RESULT.loss
    assert scoring["accuracy"] == _TRAIN_RESULT.accuracy
    assert scoring["flops"] == _TRAIN_RESULT.flops
    assert scoring["leakage"] == {
        "leakage_score": 0.0,
        "num_contaminated_matches": 0,
    }
    assert scoring["decon_error"] == 1.0
    assert state.tool_errors == {"decon": 1}
    assert state.external_failure is True
    assert state.trainer_error is None


@pytest.mark.asyncio
async def test_val_load_failure_still_runs_decon_without_val_set(
    tmp_path: Path,
) -> None:
    barrier = threading.Barrier(2)
    trainer = _BarrierTrainer(barrier)
    detector = _BarrierDetector(barrier)
    scorer, state, _, _ = _scoring_case(
        tmp_path,
        trainer,
        detector,
        val_loader=_FailingValLoader(),
    )

    scoring = await asyncio.wait_for(scorer.compute_scoring(state), timeout=3.0)

    assert detector.finished.is_set()
    assert detector.val_set is None
    assert scoring["leakage"]["leakage_score"] == 0.312346
    assert scoring["decon_error"] == 0.0
    assert scoring["val_screen_skipped"] == 1.0
    assert state.tool_errors == {}


@pytest.mark.asyncio
async def test_cancellation_drains_decon_worker_before_return(tmp_path: Path) -> None:
    class ImmediateTrainer:
        async def train_and_eval(self, corpus, config, *, runtime=None) -> TrainResult:
            del corpus, config, runtime
            return _TRAIN_RESULT

    class BlockingDetector:
        def __init__(self) -> None:
            self.started = threading.Event()
            self.release = threading.Event()
            self.finished = threading.Event()

        def score(self, docs, val_set=None) -> LeakageScores:
            del docs, val_set
            self.started.set()
            try:
                assert self.release.wait(timeout=2.0)
                return _LEAKAGE_RESULT
            finally:
                self.finished.set()

    detector = BlockingDetector()
    scorer, state, _, _ = _scoring_case(
        tmp_path,
        ImmediateTrainer(),
        detector,
    )
    scoring_task = asyncio.create_task(scorer.compute_scoring(state))
    assert await asyncio.to_thread(detector.started.wait, 1.0)

    scoring_task.cancel()
    await asyncio.sleep(0)
    assert not scoring_task.done()

    detector.release.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(scoring_task, timeout=2.0)
    assert detector.finished.is_set()


@pytest.mark.asyncio
async def test_decon_concurrency_uses_separate_heavy_scoring_bound(
    tmp_path: Path,
) -> None:
    class ImmediateTrainer:
        async def train_and_eval(self, corpus, config, *, runtime=None) -> TrainResult:
            del corpus, config, runtime
            return _TRAIN_RESULT

    class ConcurrencyProbe:
        def __init__(self) -> None:
            self.lock = threading.Lock()
            self.release = threading.Event()
            self.first_started = threading.Event()
            self.second_started = threading.Event()
            self.calls = 0
            self.active = 0
            self.max_active = 0

        def score(self, docs, val_set=None) -> LeakageScores:
            del docs, val_set
            with self.lock:
                self.calls += 1
                call = self.calls
                self.active += 1
                self.max_active = max(self.max_active, self.active)
            (self.first_started if call == 1 else self.second_started).set()
            try:
                assert self.release.wait(timeout=2.0)
                return _LEAKAGE_RESULT
            finally:
                with self.lock:
                    self.active -= 1

    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    detector = ConcurrencyProbe()
    first_scorer, first_state, _, _ = _scoring_case(
        first_dir,
        ImmediateTrainer(),
        detector,
    )
    second_scorer, second_state, _, _ = _scoring_case(
        second_dir,
        ImmediateTrainer(),
        detector,
    )

    first_task = asyncio.create_task(first_scorer.compute_scoring(first_state))
    assert await asyncio.to_thread(detector.first_started.wait, 1.0)
    second_task = asyncio.create_task(second_scorer.compute_scoring(second_state))
    assert not await asyncio.to_thread(detector.second_started.wait, 0.1)

    detector.release.set()
    await asyncio.wait_for(asyncio.gather(first_task, second_task), timeout=2.0)
    assert detector.calls == 2
    assert detector.max_active == 1


@pytest.mark.asyncio
async def test_unexpected_failures_preserve_train_first_precedence(
    tmp_path: Path,
) -> None:
    class FatalTraining(BaseException):
        pass

    class FatalDecon(BaseException):
        pass

    barrier = threading.Barrier(2)
    trainer = _BarrierTrainer(barrier, failure=FatalTraining("training fatal"))
    detector = _BarrierDetector(barrier, failure=FatalDecon("decon fatal"))
    scorer, state, _, _ = _scoring_case(tmp_path, trainer, detector)

    with pytest.raises(FatalTraining, match="training fatal"):
        await asyncio.wait_for(scorer.compute_scoring(state), timeout=3.0)

    assert trainer.finished.is_set() and detector.finished.is_set()
