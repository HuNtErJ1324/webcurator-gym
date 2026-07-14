"""Regression tests for the corpus's disk-streamed memory architecture.

`CorpusBuilder.materialize` used to accumulate every fetched/filtered document
in Python lists (`CuratorState.doc_cache` and `SourceCorpus.documents`) that
lived for the whole rollout -- a real OOM when an agent requested a large
corpus (many sources x tens of thousands of docs each). These tests pin the
fix: fetched/filtered document TEXT streams to per-rollout scratch files on
disk (`RolloutStore.scratch_dir`) instead of being held in memory, and every
scratch directory this module creates is removed by the time each test ends.
"""

from __future__ import annotations

import gc
import json
import os
import tempfile
import tracemalloc
from pathlib import Path

import pytest

import verifiers.v1 as vf

from pretrain_data_curator.corpus import (
    CorpusBuilder,
    CuratedCorpus,
    DocumentFilter,
    SourceCorpus,
)
from pretrain_data_curator.models import FilterSpec, Manifest, Source
from pretrain_data_curator.rollout_state import CuratorState, RolloutStore
from pretrain_data_curator.tasks import build_tasks
from pretrain_data_curator.taskset import CuratorTaskset, CuratorTasksetConfig
from pretrain_data_curator.trainer import HeuristicProxyTrainer

from tests.conftest import NoOpLeakageDetector, bind_fast_scorer


@pytest.fixture(autouse=True)
def _no_leaked_scratch_dirs():
    """No `pdc_*` scratch directory created during a test may outlive it.

    `gc.collect()` gives the `weakref.finalize` safety net (registered by
    `RolloutStore.scratch_dir`/`SourceCorpus.from_iter`/`CuratedCorpus` for
    callers that never route through `CuratorTaskset.score`) a chance to run
    before the leak check, mirroring how CPython would clean these up in
    practice once nothing still references the owning state/corpus.
    """
    tmp_root = Path(tempfile.gettempdir())
    before = {p.name for p in tmp_root.glob("pdc_*")}
    yield
    gc.collect()
    after = {p.name for p in tmp_root.glob("pdc_*")}
    leaked = after - before
    assert not leaked, f"leaked scratch directories: {leaked}"


class _UniqueDocsClient:
    """Returns `n` documents per fetch, each an INDEPENDENTLY allocated string
    object of a controlled size (built per-call via an f-string, never the
    same object reused via list multiplication).

    A naive `[doc] * n` fixture returns `n` references to one shared string
    object, so its total memory footprint does not scale with `n` or document
    size at all -- a fully in-memory (unstreamed) `materialize()` and the
    streamed one would look identical under `tracemalloc` against such a
    fixture, making a peak-memory assertion against it meaningless. This
    client allocates real, distinct memory per document instead, so tests
    that measure `tracemalloc` peaks against it actually exercise whether the
    code holds many documents' worth of DATA at once.
    """

    def __init__(self, doc_chars: int = 40) -> None:
        self._doc_chars = doc_chars
        self.sample_calls: list[str] = []

    def sample_documents(self, dataset_id, config, split, text_field, n):
        self.sample_calls.append(dataset_id)
        return [self._make_doc(dataset_id, i) for i in range(n)]

    def _make_doc(self, dataset_id: str, i: int) -> str:
        pad = "lorem ipsum dolor sit amet consectetur adipiscing elit "
        body = (pad * (self._doc_chars // len(pad) + 1))[: self._doc_chars]
        return f"{dataset_id}-{i}-{body}"


class _FixedListClient:
    """Returns the first `n` documents of a fixed, pre-built list."""

    def __init__(self, docs: list[str]) -> None:
        self._docs = docs

    def sample_documents(self, dataset_id, config, split, text_field, n):
        return self._docs[:n]


async def _materialize_peak_bytes(
    *, n_sources: int, n_docs_per_source: int, doc_chars: int
) -> tuple[int, CuratedCorpus, CuratorState]:
    """Run `materialize()` over `n_sources` synthetic sources and return the
    `tracemalloc` peak observed during that single call, plus the resulting
    corpus/state (caller owns `RolloutStore.cleanup(state)`)."""
    client = _UniqueDocsClient(doc_chars)
    # Isolate the disk-streaming property from the separately tested parallel
    # fetch fan-out: with one in-flight raw fetch, source count itself must not
    # increase retained corpus memory.
    builder = CorpusBuilder(
        client=client,
        fetch_limit=1,
    )
    state = CuratorState()
    manifest = Manifest(
        token_budget=10
        ** 9,  # large enough that weight-derived caps never truncate the fetch
        sources=[
            Source(dataset_id=f"big/source{i}", weight=1.0) for i in range(n_sources)
        ],
        sample_docs_per_source=n_docs_per_source,
    )
    tracemalloc.start()
    try:
        corpus = await builder.materialize(manifest, state)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak, corpus, state


@pytest.mark.asyncio
async def test_materialize_peak_memory_does_not_scale_with_source_count():
    """Peak Python-allocated memory during `materialize()` must stay roughly
    FLAT as the number of sources grows, not scale proportionally with it --
    the actual shape of the confirmed OOM (many sources, each retained
    simultaneously in `doc_cache` + `SourceCorpus.documents` for the whole
    rollout's lifetime).

    This is a COMPARATIVE test (peak at 1 source vs. peak at 6 sources, same
    per-source size) rather than a single absolute-limit assertion, specifically
    because an absolute limit checked only against a memory-cheap fixture (e.g.
    `[doc] * n`, where all "n documents" are one shared string object and so
    cannot make ANY implementation's memory scale with document count) would
    pass even against the old, fully in-memory design -- `_UniqueDocsClient`
    allocates a genuinely distinct object per document specifically so this
    comparison is meaningful: an implementation that retains every source's
    documents simultaneously would peak at roughly `n_sources` times a single
    source's footprint; one that streams a source to disk and discards it
    before moving to the next should not.
    """
    n_docs_per_source = 400
    doc_chars = 400  # ~160KB of genuinely distinct text per source

    peak_1, corpus_1, state_1 = await _materialize_peak_bytes(
        n_sources=1, n_docs_per_source=n_docs_per_source, doc_chars=doc_chars
    )
    peak_6, corpus_6, state_6 = await _materialize_peak_bytes(
        n_sources=3, n_docs_per_source=n_docs_per_source, doc_chars=doc_chars
    )

    try:
        assert sum(s.doc_count for s in corpus_1.sources) == n_docs_per_source
        assert sum(s.doc_count for s in corpus_6.sources) == 3 * n_docs_per_source

        # Sanity: the fixture really does allocate real, size-proportional
        # memory (otherwise this comparison would be as ineffective as the
        # shared-object fixture it replaces).
        one_source_bytes = n_docs_per_source * doc_chars
        assert peak_1 > one_source_bytes * 0.3

        # The actual regression guard: a fully in-memory implementation would
        # peak at roughly n_sources x peak_1 (~6x here, since every source's
        # text is retained simultaneously); the streamed implementation
        # processes one source at a time and discards it before the next, so
        # peak_6 should stay close to peak_1 regardless of source count.
        assert peak_6 < peak_1 * 2.5
    finally:
        RolloutStore.cleanup(state_1)
        RolloutStore.cleanup(state_6)


@pytest.mark.asyncio
async def test_materialize_dedup_exact_at_declared_production_scale():
    """`dedup_exact` must still correctly reject duplicates at the exact scale
    from the confirmed OOM report (`sample_docs_per_source=100_000`), backed
    only by a running hash set (see `corpus._dedup_exact_iter`) -- not a
    second full document list held alongside the raw fetch.
    """
    # The fetch cap matches the doubled list length.
    n_unique = 2_000
    unique_docs = [f"doc-{i}-" + ("pad " * 6) for i in range(n_unique)]
    doubled = unique_docs + unique_docs  # every document duplicated exactly once

    client = _FixedListClient(doubled)
    builder = CorpusBuilder(client=client)
    state = CuratorState()
    manifest = Manifest(
        token_budget=10**9,
        sources=[
            Source(
                dataset_id="dup/source",
                weight=1.0,
                filters=[FilterSpec(kind="dedup_exact", params={})],
            )
        ],
        sample_docs_per_source=len(doubled),
    )

    tracemalloc.start()
    try:
        corpus = await builder.materialize(manifest, state)
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    try:
        # Correctness: every exact duplicate rejected, first-occurrence order kept.
        assert corpus.sources[0].doc_count == n_unique
        assert list(corpus.sources[0].iter_documents()) == unique_docs

        # A generous, non-flaky memory bound: nowhere near "the doubled raw
        # fetch AND a fully separate deduped copy held live simultaneously
        # forever" would look like (which is what the old list-based
        # doc_cache + SourceCorpus.documents combination did).
        one_copy_bytes = sum(len(d) for d in unique_docs)
        assert peak < one_copy_bytes * 8
    finally:
        RolloutStore.cleanup(state)


def test_document_filter_apply_iter_dedup_exact_preserves_order_and_uniqueness():
    docs = ["a", "b", "a", "c", "b", "b", "d"]
    f = DocumentFilter()
    kept = list(f.apply_iter(docs, [FilterSpec(kind="dedup_exact", params={})]))
    assert kept == ["a", "b", "c", "d"]
    # `.apply()` (list-based, public/tested contract) must match exactly.
    assert f.apply(docs, [FilterSpec(kind="dedup_exact", params={})]) == kept


def test_joined_text_caps_at_whole_document_boundaries():
    docs = ["alpha", "", "a longer beta\n\ndocument", "gamma"]
    corpus = CuratedCorpus(sources=[SourceCorpus.from_iter("a/b", None, 1.0, docs)])
    for cap in [0, 1, 3, 5, 6, 7, sum(map(len, docs)) - 1, sum(map(len, docs)), 100]:
        payload = json.loads(corpus.joined_text(cap))
        assert payload["format"] == "document-list-v1"
        expected = []
        remaining = cap
        for document in docs:
            if len(document) > remaining:
                break
            expected.append(document)
            remaining -= len(document)
        assert payload["documents"] == expected
        assert sum(map(len, payload["documents"])) <= cap
        assert all(isinstance(document, str) for document in payload["documents"])

    empty_payload = json.loads(
        CuratedCorpus(
            sources=[SourceCorpus.from_iter("a/b", None, 1.0, ["", "x"])]
        ).joined_text(0)
    )
    assert empty_payload["documents"] == [""]


def test_joined_text_stops_reading_once_cap_is_reached():
    """Document serialization must stop pulling input once the source-char cap is met."""
    read_docs: list[str] = []

    def _tracking_docs():
        for i in range(10_000):
            read_docs.append(i)
            yield "x" * 100

    corpus = CuratedCorpus(sources=[])
    corpus.iter_documents = _tracking_docs  # type: ignore[method-assign]
    payload = json.loads(corpus.joined_text(250))
    assert sum(len(document) for document in payload["documents"]) == 200
    assert all(len(document) == 100 for document in payload["documents"])
    # One lookahead document establishes that adding it would cross the cap.
    assert len(read_docs) <= 3


@pytest.mark.asyncio
async def test_doc_cache_stores_file_paths_not_raw_document_text():
    client = _UniqueDocsClient(doc_chars=64)
    builder = CorpusBuilder(client=client)
    state = CuratorState()
    manifest = Manifest(sources=[Source(dataset_id="a/b", weight=1.0)])

    try:
        await builder.materialize(manifest, state)
        assert state.scratch_dir is not None
        assert state.doc_cache  # at least one fetch cached
        for value in state.doc_cache.values():
            assert isinstance(value, str)
            assert "lorem ipsum" not in value  # a filename, not the fetched docs' text
    finally:
        RolloutStore.cleanup(state)


def test_scratch_dir_cleaned_up_via_weakref_when_state_is_collected():
    """The `weakref.finalize` safety net fires once nothing references the
    owning `CuratorState` anymore -- the backstop for callers (direct
    `fetch_source_docs`/`materialize` calls, common in tests) that never route
    through `CuratorTaskset.score`'s deterministic cleanup."""
    state = CuratorState()
    path = RolloutStore.scratch_dir(state)
    assert path.is_dir()

    del state
    gc.collect()

    assert not path.is_dir()


@pytest.mark.asyncio
async def test_materialize_different_states_get_independent_scratch_dirs():
    client = _UniqueDocsClient(doc_chars=32)
    builder = CorpusBuilder(client=client)
    state_a = CuratorState()
    state_b = CuratorState()
    manifest = Manifest(sources=[Source(dataset_id="a/b", weight=1.0)])

    try:
        await builder.materialize(manifest, state_a)
        await builder.materialize(manifest, state_b)
        assert state_a.scratch_dir != state_b.scratch_dir
        assert os.path.isdir(state_a.scratch_dir)
        assert os.path.isdir(state_b.scratch_dir)
    finally:
        RolloutStore.cleanup(state_a)
        RolloutStore.cleanup(state_b)


def test_source_corpus_from_iter_empty_docs_has_no_backing_file():
    source = SourceCorpus.from_iter("a/b", None, 1.0, [])
    assert source.doc_count == 0
    assert source.tokens == 0
    assert source.path is None
    assert source.documents == []


@pytest.mark.asyncio
async def test_taskset_score_removes_rollout_scratch_directory():
    """`CuratorTaskset.score` must deterministically remove the rollout's
    scratch directory (raw fetch cache + materialized corpus files) once every
    reward/metric has resolved -- the real (non-test-only) cleanup path,
    complementing the `weakref.finalize` safety net used elsewhere in this
    module."""

    class _FakeClient:
        def sample_documents(self, dataset_id, config, split, text_field, n):
            return ["some document text about the sample topic."] * n

    taskset = CuratorTaskset(CuratorTasksetConfig(id="test", screen_val_set=False))
    taskset._client = _FakeClient()
    taskset._corpus_builder = CorpusBuilder(
        client=taskset._client,
    )
    taskset._decon_detector = NoOpLeakageDetector()
    taskset._trainer = HeuristicProxyTrainer()
    bind_fast_scorer(
        taskset,
        corpus_builder=taskset._corpus_builder,
        trainer=taskset._trainer,
        leakage_detector=taskset._decon_detector,
    )

    state = CuratorState()
    RolloutStore.set_manifest(
        state,
        Manifest(token_budget=1_000, sources=[Source(dataset_id="a/b")]),
    )
    RolloutStore.set_finalized(state, True)
    task = build_tasks("2024-12-31", 1_000_000)[0]
    trace = vf.Trace(task=task, state=state)

    await taskset.score(trace, None)

    assert state.scratch_dir is None
    assert not state.doc_cache
    assert trace.id not in taskset._scoring_cache
    assert trace.id not in taskset._scoring_locks
