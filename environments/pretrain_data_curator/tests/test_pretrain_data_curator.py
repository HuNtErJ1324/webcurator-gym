from __future__ import annotations

import asyncio
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import verifiers.v1 as vf
from pydantic import ValidationError
from verifiers.v1 import graph
from verifiers.v1.decorators import discover_decorated

from pretrain_data_curator.corpus import (
    CorpusBuilder,
    DocumentFilter,
    _iter_sampling,
)
from pretrain_data_curator.eval_corpus import DEFAULT_EVAL_CORPUS
from pretrain_data_curator.hf_access import (
    DatasetAccessError,
    FetchKey,
    HuggingFaceDatasetClient,
    RetryPolicy,
    classify_exception,
    loop_local_semaphore,
    run_blocking_with_retry,
)
from pretrain_data_curator.leakage import LeakageDetector, _stable_hash32
from pretrain_data_curator.models import (
    CuratorConfig,
    FilterSpec,
    Manifest,
    ProxyStudentConfig,
    Source,
)
from pretrain_data_curator.pretrain_data_curator import load_environment
from pretrain_data_curator.rewards import CuratorScorer
from pretrain_data_curator.rollout_state import CuratorState, RolloutStore
from pretrain_data_curator.tasks import build_tasks
from pretrain_data_curator.taskset import (
    SYSTEM_PROMPT,
    CuratorTaskset,
    CuratorTasksetConfig,
    extract_json_object,
    parse_manifest,
)
from pretrain_data_curator import hf_meter
from verifiers.v1.taskset import Taskset
from pretrain_data_curator.trainer import (
    HeuristicProxyTrainer,
    RuntimeSelectedTrainer,
    TrainResult,
)
from pretrain_data_curator.val_set import (
    NANOGPT_VAL_DATASET_ID,
    NANOGPT_VAL_FILENAME,
    NANOGPT_VAL_TOKENS,
    SHARD_HEADER_INTS,
    SHARD_MAGIC,
    SHARD_VERSION,
    ValidationSetConfig,
    ValTokenLoader,
    mean_held_out_ce,
    parse_token_shard,
    plan_val_windows,
)


class FakeClient:
    """In-memory HF stand-in: cutoff-relevant search + canned documents."""

    def __init__(self) -> None:
        self.sample_calls: list[str] = []
        self._docs = {
            "good/encyclopedia": [
                "The Roman Empire was one of the largest empires in ancient history, "
                "spanning three continents at its height.",
                "Volcanoes form when magma from within the Earth's upper mantle works "
                "its way to the surface and erupts.",
            ]
            * 8,
            "good/science": [
                "Newton's laws of motion describe the relationship between a body and "
                "the forces acting upon it, and its motion in response.",
                "DNA carries the genetic instructions used in the growth and "
                "functioning of all known living organisms.",
            ]
            * 8,
            "noisy/symbols": ["$$$ @@@ ### %%% ^^^ &&& !!!"] * 8,
        }

    def sample_documents(self, dataset_id, config, split, text_field, n):
        self.sample_calls.append(dataset_id)
        return list(self._docs.get(dataset_id, []))[:n]


@pytest.fixture(autouse=True)
def _fast_finalize_grace_period(monkeypatch):
    """`CuratorTaskset.finalize` polls for a late-arriving final message (see
    `_await_final_manifest`) before falling back. Shrink the poll interval for
    every test so genuine-fallback tests stay fast; the interval is still long
    enough relative to a bare `asyncio.sleep(0)` for the race-simulation test to
    land its delayed commit inside the grace window."""
    monkeypatch.setattr(CuratorTaskset, "_FINALIZE_GRACE_INTERVAL_SECONDS", 0.01)


# ---------------------------------------------------------------------------
# v1 test seam.
#
# The v0 suite drove a single `load_environment(...)` object that owned both the
# curation tools and the `CuratorRubric`. Under verifiers v1 those are two native
# objects sharing one typed `CuratorState`:
#   - `CuratorHubToolset` (the `@vf.tool` methods) — driven directly in-process,
#     reading/writing the bound `self.state` (`_inert_state` outside an MCP call).
#   - `CuratorTaskset` (the `@vf.reward`/`@vf.metric` methods + the per-rollout
#     `_prepared` cache) — scored over a `vf.Trace`, backed by `CuratorScorer`.
#
# `_Curator` binds a real toolset + taskset over one shared state with the same
# injected in-memory collaborators, so tool previews and final scoring observe a
# single per-rollout document cache + cost ledger — exactly as the v0 env did.
# It exercises the real v1 methods (the forwarders only bind state); it does not
# re-implement any curation/scoring logic.
# ---------------------------------------------------------------------------


class _Curator:
    """In-process driver over a real `CuratorTaskset` and one shared `CuratorState`.

    The agent's deliverable (the curation manifest) is built directly via
    `set_manifest` — the v1 replacement for the retired MCP `set_source` /
    `finalize_manifest` tools — and scoring drives the real `@vf.reward`/`@vf.metric`
    methods over a `vf.Trace` backed by the same injected in-memory collaborators."""

    def __init__(
        self,
        *,
        client=None,
        trainer=None,
        corpus_builder=None,
        leakage_detector=None,
        **cfg,
    ) -> None:
        self.client = client or FakeClient()
        self.taskset = CuratorTaskset(CuratorTasksetConfig(id="test", **cfg))
        # The validated CuratorConfig the reward/tools derive from (== v0 env.config).
        self.config = self.taskset.curator
        # One shared corpus builder so a tool preview and final scoring share the
        # per-rollout document cache + cost ledger (they also share `state`).
        self.corpus_builder = corpus_builder or CorpusBuilder(
            client=self.client,
            sample_docs_per_source=self.config.sample_docs_per_source,
            retry_policy=RetryPolicy(
                attempts=self.config.fetch_max_attempts,
                timeout=self.config.fetch_timeout_seconds,
                per_doc_seconds=self.config.fetch_timeout_per_doc_seconds,
            ),
            fetch_limit=self.config.max_concurrent_fetches,
        )
        self.leakage_detector = leakage_detector or LeakageDetector(
            cfg.get("eval_corpus") or DEFAULT_EVAL_CORPUS
        )
        self.trainer = trainer or HeuristicProxyTrainer()
        # Inject the shared collaborators into the taskset's lazy scoring slots so
        # `_ensure()` builds its scorer from them instead of hitting a live Hub.
        self.taskset._client = self.client
        self.taskset._corpus_builder = self.corpus_builder
        self.taskset._leakage_detector = self.leakage_detector
        self.taskset._trainer = self.trainer
        # This rollout's task (no per-task tool server is built — the taskset
        # exposes no MCP tools; the agent curates via the `hf` CLI in its shell).
        self.task = build_tasks(self.config.cutoff_date, self.config.token_budget)[0]
        self.state = CuratorState()

    async def setup(self) -> "_Curator":
        return self

    async def reset(self) -> CuratorState:
        """Bind a fresh per-rollout state."""
        self.state = CuratorState()
        return self.state

    # -- manifest setup: the agent's deliverable, built directly (no MCP tools) -
    def set_manifest(self, sources, *, finalize=True, weights=None) -> CuratorState:
        srcs = [
            Source(dataset_id=ds, weight=1.0 if weights is None else weights[i])
            for i, ds in enumerate(sources)
        ]
        manifest = Manifest(token_budget=self.config.token_budget, sources=srcs)
        RolloutStore.set_manifest(self.state, manifest)
        RolloutStore.set_finalized(self.state, finalize)
        return self.state

    # -- scoring: drive the real taskset @vf.reward/@vf.metric over a Trace -----
    @property
    def scorer(self) -> CuratorScorer:
        return self.taskset._ensure()

    def trace(self, state=None) -> vf.Trace:
        return vf.Trace(task=self.task, state=self.state if state is None else state)

    async def prepared(self, state=None) -> dict:
        return await self.taskset._prepared(self.trace(state))

    async def score(self, state=None) -> vf.Trace:
        trace = self.trace(state)
        await self.taskset.score(trace, None)
        return trace


async def _make(**kwargs) -> _Curator:
    """Build + set up a `_Curator` (the v1 replacement for the v0 `_env` helper)."""
    return await _Curator(**kwargs).setup()


async def _finalized(
    curator: _Curator, sources=("good/encyclopedia", "good/science")
) -> CuratorState:
    """Set + finalize a manifest of the given sources and return the shared state."""
    return curator.set_manifest(list(sources), finalize=True)


def _scorer(
    trainer, *, config=None, corpus_builder=None, leakage=None
) -> CuratorScorer:
    """A bare `CuratorScorer` (the framework-agnostic half of the old rubric) for
    the degrade/leakage tests that supply their own trainer + leakage detector."""
    return CuratorScorer(
        config or CuratorConfig(),
        corpus_builder or CorpusBuilder(client=FakeClient()),
        trainer,
        leakage or LeakageDetector(DEFAULT_EVAL_CORPUS),
    )


def test_hf_token_is_validated_lazily_at_first_api_use(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    env = load_environment()

    with pytest.raises(RuntimeError, match="HF_TOKEN.*required for rollouts"):
        env.taskset._ensure()


def test_hf_client_accepts_explicit_token_without_environment(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)

    client = HuggingFaceDatasetClient(token="test-token")

    assert client._token == "test-token"
    assert client._allow_script_datasets is False


def test_fetch_key_serializes_auto_text_field_stably():
    assert json.loads(FetchKey("owner/name", None, "train", None, 8).as_str()) == [
        "owner/name",
        None,
        "train",
        "__auto__",
        8,
    ]


def test_hf_client_auto_detects_text_columns_and_query_response(monkeypatch):
    rows = [
        {"wrong": "ignored", "content": "content document"},
        {"text": 42, "passage": "passage document"},
        {"query": "Solve x + 1 = 2.", "response": "x = 1."},
        {"abstract": "", "body": "body document"},
    ]
    monkeypatch.setattr(
        "datasets.load_dataset",
        lambda *args, **kwargs: iter(rows),
    )
    monkeypatch.setattr(
        HuggingFaceDatasetClient,
        "_is_script_dataset",
        lambda self, dataset_id: False,
    )
    client = object.__new__(HuggingFaceDatasetClient)
    client._token = "test-token"
    client._allow_script_datasets = False

    assert client.sample_documents("owner/name", None, "train", None, 4) == [
        "content document",
        "passage document",
        "Solve x + 1 = 2. x = 1.",
        "body document",
    ]
    assert client.sample_documents("owner/name", None, "train", "missing", 4) == [
        "content document",
        "passage document",
        "Solve x + 1 = 2. x = 1.",
        "body document",
    ]


def test_hf_client_resolves_missing_default_config_to_english(monkeypatch):
    calls = []

    def fake_load_dataset(dataset_id, *, name, **kwargs):
        calls.append(name)
        if name is None:
            raise ValueError("Config name is missing. Please pick one.")
        return iter([{"text": "configured document"}])

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    monkeypatch.setattr(
        "datasets.get_dataset_config_names",
        lambda dataset_id, token: ["20231101.ab", "20231101.en"],
    )
    monkeypatch.setattr(
        HuggingFaceDatasetClient,
        "_is_script_dataset",
        lambda self, dataset_id: False,
    )
    client = object.__new__(HuggingFaceDatasetClient)
    client._token = "test-token"
    client._allow_script_datasets = False

    assert client.sample_documents("wikimedia/wikipedia", None, "train", None, 1) == [
        "configured document"
    ]
    assert calls == [None, "20231101.en"]


def test_source_defaults_to_auto_detected_text_field():
    assert Source(dataset_id="owner/name").text_field is None


def test_package_bootstraps_full_v1_over_stale_cached_path():
    code = """
import importlib
import os
import sys
import tempfile
from pathlib import Path

import verifiers

full_paths = list(verifiers.__path__)
with tempfile.TemporaryDirectory() as tmp:
    stale = Path(tmp) / "verifiers"
    (stale / "v1").mkdir(parents=True)
    (stale / "v1" / "__init__.py").write_text("")
    (stale / "v1" / "config.py").write_text("STUB = True\\n")

    for name in [
        key
        for key in sys.modules
        if key == "verifiers.v1" or key.startswith("verifiers.v1.")
    ]:
        del sys.modules[name]
    verifiers.__path__[:] = [os.fspath(stale)]
    stale_v1 = importlib.import_module("verifiers.v1")

    import pretrain_data_curator

    bootstrapped_v1 = importlib.import_module("verifiers.v1")
    importlib.import_module("verifiers.v1.env")
    assert bootstrapped_v1 is not stale_v1
    assert verifiers.__path__[0] in full_paths
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "PYTHONPATH": ""},
        capture_output=True,
        text=True,
    )


def test_load_environment_returns_v1_environment(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-token")

    env = load_environment(candidate_limit=3)

    assert isinstance(env, vf.Environment)
    assert isinstance(env.taskset, CuratorTaskset)
    assert env.taskset.config.candidate_limit == 3
    assert env.harness.config.id == "bash"
    assert env.harness.config.env == {}
    assert env.taskset.load_tasks()


def test_load_environment_plumbs_allow_script_datasets(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-token")

    env = load_environment(allow_script_datasets=True)

    assert env.taskset.config.allow_script_datasets is True
    assert env.taskset.curator.allow_script_datasets is True
    assert env.env_args["allow_script_datasets"] is True


def test_load_environment_uses_declarative_docker_runtime_for_docker_trainer():
    docker_env = load_environment(
        use_real_trainer=True,
        proxy_student={"runtime_backend": "docker", "gpu_count": 1},
    )
    assert docker_env.harness.config.env == {"UV_REINSTALL_PACKAGE": "pydantic-core"}
    runtime = docker_env.harness.config.runtime
    assert isinstance(runtime, vf.DockerConfig)
    assert runtime.image == "pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime"
    assert runtime.workdir == "/workspace"
    assert runtime.gpu == "1"
    assert runtime.cpu == 4.0
    assert runtime.memory == 16.0
    assert runtime.disk == 20.0
    assert docker_env.config.timeout.scoring == 2340.0


def test_load_environment_rejects_remote_docker_host():
    with pytest.raises(ValueError, match="docker_host is not supported"):
        load_environment(
            use_real_trainer=True,
            proxy_student={
                "runtime_backend": "docker",
                "docker_host": "ssh://user@gpu-host",
            },
        )


def test_v1_loader_does_not_import_torch_for_default_load():
    code = """
import json
import sys

from pretrain_data_curator import load_environment

env = load_environment()
print(json.dumps({
    "environment": type(env).__name__,
    "torch_loaded": "torch" in sys.modules,
    "taskset": type(env.taskset).__name__,
}))
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "PYTHONPATH": ""},
        capture_output=True,
        text=True,
    )
    loaded = json.loads(proc.stdout)
    assert loaded == {
        "environment": "Environment",
        "torch_loaded": False,
        "taskset": "CuratorTaskset",
    }


def test_taskset_exposes_no_tools_so_non_mcp_gate_passes():
    # The redesign removes the MCP tool surface: the agent curates via the `hf`
    # CLI in its shell. The taskset must NOT override `Taskset.tools`, so the
    # non-MCP harness gate (env.py:239-247) passes for codex / kimi_code / bash.
    # The gate compares `type(self.taskset).tools is Taskset.tools` on an instance.
    assert CuratorTaskset.tools is Taskset.tools
    taskset = CuratorTaskset(CuratorTasksetConfig(id="test"))
    assert type(taskset).tools is Taskset.tools


@pytest.mark.asyncio
async def test_taskset_setup_fails_fast_when_hf_token_is_not_exported(monkeypatch):
    token_env = "PDC_TEST_HF_TOKEN"
    monkeypatch.delenv(token_env, raising=False)
    taskset = CuratorTaskset(CuratorTasksetConfig(id="test", hf_token_env=token_env))

    with pytest.raises(
        RuntimeError,
        match=r"source secrets\.env.*without `export` or `set -a`",
    ):
        await taskset.setup(
            taskset.load_tasks()[0],
            SimpleNamespace(type="subprocess"),
        )


@pytest.mark.asyncio
async def test_finalize_then_reward_aggregation():
    curator = await _make()
    state = curator.set_manifest(["good/encyclopedia", "good/science"], finalize=True)
    assert RolloutStore.is_finalized(state)

    scoring = await curator.prepared()
    assert 0.0 <= scoring["perf"] <= 1.0
    assert "quality" not in scoring
    assert "diversity" not in scoring
    assert scoring["flops"] > 0.0
    ledger = RolloutStore.ledger(curator.state)
    assert ledger.train_flops > 0.0  # FLOPs charged back to the ledger


@pytest.mark.asyncio
async def test_empty_manifest_scores_zero_perf():
    curator = await _make()
    scoring = await curator.prepared()
    assert scoring["perf"] == 0.0
    assert scoring["num_sources"] == 0


def test_document_filter_kinds():
    docs = [
        "short",
        "a much longer high quality document about science and history",
        "$$$$$",
    ]
    f = DocumentFilter()
    kept = f.apply(docs, [FilterSpec(kind="min_chars", params={"value": 10})])
    assert "short" not in kept
    cleaned = f.apply(
        docs, [FilterSpec(kind="max_symbol_ratio", params={"value": 0.3})]
    )
    assert "$$$$$" not in cleaned


@pytest.mark.asyncio
async def test_corpus_builder_applies_filters_and_sampling():
    client = FakeClient()
    builder = CorpusBuilder(client=client, sample_docs_per_source=16)
    manifest = Manifest(
        sources=[
            Source(
                dataset_id="good/encyclopedia",
                weight=1.0,
                filters=[FilterSpec(kind="min_chars", params={"value": 20})],
                sampling={"max_docs": 3},
            )
        ]
    )
    state = CuratorState()
    corpus = await builder.materialize(manifest, state)
    assert len(corpus.documents) == 3
    assert corpus.total_tokens > 0


@pytest.mark.asyncio
async def test_weight_proportional_sampling_allocates_correct_proportions():
    # Build a client with controlled documents so we can count tokens precisely.
    # Each doc is ~25 chars -> estimate_tokens = 25//4 = 6 tokens.
    doc = "a" * 25  # 6 tokens each
    n_docs = 50

    class _FixedClient(FakeClient):
        def sample_documents(self, dataset_id, config, split, text_field, n):
            return [doc] * min(n, n_docs)

    client = _FixedClient()
    builder = CorpusBuilder(client=client, sample_docs_per_source=n_docs)

    # 2:1 weight ratio with 3000-token budget -> targets: A=2000, B=1000.
    # est_docs: A = 2000//250 = 8, B = 1000//250 = 4 (both well under n_docs=50).
    manifest = Manifest(
        token_budget=3000,
        sources=[
            Source(dataset_id="good/encyclopedia", weight=2.0),
            Source(dataset_id="good/science", weight=1.0),
        ],
    )
    state = CuratorState()
    corpus = await builder.materialize(manifest, state)
    tokens_a = corpus.sources[0].tokens
    tokens_b = corpus.sources[1].tokens

    # Source A fetches 8 docs (48 tokens) <= weight target 2000; B fetches 4 (24 tokens) <= 1000.
    assert tokens_a <= 2000
    assert tokens_b <= 1000
    # Both should have fetched something meaningful.
    assert tokens_a > 0
    assert tokens_b > 0
    # A should have roughly twice as many tokens as B.
    assert tokens_a > tokens_b


@pytest.mark.asyncio
async def test_weight_proportional_explicit_max_tokens_overrides_when_tighter():
    doc = "a" * 25  # 6 tokens each
    n_docs = 50

    class _FixedClient(FakeClient):
        def sample_documents(self, dataset_id, config, split, text_field, n):
            return [doc] * min(n, n_docs)

    client = _FixedClient()
    builder = CorpusBuilder(client=client, sample_docs_per_source=n_docs)

    # Weight-derived target for source A: (2/3) * 3000 = 2000 tokens.
    # Explicit max_tokens=30 is tighter -> effective cap = 30.
    manifest = Manifest(
        token_budget=3000,
        sources=[
            Source(
                dataset_id="good/encyclopedia",
                weight=2.0,
                sampling={"max_tokens": 30},
            ),
            Source(dataset_id="good/science", weight=1.0),
        ],
    )
    state = CuratorState()
    corpus = await builder.materialize(manifest, state)
    # Source A: capped at explicit 30 tokens (tighter than the 2000-token weight target).
    assert corpus.sources[0].tokens <= 30
    # Source B: weight-derived 1000 tokens (no explicit cap); est_docs = 1000//250 = 4 docs (24 tokens).
    assert corpus.sources[1].tokens <= 1000


@pytest.mark.asyncio
async def test_weight_proportional_all_zero_weights_falls_back_to_uncapped():
    doc = "a" * 25  # 6 tokens each
    n_docs = 10

    class _FixedClient(FakeClient):
        def sample_documents(self, dataset_id, config, split, text_field, n):
            return [doc] * min(n, n_docs)

    client = _FixedClient()
    builder = CorpusBuilder(client=client, sample_docs_per_source=n_docs)

    # All sources have weight=0 -> total_weight=0 -> uncapped (current behavior).
    manifest = Manifest(
        token_budget=6,  # tiny budget that would cap everything if proportional
        sources=[
            Source(dataset_id="good/encyclopedia", weight=0.0),
            Source(dataset_id="good/science", weight=0.0),
        ],
    )
    state = CuratorState()
    corpus = await builder.materialize(manifest, state)
    # No weight-derived cap applied -> all 10 docs per source are kept.
    assert len(corpus.sources[0].documents) == n_docs
    assert len(corpus.sources[1].documents) == n_docs


@pytest.mark.asyncio
async def test_zero_weight_source_is_not_fetched_when_other_weights_are_positive():
    client = FakeClient()
    builder = CorpusBuilder(client=client, sample_docs_per_source=10)
    manifest = Manifest(
        token_budget=1_000,
        sources=[
            Source(dataset_id="good/encyclopedia", weight=0.0),
            Source(dataset_id="good/science", weight=1.0),
        ],
    )

    corpus = await builder.materialize(manifest, CuratorState())

    assert client.sample_calls == ["good/science"]
    assert corpus.sources[0].doc_count == 0
    assert corpus.sources[1].doc_count > 0


def test_sampling_continues_after_document_that_exceeds_remaining_budget():
    source = Source(dataset_id="a/b")
    oversized = "x" * 80  # 20 estimated tokens
    fitting = "y" * 16  # 4 estimated tokens

    sampled = list(_iter_sampling([oversized, fitting], source, weight_target=5))

    assert sampled == [fitting]


@pytest.mark.asyncio
async def test_materialize_backfills_unused_budget_from_cached_surplus():
    docs = {
        "filtered/out": ["x" * 1_200, "y" * 1_200],
        "has/surplus": ["a" * 1_200, "b" * 1_200],
    }

    class _BackfillClient(FakeClient):
        def sample_documents(self, dataset_id, config, split, text_field, n):
            self.sample_calls.append(dataset_id)
            return docs[dataset_id][:n]

    client = _BackfillClient()
    builder = CorpusBuilder(client=client, sample_docs_per_source=10)
    manifest = Manifest(
        token_budget=1_000,
        sources=[
            Source(
                dataset_id="filtered/out",
                weight=1.0,
                filters=[FilterSpec(kind="min_chars", params={"value": 2_000})],
            ),
            Source(dataset_id="has/surplus", weight=1.0),
        ],
    )
    state = CuratorState()

    corpus = await builder.materialize(manifest, state)

    assert sorted(client.sample_calls) == ["filtered/out", "has/surplus"]
    assert len(client.sample_calls) == 2  # the backfill made no additional fetch
    assert corpus.sources[0].doc_count == 0
    assert corpus.sources[1].doc_count == 2
    assert corpus.total_tokens == 600
    assert state.budget_fill_ratio == pytest.approx(0.6)
    assert state.source_doc_counts == [0, 2]
    assert state.source_token_counts == [0, 600]


@pytest.mark.asyncio
async def test_materialize_fetches_sources_concurrently():
    import threading
    import time

    lock = threading.Lock()
    active = 0
    max_active = 0

    class _ConcurrentClient(FakeClient):
        def sample_documents(self, dataset_id, config, split, text_field, n):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            time.sleep(0.05)
            with lock:
                active -= 1
            return ["x" * 100]

    manifest = Manifest(
        token_budget=1_000,
        sources=[
            Source(dataset_id="one/source", weight=1.0),
            Source(dataset_id="two/source", weight=1.0),
        ],
    )

    await CorpusBuilder(client=_ConcurrentClient()).materialize(
        manifest,
        CuratorState(),
    )

    assert max_active == 2


@pytest.mark.asyncio
async def test_weight_proportional_single_source_gets_full_budget():
    doc = "a" * 25  # 6 tokens each
    n_docs = 50

    class _FixedClient(FakeClient):
        def sample_documents(self, dataset_id, config, split, text_field, n):
            return [doc] * min(n, n_docs)

    client = _FixedClient()
    builder = CorpusBuilder(client=client, sample_docs_per_source=n_docs)

    # Single source gets 100% of the budget; budget large enough to fetch all n_docs.
    # est_docs = n_docs * 250 // 250 = n_docs, capped at sample_docs_per_source = n_docs.
    manifest = Manifest(
        token_budget=n_docs * 250,  # = 12500; ensures est_docs = n_docs = 50
        sources=[Source(dataset_id="good/encyclopedia", weight=1.0)],
    )
    state = CuratorState()
    corpus = await builder.materialize(manifest, state)
    # All n_docs fetched; their token total (n_docs*6=300) fits within the budget.
    assert corpus.sources[0].tokens <= n_docs * 250
    assert len(corpus.sources[0].documents) == n_docs


@pytest.mark.asyncio
async def test_fetch_count_capped_at_sample_docs_per_source_for_large_target():
    """Large token_target: est_docs hits the sample_docs_per_source cap."""
    doc = "a" * 25  # 6 tokens each
    cap = 8

    class _FixedClient(FakeClient):
        def sample_documents(self, dataset_id, config, split, text_field, n):
            return [doc] * n  # return exactly n (unbounded supply)

    client = _FixedClient()
    builder = CorpusBuilder(client=client, sample_docs_per_source=cap)

    # weight_target = 10_000 -> est_docs = 10_000 // 250 = 40 > cap=8 -> capped to 8.
    manifest = Manifest(
        token_budget=10_000,
        sources=[Source(dataset_id="good/encyclopedia", weight=1.0)],
    )
    state = CuratorState()
    corpus = await builder.materialize(manifest, state)
    assert len(corpus.sources[0].documents) == cap


@pytest.mark.asyncio
async def test_fetch_count_proportional_to_small_token_target():
    """Small token_target: est_docs is proportionally smaller than sample_docs_per_source."""
    doc = "a" * 25  # 6 tokens each
    cap = 100

    class _FixedClient(FakeClient):
        def sample_documents(self, dataset_id, config, split, text_field, n):
            return [doc] * n

    client = _FixedClient()
    builder = CorpusBuilder(client=client, sample_docs_per_source=cap)

    # weight_target = 500 -> est_docs = 500 // 250 = 2, well below cap=100.
    manifest = Manifest(
        token_budget=500,
        sources=[Source(dataset_id="good/encyclopedia", weight=1.0)],
    )
    state = CuratorState()
    corpus = await builder.materialize(manifest, state)
    assert len(corpus.sources[0].documents) == 2
    assert len(corpus.sources[0].documents) < cap


# --- manifest-level `sample_docs_per_source` override (async materialize path) ---


@pytest.mark.asyncio
async def test_materialize_manifest_sample_docs_per_source_overrides_fetch_cap():
    """A manifest-level `sample_docs_per_source` wins over the human-configured
    default for that rollout's fetch-count estimation in `materialize()`."""
    doc = "a" * 25  # 6 tokens each

    class _UnboundedClient(FakeClient):
        def sample_documents(self, dataset_id, config, split, text_field, n):
            return [doc] * n

    client = _UnboundedClient()
    builder = CorpusBuilder(client=client, sample_docs_per_source=8)
    state = CuratorState()

    # weight_target = 10_000 -> est_docs = 40, capped at the manifest's override
    # (20), NOT the builder's configured default (8).
    manifest = Manifest(
        token_budget=10_000,
        sources=[Source(dataset_id="good/encyclopedia", weight=1.0)],
        sample_docs_per_source=20,
    )
    corpus = await builder.materialize(manifest, state)
    assert len(corpus.sources[0].documents) == 20


@pytest.mark.asyncio
async def test_materialize_without_manifest_override_falls_back_to_configured_default():
    """Backward compat: a manifest with `sample_docs_per_source=None` (the
    default) leaves fetch-count estimation unchanged."""
    doc = "a" * 25

    class _UnboundedClient(FakeClient):
        def sample_documents(self, dataset_id, config, split, text_field, n):
            return [doc] * n

    client = _UnboundedClient()
    builder = CorpusBuilder(client=client, sample_docs_per_source=8)
    state = CuratorState()

    manifest = Manifest(
        token_budget=10_000,
        sources=[Source(dataset_id="good/encyclopedia", weight=1.0)],
    )
    assert manifest.sample_docs_per_source is None
    corpus = await builder.materialize(manifest, state)
    assert len(corpus.sources[0].documents) == 8


@pytest.mark.parametrize("value", [0, -1, 100_001])
def test_manifest_sample_docs_per_source_bounds_rejected(value):
    with pytest.raises(ValidationError):
        Manifest(sources=[Source(dataset_id="a/b")], sample_docs_per_source=value)


def test_manifest_sample_docs_per_source_bounds_accepted():
    assert (
        Manifest(
            sources=[Source(dataset_id="a/b")], sample_docs_per_source=1
        ).sample_docs_per_source
        == 1
    )
    assert (
        Manifest(
            sources=[Source(dataset_id="a/b")], sample_docs_per_source=100_000
        ).sample_docs_per_source
        == 100_000
    )


@pytest.mark.asyncio
async def test_materialize_different_sample_sizes_do_not_share_cache_key():
    """Two materializations over the same rollout state requesting different
    `sample_docs_per_source` for the same source must each hit the client (the
    `FetchKey.n` component must differ, so they cannot collide on one cache
    entry and silently reuse the wrong-sized fetch)."""
    call_ns: list[int] = []

    class _RecordingClient(FakeClient):
        def sample_documents(self, dataset_id, config, split, text_field, n):
            call_ns.append(n)
            return ["a" * 25] * n

    client = _RecordingClient()
    builder = CorpusBuilder(client=client, sample_docs_per_source=8)
    state = CuratorState()

    manifest_small = Manifest(
        token_budget=10_000,
        sources=[Source(dataset_id="good/encyclopedia", weight=1.0)],
        sample_docs_per_source=5,
    )
    manifest_large = Manifest(
        token_budget=10_000,
        sources=[Source(dataset_id="good/encyclopedia", weight=1.0)],
        sample_docs_per_source=20,
    )
    corpus_small = await builder.materialize(manifest_small, state)
    corpus_large = await builder.materialize(manifest_large, state)

    # Both fetches actually hit the client -- neither was served from a cache
    # entry keyed without the effective cap.
    assert call_ns == [5, 20]
    assert len(corpus_small.sources[0].documents) == 5
    assert len(corpus_large.sources[0].documents) == 20


def test_leakage_detects_exact_and_paraphrase():
    eval_docs = [
        "The mitochondrion is the powerhouse of the cell and provides energy.",
        "Binary search finds a target value within a sorted array efficiently.",
    ]
    detector = LeakageDetector(eval_docs, seed=0)
    clean = detector.score(["Unrelated text about gardening and the weather today."])
    contaminated = detector.score([eval_docs[0]])
    assert contaminated.overall > clean.overall
    assert contaminated.exact == 1.0


# ---------------------------------------------------------------------------
# Shared fakes for the robustness/concurrency tests.
# ---------------------------------------------------------------------------


class FailingClient(FakeClient):
    """FakeClient whose document sampling raises a configured exception.

    Search still succeeds, so candidates can be discovered; only `sample_documents`
    (the corpus fetch path) fails, isolating external-failure handling.
    """

    def __init__(self, exc_factory) -> None:
        super().__init__()
        self._exc_factory = exc_factory

    def sample_documents(self, dataset_id, config, split, text_field, n):
        raise self._exc_factory()


# --- Tier G: pydantic bounds + cross-field validation ----------------------


def test_config_valid_defaults_and_overrides():
    cfg = CuratorConfig(candidate_limit=4, scan_limit=10)
    assert cfg.scan_limit >= cfg.candidate_limit
    assert ProxyStudentConfig(n_embd=128, n_head=4).n_embd == 128


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scan_limit": 5, "candidate_limit": 10},  # cross-field: scan < candidate
        {"token_budget": 0},
        {"max_turns": 0},
        {"fetch_max_attempts": 0},
        {"max_concurrent_fetches": 0},
    ],
)
def test_curator_config_rejects_invalid(kwargs):
    with pytest.raises(ValidationError):
        CuratorConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"val_fraction": 1.5},
        {"val_fraction": 0.0},
        {"block_size": 0},
        {"steps": 0},
        {"n_embd": 10, "n_head": 4},  # n_embd not divisible by n_head
        {"n_embd": 8, "n_head": 4},  # head_dim 2 not a multiple of 4 (RoPE)
        {"n_layer": 3},  # odd depth breaks the symmetric U-net skips
        {"mlp_ratio": 0},
        {"lm_head_softcap": 0.0},
        {"learning_rate": 0.0},
    ],
)
def test_proxy_student_config_rejects_invalid(kwargs):
    with pytest.raises(ValidationError):
        ProxyStudentConfig(**kwargs)


# --- Tier D2: adjustable token budget -> steps / corpus cap / timeout -------


def test_train_token_budget_default_preserves_step_behavior():
    # Default (budget None) keeps the historical steps-driven behavior EXACTLY,
    # so default / CPU / heuristic runs stay cheap and unchanged.
    cfg = ProxyStudentConfig()  # steps=200, batch=16, block=256
    assert cfg.train_token_budget is None
    assert cfg.effective_steps == 200
    assert cfg.effective_train_tokens == 200 * 16 * 256  # 819_200


@pytest.mark.parametrize(
    "budget,batch,block,expected_steps",
    [
        (819_200, 16, 256, 200),  # exactly the default budget -> 200 steps
        (300_000_000, 16, 256, 73_243),  # ceil(300M / 4096)
        (1_000_000_000, 16, 256, 244_141),  # ceil(1e9 / 4096)
        (10, 1, 8, 2),  # ceil(10/8) rounds up, never truncates
    ],
)
def test_train_token_budget_derives_steps(budget, batch, block, expected_steps):
    cfg = ProxyStudentConfig(
        train_token_budget=budget, batch_size=batch, block_size=block
    )
    assert cfg.effective_steps == expected_steps
    assert cfg.effective_train_tokens == expected_steps * batch * block


@pytest.mark.parametrize("budget", [0, 1_000_000_001])
def test_train_token_budget_out_of_bounds_rejected(budget):
    with pytest.raises(ValidationError):
        ProxyStudentConfig(train_token_budget=budget)


def test_train_token_budget_max_is_accepted():
    assert ProxyStudentConfig(train_token_budget=1_000_000_000).train_token_budget == (
        1_000_000_000
    )


def test_effective_max_corpus_chars_scales_with_budget():
    # Small/default budget keeps the historical 5M cap (floor); a large budget
    # grows the cap so a few-hundred-M-token run is not capped at ~1.25M unique
    # tokens; an explicit value overrides; and the cap is ceilinged.
    assert ProxyStudentConfig().effective_max_corpus_chars == 5_000_000
    big = ProxyStudentConfig(train_token_budget=300_000_000)
    assert big.effective_max_corpus_chars == 4 * big.effective_train_tokens
    assert big.effective_max_corpus_chars > 5_000_000
    assert (
        ProxyStudentConfig(max_corpus_chars=123_456).effective_max_corpus_chars
        == 123_456
    )
    # 1e9 tokens * 4 chars/token exceeds the 2e9 ceiling -> clamped.
    assert ProxyStudentConfig(
        train_token_budget=1_000_000_000
    ).effective_max_corpus_chars == (2_000_000_000)


def test_effective_timeout_minutes_scales_and_is_bounded():
    # Default budget keeps the historical 30-minute timeout; a large budget grows
    # it; an explicit value overrides. Modal caps the derived timeout at its 24h
    # (1440-minute) platform sandbox limit and rejects an explicit value above
    # it; docker (and no runtime_backend set) has no such ceiling.
    assert ProxyStudentConfig().effective_timeout_minutes == 30
    big = ProxyStudentConfig(train_token_budget=300_000_000)
    assert 30 < big.effective_timeout_minutes <= 1440
    assert ProxyStudentConfig(timeout_minutes=45).effective_timeout_minutes == 45
    huge_docker = ProxyStudentConfig(train_token_budget=1_000_000_000)
    assert huge_docker.effective_timeout_minutes > 1440
    huge_modal = ProxyStudentConfig(
        train_token_budget=1_000_000_000, runtime_backend="modal"
    )
    assert huge_modal.effective_timeout_minutes == 1440
    assert ProxyStudentConfig(timeout_minutes=1441).effective_timeout_minutes == 1441
    with pytest.raises(ValidationError):
        ProxyStudentConfig(runtime_backend="modal", timeout_minutes=1441)


# --- Tier B: process-stable hash for fuzzy leakage -------------------------


def test_stable_hash_is_pinned_constant():
    # Pinned value: proves the shingle hash does not depend on PYTHONHASHSEED.
    assert _stable_hash32("the quick brown fox jumps") == 2016863831


def test_leakage_fuzzy_is_cross_process_deterministic():
    doc = "the quick brown fox jumps over the lazy dog near the river bank"
    code = (
        "from pretrain_data_curator.leakage import LeakageDetector;"
        "d=LeakageDetector(['%s'], seed=0);"
        "print(round(d.score(['%s']).fuzzy, 10))" % (doc, doc)
    )
    in_proc = round(LeakageDetector([doc], seed=0).score([doc]).fuzzy, 10)
    outputs = set()
    for seed in ("0", "1", "987654"):
        proc = subprocess.run(
            [sys.executable, "-c", code],
            env={"PYTHONHASHSEED": seed, "PATH": "/usr/bin:/bin"},
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
        outputs.add(proc.stdout.strip())
    assert outputs == {str(in_proc)}


# --- Tier C: per-task token budget seeds the parsed manifest ----------------
#
# The per-task `CuratorTask.token_budget` seeds the manifest when the agent's
# emitted JSON omits a `token_budget`, and is overridden when the agent supplies
# one. `parse_manifest(..., default_token_budget=task.token_budget)` is the seam.


def test_task_token_budget_seeds_manifest_when_agent_omits_it():
    manifest = parse_manifest(
        '```json\n{"sources": [{"id": "good/encyclopedia", "weight": 1.0}]}\n```',
        default_token_budget=555,
    )
    assert manifest is not None
    assert manifest.token_budget == 555


def test_agent_token_budget_overrides_task_default():
    manifest = parse_manifest(
        '{"token_budget": 4242, "sources": [{"id": "good/encyclopedia", "weight": 1.0}]}',
        default_token_budget=123456,
    )
    assert manifest is not None
    assert manifest.token_budget == 4242


def test_build_tasks_carry_typed_token_budget():
    # The typed per-task field that replaced the v0 `info` override.
    tasks = build_tasks("2024-12-31", 777)
    assert tasks and all(t.token_budget == 777 for t in tasks)
    assert all(t.cutoff_date == "2024-12-31" for t in tasks)


# --- Tier E: deterministic same-key cache; preview == score; cost once ------


@pytest.mark.asyncio
async def test_fetch_cache_same_key_identity_and_cost_once():
    client = FakeClient()
    curator = await _make(client=client)
    state = await _finalized(curator, sources=("good/encyclopedia",))
    assert client.sample_calls == []  # nothing fetched until materialize
    tokens_before = RolloutStore.ledger(state).tokens
    hub_before = RolloutStore.ledger(state).hub_calls

    manifest = RolloutStore.manifest(state)
    corpus_a = await curator.corpus_builder.materialize(manifest, state)
    assert client.sample_calls == ["good/encyclopedia"]  # fetched exactly once
    tokens_once = RolloutStore.ledger(state).tokens
    # The fetch charged one hub call and a positive token cost, exactly once.
    assert tokens_once > tokens_before
    assert RolloutStore.ledger(state).hub_calls == hub_before + 1

    corpus_b = await curator.corpus_builder.materialize(manifest, state)
    # No re-streaming on repeated same-key fetches.
    assert client.sample_calls == ["good/encyclopedia"]
    # Identical docs across fetches (preview == score).
    assert corpus_a.documents == corpus_b.documents
    # Token/corpus cost counted exactly once (no re-billing from cached builds).
    assert RolloutStore.ledger(state).tokens == tokens_once
    assert RolloutStore.ledger(state).hub_calls == hub_before + 1


@pytest.mark.asyncio
async def test_materialize_preview_and_scoring_observe_same_docs():
    # A `materialize` preview and the final scoring share one per-rollout doc cache:
    # scoring reuses the cached docs, with no extra Hub fetches.
    client = FakeClient()
    curator = await _make(client=client)
    state = await _finalized(curator, sources=("good/encyclopedia", "good/science"))
    await curator.corpus_builder.materialize(RolloutStore.manifest(state), state)
    calls_after_preview = list(client.sample_calls)
    await curator.prepared()
    assert client.sample_calls == calls_after_preview


@pytest.mark.asyncio
async def test_real_taskset_finalize_and_scoring_share_one_rollout_state():
    # Go through the REAL finalize -> score path: the agent's final-message manifest
    # is parsed by `finalize`, written to the single per-rollout CuratorState, then
    # `score` materializes that manifest's sources and trains over the SAME state.
    client = FakeClient()
    taskset = CuratorTaskset(CuratorTasksetConfig(id="test", cutoff_date="2024-12-31"))
    builder = CorpusBuilder(
        client=client, sample_docs_per_source=taskset.curator.sample_docs_per_source
    )
    taskset._client = client
    taskset._corpus_builder = builder
    taskset._leakage_detector = LeakageDetector(DEFAULT_EVAL_CORPUS)
    taskset._trainer = HeuristicProxyTrainer()

    # The taskset exposes NO tools (the non-MCP gate passes).
    assert type(taskset).tools is Taskset.tools

    task = taskset.load_tasks()[0]
    state = CuratorState()
    trace = vf.Trace(task=task, state=state)
    # Seed the rollout with the agent's FINAL message: a fenced JSON manifest.
    prompt = [vf.SystemMessage(content="sys"), vf.UserMessage(content="go")]
    final = (
        "Here is my curation decision.\n\n"
        "```json\n"
        '{"sources": [{"id": "good/encyclopedia", "weight": 1.0},'
        ' {"id": "good/science", "weight": 1.0}]}\n'
        "```"
    )
    graph.prepare_turn(trace, prompt).commit(
        vf.Response(
            id="x",
            created=0,
            model="m",
            message=vf.AssistantMessage(content=final),
            finish_reason="stop",
            usage=vf.Usage(prompt_tokens=1, completion_tokens=1),
        )
    )

    # finalize parses the manifest into the shared state (runtime=None -> the trace
    # metering fallback; no hf calls in the trace -> zero discovery cost here).
    await taskset.finalize(task, trace, None)
    assert RolloutStore.is_finalized(state)
    manifest = RolloutStore.manifest(state)
    assert {s.dataset_id for s in manifest.sources} == {
        "good/encyclopedia",
        "good/science",
    }

    # score materializes the parsed manifest's sources (once each) and trains.
    await taskset.score(trace, None)
    assert client.sample_calls == ["good/encyclopedia", "good/science"]
    assert RolloutStore.ledger(state).hub_calls == 2  # one Hub fetch per source
    assert trace.reward != 0.0  # scoring actually ran over the shared state


class _SlowCountingClient(FakeClient):
    """FakeClient whose sampling sleeps briefly to widen the concurrency window.

    The sleep runs in the worker thread (`asyncio.to_thread`), so concurrent
    callers reliably overlap inside `fetch_source_docs`; `sample_calls` records
    each real underlying fetch.
    """

    def sample_documents(self, dataset_id, config, split, text_field, n):
        import time

        time.sleep(0.02)
        return super().sample_documents(dataset_id, config, split, text_field, n)


@pytest.mark.asyncio
async def test_concurrent_same_key_fetch_coalesces_to_one_fetch_and_one_bill():
    # N concurrent same-key fetches must share ONE underlying Hub fetch and ONE
    # billing event; later callers read the cached result (single-flight).
    client = _SlowCountingClient()
    builder = CorpusBuilder(client=client, sample_docs_per_source=8)
    state = CuratorState()
    key = FetchKey("good/encyclopedia", None, "train", "text", 8)

    results = await asyncio.gather(
        *[builder.fetch_source_docs(state, key) for _ in range(12)]
    )

    # (a) the underlying HF fetch ran exactly once despite 12 racing callers.
    assert client.sample_calls == ["good/encyclopedia"]
    # (b) cost/billing was applied exactly once.
    ledger = RolloutStore.ledger(state)
    assert ledger.hub_calls == 1
    assert ledger.tokens > 0
    # (c) every caller received the same (non-empty, error-free) docs.
    docs0, err0 = results[0]
    assert err0 is None and docs0
    for docs, err in results:
        assert err is None
        assert docs == docs0


# --- Tier A: corpus build + training run exactly once per rollout ----------


class _CountingBuilder(CorpusBuilder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.materialize_calls = 0

    async def materialize(self, manifest, state, *, runtime=None):
        self.materialize_calls += 1
        return await super().materialize(manifest, state, runtime=runtime)


class _CountingTrainer(HeuristicProxyTrainer):
    def __init__(self):
        super().__init__()
        self.calls = 0

    async def train_and_eval(self, corpus, config):
        self.calls += 1
        return await super().train_and_eval(corpus, config)


@pytest.mark.asyncio
async def test_scoring_runs_build_and_training_once_under_concurrency():
    client = FakeClient()
    builder = _CountingBuilder(client=client, sample_docs_per_source=64)
    trainer = _CountingTrainer()
    curator = await _make(client=client, corpus_builder=builder, trainer=trainer)
    await _finalized(curator)

    # Drive every @vf.reward and @vf.metric method concurrently over one trace:
    # the per-rollout double-checked lock + cache (CuratorTaskset._prepared) must
    # collapse them to a single materialize + a single training run.
    trace = curator.trace()
    funcs = discover_decorated(curator.taskset, "reward") + discover_decorated(
        curator.taskset, "metric"
    )
    assert len(funcs) == 22  # 3 rewards + 19 diagnostic metrics
    await asyncio.gather(*[f(trace) for f in funcs])
    assert builder.materialize_calls == 1
    assert trainer.calls == 1


# --- Tier D: external-data robustness; structured errors + sentinel --------


@pytest.mark.parametrize(
    "exc_factory,expected_kind",
    [
        (
            lambda: __import__(
                "datasets.exceptions", fromlist=["DatasetNotFoundError"]
            ).DatasetNotFoundError("nope"),
            "missing",
        ),
        (
            lambda: ValueError("Unknown split 'bad'. Should be one of ['train']."),
            "bad_split",
        ),
        (lambda: KeyError("text_field"), "bad_field"),
        (lambda: PermissionError("401 Client Error: Unauthorized for url"), "auth"),
        (
            lambda: RuntimeError(
                "Dataset scripts are no longer supported, but found legacy.py"
            ),
            "script_dataset",
        ),
        (lambda: ConnectionError("Connection refused"), "network"),
        (lambda: TimeoutError("timed out"), "timeout"),
    ],
)
@pytest.mark.asyncio
async def test_fetch_failures_are_structured_and_scoring_degrades(
    exc_factory, expected_kind
):
    client = FailingClient(exc_factory)
    curator = await _make(
        client=client, fetch_max_attempts=1, fetch_timeout_seconds=2.0
    )

    # (a) the corpus fetch surfaces a structured error of the expected kind.
    docs, error = await curator.corpus_builder.fetch_source_docs(
        curator.state, FetchKey("good/encyclopedia", None, "train", "text", 4)
    )
    assert docs == []
    assert error["error_kind"] == expected_kind

    # (b) a finalized manifest over the failing source completes scoring without
    # raising, and (c) returns the defined sentinel with external-failure telemetry.
    curator.set_manifest(["good/encyclopedia"], finalize=True)
    scoring = await curator.prepared()
    assert scoring["perf"] == 0.0
    assert "quality" not in scoring
    assert "diversity" not in scoring
    assert RolloutStore.has_external_failure(curator.state)
    assert RolloutStore.tool_error_count(curator.state) >= 1


@pytest.mark.asyncio
async def test_real_timeout_classified_via_wait_for():
    import time as _time

    class _SlowClient(FakeClient):
        def sample_documents(self, *a, **k):
            _time.sleep(0.3)
            return ["doc"]

    policy = RetryPolicy(attempts=1, timeout=0.05, per_doc_seconds=0.0)
    builder = CorpusBuilder(client=_SlowClient(), retry_policy=policy)
    state = CuratorState()
    docs, error = await builder.fetch_source_docs(
        state, FetchKey("a/b", None, "train", "text", 4)
    )
    assert docs == []
    assert error["error_kind"] == "timeout"


def test_fetch_timeout_scales_with_requested_document_count():
    policy = RetryPolicy(timeout=30.0, per_doc_seconds=0.25)

    assert policy.timeout_for_documents(0) == pytest.approx(30.0)
    assert policy.timeout_for_documents(40) == pytest.approx(40.0)


def test_classify_exception_kinds():
    assert classify_exception(DatasetAccessError("x", kind="auth")) == "auth"
    assert classify_exception(KeyError("col")) == "bad_field"
    assert classify_exception(ConnectionError("boom")) == "network"
    assert classify_exception(TimeoutError("t")) == "timeout"
    assert (
        classify_exception(
            RuntimeError(
                "Dataset scripts are no longer supported, but found legacy.py"
            )
        )
        == "script_dataset"
    )
    assert classify_exception(RuntimeError("something weird")) == "unknown"


@pytest.mark.asyncio
async def test_script_dataset_runtime_error_is_permanent_without_retry():
    calls = 0

    def fail():
        nonlocal calls
        calls += 1
        raise RuntimeError(
            "Dataset scripts are no longer supported, but found legacy.py"
        )

    with pytest.raises(DatasetAccessError) as excinfo:
        await run_blocking_with_retry(
            fail,
            policy=RetryPolicy(attempts=3, timeout=1.0),
            semaphore=asyncio.Semaphore(1),
            dataset_id="owner/legacy",
        )

    assert calls == 1
    assert excinfo.value.kind == "script_dataset"


def test_script_dataset_probe_blocks_when_disabled(monkeypatch):
    calls = []
    load_calls = []

    class FakeHfApi:
        def __init__(self, *, token):
            assert token == "test-token"

        def file_exists(self, **kwargs):
            calls.append(kwargs)
            return True

    monkeypatch.setattr("huggingface_hub.HfApi", FakeHfApi)
    monkeypatch.setattr(
        "datasets.load_dataset",
        lambda *args, **kwargs: load_calls.append((args, kwargs)),
    )
    client = HuggingFaceDatasetClient(
        token="test-token", allow_script_datasets=False
    )

    with pytest.raises(DatasetAccessError) as excinfo:
        client.sample_documents("owner/legacy", None, "train", None, 1)

    assert excinfo.value.kind == "script_dataset"
    assert "allow_script_datasets=True" in str(excinfo.value)
    assert calls == [
        {
            "repo_id": "owner/legacy",
            "filename": "legacy.py",
            "repo_type": "dataset",
        }
    ]
    assert load_calls == []


def test_script_dataset_probe_blocks_when_datasets_runtime_is_unsupported(
    monkeypatch,
):
    class FakeHfApi:
        def __init__(self, *, token):
            pass

        def file_exists(self, **kwargs):
            return True

    monkeypatch.setattr("huggingface_hub.HfApi", FakeHfApi)
    monkeypatch.setattr("datasets.__version__", "4.6.1")
    client = HuggingFaceDatasetClient(
        token="test-token", allow_script_datasets=True
    )

    with pytest.raises(DatasetAccessError) as excinfo:
        client.sample_documents("owner/legacy", None, "train", None, 1)

    assert excinfo.value.kind == "script_dataset"
    assert "datasets==4.6.1" in str(excinfo.value)
    assert "allow_script_datasets=True" in str(excinfo.value)


def test_non_script_dataset_load_does_not_pass_trust_remote_code(monkeypatch):
    class FakeHfApi:
        def __init__(self, *, token):
            pass

        def file_exists(self, **kwargs):
            return False

    load_calls = []

    def fake_load_dataset(dataset_id, **kwargs):
        load_calls.append((dataset_id, kwargs))
        return iter([{"text": "data-only document"}])

    monkeypatch.setattr("huggingface_hub.HfApi", FakeHfApi)
    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)
    client = HuggingFaceDatasetClient(
        token="test-token", allow_script_datasets=False
    )

    assert client.sample_documents("owner/data", None, "train", None, 1) == [
        "data-only document"
    ]
    assert len(load_calls) == 1
    assert "trust_remote_code" not in load_calls[0][1]


@pytest.mark.asyncio
async def test_loop_local_semaphore_honors_most_restrictive_limit():
    # A second env instance in the same loop must not inherit a larger bound:
    # a later, smaller limit tightens the shared semaphore (and a later larger
    # one cannot loosen it).
    import weakref

    registry: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
    wide = loop_local_semaphore(registry, 8)
    assert wide._value == 8

    narrow = loop_local_semaphore(registry, 4)
    assert narrow._value == 4  # tightened to the more restrictive limit

    again = loop_local_semaphore(registry, 16)
    assert again is narrow and again._value == 4  # larger request cannot loosen it


# --- Tier F: heavy CPU work is offloaded off the event loop ----------------


@pytest.mark.asyncio
async def test_heavy_compute_is_offloaded(monkeypatch):
    import pretrain_data_curator.rewards as rewards_mod

    offloaded: list[str] = []
    real_to_thread = asyncio.to_thread

    async def spy(fn, *args, **kwargs):
        offloaded.append(getattr(fn, "__name__", repr(fn)))
        return await real_to_thread(fn, *args, **kwargs)

    monkeypatch.setattr(rewards_mod.asyncio, "to_thread", spy)

    curator = await _make()
    await _finalized(curator)
    await curator.prepared()
    # Leakage scoring goes through to_thread.
    assert "score" in offloaded  # LeakageDetector.score


def test_proxy_student_recipe_defaults_mirror_record01():
    # The record_01 recipe defaults: AdamW(betas=(0.9,0.95), eps=1e-8), weight_decay
    # 0.1, grad_clip 1.0, cosine floor 0.1, single run (cost/calibration unchanged),
    # and a derived warmup of min(256, max(1, steps//10)) = 20 for the default 200.
    cfg = ProxyStudentConfig()
    assert (cfg.adam_beta1, cfg.adam_beta2, cfg.adam_eps) == (0.9, 0.95, 1e-8)
    assert cfg.weight_decay == 0.1
    assert cfg.grad_clip == 1.0
    assert cfg.lr_min_ratio == 0.1
    assert cfg.n_train_runs == 1
    assert cfg.warmup_steps is None
    assert (
        cfg.effective_warmup_steps == min(256, max(1, cfg.effective_steps // 10)) == 20
    )
    # An explicit warmup is clamped to the run length so it never exceeds steps.
    assert ProxyStudentConfig(steps=5, warmup_steps=999).effective_warmup_steps == 5


@pytest.mark.parametrize(
    "kwargs",
    [
        {"weight_decay": -0.1},
        {"weight_decay": 1.5},
        {"adam_beta1": 1.0},
        {"adam_beta2": 0.0},
        {"adam_eps": 0.0},
        {"grad_clip": -1.0},
        {"warmup_steps": -1},
        {"lr_min_ratio": 1.5},
        {"n_train_runs": 0},
        {"n_train_runs": 65},
    ],
)
def test_proxy_student_recipe_fields_reject_invalid(kwargs):
    with pytest.raises(ValidationError):
        ProxyStudentConfig(**kwargs)


def test_heuristic_flops_scale_with_budget_when_corpus_permits():
    from pretrain_data_curator.corpus import CuratedCorpus, SourceCorpus

    # ~5000 estimated tokens (4000 words / 20000 chars -> chars//4 dominates).
    corpus = CuratedCorpus(
        sources=[SourceCorpus.from_iter("a/b", None, 1.0, ["word " * 4000])]
    )
    trainer = HeuristicProxyTrainer()
    small = ProxyStudentConfig(batch_size=1, block_size=8)  # default budget: 200*8=1600
    big = ProxyStudentConfig(batch_size=1, block_size=8, train_token_budget=40_000)

    res_small = asyncio.run(trainer.train_and_eval(corpus, small))
    res_big = asyncio.run(trainer.train_and_eval(corpus, big))

    # A bigger budget consumes more of the (larger-than-default) corpus, so both
    # tokens_trained and the billed FLOPs rise — but never beyond the corpus.
    assert res_small.tokens_trained == 1_600
    assert res_big.tokens_trained == 5_000  # min(corpus 5000, target 40000)
    assert res_big.flops > res_small.flops


# --- Tier J/K: zero-weight telemetry metrics do not affect reward ----------


def test_telemetry_metrics_are_zero_weight():
    # The external-failure diagnostics are registered as @vf.metric, never
    # @vf.reward, so they are recorded but never summed into the reward — the v1
    # structural equivalent of the v0 zero reward weight.
    taskset = _Curator().taskset
    reward_names = {f.__name__ for f in discover_decorated(taskset, "reward")}
    metric_names = {f.__name__ for f in discover_decorated(taskset, "metric")}
    for name in ("tool_error_count", "external_failure", "budget_fill_ratio"):
        assert name in metric_names
        assert name not in reward_names


@pytest.mark.asyncio
async def test_reward_unaffected_by_recorded_errors():
    curator = await _make()
    await _finalized(curator)
    baseline = (await curator.score()).reward

    # Inject telemetry on a fresh rollout; recompute. The zero-weight diagnostic
    # metrics must not change the reward.
    await curator.reset()
    await _finalized(curator)
    RolloutStore.record_tool_error(curator.state, "missing")
    RolloutStore.set_external_failure(curator.state, True)
    trace = await curator.score()
    assert trace.reward == pytest.approx(baseline)
    assert trace.metrics["tool_error_count"] == 1.0
    assert trace.metrics["external_failure"] == 1.0
    assert trace.metrics["budget_fill_ratio"] == pytest.approx(
        curator.state.budget_fill_ratio
    )


# --- Tier L: reward surface is CE performance minus penalties ---------------


def test_reward_surface_has_only_perf_cost_and_leakage():
    taskset = _Curator().taskset
    reward_names = {f.__name__ for f in discover_decorated(taskset, "reward")}
    assert reward_names == {"perf_reward", "cost_penalty", "leakage_penalty"}


@pytest.mark.asyncio
async def test_severe_leakage_remains_a_penalty_without_bonus_gating():
    client = FakeClient()
    curator = await _make(client=client)
    state = await _finalized(curator, sources=("good/encyclopedia",))
    # Eval corpus == the exact docs the source returns -> severe leakage.
    leaky_eval = client._docs["good/encyclopedia"]
    scorer = _scorer(
        HeuristicProxyTrainer(),
        config=curator.config,
        corpus_builder=curator.corpus_builder,
        leakage=LeakageDetector(leaky_eval, seed=0),
    )
    scoring = await scorer.compute_scoring(state)
    assert scoring["leakage"]["overall"] > 0.0
    assert "quality" not in scoring
    assert "diversity" not in scoring


# --- Tier M: system prompt teaches the hf-CLI + JSON-manifest workflow ------


def test_system_prompt_teaches_hf_cli_and_json_manifest():
    # The agent curates via the `hf` CLI and emits a fenced JSON manifest; the
    # prompt must teach both (and not reference the retired curator_* MCP tools).
    assert "hf datasets ls" in SYSTEM_PROMPT
    assert "hf datasets info" in SYSTEM_PROMPT
    assert "command -v hf" in SYSTEM_PROMPT
    assert "pip install -q 'huggingface-hub>=0.34'" in SYSTEM_PROMPT
    assert "--search" in SYSTEM_PROMPT
    assert "| head -c 6000" in SYSTEM_PROMPT
    assert "Never request `tags` from `datasets ls`" in SYSTEM_PROMPT
    assert "```json" in SYSTEM_PROMPT
    assert '"sources"' in SYSTEM_PROMPT
    assert "curator_" not in SYSTEM_PROMPT  # no stale MCP tool references


@pytest.mark.asyncio
async def test_discovery_output_budget_stops_before_provider_context_overflow():
    curator = await _make()
    budget = curator.taskset._discovery_output_budget_chars()
    trace = _trace_with_bash_calls(
        curator.task,
        curator.state,
        [
            (
                "hf datasets ls --search wikipedia --limit 5",
                "wikimedia/wikipedia " + ("x" * budget),
            ),
            ("hf datasets info wikimedia/wikipedia", "unused"),
        ],
    )

    assert await curator.taskset.discovery_output_budget_reached(trace)


def test_discovery_output_budget_derives_from_prompt_call_allowance():
    taskset = CuratorTaskset(
        CuratorTasksetConfig(id="test", max_turns=1000, scan_limit=1000)
    )
    _, calls = taskset._discovery_budget()
    assert calls == 24
    assert taskset._discovery_output_budget_chars() > calls * 6_000


def test_system_prompt_manifest_example_parses():
    # The schema example embedded in the prompt must itself parse into a Manifest,
    # so the documented contract and the parser cannot silently drift apart.
    manifest = parse_manifest(SYSTEM_PROMPT, default_token_budget=1_000_000)
    assert manifest is not None
    assert manifest.sources  # the example carries at least one source
    assert manifest.sources[0].text_field is None


def test_system_prompt_sample_docs_per_source_example_is_not_a_bare_literal():
    # Regression: a live eval showed an agent anchoring on a literal example
    # value (copying "sample_docs_per_source": 64 verbatim instead of computing
    # its own number from token_budget). The example's shown value must not be
    # a bare copy-pasteable integer -- match the existing `id` field's
    # placeholder-string convention so it reads as "fill this in", not a
    # literal answer -- while the field name itself still appears (so the agent
    # sees it's a real, expected top-level key) and the prose still documents
    # bounds/behavior.
    assert '"sample_docs_per_source"' in SYSTEM_PROMPT
    assert '"sample_docs_per_source": 64' not in SYSTEM_PROMPT
    assert "1-100000" in SYSTEM_PROMPT
    assert "compute" in SYSTEM_PROMPT.lower()


def test_system_prompt_is_harness_agnostic_about_running_commands():
    # Regression: with a tool-calling CLI harness (e.g. codex) the agent must be
    # told to CALL its shell tool, not to "respond with a bash command" — the latter
    # made models emit the command as prose and stop after a single turn, never
    # running discovery (finalized=0, reward=0). The prompt must drive command
    # execution in a way that works for BOTH message-executing harnesses (bash,
    # kimi_code, default) and tool-calling ones (codex, ...), with no response-format
    # wording that assumes the message itself is the command.
    low = SYSTEM_PROMPT.lower()
    assert "your first response must be a bash command" not in low
    assert "call that tool to run each command" in low  # tool-calling harnesses
    assert "reply with the command itself" in low  # message-executing harnesses
    # Explicitly warns that merely writing the command out is not running it.
    assert "does not run it" in low


# --- Tier P: held-out validation set (NanoGPT speedrun retarget) ------------
#
# The downstream cross-entropy (Perf) signal is meant to be scored against a
# fixed, held-out token stream. This tier covers the new held-out val set:
# the NanoGPT speedrun FineWeb GPT-2-BPE val shard, the first 10,485,760 tokens.


def _make_shard(token_ids, *, magic=SHARD_MAGIC, version=SHARD_VERSION, declared=None):
    """Build a modded-nanogpt .bin token shard (256-int32 header + uint16 tokens)."""
    header = np.zeros(SHARD_HEADER_INTS, dtype="<i4")
    header[0] = magic
    header[1] = version
    header[2] = len(token_ids) if declared is None else declared
    body = np.asarray(token_ids, dtype="<u2")
    return header.tobytes() + body.tobytes()


def test_validation_set_config_defaults_to_speedrun():
    cfg = ValidationSetConfig()
    assert cfg.dataset_id == NANOGPT_VAL_DATASET_ID == "kjj0/fineweb10B-gpt2"
    assert cfg.filename == NANOGPT_VAL_FILENAME == "fineweb_val_000000.bin"
    assert cfg.repo_type == "dataset"
    assert cfg.tokenizer == "gpt2"
    # The exact slice length used by modded-nanogpt's train_gpt.py.
    assert cfg.val_tokens == NANOGPT_VAL_TOKENS == 10_485_760


def test_validation_set_config_rejects_invalid():
    with pytest.raises(ValidationError):
        ValidationSetConfig(val_tokens=0)
    with pytest.raises(ValidationError):
        ValidationSetConfig(dataset_id="")


def test_curator_config_carries_validation_set_default():
    cfg = CuratorConfig()
    assert cfg.validation_set.dataset_id == "kjj0/fineweb10B-gpt2"
    assert cfg.validation_set.val_tokens == 10_485_760


def test_parse_token_shard_slices_exactly_first_n():
    # Shard has MORE tokens than the limit; the slice keeps exactly the first N.
    tokens = list(range(50))
    shard = _make_shard(tokens)
    val = parse_token_shard(shard, limit=10)
    assert val.n_tokens == 10
    assert val.tokens.tolist() == list(range(10))
    assert val.tokens.dtype == np.dtype("<u2")
    assert val.dataset_id == NANOGPT_VAL_DATASET_ID


def test_parse_token_shard_caps_at_available_tokens():
    shard = _make_shard([7, 8, 9])
    val = parse_token_shard(shard, limit=10_000)
    assert val.n_tokens == 3
    assert val.tokens.tolist() == [7, 8, 9]


def test_parse_token_shard_is_deterministic_and_roundtrips_bytes():
    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    shard = _make_shard(tokens)
    a = parse_token_shard(shard, limit=6)
    b = parse_token_shard(shard, limit=6)
    assert a.tokens.tolist() == b.tokens.tolist() == tokens[:6]
    # The header-free uint16 bytes uploaded to the sandbox are exactly the slice.
    assert a.to_uint16_bytes() == np.asarray(tokens[:6], dtype="<u2").tobytes()


@pytest.mark.parametrize(
    "shard,kind_match",
    [
        (_make_shard([1, 2, 3], magic=123), "bad magic"),
        (_make_shard([1, 2, 3], version=99), "unsupported version"),
        (b"\x00\x00", "truncated header"),
    ],
)
def test_parse_token_shard_rejects_malformed(shard, kind_match):
    with pytest.raises(DatasetAccessError) as excinfo:
        parse_token_shard(shard, limit=4)
    assert excinfo.value.kind == "bad_field"
    assert kind_match in str(excinfo.value)


def _shard_download_fn(tmp_path, token_ids, counter=None):
    """A ValTokenLoader download_fn that writes a synthetic shard to disk."""
    path = tmp_path / "fineweb_val_000000.bin"
    path.write_bytes(_make_shard(token_ids))

    def download(dataset_id, filename, repo_type):
        if counter is not None:
            counter.append((dataset_id, filename, repo_type))
        return str(path)

    return download


@pytest.mark.asyncio
async def test_val_loader_resolves_source_and_token_count(tmp_path):
    calls = []
    loader = ValTokenLoader(
        ValidationSetConfig(val_tokens=12),
        download_fn=_shard_download_fn(tmp_path, list(range(100)), calls),
    )
    val = await loader.load()
    assert val.dataset_id == "kjj0/fineweb10B-gpt2"
    assert val.filename == "fineweb_val_000000.bin"
    assert val.n_tokens == 12  # exactly the first val_tokens
    assert val.tokens.tolist() == list(range(12))
    # Resolved through the speedrun source.
    assert calls == [("kjj0/fineweb10B-gpt2", "fineweb_val_000000.bin", "dataset")]


@pytest.mark.asyncio
async def test_val_loader_caches_and_single_flights(tmp_path):
    calls = []
    loader = ValTokenLoader(
        ValidationSetConfig(val_tokens=8),
        download_fn=_shard_download_fn(tmp_path, list(range(50)), calls),
    )
    # Concurrent first loads must coalesce onto ONE download (single-flight),
    # and a later load reads the cache.
    results = await asyncio.gather(*[loader.load() for _ in range(8)])
    again = await loader.load()
    assert len(calls) == 1
    for r in (*results, again):
        assert r.tokens.tolist() == list(range(8))


@pytest.mark.asyncio
async def test_val_loader_fetch_failure_raises_typed_error(tmp_path):
    def boom(dataset_id, filename, repo_type):
        raise ConnectionError("hub unreachable")

    loader = ValTokenLoader(
        ValidationSetConfig(),
        download_fn=boom,
        retry_policy=RetryPolicy(attempts=1, timeout=2.0),
    )
    with pytest.raises(DatasetAccessError) as excinfo:
        await loader.load()
    assert excinfo.value.kind == "network"


@pytest.mark.asyncio
async def test_heuristic_trainer_ignores_val_set():
    # The default heuristic backend does NOT compute per-token CE on a held-out
    # set, so retargeting the val set must not change its (synthetic) loss.
    curator = await _make(validation_set={"val_tokens": 4096})
    assert curator.config.validation_set.val_tokens == 4096
    await _finalized(curator, sources=("good/encyclopedia",))
    scoring = await curator.prepared()
    curator2 = await _make(validation_set={"val_tokens": 9_999_999})
    await _finalized(curator2, sources=("good/encyclopedia",))
    scoring2 = await curator2.prepared()
    assert scoring["loss"] == scoring2["loss"]


def test_load_environment_accepts_validation_set_override(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-token")
    env = load_environment(
        validation_set={"dataset_id": "custom/val", "val_tokens": 1024},
    )
    # The override flows through the compat shim into the taskset's CuratorConfig.
    assert env.taskset.curator.validation_set.dataset_id == "custom/val"
    assert env.taskset.curator.validation_set.val_tokens == 1024


# --- Tier Q: held-out CE windowing/reduction (CPU-testable; guards the GPU loop)


@pytest.mark.parametrize("n_tokens", [2, 100, 257, 1000, 10_485_760])
def test_plan_val_windows_covers_every_target(n_tokens):
    block = 256
    windows = plan_val_windows(n_tokens, block)
    # Every window has <= block targets and they tile the target range with no
    # overlap and no gap: exactly the n_tokens-1 predictable next-token positions,
    # INCLUDING the final partial window.
    covered = []
    for start, length in windows:
        assert 1 <= length <= block
        covered.extend(range(start + 1, start + length + 1))
    assert covered == list(range(1, n_tokens))
    assert sum(length for _, length in windows) == n_tokens - 1


def test_plan_val_windows_short_input_scores_nonzero():
    # A val set shorter than one block must still score its len-1 targets, NOT
    # zero windows (the old `(len-1)//block` math scored 0 -> bogus 0.0 loss).
    assert plan_val_windows(100, 256) == [(0, 99)]
    assert plan_val_windows(2, 256) == [(0, 1)]


@pytest.mark.parametrize("n_tokens", [0, 1])
def test_plan_val_windows_empty_raises(n_tokens):
    # No predictable positions -> must fail loud, never silently score 0.0.
    with pytest.raises(ValueError, match="no predictable positions"):
        plan_val_windows(n_tokens, 256)


def test_mean_held_out_ce_reduces_over_all_targets():
    # A constant per-target CE of c -> mean is exactly c (denominator = #targets).
    seen = []

    def window_loss_sum(start, length):
        seen.append((start, length))
        return 3.5 * length  # constant per-target loss of 3.5

    mean = mean_held_out_ce(1000, 256, window_loss_sum)
    assert mean == pytest.approx(3.5)
    # The denominator is the actual scored-target count (= n_tokens - 1), so a
    # capped/short set cannot dilute or inflate the mean.
    assert sum(length for _, length in seen) == 999


def test_mean_held_out_ce_empty_raises_not_zero():
    # An empty val set must raise, never return a perfect 0.0 from an empty sum.
    with pytest.raises(ValueError, match="no predictable positions"):
        mean_held_out_ce(1, 256, lambda start, length: 0.0)


def test_sandbox_script_embeds_tested_windowing_helper():
    # The GPU-only script must run the SAME plan_val_windows this tier tests, and
    # must no longer contain the old, buggy non-overlapping-full-block windowing.
    import ast
    import inspect

    from pretrain_data_curator.trainer import NANOGPT_TRAIN_SCRIPT

    ast.parse(NANOGPT_TRAIN_SCRIPT)  # the injected script is valid Python
    # Exact single-source identity: the literal helper source the unit tests
    # exercise must appear verbatim in the script, proving the GPU loop runs
    # byte-identical code to the tested helper (a refactor can't silently diverge
    # the sandbox copy).
    helper_src = inspect.getsource(plan_val_windows).rstrip()
    assert helper_src in NANOGPT_TRAIN_SCRIPT
    assert "val_data) - 1) // block" not in NANOGPT_TRAIN_SCRIPT  # old logic gone
    assert "loss_sum / max(total, 1)" not in NANOGPT_TRAIN_SCRIPT  # no bogus 0.0
    assert "val_loss = loss_sum / total" in NANOGPT_TRAIN_SCRIPT


def test_parse_token_shard_rejects_odd_body():
    # A corrupt shard with a dangling odd body byte must raise a typed
    # DatasetAccessError(bad_field), not a bare NumPy ValueError.
    shard = _make_shard([], declared=3) + b"\x01\x02\x03"  # 3 body bytes (odd)
    with pytest.raises(DatasetAccessError) as excinfo:
        parse_token_shard(shard, limit=4)
    assert excinfo.value.kind == "bad_field"
    assert "not a multiple of" in str(excinfo.value)


# ===========================================================================
# Tier R: the hf-CLI redesign — manifest parsing, cost metering, finalize.
# ===========================================================================


# --- manifest parser (the agent's final-message deliverable) ---------------


def test_parse_manifest_from_fenced_json():
    text = (
        "Here is my decision.\n\n"
        "```json\n"
        '{"token_budget": 2000, "sources": [{"id": "a/b", "weight": 2.0}]}\n'
        "```\n"
    )
    m = parse_manifest(text)
    assert m is not None
    assert m.token_budget == 2000
    assert m.sources[0].dataset_id == "a/b"
    assert m.sources[0].weight == 2.0


def test_parse_manifest_from_prose_bare_object():
    text = 'final answer: {"sources": [{"id": "a/b", "weight": 1.0}]} — done'
    m = parse_manifest(text)
    assert m is not None
    assert m.sources[0].dataset_id == "a/b"


def test_parse_manifest_prefers_fenced_json_over_earlier_braces():
    # A stray non-manifest object in the prose must not win over the fenced
    # manifest block (the parser prefers an object carrying `sources`).
    text = (
        'I considered {"note": "stuff"} first.\n'
        "```json\n"
        '{"sources": [{"id": "good/science", "weight": 1.0}]}\n"'
        "```"
    )
    m = parse_manifest(text)
    assert m is not None
    assert m.sources[0].dataset_id == "good/science"


@pytest.mark.parametrize(
    "text",
    [
        "",  # empty
        "I could not find any suitable datasets.",  # prose, no JSON
        '```json\n{"sources": []}\n```',  # empty sources
        '{"sources": [{"weight": 1.0}]}',  # source has no id
        '```json\n{"sources": [{"id": "a/b"',  # truncated (unbalanced)
        '{"not_sources": 1}',  # object without sources
    ],
)
def test_parse_manifest_returns_none_for_unusable(text):
    assert parse_manifest(text) is None


def test_parse_manifest_coerces_source_fields():
    text = json.dumps(
        {
            "token_budget": 5000,
            "sources": [
                {
                    "id": "a/b",
                    "weight": 3,
                    "config": "en",
                    "split": "validation",
                    "text_field": "content",
                    "filters": [
                        {"kind": "min_chars", "params": {"value": 50}},
                        {"kind": "bogus_kind"},  # unsupported -> dropped
                    ],
                    "max_docs": 10,
                    "max_tokens": 2000,
                },
                {"dataset_id": "c/d"},  # alternate id key, default weight
                {"name": "e/f", "weight": -1},  # negative weight clamped to 0
            ],
        }
    )
    m = parse_manifest(text)
    assert m is not None
    assert m.token_budget == 5000
    assert [s.dataset_id for s in m.sources] == ["a/b", "c/d", "e/f"]
    s0 = m.sources[0]
    assert s0.config == "en" and s0.split == "validation" and s0.text_field == "content"
    assert s0.weight == 3.0
    assert [f.kind for f in s0.filters] == ["min_chars"]
    assert s0.sampling.max_docs == 10 and s0.sampling.max_tokens == 2000
    assert m.sources[1].weight == 1.0  # default
    assert m.sources[2].weight == 0.0  # clamped


def test_parse_manifest_drops_filters_with_invalid_params():
    text = json.dumps(
        {
            "sources": [
                {
                    "id": "a/b",
                    "filters": [
                        {"kind": "min_chars", "params": {"value": "200"}},
                        {"kind": "min_tokens", "params": {"value": "many"}},
                        {"kind": "max_symbol_ratio", "params": {"value": "nan"}},
                        {"kind": "drop_regex", "params": {"pattern": "["}},
                        {"kind": "keep_regex", "params": {"pattern": "^valid$"}},
                    ],
                }
            ]
        }
    )

    manifest = parse_manifest(text)

    assert manifest is not None
    assert [(spec.kind, spec.params) for spec in manifest.sources[0].filters] == [
        ("min_chars", {"value": 200}),
        ("keep_regex", {"pattern": "^valid$"}),
    ]


def test_parse_manifest_reads_sample_docs_per_source():
    text = json.dumps({"sources": [{"id": "a/b"}], "sample_docs_per_source": 500})
    m = parse_manifest(text)
    assert m is not None
    assert m.sample_docs_per_source == 500


def test_parse_manifest_missing_sample_docs_per_source_defaults_to_none():
    text = json.dumps({"sources": [{"id": "a/b"}]})
    m = parse_manifest(text)
    assert m is not None
    assert m.sample_docs_per_source is None


def test_parse_manifest_non_numeric_sample_docs_per_source_tolerated_as_none():
    text = json.dumps({"sources": [{"id": "a/b"}], "sample_docs_per_source": "lots"})
    m = parse_manifest(text)
    assert m is not None
    assert m.sample_docs_per_source is None


def test_parse_manifest_out_of_bounds_sample_docs_per_source_rejects_manifest():
    # Consistent with the existing token_budget precedent: a top-level field
    # that fails Manifest's own bounds check invalidates the whole manifest
    # (graceful zero score), rather than silently clamping a cost-relevant knob.
    text = json.dumps({"sources": [{"id": "a/b"}], "sample_docs_per_source": 500_000})
    assert parse_manifest(text) is None


def test_parse_manifest_overflow_sample_docs_per_source_tolerated_as_none():
    # `1e309` is valid JSON and parses to `float("inf")`; `int(float("inf"))`
    # raises OverflowError (not TypeError/ValueError), so this must be caught
    # and treated like any other malformed value -- falling back to None -- not
    # propagate and blow up manifest parsing.
    text = '{"sources": [{"id": "a/b"}], "sample_docs_per_source": 1e309}'
    m = parse_manifest(text)
    assert m is not None
    assert m.sample_docs_per_source is None


def test_parse_manifest_overflow_token_budget_falls_back_to_default():
    text = '{"sources": [{"id": "a/b"}], "token_budget": 1e309}'
    m = parse_manifest(text, default_token_budget=42)
    assert m is not None
    assert m.token_budget == 42


def test_extract_json_object_handles_braces_in_strings():
    obj = extract_json_object('{"q": "a } b { c", "sources": [{"id": "x/y"}]}')
    assert obj is not None
    assert obj["sources"][0]["id"] == "x/y"


# --- hf_meter: classify + JSONL cost log -> ledger -------------------------


@pytest.mark.parametrize(
    "argv,kind",
    [
        (["datasets", "ls", "--search", "code"], "search"),
        (["datasets", "ls", "--sort", "downloads", "--limit", "10"], "search"),
        (["models", "ls", "--search", "bert"], "search"),
        (["datasets", "info", "HuggingFaceFW/fineweb"], "info"),
        (["download", "foo/bar", "--repo-type", "dataset"], "download"),
        (["version"], "local"),
        (["env"], "local"),
        (["auth", "whoami"], "local"),
        (["repo", "create", "x"], "other"),
    ],
)
def test_classify_hf_argv(argv, kind):
    assert hf_meter.classify_hf_argv(argv) == kind


def test_parse_cost_log_maps_records_to_ledger():
    log = "\n".join(
        [
            json.dumps(
                {
                    "argv": ["datasets", "ls", "--search", "code"],
                    "exit": 0,
                    "bytes": 400,
                }
            ),
            json.dumps({"argv": ["datasets", "info", "a/b"], "exit": 0, "bytes": 80}),
            json.dumps(
                {
                    "argv": ["download", "a/b", "--repo-type", "dataset"],
                    "exit": 0,
                    "bytes": 4000,
                }
            ),
            json.dumps({"argv": ["version"], "exit": 0, "bytes": 12}),  # local -> free
            "   ",  # blank line tolerated
            "{not valid json",  # corrupt line tolerated
        ]
    )
    led = hf_meter.parse_cost_log(log)
    # search -> web_queries + hub_calls; info + download -> hub_calls; version free.
    assert led.web_queries == 1
    assert led.hub_calls == 3
    # downloaded bytes -> tokens (bytes // 4), summed over the network calls only.
    assert led.tokens == 400 // 4 + 80 // 4 + 4000 // 4
    assert led.train_flops == 0.0  # never set by metering


def test_parse_cost_log_charges_failed_calls_too():
    # A failed hf call still cost a round-trip; it is charged like the old tool
    # accounting (which incremented before the call), regardless of exit code.
    log = json.dumps(
        {"argv": ["datasets", "ls", "--search", "x"], "exit": 1, "bytes": 0}
    )
    led = hf_meter.parse_cost_log(log)
    assert led.web_queries == 1 and led.hub_calls == 1


# --- hf_meter: trace-reconstruction fallback -------------------------------


def test_ledger_from_messages_reconstructs_tool_call_and_text_action_calls():
    messages = [
        SimpleNamespace(role="system", content="sys"),
        SimpleNamespace(role="user", content="go"),
        # tool-call harness (bash): command in the tool-call arguments; the paired
        # ToolMessage's content is its output size.
        SimpleNamespace(
            role="assistant",
            content=None,
            tool_calls=[
                SimpleNamespace(
                    id="c1",
                    arguments=json.dumps(
                        {"command": "hf datasets ls --search code --limit 5"}
                    ),
                )
            ],
        ),
        SimpleNamespace(role="tool", tool_call_id="c1", content="x" * 400),
        # text-action harness (mini_swe_agent): command fenced in assistant text.
        SimpleNamespace(
            role="assistant",
            content="Now inspect:\n```bash\nhf datasets info good/encyclopedia\n```",
            tool_calls=None,
        ),
        # the final JSON manifest carries no hf call.
        SimpleNamespace(
            role="assistant",
            content='```json\n{"sources": [{"id": "good/encyclopedia", "weight": 1}]}\n```',
            tool_calls=None,
        ),
    ]
    led = hf_meter.ledger_from_messages(messages)
    assert led.web_queries == 1  # the `ls --search`
    assert led.hub_calls == 2  # ls + info
    assert led.tokens == 400 // 4  # only the ls had paired output bytes


def test_extract_hf_commands_splits_on_shell_separators():
    cmds = hf_meter.extract_hf_commands(
        "hf datasets ls --search a && hf datasets info b/c"
    )
    assert cmds == [["datasets", "ls", "--search", "a"], ["datasets", "info", "b/c"]]


# --- finalize: populates state.manifest + cost ledger ----------------------


class _FakeRuntime:
    """Minimal runtime exposing `read` over an optional in-memory cost log."""

    def __init__(self, log_bytes=None):
        self._log = log_bytes

    async def read(self, path):
        if self._log is None:
            raise FileNotFoundError(path)
        return self._log


def _trace_with_final(task, state, final_text):
    """A trace whose single sampled assistant message is ``final_text``."""
    trace = vf.Trace(task=task, state=state)
    prompt = [vf.SystemMessage(content="sys"), vf.UserMessage(content="go")]
    graph.prepare_turn(trace, prompt).commit(
        vf.Response(
            id="x",
            created=0,
            model="m",
            message=vf.AssistantMessage(content=final_text),
            finish_reason="stop",
            usage=vf.Usage(prompt_tokens=1, completion_tokens=1),
        )
    )
    return trace


@pytest.mark.asyncio
async def test_finalize_warns_when_fetch_cap_cannot_reach_token_budget(caplog):
    curator = await _make(sample_docs_per_source=2)
    trace = _trace_with_final(
        curator.task,
        curator.state,
        '```json\n{"token_budget": 1000, "sources": [{"id": "a/b"}]}\n```',
    )

    with caplog.at_level("WARNING"):
        await curator.taskset.finalize(curator.task, trace, None)

    assert "TOKEN BUDGET IS NOT REACHABLE" in caplog.text


@pytest.mark.asyncio
async def test_finalize_populates_manifest_and_meters_runtime_log():
    curator = await _make()
    final = (
        "```json\n"
        '{"sources": [{"id": "good/encyclopedia", "weight": 1.0},'
        ' {"id": "good/science", "weight": 2.0}]}\n'
        "```"
    )
    trace = _trace_with_final(curator.task, curator.state, final)
    log = (
        json.dumps(
            {"argv": ["datasets", "ls", "--search", "x"], "exit": 0, "bytes": 400}
        )
        + "\n"
        + json.dumps(
            {"argv": ["datasets", "info", "good/encyclopedia"], "exit": 0, "bytes": 40}
        )
        + "\n"
    )
    await curator.taskset.finalize(curator.task, trace, _FakeRuntime(log.encode()))

    state = curator.state
    assert RolloutStore.is_finalized(state)
    manifest = RolloutStore.manifest(state)
    assert {s.dataset_id for s in manifest.sources} == {
        "good/encyclopedia",
        "good/science",
    }
    led = RolloutStore.ledger(state)
    assert led.web_queries == 1 and led.hub_calls == 2
    assert led.tokens == 400 // 4 + 40 // 4
    assert led.train_flops == 0.0  # the scorer adds FLOPs later, not finalize


@pytest.mark.asyncio
async def test_finalize_meters_via_trace_when_no_runtime_log():
    curator = await _make()
    final = (
        "Searching:\n```bash\nhf datasets ls --search code\n```\n\n"
        "Final:\n```json\n"
        '{"sources": [{"id": "good/encyclopedia", "weight": 1.0}]}\n'
        "```"
    )
    trace = _trace_with_final(curator.task, curator.state, final)
    # runtime.read raises -> fall back to reconstructing hf calls from the trace.
    await curator.taskset.finalize(curator.task, trace, _FakeRuntime(None))

    assert RolloutStore.is_finalized(curator.state)
    led = RolloutStore.ledger(curator.state)
    assert led.web_queries == 1 and led.hub_calls == 1  # the `ls --search`


@pytest.mark.asyncio
async def test_finalize_graceful_zero_when_no_manifest():
    curator = await _make()
    trace = _trace_with_final(
        curator.task, curator.state, "I could not find suitable datasets, sorry."
    )
    await curator.taskset.finalize(curator.task, trace, None)

    assert not RolloutStore.is_finalized(curator.state)
    # Scoring degrades to the defined zero sentinel rather than crashing.
    scoring = await curator.prepared()
    assert scoring["perf"] == 0.0
    assert scoring["num_sources"] == 0


# --- finalize: cross-turn fallback + turn-budget prompt --------------------


def _trace_with_turns(task, state, assistant_texts):
    """A linear multi-turn trace: one sampled assistant message per text, with a
    synthetic `hf`-output user message interleaved between turns so the graph stays
    linear and ``num_turns == len(assistant_texts)``."""
    trace = vf.Trace(task=task, state=state)
    conversation = [vf.SystemMessage(content="sys"), vf.UserMessage(content="go")]
    for i, text in enumerate(assistant_texts):
        graph.prepare_turn(trace, conversation).commit(
            vf.Response(
                id=f"r{i}",
                created=0,
                model="m",
                message=vf.AssistantMessage(content=text),
                finish_reason="stop",
                usage=vf.Usage(prompt_tokens=1, completion_tokens=1),
            )
        )
        conversation.append(vf.AssistantMessage(content=text))
        conversation.append(vf.UserMessage(content="<hf output>"))
    return trace


@pytest.mark.asyncio
async def test_finalize_falls_back_to_mid_rollout_manifest_at_turn_cap():
    # The agent emits a VALID manifest mid-rollout (turn 1), then keeps issuing `hf`
    # discovery calls until the turn cap trips on a trailing tool call whose message
    # carries no manifest. The OLD finalize parsed ONLY the last message -> not
    # finalized -> num_sources=0 -> perf=0. finalize must now fall back to the most
    # recent ```json manifest across ALL assistant turns, so the rollout finalizes.
    curator = await _make(max_turns=3)
    manifest_turn = (
        "Here is my mixture so far.\n```json\n"
        '{"sources": [{"id": "good/encyclopedia", "weight": 1.0},'
        ' {"id": "good/science", "weight": 1.0}]}\n```'
    )
    later_hf_turns = [
        "Let me double-check a candidate.\n```bash\nhf datasets info good/science\n```",
        "One more search.\n```bash\nhf datasets ls --search wiki --limit 5\n```",
    ]
    trace = _trace_with_turns(
        curator.task, curator.state, [manifest_turn, *later_hf_turns]
    )
    assert trace.num_turns == 3  # the turn cap was reached
    # Regression precondition: the FINAL message alone has no usable manifest, so the
    # old last-message-only finalize would have scored zero.
    assert parse_manifest(trace.assistant_messages[-1].content or "") is None

    await curator.taskset.finalize(curator.task, trace, None)

    # The mid-rollout manifest still finalizes despite the trailing hf calls.
    assert RolloutStore.is_finalized(curator.state)
    manifest = RolloutStore.manifest(curator.state)
    assert {s.dataset_id for s in manifest.sources} == {
        "good/encyclopedia",
        "good/science",
    }
    # The training/perf reward stage is now actually reached.
    scoring = await curator.prepared()
    assert scoring["num_sources"] == 2
    assert scoring["num_sources"] > 0
    assert scoring["perf"] > 0.0


@pytest.mark.asyncio
async def test_finalize_prefers_final_message_manifest_over_earlier_draft():
    # The preferred path is UNCHANGED: when the FINAL message already carries a valid
    # manifest it wins over any earlier (draft) manifest. The cross-turn fallback must
    # never override a finalize the agent actually emitted last.
    curator = await _make()
    draft = (
        "Draft mixture:\n```json\n"
        '{"sources": [{"id": "good/encyclopedia", "weight": 1.0}]}\n```'
    )
    final = (
        "Final answer:\n```json\n"
        '{"sources": [{"id": "good/science", "weight": 2.0}]}\n```'
    )
    trace = _trace_with_turns(curator.task, curator.state, [draft, final])
    await curator.taskset.finalize(curator.task, trace, None)

    assert RolloutStore.is_finalized(curator.state)
    manifest = RolloutStore.manifest(curator.state)
    assert [s.dataset_id for s in manifest.sources] == ["good/science"]
    assert manifest.sources[0].weight == 2.0


# --- finalize: trace-fallback manifest synthesis ----------------------------


def _trace_with_bash_calls(task, state, calls):
    """A multi-turn trace where every assistant turn is a bash tool call.

    ``calls`` is a list of ``(command, result_text)`` pairs.  The result text
    is injected as a ToolMessage into the next turn's context so that
    ``trace.tool_messages`` is populated for all but the last call.
    Returns a trace with NO final text manifest (all turns are tool calls).
    """
    trace = vf.Trace(task=task, state=state)
    conversation = [vf.SystemMessage(content="sys"), vf.UserMessage(content="go")]
    for i, (cmd, result) in enumerate(calls):
        tc = vf.ToolCall(
            id=f"tc{i}", name="bash", arguments=json.dumps({"command": cmd})
        )
        graph.prepare_turn(trace, conversation).commit(
            vf.Response(
                id=f"r{i}",
                created=0,
                model="m",
                message=vf.AssistantMessage(content="", tool_calls=[tc]),
                finish_reason="tool_calls",
                usage=vf.Usage(prompt_tokens=1, completion_tokens=1),
            )
        )
        conversation.append(vf.AssistantMessage(content="", tool_calls=[tc]))
        conversation.append(vf.ToolMessage(tool_call_id=f"tc{i}", content=result))
    return trace


@pytest.mark.asyncio
async def test_finalize_synthesizes_manifest_from_inspected_tool_call_ids():
    # When the agent runs only bash tool calls and never emits a JSON manifest,
    # finalize must still produce a non-empty manifest from the ids the agent
    # explicitly inspected via `hf datasets info <id>` tool calls.
    curator = await _make(max_turns=4)
    calls = [
        (
            "hf datasets ls --search math --sort downloads --limit 5",
            "meta-math/MetaMathQA  456k downloads\nEleutherAI/hendrycks_math  200k downloads",
        ),
        (
            "hf datasets info meta-math/MetaMathQA --expand downloads,likes,tags",
            "Dataset: meta-math/MetaMathQA\ndownloads: 456789\nlicense: mit",
        ),
        (
            "hf datasets info EleutherAI/hendrycks_math --expand downloads,likes,tags",
            "Dataset: EleutherAI/hendrycks_math\ndownloads: 200000\nlicense: mit",
        ),
        (
            "hf datasets ls --search code --sort downloads --limit 5",
            "codeparrot/github-code  1.2M downloads\ncodeparrot/codeparrot-clean  300k",
        ),
    ]
    trace = _trace_with_bash_calls(curator.task, curator.state, calls)
    # Precondition: no assistant message carries a parseable JSON manifest.
    for msg in trace.assistant_messages:
        assert parse_manifest(msg.content or "") is None

    await curator.taskset.finalize(curator.task, trace, None)

    assert RolloutStore.is_finalized(curator.state), (
        "fallback must finalize the rollout"
    )
    manifest = RolloutStore.manifest(curator.state)
    assert manifest.sources, "fallback manifest must be non-empty"
    ids = {s.dataset_id for s in manifest.sources}
    # Recovery prefers deliberately inspected candidates over raw search hits,
    # which can include post-cutoff, gated, or incompatible repositories.
    assert ids == {"meta-math/MetaMathQA", "EleutherAI/hendrycks_math"}
    # Config must be null (no config was observed in tool output).
    assert all(s.config is None for s in manifest.sources)


@pytest.mark.asyncio
async def test_finalize_fallback_only_real_ids_no_invented_sources():
    # The fallback must ONLY use ids that were genuinely observed in the rollout —
    # never fabricated ids.  A rollout with zero hf tool calls produces no fallback.
    curator = await _make()
    calls = [
        ("echo hello", "hello"),  # no hf call at all
    ]
    trace = _trace_with_bash_calls(curator.task, curator.state, calls)
    await curator.taskset.finalize(curator.task, trace, None)

    # No hf ids were observed → fallback has nothing to synthesize → not finalized.
    assert not RolloutStore.is_finalized(curator.state)
    manifest = RolloutStore.manifest(curator.state)
    assert not manifest.sources


@pytest.mark.asyncio
async def test_finalize_primary_path_unchanged_when_manifest_text_present():
    # When a valid JSON manifest IS present in the assistant messages, the primary
    # parse path must win — the trace-fallback synthesizer must not override it.
    curator = await _make(max_turns=4)
    calls = [
        (
            "hf datasets info meta-math/MetaMathQA --expand downloads",
            "Dataset: meta-math/MetaMathQA\ndownloads: 456789",
        ),
    ]
    trace = _trace_with_bash_calls(curator.task, curator.state, calls)
    # Inject a final text turn with a valid manifest (primary path).
    final_manifest = (
        '```json\n{"sources": [{"id": "good/science", "weight": 3.0}]}\n```'
    )
    graph.prepare_turn(
        trace,
        [
            vf.SystemMessage(content="sys"),
            vf.UserMessage(content="go"),
            # Replay the tool-call turns so the graph prefix matches.
            *[
                msg
                for tc_cmd, tc_result in calls
                for msg in [
                    vf.AssistantMessage(
                        content="",
                        tool_calls=[
                            vf.ToolCall(
                                id="tc0",
                                name="bash",
                                arguments=json.dumps({"command": tc_cmd}),
                            )
                        ],
                    ),
                    vf.ToolMessage(tool_call_id="tc0", content=tc_result),
                ]
            ],
        ],
    ).commit(
        vf.Response(
            id="r_final",
            created=0,
            model="m",
            message=vf.AssistantMessage(content=final_manifest),
            finish_reason="stop",
            usage=vf.Usage(prompt_tokens=1, completion_tokens=1),
        )
    )

    await curator.taskset.finalize(curator.task, trace, None)

    assert RolloutStore.is_finalized(curator.state)
    manifest = RolloutStore.manifest(curator.state)
    # Primary-path manifest wins: good/science, not the fallback's MetaMathQA.
    assert [s.dataset_id for s in manifest.sources] == ["good/science"]
    assert manifest.sources[0].weight == 3.0


# --- finalize: grace-period race with the verifiers interception server ----


@pytest.mark.asyncio
async def test_finalize_grace_period_picks_up_late_final_message():
    """Reproduces the confirmed upstream race: `verifiers`' interception server
    commits the agent's real final assistant message to `trace.nodes` AFTER the
    rollout pool has already unregistered the rollout and `finalize()` has begun.
    At the moment `finalize()` first checks, only a tool-call turn (fallback
    fodder) is present; the true manifest lands a beat later, inside the grace
    window. The grace-period poll (`_await_final_manifest`) must pick up the real
    manifest instead of prematurely synthesizing one from trace-discovered ids."""
    curator = await _make()
    trace = vf.Trace(task=curator.task, state=curator.state)
    prompt = [vf.SystemMessage(content="sys"), vf.UserMessage(content="go")]

    # Turn 0: a bash tool call that discovers a real id but carries no manifest --
    # exactly what the tier-2 fallback would synthesize from if the grace period
    # were skipped.
    tc = vf.ToolCall(
        id="tc0",
        name="bash",
        arguments=json.dumps({"command": "hf datasets info good/encyclopedia"}),
    )
    graph.prepare_turn(trace, prompt).commit(
        vf.Response(
            id="r0",
            created=0,
            model="m",
            message=vf.AssistantMessage(content="", tool_calls=[tc]),
            finish_reason="tool_calls",
            usage=vf.Usage(prompt_tokens=1, completion_tokens=1),
        )
    )
    conversation = [
        *prompt,
        vf.AssistantMessage(content="", tool_calls=[tc]),
        vf.ToolMessage(tool_call_id="tc0", content="Dataset: good/encyclopedia"),
    ]

    # Precondition: as of right now, the trace has no usable manifest -- this is
    # the state finalize() sees on its first (pre-grace) check.
    assert parse_manifest(trace.assistant_messages[-1].content or "") is None

    final_manifest = (
        '```json\n{"sources": [{"id": "good/science", "weight": 2.0}]}\n```'
    )

    async def _commit_late_final_message() -> None:
        # Yield past the first grace-period poll before the interception server
        # "finishes" committing the agent's real final message.
        await asyncio.sleep(curator.taskset._FINALIZE_GRACE_INTERVAL_SECONDS * 2)
        graph.prepare_turn(trace, conversation).commit(
            vf.Response(
                id="r1",
                created=0,
                model="m",
                message=vf.AssistantMessage(content=final_manifest),
                finish_reason="stop",
                usage=vf.Usage(prompt_tokens=1, completion_tokens=1),
            )
        )

    late_commit = asyncio.create_task(_commit_late_final_message())
    await curator.taskset.finalize(curator.task, trace, None)
    await late_commit

    assert RolloutStore.is_finalized(curator.state)
    manifest = RolloutStore.manifest(curator.state)
    # The REAL agent-submitted manifest won, not the trace-discovered-ids
    # fallback (which would have synthesized good/encyclopedia instead).
    assert [s.dataset_id for s in manifest.sources] == ["good/science"]
    assert manifest.sources[0].weight == 2.0


@pytest.mark.asyncio
async def test_finalize_falls_back_when_final_message_never_arrives():
    """Companion to the grace-period test above: when the final message truly
    never arrives (no race, just an agent that never submits a manifest), the
    grace period must still expire and the existing trace-discovered-ids
    fallback must still fire exactly as before."""
    curator = await _make()
    calls = [
        (
            "hf datasets info meta-math/MetaMathQA --expand downloads",
            "Dataset: meta-math/MetaMathQA\ndownloads: 456789",
        ),
    ]
    trace = _trace_with_bash_calls(curator.task, curator.state, calls)

    await curator.taskset.finalize(curator.task, trace, None)

    assert RolloutStore.is_finalized(curator.state)
    manifest = RolloutStore.manifest(curator.state)
    assert [s.dataset_id for s in manifest.sources] == ["meta-math/MetaMathQA"]
    assert manifest.sources[0].config is None


def test_system_prompt_includes_configured_turn_budget():
    # The per-rollout system prompt must spell out the ACTUAL configured max_turns so
    # the agent stops calling `hf` and emits its final manifest before the cap trips.
    tasks = CuratorTaskset(CuratorTasksetConfig(id="test", max_turns=7)).load_tasks()
    assert tasks
    prompt = tasks[0].system_prompt
    assert "7 turns" in prompt  # the configured value is rendered in
    assert "manifest" in prompt and "scores zero" in prompt
    # The base hf-CLI teaching is preserved alongside the budget note.
    assert "hf datasets ls" in prompt
    # A different budget renders a different number (not a hard-coded constant).
    other = CuratorTaskset(CuratorTasksetConfig(id="test", max_turns=20)).load_tasks()
    assert "20 turns" in other[0].system_prompt


def test_system_prompt_scales_discovery_with_benchmark_budget():
    # Discovery is capped so the agent commits before exhausting its turns, but
    # benchmark-sized scan/turn budgets are allowed more than smoke runs.
    for max_turns in (7, 12):
        prompt = (
            CuratorTaskset(CuratorTasksetConfig(id="test", max_turns=max_turns))
            .load_tasks()[0]
            .system_prompt
        )
        assert f"{max_turns} turns" in prompt
        assert "contains multiple tool calls" in prompt
        assert "every individual `hf` call is still billed" in prompt
        assert "MUST perform at most 2 discovery rounds" in prompt
        # Commit mechanics: a plain final message, no shell command (works whether
        # the harness runs a shell tool or executes the reply as a command).
        assert "HOW TO COMMIT" in prompt
        assert "stop running commands and reply with a plain message" in prompt
        assert "do not print the manifest through the shell" in prompt
        assert "scores zero" in prompt

    smoke_prompt = (
        CuratorTaskset(CuratorTasksetConfig(id="test", max_turns=25, scan_limit=10))
        .load_tasks()[0]
        .system_prompt
    )
    assert "25 turns" in smoke_prompt
    assert "MUST perform at most 2 discovery rounds" in smoke_prompt

    benchmark_prompt = (
        CuratorTaskset(CuratorTasksetConfig(id="test", max_turns=64, scan_limit=200))
        .load_tasks()[0]
        .system_prompt
    )
    assert "64 turns" in benchmark_prompt
    assert (
        "MUST perform at most 10 discovery rounds (<=20 bash calls)" in benchmark_prompt
    )
    assert "no later than turn 56" in benchmark_prompt
    # No-invention guard: manifest ids must come from observed tool output.
    assert "copied verbatim from a dataset id" in benchmark_prompt
    assert "Do NOT invent or guess" in benchmark_prompt
    assert "config` was not explicitly listed" in benchmark_prompt
    assert "fabricated id" in benchmark_prompt


def test_system_prompt_bootstraps_missing_hf_without_diagnosis_turns():
    # Bare Modal/Prime images may not contain the CLI. The first command must
    # self-heal and search in one turn, while prohibiting unproductive alias/import
    # diagnosis and unrelated installs.
    low = SYSTEM_PROMPT.lower()
    assert "already installed" not in low
    assert "command -v hf" in low
    assert "pip install -q 'huggingface-hub>=0.34'" in low
    assert "fi; hf datasets ls" in low  # bootstrap and useful discovery share one turn
    assert "do not spend turns diagnosing missing commands" in low
    assert "do not try `huggingface-cli`" in low
    assert "only installation step allowed" in low
    assert "pip will only waste your turns" not in low
    assert "no setup required" not in low


# --- the PATH-shadow shim --------------------------------------------------


def test_install_shim_is_idempotent_and_resolves_real_hf():
    real = hf_meter._resolve_real_hf()
    if real is None:
        pytest.skip("no real `hf` CLI on PATH to shim")
    p1 = hf_meter.install_shim()
    p2 = hf_meter.install_shim()
    assert p1 == p2 == hf_meter.SHIM_HF
    assert os.path.isfile(hf_meter.SHIM_HF) and os.access(hf_meter.SHIM_HF, os.X_OK)
    # The shim dir shadows the real hf by sitting first on PATH...
    assert os.environ["PATH"].split(os.pathsep)[0] == hf_meter.SHIM_BIN_DIR
    # ...and the wrapper execs the real hf by absolute path.
    body = Path(hf_meter.SHIM_HF).read_text()
    assert real in body


# --- Tier R (cont.): last-sources-wins manifest parsing --------------------


def test_parse_manifest_prefers_last_sources_block_across_multiple_fences():
    # Multiple fenced blocks: a leading note (no sources), then a DRAFT manifest,
    # then the FINAL manifest. The parser must pick the LAST sources-bearing block
    # (so a draft or a note/plan block never shadows the real final manifest).
    text = (
        'Planning:\n```json\n{"note": "planning"}\n```\n'
        "Draft mixture:\n```json\n"
        '{"sources": [{"id": "good/encyclopedia", "weight": 1.0}]}\n```\n'
        "Final answer:\n```json\n"
        '{"sources": [{"id": "good/science", "weight": 2.0}]}\n```\n'
    )
    m = parse_manifest(text)
    assert m is not None
    assert [s.dataset_id for s in m.sources] == ["good/science"]
    assert m.sources[0].weight == 2.0


def test_parse_manifest_finds_manifest_after_leading_note_block():
    # A leading note object that has no `sources` must NOT shadow a later real
    # manifest -> the manifest is found, not None.
    text = (
        '```json\n{"note": "thinking"}\n```\n'
        "```json\n"
        '{"sources": [{"id": "good/encyclopedia", "weight": 1.0}]}\n```\n'
    )
    m = parse_manifest(text)
    assert m is not None
    assert m.sources[0].dataset_id == "good/encyclopedia"


# --- Tier M (cont.): the per-task goal prompt teaches the hf-CLI workflow ----


def test_build_tasks_prompt_describes_hf_cli_not_curation_tools():
    # The per-task goal prompt must teach the `hf` CLI + fenced-JSON manifest
    # workflow (with the metered-cost note) and no longer reference the retired
    # "curation tools".
    prompt = build_tasks("2024-12-31", 1_000_000)[0].prompt
    assert "curation tools" not in prompt  # the stale, retired wording is gone
    assert "hf datasets ls" in prompt
    assert "hf datasets info" in prompt
    assert "--search" in prompt
    assert "```json" in prompt
    assert "sources" in prompt
    assert "metered" in prompt  # the per-call cost guidance
    assert "2024-12-31" in prompt  # cutoff constraint preserved
    assert "1000000" in prompt  # token budget rendered into the target


# --- Tier R (cont.): finalize trace-fallback metering -----------------------


class _CorruptRuntime:
    """Runtime whose cost-log read fails (a corrupt / unreadable shim log)."""

    def __init__(self, exc):
        self._exc = exc

    async def read(self, path):
        raise self._exc


@pytest.mark.parametrize(
    "runtime",
    [
        None,  # no runtime object at all
        _FakeRuntime(None),  # log file missing (runtime.read raises)
        _CorruptRuntime(OSError("corrupt log")),  # log present but unreadable
    ],
    ids=["no_runtime", "missing_log", "corrupt_log"],
)
@pytest.mark.asyncio
async def test_finalize_trace_fallback_reconstructs_ledger(runtime):
    # When the shim's runtime cost log can't be read (no runtime / missing /
    # unreadable), the finalize metering reconstructs a non-empty discovery ledger
    # from the hf calls recorded in the trace itself.
    curator = await _make()
    final = (
        "Search:\n```bash\nhf datasets ls --search code --limit 5\n```\n"
        "Inspect:\n```bash\nhf datasets info good/encyclopedia\n```\n"
        "Final:\n```json\n"
        '{"sources": [{"id": "good/encyclopedia", "weight": 1.0}]}\n'
        "```"
    )
    trace = _trace_with_final(curator.task, curator.state, final)
    await curator.taskset.finalize(curator.task, trace, runtime)

    assert RolloutStore.is_finalized(curator.state)
    led = RolloutStore.ledger(curator.state)
    # Reconstructed from the trace: the `ls --search` (web query + hub call) plus
    # the `info` (hub call) — a non-empty ledger despite the unusable runtime log.
    assert led.web_queries == 1
    assert led.hub_calls == 2


# --- Tier R (cont.): the PATH-shadow shim actually records a cost line -------


def test_shim_writes_jsonl_cost_record_when_hf_invoked(tmp_path, monkeypatch):
    # Strong shim test: actually invoke `hf` THROUGH the shimmed PATH (with a stub
    # `hf` standing in for the real binary) and assert a JSONL cost record is
    # written. If the shim genuinely can't be exercised here, drive the record
    # shape -> ledger codepath directly and assert the JSONL line shape.
    import shutil

    # A stub `hf` the shim will exec as the "real" binary (prints to stdout, ok).
    real_bin = tmp_path / "realbin"
    real_bin.mkdir()
    stub_hf = real_bin / "hf"
    stub_hf.write_text("#!/bin/sh\nprintf 'DATASET a/b\\n'\nexit 0\n")
    stub_hf.chmod(0o755)

    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    # Stub first on PATH so the shim resolves IT as the real hf; force a fresh
    # install so the shim re-renders against this PATH (and clean up after).
    monkeypatch.setenv("PATH", f"{real_bin}{os.pathsep}{os.environ.get('PATH', '')}")
    hf_meter._installed = False
    try:
        shim = hf_meter.install_shim()
        can_run = bool(shim) and shutil.which("bash") is not None
        if can_run:
            proc = subprocess.run(
                ["hf", "datasets", "ls", "--search", "code"],
                cwd=str(work),
                env=os.environ.copy(),
                capture_output=True,
                text=True,
            )
            assert proc.returncode == 0
            assert "DATASET a/b" in proc.stdout  # real hf stdout passed through
            log_path = work / hf_meter.COST_LOG_NAME
            assert log_path.is_file()
            lines = [ln for ln in log_path.read_text().splitlines() if ln.strip()]
            assert len(lines) == 1  # exactly one cost record for the one hf call
            rec = json.loads(lines[0])
            # The JSONL line shape `parse_cost_log` consumes.
            assert set(rec) >= {"argv", "exit", "bytes", "duration", "ts"}
            assert rec["argv"] == ["datasets", "ls", "--search", "code"]
            assert rec["exit"] == 0
            assert isinstance(rec["bytes"], int) and rec["bytes"] >= 0
            if rec["bytes"]:
                assert rec["bytes"] == len(proc.stdout.encode())
            # ...and it folds into a non-empty ledger (search -> web query + hub call).
            led = hf_meter.parse_cost_log(log_path.read_text())
            assert led.web_queries == 1 and led.hub_calls == 1
        else:
            # Fallback: exercise the record-shape -> ledger codepath directly and
            # assert the JSONL line maps exactly as the live shim's record would.
            rec = {
                "argv": ["datasets", "ls", "--search", "code"],
                "exit": 0,
                "bytes": 400,
                "duration": 0.01,
                "ts": 123.0,
            }
            led = hf_meter.parse_cost_log(json.dumps(rec))
            assert led.web_queries == 1 and led.hub_calls == 1
            assert led.tokens == 400 // 4
    finally:
        hf_meter._installed = False
        shutil.rmtree(hf_meter.SHIM_DIR, ignore_errors=True)


# --- Tier M: selectable real-training backends ----------------------------


def _real_trainer_taskset(**proxy_student):
    use_real = proxy_student.pop("use_real_trainer", True)
    ts = CuratorTaskset(
        CuratorTasksetConfig(
            id="t", use_real_trainer=use_real, proxy_student=proxy_student
        )
    )
    # Inject the non-trainer collaborators so `_ensure` builds only the trainer
    # (no HF token / network needed); the trainer slot stays None for selection.
    ts._client = FakeClient()
    ts._corpus_builder = CorpusBuilder(client=ts._client)
    ts._leakage_detector = LeakageDetector(DEFAULT_EVAL_CORPUS)
    return ts


def test_backend_selection_builds_runtime_selected_dispatcher():
    # _build_real_trainer() always returns a RuntimeSelectedTrainer covering both
    # concrete backends; which one trains is decided at score time from the live
    # harness runtime's type, never from runtime_backend.
    ts = _real_trainer_taskset()
    ts._ensure()
    trainer = ts._trainer
    assert isinstance(trainer, RuntimeSelectedTrainer)
    assert set(trainer._trainers_by_runtime_type) == {"docker", "modal"}


def test_backend_default_is_heuristic_and_no_runtime_backend_selector():
    # There is no default runtime_backend selector, and the default
    # (use_real_trainer False) path still yields the heuristic trainer.
    assert CuratorConfig().proxy_student.runtime_backend is None
    assert ProxyStudentConfig().runtime_backend is None
    ts = _real_trainer_taskset(use_real_trainer=False)
    ts._ensure()
    assert isinstance(ts._trainer, HeuristicProxyTrainer)


def test_docker_runtime_backend_construction_has_no_platform_timeout_ceiling():
    # No more vm/gpu_type fields to set at all; an explicit > 24h timeout
    # constructs cleanly on docker (only modal has a platform ceiling).
    cfg = ProxyStudentConfig(runtime_backend="docker", timeout_minutes=5000)
    assert cfg.runtime_backend == "docker"
    assert cfg.timeout_minutes == 5000
    assert cfg.effective_timeout_minutes == 5000  # not clamped to 1440


def test_docker_image_default_is_shared_across_backends():
    # There is a single shared docker_image default now (no more prime/docker
    # split); an explicit image wins regardless of runtime_backend.
    assert (
        ProxyStudentConfig().docker_image
        == "pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime"
    )
    assert ProxyStudentConfig(runtime_backend="docker").docker_image == (
        "pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime"
    )
    assert (
        ProxyStudentConfig(
            runtime_backend="docker", docker_image="me/img:1"
        ).docker_image
        == "me/img:1"
    )


# --- Tier S: baseline-relative Perf signal (default-ON) ----------------------
#
# The Perf REWARD defaults to the bounded relative val-loss reduction over a
# neutral baseline (``perf_vs_baseline``), which is always surfaced as a
# zero-weight diagnostic too.  Setting baseline_relative_perf=False falls back
# to exp(-loss) — only meaningful for tiny toy models where loss < 1.


def test_curator_config_baseline_defaults():
    cfg = CuratorConfig()
    assert cfg.baseline_relative_perf is True  # default ON: safe for real LMs
    # The neutral reference is the CE of a uniform student over the padded GPT-2
    # vocab (ln(50304)); it is a constant — no extra training run is performed.
    assert cfg.perf_baseline_loss == pytest.approx(math.log(50304))
    with pytest.raises(ValidationError):
        CuratorConfig(perf_baseline_loss=0.0)


def test_exp_loss_perf_reward_when_flag_off():
    # Flag explicitly OFF: _perf == exp(-loss), independent of accuracy.
    # This preserves the legacy formula as a backwards-compat fallback for toy
    # models where loss < 1 and exp(-loss) is a meaningful signal.
    cfg = CuratorConfig(baseline_relative_perf=False)
    scorer = _scorer(HeuristicProxyTrainer(), config=cfg)
    assert scorer.config.baseline_relative_perf is False
    r = TrainResult(loss=2.0, accuracy=0.4, flops=0.0, tokens_trained=0, backend="x")
    assert scorer._perf(r) == scorer._perf_from_result(r)
    assert scorer._perf(r) == pytest.approx(math.exp(-2.0))
    different_accuracy = TrainResult(
        loss=2.0, accuracy=0.99, flops=0.0, tokens_trained=0, backend="x"
    )
    assert scorer._perf(different_accuracy) == pytest.approx(scorer._perf(r))
    # The sentinel still scores zero perf under either mode.
    sentinel = TrainResult(
        loss=float("inf"), accuracy=0.0, flops=0.0, tokens_trained=0, backend="error"
    )
    assert scorer._perf(sentinel) == 0.0


def test_default_perf_reward_is_baseline_relative_improvement():
    # Default (baseline_relative_perf=True): _perf == bounded relative loss reduction,
    # NOT exp(-loss).  For real LMs loss ~ 9 nats so exp(-9) ≈ 0.0001 collapses
    # reward; the relative formula gives a meaningful signal in [0, 1].
    scorer = _scorer(HeuristicProxyTrainer())  # uses CuratorConfig() defaults
    assert scorer.config.baseline_relative_perf is True
    baseline = scorer.config.perf_baseline_loss
    r = TrainResult(loss=2.0, accuracy=0.4, flops=0.0, tokens_trained=0, backend="x")
    expected = max(0.0, min(1.0, (baseline - r.loss) / baseline))
    assert scorer._perf(r) == pytest.approx(expected)
    # Must NOT equal exp(-loss) (the old collapsed formula).
    assert scorer._perf(r) != pytest.approx(math.exp(-r.loss))
    # Worse-than-baseline clamps to 0; sentinel -> 0.
    worse = TrainResult(
        loss=baseline + 1.0, accuracy=0.0, flops=0.0, tokens_trained=0, backend="x"
    )
    assert scorer._perf(worse) == 0.0
    sentinel = TrainResult(
        loss=float("inf"), accuracy=0.0, flops=0.0, tokens_trained=0, backend="error"
    )
    assert scorer._perf(sentinel) == 0.0


def test_baseline_relative_perf_reward_when_enabled():
    cfg = CuratorConfig(baseline_relative_perf=True, perf_baseline_loss=10.0)
    scorer = _scorer(HeuristicProxyTrainer(), config=cfg)
    # Bounded relative reduction over the baseline: (10 - 2)/10 = 0.8.
    r = TrainResult(loss=2.0, accuracy=0.4, flops=0.0, tokens_trained=0, backend="x")
    assert scorer._perf(r) == pytest.approx(0.8)
    # Worse-than-baseline clamps to 0; the infinite-loss sentinel -> 0.
    worse = TrainResult(
        loss=20.0, accuracy=0.0, flops=0.0, tokens_trained=0, backend="x"
    )
    assert scorer._perf(worse) == 0.0
    sentinel = TrainResult(
        loss=float("inf"), accuracy=0.0, flops=0.0, tokens_trained=0, backend="error"
    )
    assert scorer._perf(sentinel) == 0.0


@pytest.mark.asyncio
async def test_perf_vs_baseline_diagnostic_always_surfaced():
    curator = await _make()
    await _finalized(curator)
    scoring = await curator.prepared()
    baseline = curator.config.perf_baseline_loss
    assert scoring["perf_baseline_loss"] == baseline
    assert scoring["perf_vs_baseline"] == pytest.approx(
        (baseline - scoring["loss"]) / baseline
    )
    # Surfaced as a zero-weight metric: recorded in trace.metrics, never a reward.
    trace = await curator.score()
    assert "perf_vs_baseline" in trace.metrics
    reward_names = {f.__name__ for f in discover_decorated(curator.taskset, "reward")}
    assert "perf_vs_baseline" not in reward_names


@pytest.mark.asyncio
async def test_baseline_relative_flag_only_changes_perf_when_off():
    # Two curators differing ONLY in the flag, same finalized manifest + heuristic
    # trainer: the default (on) perf is the relative term; disabling the flag
    # swaps in exp(-loss), proving the two formulae differ and calibration is
    # controlled by the flag.
    on = await _make(baseline_relative_perf=True)
    await _finalized(on)
    on_scoring = await on.prepared()
    baseline = on.config.perf_baseline_loss
    assert on_scoring["perf"] == pytest.approx(
        max(0.0, min(1.0, (baseline - on_scoring["loss"]) / baseline))
    )

    off = await _make(baseline_relative_perf=False)
    await _finalized(off)
    off_scoring = await off.prepared()
    abs_perf = min(1.0, math.exp(-off_scoring["loss"]))
    assert off_scoring["perf"] == pytest.approx(abs_perf)

    # The same corpus + trainer yields the same loss but a different perf term.
    assert on_scoring["loss"] == pytest.approx(off_scoring["loss"])
    assert on_scoring["perf"] != pytest.approx(off_scoring["perf"])


def test_load_environment_accepts_baseline_relative_overrides(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-token")
    env = load_environment(baseline_relative_perf=True, perf_baseline_loss=7.5)
    assert env.taskset.curator.baseline_relative_perf is True
    assert env.taskset.curator.perf_baseline_loss == 7.5
