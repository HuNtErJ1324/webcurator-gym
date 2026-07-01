from __future__ import annotations

import asyncio
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import verifiers.v1 as vf
from pydantic import ValidationError
from verifiers.v1 import graph
from verifiers.v1.decorators import discover_decorated

from pretrain_data_curator.corpus import CorpusBuilder, DocumentFilter
from pretrain_data_curator.eval_corpus import DEFAULT_EVAL_CORPUS
from pretrain_data_curator.hf_access import (
    DatasetAccessError,
    FetchKey,
    HuggingFaceDatasetClient,
    RetryPolicy,
    classify_exception,
    loop_local_semaphore,
    parse_cutoff,
)
from pretrain_data_curator.leakage import LeakageDetector, _stable_hash32
from pretrain_data_curator.models import (
    CostLedger,
    CuratorConfig,
    FilterSpec,
    Manifest,
    ProxyStudentConfig,
    Source,
)
from pretrain_data_curator.pretrain_data_curator import load_environment
from pretrain_data_curator.rewards import CuratorScorer
from pretrain_data_curator.rollout_state import (
    STATE_SCHEMA_VERSION,
    CuratorState,
    RolloutStore,
)
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
    SandboxProxyTrainer,
    TrainerError,
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


@dataclass
class FakeDatasetInfo:
    id: str
    last_modified: datetime | None
    downloads: int = 0
    likes: int = 0
    tags: list[str] | None = None


class FakeClient:
    """In-memory HF stand-in: cutoff-relevant search + canned documents."""

    def __init__(self) -> None:
        self.search_calls: list[tuple[str, int]] = []
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

    def search_datasets(self, query: str, scan_limit: int) -> list[FakeDatasetInfo]:
        self.search_calls.append((query, scan_limit))
        return [
            FakeDatasetInfo(
                id="good/encyclopedia",
                last_modified=datetime(2024, 1, 4, tzinfo=timezone.utc),
                downloads=5000,
                likes=120,
                tags=["encyclopedia", "text"],
            ),
            FakeDatasetInfo(
                id="good/science",
                last_modified=datetime(2024, 3, 4, tzinfo=timezone.utc),
                downloads=3000,
                likes=80,
                tags=["science"],
            ),
            FakeDatasetInfo(
                id="too/new",
                last_modified=datetime(2025, 6, 4, tzinfo=timezone.utc),
                downloads=9000,
                likes=400,
                tags=["new"],
            ),
            FakeDatasetInfo(
                id="noisy/symbols",
                last_modified=datetime(2023, 5, 10, tzinfo=timezone.utc),
                downloads=10,
                likes=1,
                tags=["misc"],
            ),
        ]

    def sample_documents(self, dataset_id, config, split, text_field, n):
        self.sample_calls.append(dataset_id)
        return list(self._docs.get(dataset_id, []))[:n]


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


def _scorer(trainer, *, config=None, corpus_builder=None, leakage=None) -> CuratorScorer:
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
    client = object.__new__(HuggingFaceDatasetClient)
    client._token = "test-token"

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
    client = object.__new__(HuggingFaceDatasetClient)
    client._token = "test-token"

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


def test_load_environment_uses_declarative_docker_runtime_for_docker_trainer():
    docker_env = load_environment(
        use_real_trainer=True,
        proxy_student={"trainer_backend": "docker", "gpu_count": 1},
    )
    assert docker_env.harness.config.env == {
        "UV_REINSTALL_PACKAGE": "pydantic-core"
    }
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
                "trainer_backend": "docker",
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
    docs = ["short", "a much longer high quality document about science and history", "$$$$$"]
    f = DocumentFilter()
    kept = f.apply(docs, [FilterSpec(kind="min_chars", params={"value": 10})])
    assert "short" not in kept
    cleaned = f.apply(docs, [FilterSpec(kind="max_symbol_ratio", params={"value": 0.3})])
    assert "$$$$$" not in cleaned


def test_corpus_builder_applies_filters_and_sampling():
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
    corpus = builder.build(manifest)
    assert len(corpus.documents) == 3
    assert corpus.total_tokens > 0


def test_weight_proportional_sampling_allocates_correct_proportions():
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
    corpus = builder.build(manifest)
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


def test_weight_proportional_explicit_max_tokens_overrides_when_tighter():
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
    corpus = builder.build(manifest)
    # Source A: capped at explicit 30 tokens (tighter than the 2000-token weight target).
    assert corpus.sources[0].tokens <= 30
    # Source B: weight-derived 1000 tokens (no explicit cap); est_docs = 1000//250 = 4 docs (24 tokens).
    assert corpus.sources[1].tokens <= 1000


def test_weight_proportional_all_zero_weights_falls_back_to_uncapped():
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
    corpus = builder.build(manifest)
    # No weight-derived cap applied -> all 10 docs per source are kept.
    assert len(corpus.sources[0].documents) == n_docs
    assert len(corpus.sources[1].documents) == n_docs


def test_weight_proportional_single_source_gets_full_budget():
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
    corpus = builder.build(manifest)
    # All n_docs fetched; their token total (n_docs*6=300) fits within the budget.
    assert corpus.sources[0].tokens <= n_docs * 250
    assert len(corpus.sources[0].documents) == n_docs


def test_fetch_count_capped_at_sample_docs_per_source_for_large_target():
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
    corpus = builder.build(manifest)
    assert len(corpus.sources[0].documents) == cap


def test_fetch_count_proportional_to_small_token_target():
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
    corpus = builder.build(manifest)
    assert len(corpus.sources[0].documents) == 2
    assert len(corpus.sources[0].documents) < cap


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


# --- Tier D1: GPU-request correctness (config-level fail-loud guard) --------


def test_proxy_student_gpu_defaults_are_a_valid_request():
    # The default proxy student requests a real GPU sandbox: a concrete H100
    # gpu_type and vm=True, so the prime_sandboxes validator (which needs ALL of
    # gpu_count>0 + gpu_type + vm) is satisfied — the bug that shipped was an
    # invalid request (gpu_type=None, no vm) that degraded every rollout to perf=0.
    cfg = ProxyStudentConfig()
    assert cfg.gpu_count == 1
    assert cfg.gpu_type == "H100"
    assert cfg.vm is True
    # H200 is reachable via config, and CPU-only sandboxes are valid too.
    assert ProxyStudentConfig(gpu_type="H200").gpu_type == "H200"
    assert ProxyStudentConfig(gpu_count=0, gpu_type=None).gpu_count == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gpu_type": None},  # gpu_count default 1, no gpu_type -> invalid (the shipped bug)
        {"gpu_type": ""},  # empty gpu_type is not a valid value
        {"vm": False},  # gpu_count default 1 but vm not enabled -> invalid
        {"gpu_count": 2, "gpu_type": None},
        {"gpu_count": 0},  # gpu_type defaults to H100 -> requires gpu_count > 0
    ],
)
def test_proxy_student_rejects_invalid_gpu_combo(kwargs):
    # The fail-loud config validator rejects an unschedulable GPU request AT
    # CONSTRUCTION, before any sandbox/SDK call, instead of letting the SDK raise
    # a ValidationError that the rubric silently degrades to the perf=0 sentinel.
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
    cfg = ProxyStudentConfig(train_token_budget=budget, batch_size=batch, block_size=block)
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
    assert ProxyStudentConfig(max_corpus_chars=123_456).effective_max_corpus_chars == 123_456
    # 1e9 tokens * 4 chars/token exceeds the 2e9 ceiling -> clamped.
    assert ProxyStudentConfig(train_token_budget=1_000_000_000).effective_max_corpus_chars == (
        2_000_000_000
    )


def test_effective_timeout_minutes_scales_and_is_bounded():
    # Default budget keeps the historical 30-minute timeout; a large budget grows
    # it; it never exceeds the Prime 24h (1440-minute) platform max; an explicit
    # value overrides; and the field itself rejects > 1440.
    assert ProxyStudentConfig().effective_timeout_minutes == 30
    big = ProxyStudentConfig(train_token_budget=300_000_000)
    assert 30 < big.effective_timeout_minutes <= 1440
    assert ProxyStudentConfig(timeout_minutes=45).effective_timeout_minutes == 45
    with pytest.raises(ValidationError):
        ProxyStudentConfig(timeout_minutes=1441)


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
    RolloutStore.init(state, Manifest(), CostLedger())
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

    async def materialize(self, manifest, state):
        self.materialize_calls += 1
        return await super().materialize(manifest, state)


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
    assert len(funcs) == 17  # 3 rewards + 14 diagnostic metrics
    await asyncio.gather(*[f(trace) for f in funcs])
    assert builder.materialize_calls == 1
    assert trainer.calls == 1


# --- Tier D: external-data robustness; structured errors + sentinel --------


@pytest.mark.parametrize(
    "exc_factory,expected_kind",
    [
        (lambda: __import__("datasets.exceptions", fromlist=["DatasetNotFoundError"]).DatasetNotFoundError("nope"), "missing"),
        (lambda: ValueError("Unknown split 'bad'. Should be one of ['train']."), "bad_split"),
        (lambda: KeyError("text_field"), "bad_field"),
        (lambda: PermissionError("401 Client Error: Unauthorized for url"), "auth"),
        (lambda: ConnectionError("Connection refused"), "network"),
        (lambda: TimeoutError("timed out"), "timeout"),
    ],
)
@pytest.mark.asyncio
async def test_fetch_failures_are_structured_and_scoring_degrades(
    exc_factory, expected_kind
):
    client = FailingClient(exc_factory)
    curator = await _make(client=client, fetch_max_attempts=1, fetch_timeout_seconds=2.0)

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

    policy = RetryPolicy(attempts=1, timeout=0.05)
    builder = CorpusBuilder(client=_SlowClient(), retry_policy=policy)
    state = CuratorState()
    RolloutStore.init(state, Manifest(), CostLedger())
    docs, error = await builder.fetch_source_docs(
        state, FetchKey("a/b", None, "train", "text", 4)
    )
    assert docs == []
    assert error["error_kind"] == "timeout"


def test_classify_exception_kinds():
    assert classify_exception(DatasetAccessError("x", kind="auth")) == "auth"
    assert classify_exception(KeyError("col")) == "bad_field"
    assert classify_exception(ConnectionError("boom")) == "network"
    assert classify_exception(TimeoutError("t")) == "timeout"
    assert classify_exception(RuntimeError("something weird")) == "unknown"


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


# --- Tier H: sandbox trainer lifecycle (mocked) ----------------------------


def _corpus_with_text():
    from pretrain_data_curator.corpus import CuratedCorpus, SourceCorpus

    return CuratedCorpus(
        sources=[SourceCorpus("a/b", None, 1.0, ["hello world " * 30])]
    )


class _FakeSandbox:
    id = "sandbox-1"


class _FakeSandboxClient:
    def __init__(self, mode):
        self.mode = mode
        self.deleted = False
        self.delete_fails = False

    async def create(self, request):
        return _FakeSandbox()

    async def wait_for_creation(self, sid):
        return None

    async def upload_bytes(self, sid, path, data, name):
        return None

    async def execute_command(self, sid, cmd, timeout):
        if self.mode == "success":
            payload = json.dumps(
                {"loss": 2.5, "accuracy": 0.4, "flops": 1e9, "tokens_trained": 1000}
            )
            return SimpleNamespace(
                stdout="RESULT_JSON " + payload, stderr="", exit_code=0
            )
        if self.mode == "fail":
            return SimpleNamespace(
                stdout="", stderr="Traceback ... CUDA out of memory boom", exit_code=1
            )
        if self.mode == "timeout":
            raise asyncio.TimeoutError()
        raise AssertionError(self.mode)

    async def delete(self, sid):
        self.deleted = True
        if self.delete_fails:
            raise RuntimeError("delete failed")


def _sandbox_trainer(client):
    return SandboxProxyTrainer(
        client_factory=lambda: client,
        request_factory=lambda config, name: object(),
    )


@pytest.mark.asyncio
async def test_sandbox_trainer_success():
    client = _FakeSandboxClient("success")
    result = await _sandbox_trainer(client).train_and_eval(
        _corpus_with_text(), ProxyStudentConfig()
    )
    assert result.loss == 2.5
    assert result.backend == "sandbox"
    assert result.cleanup_error is None
    assert client.deleted is True


@pytest.mark.asyncio
async def test_sandbox_trainer_nonzero_exit_surfaces_stderr_tail():
    client = _FakeSandboxClient("fail")
    with pytest.raises(TrainerError) as excinfo:
        await _sandbox_trainer(client).train_and_eval(
            _corpus_with_text(), ProxyStudentConfig()
        )
    assert "boom" in excinfo.value.stderr_tail
    assert client.deleted is True  # cleanup still ran


@pytest.mark.asyncio
async def test_sandbox_trainer_timeout_surfaces():
    client = _FakeSandboxClient("timeout")
    with pytest.raises(TrainerError) as excinfo:
        await _sandbox_trainer(client).train_and_eval(
            _corpus_with_text(), ProxyStudentConfig()
        )
    assert "timed out" in str(excinfo.value)
    assert client.deleted is True


@pytest.mark.asyncio
async def test_sandbox_trainer_cleanup_failure_is_surfaced():
    client = _FakeSandboxClient("success")
    client.delete_fails = True
    result = await _sandbox_trainer(client).train_and_eval(
        _corpus_with_text(), ProxyStudentConfig()
    )
    assert result.cleanup_error is not None
    assert "delete failed" in result.cleanup_error


@pytest.mark.asyncio
async def test_sandbox_trainer_empty_corpus_short_circuits():
    from pretrain_data_curator.corpus import CuratedCorpus

    client = _FakeSandboxClient("success")
    result = await _sandbox_trainer(client).train_and_eval(
        CuratedCorpus(sources=[]), ProxyStudentConfig()
    )
    assert result.loss == float("inf")


# --- Tier D3: the real CreateSandboxRequest passes the SDK validator --------
# This is the gap that let the bug ship: the existing sandbox tests inject a fake
# request via request_factory=lambda ...: object(), so the REAL request was never
# validated. These build the actual SDK model from the config.


def test_default_config_builds_a_real_valid_sandbox_request():
    sandboxes = pytest.importorskip("prime_sandboxes")
    # No request_factory -> SandboxProxyTrainer builds a REAL CreateSandboxRequest.
    request = SandboxProxyTrainer()._make_request(ProxyStudentConfig(), "proxy-student")
    assert isinstance(request, sandboxes.CreateSandboxRequest)
    assert request.gpu_count == 1
    assert request.gpu_type == "H100"
    assert request.vm is True
    assert request.timeout_minutes == 30


def test_large_budget_h200_config_builds_a_real_valid_sandbox_request():
    sandboxes = pytest.importorskip("prime_sandboxes")
    cfg = ProxyStudentConfig(gpu_type="H200", train_token_budget=300_000_000)
    request = SandboxProxyTrainer()._make_request(cfg, "proxy-student")
    assert isinstance(request, sandboxes.CreateSandboxRequest)
    assert request.gpu_type == "H200"
    # The budget-derived timeout flows into the request and stays under the cap.
    assert request.timeout_minutes == cfg.effective_timeout_minutes
    assert 30 < request.timeout_minutes <= 1440


def test_sdk_rejects_the_originally_shipped_broken_request():
    # The original trainer built CreateSandboxRequest(gpu_count=1, gpu_type=None)
    # with no vm — the SDK validator rejects exactly that, which is why every real
    # rollout silently degraded to the infinite-loss sentinel. Our config validator
    # now stops this combo one layer earlier (test above), but pin the SDK behavior.
    sandboxes = pytest.importorskip("prime_sandboxes")
    with pytest.raises(ValidationError):
        sandboxes.CreateSandboxRequest(
            name="x", docker_image="img", gpu_count=1, gpu_type=None
        )


@pytest.mark.asyncio
async def test_sandbox_payload_steps_follow_train_token_budget():
    # The budget-derived training length must reach the sandbox: the script reads
    # cfg["steps"] and bills FLOPs off steps*batch*block, so this is what scales
    # tokens_trained and FLOP cost with the budget.
    client = _RecordingSandboxClient("success")
    trainer = SandboxProxyTrainer(
        client_factory=lambda: client,
        request_factory=lambda config, name: object(),
    )
    cfg = ProxyStudentConfig(train_token_budget=300_000_000)
    await trainer.train_and_eval(_corpus_with_text(), cfg)
    uploaded = json.loads(client.uploads["/workspace/config.json"].decode("utf-8"))
    assert uploaded["steps"] == cfg.effective_steps == 73_243


@pytest.mark.asyncio
async def test_sandbox_payload_carries_record01_recipe_fields():
    # The record_01 optimizer schedule + regularization + averaging knobs must reach
    # the sandbox: the script reads them to build AdamW(betas, eps, weight_decay),
    # the warmup+cosine LR schedule, the grad clip, and the n_train_runs averaging.
    client = _RecordingSandboxClient("success")
    trainer = SandboxProxyTrainer(
        client_factory=lambda: client,
        request_factory=lambda config, name: object(),
    )
    cfg = ProxyStudentConfig(
        n_train_runs=3, weight_decay=0.05, grad_clip=2.0, warmup_steps=7,
        lr_min_ratio=0.2, adam_beta1=0.8, adam_beta2=0.99, adam_eps=1e-9,
    )
    await trainer.train_and_eval(_corpus_with_text(), cfg)
    payload = json.loads(client.uploads["/workspace/config.json"].decode("utf-8"))
    assert payload["n_train_runs"] == 3
    assert payload["weight_decay"] == 0.05
    assert payload["grad_clip"] == 2.0
    assert payload["lr_min_ratio"] == 0.2
    assert payload["adam_beta1"] == 0.8
    assert payload["adam_beta2"] == 0.99
    assert payload["adam_eps"] == 1e-9
    # warmup_steps is the budget-aware EFFECTIVE value (explicit 7, clamped to steps).
    assert payload["warmup_steps"] == cfg.effective_warmup_steps == 7


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
    assert cfg.effective_warmup_steps == min(256, max(1, cfg.effective_steps // 10)) == 20
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
    corpus = CuratedCorpus(sources=[SourceCorpus("a/b", None, 1.0, ["word " * 4000])])
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


# --- Tier D4: end-to-end proof the budget-derived cap/timeout actually bite -


_OLD_CORPUS_CHAR_CAP = 5_000_000


@pytest.mark.asyncio
async def test_large_budget_uploads_corpus_beyond_old_5m_cap():
    # Prove the derived cap TRUNCATES LESS (uploads more) end to end, not just
    # that the property returns a bigger number: a corpus longer than the old 5M
    # cap is truncated to 5M under the default budget, but uploaded in FULL under
    # a large budget whose effective_max_corpus_chars exceeds the corpus length.
    from pretrain_data_curator.corpus import CuratedCorpus, SourceCorpus

    long_doc = "a" * (_OLD_CORPUS_CHAR_CAP + 1_000_000)  # 6,000,000 ASCII chars
    corpus = CuratedCorpus(sources=[SourceCorpus("a/b", None, 1.0, [long_doc])])

    # Default budget: the historical 5M cap still truncates the extra material.
    default_client = _RecordingSandboxClient("success")
    default_trainer = SandboxProxyTrainer(
        client_factory=lambda: default_client,
        request_factory=lambda config, name: object(),
    )
    await default_trainer.train_and_eval(corpus, ProxyStudentConfig())
    assert len(default_client.uploads["/workspace/corpus.txt"]) == _OLD_CORPUS_CHAR_CAP

    # Large budget: the derived cap exceeds the corpus, so the FULL 6M chars are
    # genuinely uploaded — the ~1M chars past the old cap are NOT silently dropped.
    big_cfg = ProxyStudentConfig(train_token_budget=300_000_000)
    assert big_cfg.effective_max_corpus_chars > len(long_doc)
    big_client = _RecordingSandboxClient("success")
    big_trainer = SandboxProxyTrainer(
        client_factory=lambda: big_client,
        request_factory=lambda config, name: object(),
    )
    await big_trainer.train_and_eval(corpus, big_cfg)
    uploaded = len(big_client.uploads["/workspace/corpus.txt"])
    assert uploaded == len(long_doc)  # full corpus, untruncated
    assert uploaded > _OLD_CORPUS_CHAR_CAP  # genuinely beyond the old 5M cap


class _ExecTimeoutRecordingClient(_FakeSandboxClient):
    """Records the per-command timeout train_and_eval passes to execute_command."""

    def __init__(self, mode="success"):
        super().__init__(mode)
        self.execute_timeout = None

    async def execute_command(self, sid, cmd, timeout):
        self.execute_timeout = timeout
        return await super().execute_command(sid, cmd, timeout)


@pytest.mark.asyncio
async def test_command_timeout_uses_budget_derived_effective_timeout(monkeypatch):
    # The training command's wall-clock timeout must be the budget-DERIVED
    # effective timeout (seconds), not the now-None raw timeout_minutes nor the
    # historical 30-min default. budget 300M -> effective_steps=73243 ->
    # effective_train_tokens=300,003,328 -> 15 + ceil(.../500k) = 616 minutes.
    cfg = ProxyStudentConfig(train_token_budget=300_000_000)
    assert cfg.timeout_minutes is None  # derived, never set explicitly
    assert cfg.effective_timeout_minutes == 616

    monkeypatch.setattr(
        "pretrain_data_curator.trainer._nanogpt_train_script",
        lambda: "print('stub train script')",
    )
    client = _ExecTimeoutRecordingClient("success")
    trainer = SandboxProxyTrainer(
        client_factory=lambda: client,
        request_factory=lambda config, name: object(),
    )
    await trainer.train_and_eval(_corpus_with_text(), cfg)
    # execute_command was given the derived command timeout (effective * 60s); the
    # outer asyncio.wait_for uses this same value + the documented 30s slack.
    assert client.execute_timeout == cfg.effective_timeout_minutes * 60 == 36_960

    # The same derived timeout also flows into the REAL sandbox request.
    sandboxes = pytest.importorskip("prime_sandboxes")
    request = SandboxProxyTrainer()._make_request(cfg, "proxy-student")
    assert isinstance(request, sandboxes.CreateSandboxRequest)
    assert request.timeout_minutes == cfg.effective_timeout_minutes == 616


def test_sdk_rejects_gpu_count_with_vm_disabled():
    # Pins the SDK precondition our _check_gpu_request mirrors: gpu_count > 0 with
    # vm=False is rejected (prime_sandboxes/models.py:110-111). A future SDK that
    # dropped this would fail here, flagging our validator as stale.
    sandboxes = pytest.importorskip("prime_sandboxes")
    with pytest.raises(ValidationError):
        sandboxes.CreateSandboxRequest(
            name="x", docker_image="img", gpu_count=1, gpu_type="H100", vm=False
        )


def test_sdk_rejects_gpu_type_without_gpu_count():
    # Pins the other precondition: gpu_type set with gpu_count == 0 is rejected
    # (prime_sandboxes/models.py:112-113), mirrored by our gpu_count==0 -> gpu_type
    # must be None rule.
    sandboxes = pytest.importorskip("prime_sandboxes")
    with pytest.raises(ValidationError):
        sandboxes.CreateSandboxRequest(
            name="x", docker_image="img", gpu_count=0, gpu_type="H100"
        )


# --- Tier J/K: zero-weight telemetry metrics do not affect reward ----------


def test_telemetry_metrics_are_zero_weight():
    # The external-failure diagnostics are registered as @vf.metric, never
    # @vf.reward, so they are recorded but never summed into the reward — the v1
    # structural equivalent of the v0 zero reward weight.
    taskset = _Curator().taskset
    reward_names = {f.__name__ for f in discover_decorated(taskset, "reward")}
    metric_names = {f.__name__ for f in discover_decorated(taskset, "metric")}
    for name in ("tool_error_count", "external_failure"):
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
    assert "```json" in SYSTEM_PROMPT
    assert '"sources"' in SYSTEM_PROMPT
    assert "curator_" not in SYSTEM_PROMPT  # no stale MCP tool references


def test_system_prompt_manifest_example_parses():
    # The schema example embedded in the prompt must itself parse into a Manifest,
    # so the documented contract and the parser cannot silently drift apart.
    manifest = parse_manifest(SYSTEM_PROMPT, default_token_budget=1_000_000)
    assert manifest is not None
    assert manifest.sources  # the example carries at least one source
    assert manifest.sources[0].text_field is None


# --- Tier N: state schema version + canonical hash -------------------------


@pytest.mark.asyncio
async def test_state_carries_schema_version():
    curator = await _make()
    assert RolloutStore.schema_version(curator.state) == STATE_SCHEMA_VERSION


@pytest.mark.asyncio
async def test_canonical_hash_stable_for_equal_state():
    curator = await _make()
    s1 = await _finalized(curator, sources=("good/encyclopedia",))
    h1 = RolloutStore.canonical_hash(s1)
    await curator.reset()
    s2 = await _finalized(curator, sources=("good/encyclopedia",))
    assert RolloutStore.canonical_hash(s2) == h1
    await curator.reset()
    s3 = await _finalized(curator, sources=("good/encyclopedia", "good/science"))
    assert RolloutStore.canonical_hash(s3) != h1


@pytest.mark.asyncio
async def test_canonical_state_dump_keys_and_hash():
    # Pin the EXACT serialized keys of the canonical state dump + manifest and the
    # hash formula, so a schema-key rename or canonical-hash drift fails loudly.
    import hashlib

    curator = await _make()
    state = await _finalized(curator, sources=("good/encyclopedia",))
    canonical = RolloutStore.canonical_state(state)
    assert set(canonical) == {"state_schema_version", "manifest", "finalized"}
    assert canonical["state_schema_version"] == STATE_SCHEMA_VERSION
    assert canonical["finalized"] is True
    assert set(canonical["manifest"]) == {"token_budget", "sources"}
    # The hash is the blake2b-16 of the canonical JSON (sorted, compact).
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    assert (
        RolloutStore.canonical_hash(state)
        == hashlib.blake2b(encoded.encode("utf-8"), digest_size=16).hexdigest()
    )


# --- Additional coverage: cutoff/query helpers -----------------------------


def test_parse_cutoff_forms():
    assert parse_cutoff("2024-12-31").date() == date(2024, 12, 31)
    dt = parse_cutoff("2024-06-01T12:00:00Z")
    assert dt.year == 2024 and dt.month == 6
    assert parse_cutoff(date(2023, 1, 1)).tzinfo is not None
    with pytest.raises(ValueError):
        parse_cutoff("   ")


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


class _RecordingSandboxClient(_FakeSandboxClient):
    """Records uploaded files and counts sandbox creations."""

    def __init__(self, mode="success"):
        super().__init__(mode)
        self.uploads = {}
        self.creates = 0

    async def create(self, request):
        self.creates += 1
        return await super().create(request)

    async def upload_bytes(self, sid, path, data, name):
        self.uploads[path] = data
        return None


@pytest.mark.asyncio
async def test_sandbox_trainer_uploads_exactly_first_n_val_tokens(tmp_path):
    client = _RecordingSandboxClient("success")
    loader = ValTokenLoader(
        ValidationSetConfig(val_tokens=16),
        download_fn=_shard_download_fn(tmp_path, list(range(100))),
    )
    trainer = SandboxProxyTrainer(
        client_factory=lambda: client,
        request_factory=lambda config, name: object(),
        val_loader=loader,
    )
    result = await trainer.train_and_eval(_corpus_with_text(), ProxyStudentConfig())
    assert result.backend == "sandbox"
    # The held-out val tokens were uploaded as header-free uint16 = first N tokens.
    assert "/workspace/val.bin" in client.uploads
    expected = np.asarray(range(16), dtype="<u2").tobytes()
    assert client.uploads["/workspace/val.bin"] == expected
    # The student is told to use the GPT-2 BPE tokenizer of the held-out set.
    cfg = json.loads(client.uploads["/workspace/config.json"].decode("utf-8"))
    assert cfg["tokenizer"] == "gpt2"


@pytest.mark.asyncio
async def test_sandbox_trainer_val_fetch_failure_skips_sandbox():
    client = _RecordingSandboxClient("success")

    def boom(dataset_id, filename, repo_type):
        raise ConnectionError("hub down")

    loader = ValTokenLoader(
        ValidationSetConfig(),
        download_fn=boom,
        retry_policy=RetryPolicy(attempts=1, timeout=2.0),
    )
    trainer = SandboxProxyTrainer(
        client_factory=lambda: client,
        request_factory=lambda config, name: object(),
        val_loader=loader,
    )
    # The val set is resolved before any GPU sandbox is provisioned: a fetch
    # failure raises (no sandbox created) rather than wasting a sandbox.
    with pytest.raises(DatasetAccessError):
        await trainer.train_and_eval(_corpus_with_text(), ProxyStudentConfig())
    assert client.creates == 0


@pytest.mark.asyncio
async def test_scorer_degrades_val_fetch_failure_to_sentinel():
    # End-to-end degrade: a held-out val-set fetch failure on the real-trainer
    # path collapses to the infinite-loss sentinel (Perf -> 0), not a crash.
    def boom(dataset_id, filename, repo_type):
        raise ConnectionError("hub down")

    loader = ValTokenLoader(
        ValidationSetConfig(),
        download_fn=boom,
        retry_policy=RetryPolicy(attempts=1, timeout=2.0),
    )
    trainer = SandboxProxyTrainer(
        client_factory=lambda: _RecordingSandboxClient("success"),
        request_factory=lambda config, name: object(),
        val_loader=loader,
    )
    scorer = _scorer(trainer)
    state = CuratorState()
    RolloutStore.init(state, Manifest(), CostLedger())
    result = await scorer._train(_corpus_with_text(), state)
    assert result.loss == float("inf")
    assert result.backend == "error"
    assert RolloutStore.has_external_failure(state)


@pytest.mark.asyncio
async def test_scorer_degrades_sandbox_trainer_error_to_sentinel():
    # Sibling to the val-FETCH degrade above, for the OTHER degrade path the CE
    # fix relies on: a sandbox TrainerError (the nonzero exit produced when the
    # in-sandbox plan_val_windows raises on a degenerate <=1-token val set) must
    # ALSO collapse to the infinite-loss sentinel in the scorer, not crash or
    # yield a bogus good loss.
    trainer = SandboxProxyTrainer(
        # "fail" mode returns a nonzero exit -> SandboxProxyTrainer raises a
        # TrainerError carrying the sandbox stderr tail.
        client_factory=lambda: _FakeSandboxClient("fail"),
        request_factory=lambda config, name: object(),
        val_loader=None,
    )
    scorer = _scorer(trainer)
    state = CuratorState()
    RolloutStore.init(state, Manifest(), CostLedger())
    result = await scorer._train(_corpus_with_text(), state)
    assert result.loss == float("inf")
    assert result.backend == "error"
    assert RolloutStore.has_external_failure(state)
    assert RolloutStore.tool_error_count(state) >= 1
    # The degrade preserved the sandbox failure detail (the stderr tail is only
    # attached on the TrainerError path), proving it was the trainer-error degrade
    # rather than a silently-swallowed good loss.
    assert "boom" in (RolloutStore.trainer_error(state) or "")


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
            json.dumps({"argv": ["datasets", "ls", "--search", "code"], "exit": 0, "bytes": 400}),
            json.dumps({"argv": ["datasets", "info", "a/b"], "exit": 0, "bytes": 80}),
            json.dumps({"argv": ["download", "a/b", "--repo-type", "dataset"], "exit": 0, "bytes": 4000}),
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
    log = json.dumps({"argv": ["datasets", "ls", "--search", "x"], "exit": 1, "bytes": 0})
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
    cmds = hf_meter.extract_hf_commands("hf datasets ls --search a && hf datasets info b/c")
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
        json.dumps({"argv": ["datasets", "ls", "--search", "x"], "exit": 0, "bytes": 400})
        + "\n"
        + json.dumps({"argv": ["datasets", "info", "good/encyclopedia"], "exit": 0, "bytes": 40})
        + "\n"
    )
    await curator.taskset.finalize(curator.task, trace, _FakeRuntime(log.encode()))

    state = curator.state
    assert RolloutStore.is_finalized(state)
    manifest = RolloutStore.manifest(state)
    assert {s.dataset_id for s in manifest.sources} == {"good/encyclopedia", "good/science"}
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

    assert RolloutStore.is_finalized(curator.state), "fallback must finalize the rollout"
    manifest = RolloutStore.manifest(curator.state)
    assert manifest.sources, "fallback manifest must be non-empty"
    ids = {s.dataset_id for s in manifest.sources}
    # The two explicitly-inspected ids must appear; ls-output ids may also appear.
    assert "meta-math/MetaMathQA" in ids
    assert "EleutherAI/hendrycks_math" in ids
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
        "```json\n"
        '{"sources": [{"id": "good/science", "weight": 3.0}]}\n'
        "```"
    )
    graph.prepare_turn(trace, [
        vf.SystemMessage(content="sys"),
        vf.UserMessage(content="go"),
        # Replay the tool-call turns so the graph prefix matches.
        *[
            msg
            for tc_cmd, tc_result in calls
            for msg in [
                vf.AssistantMessage(
                    content="",
                    tool_calls=[vf.ToolCall(
                        id="tc0", name="bash",
                        arguments=json.dumps({"command": tc_cmd})
                    )],
                ),
                vf.ToolMessage(tool_call_id="tc0", content=tc_result),
            ]
        ],
    ]).commit(
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
        prompt = CuratorTaskset(
            CuratorTasksetConfig(id="test", max_turns=max_turns)
        ).load_tasks()[0].system_prompt
        assert f"{max_turns} turns" in prompt
        assert "Each bash tool call uses one turn" in prompt
        assert "MUST perform at most 2 discovery rounds" in prompt
        # Commit mechanics: plain text, no bash call.
        assert "HOW TO COMMIT" in prompt
        assert "plain text response (NO bash tool call)" in prompt
        assert "Do not call bash to print the manifest" in prompt
        assert "scores zero" in prompt

    smoke_prompt = CuratorTaskset(
        CuratorTasksetConfig(id="test", max_turns=25, scan_limit=10)
    ).load_tasks()[0].system_prompt
    assert "25 turns" in smoke_prompt
    assert "MUST perform at most 2 discovery rounds" in smoke_prompt

    benchmark_prompt = CuratorTaskset(
        CuratorTasksetConfig(id="test", max_turns=64, scan_limit=200)
    ).load_tasks()[0].system_prompt
    assert "64 turns" in benchmark_prompt
    assert "MUST perform at most 10 discovery rounds (<=20 bash calls)" in benchmark_prompt
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
    assert (
        "fi; hf datasets ls" in low
    )  # bootstrap and useful discovery share one turn
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



def test_backend_selection_prime_uses_prime_path_unchanged():
    ts = _real_trainer_taskset()  # default trainer_backend == "prime"
    ts._ensure()
    trainer = ts._trainer
    assert isinstance(trainer, SandboxProxyTrainer)
    # No factories => the prime path (prime_sandboxes is imported lazily on use).
    assert trainer._client_factory is None
    assert trainer._request_factory is None
    assert ts.curator.proxy_student.trainer_backend == "prime"


def test_backend_default_is_heuristic_and_prime_selector():
    # The default selector is "prime" and the default (use_real_trainer False) path
    # still yields the heuristic trainer — both unchanged.
    assert CuratorConfig().proxy_student.trainer_backend == "prime"
    assert ProxyStudentConfig().trainer_backend == "prime"
    ts = _real_trainer_taskset(use_real_trainer=False)
    ts._ensure()
    assert isinstance(ts._trainer, HeuristicProxyTrainer)


def test_docker_backend_relaxes_prime_only_guards():
    # A docker config the PRIME validator would reject is ACCEPTED: vm=False with a
    # GPU and an explicit > 24h timeout (a self-hosted host has neither limit).
    cfg = ProxyStudentConfig(trainer_backend="docker", vm=False, timeout_minutes=5000)
    assert cfg.trainer_backend == "docker"
    assert cfg.vm is False
    assert cfg.timeout_minutes == 5000
    assert cfg.effective_timeout_minutes == 5000  # not clamped to 1440
    # gpu_type=None alongside a GPU is fine too (docker ignores gpu_type).
    cfg2 = ProxyStudentConfig(
        trainer_backend="docker", gpu_count=1, gpu_type=None, vm=False
    )
    assert cfg2.gpu_count == 1 and cfg2.gpu_type is None


def test_prime_backend_guards_still_enforced():
    # The same combos remain rejected on the (default) prime backend.
    with pytest.raises(ValidationError):
        ProxyStudentConfig(vm=False)  # default prime, gpu_count=1
    with pytest.raises(ValidationError):
        ProxyStudentConfig(timeout_minutes=1441)  # default prime, > 24h
    with pytest.raises(ValidationError):
        ProxyStudentConfig(trainer_backend="prime", gpu_count=1, gpu_type=None)


def test_docker_image_default_is_backend_aware():
    # Prime keeps the historical image default (byte-identical); docker defaults to
    # the torch 2.7 / cuda 12.6 image; an explicit image wins on either backend.
    assert (
        ProxyStudentConfig().docker_image
        == "pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime"
    )
    assert ProxyStudentConfig(trainer_backend="docker").docker_image == (
        "pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime"
    )
    assert (
        ProxyStudentConfig(trainer_backend="docker", docker_image="me/img:1").docker_image
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
    worse = TrainResult(loss=baseline + 1.0, accuracy=0.0, flops=0.0, tokens_trained=0, backend="x")
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
    worse = TrainResult(loss=20.0, accuracy=0.0, flops=0.0, tokens_trained=0, backend="x")
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
