from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import date, datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from pretrain_data_curator.corpus import CorpusBuilder, DocumentFilter
from pretrain_data_curator.eval_corpus import DEFAULT_EVAL_CORPUS
from pretrain_data_curator.hf_access import (
    DatasetAccessError,
    FetchKey,
    RetryPolicy,
    classify_exception,
    loop_local_semaphore,
    parse_cutoff,
    query_variants,
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
from pretrain_data_curator.rewards import CuratorRubric
from pretrain_data_curator.rollout_state import STATE_SCHEMA_VERSION, RolloutStore
from pretrain_data_curator.trainer import (
    HeuristicProxyTrainer,
    SandboxProxyTrainer,
    TrainerError,
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


def _env(monkeypatch, *, client=None, **kwargs):
    monkeypatch.setenv("HF_TOKEN", "test-token")
    return load_environment(
        client=client or FakeClient(), cutoff_date="2024-12-31", **kwargs
    )


def test_load_environment_validates_hf_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    with pytest.raises(Exception, match="HF_TOKEN"):
        load_environment(client=FakeClient())


@pytest.mark.asyncio
async def test_search_filters_cutoff_and_caches(monkeypatch):
    env = _env(monkeypatch)
    state = {}
    await env.setup_state(state)

    payload = await env.search_datasets("encyclopedia science", state)

    assert "too/new" not in payload  # excluded after cutoff
    candidates = RolloutStore.candidates(state)
    assert "good/encyclopedia" in candidates
    assert "too/new" not in candidates
    ledger = RolloutStore.ledger(state)
    assert ledger.hub_calls >= 1 and ledger.web_queries >= 1


@pytest.mark.asyncio
async def test_set_source_requires_prior_search(monkeypatch):
    env = _env(monkeypatch)
    state = {}
    await env.setup_state(state)

    payload = await env.set_source("good/encyclopedia", state)
    assert "must be discovered" in payload

    await env.search_datasets("encyclopedia", state)
    payload = await env.set_source("good/encyclopedia", state, weight=2.0)
    assert '"num_sources":1' in payload
    manifest = RolloutStore.manifest(state)
    assert manifest.sources[0].dataset_id == "good/encyclopedia"


@pytest.mark.asyncio
async def test_finalize_then_reward_aggregation(monkeypatch):
    env = _env(monkeypatch)
    state = {}
    await env.setup_state(state)
    await env.search_datasets("encyclopedia science", state)
    await env.set_source("good/encyclopedia", state, weight=1.0)
    await env.set_source("good/science", state, weight=1.0)
    finalize = await env.finalize_manifest(state)
    assert '"finalized":true' in finalize

    scoring = await env.curator_rubric._prepared(state)
    assert 0.0 <= scoring["perf"] <= 1.0
    assert 0.0 <= scoring["quality"] <= 1.0
    assert scoring["diversity"] > 0.0  # two distinct sources
    assert scoring["flops"] > 0.0
    ledger = RolloutStore.ledger(state)
    assert ledger.train_flops > 0.0  # FLOPs charged back to the ledger


@pytest.mark.asyncio
async def test_empty_manifest_scores_zero_perf(monkeypatch):
    env = _env(monkeypatch)
    state = {}
    await env.setup_state(state)
    scoring = await env.curator_rubric._prepared(state)
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


async def _finalized_state(env, *, sources=("good/encyclopedia", "good/science")):
    # Minimal prompt/completion so the rubric's score_objects works on a plain
    # dict state (the real rollout path populates these).
    state: dict = {"prompt": [], "completion": []}
    await env.setup_state(state)
    await env.search_datasets("encyclopedia science", state)
    for ds in sources:
        await env.set_source(ds, state, weight=1.0)
    await env.finalize_manifest(state)
    return state


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
        {"leakage_severe_threshold": 1.5},
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
        {"learning_rate": 0.0},
    ],
)
def test_proxy_student_config_rejects_invalid(kwargs):
    with pytest.raises(ValidationError):
        ProxyStudentConfig(**kwargs)


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


# --- Tier C: info round-trips as a dict; token_budget override is live ------


@pytest.mark.asyncio
async def test_info_token_budget_override_as_dict(monkeypatch):
    env = _env(monkeypatch)
    state = {"info": {"token_budget": 555}}
    await env.setup_state(state)
    assert RolloutStore.manifest(state).token_budget == 555


@pytest.mark.asyncio
async def test_info_token_budget_override_as_json_string(monkeypatch):
    env = _env(monkeypatch)
    state = {"info": json.dumps({"token_budget": 777})}
    await env.setup_state(state)
    assert RolloutStore.manifest(state).token_budget == 777


@pytest.mark.asyncio
async def test_info_absent_uses_config_default(monkeypatch):
    env = _env(monkeypatch, token_budget=123456)
    state: dict = {}
    await env.setup_state(state)
    assert RolloutStore.manifest(state).token_budget == 123456


# --- Tier E: deterministic same-key cache; preview == score; cost once ------


@pytest.mark.asyncio
async def test_fetch_cache_same_key_identity_and_cost_once(monkeypatch):
    client = FakeClient()
    env = _env(monkeypatch, client=client)
    state = await _finalized_state(env, sources=("good/encyclopedia",))
    assert client.sample_calls == []  # nothing fetched until materialize
    tokens_before = RolloutStore.ledger(state).tokens
    hub_before = RolloutStore.ledger(state).hub_calls

    manifest = RolloutStore.manifest(state)
    corpus_a = await env.corpus_builder.materialize(manifest, state)
    assert client.sample_calls == ["good/encyclopedia"]  # fetched exactly once
    tokens_once = RolloutStore.ledger(state).tokens
    # The fetch charged one hub call and a positive token cost, exactly once.
    assert tokens_once > tokens_before
    assert RolloutStore.ledger(state).hub_calls == hub_before + 1

    corpus_b = await env.corpus_builder.materialize(manifest, state)
    # No re-streaming on repeated same-key fetches.
    assert client.sample_calls == ["good/encyclopedia"]
    # Identical docs across fetches (preview == score).
    assert corpus_a.documents == corpus_b.documents
    # Token/corpus cost counted exactly once (no re-billing from cached builds).
    assert RolloutStore.ledger(state).tokens == tokens_once
    assert RolloutStore.ledger(state).hub_calls == hub_before + 1


@pytest.mark.asyncio
async def test_preview_and_scoring_observe_same_docs(monkeypatch):
    client = FakeClient()
    env = _env(monkeypatch, client=client)
    state = await _finalized_state(env, sources=("good/encyclopedia", "good/science"))
    await env.compute_manifest_stats(state)
    calls_after_preview = list(client.sample_calls)
    await env.curator_rubric._prepared(state)
    # Scoring reused cached docs from the preview; no extra fetches.
    assert client.sample_calls == calls_after_preview


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
    state: dict = {}
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
async def test_scoring_runs_build_and_training_once_under_concurrency(monkeypatch):
    client = FakeClient()
    env = _env(monkeypatch, client=client)
    state = await _finalized_state(env)
    state["trajectory_id"] = "rollout-1"

    builder = _CountingBuilder(client=client, sample_docs_per_source=64)
    trainer = _CountingTrainer()
    rubric = CuratorRubric(
        env.config, builder, trainer, LeakageDetector(DEFAULT_EVAL_CORPUS)
    )
    funcs = [
        rubric.perf_reward,
        rubric.quality_reward,
        rubric.diversity_reward,
        rubric.cost_penalty,
        rubric.leakage_penalty,
        rubric.perf_loss,
        rubric.perf_accuracy,
        rubric.train_flops,
        rubric.corpus_tokens,
        rubric.num_sources,
        rubric.leakage_exact,
        rubric.leakage_fuzzy,
        rubric.leakage_semantic,
        rubric.cost_total,
        rubric.finalized,
        rubric.viable,
        rubric.tool_error_count,
        rubric.external_failure,
    ]
    await asyncio.gather(*[f(state) for f in funcs])
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
    monkeypatch, exc_factory, expected_kind
):
    client = FailingClient(exc_factory)
    env = _env(monkeypatch, client=client, fetch_max_attempts=1, fetch_timeout_seconds=2.0)
    state: dict = {}
    await env.setup_state(state)
    await env.search_datasets("encyclopedia", state)

    # (a) tool returns a structured error object.
    payload = json.loads(await env.inspect_dataset("good/encyclopedia", state))
    assert payload["error_kind"] == expected_kind
    assert "error" in payload

    await env.set_source("good/encyclopedia", state, weight=1.0)
    await env.finalize_manifest(state)

    # (b) the rollout completes scoring without raising.
    scoring = await env.curator_rubric._prepared(state)
    # (c) scoring returns the defined sentinel.
    assert scoring["perf"] == 0.0
    assert scoring["quality"] == 0.0
    assert scoring["diversity"] == 0.0
    assert scoring["viable"] is False
    assert RolloutStore.has_external_failure(state)
    assert RolloutStore.tool_error_count(state) >= 1


@pytest.mark.asyncio
async def test_real_timeout_classified_via_wait_for(monkeypatch):
    import time as _time

    class _SlowClient(FakeClient):
        def sample_documents(self, *a, **k):
            _time.sleep(0.3)
            return ["doc"]

    policy = RetryPolicy(attempts=1, timeout=0.05)
    builder = CorpusBuilder(client=_SlowClient(), retry_policy=policy)
    state: dict = {}
    RolloutStore.init(state, Manifest(), CostLedger())
    docs, error = await builder.fetch_source_docs(
        state, FetchKey("a/b", None, "train", "text", 4)
    )
    assert docs == []
    assert error["error_kind"] == "timeout"


@pytest.mark.asyncio
async def test_search_failure_returns_structured_error(monkeypatch):
    class _NoSearchClient(FakeClient):
        def search_datasets(self, query, scan_limit):
            raise ConnectionError("network down")

    env = _env(monkeypatch, client=_NoSearchClient(), fetch_max_attempts=1)
    state: dict = {}
    await env.setup_state(state)
    payload = json.loads(await env.search_datasets("anything", state))
    assert payload["candidates"] == []
    assert payload["error_kind"] == "network"
    assert RolloutStore.has_external_failure(state)


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

    env = _env(monkeypatch)
    state = await _finalized_state(env)
    await env.curator_rubric._prepared(state)
    # Leakage scoring and per-doc quality both went through to_thread.
    assert "score" in offloaded  # LeakageDetector.score
    assert any(name.startswith("_quality") for name in offloaded)


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


# --- Tier J/K: zero-weight telemetry metrics do not affect reward ----------


@pytest.mark.asyncio
async def test_telemetry_metrics_are_zero_weight(monkeypatch):
    env = _env(monkeypatch)
    rubric = env.curator_rubric
    names_to_weights = dict(zip(rubric._get_reward_func_names(), rubric._get_reward_weights()))
    assert names_to_weights["tool_error_count"] == 0.0
    assert names_to_weights["external_failure"] == 0.0


@pytest.mark.asyncio
async def test_reward_unaffected_by_recorded_errors(monkeypatch):
    env = _env(monkeypatch)
    rubric = env.curator_rubric
    state = await _finalized_state(env)

    await rubric.score_rollout(state)
    baseline = state["reward"]

    # Inject telemetry; recompute. Zero-weight metrics must not change reward.
    state2 = await _finalized_state(env)
    RolloutStore.record_tool_error(state2, "missing")
    RolloutStore.set_external_failure(state2, True)
    await rubric.score_rollout(state2)
    assert state2["reward"] == pytest.approx(baseline)
    assert state2["metrics"]["tool_error_count"] == 1.0
    assert state2["metrics"]["external_failure"] == 1.0


# --- Tier L: reward gating on minimum viability ----------------------------


@pytest.mark.asyncio
async def test_bonuses_applied_when_viable(monkeypatch):
    env = _env(monkeypatch)
    state = await _finalized_state(env)
    scoring = await env.curator_rubric._prepared(state)
    assert scoring["viable"] is True
    assert scoring["diversity"] > 0.0  # two distinct sources, low leakage


@pytest.mark.asyncio
async def test_bonuses_gated_when_severe_leakage(monkeypatch):
    client = FakeClient()
    env = _env(monkeypatch, client=client)
    state = await _finalized_state(env, sources=("good/encyclopedia",))
    # Eval corpus == the exact docs the source returns -> severe leakage.
    leaky_eval = client._docs["good/encyclopedia"]
    rubric = CuratorRubric(
        env.config,
        env.corpus_builder,
        HeuristicProxyTrainer(),
        LeakageDetector(leaky_eval, seed=0),
    )
    scoring = await rubric._prepared(state)
    assert scoring["leakage"]["overall"] >= env.config.leakage_severe_threshold
    assert scoring["viable"] is False
    assert scoring["quality"] == 0.0
    assert scoring["diversity"] == 0.0


# --- Tier M: config-driven tool availability -------------------------------


def test_run_code_hidden_when_disabled(monkeypatch):
    env = _env(monkeypatch, enable_run_code=False)
    assert "run_code" not in env.tool_map
    assert all(t.name != "run_code" for t in env.tool_defs)


def test_run_code_advertised_when_enabled(monkeypatch):
    env = _env(monkeypatch, enable_run_code=True)
    assert "run_code" in env.tool_map
    assert any(t.name == "run_code" for t in env.tool_defs)


# --- Tier N: state schema version + canonical hash -------------------------


@pytest.mark.asyncio
async def test_state_carries_schema_version(monkeypatch):
    env = _env(monkeypatch)
    state: dict = {}
    await env.setup_state(state)
    assert RolloutStore.schema_version(state) == STATE_SCHEMA_VERSION


@pytest.mark.asyncio
async def test_canonical_hash_stable_for_equal_state(monkeypatch):
    env = _env(monkeypatch)
    s1 = await _finalized_state(env, sources=("good/encyclopedia",))
    s2 = await _finalized_state(env, sources=("good/encyclopedia",))
    assert RolloutStore.canonical_hash(s1) == RolloutStore.canonical_hash(s2)
    s3 = await _finalized_state(env, sources=("good/encyclopedia", "good/science"))
    assert RolloutStore.canonical_hash(s3) != RolloutStore.canonical_hash(s1)


# --- Additional coverage: cutoff/query/filters/remove/stats ----------------


def test_parse_cutoff_forms():
    assert parse_cutoff("2024-12-31").date() == date(2024, 12, 31)
    dt = parse_cutoff("2024-06-01T12:00:00Z")
    assert dt.year == 2024 and dt.month == 6
    assert parse_cutoff(date(2023, 1, 1)).tzinfo is not None
    with pytest.raises(ValueError):
        parse_cutoff("   ")


def test_query_variants_expansion():
    variants = query_variants("math OR code datasets for training")
    assert "math" in variants
    assert "code" in variants
    # stopwords ("for", "training", "datasets") are not emitted as bare tokens
    assert "for" not in variants
    assert len(variants) == len(set(variants))  # deduplicated


@pytest.mark.asyncio
async def test_set_source_rejects_malformed_filters(monkeypatch):
    env = _env(monkeypatch)
    state: dict = {}
    await env.setup_state(state)
    await env.search_datasets("encyclopedia", state)
    # not JSON
    bad = json.loads(await env.set_source("good/encyclopedia", state, filters="not-json"))
    assert "invalid filters" in bad["error"]
    # JSON but not a list
    bad2 = json.loads(await env.set_source("good/encyclopedia", state, filters="{}"))
    assert "invalid filters" in bad2["error"]
    # list with missing "kind"
    bad3 = json.loads(
        await env.set_source("good/encyclopedia", state, filters='[{"params": {}}]')
    )
    assert "invalid filters" in bad3["error"]


@pytest.mark.asyncio
async def test_remove_source(monkeypatch):
    env = _env(monkeypatch)
    state: dict = {}
    await env.setup_state(state)
    await env.search_datasets("encyclopedia", state)
    await env.set_source("good/encyclopedia", state, weight=1.0)
    payload = json.loads(await env.remove_source("good/encyclopedia", state))
    assert payload["removed"] is True
    assert payload["num_sources"] == 0
    # removing again is a no-op
    payload2 = json.loads(await env.remove_source("good/encyclopedia", state))
    assert payload2["removed"] is False


@pytest.mark.asyncio
async def test_compute_manifest_stats(monkeypatch):
    env = _env(monkeypatch)
    state: dict = {}
    await env.setup_state(state)
    await env.search_datasets("encyclopedia science", state)
    empty = json.loads(await env.compute_manifest_stats(state))
    assert "error" in empty  # empty manifest
    await env.set_source("good/encyclopedia", state, weight=1.0)
    stats = json.loads(await env.compute_manifest_stats(state))
    assert stats["materialized_docs"] > 0
    assert "leakage" in stats and "estimated_cost" in stats
    assert stats["external_failure"] is False


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
async def test_rubric_degrades_val_fetch_failure_to_sentinel(monkeypatch):
    # End-to-end degrade: a held-out val-set fetch failure on the real-trainer
    # path collapses to the infinite-loss sentinel (Perf -> 0), not a crash.
    env = _env(monkeypatch)

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
    rubric = CuratorRubric(
        env.config, env.corpus_builder, trainer, LeakageDetector(DEFAULT_EVAL_CORPUS)
    )
    state: dict = {}
    RolloutStore.init(state, Manifest(), CostLedger())
    result = await rubric._train(_corpus_with_text(), state)
    assert result.loss == float("inf")
    assert result.backend == "error"
    assert RolloutStore.has_external_failure(state)


@pytest.mark.asyncio
async def test_rubric_degrades_sandbox_trainer_error_to_sentinel(monkeypatch):
    # Sibling to the val-FETCH degrade above, for the OTHER degrade path the CE
    # fix relies on: a sandbox TrainerError (the nonzero exit produced when the
    # in-sandbox plan_val_windows raises on a degenerate <=1-token val set) must
    # ALSO collapse to the infinite-loss sentinel in the rubric, not crash or
    # yield a bogus good loss.
    env = _env(monkeypatch)
    trainer = SandboxProxyTrainer(
        # "fail" mode returns a nonzero exit -> SandboxProxyTrainer raises a
        # TrainerError carrying the sandbox stderr tail.
        client_factory=lambda: _FakeSandboxClient("fail"),
        request_factory=lambda config, name: object(),
        val_loader=None,
    )
    rubric = CuratorRubric(
        env.config, env.corpus_builder, trainer, LeakageDetector(DEFAULT_EVAL_CORPUS)
    )
    state: dict = {}
    RolloutStore.init(state, Manifest(), CostLedger())
    result = await rubric._train(_corpus_with_text(), state)
    assert result.loss == float("inf")
    assert result.backend == "error"
    assert RolloutStore.has_external_failure(state)
    assert RolloutStore.tool_error_count(state) >= 1
    # The degrade preserved the sandbox failure detail (the stderr tail is only
    # attached on the TrainerError path), proving it was the trainer-error degrade
    # rather than a silently-swallowed good loss.
    assert "boom" in (RolloutStore.trainer_error(state) or "")


@pytest.mark.asyncio
async def test_heuristic_trainer_ignores_val_set(monkeypatch):
    # The default heuristic backend does NOT compute per-token CE on a held-out
    # set, so retargeting the val set must not change its (synthetic) loss.
    env = _env(monkeypatch, validation_set={"val_tokens": 4096})
    assert env.config.validation_set.val_tokens == 4096
    state = await _finalized_state(env, sources=("good/encyclopedia",))
    scoring = await env.curator_rubric._prepared(state)
    env2 = _env(monkeypatch, validation_set={"val_tokens": 9_999_999})
    state2 = await _finalized_state(env2, sources=("good/encyclopedia",))
    scoring2 = await env2.curator_rubric._prepared(state2)
    assert scoring["loss"] == scoring2["loss"]


def test_load_environment_accepts_validation_set_override(monkeypatch):
    env = _env(
        monkeypatch,
        validation_set={"dataset_id": "custom/val", "val_tokens": 1024},
    )
    assert env.config.validation_set.dataset_id == "custom/val"
    assert env.config.validation_set.val_tokens == 1024
    assert env.env_args["val_dataset_id"] == "custom/val"
    assert env.env_args["val_tokens"] == 1024


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
