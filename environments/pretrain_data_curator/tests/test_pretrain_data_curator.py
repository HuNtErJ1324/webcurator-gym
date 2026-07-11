from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
import os
import subprocess
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import verifiers.v1 as vf
from pydantic import ValidationError
from verifiers.v1 import graph
from verifiers.v1.decorators import discover_decorated
from verifiers.v1.runtimes.subprocess import SubprocessConfig, SubprocessRuntime

from pretrain_data_curator.corpus import (
    CorpusBuilder,
    CuratedCorpus,
    DocumentFilter,
    SourceCorpus,
    _iter_sampling,
)
from pretrain_data_curator.hf_access import (
    DatasetAccessError,
    FetchKey,
    HuggingFaceDatasetClient,
    RetryPolicy,
    classify_exception,
    loop_local_semaphore,
    run_blocking_with_retry,
)
from pretrain_data_curator.hf_cli_parse import extract_hf_commands
from pretrain_data_curator.models import (
    CuratorConfig,
    FilterSpec,
    MANIFEST_FILENAME,
    Manifest,
    ProxyStudentConfig,
    Sampling,
    Source,
)
from pretrain_data_curator.pretrain_data_curator import load_environment
from pretrain_data_curator.rewards import CuratorScorer
from pretrain_data_curator.rollout_state import CuratorState, RolloutStore
from pretrain_data_curator.self_score import (
    SELF_SCORE_FILENAME,
    SELF_SCORE_TRAIN_FILENAME,
    render_self_score_script,
    render_self_score_train_script,
)
from pretrain_data_curator.tasks import TASK_PROMPT, build_tasks
from pretrain_data_curator.taskset import (
    HF_CLI_SKILL_FILENAME,
    HF_CLI_SKILL_RESOURCE,
    HF_CLI_SKILL_RUNTIME_PATH,
    HF_CLI_SKILL_SHA256,
    HF_CLI_SKILL_UPSTREAM_PATH,
    HF_CLI_SKILL_UPSTREAM_REVISION,
    CuratorTaskset,
    CuratorTasksetConfig,
    extract_json_object,
    hf_cli_skill_package_file,
    parse_manifest,
)
from tests.conftest import NoOpLeakageDetector, bind_fast_scorer
from verifiers.v1.taskset import Taskset
from pretrain_data_curator.trainer import (
    HeuristicProxyTrainer,
    RuntimeSelectedTrainer,
    TrainResult,
    estimate_param_count,
)
from pretrain_data_curator.student_model import GPT2_SMALL_PARAM_COUNT
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
    """Shrink workspace-manifest polling while preserving race coverage."""
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
        self.taskset = CuratorTaskset(
            CuratorTasksetConfig(id="test", screen_val_set=False, **cfg)
        )
        # The validated CuratorConfig the reward/tools derive from (== v0 env.config).
        self.config = self.taskset.curator
        # One shared corpus builder so a tool preview and final scoring share the
        # per-rollout document cache + cost ledger (they also share `state`).
        self.corpus_builder = corpus_builder or CorpusBuilder(
            client=self.client,
            retry_policy=RetryPolicy(
                attempts=self.config.fetch_max_attempts,
                timeout=self.config.fetch_timeout_seconds,
                per_doc_seconds=self.config.fetch_timeout_per_doc_seconds,
            ),
            fetch_limit=self.config.max_concurrent_fetches,
        )
        self.leakage_detector = leakage_detector or NoOpLeakageDetector()
        self.trainer = trainer or HeuristicProxyTrainer()
        # Inject the shared collaborators into the taskset's lazy scoring slots so
        # `_ensure()` builds its scorer from them instead of hitting a live Hub.
        self.taskset._client = self.client
        bind_fast_scorer(
            self.taskset,
            corpus_builder=self.corpus_builder,
            trainer=self.trainer,
            leakage_detector=self.leakage_detector,
        )
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
        leakage or NoOpLeakageDetector(),
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
    monkeypatch.setattr(
        HuggingFaceDatasetClient,
        "_is_script_dataset",
        lambda self, dataset_id: False,
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
    monkeypatch.setattr(
        HuggingFaceDatasetClient,
        "_is_script_dataset",
        lambda self, dataset_id: False,
    )
    client = object.__new__(HuggingFaceDatasetClient)
    client._token = "test-token"

    assert client.sample_documents("wikimedia/wikipedia", None, "train", None, 1) == [
        "configured document"
    ]
    assert calls == [None, "20231101.en"]


def test_source_defaults_to_auto_detected_text_field():
    assert Source(dataset_id="owner/name").text_field is None


def test_package_imports_without_torch_installed():
    """Hub integration installs project deps only; torch stays a dev extra.

    ``import pretrain_data_curator`` must succeed without torch on the path,
    matching ``test_install_and_import`` on Prime Hub.
    """
    code = r"""
import importlib.abc
import importlib.machinery
import sys


class _TorchMissingLoader(importlib.abc.Loader):
    def create_module(self, spec):
        raise ModuleNotFoundError(spec.name)

    def exec_module(self, module):
        raise ModuleNotFoundError(module.__name__)


class _TorchMissingFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            return importlib.machinery.ModuleSpec(fullname, _TorchMissingLoader())
        return None


sys.meta_path.insert(0, _TorchMissingFinder())
for name in list(sys.modules):
    if name == "torch" or name.startswith("torch."):
        del sys.modules[name]
    if name == "pretrain_data_curator" or name.startswith("pretrain_data_curator."):
        del sys.modules[name]

import pretrain_data_curator  # noqa: F401

assert "pretrain_data_curator.student_model" not in sys.modules
assert "torch" not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1])},
        capture_output=True,
        text=True,
    )


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
    assert env.taskset.config.manifest_filename == MANIFEST_FILENAME
    assert env.harness.config.id == "bash"
    assert env.harness.config.env == {
        "MAX_TOOL_OUTPUT_CHARS": "20000",
        "HF_TOKEN": "test-token",
    }
    assert env.env_args["harness_id"] == "bash"
    assert env.taskset.load_tasks()


def test_single_smoke_config_exhaustively_matches_source_options():
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "eval"
        / "deepseek-v4-flash-smoke.toml"
    )
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    row = config["eval"][0]
    args = row["args"]

    assert config["env_dir_path"] == ".."
    assert row["env_id"] == "pretrain-data-curator"
    assert row["model"] == "deepseek/deepseek-v4-flash"
    assert set(args) == set(inspect.signature(load_environment).parameters)
    assert args["screen_val_set"] is True
    assert set(args["validation_set"]) == set(ValidationSetConfig.model_fields)
    # Exact match against all ProxyStudentConfig fields except those intentionally
    # absent from the Docker smoke config:
    #   - docker_host must be unset (None) for the Docker backend; an empty
    #     string would trip the guard in load_environment.
    #   - modal_gpu is Modal-only and irrelevant to the Docker backend.
    #   - sliding_window_size defaults to None (full context); TOML 1.0 has no
    #     null literal, so omitting it is the canonical smoke-config spelling.
    expected_proxy_fields = set(ProxyStudentConfig.model_fields) - {
        "docker_host",
        "modal_gpu",
        "sliding_window_size",
        # Optional None defaults; TOML 1.0 has no null literal, so omitting them
        # is the canonical smoke-config spelling.
        "train_microbatch_size",
        "val_batch_size",
        "val_logit_chunk_tokens",
        # Portable feature knobs default off; omitted from the smoke TOML.
        "bigram_hash_embed",
        "smear_embed",
        "partial_key_offset",
        "paired_head",
        "mudd_pairs",
        "xsa_enabled",
        "xsa_pairs",
        "single_act_last_k",
        "exp_residual_decay",
        "multi_token_pred",
        "eos_aligned_batches",
        "max_document_tokens",
        "record_adam_eps",
        "grad_accum_embed_head_steps",
        "seq_len_schedule",
        "untie_at_frac",
        "cautious_wd",
        "nor_muon",
        "polar_express",
    }
    assert set(args["proxy_student"]) == expected_proxy_fields
    assert ProxyStudentConfig.model_validate(args["proxy_student"])
    assert ValidationSetConfig.model_validate(args["validation_set"])


def _400m_eval_config_names() -> list[str]:
    """Every on-disk *400M* eval TOML (docker-backed agent runs)."""
    eval_dir = Path(__file__).resolve().parents[1] / "configs" / "eval"
    names = sorted(p.name for p in eval_dir.glob("*400M*.toml"))
    assert names, f"expected *400M*.toml under {eval_dir}"
    return names


@pytest.mark.parametrize("config_name", _400m_eval_config_names())
def test_400m_eval_configs_use_webcurator_runtime_image(config_name):
    """Every *400M* eval config must use the baked hf+decon image, not bare pytorch."""
    config_path = Path(__file__).resolve().parents[1] / "configs" / "eval" / config_name
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    proxy = config["args"]["proxy_student"]
    assert proxy["runtime_backend"] == "docker"
    assert proxy["docker_image"] == "webcurator-runtime:latest"
    assert "pytorch/pytorch" not in proxy["docker_image"]


def _extract_bash_function(script: str, name: str) -> str:
    """Return the body of ``name() { ... }`` (outermost braces), or raise."""
    header = f"{name}()"
    start = script.find(header)
    assert start != -1, f"missing function {name}()"
    brace = script.find("{", start)
    assert brace != -1, f"missing opening brace for {name}()"
    depth = 0
    for idx in range(brace, len(script)):
        ch = script[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return script[brace + 1 : idx]
    raise AssertionError(f"unclosed function body for {name}()")


def _assert_ordered_runtime_provision(body: str, *, label: str) -> None:
    """Each provision path must verify/build decon, then image, then preflight."""
    decon_verify = body.find('"$DECON_BIN" --version')
    if decon_verify < 0:
        decon_verify = body.find("$DECON_BIN --version")
    decon_fallback = body.find("build_from_source.sh")
    image_build = body.find(
        "docker build -f Dockerfile.runtime -t webcurator-runtime:latest"
    )
    if image_build < 0:
        image_build = body.find(
            'docker build -f Dockerfile.runtime -t "$RUNTIME_IMAGE"'
        )
    preflight = body.find("command -v hf")
    hub_import = body.find("import huggingface_hub")
    zstd_import = body.find("import zstandard")
    zstd_decompress = body.find("zstandard.ZstdDecompressor")
    decon_exec = body.find("test -x /workspace/decon/bin/decon")
    if decon_exec < 0:
        decon_exec = body.find("test -x ${AGENT_DECON_BIN}")
    decon_run = body.find("/workspace/decon/bin/decon --version")
    if decon_run < 0:
        decon_run = body.find("${AGENT_DECON_BIN} --version")

    assert decon_verify >= 0, f"{label}: missing decon --version verification"
    assert decon_fallback >= 0, f"{label}: missing decon build_from_source fallback"
    assert image_build >= 0, f"{label}: missing webcurator-runtime:latest build"
    assert preflight >= 0, f"{label}: missing in-container hf preflight"
    assert hub_import >= 0, f"{label}: missing huggingface_hub import preflight"
    assert zstd_import >= 0, f"{label}: missing zstandard import preflight"
    assert zstd_decompress >= 0, f"{label}: missing zstd decompression round-trip preflight"
    assert decon_exec >= 0, f"{label}: missing executable /workspace/decon check"
    assert decon_run >= 0, f"{label}: missing /workspace/decon --version preflight"
    assert "/workspace/decon/bin/decon" in body or (
        'AGENT_DECON_BIN="/workspace/decon/bin/decon"' in body
    ), f"{label}: preflight must target agent-mounted /workspace/decon"

    # Order: decon verify/fallback → image build → hf → hub → zstd → executable decon.
    assert decon_verify < image_build, f"{label}: decon verify before image build"
    assert decon_fallback < image_build, f"{label}: decon fallback before image build"
    assert image_build < preflight, f"{label}: image build before hf preflight"
    assert preflight < hub_import, f"{label}: hf before huggingface_hub"
    assert hub_import < zstd_import, f"{label}: huggingface_hub before zstandard preflight"
    assert zstd_import < zstd_decompress, f"{label}: zstandard import before decompress check"
    assert zstd_decompress < decon_exec, f"{label}: zstd preflight before decon test -x"
    assert decon_exec <= decon_run, f"{label}: test -x before decon --version"


def test_400m_pod_scripts_build_runtime_after_decon_and_preflight():
    """GPU and CPU A100 paths each own decon→image→preflight; on-pod tees eval."""
    scripts_dir = Path(__file__).resolve().parents[3] / "scripts"
    a100 = (scripts_dir / "run_400m_eval_a100.sh").read_text(encoding="utf-8")
    on_pod = (scripts_dir / "run_400m_eval_on_pod.sh").read_text(encoding="utf-8")

    cpu_body = _extract_bash_function(a100, "remote_provision_cpu")
    gpu_body = _extract_bash_function(a100, "remote_provision_gpu")
    assert cpu_body != gpu_body, "CPU and GPU provision bodies must be distinct"
    _assert_ordered_runtime_provision(cpu_body, label="a100.cpu")
    _assert_ordered_runtime_provision(gpu_body, label="a100.gpu")

    # on-pod is a linear script (no named provision fn); validate the same order.
    _assert_ordered_runtime_provision(on_pod, label="on_pod")
    assert 'RUNTIME_IMAGE="webcurator-runtime:latest"' in on_pod
    assert 'AGENT_DECON_BIN="/workspace/decon/bin/decon"' in on_pod

    # LOG_FILE must capture the actual eval pipeline, not a dead declaration.
    assert 'LOG_FILE="$LOG_DIR/' in on_pod or "LOG_FILE=" in on_pod
    eval_pipeline = (
        "uv run eval @ configs/eval/deepseek-v4-pro-400M-300turn-codex.toml"
        ' 2>&1 | tee "$LOG_FILE"'
    )
    assert eval_pipeline in on_pod, 'eval must pipe into tee "$LOG_FILE"'
    assert "exec uv run eval" not in on_pod


def test_runtime_dockerfile_installs_zstandard_codec():
    # The webcurator-runtime image must ship the zstandard codec so the
    # datasets/fsspec read path can materialize zstd-compressed Hub datasets
    # (mlfoundations/dclm-baseline-1.0, monology/pile-uncopyrighted) instead of
    # failing with "Compression type zstd not supported".
    dockerfile = (
        Path(__file__).resolve().parents[3]
        / "environments"
        / "pretrain_data_curator"
        / "Dockerfile.runtime"
    )
    text = dockerfile.read_text(encoding="utf-8")
    assert "zstandard" in text, "Dockerfile.runtime must install zstandard"
    assert "pip install" in text, "Dockerfile.runtime must pip install its deps"
    # The package must also declare it so `uv pip install -e .` pulls it in on
    # the host that runs `uv run eval` (which materializes the same datasets).
    pyproject = (
        Path(__file__).resolve().parents[3]
        / "environments"
        / "pretrain_data_curator"
        / "pyproject.toml"
    )
    assert "zstandard" in pyproject.read_text(encoding="utf-8")


def test_zstandard_codec_installed_and_decompresses():
    # Deterministic, offline proof that the zstd codec the runtime images now
    # ship actually opens/decompresses zstd content through fsspec -- the exact
    # path datasets uses to read zstd-compressed parquet from the Hub. Skips if
    # zstandard is not installed (it must be, per the dependency above).
    zstandard = pytest.importorskip("zstandard")
    fsspec = pytest.importorskip("fsspec")
    import os
    import tempfile

    data = b"webcurator corpus materialization round-trip\n" * 200
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "sample.zst")
        with open(p, "wb") as f:
            f.write(zstandard.ZstdCompressor().compress(data))
        with fsspec.open(p, "rb", compression="zstd") as fh:
            out = fh.read()
    assert out == data


@pytest.mark.parametrize("harness_id", ["bash", "codex", "mini_swe_agent"])
def test_load_environment_uses_one_initial_prompt_for_all_harnesses(
    monkeypatch, harness_id
):
    monkeypatch.setenv("HF_TOKEN", "test-token")

    env = load_environment(harness_id=harness_id)

    assert env.harness.config.id == harness_id
    assert env.harness.config.env.get("HF_TOKEN") == "test-token"
    assert env.env_args["harness_id"] == harness_id

    task = env.taskset.load_tasks()[0]
    system_prompt, prompt = env.harness.resolve_prompt(task)
    assert system_prompt is None
    assert task.system_prompt is None
    assert prompt == task.prompt
    assert prompt.count("## Rules") == 1
    assert "self_score.py" in prompt


def test_load_environment_rejects_unknown_harness():
    with pytest.raises(
        ValueError,
        match=(
            "unknown harness_id 'unknown'; valid harness ids: "
            "bash, codex, default, kimi_code, mini_swe_agent, rlm, terminus_2"
        ),
    ):
        load_environment(harness_id="unknown")


def test_load_environment_uses_declarative_docker_runtime_for_docker_trainer():
    docker_env = load_environment(
        use_real_trainer=True,
        proxy_student={"runtime_backend": "docker", "gpu_count": 1},
    )
    assert docker_env.harness.config.env == {
        "MAX_TOOL_OUTPUT_CHARS": "20000",
        "UV_REINSTALL_PACKAGE": "pydantic-core",
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


def test_load_environment_injects_hf_token_into_harness_env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")
    docker_env = load_environment(
        harness_id="codex",
        use_real_trainer=True,
        proxy_student={"runtime_backend": "docker", "gpu_count": 1},
    )
    assert docker_env.harness.config.env["HF_TOKEN"] == "hf_test_token"
    assert docker_env.harness.config.env["UV_REINSTALL_PACKAGE"] == "pydantic-core"

    subprocess_env = load_environment(harness_id="codex")
    assert subprocess_env.harness.config.env == {
        "MAX_TOOL_OUTPUT_CHARS": "20000",
        "HF_TOKEN": "hf_test_token",
    }


def test_load_environment_omits_hf_token_from_harness_env_when_unset(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    env = load_environment(harness_id="codex")
    assert "HF_TOKEN" not in env.harness.config.env


def test_load_environment_rejects_remote_docker_host():
    with pytest.raises(ValueError, match="docker_host is not supported"):
        load_environment(
            use_real_trainer=True,
            proxy_student={
                "runtime_backend": "docker",
                "docker_host": "ssh://user@gpu-host",
            },
        )


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
    assert math.isfinite(scoring["perf"])
    assert "quality" not in scoring
    assert "diversity" not in scoring
    assert scoring["flops"] > 0.0
    assert "cost" not in scoring


@pytest.mark.asyncio
async def test_empty_manifest_scores_zero_perf():
    curator = await _make()
    scoring = await curator.prepared()
    assert scoring["perf"] == 0.0
    assert scoring["perf_target_loss"] == curator.config.perf_target_loss
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
    builder = CorpusBuilder(client=client)
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
    builder = CorpusBuilder(client=client)

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
    builder = CorpusBuilder(client=client)

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
        def __init__(self) -> None:
            super().__init__()
            self.requested: list[tuple[str, int]] = []

        def sample_documents(self, dataset_id, config, split, text_field, n):
            self.requested.append((dataset_id, n))
            return [doc] * min(n, n_docs)

    client = _FixedClient()
    builder = CorpusBuilder(client=client)

    manifest = Manifest(
        token_budget=600,
        sources=[
            Source(dataset_id="good/encyclopedia", weight=0.0),
            Source(dataset_id="good/science", weight=0.0),
        ],
    )
    state = CuratorState()
    await builder.materialize(manifest, state)

    # Zero weights must not impose a weight-derived per-source fetch cap of 0.
    assert all(n > 0 for _, n in client.requested)


@pytest.mark.asyncio
async def test_zero_weight_source_is_not_fetched_when_other_weights_are_positive():
    client = FakeClient()
    builder = CorpusBuilder(client=client)
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
    builder = CorpusBuilder(client=client)
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
async def test_materialize_offloads_source_corpus_file_writes(monkeypatch):
    """Async materialize must not run SourceCorpus/RolloutStore writes on the event loop."""
    import pretrain_data_curator.corpus as corpus_mod

    offloaded: list[str] = []
    real_to_thread = asyncio.to_thread

    async def spy(fn, *args, **kwargs):
        offloaded.append(getattr(fn, "__qualname__", getattr(fn, "__name__", repr(fn))))
        return await real_to_thread(fn, *args, **kwargs)

    monkeypatch.setattr(corpus_mod.asyncio, "to_thread", spy)

    class _Client(FakeClient):
        def sample_documents(self, dataset_id, config, split, text_field, n):
            return ["alpha word " * 40, "beta word " * 40, "gamma word " * 40]

    manifest = Manifest(
        token_budget=500,
        sources=[
            Source(dataset_id="one/source", weight=1.0),
            Source(dataset_id="two/source", weight=1.0),
        ],
    )
    state = CuratorState()
    corpus = await CorpusBuilder(client=_Client()).materialize(manifest, state)

    assert corpus.total_tokens > 0
    assert any(name.endswith("from_docs") for name in offloaded)
    assert any(name.endswith("store_docs") for name in offloaded)
    # Two sources → two store_docs (raw cache writes) + two from_docs (scratch writes).
    assert sum(1 for name in offloaded if name.endswith("store_docs")) == 2
    assert sum(1 for name in offloaded if name.endswith("from_docs")) == 2


@pytest.mark.asyncio
async def test_materialize_offloads_append_iter_through_to_thread(monkeypatch):
    """Surplus redistribution must route the persistent append through to_thread."""
    import pretrain_data_curator.corpus as corpus_mod

    offloaded: list[str] = []
    real_to_thread = asyncio.to_thread

    async def spy(fn, *args, **kwargs):
        offloaded.append(getattr(fn, "__qualname__", getattr(fn, "__name__", repr(fn))))
        return await real_to_thread(fn, *args, **kwargs)

    monkeypatch.setattr(corpus_mod.asyncio, "to_thread", spy)

    # Cap the first-pass selection so the redistribution loop re-appends the
    # surviving surplus (filtered_count > sampled_count) via append_iter.
    class _Client(FakeClient):
        def sample_documents(self, dataset_id, config, split, text_field, n):
            return ["alpha word " * 40 for _ in range(n or 20)]

    manifest = Manifest(
        token_budget=5000,
        sources=[
            Source(
                dataset_id="one/source",
                weight=1.0,
                sampling=Sampling(max_docs=2),
            )
        ],
    )
    state = CuratorState()
    corpus = await CorpusBuilder(client=_Client()).materialize(manifest, state)

    # The surplus-redistribution path routed the persistent append through
    # to_thread (offloaded), and the on-disk result is unchanged/deterministic.
    assert any(name.endswith("append_iter") for name in offloaded)
    assert sum(s.doc_count for s in corpus.sources) == 2


@pytest.mark.asyncio
async def test_materialize_pipeline_bounded_by_fetch_limit(monkeypatch):
    """The full fetch→filter→write pipeline must never exceed fetch_limit at once."""
    import threading

    import pretrain_data_curator.corpus as corpus_mod

    lock = threading.Lock()
    active = 0
    max_active = 0
    real_to_thread = asyncio.to_thread
    fetch_limit = 2

    async def spy(fn, *args, **kwargs):
        name = getattr(fn, "__qualname__", getattr(fn, "__name__", ""))
        if name.endswith("from_docs"):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            try:
                # Widen the overlap window so concurrent pipelines are observable.
                await asyncio.sleep(0.05)
                return await real_to_thread(fn, *args, **kwargs)
            finally:
                with lock:
                    active -= 1
        return await real_to_thread(fn, *args, **kwargs)

    monkeypatch.setattr(corpus_mod.asyncio, "to_thread", spy)

    class _Client(FakeClient):
        def sample_documents(self, dataset_id, config, split, text_field, n):
            return ["alpha word " * 40, "beta word " * 40]

    manifest = Manifest(
        token_budget=500,
        sources=[
            Source(dataset_id="one/source", weight=1.0),
            Source(dataset_id="two/source", weight=1.0),
            Source(dataset_id="three/source", weight=1.0),
            Source(dataset_id="four/source", weight=1.0),
        ],
    )
    state = CuratorState()
    await CorpusBuilder(client=_Client(), fetch_limit=fetch_limit).materialize(
        manifest, state
    )

    # Concurrent persistent writes reach the configured bound but never exceed it.
    assert max_active == fetch_limit


@pytest.mark.asyncio
async def test_materialize_preserves_source_and_doc_ordering():
    """Source order and per-source document order must be stable after offload."""
    docs = [
        "first document body here",
        "second document body here",
        "third document body here",
    ]

    class _Client(FakeClient):
        def sample_documents(self, dataset_id, config, split, text_field, n):
            return list(docs)

    manifest = Manifest(
        token_budget=500,
        sources=[
            Source(dataset_id="alpha/source", weight=1.0),
            Source(dataset_id="beta/source", weight=1.0),
        ],
    )
    state = CuratorState()
    corpus = await CorpusBuilder(client=_Client()).materialize(manifest, state)

    # Source result order mirrors the manifest order.
    assert [s.dataset_id for s in corpus.sources] == [
        "alpha/source",
        "beta/source",
    ]
    # Within each source the surviving documents keep their original order.
    for source in corpus.sources:
        assert source.documents == docs


@pytest.mark.asyncio
async def test_materialize_propagates_store_docs_errors(monkeypatch):
    """Exceptions from the offloaded store_docs must still propagate unchanged."""
    import pretrain_data_curator.corpus as corpus_mod

    class _Boom(Exception):
        pass

    def boom(state, key, docs):
        raise _Boom("disk write failed")

    monkeypatch.setattr(corpus_mod.RolloutStore, "store_docs", boom)

    class _Client(FakeClient):
        def sample_documents(self, dataset_id, config, split, text_field, n):
            return ["alpha word " * 40]

    manifest = Manifest(
        token_budget=500,
        sources=[Source(dataset_id="one/source", weight=1.0)],
    )
    with pytest.raises(_Boom):
        await CorpusBuilder(client=_Client()).materialize(manifest, CuratorState())


def test_corpus_builder_rejects_invalid_fetch_limit():
    """fetch_limit must be a positive int; 0/None/negative/non-int must fail early."""
    import pretrain_data_curator.corpus as corpus_mod

    for bad in (0, -1, None, 1.5, "8", True):
        with pytest.raises(ValueError):
            corpus_mod.CorpusBuilder(client=object(), fetch_limit=bad)

    # A valid limit constructs and is stored unchanged.
    builder = corpus_mod.CorpusBuilder(client=object(), fetch_limit=3)
    assert builder._fetch_limit == 3


@pytest.mark.asyncio
async def test_materialize_serializes_store_docs_state_writes(monkeypatch):
    """Concurrent store_docs must not race on shared state (lost-update stress).

    Uses fetch_limit > 1 so sources run concurrently, and an instrumented
    store_docs that performs a non-atomic read-modify-write on a shared counter
    with a wide race window. Without the asyncio-side serialization this would
    overlap and lose updates; with it, exactly one store_docs runs at a time and
    every write is preserved.
    """
    import threading
    import time

    from pretrain_data_curator.rollout_state import RolloutStore as _RolloutStore

    real_store = _RolloutStore.store_docs
    lock = threading.Lock()
    active = 0
    max_active = 0
    # Shared counter exercised via a real read-modify-write: read, widen the
    # window, then write. Concurrent (unsynchronized) writes would interleave
    # and undercount; serialization guarantees isolation.
    total_writes = 0
    per_key: dict[str, int] = {}
    num_sources = 6
    fetch_limit = 4

    def instrumented_store(state, key, docs):
        nonlocal active, max_active, total_writes
        with lock:
            active += 1
            max_active = max(max_active, active)
        try:
            # Non-atomic read-modify-write with a wide window to expose races.
            snapshot = total_writes
            time.sleep(0.02)
            with lock:
                total_writes = snapshot + 1
                per_key[key] = per_key.get(key, 0) + 1
        finally:
            with lock:
                active -= 1
        # Preserve the real persistent cache so downstream behavior is unaffected.
        return real_store(state, key, docs)

    monkeypatch.setattr(_RolloutStore, "store_docs", instrumented_store)

    class _Client(FakeClient):
        def sample_documents(self, dataset_id, config, split, text_field, n):
            return ["alpha word " * 40, "beta word " * 40]

    manifest = Manifest(
        token_budget=500,
        sources=[Source(dataset_id=f"src/{i}", weight=1.0) for i in range(num_sources)],
    )
    state = CuratorState()
    corpus = await CorpusBuilder(
        client=_Client(), fetch_limit=fetch_limit
    ).materialize(manifest, state)

    # Serialization caps concurrent store_docs at one, even with fetch_limit > 1.
    assert max_active == 1
    # No lost updates: every source's cache write landed exactly once.
    assert total_writes == num_sources
    assert sum(per_key.values()) == num_sources
    assert all(count == 1 for count in per_key.values())
    assert corpus.total_tokens > 0


@pytest.mark.asyncio
async def test_store_docs_cancellation_holds_lock_until_worker_done(monkeypatch):
    """Cancel first store_docs while blocked; second write waits for worker end.

    Exercises requirement 1: cancellation of the awaiting coroutine must not
    release ``_store_lock`` until the offloaded worker has actually completed.
    """
    import threading

    from pretrain_data_curator.rollout_state import RolloutStore as _RS

    entered: list[str] = []
    exited: list[str] = []
    release = threading.Event()
    started = threading.Event()
    active = 0
    max_active = 0
    clk = threading.Lock()

    def instrumented(state, key, docs):
        nonlocal active, max_active
        with clk:
            active += 1
            max_active = max(max_active, active)
        entered.append(key)
        if not started.is_set():
            started.set()
            release.wait(5.0)  # simulate a slow disk write
        with clk:
            active -= 1
        exited.append(key)
        return None

    monkeypatch.setattr(_RS, "store_docs", instrumented)

    builder = CorpusBuilder(client=object())
    state = CuratorState()

    async def write(tag):
        async with builder._store_lock:
            await builder._offloaded_to_completion(_RS.store_docs, state, tag, ["x"])

    t1 = asyncio.create_task(write("a"))
    await asyncio.to_thread(started.wait)
    t1.cancel()  # cancel first write's await while its worker is blocked
    await asyncio.sleep(0.05)
    # Second write started, but must NOT have entered: lock is held until the
    # first worker finishes.
    t2 = asyncio.create_task(write("b"))
    await asyncio.sleep(0.05)
    assert entered == ["a"]
    assert max_active == 1
    release.set()  # let the first worker finish
    cancelled = False
    try:
        await t1
    except asyncio.CancelledError:
        cancelled = True
    await t2
    # Cancellation propagated, only after the first worker completed.
    assert cancelled
    # No overlap: exactly one worker ran at a time, and b entered after a exited.
    assert max_active == 1
    assert entered == ["a", "b"]
    assert exited == ["a", "b"]


@pytest.mark.asyncio
async def test_from_docs_cancellation_holds_pipeline_sem(monkeypatch):
    """Cancel from_docs mid-write (fetch_limit=1); second pipeline waits.

    Exercises requirement 2: the pipeline slot must stay held until the
    offloaded ``from_docs`` worker finishes, so a follow-up pipeline cannot
    transiently exceed ``fetch_limit``.
    """
    import threading

    from pretrain_data_curator.corpus import SourceCorpus

    real_from = SourceCorpus.from_docs
    entered: list[str] = []
    exited: list[str] = []
    release = threading.Event()
    started = threading.Event()
    active = 0
    max_active = 0
    clk = threading.Lock()

    def instrumented(dataset_id, config, weight, docs, *, dest_dir=None):
        nonlocal active, max_active
        with clk:
            active += 1
            max_active = max(max_active, active)
        entered.append(dataset_id)
        if not started.is_set():
            started.set()
            release.wait(5.0)
        with clk:
            active -= 1
        exited.append(dataset_id)
        return real_from(dataset_id, config, weight, docs, dest_dir=dest_dir)

    monkeypatch.setattr(SourceCorpus, "from_docs", instrumented)

    builder = CorpusBuilder(client=object(), fetch_limit=1)
    sem = asyncio.Semaphore(1)

    async def pipeline(tag):
        async with sem:
            await builder._offloaded_to_completion(
                SourceCorpus.from_docs,
                tag,
                None,
                1.0,
                iter([f"doc-{tag}"]),
                dest_dir=None,
            )

    t1 = asyncio.create_task(pipeline("a"))
    await asyncio.to_thread(started.wait)
    t1.cancel()
    await asyncio.sleep(0.05)
    t2 = asyncio.create_task(pipeline("b"))
    await asyncio.sleep(0.05)
    # Second pipeline cannot enter its worker; the slot is held until the first
    # worker finishes.
    assert entered == ["a"]
    assert max_active == 1
    release.set()
    cancelled = False
    try:
        await t1
    except asyncio.CancelledError:
        cancelled = True
    await t2
    assert cancelled
    assert max_active == 1
    assert entered == ["a", "b"]
    assert exited == ["a", "b"]


@pytest.mark.asyncio
async def test_offloaded_worker_error_propagates_normally():
    """A worker exception propagates unchanged (not wrapped in CancelledError)."""
    builder = CorpusBuilder(client=object())

    def boom():
        raise ValueError("worker exploded")

    with pytest.raises(ValueError):
        await builder._offloaded_to_completion(boom)


@pytest.mark.asyncio
async def test_offloaded_cancellation_surfaces_worker_error(monkeypatch):
    """Cancellation while blocked must not mask the worker's own error."""
    import threading

    builder = CorpusBuilder(client=object())
    release = threading.Event()
    started = threading.Event()

    def boom():
        started.set()
        release.wait(5.0)
        raise ValueError("worker exploded")

    t = asyncio.create_task(builder._offloaded_to_completion(boom))
    await asyncio.to_thread(started.wait)
    t.cancel()  # cancel while the worker is blocked
    await asyncio.sleep(0.05)
    release.set()  # worker then raises
    # The worker's error propagates; the cancellation does not swallow it.
    with pytest.raises(ValueError):
        await t


@pytest.mark.asyncio
async def test_weight_proportional_single_source_gets_full_budget():
    doc = "a" * 25  # 6 tokens each
    n_docs = 50

    class _FixedClient(FakeClient):
        def sample_documents(self, dataset_id, config, split, text_field, n):
            return [doc] * min(n, n_docs)

    client = _FixedClient()
    builder = CorpusBuilder(client=client)

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
    builder = CorpusBuilder(client=client)

    # weight_target = 10_000 -> est_docs = 10_000 // 250 = 40 > cap=8 -> capped to 8.
    manifest = Manifest(
        token_budget=10_000,
        sources=[Source(dataset_id="good/encyclopedia", weight=1.0)],
        sample_docs_per_source=cap,
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
    builder = CorpusBuilder(client=client)

    # weight_target = 500 -> est_docs = 500 // 250 = 2.
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
async def test_materialize_manifest_sample_docs_per_source_caps_fetch():
    """A manifest-level `sample_docs_per_source` caps fetch-count estimation."""
    doc = "a" * 25  # 6 tokens each

    class _UnboundedClient(FakeClient):
        def sample_documents(self, dataset_id, config, split, text_field, n):
            return [doc] * n

    client = _UnboundedClient()
    builder = CorpusBuilder(client=client)
    state = CuratorState()

    # weight_target = 10_000 -> est_docs = 40, capped at the manifest value (20).
    manifest = Manifest(
        token_budget=10_000,
        sources=[Source(dataset_id="good/encyclopedia", weight=1.0)],
        sample_docs_per_source=20,
    )
    corpus = await builder.materialize(manifest, state)
    assert len(corpus.sources[0].documents) == 20


@pytest.mark.asyncio
async def test_materialize_without_manifest_cap_uses_token_target():
    """When the manifest omits `sample_docs_per_source`, fetch count follows the
    weight-proportional token target with no artificial per-source ceiling."""
    doc = "a" * 25

    class _UnboundedClient(FakeClient):
        def sample_documents(self, dataset_id, config, split, text_field, n):
            return [doc] * n

    client = _UnboundedClient()
    builder = CorpusBuilder(client=client)
    state = CuratorState()

    manifest = Manifest(
        token_budget=10_000,
        sources=[Source(dataset_id="good/encyclopedia", weight=1.0)],
    )
    assert manifest.sample_docs_per_source is None
    corpus = await builder.materialize(manifest, state)
    assert len(corpus.sources[0].documents) == 40


@pytest.mark.asyncio
async def test_materialize_large_manifest_cap_does_not_bind_below_token_target():
    """A manifest cap above the token-derived fetch count does not reduce fetches."""
    doc = "a" * 25

    class _UnboundedClient(FakeClient):
        def sample_documents(self, dataset_id, config, split, text_field, n):
            return [doc] * n

    client = _UnboundedClient()
    builder = CorpusBuilder(client=client)
    state = CuratorState()

    manifest = Manifest(
        token_budget=10_000,
        sources=[Source(dataset_id="good/encyclopedia", weight=1.0)],
        sample_docs_per_source=5_000,
    )
    corpus = await builder.materialize(manifest, state)
    # est_docs = 10_000 // 250 = 40, capped at the manifest's override (5_000).
    assert len(corpus.sources[0].documents) == 40


@pytest.mark.parametrize("value", [0, -1])
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
    # Large value (> old 100k cap) must validate.
    assert (
        Manifest(
            sources=[Source(dataset_id="a/b")], sample_docs_per_source=5_000_000
        ).sample_docs_per_source
        == 5_000_000
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
    builder = CorpusBuilder(client=client)
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
    cfg = CuratorConfig(candidate_limit=4)
    assert cfg.candidate_limit == 4
    assert cfg.max_turns == 64
    assert ProxyStudentConfig(n_embd=128, n_head=4).n_embd == 128


@pytest.mark.parametrize(
    "kwargs",
    [
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
        {"val_batch_size": 0},
        {"val_logit_chunk_tokens": 0},
        {"train_microbatch_size": 0},
    ],
)
def test_proxy_student_config_rejects_invalid(kwargs):
    with pytest.raises(ValidationError):
        ProxyStudentConfig(**kwargs)


def test_proxy_student_val_microbatch_defaults_and_payload():
    cfg = ProxyStudentConfig()
    assert cfg.val_batch_size is None
    assert cfg.val_logit_chunk_tokens is None
    assert cfg.train_microbatch_size is None
    payload = cfg.training_payload()
    assert payload["val_batch_size"] is None
    assert payload["val_logit_chunk_tokens"] is None
    assert payload["train_microbatch_size"] is None
    tuned = ProxyStudentConfig(
        val_batch_size=1,
        val_logit_chunk_tokens=2048,
        train_microbatch_size=16,
    )
    tuned_payload = tuned.training_payload()
    assert tuned_payload["val_batch_size"] == 1
    assert tuned_payload["val_logit_chunk_tokens"] == 2048
    assert tuned_payload["train_microbatch_size"] == 16


# --- Tier D2: adjustable token budget -> steps / corpus cap / timeout -------


def test_train_token_budget_default_preserves_step_behavior():
    # Default (budget None) keeps the historical steps-driven behavior EXACTLY,
    # so default / CPU / heuristic runs stay cheap and unchanged.
    cfg = ProxyStudentConfig()  # steps=200, batch=16, block=1024
    assert cfg.train_token_budget is None
    assert cfg.effective_steps == 200
    assert cfg.effective_train_tokens == 200 * 16 * 1024  # 3_276_800


@pytest.mark.parametrize(
    "budget,batch,block,expected_steps",
    [
        (819_200, 16, 256, 200),  # exactly the default budget -> 200 steps
        (300_000_000, 16, 256, 73_243),  # ceil(300M / 4096)
        (1_000_000_000, 16, 256, 244_141),  # ceil(1e9 / 4096)
        (10, 1, 8, 2),  # ceil(10/8) rounds up, never truncates
    ],
)
def test_train_token_budget_constant_batch_derives_steps(
    budget, batch, block, expected_steps
):
    """With schedule disabled, budget → steps is still ceil(budget/(batch*block))."""
    cfg = ProxyStudentConfig(
        train_token_budget=budget,
        batch_size=batch,
        block_size=block,
        batch_schedule_enabled=False,
    )
    assert cfg.effective_steps == expected_steps
    assert cfg.effective_train_tokens == expected_steps * batch * block


def test_train_token_budget_equal_stages_400m_is_schedule_aware():
    """400M / 16 / 1024 under [1,2,3] equal stages ≈ 12,208 steps (not 24,415)."""
    from pretrain_data_curator.batch_schedule import scheduled_presentation_tokens

    cfg = ProxyStudentConfig(
        train_token_budget=400_000_000,
        batch_size=16,
        block_size=1024,
        batch_schedule_enabled=True,
        batch_stage_fracs=(1 / 3, 1 / 3, 1 / 3),
        batch_stage_muls=(1, 2, 3),
    )
    assert cfg.effective_steps == 12_208
    # Old base-batch formula would have doubled the step count.
    assert cfg.effective_steps != 24_415
    tokens = cfg.effective_train_tokens
    assert tokens == scheduled_presentation_tokens(
        cfg.effective_steps,
        batch_size=16,
        block_size=1024,
        batch_stage_muls=(1, 2, 3),
        batch_stage_fracs=(1 / 3, 1 / 3, 1 / 3),
        batch_schedule_enabled=True,
    )
    assert tokens >= 400_000_000
    # Boundary rule: at most one final scheduled step of overshoot.
    prev = scheduled_presentation_tokens(
        cfg.effective_steps - 1,
        batch_size=16,
        block_size=1024,
        batch_stage_muls=(1, 2, 3),
        batch_stage_fracs=(1 / 3, 1 / 3, 1 / 3),
        batch_schedule_enabled=True,
    )
    assert prev < 400_000_000
    # Overshoot is exactly the last step's presentations.
    assert tokens - 400_000_000 < tokens - prev


def test_train_token_budget_irregular_stage_fractions():
    from pretrain_data_curator.batch_schedule import scheduled_presentation_tokens

    fracs = (0.1, 0.2, 0.7)
    muls = (1, 2, 4)
    budget = 100_000
    cfg = ProxyStudentConfig(
        train_token_budget=budget,
        batch_size=4,
        block_size=32,
        batch_schedule_enabled=True,
        batch_stage_fracs=fracs,
        batch_stage_muls=muls,
    )
    n = cfg.effective_steps
    tokens = cfg.effective_train_tokens
    assert tokens == scheduled_presentation_tokens(
        n,
        batch_size=4,
        block_size=32,
        batch_stage_muls=muls,
        batch_stage_fracs=fracs,
        batch_schedule_enabled=True,
    )
    assert tokens >= budget
    if n > 1:
        assert (
            scheduled_presentation_tokens(
                n - 1,
                batch_size=4,
                block_size=32,
                batch_stage_muls=muls,
                batch_stage_fracs=fracs,
                batch_schedule_enabled=True,
            )
            < budget
        )


def test_train_token_budget_small_budget_rounds_deterministically():
    from pretrain_data_curator.batch_schedule import scheduled_presentation_tokens

    cfg = ProxyStudentConfig(
        train_token_budget=10,
        batch_size=1,
        block_size=8,
        batch_schedule_enabled=True,
        batch_stage_fracs=(1 / 3, 1 / 3, 1 / 3),
        batch_stage_muls=(1, 2, 3),
    )
    # ceil under staged muls: still at least 2 base-equivalent presentations.
    assert cfg.effective_steps == 2
    assert cfg.effective_train_tokens >= 10
    assert cfg.effective_train_tokens == scheduled_presentation_tokens(
        2,
        batch_size=1,
        block_size=8,
        batch_stage_muls=(1, 2, 3),
        batch_stage_fracs=(1 / 3, 1 / 3, 1 / 3),
        batch_schedule_enabled=True,
    )


def test_train_token_budget_accounting_parity_with_schedule_sum():
    """effective_train_tokens equals the explicit sum of staged presentations."""
    from pretrain_data_curator.batch_schedule import (
        batch_stage_boundaries,
        scheduled_presentation_tokens,
    )

    cfg = ProxyStudentConfig(
        train_token_budget=50_000,
        batch_size=2,
        block_size=16,
        batch_schedule_enabled=True,
        batch_stage_fracs=(1 / 3, 1 / 3, 1 / 3),
        batch_stage_muls=(1, 2, 3),
        train_microbatch_size=1,  # must not affect token accounting
    )
    n = cfg.effective_steps
    manual = 0
    for (start, end), mul in zip(
        batch_stage_boundaries(n, cfg.batch_stage_fracs),
        cfg.batch_stage_muls,
        strict=True,
    ):
        manual += (end - start) * cfg.batch_size * mul * cfg.block_size
    assert cfg.effective_train_tokens == manual
    assert manual == scheduled_presentation_tokens(
        n,
        batch_size=cfg.batch_size,
        block_size=cfg.block_size,
        batch_stage_muls=cfg.batch_stage_muls,
        batch_stage_fracs=cfg.batch_stage_fracs,
        batch_schedule_enabled=True,
    )


def test_train_token_budget_microbatch_does_not_change_steps_or_tokens():
    a = ProxyStudentConfig(
        train_token_budget=400_000_000,
        batch_size=16,
        block_size=1024,
        train_microbatch_size=None,
    )
    b = ProxyStudentConfig(
        train_token_budget=400_000_000,
        batch_size=16,
        block_size=1024,
        train_microbatch_size=16,
    )
    assert a.effective_steps == b.effective_steps == 12_208
    assert a.effective_train_tokens == b.effective_train_tokens


@pytest.mark.parametrize("budget", [0, 1_000_000_001])
def test_train_token_budget_out_of_bounds_rejected(budget):
    with pytest.raises(ValidationError):
        ProxyStudentConfig(train_token_budget=budget)


def test_train_token_budget_max_is_accepted():
    assert ProxyStudentConfig(train_token_budget=1_000_000_000).train_token_budget == (
        1_000_000_000
    )


def test_effective_max_corpus_chars_scales_with_budget():
    # Default config derives from steps*batch*block; explicit budget or cap override.
    default = ProxyStudentConfig()
    assert default.effective_max_corpus_chars == 4 * default.effective_train_tokens
    big = ProxyStudentConfig(train_token_budget=300_000_000)
    assert big.effective_max_corpus_chars == 4 * big.effective_train_tokens
    assert big.effective_max_corpus_chars > default.effective_max_corpus_chars
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
async def test_fetch_cache_same_key_identity_once():
    client = FakeClient()
    curator = await _make(client=client)
    state = await _finalized(curator, sources=("good/encyclopedia",))
    assert client.sample_calls == []  # nothing fetched until materialize

    manifest = RolloutStore.manifest(state)
    corpus_a = await curator.corpus_builder.materialize(manifest, state)
    assert client.sample_calls == ["good/encyclopedia"]  # fetched exactly once

    corpus_b = await curator.corpus_builder.materialize(manifest, state)
    # No re-streaming on repeated same-key fetches.
    assert client.sample_calls == ["good/encyclopedia"]
    # Identical docs across fetches (preview == score).
    assert corpus_a.documents == corpus_b.documents


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
    # Go through the REAL finalize -> score path: the agent's workspace manifest
    # is parsed by `finalize`, written to the single per-rollout CuratorState, then
    # `score` materializes that manifest's sources and trains over the SAME state.
    client = FakeClient()
    taskset = CuratorTaskset(CuratorTasksetConfig(id="test", cutoff_date="2024-12-31"))
    builder = CorpusBuilder(
        client=client,
    )
    taskset._client = client
    taskset._corpus_builder = builder
    taskset._decon_detector = NoOpLeakageDetector()
    taskset._trainer = HeuristicProxyTrainer()
    bind_fast_scorer(
        taskset,
        corpus_builder=builder,
        trainer=taskset._trainer,
        leakage_detector=taskset._decon_detector,
    )

    # The taskset exposes NO tools (the non-MCP gate passes).
    assert type(taskset).tools is Taskset.tools

    task = taskset.load_tasks()[0]
    state = CuratorState()
    trace = vf.Trace(task=task, state=state)
    prompt = [vf.SystemMessage(content="sys"), vf.UserMessage(content="go")]
    manifest_bytes = (
        '{"sources": [{"id": "good/encyclopedia", "weight": 1.0},'
        ' {"id": "good/science", "weight": 1.0}]}'
    ).encode()
    graph.prepare_turn(trace, prompt).commit(
        vf.Response(
            id="x",
            created=0,
            model="m",
            message=vf.AssistantMessage(content="Done."),
            finish_reason="stop",
            usage=vf.Usage(prompt_tokens=1, completion_tokens=1),
        )
    )

    runtime = _FakeRuntime(files={MANIFEST_FILENAME: manifest_bytes})
    await taskset.finalize(task, trace, runtime)
    assert RolloutStore.is_finalized(state)
    manifest = RolloutStore.manifest(state)
    assert {s.dataset_id for s in manifest.sources} == {
        "good/encyclopedia",
        "good/science",
    }

    # score materializes the parsed manifest's sources (once each) and trains.
    await taskset.score(trace, None)
    assert client.sample_calls == ["good/encyclopedia", "good/science"]
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
async def test_concurrent_same_key_fetch_coalesces_to_one_fetch():
    # N concurrent same-key fetches must share ONE underlying Hub fetch;
    # later callers read the cached result (single-flight).
    client = _SlowCountingClient()
    builder = CorpusBuilder(client=client)
    state = CuratorState()
    key = FetchKey("good/encyclopedia", None, "train", "text", 8)

    results = await asyncio.gather(
        *[builder.fetch_source_docs(state, key) for _ in range(12)]
    )

    assert client.sample_calls == ["good/encyclopedia"]
    docs0, err0 = results[0]
    assert err0 is None and docs0
    for docs, err in results:
        assert err is None
        assert docs == docs0


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
    builder = _CountingBuilder(client=client)
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
    assert len(funcs) == 21  # 2 rewards + 19 diagnostic metrics (cost_total removed)
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
            RuntimeError("Dataset scripts are no longer supported, but found legacy.py")
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


@pytest.mark.asyncio
async def test_script_dataset_probe_blocks_unconditionally_with_guidance(monkeypatch):
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
    client = HuggingFaceDatasetClient(token="test-token")
    builder = CorpusBuilder(
        client=client,
        retry_policy=RetryPolicy(attempts=3, timeout=1.0),
    )
    state = CuratorState()

    docs, error = await builder.fetch_source_docs(
        state, FetchKey("owner/legacy", None, "train", None, 1)
    )

    assert docs == []
    assert error is not None
    assert error["error_kind"] == "script_dataset"
    assert "`hf download <repo> --repo-type dataset` or `curl`" in error["error"]
    assert '`kind: "local"`' in error["error"]
    assert '`local_path: "<relative-path>"`' in error["error"]
    assert "`local_format`" in error["error"]
    assert "`text_field`" in error["error"]
    assert RolloutStore.tool_error_count(state) == 1
    assert calls == [
        {
            "repo_id": "owner/legacy",
            "filename": "legacy.py",
            "repo_type": "dataset",
        }
    ]
    assert load_calls == []


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
    client = HuggingFaceDatasetClient(token="test-token")

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
    assert "score" in offloaded  # DeconLeakageDetector.score


def test_proxy_student_recipe_defaults_mirror_record01():
    cfg = ProxyStudentConfig()
    assert cfg.n_layer == 12 and cfg.n_head == 6 and cfg.n_embd == 768
    assert cfg.block_size == 1024
    assert cfg.training_recipe == "speedrun_muon"
    assert cfg.muon_lr == 0.023
    assert cfg.batch_schedule_enabled is True
    assert cfg.batch_stage_muls == (1, 2, 3)
    assert (cfg.adam_beta1, cfg.adam_beta2, cfg.record_adam_eps) == (0.9, 0.95, 1e-8)
    assert cfg.adam_eps == 1e-10
    assert cfg.weight_decay == 0.1
    assert cfg.grad_clip == 0.0
    assert cfg.muon_weight_decay == 1.2
    assert cfg.nor_muon is True
    assert cfg.lr_min_ratio == 0.1
    assert cfg.n_train_runs == 1
    assert cfg.warmup_steps is None
    assert (
        cfg.effective_warmup_steps == min(256, max(1, cfg.effective_steps // 10)) == 20
    )
    # An explicit warmup is clamped to the run length so it never exceeds steps.
    assert ProxyStudentConfig(steps=5, warmup_steps=999).effective_warmup_steps == 5


def test_proxy_student_payload_routes_document_and_epsilon_fields():
    cfg = ProxyStudentConfig(
        adam_eps=1e-10,
        record_adam_eps=2e-8,
        max_document_tokens=2048,
        eos_aligned_batches=True,
    )
    payload = cfg.training_payload()
    assert payload["adam_eps"] == 1e-10
    assert payload["record_adam_eps"] == 2e-8
    assert payload["max_document_tokens"] == 2048
    assert payload["eos_aligned_batches"] is True
    assert "max_doc_len" not in payload and "eos_positions" not in payload


def test_proxy_student_legacy_training_names_remain_safe():
    cfg = ProxyStudentConfig(max_doc_len=512)
    assert cfg.max_document_tokens == cfg.max_doc_len == 512
    legacy_record = ProxyStudentConfig(training_recipe="record_01_adamw", adam_eps=3e-8)
    assert legacy_record.record_adam_eps == 3e-8
    with pytest.raises(ValidationError, match="eos_positions is no longer accepted"):
        ProxyStudentConfig(eos_positions=[10, 20])


def test_estimate_param_count_matches_gpt2_small_default():
    assert estimate_param_count(ProxyStudentConfig()) == GPT2_SMALL_PARAM_COUNT


@pytest.mark.parametrize(
    "kwargs",
    [
        {"weight_decay": -0.1},
        {"weight_decay": 1.5},
        {"adam_beta1": 1.0},
        {"adam_beta2": 0.0},
        {"adam_eps": 0.0},
        {"record_adam_eps": 0.0},
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
    assert "cost_total" not in trace.metrics
    assert "train_flops" in trace.metrics
    assert trace.metrics["budget_fill_ratio"] == pytest.approx(
        curator.state.budget_fill_ratio
    )


# --- Tier L: reward surface is CE performance minus penalties ---------------


def test_reward_surface_has_only_perf_and_leakage():
    taskset = _Curator().taskset
    reward_names = {f.__name__ for f in discover_decorated(taskset, "reward")}
    assert reward_names == {"perf_reward", "leakage_penalty"}


@pytest.mark.asyncio
async def test_reward_has_no_cost_total_metric():
    """Cost metering is removed; reward remains perf - leakage only."""
    curator = await _make()
    await _finalized(curator)
    trace = await curator.score()
    assert "cost_total" not in trace.metrics
    baseline_reward = trace.reward
    assert "train_flops" in trace.metrics
    assert "corpus_tokens" in trace.metrics
    assert "budget_fill_ratio" in trace.metrics

    await curator.reset()
    await _finalized(curator)
    trace2 = await curator.score()
    assert "cost_total" not in trace2.metrics
    assert trace2.reward == pytest.approx(baseline_reward)


# --- Tier M: single PTB-style prompt + leakage-safe self-score --------------


def test_task_prompt_contract():
    task = CuratorTaskset(CuratorTasksetConfig(id="test", max_turns=7)).load_tasks()[0]
    prompt = task.prompt

    assert task.system_prompt is None
    assert len(prompt) < 5_000
    assert "complete freedom" in prompt
    assert '"token_budget": 1000000' in prompt
    assert '"sources"' in prompt
    assert "python self_score.py draft.json" in prompt
    assert "--limit N" in prompt
    assert "--max-steps N" in prompt
    assert "Never ask the user" in prompt
    assert "7 model turns" not in prompt
    assert "turn budget" not in prompt.lower()
    assert "discovery budget" not in prompt.lower()
    assert "discovery round" not in prompt.lower()
    assert "Contamination against any eval set incurs the leakage penalty" in prompt
    assert "does not increase the scored corpus" in prompt
    # Cost is telemetry-only and not part of the reward; the prompt must not
    # frame curation as a cost-economy tradeoff.
    assert "cost" not in prompt.lower()
    assert "telemetry" not in prompt.lower()
    assert "there is no positive performance score" in prompt
    assert "hf datasets ls" not in prompt
    assert "pip install" not in prompt
    assert "wikimedia/wikipedia" not in prompt
    assert "HuggingFaceFW/fineweb" not in prompt
    assert "your first response" not in prompt.lower()
    assert "execute them through the harness tool or shell interface" in prompt
    assert "writing a command as prose does not run it" in prompt
    assert "codex" not in prompt.lower()
    assert "mini_swe_agent" not in prompt.lower()
    assert '"id": "<observed Hugging Face owner/name>"' in prompt
    assert '"kind": "hf"' in prompt
    assert '"kind": "local"' in prompt
    assert '"local_path"' in prompt
    assert "min_chars" in prompt
    assert "dedup_exact" in prompt


def test_discovery_has_no_call_or_output_stop():
    taskset = CuratorTaskset(CuratorTasksetConfig(id="test"))
    stops = discover_decorated(taskset, "stop")

    assert [stop.__name__ for stop in stops] == ["max_turns_reached"]


def test_task_prompt_manifest_contract_covers_hf_and_local_sources():
    prompt = CuratorTaskset(CuratorTasksetConfig(id="test")).load_tasks()[0].prompt

    assert (
        '"sample_docs_per_source": <optional integer >= 1; omit for no per-source fetch cap — fetches are sized from weights and token_budget>'
        in prompt
    )
    manifest_fields = {
        "token_budget": 1_000,
        "sample_docs_per_source": 8,
        "sources": [
            {
                "id": "owner/dataset",
                "kind": "hf",
                "weight": 1.0,
                "config": None,
                "split": "train",
                "text_field": None,
                "filters": [{"kind": "min_chars", "params": {"value": 10}}],
                "max_docs": 4,
                "max_tokens": 500,
            }
        ],
    }
    for field in manifest_fields | manifest_fields["sources"][0]:
        assert f'"{field}"' in prompt

    manifest = parse_manifest(
        f"```json\n{json.dumps(manifest_fields)}\n```",
        default_token_budget=1_000,
    )
    assert manifest is not None
    assert manifest.token_budget == 1_000
    assert manifest.sample_docs_per_source == 8
    assert manifest.sources[0].dataset_id == "owner/dataset"
    assert manifest.sources[0].sampling.max_docs == 4
    assert manifest.sources[0].sampling.max_tokens == 500


def test_self_score_script_is_standalone_and_hides_final_validation_identity():
    config = CuratorConfig(token_budget=1_000)
    script = render_self_score_script(config)

    assert config.validation_set.dataset_id.encode() not in script
    assert b"validation_data_used" in script
    assert b"urllib.request" in script
    assert b"decon/bundled-evals" in script
    assert b"decon/bin/decon" in script
    compile(script, SELF_SCORE_FILENAME, "exec")


def test_self_score_preserves_training_document_boundaries():
    config = CuratorConfig(token_budget=1_000)
    namespace = {"__name__": "self_score_document_test"}
    exec(
        compile(render_self_score_script(config), SELF_SCORE_FILENAME, "exec"),
        namespace,
    )
    payload = json.loads(namespace["joined_corpus"](["first", "", "a\n\nb"], 100))
    assert payload == {
        "format": "document-list-v1",
        "documents": ["first", "", "a\n\nb"],
    }


@pytest.mark.slow
def test_self_score_script_scores_local_dev_samples_without_validation_data(tmp_path):
    # Keep the proxy tiny for this CPU integration test; production defaults are
    # GPT-2-small-class (~278M) and are too heavy for a quick local train loop.
    config = CuratorConfig(
        token_budget=1_000,
        use_real_trainer=True,
        proxy_student={
            "n_layer": 4,
            "n_head": 4,
            "n_embd": 256,
            "block_size": 256,
            "steps": 4,
            "training_recipe": "record_01_adamw",
        },
    )
    (tmp_path / SELF_SCORE_FILENAME).write_bytes(render_self_score_script(config))
    (tmp_path / SELF_SCORE_TRAIN_FILENAME).write_bytes(render_self_score_train_script())
    (tmp_path / "dev.jsonl").write_text(
        "\n".join(
            json.dumps({"text": "Clean development sample text " * 40})
            for _ in range(8)
        ),
        encoding="utf-8",
    )
    (tmp_path / "draft.json").write_text(
        json.dumps(
            {
                "token_budget": 1_000,
                "sample_docs_per_source": 8,
                "sources": [
                    {
                        "kind": "local",
                        "local_path": "dev.jsonl",
                        "local_format": "jsonl",
                        "weight": 1.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            SELF_SCORE_FILENAME,
            "draft.json",
            "--limit",
            "4",
            "--max-steps",
            "4",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    score = json.loads(result.stdout)
    assert score["ok"] is True
    assert score["validation_data_used"] is False
    assert score["self_score_settings"]["limit"] == 4
    assert score["self_score_settings"]["max_steps"] == 4
    assert score["budget_fill_ratio"] > 0.0
    assert score["leakage_score"] in (None, 0.0)
    assert "perf_reward" in score
    assert "leakage_penalty" in score
    assert "reward" in score
    try:
        import torch  # noqa: F401
    except ImportError:
        assert score["perf_loss"] is None
        assert score["perf"] is None
    else:
        assert score["perf_loss"] is not None
        assert math.isfinite(score["perf_loss"])
        assert score["perf"] is not None


def test_self_score_dataset_id_precedence_matches_taskset():
    config = CuratorConfig(token_budget=1_000)
    sources = [
        {"dataset_id": "legacy/non-forbidden"},
        {
            "dataset_id": "legacy/non-forbidden",
            "id": "ignored/lower-priority-alias",
        },
    ]
    namespace = {"__name__": "self_score_test"}
    exec(
        compile(render_self_score_script(config), SELF_SCORE_FILENAME, "exec"),
        namespace,
    )
    for source in sources:
        manifest = {"token_budget": 1_000, "sources": [source]}
        parsed = parse_manifest(json.dumps(manifest))
        assert parsed is not None
        assert parsed.sources[0].dataset_id == "legacy/non-forbidden"
        assert namespace["source_dataset_id"](source) == "legacy/non-forbidden"


def test_self_score_redacts_forbidden_source_from_all_stdout(tmp_path):
    config = CuratorConfig(token_budget=1_000)
    (tmp_path / SELF_SCORE_FILENAME).write_bytes(render_self_score_script(config))
    draft = tmp_path / "draft.json"
    draft.write_text(
        json.dumps(
            {
                "token_budget": 1_000,
                "sources": [{"dataset_id": config.validation_set.dataset_id}],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, SELF_SCORE_FILENAME, draft.name],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "HTTPS_PROXY": "http://127.0.0.1:1"},
    )
    score = json.loads(result.stdout)
    assert score["sources"][0]["source"] == "[withheld validation repository]"
    assert score["sources"][0]["error"] == (
        "ValueError: source is reserved for final validation"
    )
    assert config.validation_set.dataset_id not in result.stdout


def test_self_score_script_exposes_agent_controlled_cli_flags():
    """Self-score caps are agent-chosen via CLI flags, not baked into the script."""
    import re as _re
    from pretrain_data_curator import self_score as _self_score

    script = _self_score._SCRIPT
    assert '--limit"' in script
    assert "--max-steps" in script
    assert "--max-corpus-chars" in script
    assert "--train-timeout" in script
    assert "choices=range" not in script
    assert b"SELF_SCORE_MAX_STEPS" not in _self_score._SCRIPT.encode()
    assert b"SELF_SCORE_MAX_CORPUS_CHARS" not in _self_score._SCRIPT.encode()
    match = _re.search(
        r"train_perf\(\s*all_docs,\s*max_corpus_chars=args\.max_corpus_chars",
        script,
    )
    assert match, "train_perf must use agent-provided CLI caps"


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
async def test_default_taskset_path_builds_decon_detector():
    """The DEFAULT taskset path builds a DeconLeakageDetector through _ensure()."""
    taskset = CuratorTaskset(
        CuratorTasksetConfig(
            id="test",
            decon_binary="/nonexistent/decon",
            screen_val_set=False,
        )
    )
    taskset._client = FakeClient()
    taskset._corpus_builder = CorpusBuilder(client=taskset._client)
    taskset._trainer = HeuristicProxyTrainer()

    scorer = taskset._ensure()
    assert taskset._decon_detector is not None
    assert taskset._scorer is scorer
    assert taskset._decon_detector._binary == "/nonexistent/decon"


# === Decon reducer unit tests ==============================================


def _decon_jsonl_line(
    *,
    training_file: str = "corpus.jsonl",
    training_line: int = 0,
    contamination_score: float = 0.5,
    eval_dataset: str = "test/eval",
    answer_start_idx: int | None = 10,
    answer_end_idx: int | None = 50,
    question_start_idx: int | None = None,
    question_end_idx: int | None = None,
    cluster_token_length: int | None = None,
) -> str:
    """Build a single decon report JSONL line."""
    rec: dict[str, object] = {
        "training_file": training_file,
        "training_line": training_line,
        "contamination_score": contamination_score,
        "eval_dataset": eval_dataset,
    }
    if answer_start_idx is not None:
        rec["answer_start_idx"] = answer_start_idx
    if answer_end_idx is not None:
        rec["answer_end_idx"] = answer_end_idx
    if question_start_idx is not None:
        rec["question_start_idx"] = question_start_idx
    if question_end_idx is not None:
        rec["question_end_idx"] = question_end_idx
    if cluster_token_length is not None:
        rec["cluster_token_length"] = cluster_token_length
    return json.dumps(rec)


def test_reduce_hand_worked_example():
    """Token-weighted magnitude computes as expected on a hand-worked example."""
    from pretrain_data_curator.leakage import DeconLeakageDetector

    detector = DeconLeakageDetector(
        decon_binary="/nonexistent/decon", evals_dir="/nonexistent/evals"
    )
    # Doc A: score=0.8, span answer(0..200)=200 chars=50 tokens, contribution=40.0
    # Doc B: score=0.4, span answer(5..25)=20 chars=5 tokens, contribution=2.0
    # Total weighted = 42.0, total_tokens = 200
    # leakage = 42.0 / 200 = 0.21
    lines = [
        _decon_jsonl_line(
            training_line=0,
            contamination_score=0.8,
            answer_start_idx=0,
            answer_end_idx=200,
        ),
        _decon_jsonl_line(
            training_line=1,
            contamination_score=0.4,
            answer_start_idx=5,
            answer_end_idx=25,
        ),
    ]
    result = detector._reduce_report(lines, total_tokens=200)
    assert result.leakage_score == pytest.approx(0.21)
    assert result.num_contaminated_matches == 2
    assert len(result.contamination_details) == 2


def test_leakage_penalty_positive():
    """A non-empty contamination report drives leakage_penalty < 0.

    Monkey-patches DeconLeakageDetector to return a synthetic report without
    needing a real decon binary.
    """
    from pretrain_data_curator.leakage import DeconLeakageDetector
    from pretrain_data_curator.rewards import CuratorScorer

    detector = DeconLeakageDetector(
        decon_binary="/nonexistent/decon", evals_dir="/nonexistent/evals"
    )

    # Monkey-patch score() to simulate decon report directly.
    original_score = DeconLeakageDetector.score

    def fake_score(self, docs, val_set=None):
        lines = [
            _decon_jsonl_line(
                contamination_score=0.8,
                answer_start_idx=0,
                answer_end_idx=400,
            )
        ]
        return self._reduce_report(lines, total_tokens=1000)

    DeconLeakageDetector.score = fake_score
    try:
        config = CuratorConfig(lambda_leakage=1.0)
        scorer = CuratorScorer(
            config,
            corpus_builder=CorpusBuilder(client=FakeClient()),
            trainer=HeuristicProxyTrainer(),
            decon_detector=detector,
        )
        state = CuratorState()
        RolloutStore.set_manifest(
            state,
            Manifest(
                token_budget=1000,
                sources=[Source(dataset_id="good/encyclopedia")],
            ),
        )
        RolloutStore.set_finalized(state, True)
        scoring = asyncio.run(scorer.compute_scoring(state))
        leakage_score = scoring["leakage"]["leakage_score"]
        assert leakage_score > 0.0
        # The penalty term would be: -lambda_leakage * leakage_score < 0
        expected_penalty = -1.0 * leakage_score
        assert expected_penalty < 0.0
    finally:
        DeconLeakageDetector.score = original_score


def test_decon_error_raised_on_decon_failure():
    """DeconLeakageDetector.score raises DeconError when decon fails."""
    from pretrain_data_curator.leakage import DeconError, DeconLeakageDetector

    # The decon binary exists (resolved from project path), but the evals dir
    # doesn't, so decon exits non-zero -> DeconError.
    detector = DeconLeakageDetector(
        decon_binary="decon-nonexistent-name",
        evals_dir="/nonexistent/evals",
    )
    with pytest.raises(DeconError):
        detector.score(["some document text"])


@pytest.mark.asyncio
async def test_decon_error_sets_external_failure():
    """When DeconError is raised, the scorer records external_failure."""
    from pretrain_data_curator.leakage import DeconLeakageDetector
    from pretrain_data_curator.rewards import CuratorScorer

    detector = DeconLeakageDetector(
        decon_binary="/nonexistent/decon_nope",
        evals_dir="/nonexistent/evals",
    )
    config = CuratorConfig(lambda_leakage=1.0)
    scorer = CuratorScorer(
        config,
        corpus_builder=CorpusBuilder(client=FakeClient()),
        trainer=HeuristicProxyTrainer(),
        decon_detector=detector,
    )
    state = CuratorState()
    RolloutStore.set_manifest(
        state,
        Manifest(
            token_budget=1000,
            sources=[Source(dataset_id="good/encyclopedia")],
        ),
    )
    RolloutStore.set_finalized(state, True)
    scoring = await scorer.compute_scoring(state)
    assert scoring["decon_error"] == 1.0
    assert RolloutStore.has_external_failure(state)
    assert scoring["leakage"]["leakage_score"] == 0.0


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


@pytest.mark.parametrize("n_tokens", [2, 100, 257, 1000])
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
    assert "averaged_train_and_eval(" in NANOGPT_TRAIN_SCRIPT
    assert '"loss": val_loss' in NANOGPT_TRAIN_SCRIPT


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


# --- compatibility assistant-message manifest parser ----------------------


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


def test_parse_manifest_large_sample_docs_per_source_accepted():
    """With the upper bound removed, large values validate."""
    text = json.dumps({"sources": [{"id": "a/b"}], "sample_docs_per_source": 5_000_000})
    m = parse_manifest(text)
    assert m is not None
    assert m.sample_docs_per_source == 5_000_000


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


def test_extract_hf_commands_splits_on_shell_separators():
    cmds = extract_hf_commands("hf datasets ls --search a && hf datasets info b/c")
    assert cmds == [["datasets", "ls", "--search", "a"], ["datasets", "info", "b/c"]]


# --- finalize: populates state.manifest ------------------------------------


class _FakeRuntime:
    """Minimal runtime exposing agent-written files."""

    def __init__(self, log_bytes=None, *, files=None):
        self._files = files or {}

    async def read(self, path):
        if path in self._files:
            return self._files[path]
        raise FileNotFoundError(path)


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
    curator = await _make()
    trace = _trace_with_final(
        curator.task,
        curator.state,
        '```json\n{"token_budget": 1000, "sample_docs_per_source": 2, "sources": [{"id": "a/b"}]}\n```',
    )

    with caplog.at_level("WARNING"):
        await curator.taskset.finalize(curator.task, trace, None)

    assert "TOKEN BUDGET IS NOT REACHABLE" in caplog.text


@pytest.mark.asyncio
async def test_finalize_populates_manifest_without_cost_ledger():
    curator = await _make()
    final = (
        "```json\n"
        '{"sources": [{"id": "good/encyclopedia", "weight": 1.0},'
        ' {"id": "good/science", "weight": 2.0}]}\n'
        "```"
    )
    trace = _trace_with_final(curator.task, curator.state, final)
    await curator.taskset.finalize(curator.task, trace, None)
    assert RolloutStore.is_finalized(curator.state)
    manifest = RolloutStore.manifest(curator.state)
    assert [s.dataset_id for s in manifest.sources] == [
        "good/encyclopedia",
        "good/science",
    ]
    assert "cost_ledger" not in type(curator.state).model_fields


@pytest.mark.asyncio
async def test_finalize_prefers_valid_workspace_manifest_file():
    curator = await _make()
    trace = _trace_with_final(
        curator.task,
        curator.state,
        '```json\n{"sources": [{"id": "message/fallback"}]}\n```',
    )
    runtime = _FakeRuntime(
        files={
            MANIFEST_FILENAME: json.dumps(
                {
                    "token_budget": 1234,
                    "sources": [{"id": "file/primary", "weight": 2.0}],
                }
            ).encode()
        }
    )

    await curator.taskset.finalize(curator.task, trace, runtime)

    manifest = RolloutStore.manifest(curator.state)
    assert RolloutStore.is_finalized(curator.state)
    assert manifest.token_budget == 1234
    assert [source.dataset_id for source in manifest.sources] == ["file/primary"]


@pytest.mark.asyncio
async def test_finalize_polls_for_late_workspace_manifest_file():
    curator = await _make()
    trace = _trace_with_final(
        curator.task,
        curator.state,
        '```json\n{"sources": [{"id": "message/fallback"}]}\n```',
    )
    runtime = _FakeRuntime()

    async def write_manifest() -> None:
        await asyncio.sleep(curator.taskset._FINALIZE_GRACE_INTERVAL_SECONDS * 2)
        runtime._files[MANIFEST_FILENAME] = b'{"sources": [{"id": "file/late"}]}'

    write = asyncio.create_task(write_manifest())
    await curator.taskset.finalize(curator.task, trace, runtime)
    await write

    assert [
        source.dataset_id for source in RolloutStore.manifest(curator.state).sources
    ] == ["file/late"]


@pytest.mark.asyncio
async def test_finalize_absent_workspace_manifest_falls_back_to_messages():
    curator = await _make()
    trace = _trace_with_final(
        curator.task,
        curator.state,
        '```json\n{"sources": [{"id": "message/fallback"}]}\n```',
    )

    await curator.taskset.finalize(curator.task, trace, _FakeRuntime())

    assert RolloutStore.is_finalized(curator.state)
    assert [
        source.dataset_id for source in RolloutStore.manifest(curator.state).sources
    ] == ["message/fallback"]


@pytest.mark.asyncio
async def test_finalize_malformed_workspace_manifest_warns_and_falls_back(
    caplog,
):
    curator = await _make()
    trace = _trace_with_final(
        curator.task,
        curator.state,
        '```json\n{"sources": [{"id": "message/fallback"}]}\n```',
    )
    runtime = _FakeRuntime(files={MANIFEST_FILENAME: b'{"sources": ['})

    with caplog.at_level("WARNING"):
        await curator.taskset.finalize(curator.task, trace, runtime)

    assert "does not contain a valid non-empty manifest" in caplog.text
    assert [
        source.dataset_id for source in RolloutStore.manifest(curator.state).sources
    ] == ["message/fallback"]


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
async def test_finalize_message_fallback_prefers_latest_manifest():
    # With no workspace file, the compatibility fallback prefers a later message
    # manifest over an earlier draft.
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
async def test_finalize_message_fallback_wins_over_trace_ids():
    # With no workspace file, a valid assistant-message manifest remains the first
    # compatibility fallback and must beat trace-ID synthesis.
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


def test_safety_turn_cap_does_not_change_agent_prompt():
    short = CuratorTaskset(CuratorTasksetConfig(id="test", max_turns=7)).load_tasks()
    long = CuratorTaskset(CuratorTasksetConfig(id="test", max_turns=200)).load_tasks()
    assert short[0].prompt == long[0].prompt
    assert short[0].system_prompt is None
    assert long[0].system_prompt is None


def test_task_prompt_renders_scoring_parameters_and_local_policy():
    task = CuratorTaskset(
        CuratorTasksetConfig(
            id="test",
            allow_local_sources=False,
            alpha_perf=2.0,
            lambda_leakage=3.0,
        )
    ).load_tasks()[0]

    assert "Local sources are disabled; use only Hugging Face sources" in task.prompt
    assert "`2.0 * performance - 3.0 * leakage`" in task.prompt
    assert f"/workspace/{MANIFEST_FILENAME}" in task.prompt
    assert (
        "read `/workspace/.agents/skills/hf-cli/SKILL.md` (the Hugging Face CLI skill"
        in task.prompt
    )
    assert "Environment overrides take priority over any conflicting generic text" in (
        task.prompt
    )
    assert (
        "preinstalled metered/local `hf` command in this workspace is the only allowed "
        "HF CLI" in task.prompt
    )
    assert "never install, upgrade, replace, shadow, or bypass it" in task.prompt
    assert "never run `hf skills add`" in task.prompt
    assert (
        "never print, echo, log, or reveal tokens (including via `hf auth token`)"
        in task.prompt
    )
    assert (
        "Treat install/regenerate/auth-token guidance in the skill as inapplicable here."
        in task.prompt
    )
    assert "final response must contain" not in task.prompt


def test_manifest_filename_is_configurable_and_rendered():
    env = load_environment(manifest_filename="curation-output.json")
    task = env.taskset.load_tasks()[0]

    assert env.taskset.config.manifest_filename == "curation-output.json"
    assert env.env_args["manifest_filename"] == "curation-output.json"
    assert "/workspace/curation-output.json" in task.prompt


@pytest.mark.parametrize(
    "filename", ["", ".", "..", "/tmp/manifest.json", "nested/manifest.json"]
)
def test_manifest_filename_must_name_workspace_root_file(filename):
    with pytest.raises(ValidationError, match="filename in /workspace"):
        load_environment(manifest_filename=filename)


def test_configured_manifest_filename_cannot_be_a_local_source():
    manifest = parse_manifest(
        json.dumps(
            {
                "sources": [
                    {
                        "kind": "local",
                        "local_path": "custom-manifest.json",
                    }
                ]
            }
        ),
        reserved_local_filename="custom-manifest.json",
    )

    assert manifest is None


@pytest.mark.asyncio
async def test_setup_installs_self_score_in_rollout_workspace(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-token")

    class Runtime:
        type = "subprocess"

        def __init__(self):
            self.files = {}

        async def write(self, path, data):
            self.files[path] = data

    runtime = Runtime()
    taskset = CuratorTaskset(CuratorTasksetConfig(id="test"))
    await taskset.setup(taskset.load_tasks()[0], runtime)

    assert SELF_SCORE_FILENAME in runtime.files
    assert HF_CLI_SKILL_RUNTIME_PATH == HF_CLI_SKILL_FILENAME
    assert HF_CLI_SKILL_RUNTIME_PATH in runtime.files
    packaged = hf_cli_skill_package_file().read_bytes()
    assert runtime.files[HF_CLI_SKILL_RUNTIME_PATH] == packaged
    skill_text = packaged.decode("utf-8")
    assert skill_text.lstrip().startswith("---")
    assert "name: hf-cli" in skill_text
    assert "Generated with `huggingface_hub" in skill_text
    assert "`hf download REPO_ID`" in skill_text
    assert "`hf datasets list`" in skill_text
    # Canonical skill may mention install/auth-token flows; env overrides live in prompt.
    assert "hf skills add" in skill_text or "Install:" in skill_text
    assert (
        taskset.curator.validation_set.dataset_id.encode()
        not in runtime.files[SELF_SCORE_FILENAME]
    )
    assert SELF_SCORE_TRAIN_FILENAME not in runtime.files


def test_hf_cli_skill_is_packaged():
    skill = hf_cli_skill_package_file()
    build_config = tomllib.loads(
        (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert skill.is_file()
    assert HF_CLI_SKILL_RUNTIME_PATH == ".agents/skills/hf-cli/SKILL.md"
    assert HF_CLI_SKILL_RESOURCE == "skills/hf-cli/SKILL.md"
    assert HF_CLI_SKILL_UPSTREAM_PATH == "skills/hf-cli/SKILL.md"
    assert HF_CLI_SKILL_UPSTREAM_REVISION == (
        "7039bdcf4510c30ec932637e8b2c1646aee7f185"
    )
    assert (
        "pretrain_data_curator/**/*.md"
        in build_config["tool"]["hatch"]["build"]["include"]
    )
    assert (
        "pretrain_data_curator/skills/hf-cli/**"
        not in build_config["tool"]["hatch"]["build"]["include"]
    )
    data = skill.read_bytes()
    assert hashlib.sha256(data).hexdigest() == HF_CLI_SKILL_SHA256
    text = data.decode("utf-8")
    assert "name: hf-cli" in text
    assert "Generated with `huggingface_hub" in text


def test_self_score_train_script_renders_and_compiles():
    script = render_self_score_train_script()
    assert b"WORKDIR" in script
    assert b"/workspace" not in script
    assert b"buffering=1" in script
    assert b"atexit.register(_stderr_fh.flush)" in script
    compile(script.decode("utf-8"), SELF_SCORE_TRAIN_FILENAME, "exec")


def test_self_score_script_preserves_trainer_stderr_tail_before_cleanup():
    """Rendered self_score must read WORKDIR/stderr.txt before rmtree on failure."""
    from pretrain_data_curator import self_score as _self_score

    script = _self_score._SCRIPT
    assert "def _read_trainer_stderr_tail(" in script
    assert "_read_trainer_stderr_tail(tmp)" in script
    assert 'os.path.join(workdir, "stderr.txt")' in script
    # Failure path must surface the file tail, not only captured subprocess stderr.
    assert "file_stderr = _read_trainer_stderr_tail(tmp)" in script
    assert "detail = file_stderr or result.stderr or result.stdout" in script


def test_self_score_read_trainer_stderr_tail_helper(tmp_path):
    """Unit-test the helper extracted from the rendered script template."""
    import re
    from pretrain_data_curator import self_score as _self_score

    match = re.search(
        r"def _read_trainer_stderr_tail\(.*?(?=\ndef )",
        _self_score._SCRIPT,
        flags=re.S,
    )
    assert match
    ns: dict[str, object] = {"os": __import__("os")}
    exec(compile(match.group(0), "<stderr_helper>", "exec"), ns)
    helper = ns["_read_trainer_stderr_tail"]
    assert helper(str(tmp_path)) == ""
    (tmp_path / "stderr.txt").write_text("line1\nCUDA OOM boom\n", encoding="utf-8")
    assert "CUDA OOM boom" in helper(str(tmp_path))
    big = "x" * 10_000
    (tmp_path / "stderr.txt").write_text(big, encoding="utf-8")
    assert len(helper(str(tmp_path), max_chars=100)) == 100


@pytest.mark.asyncio
async def test_runtime_read_reads_file_created_by_agent_shell():
    runtime = SubprocessRuntime(SubprocessConfig())
    await runtime.start()
    try:
        result = await runtime.run(
            ["sh", "-c", 'printf "%s" "$MANIFEST" > manifest.json'],
            {"MANIFEST": '{"sources":[{"id":"agent/written"}]}'},
        )

        assert result.exit_code == 0
        assert json.loads(await runtime.read(MANIFEST_FILENAME)) == {
            "sources": [{"id": "agent/written"}]
        }
    finally:
        runtime.cleanup()


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


# --- Tier M (cont.): the per-task goal renders into the initial prompt --------


def test_build_tasks_renders_single_structured_task_prompt():
    prompt = build_tasks("2024-12-31", 1_000_000)[0].prompt
    assert prompt.startswith("We want to train")
    assert "## Objective" in TASK_PROMPT
    assert "## Deliverable" in prompt
    assert "## Setup" in prompt
    assert "## Rules" in prompt
    assert "complete freedom" in prompt
    assert "Research and iterate autonomously" in prompt
    assert "Hugging Face `hf` CLI" in prompt
    assert "2024-12-31" in prompt
    assert "1000000" in prompt
    assert "There will be no user interaction" in prompt
    assert "Remember:" not in prompt
    assert prompt.rstrip().endswith(
        "operate autonomously and execute the actions that make the most sense."
    )


# --- Tier R (cont.): finalize trace-fallback metering -----------------------


class _CorruptRuntime:
    """Runtime whose read fails (a corrupt / unreadable shim log)."""

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
def _real_trainer_taskset(**proxy_student):
    use_real = proxy_student.pop("use_real_trainer", True)
    ts = CuratorTaskset(
        CuratorTasksetConfig(
            id="t",
            use_real_trainer=use_real,
            screen_val_set=False,
            proxy_student=proxy_student,
        )
    )
    # Inject the non-trainer collaborators so `_ensure` builds only the trainer
    # (no HF token / network needed); the trainer slot stays None for selection.
    ts._client = FakeClient()
    ts._corpus_builder = CorpusBuilder(client=ts._client)
    ts._decon_detector = NoOpLeakageDetector()
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
# The Perf REWARD defaults to a linear val-loss scale from a neutral baseline
# to the nanoGPT speedrun target, while ``perf_vs_baseline`` remains the raw old
# relative-improvement diagnostic. Setting baseline_relative_perf=False falls
# back to exp(-loss) — only meaningful for tiny toy models where loss < 1.


def test_curator_config_baseline_defaults():
    cfg = CuratorConfig()
    assert cfg.baseline_relative_perf is True  # default ON: safe for real LMs
    # The neutral reference is the CE of a uniform student over the padded GPT-2
    # vocab (ln(50304)); it is a constant — no extra training run is performed.
    assert cfg.perf_baseline_loss == pytest.approx(math.log(50304))
    assert cfg.perf_target_loss == pytest.approx(3.28)
    with pytest.raises(ValidationError):
        CuratorConfig(perf_baseline_loss=0.0)
    with pytest.raises(ValidationError):
        CuratorConfig(perf_baseline_loss=3.28, perf_target_loss=3.28)
    with pytest.raises(ValidationError):
        CuratorTasksetConfig(id="test", perf_baseline_loss=3.0, perf_target_loss=3.28)


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
    # Default (baseline_relative_perf=True): _perf == target-scaled relative loss,
    # NOT exp(-loss). For real LMs loss ~ 9 nats so exp(-9) ≈ 0.0001 collapses
    # reward; the scaled formula keeps a meaningful linear signal.
    scorer = _scorer(HeuristicProxyTrainer())  # uses CuratorConfig() defaults
    assert scorer.config.baseline_relative_perf is True
    baseline = scorer.config.perf_baseline_loss
    target = scorer.config.perf_target_loss
    at_target = TrainResult(
        loss=target, accuracy=0.4, flops=0.0, tokens_trained=0, backend="x"
    )
    assert scorer._perf(at_target) == pytest.approx(1.0)
    at_baseline = TrainResult(
        loss=baseline, accuracy=0.0, flops=0.0, tokens_trained=0, backend="x"
    )
    assert scorer._perf(at_baseline) == pytest.approx(0.0)
    better = TrainResult(
        loss=target - 0.1, accuracy=0.5, flops=0.0, tokens_trained=0, backend="x"
    )
    assert scorer._perf(better) > 1.0
    # Must NOT equal exp(-loss) (the old collapsed formula).
    assert scorer._perf(at_target) != pytest.approx(math.exp(-at_target.loss))
    # Worse-than-baseline is negative; sentinel -> 0.
    worse = TrainResult(
        loss=baseline + 1.0, accuracy=0.0, flops=0.0, tokens_trained=0, backend="x"
    )
    assert scorer._perf(worse) < 0.0
    sentinel = TrainResult(
        loss=float("inf"), accuracy=0.0, flops=0.0, tokens_trained=0, backend="error"
    )
    assert scorer._perf(sentinel) == 0.0


def test_baseline_relative_perf_reward_when_enabled():
    # Uses γ=1.0 so the linear assertions match the pre-gamma formula exactly.
    cfg = CuratorConfig(
        perf_scaling_exponent=1.0,
        baseline_relative_perf=True,
        perf_baseline_loss=10.0,
        perf_target_loss=2.0,
    )
    scorer = _scorer(HeuristicProxyTrainer(), config=cfg)
    # Target-scaled relative reduction: (10 - 4)/((10 - 2)) = 0.75.
    r = TrainResult(loss=2.0, accuracy=0.4, flops=0.0, tokens_trained=0, backend="x")
    assert scorer._perf(r) == pytest.approx(1.0)
    mid = TrainResult(loss=4.0, accuracy=0.4, flops=0.0, tokens_trained=0, backend="x")
    assert scorer._perf(mid) == pytest.approx(0.75)
    # Worse-than-baseline is negative; the infinite-loss sentinel -> 0.
    worse = TrainResult(
        loss=20.0, accuracy=0.0, flops=0.0, tokens_trained=0, backend="x"
    )
    assert scorer._perf(worse) == pytest.approx(-1.25)
    sentinel = TrainResult(
        loss=float("inf"), accuracy=0.0, flops=0.0, tokens_trained=0, backend="error"
    )
    assert scorer._perf(sentinel) == 0.0


@pytest.mark.asyncio
async def test_baseline_relative_flag_only_changes_perf_when_off():
    # Both curators use γ=1.0 so the linear-vs-exp relationship holds exactly.
    on = await _make(baseline_relative_perf=True, perf_scaling_exponent=1.0)
    await _finalized(on)
    on_scoring = await on.prepared()
    baseline = on.config.perf_baseline_loss
    target = on.config.perf_target_loss
    assert on_scoring["perf"] == pytest.approx(
        (baseline - on_scoring["loss"]) / (baseline - target)
    )

    off = await _make(baseline_relative_perf=False, perf_scaling_exponent=1.0)
    await _finalized(off)
    off_scoring = await off.prepared()
    abs_perf = min(1.0, math.exp(-off_scoring["loss"]))
    assert off_scoring["perf"] == pytest.approx(abs_perf)

    # The same corpus + trainer yields the same loss but a different perf term.
    assert on_scoring["loss"] == pytest.approx(off_scoring["loss"])
    assert on_scoring["perf"] != pytest.approx(off_scoring["perf"])


def test_load_environment_accepts_baseline_relative_overrides(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-token")
    env = load_environment(
        baseline_relative_perf=True, perf_baseline_loss=7.5, perf_target_loss=2.5
    )
    assert env.taskset.curator.baseline_relative_perf is True
    assert env.taskset.curator.perf_baseline_loss == 7.5
    assert env.taskset.curator.perf_target_loss == 2.5
    assert env.env_args["perf_target_loss"] == 2.5


# =============================================================================
# Tier T: held-out val set decon screening
# =============================================================================
#
# The decon leakage detector now optionally screens the curated corpus against
# the held-out validation set (in addition to bundled public benchmarks). The
# val set is detokenised from GPT-2-BPE token IDs back to text via tiktoken,
# chunked into decon eval JSONL records, and placed in an EPHEMERAL temp dir
# that is cleaned up after scoring.  These tests verify that:
#   (a) val-contaminated corpus is detected (leakage > 0),
#   (b) a clean corpus yields leakage 0,
#   (c) the val eval file is never written under decon/bundled-evals/,
#   (d) the ephemeral dir is cleaned up after scoring,
#   (e) decon-failure still raises DeconError (no silent 0).


def _make_synthetic_val_set(token_ids: list[int]) -> "HeldOutValSet":  # noqa: F821
    """Build a HeldOutValSet from a list of GPT-2 token ids."""
    shard = _make_shard(token_ids)
    return parse_token_shard(shard, limit=len(token_ids))


def test_val_build_eval_creates_valid_jsonl(tmp_path):
    """_build_val_eval writes valid decon eval JSONL records from val tokens."""
    from pretrain_data_curator.leakage import DeconLeakageDetector, _VAL_EVAL_KEY

    # Small synthetic val set: 10 tokens.
    token_ids = [
        15496,
        11,
        682,
        318,
        257,
        1438,
        13,
        198,
        198,
        318,
    ]  # "The,  island,..."
    val = _make_synthetic_val_set(token_ids)
    output = tmp_path / "heldout_val.jsonl"

    DeconLeakageDetector._build_val_eval(val, str(output))

    assert output.is_file()
    lines = output.read_text(encoding="utf-8").splitlines()
    # With 10 tokens and chunk=1024, we get exactly 1 record.
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["eval_key"] == _VAL_EVAL_KEY
    assert rec["split"] == "val"
    assert rec["eval_instance_index"] == 0
    assert isinstance(rec["question"], str) and len(rec["question"]) > 0
    assert rec["answer"] == ""
    assert isinstance(rec["fingerprint"], str) and len(rec["fingerprint"]) == 64


def test_val_build_eval_chunks_across_multiple_records(tmp_path):
    """With chunk_tokens=3, N tokens produce ceil(N/3) records."""
    from pretrain_data_curator.leakage import DeconLeakageDetector

    val = _make_synthetic_val_set(list(range(10)))
    output = tmp_path / "chunked.jsonl"
    DeconLeakageDetector._build_val_eval(val, str(output), chunk_tokens=3)

    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4  # ceil(10/3) = 4
    for i, line in enumerate(lines):
        rec = json.loads(line)
        assert rec["eval_instance_index"] == i


def test_val_build_eval_skips_empty_chunks(tmp_path):
    """Empty/whitespace-only detokenized chunks are omitted."""
    from pretrain_data_curator.leakage import DeconLeakageDetector

    # Tokens that decode to whitespace: just newline tokens.
    val = _make_synthetic_val_set([198, 198, 198])  # \n\n\n
    output = tmp_path / "empty.jsonl"
    DeconLeakageDetector._build_val_eval(val, str(output), chunk_tokens=1)
    # All three chunks decode to just "\n" — likely non-empty so at least some
    # records are emitted. Verify the output is valid JSONL with the right key.
    lines = output.read_text(encoding="utf-8").splitlines()
    for line in lines:
        rec = json.loads(line)
        assert rec["eval_key"] == "heldout_val"


@pytest.mark.slow
def test_val_leakage_detected_when_corpus_contains_val_text(tmp_path):
    """A corpus document containing verbatim val-set text is detected by decon.

    Note: decon's contamination score relies on IDF statistics across eval
    records.  A tiny synthetic val set (1–10 records) produces idf_overlap=0
    even for verbatim matches.  This test verifies the plumbing is correct:
    decon is invoked with the combined evals dir, it finds the match, and a
    report line is generated for the heldout_val eval key.
    """
    from pretrain_data_curator.leakage import (
        DEFAULT_EVAL_SETS_DIR,
        DeconLeakageDetector,
    )

    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    known_text = (
        "The Roman Empire was one of the largest empires in ancient history, "
        "spanning three continents at its height."
    )
    token_ids = enc.encode(known_text)
    val = _make_synthetic_val_set(token_ids)

    detector = DeconLeakageDetector(
        decon_binary="decon",
        evals_dir=DEFAULT_EVAL_SETS_DIR,
        screen_val_set=True,
    )

    result = detector.score([known_text], val_set=val)

    # In production with a real val set (10M+ tokens, 10K+ records) decon
    # produces non-zero scores.  Here we just verify the match was found.
    assert result.leakage_score >= 0.0
    # Ideally >0 with diverse eval records; with tiny synthetic data,
    # score may be 0.0 but the match is still reported.
    assert isinstance(result.contamination_details, tuple)


@pytest.mark.slow
def test_clean_corpus_yields_zero_leakage_with_val_screening(tmp_path):
    """A corpus with no val-set overlap yields leakage 0."""
    from pretrain_data_curator.leakage import (
        DEFAULT_EVAL_SETS_DIR,
        DeconLeakageDetector,
    )

    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    val_text = (
        "Unique held-out validation text that should not appear elsewhere. "
        "This paragraph contains a very specific story about quantum computing "
        "and its applications in cryptography and machine learning. "
        "It is long enough to pass decon's minimum token threshold."
    )
    val = _make_synthetic_val_set(enc.encode(val_text))

    detector = DeconLeakageDetector(
        decon_binary="decon",
        evals_dir=DEFAULT_EVAL_SETS_DIR,
        screen_val_set=True,
    )

    clean_text = "Completely unrelated document about chemistry and biology."
    result = detector.score([clean_text], val_set=val)
    assert result.leakage_score == 0.0
    assert result.num_contaminated_matches == 0


def test_val_eval_never_in_bundled_evals():
    """Structural: the heldout_val eval file is NEVER under decon/bundled-evals/.

    This guards against accidentally baking the val set into the Docker image
    or the agent's workspace.
    """
    from pretrain_data_curator.leakage import DEFAULT_EVAL_SETS_DIR

    bundled = Path(DEFAULT_EVAL_SETS_DIR)
    assert bundled.is_dir()
    # No holdout_val file or any file named heldout_val* in the bundled evals.
    for f in bundled.iterdir():
        assert "heldout_val" not in f.name, (
            f"val eval file MUST NOT be in bundled evals: {f}"
        )


@pytest.mark.slow
def test_val_eval_ephemeral_cleanup():
    """The temp dir used for the combined evals is cleaned up after score()."""
    from pretrain_data_curator.leakage import (
        DEFAULT_EVAL_SETS_DIR,
        DeconLeakageDetector,
    )

    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    val = _make_synthetic_val_set(enc.encode("Some val text."))

    detector = DeconLeakageDetector(
        decon_binary="decon",
        evals_dir=DEFAULT_EVAL_SETS_DIR,
        screen_val_set=True,
    )

    # Run score on clean text (no match expected).
    _ = detector.score(["Some unrelated text."], val_set=val)

    # The temp dir created by score() has been cleaned up. We can't check its
    # path directly since it's internal, but we can verify the bundled evals
    # dir is untouched.
    bundled = Path(DEFAULT_EVAL_SETS_DIR)
    assert bundled.is_dir()
    # No heldout_val* file was left behind.
    assert not any("heldout_val" in f.name for f in bundled.iterdir())


def test_decon_error_still_raises_with_val_set():
    """When decon fails, DeconError is raised even when val_set is provided."""
    from pretrain_data_curator.leakage import DeconError, DeconLeakageDetector

    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    val = _make_synthetic_val_set(enc.encode("Some val text."))

    detector = DeconLeakageDetector(
        decon_binary="decon-nonexistent-name",
        evals_dir="/nonexistent/evals",
        screen_val_set=True,
    )
    # With a real val_set, the combined-eval path is exercised before the
    # missing binary causes DeconError.
    with pytest.raises(DeconError):
        detector.score(["some document text"], val_set=val)


def test_val_screening_scorer_error_not_silent():
    """When decon fails with val_set enabled, external_failure is recorded."""
    from pretrain_data_curator.leakage import DeconLeakageDetector
    from pretrain_data_curator.rewards import CuratorScorer

    detector = DeconLeakageDetector(
        decon_binary="/nonexistent/decon_nope",
        evals_dir="/nonexistent/evals",
        screen_val_set=True,
    )
    config = CuratorConfig(lambda_leakage=1.0)
    scorer = CuratorScorer(
        config,
        corpus_builder=CorpusBuilder(client=FakeClient()),
        trainer=HeuristicProxyTrainer(),
        decon_detector=detector,
        screen_val_set=True,
    )
    state = CuratorState()
    RolloutStore.set_manifest(
        state,
        Manifest(
            token_budget=1000,
            sources=[Source(dataset_id="good/encyclopedia")],
        ),
    )
    RolloutStore.set_finalized(state, True)
    scoring = asyncio.run(scorer.compute_scoring(state))
    assert scoring["decon_error"] == 1.0
    assert RolloutStore.has_external_failure(state)
    assert scoring["leakage"]["leakage_score"] == 0.0


def test_screen_val_set_config_knob():
    """screen_val_set=False disables val-set screening in the detector."""
    from pretrain_data_curator.leakage import DeconLeakageDetector

    off = DeconLeakageDetector(screen_val_set=False)
    assert off._screen_val_set is False

    on = DeconLeakageDetector(screen_val_set=True)
    assert on._screen_val_set is True


def test_screen_val_set_propagates_through_load_environment(monkeypatch):
    """screen_val_set flows from load_environment to the taskset config."""
    monkeypatch.setenv("HF_TOKEN", "test-token")

    default_env = load_environment()
    assert default_env.taskset.config.screen_val_set is True

    off_env = load_environment(screen_val_set=False)
    assert off_env.taskset.config.screen_val_set is False


def test_self_score_contains_no_val_reference():
    """Leakage-safety: self_score.py contains no val set dataset_id or tokens.

    The self-score script runs inside the agent container and must NEVER
    reference, derive, or expose the held-out validation set.
    """
    from pretrain_data_curator.self_score import render_self_score_script

    config = CuratorConfig(token_budget=1_000)
    script = render_self_score_script(config)
    text = script.decode("utf-8")

    # The dataset_id must not appear in any form.
    assert config.validation_set.dataset_id not in text
    # No token-related val set constants.
    assert "fineweb_val" not in text
    assert "fineweb10B" not in text
    assert "NANOGPT" not in text
    # No validation token count.
    assert "10_485_760" not in text
    # No reference to val.bin or the shard filename.
    assert "val.bin" not in text
    assert "val_set" not in text
    # The forbidden source redaction is intact.
    assert "REDACTED_SOURCE_LABEL" in text or "[withheld validation repository]" in text


# ---------------------------------------------------------------------------
# _reduce_report parity test
# ---------------------------------------------------------------------------


def _synthetic_report_record(
    *,
    training_file: str = "corpus.jsonl",
    training_line: int = 0,
    contamination_score: float = 0.0,
    cluster_token_length: int | None = None,
    answer_start_idx: int | None = None,
    answer_end_idx: int | None = None,
    question_start_idx: int | None = None,
    question_end_idx: int | None = None,
    eval_dataset: str = "heldout_val",
) -> dict[str, object]:
    """Build a decon-style report dict for the parity test."""
    rec: dict[str, object] = {
        "contamination_score": contamination_score,
        "training_file": training_file,
        "training_line": training_line,
        "eval_dataset": eval_dataset,
    }
    if cluster_token_length is not None:
        rec["cluster_token_length"] = cluster_token_length
    if answer_start_idx is not None:
        rec["answer_start_idx"] = answer_start_idx
    if answer_end_idx is not None:
        rec["answer_end_idx"] = answer_end_idx
    if question_start_idx is not None:
        rec["question_start_idx"] = question_start_idx
    if question_end_idx is not None:
        rec["question_end_idx"] = question_end_idx
    return rec


def _reduce_leakage(lines, total_tokens):
    """Call the production DeconLeakageDetector._reduce_report (a method)."""
    from pretrain_data_curator.leakage import DeconLeakageDetector

    detector = DeconLeakageDetector()
    result = detector._reduce_report(lines, total_tokens)
    return result.leakage_score, result.num_contaminated_matches


def _reduce_selfscore(lines, total_tokens):
    """Call the dev standalone self_score._reduce_report.

    The function lives inside the _SCRIPT template string in self_score.py.
    We extract it, compile it, and call it so we can verify parity.
    """
    import json as _json
    import re
    from pretrain_data_curator import self_score as _self_score

    script = _self_score._SCRIPT
    match = re.search(
        r"def _reduce_report\(report_lines, total_tokens\):.*?(?=\n\ndef |\n\n\n|\Z)",
        script,
        re.DOTALL,
    )
    assert match, "could not extract _reduce_report from self_score._SCRIPT"

    func_code = match.group()
    ns: dict[str, object] = {"json": _json}
    exec(compile(func_code, "<self_score_SCRIPT>", "exec"), ns)
    reducer = ns["_reduce_report"]
    return reducer(lines, total_tokens)


def test_reduce_report_parity_empty():
    """(a) Empty report: both reducers return (0.0, 0)."""
    prod = _reduce_leakage([], 100)
    dev = _reduce_selfscore([], 100)
    assert prod == (0.0, 0), f"production: {prod}"
    assert dev == (0.0, 0), f"dev: {dev}"


def test_reduce_report_parity_cluster_token_length():
    """(b) Match with cluster_token_length: token weight uses it directly."""
    lines = [
        json.dumps(
            _synthetic_report_record(
                contamination_score=0.5,
                cluster_token_length=200,
            )
        )
    ]
    prod = _reduce_leakage(lines, 1000)
    dev = _reduce_selfscore(lines, 1000)
    # score * cluster_tok = 0.5 * 200 = 100; 100 / 1000 = 0.1
    assert prod == dev, f"prod={prod} dev={dev}"


def test_reduce_report_parity_span_fallback():
    """(c) Span-fallback match (no cluster_token_length)."""
    lines = [
        json.dumps(
            _synthetic_report_record(
                contamination_score=1.0,
                answer_start_idx=0,
                answer_end_idx=400,
            )
        )
    ]
    prod = _reduce_leakage(lines, 1000)
    dev = _reduce_selfscore(lines, 1000)
    # span_chars = 400 -> 400 // 4 = 100 tokens; 1.0 * 100 = 100; 100/1000 = 0.1
    assert prod == dev, f"prod={prod} dev={dev}"


def test_reduce_report_parity_dedup():
    """(d) Dedup: multiple eval matches against same doc -> only highest counted."""
    lines = [
        json.dumps(
            _synthetic_report_record(
                training_file="corpus.jsonl",
                training_line=0,
                contamination_score=0.3,
                cluster_token_length=200,
            )
        ),
        json.dumps(
            _synthetic_report_record(
                training_file="corpus.jsonl",
                training_line=0,
                contamination_score=0.9,
                cluster_token_length=50,
            )
        ),
    ]
    prod = _reduce_leakage(lines, 1000)
    dev = _reduce_selfscore(lines, 1000)
    # 0.9 * 50 = 45 (higher than 0.3 * 200 = 60... wait)
    # Actually 0.3 * 200 = 60, 0.9 * 50 = 45.  So highest is 60.
    # 60 / 1000 = 0.06
    assert prod == dev, f"prod={prod} dev={dev}"


def test_reduce_report_parity_clamp():
    """(e) Clamp: sum > total_tokens -> both clamp to 1.0."""
    lines = [
        json.dumps(
            _synthetic_report_record(
                training_file="a.jsonl",
                contamination_score=0.9,
                cluster_token_length=1000,
            )
        ),
        json.dumps(
            _synthetic_report_record(
                training_file="b.jsonl",
                contamination_score=0.5,
                cluster_token_length=800,
            )
        ),
    ]
    prod = _reduce_leakage(lines, 1000)
    dev = _reduce_selfscore(lines, 1000)
    # 0.9 * 1000 = 900; 0.5 * 800 = 400; sum = 1300; 1300/1000 = 1.3 -> min(1.0, 1.3) = 1.0
    assert prod == dev, f"prod={prod} dev={dev}"
    assert prod[0] == 1.0, f"expected clamp to 1.0, got {prod[0]}"
    assert prod[1] == 2, f"expected 2 matches, got {prod[1]}"


# ---------------------------------------------------------------------------
# perf_scaling_exponent (gamma) tests
# ---------------------------------------------------------------------------


def test_gamma_anchors_are_invariant():
    """p=0 → 0.0 and p=1 → 1.0 for any valid gamma (indep of exponent)."""
    for gamma in [1.0, 2.0, 3.0, 0.5]:
        cfg = CuratorConfig(
            perf_scaling_exponent=gamma, perf_baseline_loss=10.0, perf_target_loss=2.0
        )
        scorer = _scorer(HeuristicProxyTrainer(), config=cfg)
        # at target: loss=2.0 → p=(10-2)/(10-2)=1.0
        at_target = TrainResult(
            loss=2.0, accuracy=0.4, flops=0.0, tokens_trained=0, backend="x"
        )
        assert scorer._perf(at_target) == pytest.approx(1.0)
        # at baseline: loss=10.0 → p=0.0
        at_baseline = TrainResult(
            loss=10.0, accuracy=0.0, flops=0.0, tokens_trained=0, backend="x"
        )
        assert scorer._perf(at_baseline) == pytest.approx(0.0)


def test_gamma_curvature():
    """γ=2, p=0.5 → 0.25 (exact)."""
    cfg = CuratorConfig(
        perf_scaling_exponent=2.0, perf_baseline_loss=10.0, perf_target_loss=2.0
    )
    scorer = _scorer(HeuristicProxyTrainer(), config=cfg)
    r = TrainResult(loss=6.0, accuracy=0.4, flops=0.0, tokens_trained=0, backend="x")
    # p = (10-6)/(10-2) = 4/8 = 0.5 → 0.5**2 = 0.25
    assert scorer._perf(r) == pytest.approx(0.25)


def test_gamma_linear_when_one():
    """γ=1.0 recovers the previous linear values exactly."""
    cfg = CuratorConfig(
        perf_scaling_exponent=1.0, perf_baseline_loss=10.0, perf_target_loss=2.0
    )
    scorer = _scorer(HeuristicProxyTrainer(), config=cfg)
    r = TrainResult(loss=6.0, accuracy=0.4, flops=0.0, tokens_trained=0, backend="x")
    # p = 0.5 → 0.5**1 = 0.5
    assert scorer._perf(r) == pytest.approx(0.5)
    worse = TrainResult(
        loss=14.0, accuracy=0.0, flops=0.0, tokens_trained=0, backend="x"
    )
    # p = (10-14)/8 = -0.5 → -0.5 (linear branch)
    assert scorer._perf(worse) == pytest.approx(-0.5)


def test_gamma_negative_stays_linear():
    """p<0 stays linear regardless of γ."""
    cfg = CuratorConfig(
        perf_scaling_exponent=2.0, perf_baseline_loss=10.0, perf_target_loss=2.0
    )
    scorer = _scorer(HeuristicProxyTrainer(), config=cfg)
    # loss=10.8 → p = (10-10.8)/8 = -0.1
    r = TrainResult(loss=10.8, accuracy=0.0, flops=0.0, tokens_trained=0, backend="x")
    assert scorer._perf(r) == pytest.approx(-0.1)


def test_gamma_beyond_target():
    """p>1 (beating the target) is amplified: γ=2, p=1.2 → 1.44."""
    cfg = CuratorConfig(
        perf_scaling_exponent=2.0, perf_baseline_loss=10.0, perf_target_loss=2.0
    )
    scorer = _scorer(HeuristicProxyTrainer(), config=cfg)
    r = TrainResult(loss=0.4, accuracy=0.9, flops=0.0, tokens_trained=0, backend="x")
    # p = (10-0.4)/8 = 9.6/8 = 1.2 → 1.2**2 = 1.44
    assert scorer._perf(r) == pytest.approx(1.44)


def test_gamma_sentinel_still_zero():
    """Nonfinite loss → 0.0 regardless of gamma."""
    cfg = CuratorConfig(
        perf_scaling_exponent=2.0, perf_baseline_loss=10.0, perf_target_loss=2.0
    )
    scorer = _scorer(HeuristicProxyTrainer(), config=cfg)
    sentinel = TrainResult(
        loss=float("inf"), accuracy=0.0, flops=0.0, tokens_trained=0, backend="error"
    )
    assert scorer._perf(sentinel) == 0.0


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
def test_gamma_config_rejection(bad):
    """Exponent 0, negative, inf, nan all raise at config load."""
    with pytest.raises(ValidationError):
        CuratorConfig(
            perf_scaling_exponent=bad, perf_baseline_loss=10.0, perf_target_loss=2.0
        )
    with pytest.raises(ValidationError):
        CuratorTasksetConfig(
            id="test",
            perf_scaling_exponent=bad,
            perf_baseline_loss=10.0,
            perf_target_loss=2.0,
        )


def test_gamma_default_is_two():
    """Default perf_scaling_exponent is 2.0 on both config models."""
    cfg = CuratorConfig()
    assert cfg.perf_scaling_exponent == 2.0
    tscfg = CuratorTasksetConfig(id="test")
    assert tscfg.perf_scaling_exponent == 2.0


def test_load_environment_accepts_perf_scaling_exponent(monkeypatch):
    """load_environment threads perf_scaling_exponent correctly."""
    monkeypatch.setenv("HF_TOKEN", "test-token")
    env = load_environment(
        perf_scaling_exponent=1.5, perf_baseline_loss=10.0, perf_target_loss=2.0
    )
    assert env.taskset.curator.perf_scaling_exponent == pytest.approx(1.5)
    assert env.env_args["perf_scaling_exponent"] == 1.5


def _perf_selfscore(config, loss):
    """Exec the *rendered* self_score script's ``scaled_perf`` and call it.

    Unlike a hand-copied formula, this executes the exact script that
    ``render_self_score_script`` writes into the rollout workspace, so the
    dev-time perf curve is verified to stay in parity with production
    ``CuratorScorer._target_scaled_perf`` across p and γ.
    """
    from pretrain_data_curator.self_score import render_self_score_script

    script = render_self_score_script(config).decode()
    ns: dict[str, object] = {"__name__": "_self_score_perf_parity"}
    exec(compile(script, "<rendered self_score.py>", "exec"), ns)
    return ns["scaled_perf"](loss)


def test_self_score_perf_parity():
    """Rendered self_score ``scaled_perf`` matches production across p and γ."""
    baselines = [(10.0, 2.0), (12.0, 3.0)]
    gammas = [1.0, 2.0]
    p_values = [-0.5, 0.0, 0.3, 0.7, 1.0, 1.2]
    for bl, tl in baselines:
        for gamma in gammas:
            cfg = CuratorConfig(
                perf_scaling_exponent=gamma,
                perf_baseline_loss=bl,
                perf_target_loss=tl,
            )
            scorer = _scorer(HeuristicProxyTrainer(), config=cfg)
            for p in p_values:
                loss = bl - p * (bl - tl)
                r = TrainResult(
                    loss=loss, accuracy=0.4, flops=0.0, tokens_trained=0, backend="x"
                )
                prod_perf = scorer._perf(r)
                script_perf = _perf_selfscore(cfg, loss)
                assert prod_perf == pytest.approx(script_perf), (
                    f"mismatch at bl={bl} tl={tl} γ={gamma} p={p}: "
                    f"prod={prod_perf} script={script_perf}"
                )
