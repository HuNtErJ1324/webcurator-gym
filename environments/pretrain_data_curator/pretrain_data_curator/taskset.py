"""The curation taskset: tasks, manifest parsing, finalize, rewards, metrics.

``CuratorTaskset`` drives a *toolless* curation rollout: the agent runs the
Hugging Face ``hf`` CLI in its own shell to search/inspect datasets and decides a
curation mixture, then writes that decision to a JSON file in the runtime
workspace. Because the taskset exposes **no** tool servers (it does not
override ``Taskset.tools``), it satisfies the non-MCP harness gate at
``verifiers/v1/env.py:239-247`` and runs under codex / kimi_code as well as the
default / bash harnesses.

After generation, ``finalize`` (run before scoring, while the runtime is live)
reads and parses the manifest JSON from the runtime workspace. Production
benchmarks require a valid non-empty ``/workspace/manifest.json``: missing or
invalid files leave ``finalized=0``, skip materialize/train, and record
``manifest_provenance`` (``missing`` vs ``invalid_workspace_file``, also in
``trace.info``). Assistant-message / opt-in trace candidates remain telemetry
only under ``manifest_candidate`` and never override those workspace outcomes.

Scoring is unchanged from v1: the finalized manifest's datasets are materialized
and used to train the fixed proxy student, and the composite reward is:

    R(M, H) = alpha_perf*Perf_scaled_to_target - lambda_leakage*Leakage

Leakage is a token-weighted scalar from the decon Rust n-gram
detector run against PUBLIC BENCHMARK eval sets (bundled under
``decon/bundled-evals/``) AND, optionally, the held-out validation set (when
``screen_val_set`` is enabled — the default).  The val eval file is detokenised
ephemerally at scoring time and never persists.

The reward coefficients are runtime config, so each ``@vf.reward`` is registered
with the framework weight ``1.0`` and folds its (signed) coefficient into the
returned value. The heavy materialize + train + leakage pass runs exactly once per
rollout, guarded by a per-trace double-checked lock + cache (``_prepared``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
from importlib import resources
from pathlib import PurePosixPath
from typing import Any, Literal

import verifiers.v1 as vf
from pydantic import ValidationError, field_validator, model_validator

from .corpus import EST_TOKENS_PER_DOC, CorpusBuilder
from .leakage import DeconLeakageDetector
from .hf_access import HuggingFaceDatasetClient, RetryPolicy
from .hf_cli_parse import content_text, extract_hf_commands
from .leakage import DEFAULT_DECON_BINARY, DEFAULT_EVAL_SETS_DIR
from .models import (
    CuratorConfig,
    FilterSpec,
    MANIFEST_CANDIDATE_ASSISTANT_MESSAGE,
    MANIFEST_CANDIDATE_TRACE_FALLBACK,
    MANIFEST_FILENAME,
    MANIFEST_PROVENANCE_INVALID_WORKSPACE_FILE,
    MANIFEST_PROVENANCE_MISSING,
    MANIFEST_PROVENANCE_WORKSPACE_FILE,
    Manifest,
    ManifestCandidate,
    ManifestProvenance,
    ProxyStudentConfig,
    Sampling,
    Source,
)
from .rewards import CuratorScorer
from .rollout_state import CuratorState, RolloutStore
from .runtime_config import derive_task_runtime_updates
from .self_score import (
    SELF_SCORE_FILENAME,
    SELF_SCORE_TRAIN_FILENAME,
    render_self_score_script,
    render_self_score_train_script,
)
from .tasks import CuratorTask, build_tasks
from .trainer import (
    HeuristicProxyTrainer,
    ProxyStudentTrainer,
    RuntimeSelectedTrainer,
    TrainerError,
)
from .val_set import ValidationSetConfig, ValTokenLoader

logger = logging.getLogger(__name__)
TRAINER_ERROR_STR_LIMIT = 20_000
# Canonical HF CLI skill (byte-for-byte) from huggingface/skills; docs:
# https://huggingface.co/docs/hub/agents-cli and
# https://github.com/huggingface/skills/tree/main/skills/hf-cli
# Delivered at the project skill path agents expect from harness install.
HF_CLI_SKILL_UPSTREAM_REVISION = "7039bdcf4510c30ec932637e8b2c1646aee7f185"
HF_CLI_SKILL_UPSTREAM_PATH = "skills/hf-cli/SKILL.md"
HF_CLI_SKILL_SHA256 = "a6b3fcf3bd0a6164aeda357f483295638cbaee54f56f0cec13462e647920ec37"
HF_CLI_SKILL_RESOURCE = "skills/hf-cli/SKILL.md"
HF_CLI_SKILL_RUNTIME_PATH = ".agents/skills/hf-cli/SKILL.md"
# Backward-compatible alias used by older tests/imports.
HF_CLI_SKILL_FILENAME = HF_CLI_SKILL_RUNTIME_PATH


def hf_cli_skill_package_file():
    """Locate the vendored skill via single-arg Traversable.joinpath (Py3.11+)."""
    node = resources.files("pretrain_data_curator")
    for part in HF_CLI_SKILL_RESOURCE.split("/"):
        node = node.joinpath(part)
    return node


_HF_CLI_SKILL = hf_cli_skill_package_file().read_bytes()

# --------------------------------------------------------------------------- #
# manifest parsing (workspace file primary; assistant messages remain fallback)
# --------------------------------------------------------------------------- #

_SUPPORTED_FILTER_KINDS = {
    "min_chars",
    "max_chars",
    "min_tokens",
    "max_symbol_ratio",
    "min_alpha_ratio",
    "drop_regex",
    "keep_regex",
    "dedup_exact",
}
_FENCE_RE = re.compile(r"```(?:json|jsonc|JSON)?\s*(.*?)```", re.DOTALL)

# Owner/name with optional trailing /config. Name body is alphanumeric chunks
# joined by ._- so three-segment prose (dataset/config) cannot backtrack into
# truncated two-segment fragments (e.g. OpenHermes-2.5/default → OpenHermes-2 +
# 5/default), and sentence-trailing ``building/starting.`` does not match.
_HF_ID_RE = re.compile(
    r"(?<![:/\w])"
    r"([A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*)"
    r"(?:/[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*)?"
    r"(?![A-Za-z0-9_./-])"
)

_NOT_HF_NAMESPACES = frozenset(
    {
        "http",
        "https",
        "hf",
        "file",
        "s3",
        "gs",
        "az",
        "usr",
        "var",
        "etc",
        "bin",
        "tmp",
        "opt",
        "home",
        "root",
        "datasets",
        "models",
        "spaces",
        "api",
        "v1",
        "v2",
    }
)
_NOT_HF_NAMES = frozenset({"train", "test", "validation", "valid", "dev", "split"})
_FILE_EXTS = frozenset(
    {
        "py",
        "json",
        "jsonl",
        "txt",
        "csv",
        "yaml",
        "yml",
        "toml",
        "sh",
        "md",
        "rst",
        "log",
        "parquet",
        "arrow",
        "gz",
        "zip",
    }
)


def _shell_command_from_tool_args(arguments: Any) -> str:
    """Extract a shell command string from tool-call arguments.

    Codex-style harnesses emit ``{"cmd": "..."}`` while bash/default harnesses
    use ``{"command": "..."}``. Prefer ``command`` when both are present.
    """
    if arguments is None:
        return ""
    raw = arguments if isinstance(arguments, str) else str(arguments)
    try:
        payload = json.loads(raw or "{}")
    except (json.JSONDecodeError, TypeError):
        return raw
    if isinstance(payload, dict):
        for key in ("command", "cmd"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return raw


def _looks_like_hf_id(s: str) -> bool:
    if "/" not in s:
        return False
    ns, name = s.split("/", 1)
    if not ns or not name:
        return False
    if ns.lower() in _NOT_HF_NAMESPACES or name.lower() in _NOT_HF_NAMES:
        return False
    if "." in name and name.rsplit(".", 1)[-1].lower() in _FILE_EXTS:
        return False
    return True


def _ids_from_trace(trace: vf.Trace) -> list[str]:
    inspected: dict[str, None] = {}
    for msg in trace.assistant_messages:
        for tc in msg.tool_calls or []:
            cmd = _shell_command_from_tool_args(getattr(tc, "arguments", "") or "")
            for argv in extract_hf_commands(cmd):
                if (
                    len(argv) >= 3
                    and argv[0] == "datasets"
                    and argv[1] == "info"
                    and "/" in argv[2]
                    and not argv[2].startswith("-")
                ):
                    inspected[argv[2]] = None
    if inspected:
        return list(inspected)
    observed: dict[str, None] = {}
    for msg in getattr(trace, "tool_messages", []):
        text = content_text(getattr(msg, "content", ""))
        for m in _HF_ID_RE.finditer(text):
            did = m.group(1)
            if _looks_like_hf_id(did):
                observed.setdefault(did, None)
    return list(observed)


def _iter_json_objects(s: str):
    depth = 0
    start: int | None = None
    in_str = False
    esc = False
    for i, ch in enumerate(s):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0 and start is not None:
                yield s[start : i + 1]
                start = None


def _last_json_object(
    s: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    last_with_sources: dict[str, Any] | None = None
    last_any: dict[str, Any] | None = None
    for chunk in _iter_json_objects(s):
        try:
            obj = json.loads(chunk)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        last_any = obj
        srcs = obj.get("sources")
        if isinstance(srcs, list) and srcs:
            last_with_sources = obj
    return last_with_sources, last_any


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract a manifest JSON object from compatibility assistant text.

    Scans every fenced ```json/``` block (and, as a final fallback candidate, the
    whole text) and prefers the LAST candidate that yields a JSON object carrying a
    non-empty ``sources`` list — so a final manifest wins over an earlier draft, and
    a real manifest wins over a leading note/plan block. When no candidate carries
    ``sources``, falls back to the last parseable JSON object (single-block
    behavior). Tolerant of `final answer:`-style prose around the JSON and of a bare
    (unfenced) object; returns ``None`` for empty / truncated / non-JSON text.
    """
    if not text:
        return None
    candidates = [m.group(1) for m in _FENCE_RE.finditer(text)]
    candidates.append(text)
    last_with_sources: dict[str, Any] | None = None
    last_any: dict[str, Any] | None = None
    for cand in candidates:
        with_sources, any_obj = _last_json_object(cand)
        if with_sources is not None:
            last_with_sources = with_sources
        if any_obj is not None:
            last_any = any_obj
    return last_with_sources if last_with_sources is not None else last_any


def _coerce_filters(raw: Any) -> list[FilterSpec]:
    if not isinstance(raw, list):
        return []
    specs: list[FilterSpec] = []
    for f in raw:
        if not isinstance(f, dict):
            continue
        kind = f.get("kind")
        if not isinstance(kind, str) or kind not in _SUPPORTED_FILTER_KINDS:
            continue
        raw_params = f.get("params")
        params = dict(raw_params) if isinstance(raw_params, dict) else {}
        try:
            if kind in {"min_chars", "max_chars", "min_tokens"} and "value" in params:
                params["value"] = int(params["value"])
            elif kind in {"max_symbol_ratio", "min_alpha_ratio"} and "value" in params:
                value = float(params["value"])
                if not math.isfinite(value):
                    continue
                params["value"] = value
            elif kind in {"drop_regex", "keep_regex"}:
                pattern = str(params.get("pattern", ""))
                re.compile(pattern)
                params["pattern"] = pattern
        except (TypeError, ValueError, OverflowError, re.error):
            continue
        specs.append(FilterSpec(kind=kind, params=params))
    return specs


def _coerce_source(raw: Any) -> Source | None:
    if isinstance(raw, str):
        return Source(dataset_id=raw.strip()) if raw.strip() else None
    if not isinstance(raw, dict):
        return None
    local_path = raw.get("local_path")
    is_local = raw.get("kind") == "local" or local_path is not None
    dataset_id = (
        raw.get("dataset_id")
        or raw.get("id")
        or raw.get("dataset")
        or raw.get("repo_id")
        or raw.get("name")
    )
    if (
        (not isinstance(dataset_id, str) or not dataset_id.strip())
        and is_local
        and isinstance(local_path, str)
        and local_path.strip()
    ):
        dataset_id = f"local:{local_path.strip()}"
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        return None
    kwargs: dict[str, Any] = {"dataset_id": dataset_id.strip()}
    if is_local:
        kwargs["kind"] = "local"
        kwargs["local_path"] = local_path
        if raw.get("local_format") is not None:
            kwargs["local_format"] = raw["local_format"]
    if raw.get("config"):
        kwargs["config"] = str(raw["config"])
    if raw.get("split"):
        kwargs["split"] = str(raw["split"])
    if raw.get("text_field"):
        kwargs["text_field"] = str(raw["text_field"])
    if "weight" in raw:
        try:
            kwargs["weight"] = max(0.0, float(raw["weight"]))
        except (TypeError, ValueError):
            pass
    kwargs["filters"] = _coerce_filters(raw.get("filters"))
    sampling = raw.get("sampling") if isinstance(raw.get("sampling"), dict) else {}
    max_docs = raw.get("max_docs", sampling.get("max_docs"))
    max_tokens = raw.get("max_tokens", sampling.get("max_tokens"))

    def _pos_int(v: Any) -> int | None:
        try:
            n = int(v)
            return n if n >= 1 else None
        except (TypeError, ValueError):
            return None

    kwargs["sampling"] = Sampling(
        max_docs=_pos_int(max_docs), max_tokens=_pos_int(max_tokens)
    )
    try:
        return Source(**kwargs)
    except ValidationError:
        return None


def parse_manifest(
    text: str,
    default_token_budget: int | None = None,
    reserved_local_filename: str | None = None,
) -> Manifest | None:
    """Parse and validate manifest text.

    Returns a :class:`Manifest` with at least one source, or ``None`` when the
    text has no usable manifest (empty / truncated / no valid sources) — the
    caller treats ``None`` as "not finalized" → graceful zero score.
    """
    data = extract_json_object(text)
    if not isinstance(data, dict):
        return None
    raw_sources = data.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        return None
    sources = [s for s in (_coerce_source(r) for r in raw_sources) if s is not None]
    if reserved_local_filename is not None:
        sources = [
            source
            for source in sources
            if not (
                source.kind == "local"
                and source.local_path is not None
                and PurePosixPath(source.local_path).name == reserved_local_filename
            )
        ]
    if not sources:
        return None
    token_budget = default_token_budget or 1_000_000
    if data.get("token_budget") is not None:
        try:
            token_budget = int(data["token_budget"])
        except (TypeError, ValueError, OverflowError):
            pass
    sample_docs_per_source: int | None = None
    if data.get("sample_docs_per_source") is not None:
        try:
            sample_docs_per_source = int(data["sample_docs_per_source"])
        except (TypeError, ValueError, OverflowError):
            sample_docs_per_source = None
    try:
        return Manifest(
            token_budget=token_budget,
            sources=sources,
            sample_docs_per_source=sample_docs_per_source,
        )
    except ValidationError:
        return None


class CuratorTasksetConfig(vf.TasksetConfig):
    """Validated configuration for the curation taskset."""

    cutoff_date: str = "2024-12-31"
    token_budget: int = 1_000_000
    hf_token_env: str = "HF_TOKEN"
    manifest_filename: str = MANIFEST_FILENAME
    candidate_limit: int = 8
    allow_trace_id_manifest_fallback: bool = False
    allow_local_sources: bool = True
    max_local_source_bytes: int = 33_554_432
    max_turns: int = 64
    alpha_perf: float = 1.0
    lambda_leakage: float = 1.0
    perf_baseline_loss: float = math.log(50304)
    perf_target_loss: float = 3.28
    perf_scaling_exponent: float = 2.0
    baseline_relative_perf: bool = True
    max_concurrent_fetches: int = 8
    max_concurrent_training: int = 1
    fetch_timeout_seconds: float = 30.0
    fetch_timeout_per_doc_seconds: float = 0.25
    fetch_max_attempts: int = 3
    use_real_trainer: bool = False
    proxy_student: dict[str, Any] = {}
    validation_set: dict[str, Any] = {}
    # Decon contamination detection knobs.
    decon_binary: str = DEFAULT_DECON_BINARY
    decon_evals_dir: str | None = None
    decon_threshold: float = 0.2
    screen_val_set: bool = True
    # Cap on tool / bash output captured from runtimes; <=0 disables truncation.
    max_tool_output_chars: int = 20_000

    @field_validator("manifest_filename")
    @classmethod
    def _manifest_filename_is_workspace_root_file(cls, value: str) -> str:
        if not value or value in {".", ".."} or "/" in value or "\\" in value:
            raise ValueError("manifest_filename must be a filename in /workspace")
        return value

    @model_validator(mode="after")
    def _check_perf_target_below_baseline(self) -> "CuratorTasksetConfig":
        if self.perf_baseline_loss <= self.perf_target_loss:
            raise ValueError(
                "perf_baseline_loss must be greater than perf_target_loss "
                f"(got baseline={self.perf_baseline_loss}, "
                f"target={self.perf_target_loss})"
            )
        return self

    @model_validator(mode="after")
    def _check_perf_scaling_exponent(self) -> "CuratorTasksetConfig":
        exp = self.perf_scaling_exponent
        if not math.isfinite(exp) or exp <= 0:
            # fmt: off
            raise ValueError(
                "perf_scaling_exponent must be finite and > 0 "
                f"(got {exp})"
            )
            # fmt: on
        return self


try:
    _TasksetBase = vf.Taskset[CuratorTask, CuratorTasksetConfig, CuratorState]
except TypeError:
    _TasksetBase = vf.Taskset


class CuratorTaskset(_TasksetBase):
    def __init__(self, config: CuratorTasksetConfig) -> None:
        super().__init__(config)
        self.curator = self._build_curator_config(config)
        self._client: HuggingFaceDatasetClient | None = None
        self._corpus_builder: CorpusBuilder | None = None
        self._trainer: ProxyStudentTrainer | None = None
        self._decon_detector: DeconLeakageDetector | None = None
        self._val_loader: ValTokenLoader | None = None
        self._scorer: CuratorScorer | None = None
        self._scoring_cache: dict[str, dict[str, Any]] = {}
        self._scoring_locks: dict[str, asyncio.Lock] = {}

    @staticmethod
    def _build_curator_config(config: CuratorTasksetConfig) -> CuratorConfig:
        return CuratorConfig(
            cutoff_date=config.cutoff_date,
            token_budget=config.token_budget,
            candidate_limit=config.candidate_limit,
            allow_trace_id_manifest_fallback=config.allow_trace_id_manifest_fallback,
            allow_local_sources=config.allow_local_sources,
            max_local_source_bytes=config.max_local_source_bytes,
            max_turns=config.max_turns,
            alpha_perf=config.alpha_perf,
            lambda_leakage=config.lambda_leakage,
            perf_baseline_loss=config.perf_baseline_loss,
            perf_target_loss=config.perf_target_loss,
            perf_scaling_exponent=config.perf_scaling_exponent,
            baseline_relative_perf=config.baseline_relative_perf,
            max_concurrent_fetches=config.max_concurrent_fetches,
            max_concurrent_training=config.max_concurrent_training,
            fetch_timeout_seconds=config.fetch_timeout_seconds,
            fetch_timeout_per_doc_seconds=config.fetch_timeout_per_doc_seconds,
            fetch_max_attempts=config.fetch_max_attempts,
            use_real_trainer=config.use_real_trainer,
            proxy_student=ProxyStudentConfig(**(config.proxy_student or {})),
            validation_set=ValidationSetConfig(**(config.validation_set or {})),
            max_tool_output_chars=config.max_tool_output_chars,
        )

    # -- collaborators ---------------------------------------------------------

    def _fetch_policy(self) -> RetryPolicy:
        return RetryPolicy(
            attempts=self.curator.fetch_max_attempts,
            timeout=self.curator.fetch_timeout_seconds,
            per_doc_seconds=self.curator.fetch_timeout_per_doc_seconds,
        )

    def _ensure(self) -> CuratorScorer:
        if self._scorer is not None:
            return self._scorer
        if self._client is None:
            self._client = HuggingFaceDatasetClient(
                token_env=self.config.hf_token_env,
            )
        if self._corpus_builder is None:
            self._corpus_builder = CorpusBuilder(
                client=self._client,
                retry_policy=self._fetch_policy(),
                fetch_limit=self.curator.max_concurrent_fetches,
                allow_local_sources=self.curator.allow_local_sources,
                max_local_source_bytes=self.curator.max_local_source_bytes,
            )
        if self._val_loader is None:
            self._val_loader = ValTokenLoader(
                self.curator.validation_set,
                retry_policy=self._fetch_policy(),
                fetch_limit=self.curator.max_concurrent_fetches,
            )
        if self._decon_detector is None:
            evals_dir = self.config.decon_evals_dir or DEFAULT_EVAL_SETS_DIR
            self._decon_detector = DeconLeakageDetector(
                decon_binary=self.config.decon_binary,
                evals_dir=evals_dir,
                threshold=self.config.decon_threshold,
                screen_val_set=self.config.screen_val_set,
            )
        if self._trainer is None:
            self._trainer = (
                self._build_real_trainer()
                if self.curator.use_real_trainer
                else HeuristicProxyTrainer()
            )
        self._scorer = CuratorScorer(
            self.curator,
            self._corpus_builder,
            self._trainer,
            self._decon_detector,
            val_loader=self._val_loader,
            screen_val_set=self.config.screen_val_set,
        )
        return self._scorer

    def _build_real_trainer(self) -> ProxyStudentTrainer:
        from .docker_backend import HarnessRuntimeProxyTrainer
        from .modal_backend import ModalProxyTrainer

        return RuntimeSelectedTrainer(
            {
                "docker": HarnessRuntimeProxyTrainer(
                    concurrency_limit=self.curator.max_concurrent_training,
                    val_loader=self._val_loader,
                ),
                "modal": ModalProxyTrainer(
                    concurrency_limit=self.curator.max_concurrent_training,
                    val_loader=self._val_loader,
                ),
            }
        )

    # -- taskset surface -------------------------------------------------------

    def load_tasks(self) -> list[CuratorTask]:
        tasks = build_tasks(
            self.curator.cutoff_date,
            self.curator.token_budget,
            manifest_filename=self.config.manifest_filename,
            allow_local_sources=self.curator.allow_local_sources,
            alpha_perf=self.curator.alpha_perf,
            lambda_leakage=self.curator.lambda_leakage,
            perf_target_loss=self.curator.perf_target_loss,
        )
        updates = derive_task_runtime_updates(
            self.curator.proxy_student,
            use_real_trainer=self.curator.use_real_trainer,
        )
        return [task.model_copy(update=updates) for task in tasks]

    async def setup(self, task: CuratorTask, runtime: vf.Runtime) -> None:
        token_env = self.config.hf_token_env
        if not os.environ.get(token_env):
            raise RuntimeError(
                f"Hugging Face token environment variable {token_env!r} is required "
                "before starting a rollout. Export it into the env-server process "
                "(for example, `set -a; source secrets.env; set +a`). Running "
                "`source secrets.env` without `export` or `set -a` does not "
                "propagate the token to the env-server."
            )
        if self.curator.use_real_trainer and runtime.type not in ("docker", "modal"):
            raise TrainerError(
                "use_real_trainer=True requires a Docker or Modal harness runtime "
                f"(got {runtime.type!r}); pass --harness.runtime.type docker or "
                "modal (or the load_environment equivalent) -- Prime sandboxes are "
                "no longer supported"
            )
        if runtime.type == "docker":
            # Real Docker trainers pin --memory; verify the live cgroup matches.
            # Heuristic / non-real-trainer Docker harness runs skip this pin.
            if (
                self.curator.use_real_trainer
                and self.curator.proxy_student.runtime_backend == "docker"
            ):
                from .container_memory import (
                    resolve_container_memory_gb,
                    verify_runtime_memory_limit,
                )

                await asyncio.to_thread(
                    verify_runtime_memory_limit,
                    runtime,
                    configured_gb=resolve_container_memory_gb(
                        self.curator.proxy_student.memory_gb,
                        backend="docker",
                    ),
                )
        await runtime.write(
            SELF_SCORE_FILENAME,
            render_self_score_script(
                self.curator,
                hf_token_env=self.config.hf_token_env,
                decon_binary=self.config.decon_binary,
                decon_evals_dir=self.config.decon_evals_dir,
                decon_threshold=self.config.decon_threshold,
            ),
        )
        await runtime.write(HF_CLI_SKILL_RUNTIME_PATH, _HF_CLI_SKILL)
        if self.curator.use_real_trainer:
            await runtime.write(
                SELF_SCORE_TRAIN_FILENAME,
                render_self_score_train_script(),
            )

    @vf.stop
    async def max_turns_reached(self, trace: vf.Trace) -> bool:
        return trace.num_turns >= self.curator.max_turns

    # -- finalize (runs before scoring, while the runtime is live) -------------

    @staticmethod
    def _final_message_text(trace: vf.Trace) -> str:
        msgs = trace.assistant_messages
        if not msgs:
            return ""
        return msgs[-1].content or ""

    async def _manifest_from_workspace_file(
        self,
        task: CuratorTask,
        runtime: Any,
        *,
        warn_invalid: bool = False,
    ) -> tuple[Literal["absent", "invalid", "valid"], Manifest | None]:
        """Read the agent-written manifest from the runtime workspace.

        Distinguishes three outcomes so finalize telemetry can separate a missing
        file from a present-but-invalid one:

        - ``absent``: runtime missing or the configured filename is not readable
        - ``invalid``: file exists but is not a valid non-empty manifest
        - ``valid``: parseable manifest with a non-empty ``sources`` list

        Runtime paths are relative to the configured workdir. Docker and Modal
        resolve this filename under ``/workspace``; the subprocess runtime
        resolves it under its per-rollout workspace.
        """
        if runtime is None:
            return "absent", None
        try:
            raw = await runtime.read(self.config.manifest_filename)
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            log = logger.warning if warn_invalid else logger.debug
            log(
                "Manifest file %r exists but is not valid UTF-8",
                self.config.manifest_filename,
            )
            return "invalid", None
        except Exception:  # noqa: BLE001 - runtime backends use different not-found errors
            return "absent", None
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = None
        manifest = parse_manifest(
            json.dumps(data) if isinstance(data, dict) else "",
            default_token_budget=task.token_budget,
            reserved_local_filename=self.config.manifest_filename,
        )
        if manifest is None or not manifest.sources:
            log = logger.warning if warn_invalid else logger.debug
            log(
                "Manifest file %r exists but does not contain a valid non-empty "
                "manifest",
                self.config.manifest_filename,
            )
            return "invalid", None
        return "valid", manifest

    def _manifest_from_messages(
        self, task: CuratorTask, trace: vf.Trace
    ) -> Manifest | None:
        """Compatibility fallback: parse assistant text messages newest first."""
        manifest = parse_manifest(
            self._final_message_text(trace),
            default_token_budget=task.token_budget,
            reserved_local_filename=self.config.manifest_filename,
        )
        if manifest is not None and manifest.sources:
            return manifest
        for message in reversed(trace.assistant_messages[:-1]):
            manifest = parse_manifest(
                message.content or "",
                default_token_budget=task.token_budget,
                reserved_local_filename=self.config.manifest_filename,
            )
            if manifest is not None and manifest.sources:
                return manifest
        return None

    def _manifest_from_trace_ids(
        self, task: CuratorTask, trace: vf.Trace
    ) -> Manifest | None:
        observed = _ids_from_trace(trace)
        if not observed:
            return None
        limit = self.curator.candidate_limit
        sources = [
            Source(dataset_id=did, config=None, weight=1.0) for did in observed[:limit]
        ]
        return Manifest(token_budget=task.token_budget, sources=sources)

    # Grace-period bound for `_await_final_manifest` (see its docstring): total
    # worst-case wait is attempts * interval seconds, and it only runs when the
    # first workspace-file read misses or catches a partial write.
    _FINALIZE_GRACE_ATTEMPTS = 6
    _FINALIZE_GRACE_INTERVAL_SECONDS = 0.5

    async def _await_final_manifest(
        self, task: CuratorTask, trace: vf.Trace, runtime: Any
    ) -> tuple[Literal["absent", "invalid", "valid"], Manifest | None]:
        """Poll briefly for the agent's workspace manifest before fallbacks.

        The shell command that writes the file and ``finalize()`` can become
        observable in either order. Retrying also tolerates reading while a shell
        redirection is still writing the JSON. A present-but-invalid read mid-poll
        is retried (partial write); the final attempt's status is authoritative.
        """
        status: Literal["absent", "invalid", "valid"] = "absent"
        manifest: Manifest | None = None
        for attempt in range(self._FINALIZE_GRACE_ATTEMPTS):
            await asyncio.sleep(self._FINALIZE_GRACE_INTERVAL_SECONDS)
            status, manifest = await self._manifest_from_workspace_file(
                task,
                runtime,
                warn_invalid=attempt == self._FINALIZE_GRACE_ATTEMPTS - 1,
            )
            if status == "valid":
                return status, manifest
        return status, manifest

    async def finalize(self, task: CuratorTask, trace: vf.Trace, runtime: Any) -> None:
        """Read the agent's manifest from the runtime workspace.

        Positional signature as invoked at ``verifiers/v1/rollout.py:241``; runs
        after generation and before ``score``. Production benchmarks require an
        explicit valid non-empty workspace ``manifest.json``: only that path sets
        ``finalized=1`` and may materialize/train. A short grace-period poll
        handles the shell-write/finalize race.

        Workspace outcomes are reported distinctly via ``manifest_provenance``:

        - ``workspace_file``: valid non-empty file
        - ``invalid_workspace_file``: file present but malformed/empty/invalid
        - ``missing``: file absent (even if assistant-message / opt-in trace
          candidates exist for telemetry under ``manifest_candidate``)

        Missing/invalid outcomes leave ``finalized=0`` and score as the zero
        sentinel (no materialize/train).
        """
        state = trace.state
        status, manifest = await self._manifest_from_workspace_file(task, runtime)
        if status != "valid":
            status, manifest = await self._await_final_manifest(task, trace, runtime)

        candidate: ManifestCandidate | None = None
        telemetry_manifest: Manifest | None = None
        if status == "valid":
            provenance: ManifestProvenance = MANIFEST_PROVENANCE_WORKSPACE_FILE
        elif status == "invalid":
            provenance = MANIFEST_PROVENANCE_INVALID_WORKSPACE_FILE
        else:
            provenance = MANIFEST_PROVENANCE_MISSING
            message_manifest = self._manifest_from_messages(task, trace)
            if message_manifest is not None:
                telemetry_manifest = message_manifest
                candidate = MANIFEST_CANDIDATE_ASSISTANT_MESSAGE
            elif self.curator.allow_trace_id_manifest_fallback:
                trace_manifest = self._manifest_from_trace_ids(task, trace)
                if trace_manifest is not None:
                    telemetry_manifest = trace_manifest
                    candidate = MANIFEST_CANDIDATE_TRACE_FALLBACK

        RolloutStore.set_manifest_provenance(state, provenance)
        trace.info["manifest_provenance"] = provenance
        if candidate is not None:
            trace.info["manifest_candidate"] = candidate
        else:
            trace.info.pop("manifest_candidate", None)

        production_ok = (
            provenance == MANIFEST_PROVENANCE_WORKSPACE_FILE
            and manifest is not None
            and bool(manifest.sources)
        )
        if production_ok:
            assert manifest is not None  # for type checkers
            if manifest.sample_docs_per_source is not None:
                fetch_cap = manifest.sample_docs_per_source
                reachable_tokens = (
                    len(manifest.sources) * fetch_cap * EST_TOKENS_PER_DOC
                )
                if manifest.token_budget > reachable_tokens:
                    logger.warning(
                        "TOKEN BUDGET IS NOT REACHABLE with the configured fetch cap: "
                        "token_budget=%d sources=%d fetch_cap=%d "
                        "estimated_tokens_per_doc=%d estimated_max_tokens=%d",
                        manifest.token_budget,
                        len(manifest.sources),
                        fetch_cap,
                        EST_TOKENS_PER_DOC,
                        reachable_tokens,
                    )
            RolloutStore.set_manifest(state, manifest)
            RolloutStore.set_finalized(state, True)
        else:
            if telemetry_manifest is not None and telemetry_manifest.sources:
                # Retain non-production candidates for telemetry; do not score them.
                RolloutStore.set_manifest(state, telemetry_manifest)
            RolloutStore.set_finalized(state, False)
        validation_id = self.curator.validation_set.dataset_id
        accessed_validation_set = False
        for message in trace.assistant_messages:
            if validation_id in content_text(getattr(message, "content", "")):
                accessed_validation_set = True
                break
            for tool_call in message.tool_calls or []:
                command = _shell_command_from_tool_args(
                    getattr(tool_call, "arguments", "") or ""
                )
                if validation_id in str(command):
                    accessed_validation_set = True
                    break
            if accessed_validation_set:
                break
        RolloutStore.set_val_set_access(state, accessed_validation_set)

    # -- prepared scoring (run once per rollout) -------------------------------

    def _lock_for(self, key: str) -> asyncio.Lock:
        lock = self._scoring_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._scoring_locks[key] = lock
        return lock

    async def _prepared(
        self, trace: vf.Trace, runtime: vf.Runtime | None = None
    ) -> dict[str, Any]:
        cached = self._scoring_cache.get(trace.id)
        if cached is not None:
            return cached
        scorer = self._ensure()
        lock = self._lock_for(trace.id)
        async with lock:
            cached = self._scoring_cache.get(trace.id)
            if cached is not None:
                return cached
            scoring = await scorer.compute_scoring(trace.state, runtime)
            self._scoring_cache[trace.id] = scoring
        self._scoring_locks.pop(trace.id, None)
        return scoring

    async def score(self, trace: vf.Trace, runtime: vf.Runtime | None) -> None:
        try:
            await super().score(trace, runtime)
        finally:
            self._scoring_cache.pop(trace.id, None)
            self._scoring_locks.pop(trace.id, None)
            RolloutStore.cleanup(trace.state)

    # -- rewards (weighted contributions; coefficients folded in) --------------

    @vf.reward(weight=1.0)
    async def perf_reward(
        self, trace: vf.Trace, runtime: vf.Runtime | None = None
    ) -> float:
        return self.curator.alpha_perf * (await self._prepared(trace, runtime))["perf"]

    @vf.reward(weight=1.0)
    async def leakage_penalty(
        self, trace: vf.Trace, runtime: vf.Runtime | None = None
    ) -> float:
        return (
            -self.curator.lambda_leakage
            * (await self._prepared(trace, runtime))["leakage"]["leakage_score"]
        )

    # -- zero-weight diagnostic metrics (recorded, not summed into reward) -----

    @vf.metric
    async def perf_loss(
        self, trace: vf.Trace, runtime: vf.Runtime | None = None
    ) -> float:
        return (await self._prepared(trace, runtime))["loss"]

    @vf.metric
    async def perf_accuracy(
        self, trace: vf.Trace, runtime: vf.Runtime | None = None
    ) -> float:
        return (await self._prepared(trace, runtime))["accuracy"]

    @vf.metric
    async def perf_vs_baseline(
        self, trace: vf.Trace, runtime: vf.Runtime | None = None
    ) -> float:
        return (await self._prepared(trace, runtime))["perf_vs_baseline"]

    @vf.metric
    async def train_flops(
        self, trace: vf.Trace, runtime: vf.Runtime | None = None
    ) -> float:
        return (await self._prepared(trace, runtime))["flops"]

    @vf.metric
    async def corpus_tokens(
        self, trace: vf.Trace, runtime: vf.Runtime | None = None
    ) -> float:
        return float((await self._prepared(trace, runtime))["tokens"])

    @vf.metric
    async def budget_fill_ratio(
        self, trace: vf.Trace, runtime: vf.Runtime | None = None
    ) -> float:
        return float((await self._prepared(trace, runtime))["budget_fill_ratio"])

    @vf.metric
    async def num_sources(
        self, trace: vf.Trace, runtime: vf.Runtime | None = None
    ) -> float:
        return float((await self._prepared(trace, runtime))["num_sources"])

    @vf.metric
    async def local_source_count(
        self, trace: vf.Trace, runtime: vf.Runtime | None = None
    ) -> float:
        await self._prepared(trace, runtime)
        return float(RolloutStore.local_source_count(trace.state))

    @vf.metric
    async def local_source_bytes(
        self, trace: vf.Trace, runtime: vf.Runtime | None = None
    ) -> float:
        await self._prepared(trace, runtime)
        return float(RolloutStore.local_source_bytes(trace.state))

    @vf.metric
    async def local_source_truncated(
        self, trace: vf.Trace, runtime: vf.Runtime | None = None
    ) -> float:
        await self._prepared(trace, runtime)
        return 1.0 if RolloutStore.local_source_truncated(trace.state) else 0.0

    @vf.metric
    async def val_set_access(
        self, trace: vf.Trace, runtime: vf.Runtime | None = None
    ) -> float:
        await self._prepared(trace, runtime)
        return 1.0 if RolloutStore.val_set_access(trace.state) else 0.0

    @vf.metric
    async def leakage_score(
        self, trace: vf.Trace, runtime: vf.Runtime | None = None
    ) -> float:
        return (await self._prepared(trace, runtime))["leakage"]["leakage_score"]

    @vf.metric
    async def num_contaminated_matches(
        self, trace: vf.Trace, runtime: vf.Runtime | None = None
    ) -> float:
        return float(
            (await self._prepared(trace, runtime))["leakage"][
                "num_contaminated_matches"
            ]
        )

    @vf.metric
    async def finalized(self, trace: vf.Trace) -> float:
        return 1.0 if RolloutStore.is_finalized(trace.state) else 0.0

    @vf.metric
    async def manifest_missing(self, trace: vf.Trace) -> float:
        """1.0 when the workspace manifest file was absent."""
        return (
            1.0
            if RolloutStore.manifest_provenance(trace.state)
            == MANIFEST_PROVENANCE_MISSING
            else 0.0
        )

    @vf.metric
    async def manifest_invalid(self, trace: vf.Trace) -> float:
        """1.0 when a workspace manifest file was present but invalid."""
        return (
            1.0
            if RolloutStore.manifest_provenance(trace.state)
            == MANIFEST_PROVENANCE_INVALID_WORKSPACE_FILE
            else 0.0
        )

    @vf.metric
    async def tool_error_count(
        self, trace: vf.Trace, runtime: vf.Runtime | None = None
    ) -> float:
        await self._prepared(trace, runtime)
        return float(RolloutStore.tool_error_count(trace.state))

    @vf.metric
    async def external_failure(
        self, trace: vf.Trace, runtime: vf.Runtime | None = None
    ) -> float:
        await self._prepared(trace, runtime)
        return 1.0 if RolloutStore.has_external_failure(trace.state) else 0.0

    @vf.metric
    async def decon_error(
        self, trace: vf.Trace, runtime: vf.Runtime | None = None
    ) -> float:
        return (await self._prepared(trace, runtime)).get("decon_error", 0.0)

    @vf.metric
    async def val_screen_skipped(
        self, trace: vf.Trace, runtime: vf.Runtime | None = None
    ) -> float:
        return (await self._prepared(trace, runtime)).get("val_screen_skipped", 0.0)

    async def trainer_error_str(
        self, trace: vf.Trace, runtime: vf.Runtime | None = None
    ) -> str:
        await self._prepared(trace, runtime)
        # fmt: off
        err = (RolloutStore.trainer_error(trace.state) or "")[
            :TRAINER_ERROR_STR_LIMIT
        ]
        # fmt: on
        if err:
            logger.warning("trainer error: %s", err)
        return err

    @vf.metric
    async def trainer_error_msg(
        self, trace: vf.Trace, runtime: vf.Runtime | None = None
    ) -> float:
        return 1.0 if await self.trainer_error_str(trace, runtime) else 0.0


__all__ = [
    "CuratorTaskset",
    "_ids_from_trace",
    "_shell_command_from_tool_args",
    "parse_manifest",
    "extract_json_object",
]

