"""The curation taskset: tasks, manifest parsing, finalize, rewards, and metrics.

``CuratorTaskset`` drives a *toolless* curation rollout: the agent runs the
Hugging Face ``hf`` CLI in its own shell to search/inspect datasets and decides a
curation mixture, then emits that decision as a single fenced ```json block as its
final message. Because the taskset exposes **no** tool servers (it does not
override ``Taskset.tools``), it satisfies the non-MCP harness gate at
``verifiers/v1/env.py:239-247`` and runs under codex / kimi_code as well as the
default / bash harnesses.

After generation, ``finalize`` (run before scoring, while the runtime is live):
  1. parses the manifest JSON from the agent's final message,
  2. meters the live ``hf`` discovery cost into the cost ledger (see
     :mod:`hf_meter`), preferring the PATH-shim's runtime log and falling back to
     reconstructing hf calls from the trace.

Scoring is unchanged from v1: the finalized manifest's datasets are materialized
and used to train the fixed proxy student, and the composite reward is:

    R(M, H) = alpha_perf*Perf - lambda_cost*Cost - lambda_leakage*Leakage

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
from typing import Any

import verifiers.v1 as vf
from pydantic import ValidationError

from .corpus import EST_TOKENS_PER_DOC, CorpusBuilder
from .docker_network import DockerHostReachability
from .eval_corpus import DEFAULT_EVAL_CORPUS
from .hf_access import HuggingFaceDatasetClient, RetryPolicy
from .hf_meter import _content_text, extract_hf_commands, install_shim, meter_ledger
from .leakage import LeakageDetector
from .leakage_reference import LeakageReferenceLoader
from .models import (
    CuratorConfig,
    FilterSpec,
    Manifest,
    ProxyStudentConfig,
    Sampling,
    Source,
)
from .rewards import CuratorScorer
from .rollout_state import CuratorState, RolloutStore
from .tasks import CuratorTask, build_tasks
from .trainer import (
    HeuristicProxyTrainer,
    ProxyStudentTrainer,
    RuntimeSelectedTrainer,
    TrainerError,
)
from .val_set import ValidationSetConfig, ValTokenLoader

logger = logging.getLogger(__name__)

_DISCOVERY_OUTPUT_CHARS_PER_CALL = 6_000
_DISCOVERY_OUTPUT_MARGIN = 1.1

SYSTEM_PROMPT = """IMPORTANT: Be extremely concise in every message, and work by RUNNING commands in your shell rather than describing them. On your very first step, actually run a command — no preamble, no plan, no explanation first. If your harness gives you a shell/terminal/exec tool, you MUST call that tool to run each command below; writing a command out as ordinary text without calling the tool does NOT run it and wastes the rollout. If instead your harness executes your reply directly as a shell command, just reply with the command itself. Bootstrap the hf CLI if needed and run a search in that same first command. If you must plan, keep it to one sentence, then immediately run a command.

You are a pretraining-data curation agent. Your job is to assemble a dataset mixture that, when used to train a fixed small GPT-2-scale student (everything fixed but the data), maximizes the student's performance.

Domain context — target large-scale, diverse, high-quality text corpora for general-purpose LLM pretraining. Good sources include encyclopedic text (Wikipedia, encyclopedias), scientific literature (papers, research), instructional text, and broad web corpora (C4, FineWeb, OpenWebText). Prefer encyclopedic sources (highest utility), then scientific, then instructional. Well-known high-quality datasets: Wikimedia/Wikipedia, allenai/c4, Skylion007/openwebtext, HuggingFaceFW/fineweb, allenai/dolma. Avoid code-only or narrow task-specific datasets. For each candidate, retrieve key metadata: downloads, likes, last modified, splits, configs. Use result limits appropriate to the turn budget. Avoid repetition: if you already have detailed info for a dataset, move on.

You have a normal bash shell. The Hugging Face `hf` CLI is the only tool you need, but some runtime images do not include it. Your FIRST command MUST defensively check and install it if missing, then continue directly to an `hf datasets ls` search in the SAME shell command (one turn total):

`if ! command -v hf >/dev/null 2>&1; then pip install -q 'huggingface-hub>=0.34'; fi; hf datasets ls --search "wikipedia" --sort downloads --limit 5 | head -c 6000`

Do not spend turns diagnosing missing commands: do not try `huggingface-cli` or `python -m huggingface_hub`. The one conditional pip install above is the ONLY installation step allowed. After it, run the `hf` subcommands below directly in bash; do not write Python, create a virtualenv, import `huggingface_hub`/`datasets`, or install anything else. Use `hf` to discover and inspect candidates, then decide a weighted curation mixture. Only use Hugging Face datasets modified on or before the cutoff date.

`hf` command cheat-sheet:
  - Search datasets:   hf datasets ls --search "<query>" --sort downloads --limit 5 | head -c 6000
  - Filter / quiet:    hf datasets ls --search "<query>" --filter text --limit 10 -q | head -c 6000
  - JSON output:        hf datasets ls --search "<query>" --limit 5 --format json --expand downloads,likes,lastModified | head -c 6000
  - Inspect a dataset: hf datasets info <dataset_id> --expand downloads,likes,tags | head -c 6000
Every `hf` command MUST end with `| head -c 6000`. Never request `tags` from `datasets ls`: multilingual repositories can return tens of thousands of tag characters and overflow your model context. Request tags only when inspecting one shortlisted dataset.
Inspect a few candidates, prefer well-downloaded, clearly-licensed text datasets, and check each was last modified on or before the cutoff date.

You are billed for live discovery: each search and each inspect/download call adds to the cost penalty, so be economical. Your reward is derived from proxy-student held-out cross-entropy loss, with penalties for cost and for leakage/contamination against a held-out evaluation set.

`"text_field": null` triggers auto-detection (the environment tries 'text', 'content', 'passage', 'abstract', 'query'/'response' concat, etc.). Set explicitly only if you know the exact column from `hf datasets info`.

Reliable starter datasets: HuggingFaceFW/fineweb (text_field: "text"), roneneldan/TinyStories (text_field: "text"), wikimedia/wikipedia (config: "20231101.en", text_field: "text"), allenai/c4 (config: "en", text_field: "text"), Salesforce/wikitext (config: "wikitext-103-v1", text_field: "text").

When you are done, emit your decision as your FINAL message: a single fenced ```json block (and nothing else after it) with this exact schema:

```json
{
  "token_budget": 1000000,
  "sample_docs_per_source": "<int, 1-100000; compute from your own token_budget, do not copy this>",
  "sources": [
    {
      "id": "<huggingface dataset id, e.g. HuggingFaceFW/fineweb>",
      "weight": 1.0,
      "config": null,
      "split": "train",
      "text_field": null,
      "filters": [{"kind": "min_chars", "params": {"value": 200}}, {"kind": "dedup_exact"}],
      "max_docs": null,
      "max_tokens": null
    }
  ]
}
```

Each source REQUIRES `id` (the Hugging Face dataset id) and `weight` (>= 0, relative mixing weight). `config`, `split`, `text_field`, `filters`, `max_docs`, and `max_tokens` are optional. Supported filter kinds: min_chars, max_chars, min_tokens, max_symbol_ratio, min_alpha_ratio, drop_regex, keep_regex, dedup_exact. Always emit a non-empty `sources` list — an empty or missing manifest scores zero.

Optional top-level `sample_docs_per_source` (integer, 1-100000) controls how many documents are fetched PER SOURCE from the Hub for this rollout — it is the fetch cap itself, not a post-fetch truncation like `max_docs`/`max_tokens`. Omit it (or set it to null) to use the environment's configured default. Fetching more documents lets the student train on more unique tokens (useful for a large `token_budget`), but also raises `cost_penalty`, since every fetched token is billed. You MUST compute this number yourself from your actual `token_budget` and number of sources — roughly `token_budget / (num_sources * 250)` tokens-per-doc, capped at 100000 — rather than reusing the schema example's placeholder verbatim."""


# --------------------------------------------------------------------------- #
# manifest parsing (the agent's final-message deliverable)
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
# ```json ... ``` (or bare ```), used to locate the agent's manifest block.
_FENCE_RE = re.compile(r"```(?:json|jsonc|JSON)?\s*(.*?)```", re.DOTALL)

# Matches "owner/name" patterns in free-form text.  Lookbehind blocks URL
# scheme colons, existing slashes, and word characters so only standalone
# identifiers match.
_HF_ID_RE = re.compile(
    r"(?<![:/\w])([A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]+)(?![/\w])"
)

# Namespace / name tokens that are path segments or field names, not HF ids.
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


def _looks_like_hf_id(s: str) -> bool:
    """True when ``s`` plausibly is a HuggingFace dataset id (owner/name)."""
    if "/" not in s:
        return False
    ns, name = s.split("/", 1)
    if not ns or not name:
        return False
    if ns.lower() in _NOT_HF_NAMESPACES or name.lower() in _NOT_HF_NAMES:
        return False
    # Skip file-extension-looking names (e.g. "some/file.json").
    if "." in name and name.rsplit(".", 1)[-1].lower() in _FILE_EXTS:
        return False
    return True


def _ids_from_trace(trace: vf.Trace) -> list[str]:
    """Dataset ids actually observed in the rollout's tool calls and outputs.

    Two tiers, in order of reliability:
    1. ``hf datasets info <id>`` command arguments — the agent explicitly
       inspected these ids, so they are definitive.
    2. Free-form text in bash tool-result messages — covers ids that appeared
       in ``hf datasets ls`` output but were never individually inspected.

    If the agent inspected any ids, only those ids are returned. Search-result
    ids are used only when no inspection happened: blindly materializing every
    search hit makes recovery select post-cutoff, gated, or incompatible
    repositories the agent deliberately did not shortlist. Within a tier,
    results preserve first-observation order and are deduplicated.
    """
    inspected: dict[str, None] = {}  # ordered-set via insertion-order dict

    # 1. Explicitly-inspected ids from assistant tool-call argument JSON.
    for msg in trace.assistant_messages:
        for tc in msg.tool_calls or []:
            try:
                cmd = json.loads(getattr(tc, "arguments", "") or "{}").get(
                    "command", ""
                )
            except (json.JSONDecodeError, AttributeError):
                cmd = getattr(tc, "arguments", "") or ""
            for argv in extract_hf_commands(cmd):
                # argv from hf_meter: ["datasets", "info", "<id>", ...]
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

    # 2. Ids from bash tool-result text (hf datasets ls output, info summaries).
    observed: dict[str, None] = {}
    for msg in getattr(trace, "tool_messages", []):
        text = _content_text(getattr(msg, "content", ""))
        for m in _HF_ID_RE.finditer(text):
            did = m.group(1)
            if _looks_like_hf_id(did):
                observed.setdefault(did, None)

    return list(observed)


def _iter_json_objects(s: str):
    """Yield each top-level balanced ``{...}`` substring (string-aware)."""
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
    """Scan ``s`` for balanced JSON objects → ``(last_with_sources, last_any)``.

    ``last_with_sources`` is the LAST object that is a dict carrying a non-empty
    ``sources`` list; ``last_any`` is the LAST parseable dict object. Either may be
    ``None`` when ``s`` contains no such object.
    """
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
    """Extract the manifest JSON object from the agent's final message.

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
    # Every fenced block (in order) plus the whole text; later candidates win.
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
            # Agent-supplied filters are best-effort. Invalid params follow the
            # same tolerant policy as unknown kinds: discard the spec rather than
            # failing the scoring pass later in DocumentFilter.
            continue
        specs.append(FilterSpec(kind=kind, params=params))
    return specs


def _coerce_source(raw: Any) -> Source | None:
    """Map one agent-supplied source dict (tolerant of `id`/`dataset_id`) to a Source."""
    if isinstance(raw, str):
        return Source(dataset_id=raw.strip()) if raw.strip() else None
    if not isinstance(raw, dict):
        return None
    dataset_id = (
        raw.get("dataset_id")
        or raw.get("id")
        or raw.get("dataset")
        or raw.get("repo_id")
        or raw.get("name")
    )
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        return None
    kwargs: dict[str, Any] = {"dataset_id": dataset_id.strip()}
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
    text: str, default_token_budget: int | None = None
) -> Manifest | None:
    """Parse + validate the agent's final-message manifest.

    Returns a :class:`Manifest` with at least one source, or ``None`` when the
    message has no usable manifest (empty / truncated / no valid sources) — the
    caller treats ``None`` as "not finalized" → graceful zero score.
    """
    data = extract_json_object(text)
    if not isinstance(data, dict):
        return None
    raw_sources = data.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        return None
    sources = [s for s in (_coerce_source(r) for r in raw_sources) if s is not None]
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
    """Validated configuration for the curation taskset (mirrors the v0
    ``load_environment`` knobs). Tunable via ``--taskset.*``."""

    cutoff_date: str = "2024-12-31"
    token_budget: int = 1_000_000
    hf_token_env: str = "HF_TOKEN"
    candidate_limit: int = 8
    scan_limit: int = 50
    sample_docs_per_source: int = 64
    allow_script_datasets: bool = False
    max_turns: int = 12
    alpha_perf: float = 1.0
    lambda_cost: float = 0.1
    lambda_leakage: float = 1.0
    # Baseline-relative Perf signal. True (default) uses relative loss reduction
    # over ``perf_baseline_loss`` as the Perf reward — the right choice for real
    # LMs where loss ~ 9 nats and exp(-loss) ≈ 0.  Set to False only for toy
    # models where absolute loss < 1 and exp(-loss) is meaningful.
    perf_baseline_loss: float = math.log(50304)
    baseline_relative_perf: bool = True
    max_concurrent_fetches: int = 8
    max_concurrent_training: int = 1
    fetch_timeout_seconds: float = 30.0
    fetch_timeout_per_doc_seconds: float = 0.25
    fetch_max_attempts: int = 3
    use_real_trainer: bool = False
    proxy_student: dict[str, Any] = {}
    validation_set: dict[str, Any] = {}
    eval_corpus: list[str] | None = None


try:
    _TasksetBase = vf.Taskset[CuratorTask, CuratorTasksetConfig, CuratorState]
except TypeError:
    _TasksetBase = vf.Taskset


class CuratorTaskset(_TasksetBase):
    def __init__(self, config: CuratorTasksetConfig) -> None:
        super().__init__(config)
        # The single validated config the reward derives from (bounds + cross-field
        # checks fail fast here, exactly as v0 ``CuratorConfig`` did).
        self.curator = self._build_curator_config(config)
        # Install the PATH-shadow `hf` cost-metering shim once per worker (best
        # effort; a missing hf just routes metering through the trace fallback).
        install_shim()
        # Collaborators are built lazily (so constructing the taskset needs no HF
        # token / network); each is also a test-injection seam.
        self._client: HuggingFaceDatasetClient | None = None
        self._corpus_builder: CorpusBuilder | None = None
        self._trainer: ProxyStudentTrainer | None = None
        self._leakage_detector: LeakageDetector | None = None
        self._leakage_reference_loader: LeakageReferenceLoader | None = None
        self._val_loader: ValTokenLoader | None = None
        self._scorer: CuratorScorer | None = None
        # Per-rollout scoring cache + locks, keyed by trace id: the heavy prepare
        # region runs once even under concurrent reward/metric evaluation.
        self._scoring_cache: dict[str, dict[str, Any]] = {}
        self._scoring_locks: dict[str, asyncio.Lock] = {}

    @staticmethod
    def _build_curator_config(config: CuratorTasksetConfig) -> CuratorConfig:
        return CuratorConfig(
            cutoff_date=config.cutoff_date,
            token_budget=config.token_budget,
            candidate_limit=config.candidate_limit,
            scan_limit=config.scan_limit,
            sample_docs_per_source=config.sample_docs_per_source,
            allow_script_datasets=config.allow_script_datasets,
            max_turns=config.max_turns,
            alpha_perf=config.alpha_perf,
            lambda_cost=config.lambda_cost,
            lambda_leakage=config.lambda_leakage,
            perf_baseline_loss=config.perf_baseline_loss,
            baseline_relative_perf=config.baseline_relative_perf,
            max_concurrent_fetches=config.max_concurrent_fetches,
            max_concurrent_training=config.max_concurrent_training,
            fetch_timeout_seconds=config.fetch_timeout_seconds,
            fetch_timeout_per_doc_seconds=config.fetch_timeout_per_doc_seconds,
            fetch_max_attempts=config.fetch_max_attempts,
            use_real_trainer=config.use_real_trainer,
            proxy_student=ProxyStudentConfig(**(config.proxy_student or {})),
            validation_set=ValidationSetConfig(**(config.validation_set or {})),
        )

    # -- collaborators ---------------------------------------------------------

    def _fetch_policy(self) -> RetryPolicy:
        return RetryPolicy(
            attempts=self.curator.fetch_max_attempts,
            timeout=self.curator.fetch_timeout_seconds,
            per_doc_seconds=self.curator.fetch_timeout_per_doc_seconds,
        )

    def _ensure(self) -> CuratorScorer:
        """Build the scoring collaborators on first use (honoring injected ones)."""
        if self._scorer is not None:
            return self._scorer
        if self._client is None:
            self._client = HuggingFaceDatasetClient(
                token_env=self.config.hf_token_env,
                allow_script_datasets=self.curator.allow_script_datasets,
            )
        if self._corpus_builder is None:
            self._corpus_builder = CorpusBuilder(
                client=self._client,
                sample_docs_per_source=self.curator.sample_docs_per_source,
                retry_policy=self._fetch_policy(),
                fetch_limit=self.curator.max_concurrent_fetches,
            )
        if self._val_loader is None:
            self._val_loader = ValTokenLoader(
                self.curator.validation_set,
                retry_policy=self._fetch_policy(),
                fetch_limit=self.curator.max_concurrent_fetches,
            )
        if self._leakage_detector is None and self.config.eval_corpus:
            self._leakage_detector = LeakageDetector(self.config.eval_corpus)
        if self._leakage_detector is None and self._leakage_reference_loader is None:
            self._leakage_reference_loader = LeakageReferenceLoader(
                self._val_loader,
                fallback_docs=DEFAULT_EVAL_CORPUS,
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
            self._leakage_detector,
            self._leakage_reference_loader,
        )
        return self._scorer

    def _build_real_trainer(self) -> ProxyStudentTrainer:
        """Construct the real (GPU) proxy-student trainer.

        Both concrete backends are built eagerly (cheap, no I/O): which one
        actually trains is decided at SCORE time from the live harness
        runtime's ``type`` (docker -> ``HarnessRuntimeProxyTrainer``, modal ->
        ``ModalProxyTrainer``), via ``RuntimeSelectedTrainer``.
        ``proxy_student.runtime_backend`` is never read here -- it only shapes
        the static task/harness declarations built ahead of any live runtime
        (see ``load_tasks`` and ``load_environment``).
        """
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
        tasks = build_tasks(self.curator.cutoff_date, self.curator.token_budget)
        system_prompt = self._system_prompt()
        updates: dict[str, Any] = {"system_prompt": system_prompt}
        ps = self.curator.proxy_student
        if self.curator.use_real_trainer and ps.runtime_backend == "docker":
            # Native taskset runs do not call load_environment(), so declare the
            # image, resources, and scoring deadline on each task. A task image
            # also makes a subprocess runtime fail before rollout.
            updates.update(
                {
                    "image": ps.docker_image,
                    "workdir": "/workspace",
                    "resources": vf.TaskResources(
                        cpu=float(ps.cpu_cores),
                        memory=float(ps.memory_gb),
                        gpu=str(ps.gpu_count) if ps.gpu_count > 0 else None,
                        disk=float(ps.disk_size_gb),
                    ),
                    "timeout": vf.TaskTimeout(
                        scoring=ps.effective_scoring_timeout_seconds
                    ),
                }
            )
        elif self.curator.use_real_trainer and ps.runtime_backend == "modal":
            from .modal_backend import _modal_gpu_for

            updates.update(
                {
                    "image": ps.docker_image,
                    "workdir": "/workspace",
                    "resources": vf.TaskResources(
                        cpu=float(ps.cpu_cores),
                        memory=float(ps.memory_gb),
                        gpu=_modal_gpu_for(ps.modal_gpu),
                        disk=float(ps.disk_size_gb),
                    ),
                    "timeout": vf.TaskTimeout(
                        scoring=ps.effective_scoring_timeout_seconds
                    ),
                }
            )
        return [task.model_copy(update=updates) for task in tasks]

    def _system_prompt(self) -> str:
        """Render the configured turn budget into the per-rollout prompt.

        Spells out the actual ``max_turns`` and strictly limits discovery so the
        agent commits a manifest before the turn cap.
        """
        max_turns = self.curator.max_turns
        discovery_rounds, discovery_calls = self._discovery_budget()
        commit_by = max(1, max_turns - max(3, max_turns // 8))
        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"You have {max_turns} turns total (model turns). A response that invokes bash "
            f"uses one model turn even if it contains multiple tool calls; every "
            f"individual `hf` call is still billed, so run one command at a time. "
            f"A discovery round = one `hf datasets ls` call + one `hf datasets info` "
            f"call (2 turns). You MUST perform at most {discovery_rounds} discovery "
            f"rounds (<={discovery_calls} bash calls). After your final discovery "
            f"round, and no later than turn "
            f"{commit_by} — you MUST commit your manifest.\n\n"
            f"HOW TO COMMIT: stop running commands and reply with a plain message "
            f"containing ONLY the fenced ```json block — do not run any shell command "
            f"in that step, and do not print the manifest through the shell. "
            f"Do not add any text after the closing ``` fence. If you still have "
            f"commands available but you have enough evidence to pick sources, commit "
            f"immediately — do not fill remaining turns with more searches.\n\n"
            f"CRITICAL — no invented sources: every `id` in your manifest MUST be "
            f"copied verbatim from a dataset id that appeared in your `hf datasets ls` "
            f"or `hf datasets info` output during this rollout. Do NOT invent or guess "
            f"dataset ids or config names. If a `config` was not explicitly listed in "
            f"command output, set `config` to null. A manifest with a fabricated id or "
            f"config materializes zero tokens and scores zero. An empty or missing "
            f"manifest also scores zero."
        )

    def _discovery_budget(self) -> tuple[int, int]:
        """Return the configured ``(rounds, calls)`` discovery allowance."""
        rounds = max(
            2,
            min(12, self.curator.max_turns // 6, self.curator.scan_limit // 10),
        )
        return rounds, rounds * 2

    def _discovery_output_budget_chars(self) -> int:
        """Match the stop budget to the prompt's per-call output contract."""
        _, calls = self._discovery_budget()
        return math.ceil(
            calls * _DISCOVERY_OUTPUT_CHARS_PER_CALL * _DISCOVERY_OUTPUT_MARGIN
        )

    # NOTE: deliberately no ``tools(...)`` override — the agent curates via the
    # `hf` CLI in its own shell, so the taskset exposes no MCP tool servers and
    # ``type(CuratorTaskset).tools is Taskset.tools`` holds. This is what lets the
    # non-MCP gate (env.py:239-247) pass for codex / kimi_code / bash harnesses.

    async def setup(self, task: CuratorTask, runtime: vf.Runtime) -> None:
        """Require a Docker or Modal harness runtime when the real trainer is on.

        Trainer selection itself is decided entirely by the live
        ``runtime.type`` (see ``RuntimeSelectedTrainer``); this only guards
        against forgetting to configure a container/sandbox harness runtime at
        all -- Prime sandboxes are no longer supported, and the subprocess
        runtime cannot host GPU training.
        """
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
            DockerHostReachability.configure()

    @vf.stop
    async def max_turns_reached(self, trace: vf.Trace) -> bool:
        """Cap the rollout at ``max_turns`` model turns (the harness also stops
        naturally once the model emits a final answer with no tool calls)."""
        return trace.num_turns >= self.curator.max_turns

    @vf.stop
    async def discovery_output_budget_reached(self, trace: vf.Trace) -> bool:
        """Stop before oversized CLI results can overflow the model context.

        Finalization can still recover a manifest from dataset ids observed in
        the trace, so this degrades to a scored fallback instead of a provider
        error that skips finalization and scoring entirely.
        """
        return (
            sum(
                len(_content_text(getattr(message, "content", "")))
                for message in trace.tool_messages
            )
            >= self._discovery_output_budget_chars()
        )

    # -- finalize (runs before scoring, while the runtime is live) -------------

    @staticmethod
    def _final_message_text(trace: vf.Trace) -> str:
        msgs = trace.assistant_messages
        if not msgs:
            return ""
        return msgs[-1].content or ""

    def _manifest_from_messages(
        self, task: CuratorTask, trace: vf.Trace
    ) -> Manifest | None:
        """Tier 1: parse a fenced JSON manifest from any assistant text message,
        newest first."""
        manifest = parse_manifest(
            self._final_message_text(trace), default_token_budget=task.token_budget
        )
        if manifest is not None and manifest.sources:
            return manifest
        for message in reversed(trace.assistant_messages[:-1]):
            manifest = parse_manifest(
                message.content or "", default_token_budget=task.token_budget
            )
            if manifest is not None and manifest.sources:
                return manifest
        return None

    def _manifest_from_trace_ids(
        self, task: CuratorTask, trace: vf.Trace
    ) -> Manifest | None:
        """Tier 2: synthesize from dataset ids actually observed in the rollout's
        bash tool calls / outputs. Uses only ids the agent genuinely discovered;
        never invents ids. ``config`` is ``null`` for all sources unless a config
        was explicitly shown in tool output."""
        observed = _ids_from_trace(trace)
        if not observed:
            return None
        limit = self.curator.candidate_limit
        sources = [
            Source(dataset_id=did, config=None, weight=1.0) for did in observed[:limit]
        ]
        return Manifest(token_budget=task.token_budget, sources=sources)

    # Grace-period bound for `_await_final_manifest` (see its docstring): total
    # worst-case wait is attempts * interval seconds, and it only ever runs on the
    # race/fallback path -- never on a rollout whose final message already landed.
    _FINALIZE_GRACE_ATTEMPTS = 6
    _FINALIZE_GRACE_INTERVAL_SECONDS = 0.5

    async def _await_final_manifest(
        self, task: CuratorTask, trace: vf.Trace
    ) -> Manifest | None:
        """Poll briefly for the agent's final assistant message before giving up
        on the primary (message-parsed) manifest path.

        WHY THIS EXISTS: the `verifiers` interception server (pinned third-party
        dependency, `verifiers==0.1.15.dev376` -- not something we control or may
        edit) streams the agent's final assistant message and commits it into
        `trace.nodes` asynchronously, and can do so AFTER the rollout pool has
        already unregistered this rollout and invoked our `finalize()`. When that
        race is lost, `_manifest_from_messages` sees a trace that is one message
        short of the agent's real, already-submitted manifest, and would otherwise
        fall straight through to the lossy trace-discovered-ids fallback (whose
        synthesized manifest always has `sample_docs_per_source=None`, silently
        defaulting away whatever the agent actually requested). `trace.nodes` is a
        plain list the interception server appends to in place, and
        `trace.assistant_messages` is a live property over it (not a cached
        snapshot), so yielding the event loop a few times via short sleeps is
        enough for the pending append to land. Bounded to a handful of short
        sleeps so a rollout that genuinely never submits a manifest still falls
        through to the fallback promptly.
        """
        for _ in range(self._FINALIZE_GRACE_ATTEMPTS):
            await asyncio.sleep(self._FINALIZE_GRACE_INTERVAL_SECONDS)
            manifest = self._manifest_from_messages(task, trace)
            if manifest is not None:
                return manifest
        return None

    async def finalize(self, task: CuratorTask, trace: vf.Trace, runtime: Any) -> None:
        """Parse the agent's manifest and meter live `hf` discovery cost.

        Positional signature as invoked at ``verifiers/v1/rollout.py:241``; runs
        after generation and before ``score``. Assistant turns are scanned most
        recent first, so the final-message behavior is preserved while a manifest
        emitted before trailing `hf` calls still finalizes at the turn cap. If the
        final message hasn't landed in the trace yet (see `_await_final_manifest`),
        a short grace-period poll gives the upstream race a chance to resolve
        before falling back. Tolerant of a missing/garbled rollout (no manifest
        anywhere -> "not finalized" -> the scorer returns the zero sentinel). The
        discovery cost ledger is computed here and persisted; the scorer still adds
        ``train_flops`` and the materialization cost on top.
        """
        state = trace.state
        manifest = self._manifest_from_messages(task, trace)
        if manifest is None:
            manifest = await self._await_final_manifest(task, trace)
        if manifest is None:
            manifest = self._manifest_from_trace_ids(task, trace)
        if manifest is not None and manifest.sources:
            fetch_cap = (
                manifest.sample_docs_per_source or self.curator.sample_docs_per_source
            )
            reachable_tokens = len(manifest.sources) * fetch_cap * EST_TOKENS_PER_DOC
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
            RolloutStore.set_finalized(state, False)
        try:
            ledger = await meter_ledger(trace, runtime)
        except Exception:  # noqa: BLE001 - metering must never fail the rollout
            ledger = RolloutStore.ledger(state)
        RolloutStore.set_ledger(state, ledger)

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
        """The single heavy scoring pass for a rollout, cached per trace.

        Double-checked locking: the cache is populated exactly once even when many
        reward/metric methods await this concurrently for the same rollout.
        """
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
        # Safe to drop: later callers short-circuit on the populated cache above.
        self._scoring_locks.pop(trace.id, None)
        return scoring

    async def score(self, trace: vf.Trace, runtime: vf.Runtime | None) -> None:
        """Score the trace, then remove its scratch directory (raw fetch cache +
        materialized corpus files under `RolloutStore.scratch_dir`).

        By the time `Taskset.score` returns, every `@vf.reward`/`@vf.metric` has
        resolved -- including the single cached `_prepared`/`compute_scoring`
        pass, so nothing still needs the on-disk corpus. Cleaning up here (rather
        than relying solely on the `weakref.finalize` safety net registered when
        the directory was created) makes disk cleanup deterministic for the real
        rollout lifecycle, matching how `doc_cache`/materialized corpus files are
        kept off the long-lived `CuratorState` in the first place.
        """
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
    async def cost_penalty(
        self, trace: vf.Trace, runtime: vf.Runtime | None = None
    ) -> float:
        return (
            -self.curator.lambda_cost * (await self._prepared(trace, runtime))["cost"]
        )

    @vf.reward(weight=1.0)
    async def leakage_penalty(
        self, trace: vf.Trace, runtime: vf.Runtime | None = None
    ) -> float:
        return (
            -self.curator.lambda_leakage
            * (await self._prepared(trace, runtime))["leakage"]["overall"]
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
        """Relative val-loss reduction over the neutral baseline (always surfaced,
        zero-weight). Positive => the curated corpus beat the no-information
        baseline; this is the sharpened, scale-anchored read on data quality."""
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
    async def leakage_exact(
        self, trace: vf.Trace, runtime: vf.Runtime | None = None
    ) -> float:
        return (await self._prepared(trace, runtime))["leakage"]["exact"]

    @vf.metric
    async def leakage_fuzzy(
        self, trace: vf.Trace, runtime: vf.Runtime | None = None
    ) -> float:
        return (await self._prepared(trace, runtime))["leakage"]["fuzzy"]

    @vf.metric
    async def leakage_semantic(
        self, trace: vf.Trace, runtime: vf.Runtime | None = None
    ) -> float:
        return (await self._prepared(trace, runtime))["leakage"]["semantic"]

    @vf.metric
    async def cost_total(
        self, trace: vf.Trace, runtime: vf.Runtime | None = None
    ) -> float:
        return (await self._prepared(trace, runtime))["cost"]

    @vf.metric
    async def finalized(self, trace: vf.Trace) -> float:
        return 1.0 if RolloutStore.is_finalized(trace.state) else 0.0

    # Diagnostics that separate "bad curation" from "external/HF/sandbox failure".
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

    async def trainer_error_str(
        self, trace: vf.Trace, runtime: vf.Runtime | None = None
    ) -> str:
        """Log and return the trainer error message, truncated for diagnostics."""
        await self._prepared(trace, runtime)
        err = (RolloutStore.trainer_error(trace.state) or "")[:500]
        if err:
            logger.warning("trainer error: %s", err)
        return err

    @vf.metric
    async def trainer_error_msg(
        self, trace: vf.Trace, runtime: vf.Runtime | None = None
    ) -> float:
        """Whether a trainer error occurred. Zero-weight diagnostic."""
        return 1.0 if await self.trainer_error_str(trace, runtime) else 0.0

    @vf.metric
    async def leakage_reference(
        self, trace: vf.Trace, runtime: vf.Runtime | None = None
    ) -> float:
        await self._prepared(trace, runtime)
        return {"unresolved": 0.0, "stub": 1.0, "real": 2.0, "custom": 3.0}.get(
            str(trace.state.leakage_reference), 0.0
        )


__all__ = [
    "CuratorTaskset",
    "SYSTEM_PROMPT",
    "_ids_from_trace",
    "parse_manifest",
    "extract_json_object",
]
