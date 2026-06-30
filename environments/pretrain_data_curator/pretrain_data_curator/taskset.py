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
and used to train the fixed proxy student, and the composite reward is preserved
1:1 from v0:

    R(M, H) = a1*Perf + a2*Quality + a3*Diversity - l1*Cost - l2*Leakage

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
import re
from typing import Any

import verifiers.v1 as vf
from pydantic import ValidationError

from .corpus import CorpusBuilder
from .eval_corpus import DEFAULT_EVAL_CORPUS
from .hf_access import HuggingFaceDatasetClient, RetryPolicy
from .hf_meter import _content_text, extract_hf_commands, install_shim, meter_ledger
from .leakage import LeakageDetector
from .models import CuratorConfig, FilterSpec, Manifest, ProxyStudentConfig, Sampling, Source
from .rewards import CuratorScorer
from .rollout_state import CuratorState, RolloutStore
from .tasks import CuratorTask, build_tasks
from .trainer import HeuristicProxyTrainer, ProxyStudentTrainer, SandboxProxyTrainer
from .val_set import ValidationSetConfig, ValTokenLoader

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """IMPORTANT: Be extremely concise in every message. Your first response MUST be a bash command — no preamble, no plan, no explanation. Just run the hf CLI. If you want to plan, do it in one sentence max, then immediately run a command.

You are a pretraining-data curation agent. Your job is to assemble a dataset mixture that, when used to train a fixed small GPT-2-scale student (everything fixed but the data), maximizes the student's performance.

Domain context — target large-scale, diverse, high-quality text corpora for general-purpose LLM pretraining. Good sources include encyclopedic text (Wikipedia, encyclopedias), scientific literature (papers, research), instructional text, and broad web corpora (C4, FineWeb, OpenWebText). Prefer encyclopedic sources (highest utility), then scientific, then instructional. Well-known high-quality datasets: Wikimedia/Wikipedia, allenai/c4, Skylion007/openwebtext, HuggingFaceFW/fineweb, allenai/dolma. Avoid code-only or narrow task-specific datasets. For each candidate, retrieve key metadata: downloads, likes, last modified, splits, configs. Use result limits appropriate to the turn budget. Avoid repetition: if you already have detailed info for a dataset, move on.

You have a normal bash shell with the Hugging Face `hf` CLI ALREADY INSTALLED and ready to use — it is the ONLY tool you need. Use it to discover and inspect candidate datasets, then decide a weighted curation mixture. Only use Hugging Face datasets modified on or before the cutoff date.

Do NOT install anything and do NOT write Python. Run the `hf` subcommands below directly in bash. There is no need to run `pip` / `pip install`, create a virtualenv, or write a `python`/`python3` script to import `huggingface_hub` or `datasets` — those imports do NOT work in this shell and pip will only waste your turns. The `hf` CLI already does everything required; reach for it, not pip/python.

`hf` command cheat-sheet (no setup required):
  - Search datasets:   hf datasets ls --search "<query>" --sort downloads --limit 10
  - Filter / quiet:    hf datasets ls --search "<query>" --filter text --limit 20 -q
  - JSON output:        hf datasets ls --search "<query>" --format json --expand downloads,likes,lastModified,tags
  - Inspect a dataset: hf datasets info <dataset_id> --expand downloads,likes,tags
Inspect a few candidates, prefer well-downloaded, clearly-licensed text datasets, and check each was last modified on or before the cutoff date.

You are billed for live discovery: each search and each inspect/download call adds to the cost penalty, so be economical. Your reward is derived from proxy-student held-out cross-entropy loss, with penalties for cost and for leakage/contamination against a held-out evaluation set.

`"text_field": null` triggers auto-detection (the environment tries 'text', 'content', 'passage', 'abstract', 'query'/'response' concat, etc.). Set explicitly only if you know the exact column from `hf datasets info`.

Reliable starter datasets: HuggingFaceFW/fineweb (text_field: "text"), roneneldan/TinyStories (text_field: "text"), wikimedia/wikipedia (config: "20231101.en", text_field: "text"), allenai/c4 (config: "en", text_field: "text"), Salesforce/wikitext (config: "wikitext-103-v1", text_field: "text").

When you are done, emit your decision as your FINAL message: a single fenced ```json block (and nothing else after it) with this exact schema:

```json
{
  "token_budget": 1000000,
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

Each source REQUIRES `id` (the Hugging Face dataset id) and `weight` (>= 0, relative mixing weight). `config`, `split`, `text_field`, `filters`, `max_docs`, and `max_tokens` are optional. Supported filter kinds: min_chars, max_chars, min_tokens, max_symbol_ratio, min_alpha_ratio, drop_regex, keep_regex, dedup_exact. Always emit a non-empty `sources` list — an empty or missing manifest scores zero."""


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
        "http", "https", "hf", "file", "s3", "gs", "az",
        "usr", "var", "etc", "bin", "tmp", "opt", "home", "root",
        "datasets", "models", "spaces", "api", "v1", "v2",
    }
)
_NOT_HF_NAMES = frozenset({"train", "test", "validation", "valid", "dev", "split"})
_FILE_EXTS = frozenset(
    {"py", "json", "jsonl", "txt", "csv", "yaml", "yml", "toml", "sh",
     "md", "rst", "log", "parquet", "arrow", "gz", "zip"}
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

    Two sources, in order of reliability:
    1. ``hf datasets info <id>`` command arguments — the agent explicitly
       inspected these ids, so they are definitive.
    2. Free-form text in bash tool-result messages — covers ids that appeared
       in ``hf datasets ls`` output but were never individually inspected.

    Returns ids in first-observation order, deduplicated.  Used as the last-
    resort fallback when no manifest text was emitted.
    """
    seen: dict[str, None] = {}  # ordered-set via insertion-order dict

    # 1. Explicitly-inspected ids from assistant tool-call argument JSON.
    for msg in trace.assistant_messages:
        for tc in msg.tool_calls or []:
            try:
                cmd = json.loads(getattr(tc, "arguments", "") or "{}").get("command", "")
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
                    seen[argv[2]] = None

    # 2. Ids from bash tool-result text (hf datasets ls output, info summaries).
    for msg in getattr(trace, "tool_messages", []):
        text = _content_text(getattr(msg, "content", ""))
        for m in _HF_ID_RE.finditer(text):
            did = m.group(1)
            if _looks_like_hf_id(did):
                seen.setdefault(did, None)

    return list(seen.keys())


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
        params = f.get("params")
        specs.append(FilterSpec(kind=kind, params=dict(params) if isinstance(params, dict) else {}))
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

    kwargs["sampling"] = Sampling(max_docs=_pos_int(max_docs), max_tokens=_pos_int(max_tokens))
    try:
        return Source(**kwargs)
    except ValidationError:
        return None


def parse_manifest(text: str, default_token_budget: int | None = None) -> Manifest | None:
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
        except (TypeError, ValueError):
            pass
    try:
        return Manifest(token_budget=token_budget, sources=sources)
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
    max_turns: int = 12
    alpha_perf: float = 1.0
    lambda_cost: float = 0.1
    lambda_leakage: float = 1.0
    # Baseline-relative Perf signal (additive; default-OFF preserves calibration).
    perf_baseline_loss: float = math.log(50304)
    baseline_relative_perf: bool = False
    max_concurrent_fetches: int = 8
    max_concurrent_training: int = 1
    fetch_timeout_seconds: float = 30.0
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
            max_turns=config.max_turns,
            alpha_perf=config.alpha_perf,
            lambda_cost=config.lambda_cost,
            lambda_leakage=config.lambda_leakage,
            perf_baseline_loss=config.perf_baseline_loss,
            baseline_relative_perf=config.baseline_relative_perf,
            max_concurrent_fetches=config.max_concurrent_fetches,
            max_concurrent_training=config.max_concurrent_training,
            fetch_timeout_seconds=config.fetch_timeout_seconds,
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
        )

    def _ensure(self) -> CuratorScorer:
        """Build the scoring collaborators on first use (honoring injected ones)."""
        if self._scorer is not None:
            return self._scorer
        if self._client is None:
            self._client = HuggingFaceDatasetClient(
                token_env=self.config.hf_token_env
            )
        if self._corpus_builder is None:
            self._corpus_builder = CorpusBuilder(
                client=self._client,
                sample_docs_per_source=self.curator.sample_docs_per_source,
                retry_policy=self._fetch_policy(),
                fetch_limit=self.curator.max_concurrent_fetches,
            )
        if self._leakage_detector is None:
            self._leakage_detector = LeakageDetector(
                self.config.eval_corpus or DEFAULT_EVAL_CORPUS
            )
        if self._val_loader is None:
            self._val_loader = ValTokenLoader(
                self.curator.validation_set,
                retry_policy=self._fetch_policy(),
                fetch_limit=self.curator.max_concurrent_fetches,
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
        )
        return self._scorer

    def _build_real_trainer(self) -> ProxyStudentTrainer:
        """Construct the real (GPU) proxy-student trainer for the selected backend.

        ``trainer_backend='prime'`` (default) builds the trainer EXACTLY as before
        — no factories, so ``SandboxProxyTrainer`` imports ``prime_sandboxes`` and
        provisions a Prime GPU sandbox. ``'docker'`` injects ``client_factory`` /
        ``request_factory`` that drive verifiers' v1 ``DockerRuntime`` against a
        (typically remote) Docker daemon, leaving the trainer lifecycle untouched.
        ``'modal'`` builds a ``ModalProxyTrainer`` that calls
        ``modal.Sandbox.create``; works from a CPU-only env-server in Hosted
        Training without a local Docker daemon or Prime account.
        """
        ps = self.curator.proxy_student

        if ps.trainer_backend == "modal":
            from .modal_backend import ModalProxyTrainer

            return ModalProxyTrainer(
                concurrency_limit=self.curator.max_concurrent_training,
                val_loader=self._val_loader,
            )

        if ps.trainer_backend == "docker":
            # Lazy import: the docker runtime is only needed on this path.
            from .docker_backend import DockerRunRequest, DockerRuntimeClient

            docker_host = ps.docker_host

            def request_factory(cfg: ProxyStudentConfig, name: str) -> DockerRunRequest:
                return DockerRunRequest(
                    name=name,
                    image=cfg.docker_image,
                    workdir="/workspace",
                    # Docker maps gpu_count -> ``--gpus N`` and ignores gpu_type.
                    gpu=str(cfg.gpu_count) if cfg.gpu_count > 0 else None,
                    cpu=float(cfg.cpu_cores),
                    memory=float(cfg.memory_gb),
                    disk=float(cfg.disk_size_gb),
                    timeout_minutes=cfg.effective_timeout_minutes,
                )

            return SandboxProxyTrainer(
                concurrency_limit=self.curator.max_concurrent_training,
                val_loader=self._val_loader,
                client_factory=lambda: DockerRuntimeClient(docker_host=docker_host),
                request_factory=request_factory,
            )

        # prime backend (default): unchanged — no factories => prime_sandboxes.
        return SandboxProxyTrainer(
            concurrency_limit=self.curator.max_concurrent_training,
            val_loader=self._val_loader,
        )

    # -- taskset surface -------------------------------------------------------

    def load_tasks(self) -> list[CuratorTask]:
        tasks = build_tasks(self.curator.cutoff_date, self.curator.token_budget)
        system_prompt = self._system_prompt()
        return [t.model_copy(update={"system_prompt": system_prompt}) for t in tasks]

    def _system_prompt(self) -> str:
        """Render the configured turn budget into the per-rollout prompt.

        Spells out the actual ``max_turns`` and strictly limits discovery so the
        agent commits a manifest before the turn cap.
        """
        max_turns = self.curator.max_turns
        discovery_rounds = max(2, min(12, max_turns // 6, self.curator.scan_limit // 10))
        discovery_calls = discovery_rounds * 2
        commit_by = max(1, max_turns - max(3, max_turns // 8))
        return (
            f"{SYSTEM_PROMPT}\n\n"
            f"You have {max_turns} turns total. Each bash tool call uses one turn. "
            f"A discovery round = one `hf datasets ls` call + one `hf datasets info` "
            f"call (2 turns). You MUST perform at most {discovery_rounds} discovery "
            f"rounds (<={discovery_calls} bash calls). After your final discovery "
            f"round, and no later than turn "
            f"{commit_by} — you MUST commit your manifest.\n\n"
            f"HOW TO COMMIT: send a plain text response (NO bash tool call) containing "
            f"ONLY the fenced ```json block. Do not call bash to print the manifest. "
            f"Do not add any text after the closing ``` fence. If you still have bash "
            f"calls available but you have enough evidence to pick sources, commit "
            f"immediately — do not fill remaining turns with more searches.\n\n"
            f"CRITICAL — no invented sources: every `id` in your manifest MUST be "
            f"copied verbatim from a dataset id that appeared in your `hf datasets ls` "
            f"or `hf datasets info` output during this rollout. Do NOT invent or guess "
            f"dataset ids or config names. If a `config` was not explicitly listed in "
            f"tool output, set `config` to null. A manifest with a fabricated id or "
            f"config materializes zero tokens and scores zero. An empty or missing "
            f"manifest also scores zero."
        )

    # NOTE: deliberately no ``tools(...)`` override — the agent curates via the
    # `hf` CLI in its own shell, so the taskset exposes no MCP tool servers and
    # ``type(CuratorTaskset).tools is Taskset.tools`` holds. This is what lets the
    # non-MCP gate (env.py:239-247) pass for codex / kimi_code / bash harnesses.

    @vf.stop
    async def max_turns_reached(self, trace: vf.Trace) -> bool:
        """Cap the rollout at ``max_turns`` model turns (the harness also stops
        naturally once the model emits a final answer with no tool calls)."""
        return trace.num_turns >= self.curator.max_turns

    # -- finalize (runs before scoring, while the runtime is live) -------------

    @staticmethod
    def _final_message_text(trace: vf.Trace) -> str:
        msgs = trace.assistant_messages
        if not msgs:
            return ""
        return msgs[-1].content or ""

    def _finalize_manifest(self, task: CuratorTask, trace: vf.Trace) -> Manifest | None:
        """Return the most recent usable manifest across assistant turns.

        Three-tier lookup — each tier is tried only if the previous produced nothing:

        1. **Parse** a fenced JSON manifest from any assistant text message (newest
           first).  This is the primary path and is unchanged.
        2. **State candidates** — synthesize from ``RolloutStore.candidates(state)``
           when populated (MCP-tool harnesses).
        3. **Trace fallback** — synthesize from dataset ids actually observed in the
           rollout's bash tool calls / outputs (bash harness).  Uses only ids the
           agent genuinely discovered; never invents ids.  ``config`` is set to
           ``null`` for all sources unless a config was explicitly shown in tool
           output (which is not parsed here — configless is the safe default).
        """
        # Tier 1: parse manifest from assistant message text.
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

        # Tier 2: synthesize from the rollout state's candidate pool (MCP harnesses).
        state = trace.state
        candidates = RolloutStore.candidates(state)
        if candidates:
            limit = self.curator.candidate_limit
            sources = [
                Source(dataset_id=did, config=None, weight=1.0)
                for did in list(candidates)[:limit]
            ]
            if sources:
                return Manifest(token_budget=task.token_budget, sources=sources)

        # Tier 3: synthesize from ids observed in bash tool calls / outputs.
        observed = _ids_from_trace(trace)
        if observed:
            limit = self.curator.candidate_limit
            sources = [
                Source(dataset_id=did, config=None, weight=1.0)
                for did in observed[:limit]
            ]
            return Manifest(token_budget=task.token_budget, sources=sources)

        return None

    async def finalize(self, task: CuratorTask, trace: vf.Trace, runtime: Any) -> None:
        """Parse the agent's manifest and meter live `hf` discovery cost.

        Positional signature as invoked at ``verifiers/v1/rollout.py:241``; runs
        after generation and before ``score``. Assistant turns are scanned most
        recent first, so the final-message behavior is preserved while a manifest
        emitted before trailing `hf` calls still finalizes at the turn cap. Tolerant
        of a missing/garbled rollout (no manifest anywhere -> "not finalized" -> the
        scorer returns the zero sentinel). The discovery cost ledger is computed here
        and persisted; the scorer still adds ``train_flops`` and the materialization
        cost on top.
        """
        state = trace.state
        manifest = self._finalize_manifest(task, trace)
        if manifest is not None and manifest.sources:
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

    async def _prepared(self, trace: vf.Trace) -> dict[str, Any]:
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
            scoring = await scorer.compute_scoring(trace.state)
            self._scoring_cache[trace.id] = scoring
        # Safe to drop: later callers short-circuit on the populated cache above.
        self._scoring_locks.pop(trace.id, None)
        return scoring

    # -- rewards (weighted contributions; coefficients folded in) --------------

    @vf.reward(weight=1.0)
    async def perf_reward(self, trace: vf.Trace) -> float:
        return self.curator.alpha_perf * (await self._prepared(trace))["perf"]

    @vf.reward(weight=1.0)
    async def cost_penalty(self, trace: vf.Trace) -> float:
        return -self.curator.lambda_cost * (await self._prepared(trace))["cost"]

    @vf.reward(weight=1.0)
    async def leakage_penalty(self, trace: vf.Trace) -> float:
        return -self.curator.lambda_leakage * (await self._prepared(trace))["leakage"]["overall"]

    # -- zero-weight diagnostic metrics (recorded, not summed into reward) -----

    @vf.metric
    async def perf_loss(self, trace: vf.Trace) -> float:
        return (await self._prepared(trace))["loss"]

    @vf.metric
    async def perf_accuracy(self, trace: vf.Trace) -> float:
        return (await self._prepared(trace))["accuracy"]

    @vf.metric
    async def perf_vs_baseline(self, trace: vf.Trace) -> float:
        """Relative val-loss reduction over the neutral baseline (always surfaced,
        zero-weight). Positive => the curated corpus beat the no-information
        baseline; this is the sharpened, scale-anchored read on data quality."""
        return (await self._prepared(trace))["perf_vs_baseline"]

    @vf.metric
    async def train_flops(self, trace: vf.Trace) -> float:
        return (await self._prepared(trace))["flops"]

    @vf.metric
    async def corpus_tokens(self, trace: vf.Trace) -> float:
        return float((await self._prepared(trace))["tokens"])

    @vf.metric
    async def num_sources(self, trace: vf.Trace) -> float:
        return float((await self._prepared(trace))["num_sources"])

    @vf.metric
    async def leakage_exact(self, trace: vf.Trace) -> float:
        return (await self._prepared(trace))["leakage"]["exact"]

    @vf.metric
    async def leakage_fuzzy(self, trace: vf.Trace) -> float:
        return (await self._prepared(trace))["leakage"]["fuzzy"]

    @vf.metric
    async def leakage_semantic(self, trace: vf.Trace) -> float:
        return (await self._prepared(trace))["leakage"]["semantic"]

    @vf.metric
    async def cost_total(self, trace: vf.Trace) -> float:
        return (await self._prepared(trace))["cost"]

    @vf.metric
    async def finalized(self, trace: vf.Trace) -> float:
        return 1.0 if RolloutStore.is_finalized(trace.state) else 0.0

    # Diagnostics that separate "bad curation" from "external/HF/sandbox failure".
    @vf.metric
    async def tool_error_count(self, trace: vf.Trace) -> float:
        return float(RolloutStore.tool_error_count(trace.state))

    @vf.metric
    async def external_failure(self, trace: vf.Trace) -> float:
        return 1.0 if RolloutStore.has_external_failure(trace.state) else 0.0

    async def trainer_error_str(self, trace: vf.Trace) -> str:
        """Log and return the trainer error message, truncated for diagnostics."""
        await self._prepared(trace)
        err = (RolloutStore.trainer_error(trace.state) or "")[:500]
        if err:
            logger.warning("trainer error: %s", err)
        return err

    @vf.metric
    async def trainer_error_msg(self, trace: vf.Trace) -> float:
        """Whether a trainer error occurred. Zero-weight diagnostic."""
        return 1.0 if await self.trainer_error_str(trace) else 0.0


__all__ = ["CuratorTaskset", "SYSTEM_PROMPT", "_ids_from_trace", "parse_manifest", "extract_json_object"]
