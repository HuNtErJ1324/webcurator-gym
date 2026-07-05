"""Typed per-rollout curation state and its accessors.

Under verifiers v1 the per-rollout shared state is a typed, mutable ``vf.State``
attached to the ``Trace`` (and synced to the tool server via the interception
channel). ``CuratorState`` declares the curation fields; ``RolloutStore`` is the
single place that (de)serializes the manifest and cost ledger, owns the
per-rollout document cache, scratch directory, and external-error telemetry.

The manifest and cost ledger are stored as plain JSON-able dicts (model dumps)
so the state round-trips cleanly over the v1 state channel; ``RolloutStore``
hands callers validated ``Manifest`` / ``CostLedger`` instances and writes them
back as dumps, exactly as the v0 ``RolloutStore`` did against the dict-state.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import uuid
import weakref
from pathlib import Path
from typing import Any

import verifiers.v1 as vf
from pydantic import Field

from .models import CostLedger, Manifest


class CuratorState(vf.State):
    """The rollout's shared curation state (manifest, ledger, caches, telemetry).

    Typed and strict (unknown fields rejected), transient (never persisted to
    disk), and shared between the tool server (``self.state``) and scoring
    (``trace.state``). Every field carries a default so the framework can build
    the initial state.
    """

    cutoff_date: str | None = None
    manifest: dict[str, Any] = Field(default_factory=lambda: Manifest().model_dump())
    cost_ledger: dict[str, Any] = Field(
        default_factory=lambda: CostLedger().model_dump()
    )
    # Cache key -> filename (relative to `scratch_dir`) of that key's raw fetched
    # documents, JSONL-encoded on disk. NOT the documents themselves -- see
    # `RolloutStore.scratch_dir`/`store_docs`/`cached_docs` for why.
    doc_cache: dict[str, str] = Field(default_factory=dict)
    # Lazily-created per-rollout scratch directory backing `doc_cache` and the
    # materialized `CuratedCorpus` produced by `CorpusBuilder.materialize`. `None`
    # until the first fetch/materialize touches disk.
    scratch_dir: str | None = None
    tool_errors: dict[str, int] = Field(default_factory=dict)
    external_failure: bool = False
    manifest_finalized: bool = False
    trainer_error: str | None = None
    budget_fill_ratio: float = 0.0
    source_doc_counts: list[int] = Field(default_factory=list)
    source_token_counts: list[int] = Field(default_factory=list)
    local_source_bytes: int = 0
    local_source_count: int = 0
    local_source_truncated: bool = False
    val_set_access: bool = False


class RolloutStore:
    """Typed accessors over a :class:`CuratorState` (the v1 ``Trace.state``)."""

    @classmethod
    def manifest(cls, state: CuratorState) -> Manifest:
        return Manifest.model_validate(state.manifest or {})

    @classmethod
    def set_manifest(cls, state: CuratorState, manifest: Manifest) -> None:
        state.manifest = manifest.model_dump()

    @classmethod
    def ledger(cls, state: CuratorState) -> CostLedger:
        return CostLedger.model_validate(state.cost_ledger or {})

    @classmethod
    def set_ledger(cls, state: CuratorState, ledger: CostLedger) -> None:
        state.cost_ledger = ledger.model_dump()

    @classmethod
    def set_materialization_stats(
        cls,
        state: CuratorState,
        *,
        budget_fill_ratio: float,
        source_doc_counts: list[int],
        source_token_counts: list[int],
    ) -> None:
        state.budget_fill_ratio = float(budget_fill_ratio)
        state.source_doc_counts = list(source_doc_counts)
        state.source_token_counts = list(source_token_counts)

    @classmethod
    def is_finalized(cls, state: CuratorState) -> bool:
        return bool(state.manifest_finalized)

    @classmethod
    def set_finalized(cls, state: CuratorState, value: bool) -> None:
        state.manifest_finalized = value

    # ---- per-rollout scratch directory (backs doc_cache + materialized corpora) --
    @classmethod
    def scratch_dir(cls, state: CuratorState) -> Path:
        """Get-or-create this rollout's scratch temp directory.

        Lazily created on first use and reused for the rest of the rollout: the
        raw-fetch document cache (`doc_cache`) and `CorpusBuilder.materialize`'s
        per-source filtered document files both live here, so peak host memory
        for a rollout's corpus stays bounded to one source's transient fetch
        rather than growing with every source/document fetched over the
        rollout's lifetime (see `cleanup`).

        `CuratorState` has no framework-level per-rollout teardown hook (it is
        toolless and `Taskset` exposes no `@vf.cleanup`), so a `weakref.finalize`
        safety net is registered against `state` itself the first time this
        directory is created -- best-effort cleanup for any caller (e.g. a
        direct `fetch_source_docs`/`materialize` call in a test) that never
        routes through `CuratorTaskset.score`, which calls `cleanup` explicitly
        and deterministically for the real rollout lifecycle.
        """
        if state.scratch_dir is None:
            path = Path(tempfile.mkdtemp(prefix="pdc_rollout_"))
            state.scratch_dir = str(path)
            weakref.finalize(state, shutil.rmtree, str(path), ignore_errors=True)
        return Path(state.scratch_dir)

    @classmethod
    def cleanup(cls, state: CuratorState) -> None:
        """Deterministically remove this rollout's scratch directory, if any.

        Called by `CuratorTaskset.score` once all rewards/metrics for the trace
        have resolved (so nothing still needs the cached/materialized files).
        Idempotent -- safe to call even if no scratch directory was ever created.
        """
        if state.scratch_dir is not None:
            shutil.rmtree(state.scratch_dir, ignore_errors=True)
            state.scratch_dir = None
        state.doc_cache = {}

    # ---- deterministic document cache (keyed by FetchKey.as_str) -------------
    @classmethod
    def cached_docs(cls, state: CuratorState, key: str) -> list[str] | None:
        """Return the cached raw docs for `key`, streamed back off disk, or
        `None` on a cache miss."""
        filename = state.doc_cache.get(key)
        if filename is None:
            return None
        path = (
            Path(state.scratch_dir) / filename
        )  # scratch_dir set whenever doc_cache is non-empty
        with path.open("r", encoding="utf-8") as fh:
            return [json.loads(line) for line in fh]

    @classmethod
    def store_docs(cls, state: CuratorState, key: str, docs: list[str]) -> None:
        """Persist `docs` to a scratch file and record its filename in `doc_cache`.

        Storing a filename (not the documents) is what keeps `state` itself
        cheap regardless of how many/large the fetched documents are -- the
        actual text only round-trips through disk, never accumulating in the
        long-lived `CuratorState` across sources or repeated fetches.
        """
        directory = cls.scratch_dir(state)
        filename = f"raw_{uuid.uuid4().hex}.jsonl"
        with (directory / filename).open("w", encoding="utf-8") as fh:
            for doc in docs:
                fh.write(json.dumps(doc))
                fh.write("\n")
        state.doc_cache[key] = filename

    # ---- external-failure telemetry -----------------------------------------
    @classmethod
    def record_tool_error(cls, state: CuratorState, kind: str) -> None:
        state.tool_errors[kind] = int(state.tool_errors.get(kind, 0)) + 1

    @classmethod
    def tool_error_count(cls, state: CuratorState) -> int:
        return sum(int(v) for v in state.tool_errors.values())

    # ---- local-source provenance and validation-set access telemetry ---------
    @classmethod
    def add_local_source(
        cls, state: CuratorState, *, bytes_pulled: int, truncated: bool
    ) -> None:
        state.local_source_count += 1
        state.local_source_bytes += int(bytes_pulled)
        state.local_source_truncated = state.local_source_truncated or bool(truncated)

    @classmethod
    def local_source_count(cls, state: CuratorState) -> int:
        return int(state.local_source_count)

    @classmethod
    def local_source_bytes(cls, state: CuratorState) -> int:
        return int(state.local_source_bytes)

    @classmethod
    def local_source_truncated(cls, state: CuratorState) -> bool:
        return bool(state.local_source_truncated)

    @classmethod
    def set_val_set_access(cls, state: CuratorState, value: bool) -> None:
        state.val_set_access = bool(value)

    @classmethod
    def val_set_access(cls, state: CuratorState) -> bool:
        return bool(state.val_set_access)

    @classmethod
    def set_external_failure(cls, state: CuratorState, value: bool = True) -> None:
        state.external_failure = bool(value)

    @classmethod
    def has_external_failure(cls, state: CuratorState) -> bool:
        return bool(state.external_failure)

    @classmethod
    def set_trainer_error(cls, state: CuratorState, message: str) -> None:
        state.trainer_error = message

    @classmethod
    def trainer_error(cls, state: CuratorState) -> str | None:
        return state.trainer_error
