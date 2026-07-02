"""Typed per-rollout curation state and its accessors.

Under verifiers v1 the per-rollout shared state is a typed, mutable ``vf.State``
attached to the ``Trace`` (and synced to the tool server via the interception
channel). ``CuratorState`` declares the curation fields; ``RolloutStore`` is the
single place that (de)serializes the manifest and cost ledger, owns the
per-rollout document cache, the external-error telemetry, the state schema
version, and the canonical state hash.

The manifest and cost ledger are stored as plain JSON-able dicts (model dumps)
so the state round-trips cleanly over the v1 state channel; ``RolloutStore``
hands callers validated ``Manifest`` / ``CostLedger`` instances and writes them
back as dumps, exactly as the v0 ``RolloutStore`` did against the dict-state.
"""

from __future__ import annotations

import hashlib
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

# Bump when the on-state layout changes in a way downstream consumers must notice.
# v2: `doc_cache` values changed from full `list[str]` (raw fetched document text,
# held in memory for the rollout's whole lifetime) to a filename string pointing
# into `scratch_dir` (a lazily-created per-rollout temp directory) -- the fix for
# the OOM this schema bump accompanies (see `RolloutStore.scratch_dir`).
STATE_SCHEMA_VERSION = 2


class CuratorState(vf.State):
    """The rollout's shared curation state (manifest, ledger, caches, telemetry).

    Typed and strict (unknown fields rejected), transient (never persisted to
    disk), and shared between the tool server (``self.state``) and scoring
    (``trace.state``). Every field carries a default so the framework can build
    the initial state, mirroring the v0 ``RolloutStore.init`` layout.
    """

    schema_version: int = STATE_SCHEMA_VERSION
    cutoff_date: str | None = None
    manifest: dict[str, Any] = Field(default_factory=lambda: Manifest().model_dump())
    cost_ledger: dict[str, Any] = Field(default_factory=lambda: CostLedger().model_dump())
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


class RolloutStore:
    """Typed accessors over a :class:`CuratorState` (the v1 ``Trace.state``)."""

    @classmethod
    def init(
        cls, state: CuratorState, manifest: Manifest, ledger: CostLedger
    ) -> None:
        """Reset a state to the given manifest/ledger (used by direct unit tests).

        The framework builds ``CuratorState`` with its field defaults; this is the
        explicit equivalent for tests/fixtures that construct a state by hand.
        """
        state.schema_version = STATE_SCHEMA_VERSION
        state.manifest = manifest.model_dump()
        state.cost_ledger = ledger.model_dump()
        state.doc_cache = {}
        state.scratch_dir = None
        state.tool_errors = {}
        state.external_failure = False
        state.manifest_finalized = False
        state.trainer_error = None

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
        path = Path(state.scratch_dir) / filename  # scratch_dir set whenever doc_cache is non-empty
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
    def tool_errors(cls, state: CuratorState) -> dict[str, int]:
        return dict(state.tool_errors)

    @classmethod
    def tool_error_count(cls, state: CuratorState) -> int:
        return sum(int(v) for v in state.tool_errors.values())

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

    # ---- schema version + canonical hash ------------------------------------
    @classmethod
    def schema_version(cls, state: CuratorState) -> int:
        return int(state.schema_version)

    @classmethod
    def canonical_state(cls, state: CuratorState) -> dict[str, Any]:
        """The canonical curation deliverable (manifest + finalize flag) as a
        JSON-able dict — the thing ``canonical_hash`` hashes.

        The schema version is serialized under ``state_schema_version`` (the
        in-state pydantic field stays the unqualified ``schema_version``).
        Transient bookkeeping (cost ledger, telemetry) is excluded,
        so it identifies *what was curated*, not how it was discovered.
        """
        return {
            "state_schema_version": cls.schema_version(state),
            "manifest": cls.manifest(state).model_dump(mode="json"),
            "finalized": cls.is_finalized(state),
        }

    @classmethod
    def canonical_hash(cls, state: CuratorState) -> str:
        """Stable hash of the canonical curation state (see ``canonical_state``).

        Equal curation states hash equal across processes/runs.
        """
        encoded = json.dumps(
            cls.canonical_state(state), sort_keys=True, separators=(",", ":")
        )
        return hashlib.blake2b(encoded.encode("utf-8"), digest_size=16).hexdigest()
