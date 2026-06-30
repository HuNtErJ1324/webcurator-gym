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
from typing import Any

import verifiers.v1 as vf
from pydantic import Field

from .models import CostLedger, Manifest

# Bump when the on-state layout changes in a way downstream consumers must notice.
STATE_SCHEMA_VERSION = 1


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
    doc_cache: dict[str, list[str]] = Field(default_factory=dict)
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

    # ---- deterministic document cache (keyed by FetchKey.as_str) -------------
    @classmethod
    def doc_cache(cls, state: CuratorState) -> dict[str, list[str]]:
        return state.doc_cache

    @classmethod
    def cached_docs(cls, state: CuratorState, key: str) -> list[str] | None:
        return state.doc_cache.get(key)

    @classmethod
    def store_docs(cls, state: CuratorState, key: str, docs: list[str]) -> None:
        state.doc_cache[key] = list(docs)

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
