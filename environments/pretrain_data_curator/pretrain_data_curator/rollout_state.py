"""Typed accessors for curation rollout state.

The manifest and cost ledger are stored as plain dicts so the rollout state stays
JSON-serializable (verifiers asserts this); this class is the single place that
(de)serializes them. It also owns the per-rollout document cache, external-error
telemetry, the state schema version, and a canonical state hash.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from .models import CostLedger, Manifest

# Bump when the on-state layout changes in a way downstream consumers must notice.
STATE_SCHEMA_VERSION = 1


class RolloutStore:
    MANIFEST = "manifest"
    LEDGER = "cost_ledger"
    CANDIDATES = "candidates"
    INSPECTED = "inspected"
    FINALIZED = "manifest_finalized"
    SCORING = "curation_scoring"
    DOC_CACHE = "doc_cache"
    TOOL_ERRORS = "tool_errors"
    EXTERNAL_FAILURE = "external_failure"
    TRAINER_ERROR = "trainer_error"
    SCHEMA_VERSION = "state_schema_version"

    @classmethod
    def init(cls, state: dict[str, Any], manifest: Manifest, ledger: CostLedger) -> None:
        state[cls.SCHEMA_VERSION] = STATE_SCHEMA_VERSION
        state[cls.MANIFEST] = manifest.model_dump()
        state[cls.LEDGER] = ledger.model_dump()
        state[cls.CANDIDATES] = {}
        state[cls.INSPECTED] = {}
        state[cls.DOC_CACHE] = {}
        state[cls.TOOL_ERRORS] = {}
        state[cls.EXTERNAL_FAILURE] = False
        state[cls.FINALIZED] = False
        state.pop(cls.SCORING, None)
        state.pop(cls.TRAINER_ERROR, None)

    @classmethod
    def manifest(cls, state: dict[str, Any]) -> Manifest:
        return Manifest.model_validate(state.get(cls.MANIFEST) or {})

    @classmethod
    def set_manifest(cls, state: dict[str, Any], manifest: Manifest) -> None:
        state[cls.MANIFEST] = manifest.model_dump()

    @classmethod
    def ledger(cls, state: dict[str, Any]) -> CostLedger:
        return CostLedger.model_validate(state.get(cls.LEDGER) or {})

    @classmethod
    def set_ledger(cls, state: dict[str, Any], ledger: CostLedger) -> None:
        state[cls.LEDGER] = ledger.model_dump()

    @classmethod
    def candidates(cls, state: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return state.setdefault(cls.CANDIDATES, {})

    @classmethod
    def inspected(cls, state: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return state.setdefault(cls.INSPECTED, {})

    @classmethod
    def is_finalized(cls, state: dict[str, Any]) -> bool:
        return bool(state.get(cls.FINALIZED))

    @classmethod
    def set_finalized(cls, state: dict[str, Any], value: bool) -> None:
        state[cls.FINALIZED] = value

    # ---- deterministic document cache (keyed by FetchKey.as_str) -------------
    @classmethod
    def doc_cache(cls, state: dict[str, Any]) -> dict[str, list[str]]:
        return state.setdefault(cls.DOC_CACHE, {})

    @classmethod
    def cached_docs(cls, state: dict[str, Any], key: str) -> list[str] | None:
        return cls.doc_cache(state).get(key)

    @classmethod
    def store_docs(cls, state: dict[str, Any], key: str, docs: list[str]) -> None:
        cls.doc_cache(state)[key] = list(docs)

    # ---- external-failure telemetry -----------------------------------------
    @classmethod
    def record_tool_error(cls, state: dict[str, Any], kind: str) -> None:
        errors = state.setdefault(cls.TOOL_ERRORS, {})
        errors[kind] = int(errors.get(kind, 0)) + 1

    @classmethod
    def tool_errors(cls, state: dict[str, Any]) -> dict[str, int]:
        return dict(state.get(cls.TOOL_ERRORS) or {})

    @classmethod
    def tool_error_count(cls, state: dict[str, Any]) -> int:
        return sum(int(v) for v in (state.get(cls.TOOL_ERRORS) or {}).values())

    @classmethod
    def set_external_failure(cls, state: dict[str, Any], value: bool = True) -> None:
        state[cls.EXTERNAL_FAILURE] = bool(value)

    @classmethod
    def has_external_failure(cls, state: dict[str, Any]) -> bool:
        return bool(state.get(cls.EXTERNAL_FAILURE))

    @classmethod
    def set_trainer_error(cls, state: dict[str, Any], message: str) -> None:
        state[cls.TRAINER_ERROR] = message

    @classmethod
    def trainer_error(cls, state: dict[str, Any]) -> str | None:
        return state.get(cls.TRAINER_ERROR)

    # ---- schema version + canonical hash ------------------------------------
    @classmethod
    def schema_version(cls, state: dict[str, Any]) -> int:
        return int(state.get(cls.SCHEMA_VERSION, STATE_SCHEMA_VERSION))

    @classmethod
    def canonical_hash(cls, state: dict[str, Any]) -> str:
        """Stable hash of the curation deliverable (manifest + finalize flag).

        Equal curation state hashes equal across processes/runs; transient
        bookkeeping (cost ledger, candidate cache, telemetry) is excluded so the
        hash identifies *what was curated*, not how it was discovered.
        """
        payload = {
            "schema_version": cls.schema_version(state),
            "manifest": cls.manifest(state).model_dump(mode="json"),
            "finalized": cls.is_finalized(state),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.blake2b(encoded.encode("utf-8"), digest_size=16).hexdigest()
