"""Typed, per-rollout state for corpus curation.

The state is the only mutable rollout-owned object.  Domain code talks to it
directly; there is no parallel store/facade to keep in sync with Verifiers.
Large document payloads live in a rollout scratch directory and only filenames
cross the state boundary.
"""

from __future__ import annotations

import shutil
import tempfile
import uuid
import weakref
from pathlib import Path
from typing import Any

import orjson

import verifiers.v1 as vf
from pydantic import Field, PrivateAttr

from .utils.models import (
    MANIFEST_PROVENANCE_MISSING,
    Manifest,
    ManifestProvenance,
    ScoringResult,
)


class CuratorState(vf.State):
    """All mutable state associated with one curation rollout."""

    cutoff_date: str | None = None
    manifest: dict[str, Any] = Field(default_factory=lambda: Manifest().model_dump())
    manifest_finalized: bool = False
    manifest_provenance: ManifestProvenance = MANIFEST_PROVENANCE_MISSING

    doc_cache: dict[str, str] = Field(default_factory=dict)
    scratch_dir: str | None = None

    tool_errors: dict[str, int] = Field(default_factory=dict)
    external_failure: bool = False
    trainer_error: str | None = None

    budget_fill_ratio: float = 0.0
    source_doc_counts: list[int] = Field(default_factory=list)
    source_token_counts: list[int] = Field(default_factory=list)
    local_source_bytes: int = 0
    local_source_count: int = 0
    local_source_truncated: bool = False
    val_set_access: bool = False

    self_score_runs: int = 0
    self_score_ok_runs: int = 0
    self_score_first_reward: float | None = None
    self_score_best_reward: float | None = None
    self_score_last_reward: float | None = None
    self_score_history: list[dict[str, Any]] = Field(default_factory=list)
    hf_cli_history: list[dict[str, Any]] = Field(default_factory=list)

    # Task.score runs @metric methods before @reward methods. Keep the one
    # expensive scoring result on rollout-owned state so both v1 primitives use
    # it without materializing/training twice.
    _scoring_result: ScoringResult | None = PrivateAttr(default=None)
    _turn_runtime: vf.Runtime | None = PrivateAttr(default=None)

    @property
    def parsed_manifest(self) -> Manifest:
        return Manifest.model_validate(self.manifest or {})

    def set_manifest(self, manifest: Manifest, *, finalized: bool) -> None:
        self.manifest = manifest.model_dump()
        self.manifest_finalized = finalized

    def set_materialization_stats(
        self,
        *,
        budget_fill_ratio: float,
        source_doc_counts: list[int],
        source_token_counts: list[int],
    ) -> None:
        self.budget_fill_ratio = float(budget_fill_ratio)
        self.source_doc_counts = list(source_doc_counts)
        self.source_token_counts = list(source_token_counts)

    def record_error(self, kind: str, *, external: bool = True) -> None:
        self.tool_errors[kind] = self.tool_errors.get(kind, 0) + 1
        self.external_failure = self.external_failure or external

    @property
    def tool_error_count(self) -> int:
        return sum(self.tool_errors.values())

    def record_local_source(self, *, bytes_pulled: int, truncated: bool) -> None:
        self.local_source_count += 1
        self.local_source_bytes += int(bytes_pulled)
        self.local_source_truncated = self.local_source_truncated or truncated

    def set_self_score_summary(
        self, *, runs: int, rewards: list[float], history: list[dict[str, Any]]
    ) -> None:
        self.self_score_runs = int(runs)
        self.self_score_ok_runs = len(rewards)
        self.self_score_first_reward = rewards[0] if rewards else None
        self.self_score_best_reward = max(rewards) if rewards else None
        self.self_score_last_reward = rewards[-1] if rewards else None
        self.self_score_history = list(history)

    def set_hf_cli_history(self, history: list[dict[str, Any]]) -> None:
        self.hf_cli_history = list(history)

    def workspace(self) -> Path:
        """Return the lazily-created scratch directory for this rollout."""
        if self.scratch_dir is None:
            path = Path(tempfile.mkdtemp(prefix="pretrain_curation_"))
            self.scratch_dir = str(path)
            weakref.finalize(self, shutil.rmtree, str(path), ignore_errors=True)
        return Path(self.scratch_dir)

    def cached_documents(self, key: str) -> list[str] | None:
        """Blocking read of a cached fetch; callers offload it via to_thread."""
        filename = self.doc_cache.get(key)
        if filename is None or self.scratch_dir is None:
            return None
        with (Path(self.scratch_dir) / filename).open("rb") as file:
            return [orjson.loads(line) for line in file]

    def cache_documents(self, key: str, documents: list[str]) -> None:
        filename = f"raw_{uuid.uuid4().hex}.jsonl"
        with (self.workspace() / filename).open("wb") as file:
            for document in documents:
                file.write(orjson.dumps(document))
                file.write(b"\n")
        self.doc_cache[key] = filename

    def cleanup(self) -> None:
        """Idempotently remove rollout-owned files after scoring."""
        if self.scratch_dir is not None:
            shutil.rmtree(self.scratch_dir, ignore_errors=True)
        self.scratch_dir = None
        self._turn_runtime = None
        self.doc_cache = {}


__all__ = ["CuratorState"]
