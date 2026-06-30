"""Materialize a curated corpus from a manifest.

`CorpusBuilder` turns a `Manifest` into concrete documents by sampling each
source through the `DatasetSearchClient`, applying its filters, and honoring
per-source sampling caps. `DocumentFilter` owns the supported filter kinds.
"""

from __future__ import annotations

import asyncio
import re
import weakref
from dataclasses import dataclass
from typing import Any

from .hf_access import (
    DatasetAccessError,
    DatasetSearchClient,
    FetchKey,
    RetryPolicy,
    estimate_tokens,
    fetch_documents,
    hf_fetch_semaphore,
)
from .models import FilterSpec, Manifest, Source
from .rollout_state import CuratorState, RolloutStore


@dataclass
class SourceCorpus:
    dataset_id: str
    config: str | None
    weight: float
    documents: list[str]

    @property
    def tokens(self) -> int:
        return sum(estimate_tokens(doc) for doc in self.documents)


@dataclass
class CuratedCorpus:
    sources: list[SourceCorpus]

    @property
    def documents(self) -> list[str]:
        return [doc for source in self.sources for doc in source.documents]

    @property
    def total_tokens(self) -> int:
        return sum(source.tokens for source in self.sources)

    def is_empty(self) -> bool:
        return all(not source.documents for source in self.sources)


class DocumentFilter:
    """Applies an ordered list of `FilterSpec` to a list of documents."""

    def apply(self, docs: list[str], filters: list[FilterSpec]) -> list[str]:
        result = list(docs)
        for spec in filters:
            result = self._apply_one(result, spec)
        return result

    def _apply_one(self, docs: list[str], spec: FilterSpec) -> list[str]:
        kind = spec.kind
        params = spec.params
        if kind == "min_chars":
            threshold = int(params.get("value", 0))
            return [d for d in docs if len(d) >= threshold]
        if kind == "max_chars":
            threshold = int(params.get("value", 10**9))
            return [d for d in docs if len(d) <= threshold]
        if kind == "min_tokens":
            threshold = int(params.get("value", 0))
            return [d for d in docs if estimate_tokens(d) >= threshold]
        if kind == "max_symbol_ratio":
            threshold = float(params.get("value", 1.0))
            return [d for d in docs if _symbol_ratio(d) <= threshold]
        if kind == "min_alpha_ratio":
            threshold = float(params.get("value", 0.0))
            return [d for d in docs if _alpha_ratio(d) >= threshold]
        if kind == "drop_regex":
            pattern = re.compile(str(params.get("pattern", "")))
            return [d for d in docs if not pattern.search(d)]
        if kind == "keep_regex":
            pattern = re.compile(str(params.get("pattern", "")))
            return [d for d in docs if pattern.search(d)]
        if kind == "dedup_exact":
            return _dedup_exact(docs)
        # Unknown filter kinds are ignored rather than raising, so an agent's
        # experimentation never hard-fails the rollout.
        return docs


def _symbol_ratio(text: str) -> float:
    if not text:
        return 1.0
    symbols = sum(1 for c in text if not c.isalnum() and not c.isspace())
    return symbols / len(text)


def _alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if c.isalpha()) / len(text)


def _dedup_exact(docs: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for doc in docs:
        key = doc.strip()
        if key in seen:
            continue
        seen.add(key)
        result.append(doc)
    return result


class CorpusBuilder:
    """Builds a `CuratedCorpus` from a `Manifest` using a search client.

    The async `materialize` path is the one used by the environment and reward:
    it fetches each source's documents through a per-rollout deterministic cache
    (so a preview and final scoring observe identical docs and cost is charged
    exactly once), offloads the blocking fetch off the event loop, bounds
    concurrency, and degrades a failed source to an empty slice (recording the
    error in state) rather than raising. The synchronous `build` is retained for
    direct, cache-free use (e.g. unit tests of filtering/sampling).
    """

    def __init__(
        self,
        client: DatasetSearchClient,
        sample_docs_per_source: int = 64,
        document_filter: DocumentFilter | None = None,
        retry_policy: RetryPolicy | None = None,
        fetch_limit: int = 8,
    ) -> None:
        self._client = client
        self._sample_docs_per_source = sample_docs_per_source
        self._filter = document_filter or DocumentFilter()
        self._retry = retry_policy or RetryPolicy()
        self._fetch_limit = fetch_limit
        # Loop-local single-flight guard: concurrent fetches sharing one
        # (rollout, source key) coalesce onto a single Hub fetch + a single
        # billing event. Locks bind to their running loop, and the rollout
        # state must stay JSON-serializable, so they live here keyed by loop
        # then by "<rollout token>\x00<cache key>", never in state (mirrors the
        # loop-local registries in hf_access).
        self._fetch_locks: weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop, dict[str, asyncio.Lock]
        ] = weakref.WeakKeyDictionary()

    def source_key(self, source: Source) -> FetchKey:
        return FetchKey(
            dataset_id=source.dataset_id,
            config=source.config,
            split=source.split,
            text_field=source.text_field,
            n=self._sample_docs_per_source,
        )

    def _fetch_lock(self, lock_key: str) -> asyncio.Lock:
        """Loop-local single-flight lock for `lock_key`, created on demand."""
        loop = asyncio.get_running_loop()
        locks = self._fetch_locks.get(loop)
        if locks is None:
            locks = {}
            self._fetch_locks[loop] = locks
        lock = locks.get(lock_key)
        if lock is None:
            lock = asyncio.Lock()
            locks[lock_key] = lock
        return lock

    def _discard_fetch_lock(self, lock_key: str) -> None:
        """Drop a single-flight lock once its result is cached (bounds growth)."""
        loop = asyncio.get_running_loop()
        locks = self._fetch_locks.get(loop)
        if locks is not None:
            locks.pop(lock_key, None)

    async def fetch_source_docs(
        self, state: CuratorState, key: FetchKey
    ) -> tuple[list[str], dict[str, Any] | None]:
        """Return `(docs, error)` for `key`, using/populating the rollout cache.

        On a cache hit the stored docs are returned with no re-streaming and no
        re-billing. On a miss the fetch is attempted (bounded + timed + retried);
        success stores the docs and charges the ledger once (one hub call plus
        the sampled tokens); failure records typed telemetry and returns an empty
        slice with the structured error (and is *not* cached, so a later attempt
        may still succeed).

        Concurrent same-key callers (e.g. a preview racing the scoring fetch)
        are coalesced by a per-(rollout, key) single-flight lock with the same
        double-checked locking as ``CuratorRubric._prepared``: the cache is
        re-checked inside the lock so the underlying fetch and its billing fire
        exactly once, and the losers read the cached result.
        """
        cache_key = key.as_str()
        cached = RolloutStore.cached_docs(state, cache_key)
        if cached is not None:
            return cached, None
        # The single-flight lock is keyed per (rollout, fetch key). The rollout's
        # identity is the live state object: in v1 each rollout owns one
        # ``CuratorState`` for its lifetime, so ``id(state)`` is stable within the
        # rollout (mirrors the v0 fallback when no trajectory id was present).
        token = str(id(state))
        lock_key = f"{token}\x00{cache_key}"
        lock = self._fetch_lock(lock_key)
        try:
            async with lock:
                # Re-check under the lock: a racing caller may have already
                # fetched, stored, and billed this key while we waited.
                cached = RolloutStore.cached_docs(state, cache_key)
                if cached is not None:
                    return cached, None
                try:
                    docs = await fetch_documents(
                        self._client.sample_documents,
                        key,
                        policy=self._retry,
                        semaphore=hf_fetch_semaphore(self._fetch_limit),
                    )
                except DatasetAccessError as exc:
                    RolloutStore.record_tool_error(state, exc.kind)
                    RolloutStore.set_external_failure(state, True)
                    return [], exc.as_dict()
                RolloutStore.store_docs(state, cache_key, docs)
                ledger = RolloutStore.ledger(state)
                ledger.hub_calls += 1
                ledger.tokens += sum(estimate_tokens(d) for d in docs)
                RolloutStore.set_ledger(state, ledger)
                return docs, None
        finally:
            self._discard_fetch_lock(lock_key)

    async def materialize(self, manifest: Manifest, state: CuratorState) -> CuratedCorpus:
        """Cache-aware async corpus build; the single materialization per rollout."""
        sources: list[SourceCorpus] = []
        for source in manifest.sources:
            raw, _error = await self.fetch_source_docs(state, self.source_key(source))
            filtered = self._filter.apply(raw, source.filters)
            documents = self._apply_sampling(filtered, source)
            sources.append(
                SourceCorpus(
                    dataset_id=source.dataset_id,
                    config=source.config,
                    weight=source.weight,
                    documents=documents,
                )
            )
        return CuratedCorpus(sources=sources)

    def build(self, manifest: Manifest) -> CuratedCorpus:
        """Synchronous, cache-free build (direct client access; testing/fallback)."""
        sources: list[SourceCorpus] = []
        for source in manifest.sources:
            documents = self._materialize_source(source)
            sources.append(
                SourceCorpus(
                    dataset_id=source.dataset_id,
                    config=source.config,
                    weight=source.weight,
                    documents=documents,
                )
            )
        return CuratedCorpus(sources=sources)

    def _materialize_source(self, source: Source) -> list[str]:
        raw = self._client.sample_documents(
            source.dataset_id,
            source.config,
            source.split,
            source.text_field,
            self._sample_docs_per_source,
        )
        filtered = self._filter.apply(raw, source.filters)
        return self._apply_sampling(filtered, source)

    def _apply_sampling(self, docs: list[str], source: Source) -> list[str]:
        capped = docs
        if source.sampling.max_docs is not None:
            capped = capped[: source.sampling.max_docs]
        if source.sampling.max_tokens is not None:
            limited: list[str] = []
            budget = source.sampling.max_tokens
            for doc in capped:
                cost = estimate_tokens(doc)
                if cost > budget:
                    break
                limited.append(doc)
                budget -= cost
            capped = limited
        return capped
