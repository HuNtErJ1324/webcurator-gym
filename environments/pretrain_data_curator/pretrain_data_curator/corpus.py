"""Materialize a curated corpus from a manifest.

`CorpusBuilder` turns a `Manifest` into concrete documents by sampling each
source through the `DatasetSearchClient`, applying its filters, and honoring
per-source sampling caps. `DocumentFilter` owns the supported filter kinds.

Fetched-and-filtered document TEXT is streamed to per-rollout scratch files on
disk as it is produced rather than accumulated in Python lists: `SourceCorpus`/
`CuratedCorpus` hold file paths + lightweight counts (doc/token counts), not the
text itself, so peak host memory for a rollout's corpus stays bounded to one
transient fetch+filter pass per allowed concurrent fetch instead of growing with
every source and document retained over the rollout's lifetime. See
`rollout_state.RolloutStore
.scratch_dir` for the backing directory and its cleanup, and `SourceCorpus
.documents`/`CuratedCorpus.documents` for the (materializing, test/debug-only)
convenience accessors -- production consumers (reward scoring, the proxy-student
trainer, the docker/modal upload) use the streaming `iter_documents()`/
`joined_text()` instead.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import shlex
import shutil
import tempfile
import uuid
import weakref
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import verifiers.v1 as vf

from .hf_access import (
    DatasetAccessError,
    DatasetSearchClient,
    FetchKey,
    RetryPolicy,
    estimate_tokens,
    extract_text_from_row,
    fetch_documents,
    hf_fetch_semaphore,
)
from .models import FilterSpec, Manifest, Source
from .rollout_state import CuratorState, RolloutStore

logger = logging.getLogger(__name__)

EST_TOKENS_PER_DOC = 250
_LOW_BUDGET_FILL_RATIO = 0.5


def _write_jsonl(path: Path, docs: Iterable[str]) -> tuple[int, int]:
    """Stream `docs` to `path` (one JSON-encoded string per line).

    Returns `(doc_count, token_count)`, accumulated as each doc is written so
    the caller never needs the full text again to know these counts.
    """
    doc_count = 0
    token_count = 0
    with path.open("w", encoding="utf-8") as fh:
        for doc in docs:
            fh.write(json.dumps(doc))
            fh.write("\n")
            doc_count += 1
            token_count += estimate_tokens(doc)
    return doc_count, token_count


def _iter_jsonl(path: Path | None) -> Iterator[str]:
    if path is None:
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            yield json.loads(line)


def _iter_local_documents(
    path: Path, fmt: str, text_field: str | None
) -> Iterator[str]:
    """Yield documents from a bounded runtime-local text or JSONL file."""
    resolved_format = fmt
    if resolved_format == "auto":
        resolved_format = (
            "jsonl"
            if path.suffix.lower() in {".jsonl", ".ndjson", ".json"}
            else "txt"
        )

    if resolved_format == "jsonl":
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    value = json.loads(raw)
                except json.JSONDecodeError:
                    yield raw
                    continue
                if isinstance(value, dict):
                    text = extract_text_from_row(value, text_field)
                    if text is not None:
                        yield text
                elif isinstance(value, str) and value.strip():
                    yield value
        return

    content = path.read_text(encoding="utf-8", errors="replace")
    for chunk in re.split(r"\n\s*\n", content):
        text = chunk.strip()
        if text:
            yield text


@dataclass
class SourceCorpus:
    dataset_id: str
    config: str | None
    weight: float
    # The surviving (post-filter, post-sampling) documents, JSONL-encoded on
    # disk; `None` when the source contributed no documents.
    path: Path | None
    doc_count: int = 0
    tokens: int = 0

    def iter_documents(self) -> Iterator[str]:
        yield from _iter_jsonl(self.path)

    @property
    def documents(self) -> list[str]:
        """Materializes every document into memory.

        Convenience for small fixture corpora and tests; production consumers
        (leakage scoring, the trainer, the docker/modal upload) must use
        `iter_documents()` instead so peak memory does not scale with corpus
        size.
        """
        return list(self.iter_documents())

    @classmethod
    def from_iter(
        cls,
        dataset_id: str,
        config: str | None,
        weight: float,
        docs: Iterable[str],
        *,
        dest_dir: Path | None = None,
    ) -> "SourceCorpus":
        """Stream `docs` to a scratch file and return the resulting corpus.

        `dest_dir` is normally a rollout's shared scratch directory (see
        `RolloutStore.scratch_dir`), whose cleanup the caller owns. When
        omitted (standalone/test construction), a private temp directory is
        created and registered for best-effort cleanup via `weakref.finalize`
        on the returned object.

        The owned directory's creation, and the `weakref.finalize`
        registration that guards its eventual cleanup, are BOTH inside the
        `try`/`except` below (not just the write in between): if `mkdtemp`,
        `_write_jsonl`, or even `weakref.finalize` itself raises before this
        function returns, there would otherwise be nothing left to ever clean
        the directory up -- so a raise anywhere in this block removes it
        synchronously instead of leaking it.
        """
        owned_dir: Path | None = None
        try:
            directory = dest_dir
            if directory is None:
                owned_dir = Path(tempfile.mkdtemp(prefix="pdc_src_"))
                directory = owned_dir
            path = directory / f"src_{uuid.uuid4().hex}.jsonl"
            doc_count, tokens = _write_jsonl(path, docs)
            if doc_count == 0:
                path.unlink(missing_ok=True)
                path = None
            source = cls(dataset_id, config, weight, path, doc_count, tokens)
            if owned_dir is not None:
                weakref.finalize(
                    source, shutil.rmtree, str(owned_dir), ignore_errors=True
                )
        except BaseException:
            if owned_dir is not None:
                shutil.rmtree(owned_dir, ignore_errors=True)
            raise
        return source

    def append_iter(self, docs: Iterable[str], *, dest_dir: Path) -> tuple[int, int]:
        """Append sampled documents and return their ``(doc_count, tokens)``."""
        path = self.path or dest_dir / f"src_{uuid.uuid4().hex}.jsonl"
        added_docs = 0
        added_tokens = 0
        with path.open("a", encoding="utf-8") as fh:
            for doc in docs:
                fh.write(json.dumps(doc))
                fh.write("\n")
                added_docs += 1
                added_tokens += estimate_tokens(doc)
        if added_docs:
            self.path = path
            self.doc_count += added_docs
            self.tokens += added_tokens
        elif self.path is None:
            path.unlink(missing_ok=True)
        return added_docs, added_tokens


@dataclass
class CuratedCorpus:
    sources: list[SourceCorpus]

    def iter_documents(self) -> Iterator[str]:
        for source in self.sources:
            yield from source.iter_documents()

    @property
    def documents(self) -> list[str]:
        """Materializes every document across every source into memory.

        Convenience for small fixture corpora and tests; production consumers
        must use `iter_documents()`/`joined_text()` instead (see `SourceCorpus
        .documents`).
        """
        return list(self.iter_documents())

    @property
    def total_tokens(self) -> int:
        return sum(source.tokens for source in self.sources)

    def is_empty(self) -> bool:
        return all(source.doc_count == 0 for source in self.sources)

    def joined_text(self, cap: int) -> str:
        """Serialize the capped source document list for trainer upload.

        The historical method name is retained for backend compatibility, but
        the payload is tagged JSON rather than blank-line-joined prose. Keeping
        documents as explicit list entries preserves first/blank/embedded-newline
        boundaries and lets the tokenizer insert EOT/BOS without guessing via
        ``text.split``. ``cap`` bounds source-document characters at whole-document
        granularity: the longest prefix whose documents fit is serialized, and the
        first document that would cross the cap stops the stream. Documents are
        never truncated, so their eventual EOT prefixes cannot be lost or detached.
        """
        documents: list[str] = []
        remaining = max(0, int(cap))
        for doc in self.iter_documents():
            if len(doc) > remaining:
                break
            documents.append(doc)
            remaining -= len(doc)
        return json.dumps(
            {"format": "document-list-v1", "documents": documents},
            ensure_ascii=False,
            separators=(",", ":"),
        )


def _iter_sampling(
    docs: Iterable[str],
    source: Source,
    weight_target: int | None,
    *,
    already_docs: int = 0,
    already_tokens: int = 0,
) -> Iterator[str]:
    """Streaming equivalent of the old list-based `_apply_sampling`.

    Applies `Sampling.max_docs` (stop after N documents) and the effective
    token cap (`weight_target`, tightened by `Sampling.max_tokens` if set).
    Documents that do not fit the remaining token budget are skipped so a later
    smaller document can still be selected. The generator stops pulling from
    upstream as soon as the document cap is hit.
    """
    max_docs = source.sampling.max_docs
    token_cap = weight_target
    if source.sampling.max_tokens is not None:
        remaining_source_tokens = max(
            source.sampling.max_tokens - already_tokens,
            0,
        )
        token_cap = (
            min(token_cap, remaining_source_tokens)
            if token_cap is not None
            else remaining_source_tokens
        )
    budget = token_cap
    count = already_docs
    for doc in docs:
        if max_docs is not None and count >= max_docs:
            break
        if token_cap is not None:
            cost = estimate_tokens(doc)
            if cost > budget:
                continue
            budget -= cost
        yield doc
        count += 1


class DocumentFilter:
    """Applies an ordered list of `FilterSpec` to a stream of documents."""

    def apply(self, docs: list[str], filters: list[FilterSpec]) -> list[str]:
        return list(self.apply_iter(docs, filters))

    def apply_iter(
        self, docs: Iterable[str], filters: list[FilterSpec]
    ) -> Iterator[str]:
        stream: Iterable[str] = docs
        for spec in filters:
            stream = self._apply_one_iter(stream, spec)
        return iter(stream)

    def _apply_one_iter(self, docs: Iterable[str], spec: FilterSpec) -> Iterator[str]:
        kind = spec.kind
        params = spec.params
        if kind == "min_chars":
            threshold = int(params.get("value", 0))
            return (d for d in docs if len(d) >= threshold)
        if kind == "max_chars":
            threshold = int(params.get("value", 10**9))
            return (d for d in docs if len(d) <= threshold)
        if kind == "min_tokens":
            threshold = int(params.get("value", 0))
            return (d for d in docs if estimate_tokens(d) >= threshold)
        if kind == "max_symbol_ratio":
            threshold = float(params.get("value", 1.0))
            return (d for d in docs if _symbol_ratio(d) <= threshold)
        if kind == "min_alpha_ratio":
            threshold = float(params.get("value", 0.0))
            return (d for d in docs if _alpha_ratio(d) >= threshold)
        if kind == "drop_regex":
            pattern = re.compile(str(params.get("pattern", "")))
            return (d for d in docs if not pattern.search(d))
        if kind == "keep_regex":
            pattern = re.compile(str(params.get("pattern", "")))
            return (d for d in docs if pattern.search(d))
        if kind == "dedup_exact":
            return _dedup_exact_iter(docs)
        # Unknown filter kinds are ignored rather than raising, so an agent's
        # experimentation never hard-fails the rollout.
        return iter(docs)


def _symbol_ratio(text: str) -> float:
    if not text:
        return 1.0
    symbols = sum(1 for c in text if not c.isalnum() and not c.isspace())
    return symbols / len(text)


def _alpha_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if c.isalpha()) / len(text)


def _dedup_exact_iter(docs: Iterable[str]) -> Iterator[str]:
    """Streaming `dedup_exact`: a running set of fixed-size digests, not the
    full document list (nor the full stripped text -- a `set[str]` of
    stripped documents would itself be a second complete copy of every
    surviving document's text), is all that is needed to detect duplicates.
    Collision probability at SHA-256 is astronomically below any realistic
    corpus size, so this is exact in practice, matching the previous
    full-text-equality semantics.
    """
    seen: set[bytes] = set()
    for doc in docs:
        digest = hashlib.sha256(doc.strip().encode("utf-8")).digest()
        if digest in seen:
            continue
        seen.add(digest)
        yield doc


def _weight_token_target(
    source: Source, token_budget: int, total_weight: float
) -> int | None:
    """Return the weight-proportional token target for `source`, or None if uncapped."""
    if total_weight <= 0:
        return None
    if source.weight == 0:
        return 0
    return int((source.weight / total_weight) * token_budget)


def _est_fetch_docs(token_target: int, cap: int | None) -> int:
    """Estimate docs needed to reach `token_target`, conservatively at 250 tokens/doc.

    When the agent sets ``sample_docs_per_source`` on the manifest, never over-fetch
    past that cap; otherwise fetch enough to cover the weight-proportional target.
    """
    est = max(token_target // EST_TOKENS_PER_DOC, 1)
    if cap is None:
        return est
    return min(est, cap)


def _iter_unsampled(
    docs: Iterable[str],
    sampled_docs: Iterable[str],
) -> Iterator[str]:
    """Yield documents not already selected by the first sampling pass."""
    sampled = Counter(
        hashlib.sha256(doc.encode("utf-8")).digest() for doc in sampled_docs
    )
    for doc in docs:
        digest = hashlib.sha256(doc.encode("utf-8")).digest()
        remaining = sampled.get(digest, 0)
        if remaining:
            if remaining == 1:
                del sampled[digest]
            else:
                sampled[digest] = remaining - 1
            continue
        yield doc


class CorpusBuilder:
    """Builds a `CuratedCorpus` from a `Manifest` using a search client.

    The async `materialize` path is the one used by the environment and reward:
    it fetches each source's documents through a per-rollout deterministic cache
    (so a preview and final scoring observe identical docs), offloads the
    blocking fetch off the event loop, bounds
    concurrency, and degrades a failed source to an empty slice (recording the
    error in state) rather than raising. The synchronous `build` is retained for
    direct, cache-free use (e.g. unit tests of filtering/sampling).
    """

    def __init__(
        self,
        client: DatasetSearchClient,
        document_filter: DocumentFilter | None = None,
        retry_policy: RetryPolicy | None = None,
        fetch_limit: int = 8,
        allow_local_sources: bool = True,
        max_local_source_bytes: int = 33_554_432,
    ) -> None:
        self._client = client
        self._filter = document_filter or DocumentFilter()
        self._retry = retry_policy or RetryPolicy()
        self._fetch_limit = fetch_limit
        self._allow_local_sources = allow_local_sources
        self._max_local_source_bytes = max_local_source_bytes
        # Loop-local single-flight guard: concurrent fetches sharing one
        # (rollout, source key) coalesce onto a single Hub fetch. Locks bind to
        # their running loop, and the rollout
        # state must stay JSON-serializable, so they live here keyed by loop
        # then by "<rollout token>\x00<cache key>", never in state (mirrors the
        # loop-local registries in hf_access).
        self._fetch_locks: weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop, dict[str, asyncio.Lock]
        ] = weakref.WeakKeyDictionary()

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

        On a cache hit the stored docs are returned with no re-streaming. On a
        miss the fetch is attempted (bounded + timed + retried); success stores
        the docs once; failure records typed telemetry and returns an empty
        slice with the structured error (and is *not* cached, so a later attempt
        may still succeed).

        Concurrent same-key callers (e.g. a preview racing the scoring fetch)
        are coalesced by a per-(rollout, key) single-flight lock with the same
        double-checked locking as ``CuratorRubric._prepared``: the cache is
        re-checked inside the lock so the underlying fetch fires exactly once,
        and the losers read the cached result.
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
                # fetched and stored this key while we waited.
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
                return docs, None
        finally:
            self._discard_fetch_lock(lock_key)

    def _local_fetch_key(self, source: Source) -> FetchKey:
        return FetchKey(
            dataset_id=f"local:{source.local_path}",
            config=source.local_format,
            split="local",
            text_field=source.text_field,
            n=self._max_local_source_bytes,
        )

    async def fetch_local_docs(
        self,
        state: CuratorState,
        source: Source,
        runtime: vf.Runtime | None,
    ) -> tuple[list[str], dict[str, Any] | None]:
        """Pull, parse, cache, and meter one bounded runtime-local source."""
        key = self._local_fetch_key(source)
        cache_key = key.as_str()

        def failure(kind: str, message: str) -> tuple[list[str], dict[str, Any]]:
            RolloutStore.record_tool_error(state, kind)
            RolloutStore.set_external_failure(state, True)
            return (
                [],
                {
                    "error": message,
                    "error_kind": kind,
                    "dataset_id": source.dataset_id,
                },
            )

        if not self._allow_local_sources:
            return failure("local_disabled", "local sources are disabled")
        if runtime is None:
            return failure(
                "local_no_runtime",
                "local source cannot be read without a live runtime",
            )

        cached = RolloutStore.cached_docs(state, cache_key)
        if cached is not None:
            return cached, None

        token = str(id(state))
        lock_key = f"{token}\x00{cache_key}"
        lock = self._fetch_lock(lock_key)
        try:
            async with lock:
                cached = RolloutStore.cached_docs(state, cache_key)
                if cached is not None:
                    return cached, None

                path = source.local_path
                assert path is not None  # validated by Source
                quoted_path = shlex.quote(path)
                try:
                    probe = await runtime.run(
                        ["sh", "-c", f"wc -c < {quoted_path}"], {}
                    )
                except Exception as exc:  # noqa: BLE001 - typed soft failure
                    return failure("local_probe_failed", str(exc))
                if probe.exit_code != 0:
                    message = probe.stderr.strip() or "local source size probe failed"
                    return failure("local_probe_failed", message)
                try:
                    size = int(probe.stdout.strip())
                except ValueError:
                    return failure(
                        "local_probe_failed",
                        f"invalid byte count {probe.stdout.strip()!r}",
                    )
                if size < 0:
                    return failure("local_probe_failed", f"invalid byte count {size}")

                cap = self._max_local_source_bytes
                try:
                    pulled = await runtime.run(
                        ["sh", "-c", f"head -c {cap} -- {quoted_path}"], {}
                    )
                except Exception as exc:  # noqa: BLE001 - typed soft failure
                    return failure("local_pull_failed", str(exc))
                if pulled.exit_code != 0:
                    message = pulled.stderr.strip() or "local source pull failed"
                    return failure("local_pull_failed", message)

                raw_path = (
                    RolloutStore.scratch_dir(state)
                    / f"local_raw_{uuid.uuid4().hex}"
                )
                try:
                    raw_path.write_text(pulled.stdout, encoding="utf-8")
                    docs = list(
                        _iter_local_documents(
                            raw_path, source.local_format, source.text_field
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - typed soft failure
                    return failure("local_parse_failed", str(exc))
                finally:
                    raw_path.unlink(missing_ok=True)

                RolloutStore.store_docs(state, cache_key, docs)
                truncated = size > cap
                RolloutStore.add_local_source(
                    state,
                    bytes_pulled=min(size, cap),
                    truncated=truncated,
                )
                return docs, None
        finally:
            self._discard_fetch_lock(lock_key)

    async def materialize(
        self,
        manifest: Manifest,
        state: CuratorState,
        *,
        runtime: vf.Runtime | None = None,
    ) -> CuratedCorpus:
        """Cache-aware async corpus build; the single materialization per rollout.

        Each source's surviving (filtered + sampled) documents are streamed
        straight to a file under the rollout's shared scratch directory (see
        `RolloutStore.scratch_dir`) as they are produced, instead of being
        accumulated into a Python list kept alive for the corpus's lifetime --
        so retained memory is bounded by in-flight fetch concurrency, not by the
        sum of every source's docs across the whole rollout.
        """
        total_weight = sum(s.weight for s in manifest.sources)
        # The agent's manifest may request its own per-rollout fetch cap; it wins over the
        # human-configured default when set. `n` (derived from this cap) feeds
        # directly into `FetchKey.n` below, so two calls that land on different
        # *effective fetch counts* get distinct cache entries; two different caps
        # that happen to estimate the same `n` (e.g. both are well above a small
        # weight-derived token target) correctly share one cache entry, since the
        # underlying fetch parameters are then identical.
        cap = manifest.sample_docs_per_source
        dest_dir = RolloutStore.scratch_dir(state)

        async def materialize_source(
            source: Source,
        ) -> tuple[SourceCorpus, FetchKey | None, dict[str, Any] | None, bool]:
            weight_target = _weight_token_target(
                source, manifest.token_budget, total_weight
            )
            if total_weight > 0 and source.weight == 0:
                return (
                    SourceCorpus(source.dataset_id, source.config, source.weight, None),
                    None,
                    {
                        "error": "zero-weight source skipped without fetching",
                        "error_kind": "zero_weight",
                        "dataset_id": source.dataset_id,
                    },
                    False,
                )
            if source.kind == "local":
                key = self._local_fetch_key(source)
                raw, error = await self.fetch_local_docs(state, source, runtime)
            else:
                if weight_target is not None:
                    n = _est_fetch_docs(weight_target, cap)
                elif cap is not None:
                    n = cap
                else:
                    n = max(manifest.token_budget // EST_TOKENS_PER_DOC, 1)
                key = FetchKey(
                    dataset_id=source.dataset_id,
                    config=source.config,
                    split=source.split,
                    text_field=source.text_field,
                    n=n,
                )
                raw, error = await self.fetch_source_docs(state, key)
            filtered_count = 0
            filtered_exhausted = False

            def tracked_filtered() -> Iterator[str]:
                nonlocal filtered_count, filtered_exhausted
                for doc in self._filter.apply_iter(raw, source.filters):
                    filtered_count += 1
                    yield doc
                filtered_exhausted = True

            filtered = tracked_filtered()
            sampled = _iter_sampling(filtered, source, weight_target)
            corpus = SourceCorpus.from_iter(
                source.dataset_id,
                source.config,
                source.weight,
                sampled,
                dest_dir=dest_dir,
            )
            return (
                corpus,
                key,
                error,
                not filtered_exhausted or filtered_count > corpus.doc_count,
            )

        materialized = await asyncio.gather(
            *(materialize_source(source) for source in manifest.sources)
        )
        sources = [result[0] for result in materialized]

        # Redistribute unused weighted budget to already-fetched documents that
        # survived filtering but were not selected in the first pass. Re-reading
        # the rollout's on-disk raw cache avoids additional Hub calls.
        if total_weight > 0:
            remaining_budget = max(
                manifest.token_budget - sum(source.tokens for source in sources),
                0,
            )
            for source, corpus, (_, key, _, has_surplus) in zip(
                manifest.sources,
                sources,
                materialized,
                strict=True,
            ):
                if remaining_budget <= 0:
                    break
                if key is None or not has_surplus:
                    continue
                raw = RolloutStore.cached_docs(state, key.as_str())
                if raw is None:
                    continue
                filtered = self._filter.apply_iter(raw, source.filters)
                surplus = _iter_unsampled(filtered, corpus.iter_documents())
                _, added_tokens = corpus.append_iter(
                    _iter_sampling(
                        surplus,
                        source,
                        remaining_budget,
                        already_docs=corpus.doc_count,
                        already_tokens=corpus.tokens,
                    ),
                    dest_dir=dest_dir,
                )
                remaining_budget -= added_tokens

        for source, corpus, (_, _, error, _) in zip(
            manifest.sources,
            sources,
            materialized,
            strict=True,
        ):
            if corpus.doc_count == 0:
                logger.warning(
                    "source materialized empty: dataset_id=%s config=%r error=%s",
                    source.dataset_id,
                    source.config,
                    error,
                )

        corpus = CuratedCorpus(sources=sources)
        fill_ratio = corpus.total_tokens / manifest.token_budget
        RolloutStore.set_materialization_stats(
            state,
            budget_fill_ratio=fill_ratio,
            source_doc_counts=[source.doc_count for source in sources],
            source_token_counts=[source.tokens for source in sources],
        )
        if fill_ratio < _LOW_BUDGET_FILL_RATIO:
            logger.warning(
                "corpus severely undershot token budget: total_tokens=%d "
                "token_budget=%d budget_fill_ratio=%.3f",
                corpus.total_tokens,
                manifest.token_budget,
                fill_ratio,
            )
        return corpus
