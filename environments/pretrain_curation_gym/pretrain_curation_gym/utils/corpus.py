"""Materialize a curated corpus from a manifest."""

from __future__ import annotations

import asyncio
import hashlib
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

import orjson
import verifiers.v1 as vf

from ..gpu.scoring_shared import apply_filters_iter, weighted_token_target
from .async_utils import LoopLocalLocks, hf_fetch_semaphore, run_blocking_drained
from .hf_access import (
    DatasetAccessError,
    DatasetSearchClient,
    FetchKey,
    RetryPolicy,
    estimate_tokens,
    extract_text_from_row,
    fetch_documents,
)
from .models import FilterSpec, Manifest, Source
from ..state import CuratorState

logger = logging.getLogger(__name__)

EST_TOKENS_PER_DOC = 250
_LOW_BUDGET_FILL_RATIO = 0.5


def _write_jsonl(path: Path, docs: Iterable[str]) -> tuple[int, int]:
    """Stream `docs` to `path` (one JSON-encoded string per line)."""
    doc_count = 0
    token_count = 0
    with path.open("wb") as fh:
        for doc in docs:
            fh.write(orjson.dumps(doc))
            fh.write(b"\n")
            doc_count += 1
            token_count += estimate_tokens(doc)
    return doc_count, token_count


def _iter_jsonl(path: Path | None) -> Iterator[str]:
    if path is None:
        return
    with path.open("rb") as fh:
        for line in fh:
            yield orjson.loads(line)


def _iter_local_documents(
    path: Path, fmt: str, text_field: str | None
) -> Iterator[str]:
    """Yield documents from a bounded runtime-local text or JSONL file."""
    resolved_format = fmt
    if resolved_format == "auto":
        resolved_format = (
            "jsonl" if path.suffix.lower() in {".jsonl", ".ndjson", ".json"} else "txt"
        )

    if resolved_format == "jsonl":
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    value = orjson.loads(raw)
                except orjson.JSONDecodeError:
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


def _materialize_local_docs(
    raw_path: Path, raw_text: str, fmt: str, text_field: str | None
) -> list[str]:
    """Write a runtime-pulled file and parse it off the event loop."""
    raw_path.write_text(raw_text, encoding="utf-8")
    return list(_iter_local_documents(raw_path, fmt, text_field))


@dataclass
class SourceCorpus:
    dataset_id: str
    config: str | None
    weight: float
    path: Path | None
    doc_count: int = 0
    tokens: int = 0

    def iter_documents(self) -> Iterator[str]:
        yield from _iter_jsonl(self.path)

    @property
    def documents(self) -> list[str]:
        """Materializes every document into memory."""
        return list(self.iter_documents())

    @classmethod
    def from_docs(
        cls,
        dataset_id: str,
        config: str | None,
        weight: float,
        docs: Iterable[str],
        *,
        dest_dir: Path | None = None,
    ) -> "SourceCorpus":
        """Stream `docs` to a scratch file and return the resulting corpus."""
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
        with path.open("ab") as fh:
            for doc in docs:
                fh.write(orjson.dumps(doc))
                fh.write(b"\n")
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
        """Materializes every document across every source into memory."""
        return list(self.iter_documents())

    @property
    def total_tokens(self) -> int:
        return sum(source.tokens for source in self.sources)

    def is_empty(self) -> bool:
        return all(source.doc_count == 0 for source in self.sources)

    def joined_text(self, cap: int) -> str:
        """Serialize the capped source document list for trainer upload."""
        documents: list[str] = []
        remaining = max(0, int(cap))
        for doc in self.iter_documents():
            if len(doc) > remaining:
                break
            documents.append(doc)
            remaining -= len(doc)
        return orjson.dumps(
            {"format": "document-list-v1", "documents": documents}
        ).decode("utf-8")


def _iter_sampling(
    docs: Iterable[str],
    source: Source,
    weight_target: int | None,
    *,
    already_docs: int = 0,
    already_tokens: int = 0,
) -> Iterator[str]:
    """Streaming equivalent of the old list-based `_apply_sampling`."""
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
        if budget is not None:
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
        return apply_filters_iter(docs, filters)


def _weight_token_target(
    source: Source, token_budget: int, total_weight: float
) -> int | None:
    """Return the weight-proportional token target for `source`, or None if uncapped."""
    return weighted_token_target(source.weight, total_weight, token_budget)


def _est_fetch_docs(token_target: int, cap: int | None) -> int:
    """Estimate docs needed to reach `token_target`, conservatively at 250 tokens/doc."""
    est = max(token_target // EST_TOKENS_PER_DOC, 1)
    if cap is None:
        return est
    return min(est, cap)


def _doc_digest(text: str) -> bytes:
    """16-byte identity digest for dedup bookkeeping (not adversarial hashing)."""
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()


def _iter_unsampled(
    docs: Iterable[str],
    sampled_docs: Iterable[str],
) -> Iterator[str]:
    """Yield documents not already selected by the first sampling pass."""
    sampled = Counter(_doc_digest(doc) for doc in sampled_docs)
    for doc in docs:
        digest = _doc_digest(doc)
        remaining = sampled.get(digest, 0)
        if remaining:
            if remaining == 1:
                del sampled[digest]
            else:
                sampled[digest] = remaining - 1
            continue
        yield doc


class CorpusBuilder:
    """Builds a `CuratedCorpus` from a `Manifest` using a search client."""

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
        # fetch_limit covers Hub fan-out and filter/write pipeline.
        if not isinstance(fetch_limit, int) or isinstance(fetch_limit, bool):
            raise ValueError(
                f"fetch_limit must be a positive int, got {fetch_limit!r} "
                f"({type(fetch_limit).__name__})"
            )
        if fetch_limit < 1:
            raise ValueError(f"fetch_limit must be >= 1, got {fetch_limit}")
        self._fetch_limit = fetch_limit
        self._allow_local_sources = allow_local_sources
        self._max_local_source_bytes = max_local_source_bytes
        # Serialize cache_documents RMW on shared CuratorState.
        self._store_lock = asyncio.Lock()
        # Single-flight per (rollout, source) key.
        self._fetch_locks = LoopLocalLocks()

    async def fetch_source_docs(
        self, state: CuratorState, key: FetchKey
    ) -> tuple[list[str], dict[str, Any] | None]:
        """Return `(docs, error)` for `key`, using/populating the rollout cache."""
        cache_key = key.as_str()
        cached = await run_blocking_drained(state.cached_documents, cache_key)
        if cached is not None:
            return cached, None
        token = str(id(state))
        lock_key = f"{token}\x00{cache_key}"
        lock = self._fetch_locks.get(lock_key)
        try:
            async with lock:
                cached = await run_blocking_drained(
                    state.cached_documents, cache_key
                )
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
                    state.record_error(exc.kind)
                    return [], exc.as_dict()
                async with self._store_lock:
                    await run_blocking_drained(
                        state.cache_documents, cache_key, docs
                    )
                return docs, None
        finally:
            self._fetch_locks.discard(lock_key)

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
            state.record_error(kind)
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

        cached = await run_blocking_drained(state.cached_documents, cache_key)
        if cached is not None:
            return cached, None

        token = str(id(state))
        lock_key = f"{token}\x00{cache_key}"
        lock = self._fetch_locks.get(lock_key)
        try:
            async with lock:
                cached = await run_blocking_drained(
                    state.cached_documents, cache_key
                )
                if cached is not None:
                    return cached, None

                path = source.local_path
                assert path is not None
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

                raw_path = state.workspace() / f"local_raw_{uuid.uuid4().hex}"
                try:
                    docs = await run_blocking_drained(
                        _materialize_local_docs,
                        raw_path,
                        pulled.stdout,
                        source.local_format,
                        source.text_field,
                    )
                except Exception as exc:  # noqa: BLE001 - typed soft failure
                    return failure("local_parse_failed", str(exc))
                finally:
                    raw_path.unlink(missing_ok=True)

                async with self._store_lock:
                    await run_blocking_drained(
                        state.cache_documents, cache_key, docs
                    )
                truncated = size > cap
                state.record_local_source(
                    bytes_pulled=min(size, cap),
                    truncated=truncated,
                )
                return docs, None
        finally:
            self._fetch_locks.discard(lock_key)

    async def materialize(
        self,
        manifest: Manifest,
        state: CuratorState,
        *,
        runtime: vf.Runtime | None = None,
    ) -> CuratedCorpus:
        """Cache-aware async corpus build; the single materialization per rollout."""
        total_weight = sum(s.weight for s in manifest.sources)
        cap = manifest.sample_docs_per_source
        dest_dir = state.workspace()
        pipeline_sem = asyncio.Semaphore(self._fetch_limit)

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
            async with pipeline_sem:
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
                corpus = await run_blocking_drained(
                    SourceCorpus.from_docs,
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
                raw = await run_blocking_drained(
                    state.cached_documents, key.as_str()
                )
                if raw is None:
                    continue
                filtered = self._filter.apply_iter(raw, source.filters)
                surplus = _iter_unsampled(filtered, corpus.iter_documents())
                _, added_tokens = await run_blocking_drained(
                    corpus.append_iter,
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
        state.set_materialization_stats(
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
