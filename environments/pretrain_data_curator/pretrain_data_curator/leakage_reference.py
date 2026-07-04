"""Bounded leakage-reference construction from the held-out validation tokens."""

from __future__ import annotations

import asyncio
import logging
import time
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .eval_corpus import DEFAULT_EVAL_CORPUS
from .leakage import LeakageDetector
from .val_set import HeldOutValSet, ValTokenLoader

logger = logging.getLogger(__name__)

ReferenceSource = Literal["real", "stub"]
TokenDecoder = Callable[[list[int]], str]
DecoderFactory = Callable[[str], TokenDecoder]

# Reference windows are 1024 GPT-2 BPE tokens; GPT-2 averages ~1.3
# tokens/word for English web text, so 1024 / 1.3 ≈ 788 words.
# Round to 800: the fuzzy-chunk size the curated-document side is
# partitioned into so a verbatim reference window fits inside at
# least one chunk without straddling.
FUZZY_CHUNK_WORDS = 800


@dataclass(frozen=True)
class LeakageReference:
    """A ready-to-score detector plus provenance and bounded-size diagnostics."""

    detector: LeakageDetector
    source: ReferenceSource
    documents: tuple[str, ...]
    sampled_tokens: int


class LeakageReferenceLoader:
    """Build and cache a bounded text reference from the real validation shard.

    The default sample takes one deterministic pseudo-random window from each of
    64 equal strata. Each window is at most 1,024 tokens, so decoding and detector
    construction see no more than 65,536 validation tokens regardless of shard
    size. If validation loading or decoding fails, the small built-in corpus is
    used with an explicit warning and ``source="stub"``.
    """

    DEFAULT_SAMPLE_COUNT = 64
    DEFAULT_CHUNK_TOKENS = 1_024
    MAX_CHUNK_CHARS = 8_192
    MAX_SEMANTIC_FEATURES = 32_768
    DEFAULT_SEED = 0

    def __init__(
        self,
        val_loader: ValTokenLoader,
        *,
        fallback_docs: list[str] | None = None,
        decoder_factory: DecoderFactory | None = None,
        sample_count: int = DEFAULT_SAMPLE_COUNT,
        chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
        seed: int = DEFAULT_SEED,
        stub_ttl_seconds: float = 60.0,
    ) -> None:
        if sample_count < 1:
            raise ValueError(f"sample_count must be >= 1, got {sample_count}")
        if chunk_tokens < 1:
            raise ValueError(f"chunk_tokens must be >= 1, got {chunk_tokens}")
        self._val_loader = val_loader
        self._fallback_docs = list(fallback_docs or DEFAULT_EVAL_CORPUS)
        self._decoder_factory = decoder_factory or self._tiktoken_decoder
        self._sample_count = sample_count
        self._chunk_tokens = chunk_tokens
        self._seed = seed
        self._stub_ttl = stub_ttl_seconds
        self._cached: LeakageReference | None = None
        self._stub_cache: tuple[float, LeakageReference] | None = None
        self._locks: weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop, asyncio.Lock
        ] = weakref.WeakKeyDictionary()

    async def load(self) -> LeakageReference:
        """Cache the real reference once built; stub cached with TTL.

        On transient failure the stub is cached for ``stub_ttl_seconds`` so
        concurrent/serial rollouts avoid repeated retry cost while still
        converging to real once connectivity returns.
        """
        if self._cached is not None:
            return self._cached
        if self._stub_cache is not None:
            ts, ref = self._stub_cache
            if time.monotonic() - ts < self._stub_ttl:
                return ref
        lock = self._lock()
        async with lock:
            if self._cached is not None:
                return self._cached
            if self._stub_cache is not None:
                ts, ref = self._stub_cache
                if time.monotonic() - ts < self._stub_ttl:
                    return ref
            try:
                val_set = await self._val_loader.load()
                reference = await asyncio.to_thread(self._build_real, val_set)
                self._cached = reference
                self._stub_cache = None
            except Exception as exc:  # noqa: BLE001 - fallback is explicit telemetry
                logger.warning(
                    "[curator] leakage_reference=stub: real validation reference "
                    "unavailable (%s: %s)",
                    type(exc).__name__,
                    exc,
                )
                docs = tuple(self._fallback_docs)
                reference = LeakageReference(
                    detector=LeakageDetector(
                        list(docs), fuzzy_chunk_words=FUZZY_CHUNK_WORDS
                    ),
                    source="stub",
                    documents=docs,
                    sampled_tokens=0,
                )
                self._stub_cache = (time.monotonic(), reference)
            return reference

    def _lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        lock = self._locks.get(loop)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[loop] = lock
        return lock

    def _build_real(self, val_set: HeldOutValSet) -> LeakageReference:
        decoder = self._decoder_factory(val_set.tokenizer)
        starts = self._sample_starts(val_set.n_tokens)
        documents: list[str] = []
        sampled_tokens = 0
        for start in starts:
            stop = min(start + self._chunk_tokens, val_set.n_tokens)
            token_ids = val_set.tokens[start:stop].astype(np.int64).tolist()
            text = decoder(token_ids).strip()[: self.MAX_CHUNK_CHARS]
            if text:
                documents.append(text)
                sampled_tokens += stop - start
        if not documents:
            raise ValueError("sampled validation windows decoded to no text")
        docs = tuple(documents)
        return LeakageReference(
            detector=LeakageDetector(
                list(docs),
                max_semantic_features=self.MAX_SEMANTIC_FEATURES,
                fuzzy_chunk_words=FUZZY_CHUNK_WORDS,
            ),
            source="real",
            documents=docs,
            sampled_tokens=sampled_tokens,
        )

    def _sample_starts(self, n_tokens: int) -> list[int]:
        if n_tokens < 1:
            raise ValueError("validation token stream is empty")
        max_start = max(0, n_tokens - self._chunk_tokens)
        if max_start == 0:
            return [0]
        count = min(self._sample_count, max_start + 1)
        boundaries = np.linspace(0, max_start + 1, count + 1, dtype=np.int64)
        rng = np.random.default_rng(self._seed)
        starts: list[int] = []
        for index in range(count):
            low = int(boundaries[index])
            high = int(boundaries[index + 1])
            starts.append(low if high <= low + 1 else int(rng.integers(low, high)))
        return starts

    @staticmethod
    def _tiktoken_decoder(tokenizer: str) -> TokenDecoder:
        import tiktoken

        encoding = tiktoken.get_encoding(tokenizer)
        return encoding.decode
