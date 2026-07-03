"""Bounded leakage-reference construction from the held-out validation tokens."""

from __future__ import annotations

import asyncio
import logging
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
        self._cached: LeakageReference | None = None
        self._locks: weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop, asyncio.Lock
        ] = weakref.WeakKeyDictionary()

    async def load(self) -> LeakageReference:
        """Cache the real reference once built; always retry on stub fallback."""
        if self._cached is not None:
            return self._cached
        lock = self._lock()
        async with lock:
            if self._cached is not None:
                return self._cached
            try:
                val_set = await self._val_loader.load()
                reference = await asyncio.to_thread(self._build_real, val_set)
                self._cached = reference
            except Exception as exc:  # noqa: BLE001 - fallback is explicit telemetry
                logger.warning(
                    "[curator] leakage_reference=stub: real validation reference "
                    "unavailable (%s: %s)",
                    type(exc).__name__,
                    exc,
                )
                docs = tuple(self._fallback_docs)
                reference = LeakageReference(
                    detector=LeakageDetector(list(docs)),
                    source="stub",
                    documents=docs,
                    sampled_tokens=0,
                )
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
