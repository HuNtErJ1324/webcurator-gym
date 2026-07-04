"""Contamination detection against a fixed held-out evaluation corpus.

Combines three overlap signals between curated documents and the eval set:
  - exact: normalized full-document hash match
  - fuzzy: MinHash-estimated Jaccard over word shingles
  - semantic: cosine similarity over character-trigram frequency vectors

The neural-free semantic signal keeps the detector deterministic and dependency
light; it can be swapped for a real embedding model later.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"\w+")


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _doc_hash(text: str) -> str:
    return hashlib.sha1(_normalize(text).encode("utf-8")).hexdigest()


def _stable_hash32(text: str) -> int:
    """Process-stable 32-bit hash of a shingle.

    Python's built-in ``hash()`` is salted per process (``PYTHONHASHSEED``), so
    MinHash signatures built with it differ across runs/processes and make the
    estimated Jaccard / fuzzy-leakage score non-reproducible. blake2b is stable.
    """
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big")


def _word_shingles(text: str, k: int = 5) -> set[int]:
    words = _WORD_RE.findall(text.lower())
    if len(words) < k:
        if not words:
            return set()
        return {_stable_hash32(" ".join(words))}
    shingles = set()
    for i in range(len(words) - k + 1):
        shingle = " ".join(words[i : i + k])
        shingles.add(_stable_hash32(shingle))
    return shingles


@dataclass
class LeakageScores:
    exact: float
    fuzzy: float
    semantic: float
    overall: float

    def as_dict(self) -> dict[str, float]:
        return {
            "exact": round(self.exact, 4),
            "fuzzy": round(self.fuzzy, 4),
            "semantic": round(self.semantic, 4),
            "overall": round(self.overall, 4),
        }


class LeakageDetector:
    """Precomputes eval-set signatures and scores curated documents against them."""

    def __init__(
        self,
        eval_docs: list[str],
        num_perm: int = 64,
        shingle_k: int = 5,
        fuzzy_threshold: float = 0.5,
        semantic_threshold: float = 0.8,
        seed: int = 0,
        max_semantic_features: int | None = None,
        fuzzy_chunk_words: int = 0,
    ) -> None:
        if max_semantic_features is not None and max_semantic_features < 1:
            raise ValueError(
                "max_semantic_features must be >= 1 when set, "
                f"got {max_semantic_features}"
            )
        self._num_perm = num_perm
        self._shingle_k = shingle_k
        self._fuzzy_threshold = fuzzy_threshold
        self._semantic_threshold = semantic_threshold
        self._max_semantic_features = max_semantic_features
        self._fuzzy_chunk_words = fuzzy_chunk_words
        rng = np.random.default_rng(seed)
        mask = (1 << 32) - 1
        self._a = rng.integers(1, mask, size=num_perm, dtype=np.uint64)
        self._b = rng.integers(0, mask, size=num_perm, dtype=np.uint64)
        self._prime = np.uint64((1 << 61) - 1)

        self._eval_hashes = {_doc_hash(d) for d in eval_docs}
        self._eval_minhashes = np.array(
            [self._minhash(d) for d in eval_docs], dtype=np.uint64
        ) if eval_docs else np.empty((0, num_perm), dtype=np.uint64)
        self._eval_shingle_sets = [_word_shingles(d, self._shingle_k) for d in eval_docs]
        self._trigram_index, self._eval_vectors = self._build_vectors(eval_docs)

    def score(self, docs: Iterable[str]) -> LeakageScores:
        """Score `docs` (any iterable, e.g. a streaming `CuratedCorpus.iter_documents()`)
        against the eval set in a SINGLE pass, so a one-shot generator works: each of
        the three signals is independent per-document, so exact/fuzzy/semantic hits are
        all tallied together against a running document count, rather than requiring
        three separate passes over (and thus a fully materialized) document list.
        """
        has_exact = bool(self._eval_hashes)
        has_fuzzy = self._eval_minhashes.shape[0] > 0
        has_semantic = self._eval_vectors.shape[0] > 0 and bool(self._trigram_index)
        exact_hits = fuzzy_hits = semantic_hits = count = 0
        for doc in docs:
            count += 1
            if has_exact and _doc_hash(doc) in self._eval_hashes:
                exact_hits += 1
            if has_fuzzy and self._is_fuzzy_hit(doc):
                fuzzy_hits += 1
            if has_semantic and self._is_semantic_hit(doc):
                semantic_hits += 1
        if count == 0:
            return LeakageScores(0.0, 0.0, 0.0, 0.0)
        exact = exact_hits / count if has_exact else 0.0
        fuzzy = fuzzy_hits / count if has_fuzzy else 0.0
        semantic = semantic_hits / count if has_semantic else 0.0
        overall = max(exact, fuzzy, semantic)
        return LeakageScores(exact, fuzzy, semantic, overall)

    def _minhash(self, text: str) -> np.ndarray:
        shingles = _word_shingles(text, self._shingle_k)
        if not shingles:
            return np.full(self._num_perm, np.uint64(0), dtype=np.uint64)
        arr = np.array(list(shingles), dtype=np.uint64)
        # (num_perm, num_shingles) affine permutations, then per-row minimum.
        hashed = (self._a[:, None] * arr[None, :] + self._b[:, None]) % self._prime
        return hashed.min(axis=1)

    def _is_fuzzy_hit(self, doc: str) -> bool:
        if self._fuzzy_chunk_words > 0:
            words = _WORD_RE.findall(doc.lower())
            if not words:
                return False
            chunk_size = self._fuzzy_chunk_words
            stride = max(1, chunk_size // 2)
            i = 0
            while i < len(words):
                chunk = " ".join(words[i : i + chunk_size])
                sig = self._minhash(chunk)
                equal = (self._eval_minhashes == sig[None, :]).mean(axis=1)
                if bool(equal.max() >= self._fuzzy_threshold):
                    return True
                if self._eval_shingle_sets:
                    chunk_shingles = _word_shingles(chunk, self._shingle_k)
                    if chunk_shingles:
                        for eval_shingles in self._eval_shingle_sets:
                            if not eval_shingles:
                                continue
                            containment = len(chunk_shingles & eval_shingles) / len(
                                chunk_shingles
                            )
                            if containment >= self._fuzzy_threshold:
                                return True
                i += stride
            return False
        sig = self._minhash(doc)
        equal = (self._eval_minhashes == sig[None, :]).mean(axis=1)
        return bool(equal.max() >= self._fuzzy_threshold)

    def _build_vectors(
        self, docs: list[str]
    ) -> tuple[dict[str, int], np.ndarray]:
        index: dict[str, int] = {}
        _truncation_warned = False
        for doc in docs:
            for tri in self._trigrams(doc):
                if (
                    tri not in index
                    and self._max_semantic_features is not None
                    and len(index) >= self._max_semantic_features
                ):
                    if not _truncation_warned:
                        logger.warning(
                            "[curator] max_semantic_features=%d reached: "
                            "semantic trigram index truncated "
                            "(unique trigrams exceed cap)",
                            self._max_semantic_features,
                        )
                        _truncation_warned = True
                    continue
                index.setdefault(tri, len(index))
        if not docs or not index:
            return index, np.empty((0, len(index)), dtype=np.float64)
        vectors = np.zeros((len(docs), len(index)), dtype=np.float64)
        for row, doc in enumerate(docs):
            for tri in self._trigrams(doc):
                col = index.get(tri)
                if col is not None:
                    vectors[row, col] += 1.0
        vectors = _l2_normalize(vectors)
        return index, vectors

    def _is_semantic_hit(self, doc: str) -> bool:
        vec = self._vectorize(doc)
        if vec is None:
            return False
        sims = self._eval_vectors @ vec
        return bool(sims.size and sims.max() >= self._semantic_threshold)

    def _vectorize(self, doc: str) -> np.ndarray | None:
        vec = np.zeros(len(self._trigram_index), dtype=np.float64)
        any_hit = False
        for tri in self._trigrams(doc):
            idx = self._trigram_index.get(tri)
            if idx is not None:
                vec[idx] += 1.0
                any_hit = True
        if not any_hit:
            return None
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else None

    @staticmethod
    def _trigrams(text: str) -> list[str]:
        cleaned = re.sub(r"\s+", " ", text.lower()).strip()
        if len(cleaned) < 3:
            return [cleaned] if cleaned else []
        return [cleaned[i : i + 3] for i in range(len(cleaned) - 2)]


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms
