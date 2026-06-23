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
import re
from dataclasses import dataclass

import numpy as np

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
    ) -> None:
        self._num_perm = num_perm
        self._shingle_k = shingle_k
        self._fuzzy_threshold = fuzzy_threshold
        self._semantic_threshold = semantic_threshold
        rng = np.random.default_rng(seed)
        mask = (1 << 32) - 1
        self._a = rng.integers(1, mask, size=num_perm, dtype=np.uint64)
        self._b = rng.integers(0, mask, size=num_perm, dtype=np.uint64)
        self._prime = np.uint64((1 << 61) - 1)

        self._eval_hashes = {_doc_hash(d) for d in eval_docs}
        self._eval_minhashes = np.array(
            [self._minhash(d) for d in eval_docs], dtype=np.uint64
        ) if eval_docs else np.empty((0, num_perm), dtype=np.uint64)
        self._trigram_index, self._eval_vectors = self._build_vectors(eval_docs)

    def score(self, docs: list[str]) -> LeakageScores:
        if not docs:
            return LeakageScores(0.0, 0.0, 0.0, 0.0)
        exact = self._exact_fraction(docs)
        fuzzy = self._fuzzy_fraction(docs)
        semantic = self._semantic_fraction(docs)
        overall = max(exact, fuzzy, semantic)
        return LeakageScores(exact, fuzzy, semantic, overall)

    def _exact_fraction(self, docs: list[str]) -> float:
        if not self._eval_hashes:
            return 0.0
        hits = sum(1 for d in docs if _doc_hash(d) in self._eval_hashes)
        return hits / len(docs)

    def _minhash(self, text: str) -> np.ndarray:
        shingles = _word_shingles(text, self._shingle_k)
        if not shingles:
            return np.full(self._num_perm, np.uint64(0), dtype=np.uint64)
        arr = np.array(list(shingles), dtype=np.uint64)
        # (num_perm, num_shingles) affine permutations, then per-row minimum.
        hashed = (self._a[:, None] * arr[None, :] + self._b[:, None]) % self._prime
        return hashed.min(axis=1)

    def _fuzzy_fraction(self, docs: list[str]) -> float:
        if self._eval_minhashes.shape[0] == 0:
            return 0.0
        hits = 0
        for doc in docs:
            sig = self._minhash(doc)
            equal = (self._eval_minhashes == sig[None, :]).mean(axis=1)
            if equal.max() >= self._fuzzy_threshold:
                hits += 1
        return hits / len(docs)

    def _build_vectors(
        self, docs: list[str]
    ) -> tuple[dict[str, int], np.ndarray]:
        index: dict[str, int] = {}
        for doc in docs:
            for tri in self._trigrams(doc):
                index.setdefault(tri, len(index))
        if not docs or not index:
            return index, np.empty((0, len(index)), dtype=np.float64)
        vectors = np.zeros((len(docs), len(index)), dtype=np.float64)
        for row, doc in enumerate(docs):
            for tri in self._trigrams(doc):
                vectors[row, index[tri]] += 1.0
        vectors = _l2_normalize(vectors)
        return index, vectors

    def _semantic_fraction(self, docs: list[str]) -> float:
        if self._eval_vectors.shape[0] == 0 or not self._trigram_index:
            return 0.0
        hits = 0
        for doc in docs:
            vec = self._vectorize(doc)
            if vec is None:
                continue
            sims = self._eval_vectors @ vec
            if sims.size and sims.max() >= self._semantic_threshold:
                hits += 1
        return hits / len(docs)

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
