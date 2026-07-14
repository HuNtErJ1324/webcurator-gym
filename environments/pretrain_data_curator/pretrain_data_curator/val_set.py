"""Held-out validation token stream for the downstream cross-entropy signal.

`Perf(M)` (the reward's downstream-loss term) is meant to be the cross-entropy a
fixed proxy-student — trained on the curated corpus — achieves on a *fixed,
held-out* token stream. This module owns that held-out set.

The default is the **NanoGPT speedrun** (``KellerJordan/modded-nanogpt``)
validation set: the FineWeb ``sample-10BT`` subset, GPT-2-BPE-tokenized, published
as ``kjj0/fineweb10B-gpt2`` (the ``fineweb_val_*.bin`` shard), scored as
cross-entropy in nats/token over the FIRST ``val_tokens`` tokens
(``val_tokens = 10_485_760`` in modded-nanogpt's ``train_gpt.py``).

The ``.bin`` format mirrors modded-nanogpt's ``_load_data_shard``: a 256-int32
header (``header[0]`` magic ``20240520``, ``header[1]`` version ``1``,
``header[2]`` token count), followed by ``num_tokens`` little-endian ``uint16``
GPT-2 token ids.

Loading goes through the SAME robustness machinery as Hub document fetches in
``hf_access`` — off-event-loop work, a per-attempt timeout, bounded retry/backoff,
a loop-local concurrency semaphore, and typed ``DatasetAccessError`` — plus a
process-level deterministic cache (keyed by ``(dataset_id, filename, val_tokens)``)
with single-flight, so the (large) shard is downloaded and parsed exactly once and
a fetch failure degrades to a typed error the caller turns into a sentinel rather
than crashing.
"""

from __future__ import annotations

import asyncio
import json
import weakref
from dataclasses import dataclass
from typing import Callable

import numpy as np
from pydantic import BaseModel, Field

from .hf_access import (
    DatasetAccessError,
    RetryPolicy,
    hf_fetch_semaphore,
    run_blocking_with_retry,
)
from .train_gpt import plan_val_windows

# --- NanoGPT speedrun validation-set spec (verified against modded-nanogpt) ---
# Source dataset (GPT-2-BPE-tokenized FineWeb sample-10BT) and the val shard.
NANOGPT_VAL_DATASET_ID = "kjj0/fineweb10B-gpt2"
NANOGPT_VAL_FILENAME = "fineweb_val_000000.bin"
NANOGPT_VAL_REPO_TYPE = "dataset"
# Tokenizer the shard is encoded with; the proxy-student must use the same BPE.
NANOGPT_VAL_TOKENIZER = "gpt2"
# modded-nanogpt train_gpt.py: ``val_tokens: int = 10485760`` — the validation
# slice is the FIRST this-many tokens of the val shard.
NANOGPT_VAL_TOKENS = 10_485_760

# modded-nanogpt .bin shard header layout (``_load_data_shard``).
SHARD_HEADER_INTS = 256
SHARD_HEADER_BYTES = SHARD_HEADER_INTS * 4  # 256 * int32
SHARD_MAGIC = 20240520
SHARD_VERSION = 1
_TOKEN_DTYPE = np.dtype("<u2")  # little-endian uint16 GPT-2 token ids


class ValidationSetConfig(BaseModel):
    """Held-out validation set for the downstream cross-entropy (``Perf``) signal.

    Defaults to the NanoGPT speedrun set: FineWeb ``sample-10BT`` GPT-2-BPE val
    tokens (``kjj0/fineweb10B-gpt2`` / ``fineweb_val_000000.bin``), the first
    ``val_tokens`` tokens, scored as cross-entropy in nats/token. Every field is
    bounded so a nonsensical val-set spec fails fast at load time.
    """

    dataset_id: str = Field(default=NANOGPT_VAL_DATASET_ID, min_length=1)
    filename: str = Field(default=NANOGPT_VAL_FILENAME, min_length=1)
    repo_type: str = Field(default=NANOGPT_VAL_REPO_TYPE, min_length=1)
    tokenizer: str = Field(default=NANOGPT_VAL_TOKENIZER, min_length=1)
    val_tokens: int = Field(default=NANOGPT_VAL_TOKENS, ge=1)


@dataclass(frozen=True)
class HeldOutValSet:
    """An immutable, GPT-2-BPE-tokenized held-out validation token stream."""

    dataset_id: str
    filename: str
    tokenizer: str
    tokens: np.ndarray  # 1-D uint16 GPT-2 token ids
    n_tokens: int

    def to_uint16_bytes(self) -> bytes:
        """Raw little-endian uint16 bytes (header-free) for upload to the trainer."""
        return np.ascontiguousarray(self.tokens, dtype=_TOKEN_DTYPE).tobytes()

    def as_dict(self) -> dict[str, object]:
        return {
            "dataset_id": self.dataset_id,
            "filename": self.filename,
            "tokenizer": self.tokenizer,
            "n_tokens": self.n_tokens,
        }


def parse_token_shard(
    raw: bytes,
    *,
    limit: int,
    dataset_id: str = NANOGPT_VAL_DATASET_ID,
    filename: str = NANOGPT_VAL_FILENAME,
    tokenizer: str = NANOGPT_VAL_TOKENIZER,
) -> HeldOutValSet:
    """Parse a modded-nanogpt ``.bin`` token shard and slice the first ``limit``.

    Validates the header (magic + version), reads the declared token count, and
    returns exactly the first ``min(limit, num_tokens)`` ``uint16`` tokens. A
    malformed shard raises a permanent ``DatasetAccessError(kind="bad_field")`` so
    it is surfaced (and not retried) like any other classified access failure.
    """
    if limit < 1:
        raise DatasetAccessError(
            f"val_tokens must be >= 1, got {limit}",
            kind="bad_field",
            dataset_id=dataset_id,
        )
    if len(raw) < SHARD_HEADER_BYTES:
        raise DatasetAccessError(
            f"{filename}: truncated header ({len(raw)} < {SHARD_HEADER_BYTES} bytes)",
            kind="bad_field",
            dataset_id=dataset_id,
        )
    header = np.frombuffer(raw[:SHARD_HEADER_BYTES], dtype="<i4")
    if int(header[0]) != SHARD_MAGIC:
        raise DatasetAccessError(
            f"{filename}: bad magic {int(header[0])} (expected {SHARD_MAGIC})",
            kind="bad_field",
            dataset_id=dataset_id,
        )
    if int(header[1]) != SHARD_VERSION:
        raise DatasetAccessError(
            f"{filename}: unsupported version {int(header[1])} (expected {SHARD_VERSION})",
            kind="bad_field",
            dataset_id=dataset_id,
        )
    num_tokens = int(header[2])
    body_bytes = raw[
        SHARD_HEADER_BYTES : SHARD_HEADER_BYTES + num_tokens * _TOKEN_DTYPE.itemsize
    ]
    # A truncated/corrupt shard can leave a dangling odd byte; np.frombuffer would
    # raise a bare ValueError, so classify it as a typed bad_field access failure.
    if len(body_bytes) % _TOKEN_DTYPE.itemsize != 0:
        raise DatasetAccessError(
            f"{filename}: token body has {len(body_bytes)} bytes, "
            f"not a multiple of {_TOKEN_DTYPE.itemsize}",
            kind="bad_field",
            dataset_id=dataset_id,
        )
    body = np.frombuffer(body_bytes, dtype=_TOKEN_DTYPE)
    n = min(int(limit), int(body.shape[0]))
    if n < 1:
        raise DatasetAccessError(
            f"{filename}: shard declares {num_tokens} tokens but body is empty",
            kind="bad_field",
            dataset_id=dataset_id,
        )
    tokens = np.ascontiguousarray(body[:n], dtype=_TOKEN_DTYPE)
    return HeldOutValSet(
        dataset_id=dataset_id,
        filename=filename,
        tokenizer=tokenizer,
        tokens=tokens,
        n_tokens=n,
    )


def mean_held_out_ce(n_tokens, block, window_loss_sum):
    """Mean cross-entropy (nats/token) over every held-out predictable position.

    ``window_loss_sum(start, length)`` returns the SUMMED token-level CE over that
    window's ``length`` targets. Pure and torch-free for testability; it raises
    (via ``plan_val_windows``) on an empty val set, and the denominator is the
    actual number of scored targets (``n_tokens - 1``, always > 0 here), so a
    zero-position eval degrades rather than reporting a bogus ``0.0``.
    """
    windows = plan_val_windows(n_tokens, block)
    loss_sum = 0.0
    total = 0
    for start, length in windows:
        loss_sum += float(window_loss_sum(start, length))
        total += int(length)
    return loss_sum / total


# Download a repo file to a local path. Injectable so the loader is testable
# without the Hub; the live default uses ``huggingface_hub.hf_hub_download``.
DownloadFn = Callable[[str, str, str], str]


def _default_download(dataset_id: str, filename: str, repo_type: str) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=dataset_id, filename=filename, repo_type=repo_type)


class ValTokenLoader:
    """Loads (and caches) the held-out validation token stream, robustly.

    Mirrors ``CorpusBuilder``'s deterministic-cache + single-flight discipline and
    reuses ``hf_access``'s off-loop/timeout/retry/semaphore/typed-error wrapper, so
    the val set is fetched through the same paths as every other external access —
    never around them. The val set is global (identical across rollouts), so the
    cache lives on the loader instance rather than in per-rollout state.
    """

    def __init__(
        self,
        config: ValidationSetConfig,
        *,
        download_fn: DownloadFn | None = None,
        retry_policy: RetryPolicy | None = None,
        fetch_limit: int = 8,
    ) -> None:
        self._config = config
        self._download_fn = download_fn or _default_download
        self._retry = retry_policy or RetryPolicy()
        self._fetch_limit = fetch_limit
        self._cache: dict[str, HeldOutValSet] = {}
        # Loop-local single-flight locks (one per cache key), so concurrent first
        # loads coalesce onto a single download+parse and a single cache write.
        self._locks: weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop, dict[str, asyncio.Lock]
        ] = weakref.WeakKeyDictionary()

    @property
    def cache_key(self) -> str:
        return json.dumps(
            [self._config.dataset_id, self._config.filename, self._config.val_tokens],
            separators=(",", ":"),
        )

    def _lock(self, key: str) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        locks = self._locks.get(loop)
        if locks is None:
            locks = {}
            self._locks[loop] = locks
        lock = locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            locks[key] = lock
        return lock

    async def load(self) -> HeldOutValSet:
        """Return the held-out val set, fetching once. Raises ``DatasetAccessError``.

        Double-checked locking against the process-level cache, so even under
        concurrent first calls the shard is downloaded and parsed exactly once.
        """
        key = self.cache_key
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        async with self._lock(key):
            cached = self._cache.get(key)
            if cached is not None:
                return cached
            val_set = await run_blocking_with_retry(
                self._load_sync,
                policy=self._retry,
                semaphore=hf_fetch_semaphore(self._fetch_limit),
                dataset_id=self._config.dataset_id,
            )
            self._cache[key] = val_set
            return val_set

    def _load_sync(self) -> HeldOutValSet:
        cfg = self._config
        path = self._download_fn(cfg.dataset_id, cfg.filename, cfg.repo_type)
        # Only the header plus the first ``val_tokens`` tokens are needed, so we
        # never materialize more of the (multi-GB) shard than the slice requires.
        want = SHARD_HEADER_BYTES + cfg.val_tokens * _TOKEN_DTYPE.itemsize
        with open(path, "rb") as fh:
            raw = fh.read(want)
        return parse_token_shard(
            raw,
            limit=cfg.val_tokens,
            dataset_id=cfg.dataset_id,
            filename=cfg.filename,
            tokenizer=cfg.tokenizer,
        )
