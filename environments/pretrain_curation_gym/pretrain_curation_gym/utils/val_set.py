"""Held-out validation token stream for the downstream cross-entropy signal."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import partial
from typing import Callable

import numpy as np
from pydantic import BaseModel, Field

from .async_utils import LoopLocalLocks, hf_fetch_semaphore
from .hf_access import (
    DatasetAccessError,
    RetryPolicy,
    run_blocking_with_retry,
)
from ..gpu.train_gpt import plan_val_windows

NANOGPT_VAL_DATASET_ID = "kjj0/fineweb10B-gpt2"
NANOGPT_VAL_FILENAME = "fineweb_val_000000.bin"
NANOGPT_VAL_REPO_TYPE = "dataset"
# Must match proxy-student BPE.
NANOGPT_VAL_TOKENIZER = "gpt2"
NANOGPT_VAL_TOKENS = 10_485_760

SHARD_HEADER_INTS = 256
SHARD_HEADER_BYTES = SHARD_HEADER_INTS * 4
SHARD_MAGIC = 20240520
SHARD_VERSION = 1
_TOKEN_DTYPE = np.dtype("<u2")


class ValidationSetConfig(BaseModel):
    """Held-out validation set for the downstream cross-entropy (``Perf``) signal."""

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
    tokens: np.ndarray
    n_tokens: int

    def to_uint16_bytes(self) -> bytes:
        """Raw little-endian uint16 bytes (header-free) for upload to the trainer."""
        return np.ascontiguousarray(self.tokens, dtype=_TOKEN_DTYPE).tobytes()


def parse_token_shard(
    raw: bytes,
    *,
    limit: int,
    dataset_id: str = NANOGPT_VAL_DATASET_ID,
    filename: str = NANOGPT_VAL_FILENAME,
    tokenizer: str = NANOGPT_VAL_TOKENIZER,
) -> HeldOutValSet:
    """Parse a modded-nanogpt ``.bin`` token shard and slice the first ``limit``."""
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
    """Mean cross-entropy (nats/token) over every held-out predictable position."""
    windows = plan_val_windows(n_tokens, block)
    loss_sum = 0.0
    total = 0
    for start, length in windows:
        loss_sum += float(window_loss_sum(start, length))
        total += int(length)
    return loss_sum / total


DownloadFn = Callable[[str, str, str], str]


def _default_download(
    dataset_id: str,
    filename: str,
    repo_type: str,
    *,
    token: str | None = None,
) -> str:
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=dataset_id,
        filename=filename,
        repo_type=repo_type,
        token=token,
    )


class ValTokenLoader:
    """Loads (and caches) the held-out validation token stream, robustly."""

    def __init__(
        self,
        config: ValidationSetConfig,
        *,
        download_fn: DownloadFn | None = None,
        token: str | None = None,
        retry_policy: RetryPolicy | None = None,
        fetch_limit: int = 8,
    ) -> None:
        self._config = config
        self._download_fn = download_fn or partial(_default_download, token=token)
        self._retry = retry_policy or RetryPolicy()
        self._fetch_limit = fetch_limit
        self._cache: dict[str, HeldOutValSet] = {}
        self._locks = LoopLocalLocks()

    @property
    def cache_key(self) -> str:
        return json.dumps(
            [self._config.dataset_id, self._config.filename, self._config.val_tokens],
            separators=(",", ":"),
        )

    async def load(self) -> HeldOutValSet:
        """Return the held-out val set, fetching once."""
        key = self.cache_key
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        async with self._locks.get(key):
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
        # Slice header + val_tokens only (shard is multi-GB).
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
