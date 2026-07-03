"""Cutoff-filtered Hugging Face universe access.

The `DatasetSearchClient` Protocol keeps the environment testable with a fake
client, while `HuggingFaceDatasetClient` provides the live Hub implementation.

External access is wrapped with timeouts, bounded retry/backoff, typed errors
(`DatasetAccessError`), and a process-wide concurrency bound so a flaky or
missing dataset never crashes a tool call or the reward pass.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import weakref
from dataclasses import dataclass
from typing import Any, Callable, Protocol, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Error kinds that are deterministic facts about the request and must NOT be
# retried (the answer will not change); everything else is treated as transient.
PERMANENT_KINDS = frozenset({"missing", "auth", "bad_split", "bad_config", "bad_field"})


class DatasetAccessError(RuntimeError):
    """A classified, structured failure accessing the Hugging Face Hub.

    `kind` is one of: ``missing``, ``auth``, ``bad_split``, ``bad_config``,
    ``bad_field``, ``network``, ``timeout``, ``unknown``.
    """

    def __init__(
        self, message: str, *, kind: str = "unknown", dataset_id: str | None = None
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.dataset_id = dataset_id

    def as_dict(self) -> dict[str, Any]:
        return {
            "error": str(self),
            "error_kind": self.kind,
            "dataset_id": self.dataset_id,
        }


def classify_exception(exc: BaseException) -> str:
    """Map an arbitrary access exception onto a stable `DatasetAccessError.kind`.

    Classification is intentionally duck-typed (by class name and message text)
    so it works regardless of which optional `datasets`/`huggingface_hub` error
    classes are importable at runtime, and so tests can simulate failures with
    plain exceptions.
    """
    if isinstance(exc, DatasetAccessError):
        return exc.kind
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return "timeout"
    name = type(exc).__name__
    msg = str(exc).lower()
    if name in {
        "DatasetNotFoundError",
        "RepositoryNotFoundError",
        "EntryNotFoundError",
    }:
        return "missing"
    if name in {"GatedRepoError", "UnauthorizedError", "PaymentRequiredError"}:
        return "auth"
    if name in {"ConnectionError", "ConnectTimeout", "ReadTimeout", "Timeout"}:
        return "network"
    if "401" in msg or "403" in msg or "unauthorized" in msg or "authentication" in msg:
        return "auth"
    if "gated" in msg or "private" in msg or "access to this dataset" in msg:
        return "auth"
    if (
        "404" in msg
        or "not found" in msg
        or "doesn't exist" in msg
        or "does not exist" in msg
    ):
        return "missing"
    if "split" in msg and ("unknown" in msg or "invalid" in msg or "not" in msg):
        return "bad_split"
    if "config" in msg or "builderconfig" in msg or "subset" in msg:
        return "bad_config"
    if (
        isinstance(exc, KeyError)
        or "column" in msg
        or "text_field" in msg
        or "field" in msg
        or "key" in msg
    ):
        return "bad_field"
    if isinstance(exc, (ConnectionError, OSError)):
        return "network"
    if "timed out" in msg or "timeout" in msg:
        return "timeout"
    return "unknown"


@dataclass(frozen=True)
class FetchKey:
    """Deterministic cache identity for a sampled document slice.

    Two fetches with the same `(dataset_id, config, split, text_field, n)` MUST
    observe identical documents (so a preview and final scoring agree, and cost
    is charged once).
    """

    dataset_id: str
    config: str | None
    split: str
    text_field: str | None
    n: int

    def as_str(self) -> str:
        return json.dumps(
            [
                self.dataset_id,
                self.config,
                self.split,
                "__auto__" if self.text_field is None else self.text_field,
                self.n,
            ],
            separators=(",", ":"),
        )


@dataclass
class RetryPolicy:
    """Bounded retry/backoff + per-attempt timeout for blocking external calls."""

    attempts: int = 3
    base_delay: float = 0.05
    max_delay: float = 2.0
    timeout: float = 30.0
    per_doc_seconds: float = 0.25

    def timeout_for_documents(self, n: int) -> float:
        """Scale a streaming fetch attempt by the requested document count."""
        return self.timeout + max(n, 0) * self.per_doc_seconds


# Loop-local concurrency bound for Hub fetches. Semaphores are bound to the
# running event loop on first use, so we key one per loop (a rare, explicitly
# sanctioned process-level handle). Finished loops are dropped automatically.
_FETCH_SEMAPHORES: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore]" = weakref.WeakKeyDictionary()


def loop_local_semaphore(
    registry: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore]",
    limit: int,
) -> asyncio.Semaphore:
    """Return the loop-local semaphore, bound to the MOST RESTRICTIVE limit yet
    requested for that loop (a later, smaller limit tightens it, so a second env
    instance sharing the loop never inherits a larger bound than it asked for).
    """
    loop = asyncio.get_running_loop()
    sem = registry.get(loop)
    if sem is None or getattr(sem, "_pdc_limit", limit) > limit:
        sem = asyncio.Semaphore(limit)
        sem._pdc_limit = limit  # remember the bound, so a smaller later limit wins
        registry[loop] = sem
    return sem


def hf_fetch_semaphore(limit: int) -> asyncio.Semaphore:
    """Process-wide (loop-local) bound on concurrent Hugging Face fetches."""
    return loop_local_semaphore(_FETCH_SEMAPHORES, limit)


async def run_blocking_with_retry(
    fn: Callable[[], T],
    *,
    policy: RetryPolicy,
    semaphore: asyncio.Semaphore,
    dataset_id: str | None = None,
    timeout: float | None = None,
) -> T:
    """Run a blocking callable off the event loop with bound+timeout+retry.

    Offloads `fn` via `asyncio.to_thread`, caps each attempt with
    `asyncio.wait_for`, retries transient failures with exponential backoff, and
    raises a classified `DatasetAccessError` on permanent failure or exhaustion.
    """
    last: DatasetAccessError | None = None
    attempt_timeout = policy.timeout if timeout is None else timeout
    for attempt in range(1, policy.attempts + 1):
        try:
            async with semaphore:
                return await asyncio.wait_for(
                    asyncio.to_thread(fn), timeout=attempt_timeout
                )
        except (asyncio.TimeoutError, TimeoutError):
            last = DatasetAccessError(
                f"access to {dataset_id or 'dataset'} timed out after "
                f"{attempt_timeout}s",
                kind="timeout",
                dataset_id=dataset_id,
            )
        except DatasetAccessError as exc:
            last = exc
            if exc.kind in PERMANENT_KINDS:
                raise
        except Exception as exc:  # noqa: BLE001 - classify, then decide retry
            kind = classify_exception(exc)
            err = DatasetAccessError(
                str(exc) or type(exc).__name__, kind=kind, dataset_id=dataset_id
            )
            if kind in PERMANENT_KINDS:
                raise err from exc
            last = err
        if attempt < policy.attempts:
            await asyncio.sleep(
                min(policy.base_delay * (2 ** (attempt - 1)), policy.max_delay)
            )
    assert last is not None  # loop runs at least once
    logger.warning("HF access failed after %d attempts: %s", policy.attempts, last)
    raise last


async def fetch_documents(
    sample_fn: Callable[[str, str | None, str, str | None, int], list[str]],
    key: FetchKey,
    *,
    policy: RetryPolicy,
    semaphore: asyncio.Semaphore,
) -> list[str]:
    """Fetch documents for `key` via `sample_fn`, robustly. Raises DatasetAccessError."""

    def _call() -> list[str]:
        return list(
            sample_fn(key.dataset_id, key.config, key.split, key.text_field, key.n)
        )

    return await run_blocking_with_retry(
        _call,
        policy=policy,
        semaphore=semaphore,
        dataset_id=key.dataset_id,
        timeout=policy.timeout_for_documents(key.n),
    )


class DatasetSearchClient(Protocol):
    def sample_documents(
        self,
        dataset_id: str,
        config: str | None,
        split: str,
        text_field: str | None,
        n: int,
    ) -> list[str]: ...


class HuggingFaceDatasetClient:
    """Live Hugging Face Hub client (streaming document sampling)."""

    def __init__(
        self, token: str | None = None, *, token_env: str = "HF_TOKEN"
    ) -> None:
        token = token or os.environ.get(token_env)
        if not token:
            raise RuntimeError(
                f"Hugging Face token environment variable {token_env!r} is "
                "required for rollouts; set it in the env-server container "
                "before the first Hub API use."
            )

        self._token = token

    def sample_documents(
        self,
        dataset_id: str,
        config: str | None,
        split: str,
        text_field: str | None,
        n: int,
    ) -> list[str]:
        from datasets import get_dataset_config_names, load_dataset

        try:
            stream = load_dataset(
                dataset_id,
                name=config,
                split=split,
                streaming=True,
                token=self._token,
            )
        except ValueError as exc:
            if config is not None or "config name is missing" not in str(exc).lower():
                raise
            configs = get_dataset_config_names(dataset_id, token=self._token)
            resolved = self._preferred_config(configs)
            if resolved is None:
                raise
            logger.info(
                "dataset %s has no default config; selected %s",
                dataset_id,
                resolved,
            )
            stream = load_dataset(
                dataset_id,
                name=resolved,
                split=split,
                streaming=True,
                token=self._token,
            )
        docs: list[str] = []
        for i, row in enumerate(stream):
            if i >= n:
                break
            if not isinstance(row, dict):
                continue

            value = row.get(text_field) if text_field is not None else None
            if not isinstance(value, str) or not value.strip():
                candidates: list[object] = []
                for field in (
                    "text",
                    "content",
                    "passage",
                    "document",
                    "abstract",
                    "body",
                    "article",
                    "sentence",
                    "query",
                    "answer",
                    "response",
                    "output",
                    "instruction",
                    "input",
                    "context",
                ):
                    if field == "query" and (
                        row.get("query") is not None or row.get("response") is not None
                    ):
                        candidates.append(
                            str(row.get("query", ""))
                            + " "
                            + str(row.get("response", ""))
                        )
                    candidates.append(row.get(field))
                value = next(
                    (
                        candidate
                        for candidate in candidates
                        if isinstance(candidate, str) and candidate.strip()
                    ),
                    None,
                )
            if isinstance(value, str) and value.strip():
                docs.append(value)
        return docs

    @staticmethod
    def _preferred_config(configs: list[str]) -> str | None:
        """Choose a stable English/default config when a dataset has no default."""
        if not configs:
            return None
        for preferred in ("default", "en", "english", "plain_text"):
            if preferred in configs:
                return preferred
        for config in configs:
            if config.endswith(".en") or config.startswith("en."):
                return config
        return configs[0]


def estimate_tokens(text: str) -> int:
    """Cheap, deterministic token estimate (~4 chars/token, min by whitespace)."""
    if not text:
        return 0
    words = len(text.split())
    chars = len(text)
    return max(words, chars // 4)
