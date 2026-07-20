"""Cutoff-filtered Hugging Face data access."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from itertools import islice
from typing import Any, Callable, Protocol, TypeVar

from huggingface_hub.errors import (
    EntryNotFoundError,
    GatedRepoError,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from ..gpu.scoring_shared import CHARS_PER_TOKEN, estimate_tokens
from .async_utils import AdaptiveSemaphore, run_shielded

logger = logging.getLogger(__name__)

T = TypeVar("T")
_DRAIN_TASKS: set[asyncio.Task[None]] = set()

# Non-retryable Hub errors; all others treated as transient.
PERMANENT_KINDS = frozenset(
    {
        "missing",
        "auth",
        "bad_split",
        "bad_config",
        "bad_field",
        "script_dataset",
    }
)


class DatasetAccessError(RuntimeError):
    """A classified, structured failure accessing the Hugging Face Hub."""

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
    """Map an arbitrary access exception onto a stable `DatasetAccessError.kind`."""
    if isinstance(exc, DatasetAccessError):
        return exc.kind
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)):
        return "timeout"
    if isinstance(exc, GatedRepoError):
        return "auth"
    if isinstance(
        exc, (RepositoryNotFoundError, EntryNotFoundError, RevisionNotFoundError)
    ):
        return "missing"
    name = type(exc).__name__
    msg = str(exc).lower()
    if "dataset scripts are no longer supported" in msg:
        return "script_dataset"
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
    """Deterministic cache identity for a sampled document slice."""

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


async def run_blocking_with_retry(
    fn: Callable[[], T],
    *,
    policy: RetryPolicy,
    semaphore: AdaptiveSemaphore,
    dataset_id: str | None = None,
    timeout: float | None = None,
) -> T:
    """Run a blocking callable off the event loop with bound+timeout+retry."""
    attempt_timeout = policy.timeout if timeout is None else timeout

    def drain_and_release(
        worker: asyncio.Task[T], limiter: AdaptiveSemaphore
    ) -> None:
        async def drain() -> None:
            try:
                await run_shielded(worker)
            except BaseException:
                pass
            finally:
                await limiter.release()

        task = asyncio.create_task(drain())
        _DRAIN_TASKS.add(task)
        task.add_done_callback(_DRAIN_TASKS.discard)

    async def attempt() -> T:
        acquired = False
        delegated_release = False
        try:
            await semaphore.acquire()
            acquired = True
            worker = asyncio.create_task(asyncio.to_thread(fn))
            try:
                return await asyncio.wait_for(
                    asyncio.shield(worker), timeout=attempt_timeout
                )
            except (asyncio.TimeoutError, TimeoutError):
                # On timeout, drain the slot in background; return promptly.
                delegated_release = True
                drain_and_release(worker, semaphore)
                raise
            except asyncio.CancelledError:
                delegated_release = True
                drain_and_release(worker, semaphore)
                raise
        except (asyncio.TimeoutError, TimeoutError) as exc:
            raise DatasetAccessError(
                f"access to {dataset_id or 'dataset'} timed out after "
                f"{attempt_timeout}s",
                kind="timeout",
                dataset_id=dataset_id,
            ) from exc
        except DatasetAccessError:
            raise
        except Exception as exc:  # noqa: BLE001 - classify, then tenacity decides
            raise DatasetAccessError(
                str(exc) or type(exc).__name__,
                kind=classify_exception(exc),
                dataset_id=dataset_id,
            ) from exc
        finally:
            if acquired and not delegated_release:
                await semaphore.release()

    def transient(exc: BaseException) -> bool:
        return isinstance(exc, DatasetAccessError) and exc.kind not in PERMANENT_KINDS

    retryer = AsyncRetrying(
        stop=stop_after_attempt(policy.attempts),
        wait=wait_exponential(multiplier=policy.base_delay / 2, max=policy.max_delay),
        retry=retry_if_exception(transient),
        reraise=True,
    )
    try:
        return await retryer(attempt)
    except DatasetAccessError as exc:
        if exc.kind not in PERMANENT_KINDS:
            logger.warning(
                "HF access failed after %d attempts: %s", policy.attempts, exc
            )
        raise


async def fetch_documents(
    sample_fn: Callable[[str, str | None, str, str | None, int], list[str]],
    key: FetchKey,
    *,
    policy: RetryPolicy,
    semaphore: AdaptiveSemaphore,
) -> list[str]:
    """Fetch documents for `key` via `sample_fn`, robustly."""

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


def extract_text_from_row(row: dict, text_field: str | None) -> str | None:
    """Extract one non-empty text value using the shared HF/local heuristic."""
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
                    str(row.get("query", "")) + " " + str(row.get("response", ""))
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
    return value if isinstance(value, str) and value.strip() else None


class HuggingFaceDatasetClient:
    """Live Hugging Face Hub client (streaming document sampling)."""

    def __init__(
        self,
        token: str | None = None,
        *,
        token_env: str = "HF_TOKEN",
    ) -> None:
        token = token or os.environ.get(token_env)
        if not token:
            raise RuntimeError(
                f"Hugging Face token environment variable {token_env!r} is "
                "required for rollouts; set it in the env-server container "
                "before the first Hub API use."
            )

        self._token = token

    @staticmethod
    def _reject_script_dataset(dataset_id: str) -> None:
        raise DatasetAccessError(
            f"{dataset_id} is a script-based Hugging Face dataset, which the "
            "installed datasets runtime cannot load. Download its raw files in "
            "your shell with `hf download <repo> --repo-type dataset` or `curl`, "
            "then convert them to plain text or JSONL in your workspace. Cite the "
            'result in the manifest as a source with `kind: "local"` and '
            '`local_path: "<relative-path>"`. Set `local_format` to `"auto"`, '
            '`"jsonl"`, or `"txt"`, and set `text_field` to the document text '
            "column or `null` for auto-detection.",
            kind="script_dataset",
            dataset_id=dataset_id,
        )

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
        except RuntimeError as exc:
            # datasets detects script-backed repos at resolve time.
            if classify_exception(exc) == "script_dataset":
                self._reject_script_dataset(dataset_id)
            raise
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
            try:
                stream = load_dataset(
                    dataset_id,
                    name=resolved,
                    split=split,
                    streaming=True,
                    token=self._token,
                )
            except RuntimeError as retry_exc:
                if classify_exception(retry_exc) == "script_dataset":
                    self._reject_script_dataset(dataset_id)
                raise
        docs: list[str] = []
        for row in islice(stream, max(n, 0)):
            if not isinstance(row, dict):
                continue
            value = extract_text_from_row(row, text_field)
            if value is not None:
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
