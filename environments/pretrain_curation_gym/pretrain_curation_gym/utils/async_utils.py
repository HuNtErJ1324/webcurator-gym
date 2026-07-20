"""Asyncio adapters for blocking scoring work."""

from __future__ import annotations

import asyncio
import weakref
from collections.abc import Callable
from typing import ParamSpec, TypeVar

from verifiers.v1.utils.aio import (
    run_shielded as run_shielded,
)

P = ParamSpec("P")
R = TypeVar("R")


async def run_blocking_drained(
    fn: Callable[P, R], /, *args: P.args, **kwargs: P.kwargs
) -> R:
    """Run ``fn`` off-loop and finish its thread before propagating cancellation."""
    return await run_shielded(asyncio.to_thread(fn, *args, **kwargs))


class LoopLocalLocks:
    """Per-event-loop, per-key ``asyncio.Lock`` registry for single-flight work."""

    def __init__(self) -> None:
        self._locks: weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop, dict[str, asyncio.Lock]
        ] = weakref.WeakKeyDictionary()

    def get(self, key: str) -> asyncio.Lock:
        """Return the loop-local lock for ``key``, creating it on first use."""
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

    def discard(self, key: str) -> None:
        """Drop ``key``'s lock once its result is cached (bounds growth)."""
        loop = asyncio.get_running_loop()
        locks = self._locks.get(loop)
        if locks is not None:
            locks.pop(key, None)


class AdaptiveSemaphore:
    """Semaphore whose limit may tighten without forgetting active holders."""

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._active = 0
        self._condition = asyncio.Condition()

    def tighten(self, limit: int) -> None:
        self._limit = min(self._limit, limit)

    async def acquire(self) -> None:
        async with self._condition:
            await self._condition.wait_for(lambda: self._active < self._limit)
            self._active += 1

    async def release(self) -> None:
        async with self._condition:
            self._active -= 1
            self._condition.notify_all()

    async def __aenter__(self) -> "AdaptiveSemaphore":
        await self.acquire()
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.release()


class LoopLocalSemaphore:
    """Loop-local adaptive semaphore using the most restrictive requested limit."""

    def __init__(self) -> None:
        self._semaphores: weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop, AdaptiveSemaphore
        ] = weakref.WeakKeyDictionary()

    def get(self, limit: int) -> AdaptiveSemaphore:
        if limit < 1:
            raise ValueError(f"semaphore limit must be >= 1, got {limit}")
        loop = asyncio.get_running_loop()
        entry = self._semaphores.get(loop)
        if entry is None:
            entry = AdaptiveSemaphore(limit)
            self._semaphores[loop] = entry
        else:
            entry.tighten(limit)
        return entry


_FETCH_SEMAPHORES = LoopLocalSemaphore()
_TRAIN_SEMAPHORES = LoopLocalSemaphore()
_DECON_SEMAPHORES = LoopLocalSemaphore()


def hf_fetch_semaphore(limit: int) -> AdaptiveSemaphore:
    """Process-wide (loop-local) bound on concurrent Hugging Face fetches."""
    return _FETCH_SEMAPHORES.get(limit)


def training_semaphore(limit: int) -> AdaptiveSemaphore:
    """Process-wide bound on concurrent sandbox proxy-student training jobs."""
    return _TRAIN_SEMAPHORES.get(limit)


def decon_semaphore(limit: int) -> AdaptiveSemaphore:
    """Process-wide bound on concurrent Decon subprocesses (CPU/RAM heavy)."""
    return _DECON_SEMAPHORES.get(limit)


__all__ = [
    "AdaptiveSemaphore",
    "LoopLocalLocks",
    "LoopLocalSemaphore",
    "decon_semaphore",
    "hf_fetch_semaphore",
    "run_blocking_drained",
    "run_shielded",
    "training_semaphore",
]
