"""Asyncio adapters for blocking scoring work.

One shared home for the three loop-bound patterns the environment needs:
draining offloaded work through cancellation (``run_blocking_drained``),
single-flight keyed locks (``LoopLocalLocks``), and most-restrictive-limit
semaphores (``LoopLocalSemaphore``). The drain semantics come from the
framework's ``run_shielded`` rather than a local re-implementation.
"""

from __future__ import annotations

import asyncio
import weakref
from collections.abc import Callable
from typing import ParamSpec, TypeVar

from verifiers.v1.utils.aio import (  # pyright: ignore[reportMissingImports]
    run_shielded as run_shielded,
)

P = ParamSpec("P")
R = TypeVar("R")


async def run_blocking_drained(
    fn: Callable[P, R], /, *args: P.args, **kwargs: P.kwargs
) -> R:
    """Run ``fn`` off-loop and finish its thread before propagating cancellation.

    Cancelling an ``asyncio.to_thread`` await does not stop the underlying
    function. Scoring callers own temporary corpus files (and hold semaphore
    slots or locks around offloaded writes), so control must not return to
    cleanup — or release a guard — while a worker thread can still be touching
    those files. ``run_shielded`` provides the framework-native drain: the
    worker always runs to completion, cancellation wins, and a worker error
    raised during cancellation is chained under the ``CancelledError`` instead
    of being silently lost.
    """
    return await run_shielded(asyncio.to_thread(fn, *args, **kwargs))


class LoopLocalLocks:
    """Per-event-loop, per-key ``asyncio.Lock`` registry for single-flight work.

    Locks bind to their running loop, and rollout state must stay
    JSON-serializable, so single-flight guards live in registries like this
    one — keyed first by loop (weakly, so finished loops are dropped) and then
    by caller key — never in state.
    """

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


class LoopLocalSemaphore:
    """Loop-local semaphore bound to the MOST RESTRICTIVE limit yet requested.

    Semaphores bind to the running event loop on first use, so one is kept per
    loop (a rare, explicitly sanctioned process-level handle). A later, smaller
    limit tightens the bound, so a second env instance sharing the loop never
    inherits a larger bound than it asked for. Finished loops are dropped
    automatically.
    """

    def __init__(self) -> None:
        # Each loop maps to its (granted limit, semaphore) pair; the recorded
        # limit is what lets a smaller later request win.
        self._semaphores: weakref.WeakKeyDictionary[
            asyncio.AbstractEventLoop, tuple[int, asyncio.Semaphore]
        ] = weakref.WeakKeyDictionary()

    def get(self, limit: int) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        entry = self._semaphores.get(loop)
        if entry is None or entry[0] > limit:
            entry = (limit, asyncio.Semaphore(limit))
            self._semaphores[loop] = entry
        return entry[1]


__all__ = [
    "LoopLocalLocks",
    "LoopLocalSemaphore",
    "run_blocking_drained",
    "run_shielded",
]
