"""Small asyncio adapters for blocking scoring work."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")


async def run_blocking_drained(fn: Callable[P, R], /, *args: P.args, **kwargs: P.kwargs) -> R:
    """Run ``fn`` off-loop and finish its thread before propagating cancellation.

    Cancelling an ``asyncio.to_thread`` await does not stop the underlying
    function. Scoring callers own temporary corpus files, so they must not
    return control to cleanup while a worker can still be reading those files.
    """
    worker = asyncio.create_task(asyncio.to_thread(fn, *args, **kwargs))
    cancelled = False
    while not worker.done():
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError:
            task = asyncio.current_task()
            if task is None or not task.cancelling():
                raise
            cancelled = True
        except BaseException:
            if not cancelled:
                raise
            break

    if cancelled:
        # Retrieve any worker exception before preserving caller cancellation.
        try:
            worker.result()
        except BaseException:
            pass
        raise asyncio.CancelledError()
    return worker.result()
