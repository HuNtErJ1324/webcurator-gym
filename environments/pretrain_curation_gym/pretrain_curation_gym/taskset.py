"""Thin v1 taskset composition for the pretraining curator."""

from __future__ import annotations

import verifiers.v1 as vf

from .config import CuratorTasksetConfig
from .task import CuratorTask
from .taskdata import CuratorTaskData


# The framework's TaskT bound is invariant over Task state/config today, even
# though Taskset only yields it. The concrete specialization is runtime-valid.
class CuratorTaskset(  # pyright: ignore[reportInvalidTypeArguments]
    vf.Taskset[CuratorTask, CuratorTasksetConfig]  # pyright: ignore[reportInvalidTypeArguments]
):
    """Load the single curation row; behavior remains on ``CuratorTask``."""

    def load(self) -> list[CuratorTask]:
        config = self.config.task
        return [CuratorTask(CuratorTaskData.from_config(config), config)]


__all__ = ["CuratorTaskset"]
