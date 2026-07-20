"""Thin v1 taskset composition for the pretraining curator."""

from __future__ import annotations

import verifiers.v1 as vf

from .config import CuratorTasksetConfig
from .task import CuratorTask
from .taskdata import CuratorTaskData


# TaskT is invariant; Taskset only yields it.
class CuratorTaskset(
    vf.Taskset[CuratorTask, CuratorTasksetConfig]
):
    """Load the single curation row; behavior remains on ``CuratorTask``."""

    def load(self) -> list[CuratorTask]:
        config = self.config.task
        return [CuratorTask(CuratorTaskData.from_config(config), config)]


__all__ = ["CuratorTaskset"]
