"""Thin v1 taskset composition for the pretraining curator."""

from __future__ import annotations

import verifiers.v1 as vf

from .config import CuratorTasksetConfig
from .runtime_config import derive_task_runtime_updates
from .task import CuratorTask
from .tasks import CuratorTaskData


# The framework's TaskT bound is invariant over Task state/config today, even
# though Taskset only yields it. The concrete specialization is runtime-valid.
class CuratorTaskset(  # pyright: ignore[reportInvalidTypeArguments]
    vf.Taskset[CuratorTask, CuratorTasksetConfig]  # pyright: ignore[reportInvalidTypeArguments]
):
    """Load the single curation row; behavior remains on ``CuratorTask``."""

    def load(self) -> list[CuratorTask]:
        config = self.config.task
        data = CuratorTaskData.from_config(config)
        runtime_fields = derive_task_runtime_updates(
            config.curator.proxy_student,
            use_real_trainer=config.curator.use_real_trainer,
        )
        return [CuratorTask(data.model_copy(update=runtime_fields), config)]


__all__ = ["CuratorTaskset"]
