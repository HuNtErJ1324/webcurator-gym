"""Configuration at the three v1 ownership boundaries."""

from __future__ import annotations

from pydantic import Field, field_validator, model_validator

import verifiers.v1 as vf

from .leakage import DEFAULT_DECON_BINARY
from .models import CuratorConfig, MANIFEST_FILENAME

ENV_ID = "pretrain-curation-gym"


class CuratorTaskConfig(vf.TaskConfig):
    """Knobs consumed by one curation task and its scoring lifecycle."""

    curator: CuratorConfig = Field(default_factory=CuratorConfig)
    hf_token_env: str = "HF_TOKEN"
    manifest_filename: str = MANIFEST_FILENAME
    decon_binary: str = DEFAULT_DECON_BINARY
    decon_evals_dir: str | None = None
    decon_threshold: float = 0.2
    screen_val_set: bool = True

    @field_validator("manifest_filename")
    @classmethod
    def workspace_root_filename(cls, value: str) -> str:
        if not value or value in {".", ".."} or "/" in value or "\\" in value:
            raise ValueError("manifest_filename must be a runtime-workspace filename")
        return value


class CuratorTasksetConfig(vf.TasksetConfig):
    """The taskset owns one typed task configuration and no duplicate knobs."""

    id: str = ENV_ID
    # Plugin configs intentionally specialize the framework's mutable Pydantic
    # field; pyright cannot express that runtime narrowing through SerializeAsAny.
    task: CuratorTaskConfig = Field(  # pyright: ignore[reportIncompatibleVariableOverride]
        default_factory=CuratorTaskConfig
    )


class CuratorEnvConfig(vf.EnvConfig):
    """Concrete composition config used by ``load_environment``."""

    taskset: CuratorTasksetConfig = Field(  # pyright: ignore[reportIncompatibleVariableOverride]
        default_factory=CuratorTasksetConfig
    )
    harness: vf.HarnessConfig = Field(
        default_factory=lambda: vf.HarnessConfig(id="default")
    )
    max_turns: int | None = 64

    @model_validator(mode="after")
    def align_turn_limits(self) -> "CuratorEnvConfig":
        task_limit = self.taskset.task.curator.max_turns
        if self.max_turns is None:
            self.max_turns = task_limit
        elif self.max_turns != task_limit:
            raise ValueError(
                "max_turns and taskset.task.curator.max_turns must match "
                f"({self.max_turns} != {task_limit})"
            )
        return self


__all__ = [
    "ENV_ID",
    "CuratorEnvConfig",
    "CuratorTaskConfig",
    "CuratorTasksetConfig",
]
