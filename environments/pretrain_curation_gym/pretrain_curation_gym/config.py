"""Configuration at the three v1 ownership boundaries."""

from __future__ import annotations

import re

from pydantic import Field, field_validator

import verifiers.v1 as vf

from .utils.leakage import DEFAULT_DECON_BINARY
from .utils.models import CuratorConfig, MANIFEST_FILENAME

ENV_ID = "pretrain-curation-gym"
DEFAULT_MAX_TURNS = 64


class CuratorTaskConfig(vf.TaskConfig):
    """Knobs consumed by one curation task and its scoring lifecycle."""

    curator: CuratorConfig = Field(default_factory=CuratorConfig)
    max_turns: int | None = Field(default=None, ge=1, le=1000)
    """Prompted turn budget (must match EnvConfig.max_turns when set). None = omit total."""
    hf_token_env: str = "HF_TOKEN"
    manifest_filename: str = MANIFEST_FILENAME
    decon_binary: str = DEFAULT_DECON_BINARY
    decon_evals_dir: str | None = None
    runtime_decon_binary: str = "decon/bin/decon"
    runtime_decon_evals_dir: str = "decon/bundled-evals"
    decon_threshold: float = 0.8
    screen_val_set: bool = True
    error_on_empty_rollout: bool = False
    """If True, empty rollouts raise retryable EmptyRolloutError (off for RL zeros)."""
    error_on_decon_failure: bool = False
    """If True, decon failures raise retryable DeconUnavailableError (off for RL)."""

    @field_validator("manifest_filename")
    @classmethod
    def workspace_root_filename(cls, value: str) -> str:
        if not value or value in {".", ".."} or "/" in value or "\\" in value:
            raise ValueError("manifest_filename must be a runtime-workspace filename")
        return value

    @field_validator("hf_token_env")
    @classmethod
    def environment_variable_name(cls, value: str) -> str:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
            raise ValueError("hf_token_env must be a valid environment variable name")
        return value

    @field_validator("runtime_decon_binary", "runtime_decon_evals_dir")
    @classmethod
    def nonempty_runtime_path(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("runtime decon paths must not be empty")
        return value


class CuratorTasksetConfig(vf.TasksetConfig):
    """The taskset owns one typed task configuration and no duplicate knobs."""

    id: str = ENV_ID
    # pyright: TaskConfig specialization via SerializeAsAny
    task: CuratorTaskConfig = Field(
        default_factory=CuratorTaskConfig
    )


class CuratorEnvConfig(vf.EnvConfig):
    """Concrete composition config used by ``load_environment``."""

    taskset: CuratorTasksetConfig = Field(
        default_factory=CuratorTasksetConfig
    )
    harness: vf.HarnessConfig = Field(
        default_factory=lambda: vf.HarnessConfig(id="default")
    )
    max_turns: int = Field(default=DEFAULT_MAX_TURNS, ge=1, le=1000)


__all__ = [
    "ENV_ID",
    "DEFAULT_MAX_TURNS",
    "CuratorEnvConfig",
    "CuratorTaskConfig",
    "CuratorTasksetConfig",
]
