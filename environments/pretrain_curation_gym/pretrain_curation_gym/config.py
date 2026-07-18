"""Configuration at the three v1 ownership boundaries."""

from __future__ import annotations

from pydantic import Field, PrivateAttr, field_validator

import verifiers.v1 as vf

from .utils.leakage import DEFAULT_DECON_BINARY
from .utils.models import CuratorConfig, MANIFEST_FILENAME

ENV_ID = "pretrain-curation-gym"
DEFAULT_MAX_TURNS = 64


class CuratorTaskConfig(vf.TaskConfig):
    """Knobs consumed by one curation task and its scoring lifecycle."""

    curator: CuratorConfig = Field(default_factory=CuratorConfig)
    hf_token_env: str = "HF_TOKEN"
    manifest_filename: str = MANIFEST_FILENAME
    decon_binary: str = DEFAULT_DECON_BINARY
    decon_evals_dir: str | None = None
    # OLMo 3's calibrated contamination decision threshold (paper Appendix A.5;
    # the decon package's production default). Matches below it are increasingly
    # likely to be eval *source material* rather than eval-derived text.
    decon_threshold: float = 0.8
    screen_val_set: bool = True
    error_on_empty_rollout: bool = False
    """Raise a retryable ``EmptyRolloutError`` when a rollout produced no usable
    artifact — no valid workspace manifest and zero self-scores. Off by default
    so RL keeps its legitimate silent zero-reward signal; smoke/eval configs
    enable it together with ``[retries.rollout] include=['EmptyRolloutError']``
    so a transient model-endpoint failure retries instead of being recorded as a
    spurious zero-reward success."""

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
    # Derived from EnvConfig.max_turns by load_environment. It is deliberately
    # private so there is still exactly one user-configurable turn-limit field.
    _resolved_max_turns: int = PrivateAttr(default=DEFAULT_MAX_TURNS)


class CuratorEnvConfig(vf.EnvConfig):
    """Concrete composition config used by ``load_environment``."""

    taskset: CuratorTasksetConfig = Field(  # pyright: ignore[reportIncompatibleVariableOverride]
        default_factory=CuratorTasksetConfig
    )
    harness: vf.HarnessConfig = Field(
        default_factory=lambda: vf.HarnessConfig(id="default")
    )
    # This is the sole turn limit. EnvConfig passes it to the framework's
    # interception session, which enforces it for every harness.
    max_turns: int = Field(default=DEFAULT_MAX_TURNS, ge=1, le=1000)


__all__ = [
    "ENV_ID",
    "DEFAULT_MAX_TURNS",
    "CuratorEnvConfig",
    "CuratorTaskConfig",
    "CuratorTasksetConfig",
]
