"""Configuration at the three v1 ownership boundaries."""

from __future__ import annotations

from pydantic import Field, field_validator

import verifiers.v1 as vf

from .utils.leakage import DEFAULT_DECON_BINARY
from .utils.models import CuratorConfig, MANIFEST_FILENAME

ENV_ID = "pretrain-curation-gym"
DEFAULT_MAX_TURNS = 64


class CuratorTaskConfig(vf.TaskConfig):
    """Knobs consumed by one curation task and its scoring lifecycle."""

    curator: CuratorConfig = Field(default_factory=CuratorConfig)
    max_turns: int = Field(default=DEFAULT_MAX_TURNS, ge=1, le=1000)
    """Turn budget the agent is TOLD it has: rendered into the task prompt and
    reported by the workspace ``turns.py``.

    It must equal ``EnvConfig.max_turns``, which is what actually enforces the
    limit. It cannot be derived from that field at load time: v1 never plumbs
    ``RolloutLimits`` to task code, and the CLI builds the taskset straight from
    ``[taskset.*]`` via ``loaders.load_taskset`` without calling
    ``load_environment``. So this is a real, serialized field — a private
    attribute set by ``load_environment`` reaches neither the CLI (which skips
    that function) nor an env-server worker (which rebuilds the config from
    ``model_dump``, and private attributes do not serialize).

    ``load_environment`` reconciles the two for programmatic callers, so setting
    either one there is enough. A TOML-driven run has no such hook: set the
    framework's ``max_turns`` and this to the same value."""
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
    error_on_decon_failure: bool = False
    """Raise a retryable ``DeconUnavailableError`` when the contamination screen
    could not run. Independent of the reward, which is always withheld in that
    case — an unscreened corpus is never scoreable. Off by default so RL records
    a zero-reward sample rather than erroring; eval configs enable it together
    with ``[retries.rollout] include=['DeconUnavailableError']`` so a transient
    detector failure retries instead of banking an unscoreable rollout."""

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
    # The ENFORCED turn limit: EnvConfig passes it to the framework's interception
    # session, which refuses turns past it for every harness. The agent-visible
    # copy is CuratorTaskConfig.max_turns; load_environment reconciles them.
    max_turns: int = Field(default=DEFAULT_MAX_TURNS, ge=1, le=1000)


__all__ = [
    "ENV_ID",
    "DEFAULT_MAX_TURNS",
    "CuratorEnvConfig",
    "CuratorTaskConfig",
    "CuratorTasksetConfig",
]
