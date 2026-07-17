"""v1 environment composition and up-front runtime validation."""

from __future__ import annotations

import verifiers.v1 as vf
from verifiers import ensure_keys

from .config import CuratorEnvConfig, CuratorTasksetConfig
from .taskset import CuratorTaskset

_MODAL_MAX_TIMEOUT_MINUTES = 1440


def load_taskset(config: CuratorTasksetConfig) -> CuratorTaskset:
    return CuratorTaskset(config)


def load_environment(config: CuratorEnvConfig | None = None) -> vf.Environment:
    """Compose the curator taskset with a stock v1 harness.

    Runtime placement is owned by ``config.harness.runtime`` so discovery,
    finalization, and real training share one rollout runtime.
    """
    config = config or CuratorEnvConfig()
    task = config.taskset.task
    curator = task.curator
    ensure_keys([task.hf_token_env])

    student = curator.proxy_student
    runtime = config.harness.runtime
    if curator.use_real_trainer and runtime.type not in {"docker", "modal"}:
        raise ValueError(
            "use_real_trainer=True requires harness.runtime.type to be "
            "'docker' or 'modal'"
        )
    if curator.use_real_trainer and runtime.type == "docker":
        if runtime.memory is None:
            raise ValueError(
                "real Docker training requires harness.runtime.memory"
            )
        from .util.container_memory import assert_host_supports_container_memory

        assert_host_supports_container_memory(runtime.memory)
    if curator.use_real_trainer and runtime.type == "modal":
        ensure_keys(["MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"])
        if student.effective_timeout_minutes > _MODAL_MAX_TIMEOUT_MINUTES:
            raise ValueError(
                f"timeout_minutes ({student.effective_timeout_minutes}) exceeds "
                f"the Modal 24h sandbox maximum ({_MODAL_MAX_TIMEOUT_MINUTES}); "
                "lower it"
            )

    forwarded = list(dict.fromkeys([*config.harness.forward_env, task.hf_token_env]))
    harness_env = dict(config.harness.env)
    if curator.use_real_trainer and runtime.type == "docker":
        harness_env.setdefault("UV_REINSTALL_PACKAGE", "pydantic-core")

    harness = config.harness.model_copy(
        update={
            "env": harness_env,
            "forward_env": forwarded,
        }
    )
    taskset = config.taskset.model_copy(deep=True)
    taskset._resolved_max_turns = config.max_turns
    resolved = config.model_copy(
        update={"taskset": taskset, "harness": harness}
    )
    return vf.Environment(resolved)


__all__ = ["load_environment", "load_taskset"]
