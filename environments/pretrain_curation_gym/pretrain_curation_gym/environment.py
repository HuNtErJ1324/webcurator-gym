"""v1 environment composition and up-front runtime validation."""

from __future__ import annotations

import verifiers.v1 as vf
from verifiers import ensure_keys

from .config import CuratorEnvConfig, CuratorTasksetConfig
from .runtime_config import derive_env_harness_runtime
from .taskset import CuratorTaskset


def load_taskset(config: CuratorTasksetConfig) -> CuratorTaskset:
    return CuratorTaskset(config)


def load_environment(config: CuratorEnvConfig | None = None) -> vf.Environment:
    """Compose the curator taskset with a stock v1 harness.

    Runtime placement remains derived from the fixed proxy-student config so
    discovery, finalization, and real training share one rollout runtime.
    """
    config = config or CuratorEnvConfig()
    task = config.taskset.task
    curator = task.curator
    ensure_keys([task.hf_token_env])

    student = curator.proxy_student
    if curator.use_real_trainer and student.runtime_backend not in {"docker", "modal"}:
        raise ValueError(
            "use_real_trainer=True requires proxy_student.runtime_backend to be "
            "'docker' or 'modal'"
        )
    if curator.use_real_trainer and student.runtime_backend == "docker":
        if student.docker_host is not None:
            raise ValueError("proxy_student.docker_host is not supported")
        from .util.container_memory import (
            assert_host_supports_container_memory,
            resolve_container_memory_gb,
        )

        assert_host_supports_container_memory(
            resolve_container_memory_gb(student.memory_gb)
        )
    if curator.use_real_trainer and student.runtime_backend == "modal":
        ensure_keys(["MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"])

    runtime, derived_timeout = derive_env_harness_runtime(
        student, use_real_trainer=curator.use_real_trainer
    )
    forwarded = list(dict.fromkeys([*config.harness.forward_env, task.hf_token_env]))
    harness_env = dict(config.harness.env)
    if curator.use_real_trainer and student.runtime_backend == "docker":
        harness_env.setdefault("UV_REINSTALL_PACKAGE", "pydantic-core")

    harness = config.harness.model_copy(
        update={
            "runtime": runtime,
            "env": harness_env,
            "forward_env": forwarded,
        }
    )
    timeout = config.timeout.model_copy(
        update={
            "scoring": derived_timeout.scoring
            if derived_timeout.scoring is not None
            else config.timeout.scoring
        }
    )
    resolved = config.model_copy(update={"harness": harness, "timeout": timeout})
    return vf.Environment(resolved)


__all__ = ["load_environment", "load_taskset"]
