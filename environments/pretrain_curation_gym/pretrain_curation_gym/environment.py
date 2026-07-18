"""v1 environment composition and up-front runtime validation."""

from __future__ import annotations

import verifiers.v1 as vf
from verifiers import ensure_keys

from .config import CuratorEnvConfig, CuratorTasksetConfig
from .taskset import CuratorTaskset

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

    runtime = config.harness.runtime
    harness_env = dict(config.harness.env)
    if curator.use_real_trainer:
        if runtime.type not in {"docker", "modal"}:
            raise ValueError(
                "use_real_trainer=True requires harness.runtime.type to be "
                "'docker' or 'modal'"
            )
        if runtime.type == "docker":
            if runtime.memory is None:
                raise ValueError(
                    "real Docker training requires harness.runtime.memory"
                )
            harness_env.setdefault("UV_REINSTALL_PACKAGE", "pydantic-core")
        else:
            ensure_keys(["MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"])

    forwarded = list(dict.fromkeys([*config.harness.forward_env, task.hf_token_env]))

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
