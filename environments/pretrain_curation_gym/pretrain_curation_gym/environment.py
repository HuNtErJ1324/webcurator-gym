"""v1 environment composition and up-front runtime validation."""

from __future__ import annotations

import posixpath

import verifiers.v1 as vf
from verifiers import ensure_keys

from .config import CuratorEnvConfig, CuratorTasksetConfig
from .gpu.hf_cli_audit import HF_CLI_WRAPPER_FILENAME
from .taskset import CuratorTaskset

def load_taskset(config: CuratorTasksetConfig) -> CuratorTaskset:
    return CuratorTaskset(config)


def reconcile_max_turns(config: CuratorEnvConfig) -> int:
    """Agree the enforced turn limit and the one the agent is told about.

    ``EnvConfig.max_turns`` is enforced by the framework; ``task.max_turns`` is
    what the prompt and ``turns.py`` report. They are separate fields because
    task code cannot read the framework's limit (see ``CuratorTaskConfig``), so
    a programmatic caller that sets only one would otherwise silently mislead
    the agent about its budget. Setting either here is enough; setting both to
    different values is a mistake worth failing on rather than picking a winner.
    """
    env_set = "max_turns" in config.model_fields_set
    task_set = "max_turns" in config.taskset.task.model_fields_set
    if env_set and task_set and config.max_turns != config.taskset.task.max_turns:
        raise ValueError(
            "max_turns is set in two places and they disagree: "
            f"max_turns={config.max_turns} (enforced by the framework) vs "
            f"taskset.task.max_turns={config.taskset.task.max_turns} (shown to "
            "the agent). Set one, or set both to the same value."
        )
    if task_set and not env_set:
        return config.taskset.task.max_turns
    return config.max_turns


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

    # The workspace `hf` audit wrapper records nothing unless its directory
    # precedes the real CLI on PATH. The runtime image puts it there (see
    # Dockerfile.runtime), but an explicit harness PATH replaces that value
    # wholesale, which would silently empty the audit log rather than fail. Repair
    # such an override here, where the wrapper's location is already known.
    workdir = getattr(runtime, "workdir", None)
    if harness_env.get("PATH") and workdir:
        wrapper_dir = posixpath.join(
            workdir, posixpath.dirname(HF_CLI_WRAPPER_FILENAME)
        )
        entries = harness_env["PATH"].split(":")
        if wrapper_dir not in entries:
            harness_env["PATH"] = ":".join([wrapper_dir, *entries])

    forwarded = list(dict.fromkeys([*config.harness.forward_env, task.hf_token_env]))

    harness = config.harness.model_copy(
        update={
            "env": harness_env,
            "forward_env": forwarded,
        }
    )
    max_turns = reconcile_max_turns(config)
    taskset = config.taskset.model_copy(deep=True)
    taskset.task = taskset.task.model_copy(update={"max_turns": max_turns})
    resolved = config.model_copy(
        update={"taskset": taskset, "harness": harness, "max_turns": max_turns}
    )
    return vf.Environment(resolved)


__all__ = ["load_environment", "load_taskset"]
