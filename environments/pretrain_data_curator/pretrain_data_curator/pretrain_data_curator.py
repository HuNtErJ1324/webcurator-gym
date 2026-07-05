"""Loader for the native verifiers v1 pretraining-data curation environment."""

from __future__ import annotations

import math
import pkgutil
from typing import Any

import verifiers as legacy_vf
import verifiers.v1 as vf
import verifiers.v1.harnesses as vf_harnesses
from verifiers.v1.runtimes.modal import ModalConfig

from .hosted_compat import Environment
from .modal_backend import _modal_gpu_for
from .models import ProxyStudentConfig
from .leakage import DEFAULT_DECON_BINARY
from .taskset import CuratorTasksetConfig
from .tasks import TASK_PROMPT

TASKSET_ID = "pretrain-data-curator"

__all__ = ["TASK_PROMPT", "load_environment"]


def load_environment(
    cutoff_date: str = "2024-12-31",
    token_budget: int = 1_000_000,
    hf_token_env: str = "HF_TOKEN",
    candidate_limit: int = 8,
    sample_docs_per_source: int = 64,
    allow_local_sources: bool = True,
    max_local_source_bytes: int = 33_554_432,
    max_turns: int = 64,
    alpha_perf: float = 1.0,
    lambda_cost: float = 0.1,
    lambda_leakage: float = 1.0,
    perf_baseline_loss: float = math.log(50304),
    baseline_relative_perf: bool = True,
    max_concurrent_fetches: int = 8,
    max_concurrent_training: int = 1,
    fetch_timeout_seconds: float = 30.0,
    fetch_max_attempts: int = 3,
    use_real_trainer: bool = False,
    proxy_student: dict[str, Any] | None = None,
    validation_set: dict[str, Any] | None = None,
    fetch_timeout_per_doc_seconds: float = 0.25,
    harness_id: str = "bash",
    decon_binary: str = DEFAULT_DECON_BINARY,
    decon_evals_dir: str | None = None,
    decon_threshold: float = 0.2,
) -> vf.Environment:
    """Build the native verifiers v1 curation environment.

    The agent curates via the ``hf`` CLI in its own shell rather than MCP tools,
    so there is no tool server to inject a client into; scoring collaborators are
    injected on the taskset directly in tests. Hugging Face credentials are
    checked in taskset setup before a rollout starts (and again lazily at first
    Hub API use), so constructing the environment itself does not require
    ``HF_TOKEN`` in the orchestrator process.
    ``harness_id`` selects one of the harnesses bundled with the installed
    Verifiers package.
    Unsupported keywords are rejected by Python with a clear ``TypeError`` rather
    than being silently dropped, so a misspelled or stale eval arg fails loudly.
    """
    valid_harness_ids = sorted(
        module.name for module in pkgutil.iter_modules(vf_harnesses.__path__)
    )
    if harness_id not in valid_harness_ids:
        raise ValueError(
            f"unknown harness_id {harness_id!r}; valid harness ids: "
            f"{', '.join(valid_harness_ids)}"
        )

    config = CuratorTasksetConfig(
        id=TASKSET_ID,
        cutoff_date=cutoff_date,
        token_budget=token_budget,
        hf_token_env=hf_token_env,
        candidate_limit=candidate_limit,
        sample_docs_per_source=sample_docs_per_source,
        allow_local_sources=allow_local_sources,
        max_local_source_bytes=max_local_source_bytes,
        max_turns=max_turns,
        alpha_perf=alpha_perf,
        lambda_cost=lambda_cost,
        lambda_leakage=lambda_leakage,
        perf_baseline_loss=perf_baseline_loss,
        baseline_relative_perf=baseline_relative_perf,
        max_concurrent_fetches=max_concurrent_fetches,
        max_concurrent_training=max_concurrent_training,
        fetch_timeout_seconds=fetch_timeout_seconds,
        fetch_timeout_per_doc_seconds=fetch_timeout_per_doc_seconds,
        fetch_max_attempts=fetch_max_attempts,
        use_real_trainer=use_real_trainer,
        proxy_student=proxy_student or {},
        validation_set=validation_set or {},
        decon_binary=decon_binary,
        decon_evals_dir=decon_evals_dir,
        decon_threshold=decon_threshold,
    )
    env_args = {
        "cutoff_date": cutoff_date,
        "token_budget": token_budget,
        "hf_token_env": hf_token_env,
        "candidate_limit": candidate_limit,
        "sample_docs_per_source": sample_docs_per_source,
        "harness_id": harness_id,
        "allow_local_sources": allow_local_sources,
        "max_local_source_bytes": max_local_source_bytes,
        "max_turns": max_turns,
        "alpha_perf": alpha_perf,
        "lambda_cost": lambda_cost,
        "lambda_leakage": lambda_leakage,
        "perf_baseline_loss": perf_baseline_loss,
        "baseline_relative_perf": baseline_relative_perf,
        "max_concurrent_fetches": max_concurrent_fetches,
        "max_concurrent_training": max_concurrent_training,
        "fetch_timeout_seconds": fetch_timeout_seconds,
        "fetch_timeout_per_doc_seconds": fetch_timeout_per_doc_seconds,
        "fetch_max_attempts": fetch_max_attempts,
        "use_real_trainer": use_real_trainer,
        "proxy_student": proxy_student,
        "validation_set": validation_set,
        "decon_binary": decon_binary,
        "decon_evals_dir": decon_evals_dir,
        "decon_threshold": decon_threshold,
    }
    harness_env: dict[str, str] = {}
    harness_runtime: vf.RuntimeConfig = vf.SubprocessConfig()
    timeout = vf.TimeoutConfig()
    ps = ProxyStudentConfig(**config.proxy_student)
    if use_real_trainer and ps.runtime_backend is None:
        raise ValueError(
            "use_real_trainer=True requires proxy_student.runtime_backend to be "
            "'docker' or 'modal' (Prime sandboxes are no longer supported)"
        )
    if use_real_trainer and ps.runtime_backend == "docker":
        if ps.docker_host is not None:
            raise ValueError(
                "proxy_student.docker_host is not supported by the shared harness "
                "runtime Docker backend; run the rollout and Docker daemon on the "
                "same machine and leave docker_host unset"
            )
        # The v1 bash harness runs as a cached PEP 723 uv script. On local Docker
        # trainer runs, a stale script env can miss pydantic-core's compiled
        # extension and fail before reward training starts. Reinstall only that
        # package on this path; hosted/Prime/default paths keep the normal cache.
        harness_env["UV_REINSTALL_PACKAGE"] = "pydantic-core"
        harness_runtime = vf.DockerConfig(
            image=ps.docker_image,
            workdir="/workspace",
            gpu=str(ps.gpu_count) if ps.gpu_count > 0 else None,
            cpu=float(ps.cpu_cores),
            memory=float(ps.memory_gb),
            disk=float(ps.disk_size_gb),
        )
        # Scoring includes corpus materialization, input writes, training, and
        # leakage computation. Keep the framework deadline above the trainer's
        # own multi-hour command deadline so the trainer can report/clean up.
        timeout = vf.TimeoutConfig(scoring=ps.effective_scoring_timeout_seconds)
    elif use_real_trainer and ps.runtime_backend == "modal":
        legacy_vf.ensure_keys(["MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"])
        # Intentionally do not set UV_REINSTALL_PACKAGE here. That workaround
        # originated when the Docker trainer's bash harness ran on the env-server
        # and could reuse its host-cached PEP 723 environment. ModalRuntime creates
        # a fresh sandbox per rollout from the registry image and mounts no
        # persistent Volume or snapshot, so its uv script environment cannot carry
        # a stale pydantic-core extension across rollouts.
        harness_runtime = ModalConfig(
            image=ps.docker_image,
            workdir="/workspace",
            gpu=_modal_gpu_for(ps.modal_gpu),
            cpu=float(ps.cpu_cores),
            memory=float(ps.memory_gb),
            disk=float(ps.disk_size_gb),
        )
        timeout = vf.TimeoutConfig(scoring=ps.effective_scoring_timeout_seconds)

    return Environment(
        vf.EnvConfig(
            taskset=config,
            max_turns=max_turns,
            harness=vf.HarnessConfig(
                id=harness_id, env=harness_env, runtime=harness_runtime
            ),
            timeout=timeout,
        ),
        env_args=env_args,
    )
