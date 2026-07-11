"""Shared runtime/resource derivation for the Docker and Modal trainer backends.

Both the v1 env-level ``harness_runtime`` (``DockerConfig``/``ModalConfig``) and
the taskset task-level ``resources``/``timeout`` updates derive from the same
``ProxyStudentConfig`` fields. Centralizing that derivation here keeps the
Docker and Modal semantics in one place and prevents the two code paths from
drifting. Modal-specific resolution is imported lazily so a package import
stays safe (and free of Modal import side effects) until a Modal backend is
actually selected.
"""

from __future__ import annotations

from typing import Any

import verifiers.v1 as vf

from .models import ProxyStudentConfig


def derive_trainer_resources(
    ps: ProxyStudentConfig,
    *,
    backend: str,
) -> dict[str, Any]:
    """Primitive trainer resources shared by both runtime mappings.

    ``backend`` is ``"docker"`` or ``"modal"``. The only backend-specific field
    is the GPU specifier: Docker passes ``gpu_count`` (``None`` when zero), while
    Modal maps ``modal_gpu`` through ``_modal_gpu_for`` (lazy import).
    """
    if backend == "modal":
        from .modal_backend import _modal_gpu_for

        gpu = _modal_gpu_for(ps.modal_gpu)
    else:
        gpu = str(ps.gpu_count) if ps.gpu_count > 0 else None
    return {
        "image": ps.docker_image,
        "workdir": "/workspace",
        "cpu": float(ps.cpu_cores),
        "memory": float(ps.memory_gb),
        "gpu": gpu,
        "disk": float(ps.disk_size_gb),
        "scoring_timeout_seconds": ps.effective_scoring_timeout_seconds,
    }


def derive_env_harness_runtime(
    ps: ProxyStudentConfig,
    *,
    use_real_trainer: bool,
) -> tuple[vf.RuntimeConfig, vf.TimeoutConfig]:
    """Build the v1 env-level harness runtime and scoring timeout.

    Mirrors ``load_environment``'s runtime derivation: a real ``docker``/``modal``
    trainer backend yields a ``DockerConfig``/``ModalConfig`` plus an extended
    scoring timeout, otherwise the default subprocess runtime and timeout are
    returned. Validation of ``runtime_backend``/``docker_host`` is the caller's
    responsibility and lives in ``load_environment``.
    """
    if use_real_trainer and ps.runtime_backend == "docker":
        res = derive_trainer_resources(ps, backend="docker")
        runtime: vf.RuntimeConfig = vf.DockerConfig(
            image=res["image"],
            workdir=res["workdir"],
            gpu=res["gpu"],
            cpu=res["cpu"],
            memory=res["memory"],
            disk=res["disk"],
        )
        timeout = vf.TimeoutConfig(scoring=res["scoring_timeout_seconds"])
    elif use_real_trainer and ps.runtime_backend == "modal":
        from verifiers.v1.runtimes.modal import ModalConfig

        res = derive_trainer_resources(ps, backend="modal")
        runtime = ModalConfig(
            image=res["image"],
            workdir=res["workdir"],
            gpu=res["gpu"],
            cpu=res["cpu"],
            memory=res["memory"],
            disk=res["disk"],
        )
        timeout = vf.TimeoutConfig(scoring=res["scoring_timeout_seconds"])
    else:
        runtime = vf.SubprocessConfig()
        timeout = vf.TimeoutConfig()
    return runtime, timeout


def derive_task_runtime_updates(
    ps: ProxyStudentConfig,
    *,
    use_real_trainer: bool,
) -> dict[str, Any]:
    """Build the per-task runtime/resources/timeout updates for the taskset.

    Returns ``{}`` when no real trainer backend is selected, matching the
    historical behavior of ``CuratorTaskset.load_tasks`` (which leaves the base
    tasks untouched). For a real ``docker``/``modal`` backend it returns the
    shared image/workdir plus a ``vf.TaskResources``/``vf.TaskTimeout`` pair.
    """
    if not (use_real_trainer and ps.runtime_backend in ("docker", "modal")):
        return {}
    res = derive_trainer_resources(ps, backend=ps.runtime_backend)
    return {
        "image": res["image"],
        "workdir": res["workdir"],
        "resources": vf.TaskResources(
            cpu=res["cpu"],
            memory=res["memory"],
            gpu=res["gpu"],
            disk=res["disk"],
        ),
        "timeout": vf.TaskTimeout(scoring=res["scoring_timeout_seconds"]),
    }
