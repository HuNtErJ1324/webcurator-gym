"""Host/container memory preflight, cgroup verification, and OOM diagnostics.

Production 400M Docker runs pin ``proxy_student.memory_gb`` into Verifiers'
``DockerConfig.memory`` / ``TaskResources.memory``, which becomes
``docker run --memory Ng``. This module:

- resolves an optional ``PDC_CONTAINER_MEMORY_GB`` override
- fails before evaluation when host RAM cannot support the requested limit
  plus a configurable headroom reserve
- verifies the live container's cgroup / ``HostConfig.Memory`` matches
- collects OOMKilled / limit / host / process-group details for diagnostics
  without changing reward math
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from typing import Any

logger = logging.getLogger(__name__)

GIB = 1024**3
# Leave room for the env-server, HF caches, and agent shell outside the trainer
# cgroup. Large A100-80GB pods typically have well above 96+16 GiB host RAM.
DEFAULT_HOST_HEADROOM_GIB = 16.0
# Docker may round the applied limit by a page or two; tolerate 16 MiB.
MEMORY_LIMIT_TOLERANCE_BYTES = 16 * 1024 * 1024

ENV_CONTAINER_MEMORY_GB = "PDC_CONTAINER_MEMORY_GB"
ENV_MEMORY_HEADROOM_GB = "PDC_MEMORY_HEADROOM_GB"
ENV_SKIP_MEMORY_PREFLIGHT = "PDC_SKIP_MEMORY_PREFLIGHT"


class ContainerMemoryError(RuntimeError):
    """Host preflight or post-create cgroup memory verification failed."""


def memory_gb_to_bytes(memory_gb: float) -> int:
    return int(float(memory_gb) * GIB)


def bytes_to_gib(num_bytes: int | float) -> float:
    return float(num_bytes) / GIB


def resolve_headroom_gb(explicit: float | None = None) -> float:
    if explicit is not None:
        return float(explicit)
    raw = os.environ.get(ENV_MEMORY_HEADROOM_GB)
    if raw is None or raw.strip() == "":
        return DEFAULT_HOST_HEADROOM_GIB
    return float(raw)


def resolve_container_memory_gb(configured_gb: float | int) -> float:
    """Return the effective container memory limit in GiB.

    ``PDC_CONTAINER_MEMORY_GB`` overrides the TOML/config value when set, so ops
    can raise/lower the Docker ``--memory`` pin without editing eval configs.
    """
    raw = os.environ.get(ENV_CONTAINER_MEMORY_GB)
    if raw is None or raw.strip() == "":
        return float(configured_gb)
    value = float(raw)
    if value < 1:
        raise ContainerMemoryError(f"{ENV_CONTAINER_MEMORY_GB}={raw!r} must be >= 1")
    return value


def parse_meminfo(text: str) -> dict[str, int]:
    """Parse ``/proc/meminfo`` into a ``{Field: bytes}`` map."""
    out: dict[str, int] = {}
    for line in text.splitlines():
        match = re.match(r"^(\w+):\s+(\d+)(?:\s+kB)?\s*$", line)
        if not match:
            continue
        key, raw = match.group(1), int(match.group(2))
        # Values are reported in kB; bare counts (rare) are treated as bytes.
        out[key] = raw * 1024 if line.rstrip().endswith("kB") else raw
    return out


def read_host_meminfo(path: str = "/proc/meminfo") -> dict[str, int]:
    return parse_meminfo(open(path, encoding="utf-8").read())


def host_mem_total_bytes(meminfo: dict[str, int] | None = None) -> int:
    info = meminfo if meminfo is not None else read_host_meminfo()
    total = info.get("MemTotal")
    if total is None:
        raise ContainerMemoryError("host /proc/meminfo missing MemTotal")
    return int(total)


def required_host_bytes(
    memory_gb: float,
    *,
    headroom_gb: float | None = None,
) -> int:
    return memory_gb_to_bytes(memory_gb) + memory_gb_to_bytes(
        resolve_headroom_gb(headroom_gb)
    )


def assert_host_supports_container_memory(
    memory_gb: float,
    *,
    headroom_gb: float | None = None,
    meminfo: dict[str, int] | None = None,
    meminfo_text: str | None = None,
) -> None:
    """Fail fast when host RAM cannot back the Docker ``--memory`` limit.

    Skipped only when ``PDC_SKIP_MEMORY_PREFLIGHT=1`` (tests / deliberately
    unconstrained local sandboxes). Production eval launchers leave this unset.
    """
    if os.environ.get(ENV_SKIP_MEMORY_PREFLIGHT, "").strip() in {"1", "true", "yes"}:
        return
    if meminfo is None and meminfo_text is not None:
        meminfo = parse_meminfo(meminfo_text)
    total = host_mem_total_bytes(meminfo)
    headroom = resolve_headroom_gb(headroom_gb)
    needed = required_host_bytes(memory_gb, headroom_gb=headroom)
    if total < needed:
        raise ContainerMemoryError(
            "host RAM cannot support requested container memory limit: "
            f"need {bytes_to_gib(needed):.1f} GiB "
            f"({float(memory_gb):.1f} GiB container + {headroom:.1f} GiB headroom) "
            f"but host MemTotal is {bytes_to_gib(total):.1f} GiB"
        )


def parse_docker_memory_limit_bytes(raw: str | int | float | None) -> int | None:
    """Normalize a Docker inspect memory field to bytes.

    ``HostConfig.Memory`` is an integer byte count. ``0`` means unlimited.
    """
    if raw is None:
        return None
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return int(raw)
    text = str(raw).strip()
    if not text:
        return None
    if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
        return int(text)
    match = re.fullmatch(r"(?i)(\d+(?:\.\d+)?)([kmgtpe]i?b?)?", text)
    if not match:
        raise ContainerMemoryError(f"unrecognized docker memory value: {raw!r}")
    amount = float(match.group(1))
    unit = (match.group(2) or "b").lower()
    multipliers = {
        "b": 1,
        "k": 1024,
        "kb": 1024,
        "ki": 1024,
        "kib": 1024,
        "m": 1024**2,
        "mb": 1024**2,
        "mi": 1024**2,
        "mib": 1024**2,
        "g": GIB,
        "gb": GIB,
        "gi": GIB,
        "gib": GIB,
        "t": 1024**4,
        "tb": 1024**4,
        "ti": 1024**4,
        "tib": 1024**4,
    }
    if unit not in multipliers:
        raise ContainerMemoryError(f"unrecognized docker memory unit in {raw!r}")
    return int(amount * multipliers[unit])


def memory_limits_match(
    *,
    configured_gb: float,
    effective_bytes: int | None,
    tolerance_bytes: int = MEMORY_LIMIT_TOLERANCE_BYTES,
) -> bool:
    if effective_bytes is None:
        return False
    if effective_bytes <= 0:
        return False
    expected = memory_gb_to_bytes(configured_gb)
    return abs(int(effective_bytes) - expected) <= tolerance_bytes


def assert_cgroup_memory_limit(
    *,
    configured_gb: float,
    effective_bytes: int | None,
    source: str = "container",
    tolerance_bytes: int = MEMORY_LIMIT_TOLERANCE_BYTES,
) -> None:
    if effective_bytes is None:
        raise ContainerMemoryError(
            f"{source} memory limit unavailable; cannot verify "
            f"configured {float(configured_gb):.1f} GiB"
        )
    if effective_bytes <= 0:
        raise ContainerMemoryError(
            f"{source} memory limit is unlimited (0); expected "
            f"{float(configured_gb):.1f} GiB"
        )
    if not memory_limits_match(
        configured_gb=configured_gb,
        effective_bytes=effective_bytes,
        tolerance_bytes=tolerance_bytes,
    ):
        raise ContainerMemoryError(
            f"{source} memory limit mismatch: configured "
            f"{float(configured_gb):.1f} GiB "
            f"({memory_gb_to_bytes(configured_gb)} bytes) but effective is "
            f"{int(effective_bytes)} bytes "
            f"({bytes_to_gib(effective_bytes):.3f} GiB)"
        )


def inspect_container_memory(
    container: str,
    *,
    docker_bin: str = "docker",
) -> dict[str, Any]:
    """Return HostConfig.Memory / State.OOMKilled for a live container."""
    proc = subprocess.run(
        [
            docker_bin,
            "inspect",
            "--format",
            "{{json .}}",
            container,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise ContainerMemoryError(f"docker inspect failed for {container!r}: {detail}")
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise ContainerMemoryError(
            f"docker inspect returned non-JSON for {container!r}"
        ) from exc
    host_config = payload.get("HostConfig") or {}
    state = payload.get("State") or {}
    return {
        "memory_bytes": parse_docker_memory_limit_bytes(host_config.get("Memory")),
        "oom_killed": bool(state.get("OOMKilled", False)),
        "status": state.get("Status"),
        "exit_code": state.get("ExitCode"),
        "error": state.get("Error") or None,
        "raw": payload,
    }


def verify_runtime_memory_limit(
    runtime: Any,
    *,
    configured_gb: float,
    docker_bin: str = "docker",
) -> dict[str, Any]:
    """Verify the live Docker runtime's cgroup memory matches ``configured_gb``."""
    container = getattr(runtime, "_container", None) or getattr(runtime, "name", None)
    if not container:
        raise ContainerMemoryError(
            "docker runtime has no container name; cannot verify memory limit"
        )
    info = inspect_container_memory(str(container), docker_bin=docker_bin)
    assert_cgroup_memory_limit(
        configured_gb=configured_gb,
        effective_bytes=info.get("memory_bytes"),
        source=f"container {container}",
    )
    return info


def collect_oom_diagnostics(
    *,
    configured_gb: float | None = None,
    effective_memory_bytes: int | None = None,
    oom_killed: bool | None = None,
    host_meminfo: dict[str, int] | None = None,
    process_group: dict[str, Any] | None = None,
    container: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable OOM / memory diagnostic payload."""
    try:
        host = host_meminfo if host_meminfo is not None else read_host_meminfo()
        host_total = host.get("MemTotal")
        host_available = host.get("MemAvailable")
    except Exception as exc:  # noqa: BLE001 - diagnostics must not raise
        host = None
        host_total = None
        host_available = None
        host_error = str(exc)
    else:
        host_error = None

    payload: dict[str, Any] = {
        "configured_memory_gb": (
            float(configured_gb) if configured_gb is not None else None
        ),
        "configured_memory_bytes": (
            memory_gb_to_bytes(configured_gb) if configured_gb is not None else None
        ),
        "effective_memory_bytes": effective_memory_bytes,
        "effective_memory_gib": (
            bytes_to_gib(effective_memory_bytes)
            if effective_memory_bytes is not None and effective_memory_bytes > 0
            else None
        ),
        "oom_killed": oom_killed,
        "host_memory_bytes": host_total,
        "host_memory_gib": bytes_to_gib(host_total) if host_total else None,
        "host_memory_available_bytes": host_available,
        "host_memory_error": host_error,
        "container": container,
        "process_group": process_group,
    }
    if extra:
        payload.update(extra)
    return payload


def format_oom_diagnostics(diagnostics: dict[str, Any]) -> str:
    return "--- memory/oom diagnostics ---\n" + json.dumps(
        diagnostics, sort_keys=True, default=str
    )


__all__ = [
    "ContainerMemoryError",
    "DEFAULT_HOST_HEADROOM_GIB",
    "ENV_CONTAINER_MEMORY_GB",
    "ENV_MEMORY_HEADROOM_GB",
    "ENV_SKIP_MEMORY_PREFLIGHT",
    "GIB",
    "MEMORY_LIMIT_TOLERANCE_BYTES",
    "assert_cgroup_memory_limit",
    "assert_host_supports_container_memory",
    "bytes_to_gib",
    "collect_oom_diagnostics",
    "format_oom_diagnostics",
    "host_mem_total_bytes",
    "inspect_container_memory",
    "memory_gb_to_bytes",
    "memory_limits_match",
    "parse_docker_memory_limit_bytes",
    "parse_meminfo",
    "read_host_meminfo",
    "required_host_bytes",
    "resolve_container_memory_gb",
    "resolve_headroom_gb",
    "verify_runtime_memory_limit",
]
