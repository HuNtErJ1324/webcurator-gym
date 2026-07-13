"""Regression tests for container memory preflight/cgroup checks and self_score single-flight."""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from pretrain_data_curator.container_memory import (
    ENV_CONTAINER_MEMORY_GB,
    ENV_SKIP_MEMORY_PREFLIGHT,
    GIB,
    ContainerMemoryError,
    assert_cgroup_memory_limit,
    assert_host_supports_container_memory,
    collect_oom_diagnostics,
    memory_gb_to_bytes,
    parse_docker_memory_limit_bytes,
    parse_meminfo,
    resolve_container_memory_gb,
    verify_runtime_memory_limit,
)
from pretrain_data_curator.models import CuratorConfig, ProxyStudentConfig
from pretrain_data_curator.runtime_config import derive_trainer_resources
from pretrain_data_curator.self_score import render_self_score_script


def _meminfo(total_kib: int, available_kib: int | None = None) -> str:
    avail = total_kib if available_kib is None else available_kib
    return f"MemTotal:       {total_kib} kB\nMemAvailable:   {avail} kB\n"


def test_production_400m_configs_default_to_96gib():
    root = Path(__file__).resolve().parents[1] / "configs" / "eval"
    configs = sorted(root.glob("*400M*.toml"))
    assert configs
    for path in configs:
        text = path.read_text(encoding="utf-8")
        assert re.search(r"^memory_gb\s*=\s*96\b", text, re.M), path.name


def test_resolve_container_memory_gb_env_override(monkeypatch):
    monkeypatch.delenv(ENV_CONTAINER_MEMORY_GB, raising=False)
    assert resolve_container_memory_gb(48) == 48.0
    monkeypatch.setenv(ENV_CONTAINER_MEMORY_GB, "96")
    assert resolve_container_memory_gb(48) == 96.0


def test_derive_trainer_resources_honors_memory_override(monkeypatch):
    monkeypatch.setenv(ENV_CONTAINER_MEMORY_GB, "96")
    ps = ProxyStudentConfig(runtime_backend="docker", memory_gb=48, gpu_count=1)
    resources = derive_trainer_resources(ps, backend="docker")
    assert resources["memory"] == 96.0


def test_host_preflight_passes_with_sufficient_ram(monkeypatch):
    monkeypatch.delenv(ENV_SKIP_MEMORY_PREFLIGHT, raising=False)
    # 96 GiB container + 16 GiB headroom = 112 GiB => need >= 112 * 1024^2 KiB
    total_kib = int(112 * GIB / 1024)
    assert_host_supports_container_memory(
        96, headroom_gb=16, meminfo_text=_meminfo(total_kib)
    )


def test_host_preflight_fails_when_host_ram_too_small(monkeypatch):
    monkeypatch.delenv(ENV_SKIP_MEMORY_PREFLIGHT, raising=False)
    total_kib = int(64 * GIB / 1024)  # 64 GiB host cannot back 96+16
    with pytest.raises(ContainerMemoryError, match="host RAM cannot support"):
        assert_host_supports_container_memory(
            96, headroom_gb=16, meminfo_text=_meminfo(total_kib)
        )


def test_parse_meminfo_and_docker_memory_units():
    info = parse_meminfo(_meminfo(1024))
    assert info["MemTotal"] == 1024 * 1024
    assert parse_docker_memory_limit_bytes(96 * GIB) == 96 * GIB
    assert parse_docker_memory_limit_bytes("96g") == 96 * GIB
    assert parse_docker_memory_limit_bytes("0") == 0


def test_cgroup_memory_verification_accepts_matching_limit():
    assert_cgroup_memory_limit(configured_gb=96, effective_bytes=memory_gb_to_bytes(96))


def test_cgroup_memory_verification_rejects_mismatch_and_unlimited():
    with pytest.raises(ContainerMemoryError, match="mismatch"):
        assert_cgroup_memory_limit(
            configured_gb=96, effective_bytes=memory_gb_to_bytes(48)
        )
    with pytest.raises(ContainerMemoryError, match="unlimited"):
        assert_cgroup_memory_limit(configured_gb=96, effective_bytes=0)


def test_verify_runtime_memory_limit_uses_inspect(monkeypatch):
    runtime = SimpleNamespace(_container="pdc-test", name="pdc-test")

    def fake_inspect(container, *, docker_bin="docker"):
        assert container == "pdc-test"
        return {
            "memory_bytes": memory_gb_to_bytes(96),
            "oom_killed": False,
            "status": "running",
            "exit_code": 0,
            "error": None,
            "raw": {},
        }

    monkeypatch.setattr(
        "pretrain_data_curator.container_memory.inspect_container_memory",
        fake_inspect,
    )
    info = verify_runtime_memory_limit(runtime, configured_gb=96)
    assert info["memory_bytes"] == memory_gb_to_bytes(96)


def test_collect_oom_diagnostics_payload():
    payload = collect_oom_diagnostics(
        configured_gb=96,
        effective_memory_bytes=memory_gb_to_bytes(96),
        oom_killed=True,
        host_meminfo={"MemTotal": 200 * GIB, "MemAvailable": 120 * GIB},
        process_group={"pid": 7, "returncode": -9, "killed": True},
        container="abc",
    )
    assert payload["oom_killed"] is True
    assert payload["configured_memory_gb"] == 96.0
    assert payload["effective_memory_bytes"] == memory_gb_to_bytes(96)
    assert payload["host_memory_bytes"] == 200 * GIB
    assert payload["process_group"]["killed"] is True
    assert payload["container"] == "abc"


def _load_self_score_helpers():
    """Exec process-group helpers from the rendered self_score script template."""
    from pretrain_data_curator import self_score as mod

    script = mod._SCRIPT
    start = script.index("# Single-flight GPU trainer lock")
    end = script.index("\ndef decon_score")
    ns: dict[str, object] = {
        "os": os,
        "fcntl": __import__("fcntl"),
        "signal": signal,
        "subprocess": subprocess,
        "sys": sys,
        "time": time,
        "atexit": __import__("atexit"),
        "json": json,
    }
    exec(compile(script[start:end], "<self_score_helpers>", "exec"), ns)
    return ns


def test_self_score_script_embeds_single_flight_and_process_group_controls():
    from pretrain_data_curator import self_score as mod

    script = mod._SCRIPT
    assert (
        b"start_new_session=True" in script.encode()
        or "start_new_session=True" in script
    )
    assert "_train_lock" in script
    assert "_terminate_process_group" in script
    assert "_run_in_process_group" in script
    assert "killpg" in script
    rendered = render_self_score_script(CuratorConfig(use_real_trainer=True))
    text = rendered.decode()
    assert "start_new_session=True" in text
    assert "_train_lock" in text


def test_process_group_timeout_kills_and_reaps_children():
    helpers = _load_self_score_helpers()
    run = helpers["_run_in_process_group"]
    child = (
        "import os, signal, time\n"
        "if os.fork() == 0:\n"
        "    while True:\n"
        "        time.sleep(0.2)\n"
        "while True:\n"
        "    time.sleep(0.2)\n"
    )
    with pytest.raises(subprocess.TimeoutExpired) as excinfo:
        run([sys.executable, "-c", child], timeout=0.3)
    details = getattr(excinfo.value, "process_group", {})
    assert details.get("timed_out") is True
    assert details.get("terminated") or details.get("killed")
    # Parent session leader should be reaped; no orphaned living pid.
    pid = details.get("pid")
    assert pid is not None
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_process_group_success_path():
    helpers = _load_self_score_helpers()
    run = helpers["_run_in_process_group"]
    result, details = run(
        [sys.executable, "-c", "print('ok')"],
        timeout=5,
    )
    assert result.returncode == 0
    assert "ok" in (result.stdout or "")
    assert details["returncode"] == 0
    assert details.get("timed_out") is False


def test_train_lock_serializes_overlapping_holders(tmp_path):
    helpers = _load_self_score_helpers()
    lock_path = str(tmp_path / "train.lock")
    hold = threading.Event()
    first_holding = threading.Event()
    second_acquired = threading.Event()
    overlapping = {"seen": False}

    def first():
        fh = helpers["_train_lock"](lock_path)
        first_holding.set()
        # If second acquires while we still hold, that is a failure.
        hold.wait(timeout=2)
        time.sleep(0.05)
        overlapping["seen"] = second_acquired.is_set()
        helpers["_release_train_lock"](fh)

    def second():
        first_holding.wait(timeout=2)
        fh = helpers["_train_lock"](lock_path)
        second_acquired.set()
        helpers["_release_train_lock"](fh)

    t1 = threading.Thread(target=first)
    t2 = threading.Thread(target=second)
    t1.start()
    assert first_holding.wait(timeout=1)
    t2.start()
    time.sleep(0.1)
    assert not second_acquired.is_set(), "second holder acquired while first still held"
    hold.set()
    t1.join(timeout=2)
    t2.join(timeout=2)
    assert second_acquired.is_set()
    assert overlapping["seen"] is False
