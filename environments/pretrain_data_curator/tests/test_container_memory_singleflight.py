"""Regression tests for container memory preflight/cgroup checks and self_score single-flight."""

from __future__ import annotations

import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import verifiers.v1 as vf

from pretrain_data_curator.container_memory import (
    DEFAULT_HOST_HEADROOM_GIB,
    ENV_CONTAINER_MEMORY_GB,
    ENV_DOCKER_CONTAINER_MEMORY_GB,
    ENV_SKIP_MEMORY_PREFLIGHT,
    GIB,
    ContainerMemoryError,
    assert_cgroup_memory_limit,
    assert_host_supports_container_memory,
    classify_trainer_failure,
    collect_oom_diagnostics,
    memory_gb_to_bytes,
    parse_cgroup_memory_events,
    parse_cgroup_memory_max,
    parse_docker_memory_limit_bytes,
    parse_meminfo,
    read_host_meminfo,
    resolve_container_memory_gb,
    verify_runtime_memory_limit,
)
from pretrain_data_curator.models import CuratorConfig, ProxyStudentConfig
from pretrain_data_curator.runtime_config import derive_trainer_resources
from pretrain_data_curator.self_score import render_self_score_script
from pretrain_data_curator.rollout_state import CuratorState
from pretrain_data_curator.taskset import CuratorTaskset, CuratorTasksetConfig


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


def test_resolve_container_memory_gb_docker_override_bounds(monkeypatch):
    monkeypatch.delenv(ENV_DOCKER_CONTAINER_MEMORY_GB, raising=False)
    monkeypatch.delenv(ENV_CONTAINER_MEMORY_GB, raising=False)
    assert resolve_container_memory_gb(48, backend="docker") == 48.0
    monkeypatch.setenv(ENV_DOCKER_CONTAINER_MEMORY_GB, "96")
    assert resolve_container_memory_gb(48, backend="docker") == 96.0
    monkeypatch.setenv(ENV_DOCKER_CONTAINER_MEMORY_GB, "0.5")
    with pytest.raises(ContainerMemoryError, match="must be in"):
        resolve_container_memory_gb(48, backend="docker")
    monkeypatch.setenv(ENV_DOCKER_CONTAINER_MEMORY_GB, "4096")
    with pytest.raises(ContainerMemoryError, match="must be in"):
        resolve_container_memory_gb(48, backend="docker")


def test_docker_memory_override_does_not_affect_modal(monkeypatch):
    monkeypatch.setenv(ENV_DOCKER_CONTAINER_MEMORY_GB, "96")
    monkeypatch.setenv(ENV_CONTAINER_MEMORY_GB, "128")
    docker = derive_trainer_resources(
        ProxyStudentConfig(runtime_backend="docker", memory_gb=48, gpu_count=1),
        backend="docker",
    )
    modal = derive_trainer_resources(
        ProxyStudentConfig(runtime_backend="modal", memory_gb=48, gpu_count=1),
        backend="modal",
    )
    assert docker["memory"] == 96.0
    assert modal["memory"] == 48.0


def test_legacy_container_memory_env_still_honored_for_docker(monkeypatch):
    monkeypatch.delenv(ENV_DOCKER_CONTAINER_MEMORY_GB, raising=False)
    monkeypatch.setenv(ENV_CONTAINER_MEMORY_GB, "96")
    assert resolve_container_memory_gb(48, backend="docker") == 96.0


def test_derive_trainer_resources_honors_memory_override(monkeypatch):
    monkeypatch.setenv(ENV_DOCKER_CONTAINER_MEMORY_GB, "96")
    ps = ProxyStudentConfig(runtime_backend="docker", memory_gb=48, gpu_count=1)
    resources = derive_trainer_resources(ps, backend="docker")
    assert resources["memory"] == 96.0


def test_default_headroom_allows_100gib_hosts_for_96gib_pin():
    assert DEFAULT_HOST_HEADROOM_GIB <= 4.0
    total_kib = int(100 * GIB / 1024)
    assert_host_supports_container_memory(96, meminfo_text=_meminfo(total_kib))


def test_host_preflight_passes_with_sufficient_ram(monkeypatch):
    monkeypatch.delenv(ENV_SKIP_MEMORY_PREFLIGHT, raising=False)
    total_kib = int(100 * GIB / 1024)
    assert_host_supports_container_memory(
        96, headroom_gb=DEFAULT_HOST_HEADROOM_GIB, meminfo_text=_meminfo(total_kib)
    )


def test_host_preflight_fails_when_host_ram_too_small(monkeypatch):
    monkeypatch.delenv(ENV_SKIP_MEMORY_PREFLIGHT, raising=False)
    total_kib = int(64 * GIB / 1024)
    with pytest.raises(ContainerMemoryError, match="host RAM cannot support"):
        assert_host_supports_container_memory(
            96, headroom_gb=DEFAULT_HOST_HEADROOM_GIB, meminfo_text=_meminfo(total_kib)
        )


def test_read_host_meminfo_closes_handle(tmp_path):
    path = tmp_path / "meminfo"
    path.write_text(_meminfo(2048), encoding="utf-8")
    info = read_host_meminfo(str(path))
    assert info["MemTotal"] == 2048 * 1024
    # Re-open/replace should succeed if the previous handle was closed.
    path.write_text(_meminfo(4096), encoding="utf-8")
    assert read_host_meminfo(str(path))["MemTotal"] == 4096 * 1024


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


def test_classify_trainer_failure_from_cgroup_events_and_signals():
    assert (
        classify_trainer_failure(
            timed_out=True,
            events_before={"events": {"oom_kill": 0}},
            events_after={"events": {"oom_kill": 1}},
        )
        == "timeout"
    )
    assert (
        classify_trainer_failure(
            returncode=1,
            stderr="RuntimeError: CUDA out of memory",
            events_before={"events": {"oom_kill": 0}},
            events_after={"events": {"oom_kill": 0}},
        )
        == "cuda_oom"
    )
    # cgroup oom_kill delta takes precedence over CUDA text.
    assert (
        classify_trainer_failure(
            returncode=1,
            stderr="RuntimeError: CUDA out of memory",
            events_before={"events": {"oom_kill": 0}},
            events_after={"events": {"oom_kill": 1}},
        )
        == "cgroup_oom"
    )
    assert (
        classify_trainer_failure(
            returncode=-9,
            stderr="",
            events_before={"events": {"oom": 0, "oom_kill": 0}},
            events_after={"events": {"oom": 1, "oom_kill": 1}},
        )
        == "cgroup_oom"
    )
    assert (
        classify_trainer_failure(
            returncode=-9,
            stderr="",
            events_before={"events": {"oom_kill": 0}},
            events_after={"events": {"oom_kill": 0}},
            docker_oom_killed=True,
        )
        == "cgroup_oom"
    )
    assert (
        classify_trainer_failure(
            returncode=-9,
            stderr="",
            events_before={"events": {"oom_kill": 0}},
            events_after={"events": {"oom_kill": 0}},
            docker_oom_killed=False,
        )
        == "external_sigkill"
    )
    assert (
        classify_trainer_failure(
            returncode=-9,
            process_group={"killed": True, "cleanup_reason": "error"},
        )
        == "unknown"
    )
    assert (
        classify_trainer_failure(
            returncode=-9,
            process_group={
                "killed": True,
                "cleanup_reason": "timeout",
                "timed_out": True,
            },
        )
        == "timeout"
    )
    assert parse_cgroup_memory_events("oom 1\noom_kill 2\n") == {
        "oom": 1,
        "oom_kill": 2,
    }
    assert parse_cgroup_memory_max("max") is None
    assert parse_cgroup_memory_max("12345") == 12345


def test_collect_oom_diagnostics_payload():
    payload = collect_oom_diagnostics(
        configured_gb=96,
        effective_memory_bytes=memory_gb_to_bytes(96),
        oom_killed=True,
        host_meminfo={"MemTotal": 200 * GIB, "MemAvailable": 120 * GIB},
        process_group={"pid": 7, "returncode": -9, "killed": True},
        container="abc",
        events_before={"events": {"oom_kill": 0}},
        events_after={"events": {"oom_kill": 1}},
        returncode=-9,
        stderr="",
    )
    assert payload["oom_killed"] is True
    assert payload["configured_memory_gb"] == 96.0
    assert payload["effective_memory_bytes"] == memory_gb_to_bytes(96)
    assert payload["host_memory_bytes"] == 200 * GIB
    assert payload["process_group"]["killed"] is True
    assert payload["container"] == "abc"
    assert payload["kill_class"] == "cgroup_oom"


def _load_self_score_helpers():
    """Exec process-group helpers from the rendered self_score script template."""
    from pretrain_data_curator import self_score as mod

    script = mod._SCRIPT
    start = script.index("# --- Progress heartbeats")
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
        "re": re,
    }
    exec(compile(script[start:end], "<self_score_helpers>", "exec"), ns)
    return ns


def test_self_score_script_embeds_single_flight_and_process_group_controls():
    from pretrain_data_curator import self_score as mod

    script = mod._SCRIPT
    assert "start_new_session=True" in script
    assert "pass_fds=" in script
    assert "_write_lock_pgid" in script
    assert "_reap_stale_lock_holder" in script
    assert "_self_score_signal_handler" in script
    assert "_train_lock" in script
    assert "_terminate_process_group" in script
    assert "_run_in_process_group" in script
    assert "killpg" in script
    assert "LOCK_NB" in script
    assert "def _snapshot_cgroup_memory" in script
    assert "def _classify_trainer_kill" in script
    assert "_pgid_starttime" in script
    rendered = render_self_score_script(CuratorConfig(use_real_trainer=True))
    text = rendered.decode()
    assert "start_new_session=True" in text
    assert "pass_fds=" in text
    assert "_train_lock" in text
    assert "def _snapshot_cgroup_memory" in text
    assert "def _classify_trainer_kill" in text
    compile(text, "<rendered_self_score>", "exec")


def test_process_group_timeout_kills_and_reaps_grandchild(tmp_path):
    helpers = _load_self_score_helpers()
    run = helpers["_run_in_process_group"]
    pidfile = tmp_path / "grandchild.pid"
    child = textwrap.dedent(
        f"""
        import os, time
        from pathlib import Path
        pid = os.fork()
        if pid == 0:
            while True:
                time.sleep(0.2)
        Path({str(pidfile)!r}).write_text(str(pid), encoding="utf-8")
        while True:
            time.sleep(0.2)
        """
    )
    with pytest.raises(subprocess.TimeoutExpired) as excinfo:
        run([sys.executable, "-c", child], timeout=0.4)
    details = getattr(excinfo.value, "process_group", {})
    assert details.get("timed_out") is True
    assert details.get("terminated") or details.get("killed")
    pid = details.get("pid")
    assert pid is not None
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    # Wait briefly for pidfile flush races, then prove grandchild is dead.
    deadline = time.monotonic() + 2
    grandchild = None
    while time.monotonic() < deadline:
        if pidfile.exists():
            raw = pidfile.read_text(encoding="utf-8").strip()
            if raw.isdigit():
                grandchild = int(raw)
                break
        time.sleep(0.05)
    assert grandchild is not None, "grandchild pid was not captured"
    with pytest.raises(ProcessLookupError):
        os.kill(grandchild, 0)


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


def test_cross_process_single_flight_lock(tmp_path):
    helpers = _load_self_score_helpers()
    lock_path = str(tmp_path / "cross.lock")
    events_path = tmp_path / "events.log"
    # Avoid nested f-strings; %% escapes feed the outer % path interpolation.
    script = textwrap.dedent(
        """
        import fcntl, time
        path = %r
        events = %r
        started = time.monotonic()
        fh = open(path, "a+", encoding="utf-8")
        contended = False
        while True:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                contended = True
                if time.monotonic() - started > 2.0:
                    fh.close()
                    raise SystemExit("lock wait timed out")
                time.sleep(0.02)
        held_at = time.monotonic()
        with open(events, "a", encoding="utf-8") as ev:
            ev.write('acquired contended=%%s t=%%.6f\\n' %% (contended, held_at))
        time.sleep(0.2)
        fh.close()
        """
    ) % (lock_path, str(events_path))
    fh = helpers["_train_lock"](lock_path)
    child = subprocess.Popen(
        [sys.executable, "-c", script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    time.sleep(0.2)
    assert child.poll() is None, (
        "child acquired/finished while parent still held lock: "
        f"rc={child.poll()} stderr={child.stderr.read() if child.stderr else ''}"
    )
    parent_release = time.monotonic()
    helpers["_release_train_lock"](fh)
    assert child.wait(timeout=2) == 0, child.stderr.read()
    lines = events_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines, "child never recorded lock acquisition"
    assert "contended=True" in lines[0], lines
    child_t = float(lines[0].split("t=")[1])
    assert child_t + 1e-3 >= parent_release, (child_t, parent_release)


def test_stale_pgid_recovery_terminates_orphan_group(tmp_path):
    helpers = _load_self_score_helpers()
    lock_path = str(tmp_path / "stale.lock")
    orphan = subprocess.Popen(
        [sys.executable, "-c", "import time\nwhile True:\n    time.sleep(0.2)\n"],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        with open(lock_path, "a+", encoding="utf-8") as fh:
            helpers["_write_lock_pgid"](fh, orphan.pid)
        # New holder must reap the recorded stale pgid after acquiring the lock.
        fh = helpers["_train_lock"](lock_path)
        helpers["_release_train_lock"](fh)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and orphan.poll() is None:
            time.sleep(0.05)
        assert orphan.poll() is not None
        with pytest.raises(ProcessLookupError):
            os.kill(orphan.pid, 0)
    finally:
        if orphan.poll() is None:
            os.killpg(orphan.pid, signal.SIGKILL)
            orphan.wait(timeout=2)


def test_stale_pgid_recovery_skips_pid_reuse(tmp_path):
    helpers = _load_self_score_helpers()
    lock_path = str(tmp_path / "reuse.lock")
    victim = subprocess.Popen(
        [sys.executable, "-c", "import time\nwhile True:\n    time.sleep(0.2)\n"],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        # Record a matching pgid with a fake starttime so identity mismatches.
        with open(lock_path, "w", encoding="utf-8") as fh:
            fh.write(f"{victim.pid}:0\n")
        fh = helpers["_train_lock"](lock_path)
        helpers["_release_train_lock"](fh)
        time.sleep(0.1)
        assert victim.poll() is None, "PID-reuse mismatch must not kill live process"
    finally:
        os.killpg(victim.pid, signal.SIGKILL)
        victim.wait(timeout=2)


def test_train_lock_times_out_instead_of_blocking_forever(tmp_path):
    helpers = _load_self_score_helpers()
    lock_path = str(tmp_path / "busy.lock")
    holder = open(lock_path, "a+", encoding="utf-8")
    fcntl.flock(holder.fileno(), fcntl.LOCK_EX)
    try:
        t0 = time.monotonic()
        with pytest.raises(RuntimeError, match="another trainer holds the GPU"):
            helpers["_train_lock"](lock_path, timeout=0.25)
        assert time.monotonic() - t0 < 1.5
    finally:
        fcntl.flock(holder.fileno(), fcntl.LOCK_UN)
        holder.close()


def test_terminate_process_group_reaps_zombie_quickly():
    helpers = _load_self_score_helpers()
    # Child ignores SIGTERM briefly then exits; parent must reap without ~10s stall.
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import signal,time\n"
                "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
                "time.sleep(0.2)\n"
                "signal.signal(signal.SIGTERM, signal.SIG_DFL)\n"
                "time.sleep(0.05)\n"
            ),
        ],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    t0 = time.monotonic()
    details = helpers["_terminate_process_group"](child, grace_seconds=0.5)
    elapsed = time.monotonic() - t0
    assert details["reaped"] is True
    assert child.poll() is not None
    assert elapsed < 3.0, elapsed


def test_signal_handler_cleans_active_process_group(tmp_path):
    helpers = _load_self_score_helpers()
    # Drive the handler directly with a live fake active proc.
    child = subprocess.Popen(
        [sys.executable, "-c", "import time\nwhile True:\n    time.sleep(0.2)\n"],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    helpers["_ACTIVE_TRAIN_PROC"] = child
    # Avoid re-raising SIGTERM into the pytest process: call cleanup path used by handler.
    helpers["_cleanup_active_train_proc"]()
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline and child.poll() is None:
        time.sleep(0.05)
    assert child.poll() is not None
    # Handler restores default then re-raises; verify install + restore wiring exists.
    assert callable(helpers["_self_score_signal_handler"])
    helpers["_install_train_signal_handlers"]()
    # Restore pytest-friendly dispositions after install.
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGHUP, signal.SIG_DFL)


def test_signal_delivery_terminates_trainer_group_and_exits(tmp_path):
    """Real subprocess: helper installs handlers, gets SIGTERM, trainer PGID dies."""
    from pretrain_data_curator import self_score as mod

    script = mod._SCRIPT
    hs = script.index("# --- Progress heartbeats")
    he = script.index("\ndef decon_score")
    helper_src = script[hs:he]
    ready = tmp_path / "ready.txt"
    helper_path = tmp_path / "signal_helper.py"
    helper_path.write_text(
        "import atexit, fcntl, json, os, re, signal, subprocess, sys, time\n"
        + helper_src
        + "\n"
        + textwrap.dedent(
            f"""
            def main():
                _install_train_signal_handlers()
                trainer = subprocess.Popen(
                    [sys.executable, "-c", "import time\\nwhile True:\\n    time.sleep(0.2)\\n"],
                    start_new_session=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                global _ACTIVE_TRAIN_PROC
                _ACTIVE_TRAIN_PROC = trainer
                Path = __import__("pathlib").Path
                Path({str(ready)!r}).write_text(str(trainer.pid), encoding="utf-8")
                while True:
                    time.sleep(0.2)

            if __name__ == "__main__":
                main()
            """
        ),
        encoding="utf-8",
    )
    proc = subprocess.Popen(
        [sys.executable, str(helper_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    trainer_pid = None
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if ready.exists():
            raw = ready.read_text(encoding="utf-8").strip()
            if raw.isdigit():
                trainer_pid = int(raw)
                break
        if proc.poll() is not None:
            break
        time.sleep(0.05)
    assert trainer_pid is not None, (
        proc.poll(),
        proc.stderr.read() if proc.stderr else None,
    )
    os.kill(proc.pid, signal.SIGTERM)
    rc = proc.wait(timeout=5)
    # Default disposition after re-raise yields 128+signum or -SIGTERM depending on platform.
    assert rc in (128 + signal.SIGTERM, -signal.SIGTERM, signal.SIGTERM), rc
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            os.kill(trainer_pid, 0)
            time.sleep(0.05)
        except ProcessLookupError:
            break
    with pytest.raises(ProcessLookupError):
        os.kill(trainer_pid, 0)
    # Wiring covers SIGINT/SIGHUP as well as SIGTERM.
    assert "signal.SIGINT" in helper_src and "signal.SIGHUP" in helper_src


def test_rendered_self_score_script_compiles_and_train_perf_smoke(tmp_path):
    """Compile the full rendered script and exercise train_perf with a stub trainer."""
    cfg = CuratorConfig(use_real_trainer=True)
    rendered = render_self_score_script(cfg).decode()
    compile(rendered, "<rendered_self_score_full>", "exec")
    assert "def _snapshot_cgroup_memory" in rendered
    assert "def _classify_trainer_kill" in rendered

    ss_path = tmp_path / "self_score.py"
    train_path = tmp_path / "self_score_train.py"
    ss_path.write_text(rendered, encoding="utf-8")
    train_path.write_text(
        textwrap.dedent(
            """
            #!/usr/bin/env python3
            import json, pathlib, sys
            workdir = sys.argv[1]
            pathlib.Path(workdir, "result.json").write_text(
                json.dumps({"loss": 2.5, "val_source": "stub"}),
                encoding="utf-8",
            )
            print("RESULT_JSON " + json.dumps({"loss": 2.5, "val_source": "stub"}))
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )

    ns: dict[str, object] = {
        "__name__": "self_score_under_test",
        "__file__": str(ss_path),
    }
    exec(compile(rendered, str(ss_path), "exec"), ns)
    train_perf = ns["train_perf"]
    loss, perf, backend = train_perf(
        ["hello world document for stub training"],
        max_corpus_chars=1000,
        max_steps=1,
        train_timeout=30,
    )
    assert loss == 2.5
    assert perf is not None and backend == "stub"
    # Undefined helper regression: calling classify/snapshot must work.
    snap = ns["_snapshot_cgroup_memory"]()
    assert isinstance(snap, dict) and "events" in snap
    assert (
        ns["_classify_trainer_kill"](
            returncode=-9,
            stderr="",
            events_before={"events": {"oom_kill": 0}},
            events_after={"events": {"oom_kill": 1}},
        )
        == "cgroup_oom"
    )


@pytest.mark.asyncio
async def test_heuristic_docker_setup_skips_memory_pin(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-token")

    class FakeRuntime:
        type = "docker"
        _container = "should-not-inspect"

        async def write(self, path, data):
            return None

    called = {"verify": False}

    def boom(*args, **kwargs):
        called["verify"] = True
        raise AssertionError("heuristic docker setup must not verify memory pin")

    monkeypatch.setattr(
        "pretrain_data_curator.container_memory.verify_runtime_memory_limit",
        boom,
    )
    taskset = CuratorTaskset(
        CuratorTasksetConfig(
            id="pretrain-data-curator",
            use_real_trainer=False,
            proxy_student={"runtime_backend": "docker"},
        )
    )
    task = taskset.load()[0]
    trace = vf.Trace(
        task=vf.TraceTask(type=type(task).__name__, data=task.data),
        state=CuratorState(),
    )
    await task.setup(trace, FakeRuntime())
    assert called["verify"] is False


@pytest.mark.asyncio
async def test_real_docker_setup_verifies_memory_pin(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-token")

    class FakeRuntime:
        type = "docker"
        _container = "pdc-real"

        async def write(self, path, data):
            return None

    seen = {}

    def fake_verify(runtime, *, configured_gb, docker_bin="docker"):
        seen["container"] = getattr(runtime, "_container", None)
        seen["configured_gb"] = configured_gb
        return {"memory_bytes": memory_gb_to_bytes(configured_gb)}

    monkeypatch.setattr(
        "pretrain_data_curator.container_memory.verify_runtime_memory_limit",
        fake_verify,
    )
    taskset = CuratorTaskset(
        CuratorTasksetConfig(
            id="pretrain-data-curator",
            use_real_trainer=True,
            proxy_student={"runtime_backend": "docker", "memory_gb": 96},
        )
    )
    task = taskset.load()[0]
    trace = vf.Trace(
        task=vf.TraceTask(type=type(task).__name__, data=task.data),
        state=CuratorState(),
    )
    await task.setup(trace, FakeRuntime())
    assert seen["container"] == "pdc-real"
    assert seen["configured_gb"] == 96.0


def test_on_pod_eval_toml_quoting_and_missing_memory_gb(tmp_path):
    script = (
        Path(__file__).resolve().parents[3] / "scripts" / "run_400m_eval_on_pod.sh"
    ).read_text(encoding="utf-8")
    assert "uv run python - <<'PY'" in script
    assert 'os.environ.get("EVAL_TOML")' in script
    assert "memory_gb is required" in script
    assert 'Path("$EVAL_TOML")' not in script

    # Extract the quoted heredoc body and execute it against crafted TOMLs.
    start = script.index("uv run python - <<'PY'\n") + len("uv run python - <<'PY'\n")
    end = script.index("\nPY\n", start)
    body = script[start:end]

    # The on-pod script reads top-level args.proxy_student; mirror that shape,
    # including metacharacters in the path to prove EVAL_TOML quoting.
    good = tmp_path / 'cfg with spaces & "quotes".toml'
    good.write_text(
        textwrap.dedent(
            """
            [args.proxy_student]
            memory_gb = 96
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "EVAL_TOML": str(good),
        "PDC_SKIP_MEMORY_PREFLIGHT": "1",
        "PYTHONPATH": str(Path(__file__).resolve().parents[1]),
    }
    # Prepend package root so imports resolve without uv sync.
    pkg_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = os.pathsep.join([pkg_root, env.get("PYTHONPATH", "")]).rstrip(
        os.pathsep
    )
    proc = subprocess.run(
        [sys.executable, "-c", body],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "host memory OK" in proc.stdout

    missing = tmp_path / "missing.toml"
    missing.write_text("[args.proxy_student]\nsteps = 1\n", encoding="utf-8")
    env["EVAL_TOML"] = str(missing)
    proc = subprocess.run(
        [sys.executable, "-c", body],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "memory_gb is required" in (proc.stderr + proc.stdout)
