"""Regressions for the "healthy long run looks like a hang" failure mode.

A 300-turn A100 rollout ended with reward 0 and `harness 'codex' exited 143`:
`self_score.py` ran silently for minutes (sampling + decon + trainer startup
produce no output), the agent read the silence as a hang, killed "hanging"
processes, and took down its own Codex process group. The eval CLI still exited
0, so the detached run recorded `EXIT=0`.

These tests pin the four defenses:

* self_score emits flushed, bounded stderr heartbeats and keeps stdout pure JSON;
* a source (or the whole candidate) that samples zero documents reports
  ``ok: false`` with an actionable reason, never a healthy-looking reward 0;
* the task prompt tells the agent that a silent multi-minute self_score is
  normal and that killing processes can kill its own harness;
* the detached A100 status marker is never ``EXIT=0`` for a failed rollout;
* stale process-group cleanup never signals a group it cannot verify.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pytest
from pretrain_data_curator.models import CuratorConfig
from pretrain_data_curator.self_score import (
    SELF_SCORE_FILENAME,
    render_self_score_script,
)
from pretrain_data_curator.tasks import TASK_PROMPT, build_tasks
from test_400m_eval_detached import _extract_bash_heredoc

REPO_ROOT = Path(__file__).resolve().parents[3]
EVAL_SCRIPT = REPO_ROOT / "scripts" / "run_400m_eval_a100.sh"

# Same budget the existing prompt contract enforces (test_task_prompt_contract):
# the new hang/kill guidance has to fit inside it, not extend it.
TASK_PROMPT_MAX_CHARS = 5_850


def _self_score_helpers() -> dict[str, Any]:
    """Exec the rendered script (no __main__) to drive its helpers directly."""
    namespace: dict[str, Any] = {"__name__": "self_score_under_test"}
    exec(
        compile(
            render_self_score_script(CuratorConfig(token_budget=1_000)),
            SELF_SCORE_FILENAME,
            "exec",
        ),
        namespace,
    )
    return namespace


def _write_manifest(tmp_path: Path, sources: list[dict], **extra) -> Path:
    path = tmp_path / "draft.json"
    path.write_text(
        json.dumps({"token_budget": 1_000, "sources": sources, **extra}),
        encoding="utf-8",
    )
    return path


def _render_script(tmp_path: Path) -> bytes:
    """Render with an absent decon evals dir: these tests exercise sampling and
    reporting, and the bundled decon detector takes minutes on any real corpus."""
    return render_self_score_script(
        CuratorConfig(token_budget=1_000),
        decon_evals_dir=str(tmp_path / "absent-evals"),
    )


def _run_self_score(tmp_path: Path, manifest: Path, *args: str, env=None):
    (tmp_path / SELF_SCORE_FILENAME).write_bytes(_render_script(tmp_path))
    result = subprocess.run(
        [sys.executable, SELF_SCORE_FILENAME, manifest.name, *args],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, **(env or {})},
    )
    return result


# --- heartbeats -------------------------------------------------------------


def test_self_score_emits_phase_heartbeats_without_corrupting_json(tmp_path: Path):
    (tmp_path / "dev.jsonl").write_text(
        "\n".join(
            json.dumps({"text": "Clean development sample " * 30}) for _ in range(4)
        ),
        encoding="utf-8",
    )
    manifest = _write_manifest(
        tmp_path,
        [
            {
                "kind": "local",
                "local_path": "dev.jsonl",
                "local_format": "jsonl",
                "weight": 1.0,
            }
        ],
    )
    result = _run_self_score(tmp_path, manifest, "--limit", "4")

    assert result.returncode == 0, result.stderr
    # stdout stays a single machine-readable JSON object: no heartbeat leaked in.
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert "[self-score]" not in result.stdout

    phases = [
        line.split("phase=")[1].split()[0]
        for line in result.stderr.splitlines()
        if line.startswith("[self-score] phase=")
    ]
    # corpus/sample, scoring and completion are all distinguishable from a hang.
    for phase in (
        "start",
        "sampling",
        "sampled",
        "corpus_complete",
        "scoring",
        "complete",
    ):
        assert phase in phases, (phase, result.stderr)
    assert all(
        "elapsed=" in line
        for line in result.stderr.splitlines()
        if line.startswith("[self-score] phase=")
    )


def test_progress_lines_are_flushed_immediately(tmp_path: Path):
    """A heartbeat must reach stderr as it happens, not at interpreter shutdown."""
    rendered = tmp_path / SELF_SCORE_FILENAME
    rendered.write_bytes(_render_script(tmp_path))
    driver = (
        "import os\n"
        f"ns = {{'__name__': 'self_score_flush_test'}}\n"
        f"exec(compile(open({str(rendered)!r}).read(), 'self_score.py', 'exec'), ns)\n"
        "ns['progress']('train_running', pid=1)\n"
        "os._exit(0)\n"  # hard exit: buffered output would be lost
    )
    result = subprocess.run(
        [sys.executable, "-c", driver],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert "[self-score] phase=train_running" in result.stderr, result.stderr


def test_communicate_with_heartbeat_reports_liveness_and_still_times_out():
    helpers = _self_score_helpers()
    communicate = helpers["_communicate_with_heartbeat"]

    os.environ["PDC_SELF_SCORE_HEARTBEAT_SECONDS"] = "1"
    try:
        # 1) slow-but-healthy child: heartbeats emitted, output still returned.
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(2.5); print('done')"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        started = time.monotonic()
        stdout, _stderr = communicate(proc, 30, "train_running")
        assert "done" in stdout
        assert proc.returncode == 0
        assert time.monotonic() - started >= 2.0

        # 2) the caller's overall timeout still fires (heartbeats do not extend it).
        slow = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            with pytest.raises(subprocess.TimeoutExpired):
                communicate(slow, 2, "train_running")
        finally:
            slow.kill()
            slow.communicate()
    finally:
        os.environ.pop("PDC_SELF_SCORE_HEARTBEAT_SECONDS", None)


# --- zero-document diagnostics ----------------------------------------------


def test_empty_source_file_is_not_reported_as_healthy(tmp_path: Path):
    (tmp_path / "empty.jsonl").write_text("", encoding="utf-8")
    manifest = _write_manifest(
        tmp_path,
        [
            {
                "kind": "local",
                "local_path": "empty.jsonl",
                "local_format": "jsonl",
                "weight": 1.0,
            }
        ],
    )
    payload = json.loads(_run_self_score(tmp_path, manifest).stdout)

    assert payload["ok"] is False
    assert payload["error"], "a zero-document candidate must explain itself"
    assert payload["sampled_documents"] == 0
    # never a healthy-looking zero: an untrained candidate has no reward at all.
    assert payload["reward"] is None
    assert payload["perf"] is None
    source = payload["sources"][0]
    assert source["ok"] is False
    assert "read 0 lines" in source["reason"]
    assert source["observed"]["records_read"] == 0


def test_missing_text_field_names_the_observed_fields(tmp_path: Path):
    (tmp_path / "rows.jsonl").write_text(
        "\n".join(
            json.dumps({"body": "text under the wrong key " * 20}) for _ in range(3)
        ),
        encoding="utf-8",
    )
    manifest = _write_manifest(
        tmp_path,
        [
            {
                "kind": "local",
                "local_path": "rows.jsonl",
                "local_format": "jsonl",
                "text_field": "content",
                "weight": 1.0,
            }
        ],
    )
    payload = json.loads(_run_self_score(tmp_path, manifest).stdout)

    assert payload["ok"] is False
    source = payload["sources"][0]
    assert source["sampled_documents"] == 0
    assert "text_field='content' matched no field" in source["reason"]
    assert "body" in source["reason"]
    assert source["observed"]["records_read"] == 3
    assert source["observed"]["observed_fields"] == ["body"]


def test_filters_eliminating_every_document_are_reported(tmp_path: Path):
    (tmp_path / "short.jsonl").write_text(
        "\n".join(json.dumps({"text": "short"}) for _ in range(3)), encoding="utf-8"
    )
    manifest = _write_manifest(
        tmp_path,
        [
            {
                "kind": "local",
                "local_path": "short.jsonl",
                "local_format": "jsonl",
                "weight": 1.0,
                "filters": [{"kind": "min_chars", "params": {"value": 200}}],
            }
        ],
    )
    payload = json.loads(_run_self_score(tmp_path, manifest).stdout)

    assert payload["ok"] is False
    source = payload["sources"][0]
    assert "filters removed all 3 sampled documents" in source["reason"]
    assert "min_chars" in source["reason"]


def test_one_dead_source_among_live_ones_is_surfaced(tmp_path: Path):
    (tmp_path / "good.jsonl").write_text(
        "\n".join(
            json.dumps({"text": "Clean development sample " * 30}) for _ in range(4)
        ),
        encoding="utf-8",
    )
    (tmp_path / "empty.jsonl").write_text("", encoding="utf-8")
    manifest = _write_manifest(
        tmp_path,
        [
            {
                "kind": "local",
                "local_path": "good.jsonl",
                "local_format": "jsonl",
                "weight": 1.0,
            },
            {
                "kind": "local",
                "local_path": "empty.jsonl",
                "local_format": "jsonl",
                "weight": 1.0,
            },
        ],
    )
    payload = json.loads(_run_self_score(tmp_path, manifest).stdout)

    assert payload["ok"] is False
    assert "1 of 2 sources sampled zero documents" in payload["error"]
    assert "empty.jsonl" in payload["error"]
    by_source = {s["source"]: s for s in payload["sources"]}
    assert by_source["good.jsonl"]["ok"] is True
    assert by_source["empty.jsonl"]["ok"] is False
    # the live source still trained/scored: real reward semantics are preserved.
    assert payload["sampled_documents"] == 4
    assert isinstance(payload["reward"], float)


def test_valid_candidate_keeps_genuine_numeric_reward(tmp_path: Path):
    (tmp_path / "good.jsonl").write_text(
        "\n".join(
            json.dumps({"text": "Clean development sample " * 30}) for _ in range(4)
        ),
        encoding="utf-8",
    )
    manifest = _write_manifest(
        tmp_path,
        [
            {
                "kind": "local",
                "local_path": "good.jsonl",
                "local_format": "jsonl",
                "weight": 1.0,
            }
        ],
    )
    payload = json.loads(_run_self_score(tmp_path, manifest).stdout)

    assert payload["ok"] is True
    assert payload["error"] is None
    assert payload["sources"][0]["reason"] is None
    # A trained (or heuristic) candidate keeps a numeric reward -- including 0.0.
    assert isinstance(payload["reward"], float)
    assert isinstance(payload["perf_reward"], float)


# --- task prompt ------------------------------------------------------------


def test_prompt_warns_that_self_score_is_slow_and_must_not_be_killed():
    prompt = str(build_tasks("2024-12-31", 1_000_000)[0].prompt)
    lowered = prompt.lower()
    assert "many minutes" in lowered
    assert "heartbeat" in lowered
    assert "kill" in lowered
    assert "harness" in lowered
    assert "--train-timeout" in prompt
    # existing guidance is preserved
    assert "hf papers" in prompt
    assert "/workspace/.agents/skills/hf-cli/SKILL.md" in prompt
    assert "loading script" in prompt
    assert "python self_score.py /workspace/manifest.json" in prompt
    assert "does not end the episode" in prompt


def test_prompt_stays_within_its_length_contract():
    prompt = str(build_tasks("2024-12-31", 1_000_000)[0].prompt)
    assert len(TASK_PROMPT) <= TASK_PROMPT_MAX_CHARS, len(TASK_PROMPT)
    assert len(prompt) <= TASK_PROMPT_MAX_CHARS, len(prompt)


# --- detached A100 status propagation ---------------------------------------


def _eval_wrapper_repo(
    tmp_path: Path, row: dict | None, *, uv_exit: int = 0, log_results=True
):
    """Temp repo + fake `uv` so the real eval.sh wrapper can run end to end.

    The wrapper prepends ``$HOME/.local/bin`` to PATH, so the fake `uv` has to
    live there (a fake earlier in PATH would still lose to the real one).
    """
    repo = tmp_path / "webcurator-gym"
    env_dir = repo / "environments" / "pretrain_data_curator"
    run_dir = env_dir / "outputs" / "run-abc"
    run_dir.mkdir(parents=True)
    (repo / "secrets.env").write_text("HF_TOKEN=dummy\nPRIME_API_KEY=dummy\n")
    if row is not None:
        (run_dir / "results.jsonl").write_text(json.dumps(row) + "\n")

    home = tmp_path / "home"
    bin_dir = home / ".local" / "bin"
    bin_dir.mkdir(parents=True)
    uv = bin_dir / "uv"
    line = (
        'echo "results: outputs/run-abc"' if log_results else 'echo "no results line"'
    )
    uv.write_text("#!/usr/bin/env bash\n%s\nexit %d\n" % (line, uv_exit))
    uv.chmod(0o755)

    script = tmp_path / "eval.sh"
    script.write_text(
        _extract_bash_heredoc(EVAL_SCRIPT.read_text(encoding="utf-8"), "EVAL")
    )
    script.chmod(0o755)
    return repo, home, script


def _run_eval_wrapper_repo(repo: Path, home: Path, script: Path) -> tuple[str, str]:
    status = script.parent / "status"
    log = script.parent / "eval.log"
    env = dict(os.environ)
    env["HOME"] = str(home)
    env["WCG_REPO_ROOT"] = str(repo)
    with log.open("w") as handle:
        subprocess.run(
            ["bash", str(script), "model", "config.toml", str(status), str(log)],
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            timeout=120,
        )
    return status.read_text().strip(), log.read_text()


def _run_eval_wrapper(tmp_path: Path, row: dict | None, **kwargs) -> tuple[str, str]:
    repo, home, script = _eval_wrapper_repo(tmp_path, row, **kwargs)
    return _run_eval_wrapper_repo(repo, home, script)


_HEALTHY_ROW = {
    "is_completed": True,
    "stop_condition": "completed",
    "errors": [],
    "rewards": {"reward": 0.42},
    "metrics": {"perf_loss": 3.1, "corpus_tokens": 12_000},
}


def test_status_is_exit_zero_only_for_a_finalized_successful_rollout(tmp_path: Path):
    status, log = _run_eval_wrapper(tmp_path, _HEALTHY_ROW)
    assert status == "EXIT=0", log
    assert "[validate] OK" in log


@pytest.mark.parametrize(
    "row,expected",
    [
        pytest.param(
            {**_HEALTHY_ROW, "stop_condition": "error", "rewards": {"reward": 0.0}},
            "stop_condition=error",
            id="stop_condition_error",
        ),
        pytest.param(
            {**_HEALTHY_ROW, "stop_condition": "truncation"},
            "stop_condition=truncation",
            id="stop_condition_truncation",
        ),
        pytest.param(
            {
                **_HEALTHY_ROW,
                "errors": [
                    {"type": "HarnessError", "message": "harness 'codex' exited 143"}
                ],
            },
            "rollout error",
            id="harness_error",
        ),
        pytest.param(
            {**_HEALTHY_ROW, "metrics": {}}, "empty metrics", id="empty_metrics"
        ),
        pytest.param(
            {**_HEALTHY_ROW, "is_completed": False}, "not finalized", id="not_finalized"
        ),
        pytest.param(None, "missing or empty", id="no_results_file"),
    ],
)
def test_failed_rollout_never_reports_exit_zero(tmp_path: Path, row, expected):
    status, log = _run_eval_wrapper(tmp_path, row)
    assert status == "EXIT=65", log
    assert expected in log


def test_validation_uses_this_runs_logged_dir_not_a_repo_scan(tmp_path: Path):
    # A stale healthy results dir elsewhere in outputs/ must not rescue a run
    # whose own logged directory is bad.
    repo, home, script = _eval_wrapper_repo(
        tmp_path,
        {**_HEALTHY_ROW, "stop_condition": "error"},
    )
    stale = repo / "environments" / "pretrain_data_curator" / "outputs" / "old-good"
    stale.mkdir(parents=True)
    (stale / "results.jsonl").write_text(json.dumps(_HEALTHY_ROW) + "\n")

    status, log = _run_eval_wrapper_repo(repo, home, script)
    assert status == "EXIT=65", log


def test_nonzero_eval_exit_is_preserved(tmp_path: Path):
    status, _log = _run_eval_wrapper(tmp_path, _HEALTHY_ROW, uv_exit=3)
    assert status == "EXIT=3"


def test_missing_results_line_in_log_fails_closed(tmp_path: Path):
    status, log = _run_eval_wrapper(tmp_path, _HEALTHY_ROW, log_results=False)
    assert status == "EXIT=65", log
    assert "no usable results path" in log


# --- process-group safety ---------------------------------------------------


def test_guard_refuses_to_signal_a_pid_that_is_not_its_group_leader():
    """A recorded pid that is no longer its own group leader must never be killed:
    killpg would hit whatever group now owns that id -- possibly the harness's."""
    helpers = _self_score_helpers()
    guard = helpers["_group_signal_guard"]
    # No start_new_session -> the child lives in THIS process's group, so its pid
    # is not a pgid (exactly the PID-reuse shape the guard exists to reject).
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        assert helpers["_process_pgid"](child.pid) == os.getpgid(child.pid) != child.pid
        assert guard(child.pid) == "not_session_leader"

        details = helpers["_terminate_pgid"](child.pid, grace_seconds=0.2)
        assert details["skipped"] is True
        assert details["reason"] == "not_session_leader"
        assert details["terminated"] is False and details["killed"] is False
        time.sleep(0.1)
        assert child.poll() is None, (
            "guarded cleanup must not signal an unverified group"
        )
    finally:
        child.kill()
        child.wait(timeout=5)


def test_guard_rejects_mismatched_identity_and_missing_leader():
    helpers = _self_score_helpers()
    guard = helpers["_group_signal_guard"]
    leader = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        starttime = helpers["_pgid_starttime"](leader.pid)
        # a live leader with the recorded starttime is the one case we may signal
        assert guard(leader.pid, expected_starttime=starttime) is None
        assert guard(leader.pid, expected_starttime="0") == "identity_mismatch"
    finally:
        leader.kill()
        leader.wait(timeout=5)

    # a pid with no /proc entry is only signalled for a group we still own
    dead = leader.pid
    assert guard(dead, expected_starttime=None) == "leader_gone"
    assert guard(dead, expected_starttime=None, allow_missing_leader=True) is None
    assert guard(None) == "no_pgid"


def test_stale_lock_holder_that_is_not_a_leader_is_never_signalled(tmp_path: Path):
    helpers = _self_score_helpers()
    lock_path = str(tmp_path / "stale.lock")
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        with open(lock_path, "a+", encoding="utf-8") as fh:
            helpers["_write_lock_pgid"](fh, child.pid)
        lock = helpers["_train_lock"](lock_path)
        helpers["_release_train_lock"](lock)
        time.sleep(0.1)
        assert child.poll() is None, "stale-lock recovery signalled a non-leader pid"
    finally:
        child.kill()
        child.wait(timeout=5)


def test_owned_process_group_still_sweeps_grandchildren(tmp_path: Path):
    """The guard must not weaken cleanup of a trainer group we created."""
    helpers = _self_score_helpers()
    pidfile = tmp_path / "grandchild.pid"
    child_src = (
        "import os, time\n"
        "from pathlib import Path\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"
        "    while True:\n"
        "        time.sleep(0.2)\n"
        f"Path({str(pidfile)!r}).write_text(str(pid), encoding='utf-8')\n"
        "while True:\n"
        "    time.sleep(0.2)\n"
    )
    with pytest.raises(subprocess.TimeoutExpired) as excinfo:
        helpers["_run_in_process_group"]([sys.executable, "-c", child_src], timeout=0.5)
    details = getattr(excinfo.value, "process_group", {})
    assert details.get("timed_out") is True
    assert not details.get("skipped")
    assert details.get("terminated") or details.get("killed")

    deadline = time.monotonic() + 5
    grandchild = None
    while time.monotonic() < deadline:
        raw = pidfile.read_text(encoding="utf-8").strip() if pidfile.exists() else ""
        if raw.isdigit():
            grandchild = int(raw)
            break
        time.sleep(0.05)
    assert grandchild is not None
    with pytest.raises(ProcessLookupError):
        os.kill(grandchild, 0)
