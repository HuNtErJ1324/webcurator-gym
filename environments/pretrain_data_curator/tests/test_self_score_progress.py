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
import shlex
import signal
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

# Same budget the existing prompt contract enforces (test_task_prompt_contract).
TASK_PROMPT_MAX_CHARS = 6_000


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


def test_heartbeats_reach_stderr_while_a_child_is_still_running(tmp_path: Path):
    """Heartbeat lines must be readable *during* the wait, before any timeout kill.

    Reads the driver's stderr live: a heartbeat has to arrive while the child is
    still alive, which is the only thing that distinguishes a slow run from a hang.
    """
    rendered = tmp_path / SELF_SCORE_FILENAME
    rendered.write_bytes(_render_script(tmp_path))
    driver = (
        "import subprocess, sys\n"
        f"ns = {{'__name__': 'self_score_live_test'}}\n"
        f"exec(compile(open({str(rendered)!r}).read(), 'self_score.py', 'exec'), ns)\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(20)'],\n"
        "                        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)\n"
        "try:\n"
        "    ns['_communicate_with_heartbeat'](child, 20, 'train_running')\n"
        "finally:\n"
        "    child.kill()\n"
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", driver],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env={**os.environ, "PDC_SELF_SCORE_HEARTBEAT_SECONDS": "1"},
    )
    try:
        assert proc.stderr is not None
        line = proc.stderr.readline()  # blocks only until the first heartbeat
        assert "[self-score] phase=train_running" in line, line
        # ... and it arrived while both the driver and its child were still running
        assert proc.poll() is None
    finally:
        proc.kill()
        proc.communicate()


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
    assert payload["sampled_documents"] == 4
    # A partially sampled candidate was never scored as written: no number at all,
    # so it can never be misread as a score (least of all a 0.0 one).
    assert payload["reward"] is None
    assert payload["perf_reward"] is None
    assert payload["leakage_penalty"] is None


def test_valid_candidate_keeps_genuine_numeric_zero_reward(tmp_path: Path):
    """The heuristic (no-trainer) path scores exactly 0.0 -- and stays ok=true.

    This is the case the zero-document contract must never swallow: a real,
    fully sampled candidate whose reward genuinely is 0.0.
    """
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
    assert payload["sources"][0]["ok"] is True
    # no trainer and no decon here: perf term 0.0, leakage term 0.0 -> reward 0.0
    assert payload["perf"] is None
    assert payload["perf_reward"] == 0.0
    assert payload["leakage_penalty"] == 0.0
    assert payload["reward"] == 0.0


def test_decon_timeout_still_produces_the_single_json_result(tmp_path: Path):
    """A decon that never finishes must not cost the run its result."""
    evals = tmp_path / "evals"
    evals.mkdir()
    (evals / "eval.jsonl").write_text(json.dumps({"text": "eval"}) + "\n")
    slow_decon = tmp_path / "slow-decon"
    slow_decon.write_text("#!/usr/bin/env bash\nsleep 120\n")
    slow_decon.chmod(0o755)

    (tmp_path / SELF_SCORE_FILENAME).write_bytes(
        render_self_score_script(
            CuratorConfig(token_budget=1_000),
            decon_binary=str(slow_decon),
            decon_evals_dir=str(evals),
        )
    )
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
    result = subprocess.run(
        [sys.executable, SELF_SCORE_FILENAME, manifest.name],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        env={
            **os.environ,
            "PDC_SELF_SCORE_DECON_TIMEOUT": "2",
            "PDC_SELF_SCORE_HEARTBEAT_SECONDS": "1",
        },
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)  # stdout is still exactly one JSON object
    assert payload["ok"] is True
    assert payload["leakage_score"] is None
    assert payload["leakage_penalty"] == 0.0
    assert payload["reward"] == 0.0
    assert "decon timed out after 2s" in result.stderr
    assert "phase=decon_timeout" in result.stderr
    # the detector was reaped, not left hanging around the workspace
    assert "phase=complete" in result.stderr


# --- task prompt ------------------------------------------------------------


def test_prompt_warns_that_self_score_is_slow_and_must_not_be_killed():
    prompt = str(build_tasks("2024-12-31", 1_000_000)[0].data.prompt)
    lowered = prompt.lower()
    assert "long silent runs" in lowered
    assert "idle gpu" in lowered
    assert "heartbeat" in lowered
    assert "never kill or signal" in lowered
    assert "harness" in lowered
    assert "wait for it to return or time out" in lowered
    assert "--train-timeout" in prompt


def test_prompt_distinguishes_zero_doc_diagnostic_from_a_trained_zero():
    prompt = str(build_tasks("2024-12-31", 1_000_000)[0].data.prompt)
    assert "`ok: false` + `reward: null`" in prompt
    assert "not a score" in prompt
    assert "sampled zero documents" in prompt
    # ... and that a real scored mixture may legitimately report exactly 0.0
    assert "`ok: true` with a numeric reward, even 0.0" in prompt


def test_prompt_preserves_existing_guidance():
    prompt = str(build_tasks("2024-12-31", 1_000_000)[0].data.prompt)
    # existing guidance is preserved
    assert "hf papers" in prompt
    assert "/workspace/.agents/skills/hf-cli/SKILL.md" in prompt
    assert "loading script" in prompt
    assert "python self_score.py /workspace/manifest.json" in prompt
    assert "does not end the episode" in prompt


def test_prompt_section_order_is_objective_setup_research_deliverable_self_score_rules():
    prompt = str(build_tasks("2024-12-31", 1_000_000)[0].data.prompt)
    section_headers = [
        "## Objective",
        "## Setup",
        "## Research",
        "## Deliverable",
        "## Self-score (you run it)",
        "## Rules",
    ]
    positions = [prompt.index(header) for header in section_headers]
    assert positions == sorted(positions)
    for header in section_headers:
        assert TASK_PROMPT.count(header) == 1


def test_prompt_stays_within_its_length_contract():
    prompt = str(build_tasks("2024-12-31", 1_000_000)[0].data.prompt)
    assert len(TASK_PROMPT) <= TASK_PROMPT_MAX_CHARS, len(TASK_PROMPT)
    assert len(prompt) <= TASK_PROMPT_MAX_CHARS, len(prompt)


def test_prompt_explains_squared_performance_matching_reward_default():
    """Setup must document squared nonnegative progress; reward default is 2.0."""
    prompt = str(build_tasks("2024-12-31", 1_000_000)[0].data.prompt)
    setup = prompt[prompt.index("## Setup") : prompt.index("## Research")]
    assert "normalized loss progress is squared in the performance term" in setup
    assert "default exponent 2.0" in setup
    assert "equal loss improvements earn more reward later than earlier" in setup
    assert "negative progress stays linear" in setup
    # Live default must match the documented exponent (do not hard-code drift).
    assert CuratorConfig().perf_scaling_exponent == 2.0
    assert "default exponent 2.0" in TASK_PROMPT


# --- detached A100 status propagation ---------------------------------------

# The wrapper runs the result gate with the pod's project Python and refuses any
# interpreter that is not 3.12 (the pod provisions `uv venv -p 3.12`). This project
# supports 3.11 too, so the test session's own interpreter cannot stand in for the
# pod's -- it would be rejected at the preflight and no wrapper test would ever
# reach semantic validation.
#
# The fixture therefore provisions a shim interpreter that models the pod's 3.12
# exactly where it matters and nowhere else: it answers the wrapper's *exact*
# version probe with a 3.12 version, and delegates every other invocation --
# notably running result_gate.py -- transparently to the real test interpreter.
# Nothing depends on which interpreters happen to be installed on the machine.
#
# Kept byte-identical to the probe in scripts/run_400m_eval_a100.sh; the pairing is
# pinned by test_project_python_shim_answers_the_wrappers_exact_probe.
_VERSION_PROBE_CODE = (
    'import sys, tomllib; print(".".join(map(str, sys.version_info[:3])))'
)


def _python_calls(tmp_path: Path) -> Path:
    """Every invocation the wrapper makes of the provisioned project Python."""
    return tmp_path / "project-python-calls.txt"


def _write_project_python_shim(
    python: Path, *, version: str, calls: Path, delegate: str | None
) -> None:
    """Write a project-Python shim that spoofs *only* the wrapper's version probe.

    ``delegate`` is the real interpreter every other invocation is handed to; when
    it is None (the rejected-interpreter case), any second invocation fails loudly,
    so a run that slipped past the preflight is visible as a recorded gate call.
    """
    if delegate is None:
        fallthrough = (
            'echo "rejected project Python was asked to run: $*" >&2\nexit 1\n'
        )
    else:
        fallthrough = f'exec {shlex.quote(delegate)} "$@"\n'
    python.write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%s\\n' \"$*\" >> {shlex.quote(str(calls))}\n"
        f'if [[ $# -eq 2 && "$1" == "-c" && "$2" == {shlex.quote(_VERSION_PROBE_CODE)} ]]; then\n'
        f"  printf '%s\\n' {shlex.quote(version)}\n"
        "  exit 0\n"
        "fi\n" + fallthrough,
        encoding="utf-8",
    )
    python.chmod(0o755)


def _eval_wrapper_repo(
    tmp_path: Path,
    row: dict | None,
    *,
    uv_exit: int = 0,
    log_results=True,
    python_version: str = "3.12.13",
    delegate: bool = True,
):
    """Temp repo + fake `uv` so the real eval.sh wrapper can run end to end.

    The wrapper prepends ``$HOME/.local/bin`` to PATH, so the fake `uv` has to
    live there (a fake earlier in PATH would still lose to the real one).
    """
    repo = tmp_path / "webcurator-gym"
    env_dir = repo / "environments" / "pretrain_data_curator"
    run_dir = env_dir / "outputs" / "run-abc"
    run_dir.mkdir(parents=True)
    gate_src = (
        REPO_ROOT
        / "environments"
        / "pretrain_data_curator"
        / "pretrain_data_curator"
        / "result_gate.py"
    )
    gate_dst = env_dir / "pretrain_data_curator"
    gate_dst.mkdir()
    (gate_dst / gate_src.name).write_bytes(gate_src.read_bytes())
    (env_dir / "config.toml").write_text(
        "[args]\n"
        "token_budget = 400000000\n"
        "use_real_trainer = true\n"
        "[args.proxy_student]\n"
        "train_token_budget = 400000000\n"
    )
    project_bin = repo / ".venv" / "bin"
    project_bin.mkdir(parents=True)
    _write_project_python_shim(
        project_bin / "python",
        version=python_version,
        calls=_python_calls(tmp_path),
        delegate=sys.executable if delegate else None,
    )
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
    "metrics": {
        "finalized": 1.0,
        "manifest_missing": 0.0,
        "manifest_invalid": 0.0,
        "corpus_tokens": 400_000_000.0,
        "num_sources": 3.0,
        "train_flops": 1.2e18,
        "perf_loss": 3.1,
        "trainer_error_msg": 0.0,
    },
}


def test_project_python_shim_spoofs_only_the_wrappers_exact_probe(tmp_path: Path):
    """The shim models 3.12 for one exact command and is transparent otherwise.

    Pins the shim to the probe production actually runs: if the wrapper's probe
    changes, the shim stops matching it (and delegates, reporting the session's
    real 3.11) rather than silently spoofing a command production no longer uses.
    """
    script = EVAL_SCRIPT.read_text(encoding="utf-8")
    assert f"\"$PROJECT_PYTHON\" -c '{_VERSION_PROBE_CODE}'" in script

    shim = tmp_path / "python"
    _write_project_python_shim(
        shim,
        version="3.12.13",
        calls=_python_calls(tmp_path),
        delegate=sys.executable,
    )

    def run(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [str(shim), *args], capture_output=True, text=True, timeout=60
        )

    assert run("-c", _VERSION_PROBE_CODE).stdout.strip() == "3.12.13"
    # Any other invocation is the real interpreter, unspoofed: same code with an
    # extra argument, a different snippet, and real script execution all pass
    # straight through.
    assert run("-c", _VERSION_PROBE_CODE, "extra").stdout.strip() == (
        ".".join(map(str, sys.version_info[:3]))
    )
    assert run("-c", "print('delegated')").stdout.strip() == "delegated"
    assert run("-c", "import sys; print(sys.executable)").stdout.strip() == (
        sys.executable
    )


def test_status_is_exit_zero_only_for_a_finalized_successful_rollout(tmp_path: Path):
    status, log = _run_eval_wrapper(tmp_path, _HEALTHY_ROW)
    assert status == "EXIT=0", log
    assert "valid_rows=1 mode=production" in log
    # Non-vacuous: the shim answered the version probe, then handed the *real*
    # interpreter the result gate, which is what produced that verdict.
    recorded = _python_calls(tmp_path).read_text(encoding="utf-8").splitlines()
    assert len(recorded) == 2, recorded
    assert recorded[0] == f"-c {_VERSION_PROBE_CODE}"
    assert "pretrain_data_curator/result_gate.py" in recorded[1]
    assert "results.jsonl" in recorded[1]


def test_non_3_12_project_python_fails_before_semantic_validation(tmp_path: Path):
    """A provisioned interpreter that is not 3.12 never reaches the result gate."""
    status, log = _run_eval_wrapper(
        tmp_path, _HEALTHY_ROW, python_version="3.11.14", delegate=False
    )
    assert status == "SEMANTIC_INVALID=65", log
    assert "provisioned project Python must be 3.12 (got 3.11.14" in log
    # ... and it stopped there: the gate never ran, so nothing was validated.
    assert "valid_rows=" not in log
    recorded = _python_calls(tmp_path).read_text(encoding="utf-8").splitlines()
    assert recorded == [f"-c {_VERSION_PROBE_CODE}"], recorded


@pytest.mark.parametrize("mode", ["missing", "not_executable"])
def test_result_gate_missing_project_python_fails_closed_with_diagnostic(
    tmp_path: Path, mode: str
):
    repo, home, script = _eval_wrapper_repo(tmp_path, _HEALTHY_ROW)
    project_python = repo / ".venv" / "bin" / "python"
    project_python.unlink()
    if mode == "not_executable":
        project_python.write_text("not an executable\n")
        project_python.chmod(0o644)

    status, log = _run_eval_wrapper_repo(repo, home, script)
    assert status == "SEMANTIC_INVALID=65", log
    assert "provisioned project Python is unavailable or not executable" in log
    assert str(project_python) in log


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
            "has errors=",
            id="harness_error",
        ),
        pytest.param(
            {**_HEALTHY_ROW, "metrics": {}},
            "empty or missing metrics",
            id="empty_metrics",
        ),
        pytest.param(
            {**_HEALTHY_ROW, "is_completed": False},
            "not completed",
            id="not_finalized",
        ),
        pytest.param(
            {**_HEALTHY_ROW, "rewards": {"reward": None}},
            "missing numeric reward",
            id="nested_null_reward",
        ),
        pytest.param(
            {**_HEALTHY_ROW, "rewards": {}, "reward": None},
            "missing numeric reward",
            id="flat_null_reward",
        ),
        pytest.param(None, "missing or empty results file", id="no_results_file"),
    ],
)
def test_failed_rollout_never_reports_exit_zero(tmp_path: Path, row, expected):
    status, log = _run_eval_wrapper(tmp_path, row)
    assert status == "SEMANTIC_INVALID=65", log
    assert expected in log
    # the verdict came from the real result gate, not from the interpreter preflight
    recorded = _python_calls(tmp_path).read_text(encoding="utf-8").splitlines()
    assert len(recorded) == 2, recorded
    assert "pretrain_data_curator/result_gate.py" in recorded[1]


def test_finalized_rollout_with_genuine_zero_reward_exits_zero(tmp_path: Path):
    """reward 0.0 is a real score, not a missing one: it must not fail the run."""
    status, log = _run_eval_wrapper(
        tmp_path, {**_HEALTHY_ROW, "rewards": {"reward": 0.0}}
    )
    assert status == "EXIT=0", log
    assert "valid_rows=1 mode=production" in log
    recorded = _python_calls(tmp_path).read_text(encoding="utf-8").splitlines()
    assert "pretrain_data_curator/result_gate.py" in recorded[1], recorded


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
    assert status == "SEMANTIC_INVALID=65", log


def test_nonzero_eval_exit_is_preserved(tmp_path: Path):
    status, _log = _run_eval_wrapper(tmp_path, _HEALTHY_ROW, uv_exit=3)
    assert status == "EXIT=3"


def test_missing_results_line_in_log_fails_closed(tmp_path: Path):
    status, log = _run_eval_wrapper(tmp_path, _HEALTHY_ROW, log_results=False)
    assert status == "SEMANTIC_INVALID=65", log
    assert "no usable results path" in log


# --- process-group safety ---------------------------------------------------


class _KillpgRecorder:
    """Stand-in for the script's ``os`` that records every real signal sent.

    Signal 0 (liveness probe) is delegated; anything else is recorded and
    swallowed, so a regression that signals an unverified group is caught as a
    recorded call instead of by killing the test session's own group.
    """

    def __init__(self):
        self.signals: list[tuple[int, int]] = []

    def killpg(self, pgid, sig):
        if sig == 0:
            return os.killpg(pgid, 0)
        self.signals.append((int(pgid), int(sig)))

    def __getattr__(self, name):
        return getattr(os, name)


def test_guard_fails_closed_without_a_recorded_identity():
    """No recorded starttime means nothing proves the pid is still ours."""
    helpers = _self_score_helpers()
    guard = helpers["_group_signal_guard"]
    leader = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        # Live, is its own group leader, but no identity was recorded: refuse.
        assert guard(leader.pid) == "no_recorded_identity"
        assert guard(leader.pid, allow_missing_leader=True) == "no_recorded_identity"
        assert guard(None) == "no_pgid"

        recorder = _KillpgRecorder()
        helpers["os"] = recorder
        details = helpers["_terminate_pgid"](leader.pid, grace_seconds=0.2)
        assert details["skipped"] is True
        assert details["reason"] == "no_recorded_identity"
        assert recorder.signals == [], "signalled a group with no recorded identity"
        assert leader.poll() is None
    finally:
        helpers["os"] = os
        leader.kill()
        leader.wait(timeout=5)


def test_guard_refuses_a_reused_pid_and_a_pid_that_is_not_its_group_leader():
    """PID reuse: same number, different process -> never signalled."""
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
        # the one signallable case: live leader of its own group, identity matches
        assert guard(leader.pid, expected_starttime=starttime) is None
        # same pid, different starttime == a recycled pid: refuse
        assert guard(leader.pid, expected_starttime="0") == "identity_mismatch"

        recorder = _KillpgRecorder()
        helpers["os"] = recorder
        details = helpers["_terminate_pgid"](
            leader.pid, grace_seconds=0.2, expected_starttime="0"
        )
        assert details["skipped"] is True
        assert details["reason"] == "identity_mismatch"
        assert recorder.signals == [], "signalled a recycled pid"
        assert leader.poll() is None
    finally:
        helpers["os"] = os
        leader.kill()
        leader.wait(timeout=5)

    # a pid living in someone else's group (no start_new_session) is not a pgid
    child = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        starttime = helpers["_pgid_starttime"](child.pid)
        assert helpers["_process_pgid"](child.pid) == os.getpgid(child.pid) != child.pid
        assert guard(child.pid, expected_starttime=starttime) == "not_session_leader"
        recorder = _KillpgRecorder()
        helpers["os"] = recorder
        details = helpers["_terminate_pgid"](
            child.pid, grace_seconds=0.2, expected_starttime=starttime
        )
        assert details["skipped"] is True
        assert details["reason"] == "not_session_leader"
        assert recorder.signals == [], "signalled a group we do not lead"
        assert child.poll() is None
    finally:
        helpers["os"] = os
        child.kill()
        child.wait(timeout=5)


def test_guard_treats_a_missing_leader_as_signallable_only_for_an_owned_group():
    helpers = _self_score_helpers()
    guard = helpers["_group_signal_guard"]
    dead = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    dead.wait(timeout=5)  # reaped: no /proc entry remains

    # Without a recorded identity we never signal, even for an owned group.
    assert guard(dead.pid, allow_missing_leader=True) == "no_recorded_identity"
    # With one: a stale lock holder is not signalled ...
    assert guard(dead.pid, expected_starttime="123") == "leader_gone"
    # ... while a group we created may still be swept (killpg can only reach
    # surviving members or raise ESRCH -- a reused pgid needs a live pid==pgid).
    assert guard(dead.pid, expected_starttime="123", allow_missing_leader=True) is None


def test_pid_reuse_between_sigterm_and_sigkill_blocks_the_second_signal():
    """TOCTOU: identity is revalidated immediately before *each* signal."""
    helpers = _self_score_helpers()
    real_stat = helpers["_proc_stat_fields"]
    leader = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import signal, time\n"
            "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
            "time.sleep(30)\n",
        ],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        starttime = helpers["_pgid_starttime"](leader.pid)
        recorder = _KillpgRecorder()
        helpers["os"] = recorder

        def flaky_stat(pid):
            """After the first real signal, /proc reports a *different* process."""
            fields = real_stat(pid)
            if fields is not None and recorder.signals:
                fields = list(fields)
                fields[19] = "999999999"  # pid recycled between SIGTERM and SIGKILL
            return fields

        helpers["_proc_stat_fields"] = flaky_stat
        details = helpers["_terminate_pgid"](
            leader.pid, grace_seconds=0.3, expected_starttime=starttime
        )
        # SIGTERM went out under a matching identity; SIGKILL was refused.
        assert details["terminated"] is True
        assert details.get("killed") is not True
        assert details.get("reason") == "identity_mismatch"
        assert [sig for _pgid, sig in recorder.signals] == [signal.SIGTERM]
        assert leader.poll() is None
    finally:
        helpers["_proc_stat_fields"] = real_stat
        helpers["os"] = os
        leader.kill()
        leader.wait(timeout=5)


def test_fast_exiting_child_is_reaped_without_any_signal():
    helpers = _self_score_helpers()
    result, details = helpers["_run_in_process_group"](
        [sys.executable, "-c", "print('quick')"], timeout=30
    )
    assert result.returncode == 0
    assert "quick" in (result.stdout or "")
    assert details["timed_out"] is False
    assert details["returncode"] == 0
    assert not details.get("skipped")


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
        recorder = _KillpgRecorder()
        helpers["os"] = recorder
        lock = helpers["_train_lock"](lock_path)
        helpers["_release_train_lock"](lock)
        assert recorder.signals == [], "stale-lock recovery signalled a non-leader pid"
        assert child.poll() is None
    finally:
        helpers["os"] = os
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

    # non-vacuous: the grandchild really existed, and it is really gone
    deadline = time.monotonic() + 5
    grandchild = None
    while time.monotonic() < deadline:
        raw = pidfile.read_text(encoding="utf-8").strip() if pidfile.exists() else ""
        if raw.isdigit():
            grandchild = int(raw)
            break
        time.sleep(0.05)
    assert grandchild is not None, "grandchild never started; test would be vacuous"
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        try:
            os.kill(grandchild, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    with pytest.raises(ProcessLookupError):
        os.kill(grandchild, 0)
