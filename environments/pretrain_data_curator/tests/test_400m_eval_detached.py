"""Deterministic tests for the durable detached remote eval design.

These drive the *actual* launcher script under a stateful fake SSH/rsync that
simulates a remote pod filesystem and the eval lifecycle, so we can prove:

* after the eval is launched detached (setsid/nohup + PID/status markers), a
  transient SSH disconnect (monitor probe failures) does NOT kill it and does
  NOT start a duplicate eval;
* the monitor reconnects, sees RUNNING, and eventually sees a successful
  completion before result validation/download proceeds;
* an explicit non-zero remote exit propagates to a launcher failure;
* a missing/dead PID with no completion marker fails safe;
* a monitor timeout triggers cleanup (pod termination) while preserving the
  remote log/status;
* secrets are sourced, never echoed.

The remote state is modelled by ``WCG_RECORD_DIR/remote_root`` plus a few
mode env vars the fake SSH reads (WCG_TRANSIENT, WCG_NEVER_DONE,
WCG_EVAL_EXIT_CODE, WCG_NO_STATUS, WCG_DONE_AFTER_PROBES).
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import textwrap
from pathlib import Path

from test_400m_eval_a100_launcher import _extract_bash_function

REPO_ROOT = Path(__file__).resolve().parents[3]
EVAL_SCRIPT = REPO_ROOT / "scripts" / "run_400m_eval_a100.sh"

_AVAIL_MIXED = {
    "gpu_resources": [
        {
            "cloud_id": "mc-a100",
            "provider": "MassedCompute",
            "stock_status": "available",
            "price_per_hour": "0.90",
            "is_spot": False,
            "gpu_type": "A100_80GB",
        },
        {
            "cloud_id": "dc-a100",
            "provider": "DataCrunch",
            "stock_status": "available",
            "price_per_hour": "1.20",
            "is_spot": False,
            "gpu_type": "A100_80GB",
        },
    ]
}
_STATUS_ROOT = (
    '{"status":"ACTIVE","installation_status":"FINISHED",'
    '"ip":"203.0.113.9","ssh":"ssh root@203.0.113.9",'
    '"port_mappings":[{"internal":"22","external":"22"}]}'
)


def _write_script(path: Path, text: str) -> None:
    path.write_text(textwrap.dedent(text))
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _make_temp_repo(tmp_path: Path) -> Path:
    """Self-contained temp repo: copy the launcher and a minimal environments
    tree (only the eval configs the script references by path) so the run's
    LOCAL_OUT_DIR stays inside tmp_path and never touches the real repo.
    """
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy(EVAL_SCRIPT, repo / "scripts" / EVAL_SCRIPT.name)
    cfg_src = REPO_ROOT / "environments" / "pretrain_data_curator" / "configs" / "eval"
    cfg_dst = repo / "environments" / "pretrain_data_curator" / "configs" / "eval"
    cfg_dst.mkdir(parents=True)
    for toml in cfg_src.glob("*.toml"):
        shutil.copy(toml, cfg_dst / toml.name)
    package_src = (
        REPO_ROOT
        / "environments"
        / "pretrain_data_curator"
        / "pretrain_data_curator"
        / "result_gate.py"
    )
    package_dst = (
        repo / "environments" / "pretrain_data_curator" / "pretrain_data_curator"
    )
    package_dst.mkdir(parents=True)
    shutil.copy(package_src, package_dst / package_src.name)
    docs = repo / "docs"
    docs.mkdir(parents=True)
    (docs / "build_site.py").write_text(
        "import os, pathlib\n"
        "pathlib.Path(os.environ['WCG_RECORD_DIR'], 'site_rebuilt.log').write_text('yes\\n')\n"
    )
    site_data = docs / "site" / "data"
    site_data.mkdir(parents=True)
    (site_data / "manifest.json").write_text(
        json.dumps(
            {
                "run_count": 1,
                "runs": [
                    {"model": "z-ai/glm-5.2", "reward": 0.42},
                ],
            }
        )
    )
    (repo / "secrets.env").write_text("HF_TOKEN=dummy\nPRIME_API_KEY=dummy\n")
    return repo


_SSH_FAKE = r"""#!/usr/bin/env bash
set +e
stdin="$(cat)"
REC="${WCG_RECORD_DIR:?}"
ROOT="$REC/remote_root/"
map() { echo "$ROOT$(printf '%s' "$1" | sed 's#^/##')"; }

# PATCH (Prime Inference namespace-tool compat)
if printf '%s' "$stdin" | grep -q "PRIME_INFERENCE_DISABLE_NAMESPACE_TOOLS"; then
  if [ "${WCG_PATCH_FAIL:-0}" = "1" ]; then echo "forced patch failure" >&2; exit 1; fi
  exit 0
fi

# LAUNCH (detached eval)
if printf '%s' "$stdin" | grep -q "setsid nohup bash"; then
  # script passes dynamic values as positional args after `bash -s`:
  # run_dir eval_log eval_pid eval_status MODEL EVAL_CONFIG REMOTE_ROOT
  args=("$@")
  n=${#args[@]}
  rd="${args[n-7]}"
  logf="${args[n-6]}"
  pidf="${args[n-5]}"
  stf="${args[n-4]}"
  MODEL="${args[n-3]}"
  EVAL_CONFIG="${args[n-2]}"
  REMOTE_ROOT="${args[n-1]}"
  # capture the generated remote shell text for quoting assertions
  printf '%s\n' "$stdin" > "$REC/launch_stdin.log"
  if [ -f "$(map "$pidf")" ]; then
    printf 'dup\n' >> "$REC/launch_count.log"
    echo "already running (idempotent skip)"
    exit 0
  fi
  mkdir -p "$(dirname "$(map "$pidf")")"
  echo "99999" > "$(map "$pidf")"
  if [ "${WCG_NO_STATUS:-0}" != "1" ]; then
    echo "RUNNING" > "$(map "$stf")"
  fi
  echo "launched" > "$(map "$logf")"
  date +%s > "$(map "$stf").launched"
  printf '1\n' >> "$REC/launch_count.log"
  echo "launched"
  exit 0
fi

# PROBE
if printf '%s' "$stdin" | grep -q "WCG_PROBE"; then
  # script passes paths as args after `bash -s`: run_dir eval_log eval_pid eval_status
  args=("$@")
  rd="${args[${#args[@]}-4]}"
  logf="${args[${#args[@]}-3]}"
  pidf="${args[${#args[@]}-2]}"
  stf="${args[${#args[@]}-1]}"
  pcfile="$REC/probe_count"
  pc=$(cat "$pcfile" 2>/dev/null || echo 0); pc=$((pc+1)); echo "$pc" > "$pcfile"
  if [ "${WCG_TRANSIENT:-0}" != "0" ] && [ "$pc" -le "${WCG_TRANSIENT:-0}" ]; then
    exit 1
  fi
  sf="$(map "$stf")"
  if [ -f "$sf" ]; then
    s=$(cat "$sf")
    if [[ "$s" == EXIT=* ]]; then echo "STATUS=done EXIT=${s#EXIT=}"; exit 0; fi
    # RUNNING marker present: flip to done once enough probes have elapsed
    if [ "${WCG_NEVER_DONE:-0}" = "1" ]; then echo "STATUS=running"; exit 0; fi
    if [ "${WCG_NO_STATUS:-0}" != "1" ] && [ "$pc" -ge "${WCG_DONE_AFTER_PROBES:-3}" ]; then
      code="${WCG_EVAL_EXIT_CODE:-0}"
      printf 'EXIT=%s\n' "$code" > "$sf.tmp" && mv "$sf.tmp" "$sf"
      printf 'results: outputs/2026-01-01/run0\n' >> "$(map "$logf")"
      echo "STATUS=done EXIT=$code"; exit 0
    fi
    echo "STATUS=running"; exit 0
  fi
  if [ "${WCG_NEVER_DONE:-0}" = "1" ]; then echo "STATUS=running"; exit 0; fi
  if [ "${WCG_NO_STATUS:-0}" != "1" ] && [ "$pc" -ge "${WCG_DONE_AFTER_PROBES:-3}" ]; then
    code="${WCG_EVAL_EXIT_CODE:-0}"
    printf 'EXIT=%s\n' "$code" > "$sf.tmp" && mv "$sf.tmp" "$sf"
    printf 'results: outputs/2026-01-01/run0\n' >> "$(map "$logf")"
    echo "STATUS=done EXIT=$code"; exit 0
  fi
  if [ -f "$(map "$pidf")" ]; then echo "STATUS=nostatus_deadpid"; exit 0; fi
  echo "STATUS=nostatus_nopid"; exit 0
fi

# FIND RESULTS
if printf '%s' "$stdin" | grep -q "WCG_FIND_RESULTS"; then
  # mirror the real function: grep the remote log for `results: outputs/...`,
  # strip exactly one leading outputs/, and reject unsafe paths.
  logf=$(printf '%s' "$stdin" | sed -n 's/.*LOG="\([^"]*\)".*/\1/p' | head -1)
  line=$(grep -Eo 'results: outputs/[^[:space:]]+' "$(map "$logf")" 2>/dev/null | tail -1)
  if [ -n "$line" ]; then
    rel="${line#results: }"
    rel="${rel#outputs/}"
    case "$rel" in
      ""|/*|*..*|*/.|*/.|./*) exit 1 ;;
    esac
    echo "$rel"; exit 0
  fi
  # fallback: deterministic relative dir (no outputs/ prefix), contract identical
  echo "2026-01-01/run0"; exit 0
fi

# TAIL
if printf '%s' "$stdin" | grep -q "WCG_TAIL"; then
  # script passes the log path as the sole arg after `bash -s`
  args=("$@")
  f="$(map "${args[${#args[@]}-1]}")"
  if [ -f "$f" ]; then tail -c 2000 "$f"; fi
  exit 0
fi

# default: provisioning / other heredocs -> success
exit 0
"""


_RSYNC_FAKE = r"""#!/usr/bin/env bash
printf '%s\n' "$*" >> "${WCG_RECORD_DIR:?}/rsync_args.log"
printf 'rsync\n' >> "${WCG_RECORD_DIR:?}/rsync_count.txt"
mode="${WCG_RSYNC_MODE:-ok}"
if [ "$mode" = "permanent" ]; then
  echo "rsync: send_files failed to open \"/root/webcurator-gym/secrets.env\": Permission denied (13)" >&2
  echo "rsync error: some files/attrs were not transferred (see previous errors) (code 23)" >&2
  exit 23
elif [ "$mode" = "mkdir_nosuchfile" ]; then
  echo 'rsync: mkdir "/root/webcurator-gym" failed: No such file or directory (2)' >&2
  exit 23
fi
# download destination is a local path (no @host:); write a dummy results.jsonl
last="${@: -1}"
if [[ "$last" != *"@"* ]]; then
  mkdir -p "$last"
  if [ -n "${WCG_RESULTS_JSON:-}" ]; then
    printf '%s\n' "$WCG_RESULTS_JSON" > "$last/results.jsonl"
  else
    cat > "$last/results.jsonl" <<'JSON'
{"is_completed":true,"stop_condition":"agent_completed","errors":[],"rewards":{"reward":0.42},"metrics":{"finalized":1.0,"manifest_missing":0.0,"manifest_invalid":0.0,"corpus_tokens":400000000.0,"num_sources":3.0,"train_flops":1.2e18,"perf_loss":3.1,"trainer_error_msg":0.0}}
JSON
  fi
  printf 'preserve me\n' > "$last/downloaded-artifact.txt"
fi
exit 0
"""


def _build_fakes(record_dir: Path) -> Path:
    bin_dir = record_dir / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    _write_script(
        bin_dir / "prime",
        r"""#!/usr/bin/env bash
set +e
case "$1" in
  whoami) exit 0 ;;
  availability)
    cat "${WCG_AVAIL_JSON:?}"
    exit 0
    ;;
  pods)
    sub="$2"
    if [ "$sub" = "create" ]; then
      cid=""
      while [ $# -gt 0 ]; do
        case "$1" in
          --cloud-id) cid="$2"; shift 2 ;;
          *) shift ;;
        esac
      done
      printf '%s\n' "$cid" >> "${WCG_RECORD_DIR:?}/picked.txt"
      echo "Successfully created pod POD123"
      exit 0
    elif [ "$sub" = "status" ]; then
      cat "${WCG_POD_STATUS_JSON:?}"
      exit 0
    elif [ "$sub" = "terminate" ]; then
      printf 'terminated\n' >> "${WCG_RECORD_DIR:?}/terminated.log"
      exit 0
    fi
    ;;
esac
exit 0
""",
    )
    _write_script(bin_dir / "ssh", _SSH_FAKE)
    _write_script(
        bin_dir / "scp",
        r"""#!/usr/bin/env bash
printf 'scp %s\n' "$*" >> "${WCG_RECORD_DIR:?}/scp.log"
exit 0
""",
    )
    _write_script(bin_dir / "rsync", _RSYNC_FAKE)
    _write_script(
        bin_dir / "ssh-keygen",
        r"""#!/usr/bin/env bash
exit 0
""",
    )
    _write_script(
        bin_dir / "sleep",
        r"""#!/usr/bin/env bash
exit 0
""",
    )
    return bin_dir


def _run_script(
    tmp_path: Path,
    repo: Path,
    *,
    modes: dict | None = None,
    model="z-ai/glm-5.2",
    skip_site: bool = True,
):
    record_dir = tmp_path / "record"
    record_dir.mkdir(parents=True, exist_ok=True)
    bin_dir = _build_fakes(record_dir)
    avail_json = tmp_path / "avail.json"
    avail_json.write_text(json.dumps(_AVAIL_MIXED))
    home = tmp_path / "home"
    (home / ".ssh").mkdir(parents=True)
    (home / ".ssh" / "id_rsa").write_text("dummy")
    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["HOME"] = str(home)
    env["WCG_AVAIL_JSON"] = str(avail_json)
    status_json = tmp_path / "status.json"
    status_json.write_text(_STATUS_ROOT)
    env["WCG_POD_STATUS_JSON"] = str(status_json)
    env["WCG_RSYNC_MODE"] = "ok"
    env["WCG_RECORD_DIR"] = str(record_dir)
    for k, v in (modes or {}).items():
        env[k] = str(v)
    script = repo / "scripts" / EVAL_SCRIPT.name
    argv = [str(script), "--model", model]
    if skip_site:
        argv.append("--skip-site")
    result = subprocess.run(
        argv,
        env=env,
        cwd=str(repo),
        capture_output=True,
        text=True,
        timeout=120,
    )
    return result, record_dir


def _extract_bash_heredoc(text: str, marker: str) -> str:
    """Return the body of a `<<'MARKER' ... MARKER` heredoc (quoted delimiter).

    Works even when the same marker is reused (e.g. two `<<'REMOTE'` blocks),
    because it searches from the optional enclosing function name first.
    """
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if (
            f"<<'{marker}'" in line
            or f"<<-{marker}'" in line
            or f"<<{marker}" in line
            or f"<<-{marker}" in line
        ):
            start = i + 1
            break
    if start is None:
        raise AssertionError(f"heredoc marker '{marker}' not found")
    for j in range(start, len(lines)):
        if lines[j].strip() == marker:
            return "\n".join(lines[start:j])
    raise AssertionError(f"heredoc terminator '{marker}' not found")


def _extract_function_heredoc(text: str, func: str, marker: str) -> str:
    """Extract a heredoc that lives inside a specific bash function body."""
    body = _extract_bash_function(text, func)
    return _extract_bash_heredoc(body, marker)


def _run_bash_heredoc(body: str, *args: str) -> subprocess.CompletedProcess:
    """Execute a quoted-heredoc body as a standalone bash script with args."""
    with subprocess.Popen(
        ["bash", "-s", *args],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as proc:
        out, err = proc.communicate(body)
    return subprocess.CompletedProcess(proc.args, proc.returncode, out, err)


def _run_find_results(
    body: str, remote_root: Path, run_name: str
) -> subprocess.CompletedProcess:
    """Execute the find_remote_results_dir heredoc (the remote-side body) with
    remote_log/REMOTE_ROOT set, mirroring what the launcher's `remote` wrapper
    would expand before invoking `bash -s` on the pod.

    The heredoc is unquoted in the script, so the source escapes `$` (e.g.
    `\\$LOG`); unescape to simulate the text the remote bash actually receives.
    The nested `<<'PY'` fallback block is quoted, so `$REMOTE_ROOT` is expanded
    by the outer heredoc on the launcher host -- substitute it textually here.
    """
    body = body.replace("\\$", "$").replace("$REMOTE_ROOT", str(remote_root))
    env = dict(os.environ)
    env["REMOTE_ROOT"] = str(remote_root)
    env["remote_log"] = str(remote_root / f"wcg-eval-{run_name}.d" / "eval.log")
    with subprocess.Popen(
        ["bash", "-s"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    ) as proc:
        out, err = proc.communicate(body)
    return subprocess.CompletedProcess(proc.args, proc.returncode, out, err)


# --- actual-probe-herdoc regression (blocker 2) ----------------------------


def test_remote_eval_probe_heredoc_runs_against_real_files(tmp_path: Path):
    text = EVAL_SCRIPT.read_text(encoding="utf-8")
    body = _extract_function_heredoc(text, "_remote_eval_probe", "RM")

    rd = tmp_path / "run"
    log = tmp_path / "eval.log"
    pid = tmp_path / "eval.pid"
    st = tmp_path / "status"
    rd.mkdir()

    # 1) RUNNING marker -> STATUS=running
    st.write_text("RUNNING\n")
    out = _run_bash_heredoc(body, str(rd), str(log), str(pid), str(st))
    assert "STATUS=running" in out.stdout, out.stdout + out.stderr

    # 2) done EXIT=0
    st.write_text("EXIT=0\n")
    out = _run_bash_heredoc(body, str(rd), str(log), str(pid), str(st))
    assert "STATUS=done EXIT=0" in out.stdout, out.stdout + out.stderr

    # 3) done EXIT != 0 (non-zero preserved)
    st.write_text("EXIT=3\n")
    out = _run_bash_heredoc(body, str(rd), str(log), str(pid), str(st))
    assert "STATUS=done EXIT=3" in out.stdout, out.stdout + out.stderr

    # 4) semantic failure is distinct from an eval process exit of 65, allowing
    # artifact download without accidentally accepting an unrelated exit code.
    st.write_text("SEMANTIC_INVALID=65\n")
    out = _run_bash_heredoc(body, str(rd), str(log), str(pid), str(st))
    assert "STATUS=semantic_invalid EXIT=65" in out.stdout, out.stdout + out.stderr

    # 5) dead PID with missing status -> nostatus_deadpid (fails safe)
    st.unlink()
    pid.write_text("999999\n")  # not a live pid on the test host
    out = _run_bash_heredoc(body, str(rd), str(log), str(pid), str(st))
    assert "STATUS=nostatus_deadpid" in out.stdout, out.stdout + out.stderr

    # 6) missing status AND missing pid -> nostatus_nopid
    pid.unlink()
    out = _run_bash_heredoc(body, str(rd), str(log), str(pid), str(st))
    assert "STATUS=nostatus_nopid" in out.stdout, out.stdout + out.stderr


def test_remote_eval_probe_argument_mapping_is_correct():
    text = EVAL_SCRIPT.read_text(encoding="utf-8")
    body = _extract_function_heredoc(text, "_remote_eval_probe", "RM")
    # The script calls `_remote_eval_probe run_dir eval_log eval_pid eval_status`,
    # so the heredoc must map: $1=RD, $2=LOG, $3=PID, $4=ST.
    assert 'RD="$1"; LOG="$2"; PID="$3"; ST="$4"' in body


# --- source-inspection tests ------------------------------------------------


def test_run_remote_eval_launches_detached_with_markers():
    text = EVAL_SCRIPT.read_text(encoding="utf-8")
    # Search the whole script: _extract_bash_function truncates on ${...} heredocs.
    assert "setsid nohup bash" in text
    assert "eval.pid" in text and "status" in text
    # idempotent guard: do not start a duplicate if a live PID marker exists
    assert 'kill -0 "$OPID"' in text
    assert "not starting duplicate" in text
    # durable remote log path under a run dir
    assert "wcg-eval-${RUN_NAME}.d/eval.log" in text


def test_run_remote_eval_has_monitor_tolerating_transient_ssh():
    text = EVAL_SCRIPT.read_text(encoding="utf-8")
    mon = _extract_bash_function(text, "monitor_remote_eval")
    # retries + backoff within the probe loop
    assert 'for attempt in $(seq 1 "$retries")' in mon
    assert 'sleep "$backoff"' in mon
    # transient probe failure must NOT kill the eval; it only logs + retries
    assert "monitor SSH probe failed (transient)" in mon
    # only proceeds after confirmed completion
    assert "STATUS=done*)" in mon
    assert "exited non-zero" in mon
    # timeout path preserves remote log/status
    assert "monitor timed out after" in mon


def test_monitor_proceeds_to_results_only_after_completion():
    text = EVAL_SCRIPT.read_text(encoding="utf-8")
    # main flow still calls find/download AFTER run_remote_eval returns
    body = _extract_bash_function(text, "run_remote_eval")
    assert "monitor_remote_eval" in body
    assert "find_remote_results_dir" in text
    assert "download_results" in text
    # find_remote_results_dir now reads the durable detached log
    assert "wcg-eval-${RUN_NAME}.d/eval.log" in text


def test_secrets_sourced_not_echoed():
    text = EVAL_SCRIPT.read_text(encoding="utf-8")
    assert "source secrets.env" in text
    assert "set -a" in text
    # never print secrets to any log
    assert 'echo "$PRIME_API_KEY' not in text
    assert 'echo "$HF_TOKEN' not in text
    assert "echo $PRIME_API_KEY" not in text
    assert "echo $HF_TOKEN" not in text


# --- behavioral tests (stateful fake SSH) ------------------------------------


def test_detached_eval_survives_transient_disconnect_and_completes(tmp_path: Path):
    repo = _make_temp_repo(tmp_path)
    result, record = _run_script(
        tmp_path,
        repo,
        modes={
            "WCG_EVAL_TIMEOUT_SECONDS": 30,
            "WCG_EVAL_POLL_INTERVAL": 1,
            "WCG_EVAL_MON_RETRIES": 2,
            "WCG_EVAL_MON_BACKOFF": 1,
            "WCG_TRANSIENT": 4,
            "WCG_DONE_AFTER_PROBES": 3,
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    # exactly one launch -> transient monitor failures did NOT start a duplicate
    launches = (record / "launch_count.log").read_text().split()
    assert launches.count("1") == 1, launches
    assert "dup" not in launches
    # results were downloaded
    assert (record / "terminated.log").exists()  # cleanup ran on success exit


def test_explicit_remote_nonzero_exit_propagates(tmp_path: Path):
    repo = _make_temp_repo(tmp_path)
    result, record = _run_script(
        tmp_path,
        repo,
        modes={
            "WCG_EVAL_TIMEOUT_SECONDS": 30,
            "WCG_EVAL_POLL_INTERVAL": 1,
            "WCG_EVAL_MON_RETRIES": 2,
            "WCG_EVAL_MON_BACKOFF": 1,
            "WCG_DONE_AFTER_PROBES": 2,
            "WCG_EVAL_EXIT_CODE": 3,
        },
    )
    assert result.returncode != 0, result.stdout + result.stderr
    # remote exit code preserved in the status marker
    status = list((record / "remote_root").rglob("status"))[0]
    assert "EXIT=3" in status.read_text()
    # cleanup still terminated the pod
    assert (record / "terminated.log").exists()


def test_missing_status_with_pid_fails_safe(tmp_path: Path):
    repo = _make_temp_repo(tmp_path)
    result, record = _run_script(
        tmp_path,
        repo,
        modes={
            "WCG_EVAL_TIMEOUT_SECONDS": 30,
            "WCG_EVAL_POLL_INTERVAL": 1,
            "WCG_EVAL_MON_RETRIES": 2,
            "WCG_EVAL_MON_BACKOFF": 1,
            "WCG_NO_STATUS": 1,
        },
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert (record / "terminated.log").exists()


def test_monitor_timeout_triggers_cleanup(tmp_path: Path):
    repo = _make_temp_repo(tmp_path)
    result, record = _run_script(
        tmp_path,
        repo,
        modes={
            "WCG_EVAL_TIMEOUT_SECONDS": 5,
            "WCG_EVAL_POLL_INTERVAL": 1,
            "WCG_EVAL_MON_RETRIES": 2,
            "WCG_EVAL_MON_BACKOFF": 1,
            "WCG_NEVER_DONE": 1,
        },
    )
    assert result.returncode != 0, result.stdout + result.stderr
    # pod terminated by cleanup on launcher exit
    assert (record / "terminated.log").exists()
    # remote log/status preserved (never flipped to done)
    status = list((record / "remote_root").rglob("status"))[0]
    assert "EXIT=" not in status.read_text()


def test_patch_failure_still_aborts_before_launch(tmp_path: Path):
    repo = _make_temp_repo(tmp_path)
    result, record = _run_script(
        tmp_path,
        repo,
        modes={"WCG_PATCH_FAIL": 1},
    )
    assert result.returncode != 0, result.stdout + result.stderr
    # eval never launched
    assert not (record / "launch_count.log").exists()


def test_hostile_model_value_cannot_break_out(tmp_path: Path):
    """A hostile MODEL (command substitution / quote breakout attempt) must not
    execute locally or be embedded unescaped in remote shell text. It is passed
    as a positional arg to a quoted heredoc, so it is treated as data only."""
    repo = _make_temp_repo(tmp_path)
    pwn = tmp_path / "PWNED_HOSTILE"
    hostile = "$(touch " + str(pwn) + ")"
    result, record = _run_script(
        tmp_path,
        repo,
        model=hostile,
        modes={
            "WCG_EVAL_TIMEOUT_SECONDS": 20,
            "WCG_EVAL_POLL_INTERVAL": 1,
            "WCG_EVAL_MON_RETRIES": 2,
            "WCG_EVAL_MON_BACKOFF": 1,
            "WCG_DONE_AFTER_PROBES": 1,
        },
    )
    # (1) local command substitution must NOT have fired
    assert not pwn.exists(), "hostile MODEL triggered local command substitution"
    # (2) the eval still ran to completion (hostile value is inert data)
    assert result.returncode == 0, result.stdout + result.stderr
    # (3) the generated remote shell text passes MODEL as a positional arg and
    #     never embeds the raw hostile value unescaped
    launch_stdin = (record / "launch_stdin.log").read_text()
    assert 'MODEL="$5"' in launch_stdin
    assert hostile not in launch_stdin


# --- find_remote_results_dir / download path logic (artifact-path blocker) ---


def test_find_remote_results_strips_outputs_prefix(tmp_path: Path):
    """The real function must return the results dir RELATIVE to the outputs
    root (strip exactly one leading `outputs/`), not `outputs/...`."""
    text = EVAL_SCRIPT.read_text(encoding="utf-8")
    body = _extract_function_heredoc(text, "find_remote_results_dir", "REMOTE")
    run_dir = tmp_path / "wcg-eval-x.d"
    run_dir.mkdir()
    (run_dir / "eval.log").write_text(
        "some preamble\nresults: outputs/pretrain-data-curator--abc/uuid-123\n"
    )
    out = _run_find_results(body, tmp_path, "x")
    assert out.returncode == 0, out.stdout + out.stderr
    rel = out.stdout.strip()
    assert rel == "pretrain-data-curator--abc/uuid-123"
    assert not rel.startswith("outputs/")


def test_find_remote_results_rejects_unsafe_paths(tmp_path: Path):
    cases = {
        "empty after strip": "results: outputs/\n",
        "absolute after strip": "results: outputs//abs/path\n",
        "parent traversal": "results: outputs/foo/../bar\n",
        "no expected prefix": "results: other/foo\n",
    }
    text = EVAL_SCRIPT.read_text(encoding="utf-8")
    body = _extract_function_heredoc(text, "find_remote_results_dir", "REMOTE")
    for name, content in cases.items():
        run_dir = tmp_path / f"wcg-eval-{name.replace(' ', '_')}.d"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "eval.log").write_text(content)
        out = _run_find_results(body, tmp_path, name.replace(" ", "_"))
        # unsafe -> fail safe (non-zero), nothing emitted to stdout
        assert out.returncode != 0, f"{name}: expected fail-safe, got {out.stdout!r}"
        assert out.stdout.strip() == "", f"{name}: must not emit an unsafe path"


def test_find_remote_results_fallback_returns_relative(tmp_path: Path):
    """Fallback (no results line in log) scans outputs/ and returns a path
    relative to the outputs root -- identical contract to the grep path."""
    text = EVAL_SCRIPT.read_text(encoding="utf-8")
    body = _extract_function_heredoc(text, "find_remote_results_dir", "REMOTE")
    run_dir = tmp_path / "wcg-eval-x.d"
    run_dir.mkdir()
    (run_dir / "eval.log").write_text("no results line here\n")
    outputs_root = (
        tmp_path
        / "webcurator-gym"
        / "environments"
        / "pretrain_data_curator"
        / "outputs"
    )
    run = outputs_root / "pretrain-data-curator--abc" / "run-uuid"
    run.mkdir(parents=True)
    (run / "results.jsonl").write_text(
        json.dumps({"is_completed": True, "rewards": {"reward": 0.42}})
    )
    out = _run_find_results(body, tmp_path, "x")
    assert out.returncode == 0, out.stdout + out.stderr
    rel = out.stdout.strip()
    assert rel == "pretrain-data-curator--abc/run-uuid"
    assert not rel.startswith("outputs/")


def test_download_constructs_single_outputs_and_lands_local(tmp_path: Path):
    """End-to-end: downloaded artifacts land in the requested local dir, and the
    rsync source contains exactly one `/outputs/` (no doubled prefix)."""
    repo = _make_temp_repo(tmp_path)
    result, record = _run_script(
        tmp_path,
        repo,
        modes={
            "WCG_EVAL_TIMEOUT_SECONDS": 30,
            "WCG_EVAL_POLL_INTERVAL": 1,
            "WCG_EVAL_MON_RETRIES": 2,
            "WCG_EVAL_MON_BACKOFF": 1,
            "WCG_DONE_AFTER_PROBES": 1,
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    local_dir = (
        repo
        / "environments"
        / "pretrain_data_curator"
        / "outputs"
        / "evals-400m"
        / "z-ai-glm-5.2-400M-300turn-codex"
    )
    assert (local_dir / "results.jsonl").exists()
    # the rsync SOURCE contains the relative results dir under exactly one
    # /outputs/ (no doubled prefix from caller + function)
    args = (record / "rsync_args.log").read_text()
    assert "outputs/2026-01-01/run0/" in args
    assert "/outputs/outputs/" not in args
    # the rsync SOURCE (remote) carries exactly one /outputs/ prefix
    src = "root@203.0.113.9:/root/webcurator-gym/environments/pretrain_data_curator/outputs/2026-01-01/run0/"
    assert src in args
    assert src.count("/outputs/") == 1
    assert "/outputs/outputs/" not in args


def test_invalid_download_preserves_artifacts_skips_site_and_cleans_pod(
    tmp_path: Path,
):
    """The original bad row must fail only after its artifacts are retained."""
    repo = _make_temp_repo(tmp_path)
    invalid = {
        "is_completed": True,
        "stop_condition": "agent_completed",
        "errors": [],
        "rewards": {"reward": 0.0},
        "metrics": {
            "finalized": 0.0,
            "manifest_missing": 1.0,
            "manifest_invalid": 0.0,
            "corpus_tokens": 0.0,
            "num_sources": 0.0,
            "train_flops": 0.0,
            "perf_loss": None,
            "trainer_error_msg": 0.0,
        },
    }
    result, record = _run_script(
        tmp_path,
        repo,
        skip_site=False,
        modes={
            "WCG_EVAL_TIMEOUT_SECONDS": 30,
            "WCG_EVAL_POLL_INTERVAL": 1,
            "WCG_EVAL_MON_RETRIES": 2,
            "WCG_EVAL_MON_BACKOFF": 1,
            "WCG_DONE_AFTER_PROBES": 1,
            "WCG_RESULTS_JSON": json.dumps(invalid),
        },
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert "semantic validation" in result.stderr

    local_dir = (
        repo
        / "environments"
        / "pretrain_data_curator"
        / "outputs"
        / "evals-400m"
        / "z-ai-glm-5.2-400M-300turn-codex"
    )
    assert json.loads((local_dir / "results.jsonl").read_text()) == invalid
    assert (local_dir / "downloaded-artifact.txt").read_text() == "preserve me\n"
    assert not (record / "site_rebuilt.log").exists()
    assert (record / "terminated.log").exists()


def test_valid_download_rebuilds_site_via_repo_root_builder(tmp_path: Path):
    """A results row that PASSES semantic validation must reach the site rebuild,
    exercising the fake builder at the new repo-root docs/ location (proving the
    launcher's "$ROOT/docs/build_site.py" resolves correctly end-to-end)."""
    repo = _make_temp_repo(tmp_path)
    valid = {
        "is_completed": True,
        "stop_condition": "agent_completed",
        "errors": [],
        "rewards": {"reward": 0.42},
        "metrics": {
            "finalized": 1.0,
            "manifest_missing": 0.0,
            "manifest_invalid": 0.0,
            "corpus_tokens": 400000000.0,
            "num_sources": 3.0,
            "train_flops": 1.2e18,
            "perf_loss": 3.1,
            "trainer_error_msg": 0.0,
        },
    }
    result, record = _run_script(
        tmp_path,
        repo,
        skip_site=False,
        modes={
            "WCG_EVAL_TIMEOUT_SECONDS": 30,
            "WCG_EVAL_POLL_INTERVAL": 1,
            "WCG_EVAL_MON_RETRIES": 2,
            "WCG_EVAL_MON_BACKOFF": 1,
            "WCG_DONE_AFTER_PROBES": 1,
            "WCG_RESULTS_JSON": json.dumps(valid),
        },
    )
    assert result.returncode == 0, result.stdout + result.stderr
    # validation passed AND the repo-root builder was actually invoked
    assert (record / "site_rebuilt.log").read_text().strip() == "yes"
    assert (record / "terminated.log").exists()
