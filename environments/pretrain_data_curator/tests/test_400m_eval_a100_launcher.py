"""Deterministic tests for 400M A100 launcher root-provider safeguards.

Coverage is split between two styles:

* source-inspection tests that read the actual launcher script and assert the
  safeguards (provider exclusion, root-user guard ordering, permanent-rsync
  regex, chmod 0600, A100/Docker/cleanup preservation) are present; and
* real-script-path regressions that copy the launcher into an isolated temp
  repo and run it end-to-end behind a fake ``PATH`` (no real ``prime``/``ssh``/
  ``rsync`` calls, no pod created, and the live repo-root secrets.env is never
  touched thanks to the temp repo's own secrets.env).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import stat
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
EVAL_SCRIPT = REPO_ROOT / "scripts" / "run_400m_eval_a100.sh"

# --- canned prime availability / status payloads for real-script regressions --

_STATUS_ROOT = (
    '{"status":"ACTIVE","installation_status":"FINISHED",'
    '"ip":"203.0.113.9","ssh":"ssh root@203.0.113.9",'
    '"port_mappings":[{"internal":"22","external":"22"}]}'
)
_STATUS_NONROOT = (
    '{"status":"ACTIVE","installation_status":"FINISHED",'
    '"ip":"203.0.113.9","ssh":"ssh ubuntu@203.0.113.9",'
    '"port_mappings":[{"internal":"22","external":"22"}]}'
)
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


def _make_temp_repo(tmp_path: Path) -> Path:
    """Isolate tests from the live repo: copy the launcher into a temp repo with
    a throwaway secrets.env so the actual script runs without ever touching the
    real repo-root secrets.env. environments/ is symlinked so configs resolve.
    """
    repo = tmp_path / "repo"
    (repo / "scripts").mkdir(parents=True)
    shutil.copy(EVAL_SCRIPT, repo / "scripts" / EVAL_SCRIPT.name)
    (repo / "environments").symlink_to(REPO_ROOT / "environments", target_is_directory=True)
    (repo / "secrets.env").write_text("HF_TOKEN=dummy\nPRIME_API_KEY=dummy\n")
    return repo


def _write_script(path: Path, text: str) -> None:
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _build_fakes(record_dir: Path) -> Path:
    """Create a fake bin dir recording the script's external calls."""
    bin_dir = record_dir / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)

    _write_script(
        bin_dir / "prime",
        """#!/usr/bin/env bash
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
      printf '%s\\n' "$cid" >> "${WCG_RECORD_DIR:?}/picked.txt"
      echo "Successfully created pod POD123"
      exit 0
    elif [ "$sub" = "status" ]; then
      cat "${WCG_POD_STATUS_JSON:?}"
      exit 0
    elif [ "$sub" = "terminate" ]; then
      exit 0
    fi
    ;;
esac
exit 0
""",
    )

    _write_script(
        bin_dir / "ssh",
        """#!/usr/bin/env bash
args="$*"
case "$args" in
  *"chmod 0600"*"secrets.env"*) printf 'chmod\\n' >> "${WCG_RECORD_DIR:?}/chmod.log" ;;
esac
stdin="$(cat)"
if [ "${WCG_PATCH_FAIL:-0}" = "1" ] && printf '%s' "$stdin" | grep -q "PRIME_INFERENCE_DISABLE_NAMESPACE_TOOLS"; then
  echo "forced patch failure" >&2
  exit 1
fi
if printf '%s' "$stdin" | grep -q "uv run eval -m"; then
  printf 'eval_ran\n' >> "${WCG_RECORD_DIR:?}/eval_ran.log"
fi
exit 0
""",
    )

    _write_script(
        bin_dir / "scp",
        """#!/usr/bin/env bash
printf 'scp %s\\n' "$*" >> "${WCG_RECORD_DIR:?}/scp.log"
exit 0
""",
    )

    _write_script(
        bin_dir / "rsync",
        """#!/usr/bin/env bash
printf 'rsync\\n' >> "${WCG_RECORD_DIR:?}/rsync_count.txt"
mode="${WCG_RSYNC_MODE:-ok}"
if [ "$mode" = "permanent" ]; then
  echo "rsync: send_files failed to open \\"/root/webcurator-gym/secrets.env\\": Permission denied (13)" >&2
  echo "rsync error: some files/attrs were not transferred (see previous errors) (code 23)" >&2
  exit 23
elif [ "$mode" = "mkdir_nosuchfile" ]; then
  echo 'rsync: mkdir "/root/webcurator-gym" failed: No such file or directory (2)' >&2
  exit 23
fi
exit 0
""",
    )

    _write_script(bin_dir / "ssh-keygen", """#!/usr/bin/env bash\nexit 0\n""")
    _write_script(bin_dir / "sleep", """#!/usr/bin/env bash\nexit 0\n""")
    return bin_dir


def _run_script(
    tmp_path: Path,
    repo: Path,
    *,
    avail,
    status,
    rsync_mode="ok",
    model="z-ai/glm-5.2",
):
    """Run the actual launcher script inside ``repo`` via a fake PATH."""
    record_dir = tmp_path / "record"
    record_dir.mkdir(parents=True, exist_ok=True)
    bin_dir = _build_fakes(record_dir)

    avail_json = tmp_path / "avail.json"
    avail_json.write_text(json.dumps(avail))

    home = tmp_path / "home"
    (home / ".ssh").mkdir(parents=True)
    (home / ".ssh" / "id_rsa").write_text("dummy")

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["HOME"] = str(home)
    env["WCG_AVAIL_JSON"] = str(avail_json)
    status_json = tmp_path / "status.json"
    status_json.write_text(status)
    env["WCG_POD_STATUS_JSON"] = str(status_json)
    env["WCG_RSYNC_MODE"] = rsync_mode
    env["WCG_RECORD_DIR"] = str(record_dir)

    script = repo / "scripts" / EVAL_SCRIPT.name
    result = subprocess.run(
        [str(script), "--model", model, "--skip-site"],
        env=env,
        cwd=str(repo),
        capture_output=True,
        text=True,
        timeout=120,
    )
    return result, record_dir


def _extract_bash_function(script: str, name: str) -> str:
    header = f"{name}()"
    start = script.find(header)
    assert start != -1, f"missing function {name}()"
    brace = script.find("{", start)
    assert brace != -1, f"missing opening brace for {name}()"
    depth = 0
    for idx in range(brace, len(script)):
        ch = script[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return script[brace + 1 : idx]
    raise AssertionError(f"unclosed function body for {name}()")


def _extract_heredoc_python(script: str, func_name: str) -> str:
    body = _extract_bash_function(script, func_name)
    marker = "<<'PY'"
    start = body.find(marker)
    assert start != -1, f"missing python heredoc in {func_name}"
    py_start = body.find("\n", start) + 1
    end = body.find("\nPY", py_start)
    assert end != -1, f"missing PY terminator in {func_name}"
    return body[py_start:end]


def _select_cloud_id(func_name: str, resources: list, excluded: str = "") -> str:
    import io
    import sys as _sys

    script = EVAL_SCRIPT.read_text(encoding="utf-8")
    code = _extract_heredoc_python(script, func_name)
    buf = io.StringIO()
    ns = {"__name__": "__main__", "sys": _sys}
    _sys.argv = [func_name, json.dumps({"gpu_resources": resources}), excluded]
    old = _sys.stdout
    _sys.stdout = buf
    try:
        exec(compile(code, func_name, "exec"), ns)
    except SystemExit as exc:
        _sys.stdout = old
        if exc.code:
            raise AssertionError(
                f"{func_name} exited {exc.code}: {buf.getvalue().strip()}"
            ) from exc
        return buf.getvalue().strip()
    _sys.stdout = old
    return buf.getvalue().strip()


def _gpu_resource(cloud_id, provider, price="1.00", spot=False):
    return {
        "cloud_id": cloud_id,
        "provider": provider,
        "stock_status": "available",
        "price_per_hour": price,
        "is_spot": spot,
    }


def _cpu_resource(cloud_id, provider, mem="64", vcpus="16", price="1.00"):
    return {
        "cloud_id": cloud_id,
        "provider": provider,
        "stock_status": "available",
        "memory_gb": mem,
        "vcpus": vcpus,
        "price_per_hour": price,
    }


def test_gpu_filter_cheapest_massedcompute_vs_datacrunch_picks_datacrunch():
    assert EVAL_SCRIPT.is_file()
    resources = [
        _gpu_resource("mc-cheapest", "MassedCompute", price="0.10"),
        _gpu_resource("mc-low", "massedcompute", price="0.11"),
        _gpu_resource("cr-cheap", "crusoecloud", price="0.20"),
        _gpu_resource("dc-root", "DataCrunch", price="0.99"),
    ]
    assert _select_cloud_id("pick_cloud_id", resources) == "dc-root"


def test_gpu_filter_only_excluded_providers_yields_no_offer():
    with pytest.raises(AssertionError):
        _select_cloud_id(
            "pick_cloud_id",
            [
                _gpu_resource("mc-only", "massedcompute"),
                _gpu_resource("cr-only", "CrusoeCloud"),
            ],
        )


def test_gpu_filter_excludes_massedcompute_and_crusoe_case_insensitive():
    resources = [
        _gpu_resource("mc-cheap", "MassedCompute", price="0.40"),
        _gpu_resource("cr-cheap", "CRUSOECLOUD", price="0.50"),
        _gpu_resource("dc-root", "datacrunch", price="0.99"),
    ]
    assert _select_cloud_id("pick_cloud_id", resources) == "dc-root"


def test_cpu_filter_excludes_massedcompute_and_crusoe_case_insensitive():
    resources = [
        _cpu_resource("mc-cpu", "MassedCompute", price="0.40"),
        _cpu_resource("cr-cpu", "crusoecloud", price="0.50"),
        _cpu_resource("dc-cpu", "DataCrunch", price="0.90"),
    ]
    assert _select_cloud_id("pick_cpu_cloud_id", resources) == "dc-cpu"
    with pytest.raises(AssertionError):
        _select_cloud_id(
            "pick_cpu_cloud_id", [_cpu_resource("mc-only", "massedcompute")]
        )


def test_non_root_ssh_user_fail_fast_before_sync(tmp_path: Path):
    text = EVAL_SCRIPT.read_text(encoding="utf-8")
    auth_at = text.find("wait_for_ssh_auth\n")
    root_check_at = text.find('if [[ "$SSH_USER" != "root" ]]; then')
    rsync_at = text.find("\nremote_rsync\n")
    assert 0 <= auth_at < root_check_at < rsync_at, "root check misplaced"
    assert "FATAL: pod authenticated as non-root user" in text
    assert "Unsupported provider image" in text
    assert "trap cleanup EXIT" in text
    cleanup = _extract_bash_function(text, "cleanup")
    assert "prime pods terminate" in cleanup

    probe = tmp_path / "non_root_probe.sh"
    probe.write_text(
        r"""#!/usr/bin/env bash
set -euo pipefail
SSH_USER="ubuntu"
POD_ID="pod-xyz"
KEEP_POD=0
die() { echo "FATAL: $*" >&2; exit 3; }
log() { :; }
cleanup() {
  local code=$?
  if [[ -n "$POD_ID" && "$KEEP_POD" -eq 0 ]]; then
    echo "terminated=$POD_ID"
  fi
  exit "$code"
}
trap cleanup EXIT
if [[ "$SSH_USER" != "root" ]]; then
  die "FATAL: pod authenticated as non-root user '$SSH_USER'; this launcher requires root SSH access (rootful Docker / root home). Unsupported provider image -- terminating pod."
fi
echo "SHOULD NOT REACH SYNC"
""",
        encoding="utf-8",
    )
    probe.chmod(0o755)
    proc = subprocess.run(
        ["bash", str(probe)], check=False, capture_output=True, text=True
    )
    assert proc.returncode == 3, proc.stdout + proc.stderr
    assert "SHOULD NOT REACH SYNC" not in proc.stdout
    combined = proc.stdout + proc.stderr
    assert "FATAL: pod authenticated as non-root user 'ubuntu'" in combined
    assert "terminated=pod-xyz" in combined


def test_rsync_permanent_failure_fails_fast_no_retry():
    text = EVAL_SCRIPT.read_text(encoding="utf-8")
    rsync_fn = _extract_bash_function(text, "remote_rsync")
    m = re.search(r"PERMANENT_RE='([^']+)'", rsync_fn)
    assert m, "could not find PERMANENT_RE pattern in remote_rsync"
    pattern = re.compile(m.group(1), re.IGNORECASE)

    permanent = [
        'rsync: failed to set times on "/root/webcurator-gym": Permission denied (13)',
        'rsync: link_stat "/root/webcurator-gym" failed: No such file or directory (2)',
        'rsync: mkdir "/root/webcurator-gym/sub" failed: Permission denied (13)',
        'rsync: write failed on "/root/webcurator-gym/x": Read-only file system (30)',
        'rsync: mkdir "/root/webcurator-gym" failed: No such file or directory (2)',
        "rsync: cannot stat destination /root/webcurator-gym: Permission denied",
        "rsync: read-only filesystem while writing",
    ]
    for line in permanent:
        assert pattern.search(line), f"permanent pattern should match: {line}"

    transient = [
        "rsync: connection unexpectedly closed (0 bytes received so far) [Receiver]",
        "rsync error: timeout in data send/receive (code 30)",
        "rsync: failed to connect to host: Connection timed out (110)",
        "rsync: recv_generator: failed to stat due to Input/output error",
        "ssh: connect to host 1.2.3.4 port 22: Connection refused",
    ]
    for line in transient:
        assert not pattern.search(line), f"transient must NOT match: {line}"

    perm_die_at = rsync_fn.find('die "rsync permanent failure')
    sleep_at = rsync_fn.find("sleep 20")
    assert 0 <= perm_die_at < sleep_at, "permanent failure must die before retry"
    assert rsync_fn.count("sleep 20") >= 1, "transient retry/backoff preserved"
    assert "/root/webcurator-gym/" in rsync_fn


def test_rsync_permanent_mkdir_nosuchfile_fails_fast_real_script(tmp_path: Path):
    """Real-script-path regression: a missing-destination path error
    (`rsync: mkdir "/root/webcurator-gym" failed: No such file or directory (2)`)
    must fail fast -- exactly one rsync attempt, no retry sleep -- even though it
    carries no 'permission denied' text. Runs the actual launcher script.
    """
    repo = _make_temp_repo(tmp_path)
    result, record = _run_script(
        tmp_path,
        repo,
        avail=_AVAIL_MIXED,
        status=_STATUS_ROOT,
        rsync_mode="mkdir_nosuchfile",
    )
    count = record / "rsync_count.txt"
    assert count.exists(), f"rsync was never invoked; stderr={result.stderr}"
    # Permanent path failure must not be retried (5x). Exactly one attempt.
    assert len(count.read_text().split()) == 1, "permanent failure must not be retried"
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "rsync permanent failure" in combined


def test_upload_secrets_enforces_chmod_0600():
    text = EVAL_SCRIPT.read_text(encoding="utf-8")
    fn = _extract_bash_function(text, "upload_secrets")
    assert "chmod 0600 /root/webcurator-gym/secrets.env" in fn
    assert "Failed to chmod 0600 the remote secrets file" in fn
    assert re.search(r"if ! remote chmod 0600", fn)
    assert "scp -i" in fn
    assert ">/dev/null" in fn
    assert 'echo "$tmp"' not in fn
    assert 'cat "$tmp"' not in fn


def _run_patch_heredoc(harness_path: Path, *, verifiers_file: Path):
    """Exec the Python heredoc of remote_patch_codex_for_prime_inference against
    a fake harness file by monkeypatching verifiers.__file__."""
    import io
    import sys as _sys

    import verifiers as _v

    text = EVAL_SCRIPT.read_text(encoding="utf-8")
    code = _extract_heredoc_python(text, "remote_patch_codex_for_prime_inference")
    buf = io.StringIO()
    ns = {"__name__": "__main__"}
    old_file = _v.__file__
    old_out = _sys.stdout
    _sys.stdout = buf
    _v.__file__ = str(verifiers_file)
    try:
        try:
            exec(compile(code, "patch", "exec"), ns)
        except SystemExit as exc:
            _sys.stdout = old_out
            _v.__file__ = old_file
            msg = exc.args[0] if exc.args else ""
            return buf.getvalue() + str(msg), exc.code
    finally:
        _sys.stdout = old_out
        _v.__file__ = old_file
    return buf.getvalue(), 0


def test_codex_patch_function_present_and_targets_both_namespace_tools():
    text = EVAL_SCRIPT.read_text(encoding="utf-8")
    body = _extract_bash_function(text, "remote_patch_codex_for_prime_inference")
    assert "PRIME_INFERENCE_DISABLE_NAMESPACE_TOOLS" in body
    # Both namespace features must be disabled (Prime rejects every type=namespace tool).
    assert r'\"apps\"' in body
    assert r'\"multi_agent\"' in body
    # Function tools / web_search / shell / HF CLI skill access must NOT be disabled.
    for keep in ("function", "web_search", "shell"):
        assert f'"--disable {keep}"' not in body
        assert f'"{keep}"' not in body
    assert "raise SystemExit" in body
    assert "codex harness patch needle missing" in body
    assert "already patched" in body
    # idempotent loop over both features (bash-escaped quotes in the script)
    assert r'for _feature in (\"apps\", \"multi_agent\")' in body


def test_codex_patch_invoked_before_eval_in_run_remote_eval():
    text = EVAL_SCRIPT.read_text(encoding="utf-8")
    run_fn = _extract_bash_function(text, "run_remote_eval")
    call_at = run_fn.find("remote_patch_codex_for_prime_inference")
    eval_at = run_fn.find("uv run eval")
    assert 0 <= call_at < eval_at, "patch must run before uv run eval"
    # patch call sits directly in run_remote_eval, not behind any conditional
    assert run_fn.find("uv run eval") != -1


def test_codex_patch_heredoc_unit_inserts_and_idempotent(tmp_path: Path):
    vf = tmp_path / "verifiers"
    code_dir = vf / "v1" / "harnesses" / "codex"
    code_dir.mkdir(parents=True)
    harness = code_dir / "harness.py"
    needle_block = (
        "        tool_config = [\n"
        "            arg\n"
        "            for tool in self.config.disabled_tools or []\n"
        "            for arg in (\"--disable\", tool)\n"
        "        ]\n"
    )
    harness.write_text('class C:\n    def f(self):\n' + needle_block)

    out, code = _run_patch_heredoc(harness, verifiers_file=vf / "__init__.py")
    assert code == 0, out
    assert "patched:" in out
    patched = harness.read_text()
    assert '_feature in ("apps", "multi_agent")' in patched
    assert "if _feature not in _disabled:" in patched
    assert "_disabled.append(_feature)" in patched
    # original single-line comprehension replaced by the guarded version
    assert 'for tool in self.config.disabled_tools or []' not in patched

    first = harness.read_text()
    out2, code2 = _run_patch_heredoc(harness, verifiers_file=vf / "__init__.py")
    assert code2 == 0, out2
    assert "already patched:" in out2
    assert harness.read_text() == first  # idempotent: no second insertion


def test_codex_patch_generated_launch_args_disable_both_keep_function_tools(tmp_path: Path):
    """Regression: after patching, the harness's tool_config (what becomes the
    `codex exec --disable ...` argv) must disable BOTH namespace features and
    must NOT disable function tools / web_search / shell / HF CLI skill access,
    even when the harness config sets no disabled_tools.
    """
    import types

    vf = tmp_path / "verifiers"
    code_dir = vf / "v1" / "harnesses" / "codex"
    code_dir.mkdir(parents=True)
    harness = code_dir / "harness.py"
    needle_block = (
        "        tool_config = [\n"
        "            arg\n"
        "            for tool in self.config.disabled_tools or []\n"
        "            for arg in (\"--disable\", tool)\n"
        "        ]\n"
    )
    harness.write_text('class C:\n    def f(self):\n' + needle_block)

    out, code = _run_patch_heredoc(harness, verifiers_file=vf / "__init__.py")
    assert code == 0, out
    patched = harness.read_text()

    # Extract the patched block (marker comment line through the tool_config list end).
    lines = patched.splitlines()
    start = next(i for i, ln in enumerate(lines) if "PRIME_INFERENCE_DISABLE_NAMESPACE_TOOLS" in ln)
    end = next(
        j for j in range(start, len(lines))
        if lines[j].rstrip() == "        ]" and j > start
    )
    block = textwrap.dedent("\n".join(lines[start : end + 1]))

    ns = {"self": types.SimpleNamespace(
        config=types.SimpleNamespace(disabled_tools=None)
    )}
    exec(compile(block, "patched_block", "exec"), ns)
    tool_config = ns["tool_config"]
    args = list(tool_config)  # flat: ["--disable", "apps", "--disable", "multi_agent"]

    assert args == ["--disable", "apps", "--disable", "multi_agent"], args
    # no namespace-producing default remains; function/web_search/shell intact
    assert "--disable" in args
    for preserved in ("function", "web_search", "shell"):
        assert preserved not in args


def test_codex_patch_heredoc_fails_fast_on_missing_needle(tmp_path: Path):
    vf = tmp_path / "verifiers"
    code_dir = vf / "v1" / "harnesses" / "codex"
    code_dir.mkdir(parents=True)
    harness = code_dir / "harness.py"
    harness.write_text('class C:\n    def f(self):\n        pass\n')

    out, code = _run_patch_heredoc(harness, verifiers_file=vf / "__init__.py")
    assert code, "must exit non-zero on missing needle"
    assert "patch needle missing" in out


def test_codex_patch_fails_fast_real_script_no_eval(tmp_path: Path):
    """Real-script-path regression: when the remote Codex harness patch fails
    (forced via WCG_PATCH_FAIL), run_remote_eval must die (set -e) before the
    `uv run eval` heredoc is ever sent -- eval_ran.log must stay absent.
    """
    repo = _make_temp_repo(tmp_path)
    # Run with the patch forced to fail (WCG_PATCH_FAIL=1 makes the fake ssh
    # exit 1 when stdin contains the patch marker).
    record_dir = tmp_path / "record2"
    record_dir.mkdir(parents=True, exist_ok=True)
    bin_dir = _build_fakes(record_dir)
    avail_json = tmp_path / "avail.json"
    avail_json.write_text(json.dumps(_AVAIL_MIXED))
    home = tmp_path / "home2"
    (home / ".ssh").mkdir(parents=True)
    (home / ".ssh" / "id_rsa").write_text("dummy")
    script_env = dict(os.environ)
    script_env["PATH"] = f"{bin_dir}{os.pathsep}{script_env['PATH']}"
    script_env["HOME"] = str(home)
    script_env["WCG_AVAIL_JSON"] = str(avail_json)
    status_json = tmp_path / "status2.json"
    status_json.write_text(_STATUS_ROOT)
    script_env["WCG_POD_STATUS_JSON"] = str(status_json)
    script_env["WCG_RSYNC_MODE"] = "ok"
    script_env["WCG_RECORD_DIR"] = str(record_dir)
    script_env["WCG_PATCH_FAIL"] = "1"
    proc = subprocess.run(
        [str(repo / "scripts" / EVAL_SCRIPT.name), "--model", "z-ai/glm-5.2", "--skip-site"],
        env=script_env,
        cwd=str(repo),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert not (record_dir / "eval_ran.log").exists(), "eval must not run if patch fails"


def test_preserves_a100_80gb_docker_artifacts_and_cleanup():
    text = EVAL_SCRIPT.read_text(encoding="utf-8")
    assert 'GPU_TYPE="A100_80GB"' in text
    fallback = _extract_bash_function(text, "pick_cloud_id_with_fallback")
    assert "A100_80GB" in fallback
    assert "A100_40GB" in fallback

    gpu = _extract_bash_function(text, "remote_provision_gpu")
    assert "docker" in gpu
    assert "webcurator-runtime" in gpu or "Dockerfile.runtime" in gpu

    assert "download_results" in text
    assert "trap cleanup EXIT" in text
    cleanup = _extract_bash_function(text, "cleanup")
    assert "prime pods terminate" in cleanup
    assert 'exit "$code"' in cleanup


# --- optional live Responses-API A/B (no GPU, no secret printing) ---------------
# Opt-in only: set WCG_LIVE_AB=1 and ensure PRIME_API_KEY (or ~/.prime/config.json)
# is available AND the Prime Responses endpoint is reachable (e.g. via prime_tunnel).
# Skips otherwise. Never prints the API key. The patch exists because Codex's
# default `multi_agent` (Responses-API `type=namespace` tool `multi_agent_v1`)
# makes Prime Inference return HTTP 400 invalid_request on Codex/Responses models;
# a normal `type=function` tool must still be accepted.

def _live_responses_key() -> str | None:
    if os.environ.get("PRIME_API_KEY"):
        return os.environ["PRIME_API_KEY"]
    cfg = Path.home() / ".prime" / "config.json"
    if cfg.is_file():
        try:
            return json.loads(cfg.read_text()).get("api_key")
        except Exception:
            return None
    return None


def _live_responses_call(base: str, key: str, model: str, tools: list) -> tuple[int, str]:
    import urllib.error
    import urllib.request

    body = json.dumps(
        {"model": model, "input": "ping", "tools": tools, "stream": False}
    ).encode()
    req = urllib.request.Request(
        base,
        data=body,
        method="POST",
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        resp = urllib.request.urlopen(req, timeout=60)
        return resp.status, resp.read().decode("utf-8", "replace")[:400]
    except urllib.error.HTTPError as exc:  # noqa: BLE001
        return exc.code, exc.read().decode("utf-8", "replace")[:600]
    except Exception as exc:  # noqa: BLE001 - connection failures -> skip
        return -1, f"{type(exc).__name__}: {exc}"


def test_live_codex_responses_namespace_tools_reproduce_400():
    if os.environ.get("WCG_LIVE_AB") != "1":
        pytest.skip("opt-in: set WCG_LIVE_AB=1 with reachable Prime Responses endpoint")
    key = _live_responses_key()
    if not key:
        pytest.skip("no PRIME_API_KEY available")
    base = os.environ.get(
        "WCG_RESPONSES_URL", "https://api.primeintellect.ai/v1/responses"
    )
    model = os.environ.get("WCG_LIVE_MODEL", "z-ai/glm-5.2")
    func_tool = {
        "type": "function",
        "name": "get_weather",
        "description": "x",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    }
    # Both Codex `type=namespace` features Prime rejects.
    ns_tools = [
        {"type": "namespace", "name": "apps"},
        {"type": "namespace", "name": "multi_agent_v1"},
    ]

    func_status, func_body = _live_responses_call(base, key, model, [func_tool])
    if func_status == -1:
        pytest.skip(f"Responses endpoint unreachable: {func_body}")
    # A normal function tool must be accepted (the patch preserves function tools).
    assert func_status != 400, f"function tool unexpectedly 400: {func_body}"

    for ns_tool in ns_tools:
        ns_status, ns_body = _live_responses_call(base, key, model, [ns_tool])
        assert ns_status == 400, (
            f"namespace {ns_tool['name']} expected 400, got {ns_status}: {ns_body}"
        )
        assert "invalid" in ns_body.lower(), (
            f"expected invalid_request 400 for {ns_tool['name']}: {ns_body}"
        )

    # With both namespace features disabled (the patch's argv), the same request
    # must no longer 400 -- only function/web_search tools remain.
    disabled_status, disabled_body = _live_responses_call(
        base, key, model, [func_tool],
    )
    assert disabled_status != 400, f"disabled-namespace request 400: {disabled_body}"
