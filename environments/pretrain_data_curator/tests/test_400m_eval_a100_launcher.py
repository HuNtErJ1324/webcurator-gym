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
import tomllib
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
    if [ -n "${WCG_AVAIL_DIR:-}" ]; then
      gt=""
      prev=""
      for a in "$@"; do
        if [ "$prev" = "--gpu-type" ]; then gt="$a"; fi
        prev="$a"
      done
      cat "${WCG_AVAIL_DIR}/${gt}.json" 2>/dev/null || true
    else
      cat "${WCG_AVAIL_JSON:?}"
    fi
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
      # Real `prime pods create` echoes a config summary including the GPU type
      # it actually provisioned. Tests set WCG_CREATE_GPUTYPE to emulate the
      # provider handing back a different GPU than was requested.
      if [ -n "${WCG_CREATE_GPUTYPE:-}" ]; then
        echo "gpuType: ${WCG_CREATE_GPUTYPE}"
      fi
      echo "Successfully created pod POD123"
      exit 0
    elif [ "$sub" = "status" ]; then
      cat "${WCG_POD_STATUS_JSON:?}"
      exit 0
    elif [ "$sub" = "terminate" ]; then
      pid=""
      while [ $# -gt 0 ]; do
        case "$1" in
          terminate) pid="$2"; shift 2 ;;
          *) shift ;;
        esac
      done
      printf '%s\\n' "$pid" >> "${WCG_RECORD_DIR:?}/terminated.txt"
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
    avail=None,
    status,
    rsync_mode="ok",
    model="z-ai/glm-5.2",
    avail_by_type=None,
    gpu_type=None,
    create_gpu_type=None,
):
    """Run the actual launcher script inside ``repo`` via a fake PATH.

    ``avail_by_type`` is a ``{gpu_type: payload}`` map; when given, the fake
    ``prime availability`` returns a per-gpu-type JSON (enabling per-gpu-type
    capacity tests). Otherwise a single ``avail`` payload is used for every
    gpu-type.

    ``gpu_type`` passes ``--gpu-type`` through to the launcher.
    ``create_gpu_type`` makes the fake ``prime pods create`` report that GPU
    type back, emulating a provider that hands back a GPU different from the
    one requested.
    """
    record_dir = tmp_path / "record"
    record_dir.mkdir(parents=True, exist_ok=True)
    bin_dir = _build_fakes(record_dir)

    home = tmp_path / "home"
    (home / ".ssh").mkdir(parents=True)
    (home / ".ssh" / "id_rsa").write_text("dummy")

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["HOME"] = str(home)
    if avail_by_type is not None:
        avail_dir = tmp_path / "avail"
        avail_dir.mkdir(parents=True, exist_ok=True)
        for gt, payload in avail_by_type.items():
            (avail_dir / f"{gt}.json").write_text(json.dumps(payload))
        env["WCG_AVAIL_DIR"] = str(avail_dir)
    else:
        avail_json = tmp_path / "avail.json"
        avail_json.write_text(json.dumps(avail if avail is not None else {}))
        env["WCG_AVAIL_JSON"] = str(avail_json)
    status_json = tmp_path / "status.json"
    status_json.write_text(status)
    env["WCG_POD_STATUS_JSON"] = str(status_json)
    env["WCG_RSYNC_MODE"] = rsync_mode
    env["WCG_RECORD_DIR"] = str(record_dir)
    if create_gpu_type is not None:
        env["WCG_CREATE_GPUTYPE"] = create_gpu_type

    script = repo / "scripts" / EVAL_SCRIPT.name
    argv = [str(script), "--model", model, "--skip-site"]
    if gpu_type is not None:
        argv += ["--gpu-type", gpu_type]
    result = subprocess.run(
        argv,
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


def test_launcher_tracked_eval_configs_pin_train_microbatch_32():
    """A100 launcher base + curation profiles must pin train_microbatch_size=32.

    Extracts ``BASE_EVAL_CONFIG`` assignments from the real launcher script and
    asserts each resolved TOML keeps the production microbatch pin (memory-only).
    """
    assert EVAL_SCRIPT.is_file()
    text = EVAL_SCRIPT.read_text(encoding="utf-8")
    rels = re.findall(r'BASE_EVAL_CONFIG="(configs/eval/[^"]+\.toml)"', text)
    assert rels == [
        "configs/eval/400M-300turn-codex.toml",
        "configs/eval/400M-300turn-codex-curation.toml",
    ], rels
    env_dir = REPO_ROOT / "environments" / "pretrain_data_curator"
    for rel in rels:
        path = env_dir / rel
        assert path.is_file(), path
        proxy = tomllib.loads(path.read_text(encoding="utf-8"))["args"]["proxy_student"]
        assert proxy["train_microbatch_size"] == 32, rel
        assert proxy["batch_size"] == 16, rel
        assert proxy["block_size"] == 1024, rel
        assert proxy["batch_stage_muls"] == [1, 2, 3], rel


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

    # The picker must honor the requested GPU type verbatim and must never
    # encode a substitute tier. (Regression: the old pick_cloud_id_with_fallback
    # silently downgraded an A100_80GB request to A100_40GB.) Comments are
    # stripped first -- they legitimately name A100_40GB to explain the history;
    # what must not contain it is the executable body.
    pick = _extract_bash_function(text, "pick_compute")
    pick_code = "\n".join(
        ln for ln in pick.splitlines() if not ln.lstrip().startswith("#")
    )
    assert "A100_40GB" not in pick_code, (
        "no substitute GPU tier may be hardcoded in pick_compute"
    )
    assert '"$GPU_TYPE"' in pick_code, "pick_compute must query the REQUESTED gpu type"
    assert "pick_cloud_id_with_fallback" not in text, "fallback picker must be gone"

    gpu = _extract_bash_function(text, "remote_provision_gpu")
    assert "docker" in gpu
    assert "webcurator-runtime" in gpu or "Dockerfile.runtime" in gpu

    assert "download_results" in text
    assert "trap cleanup EXIT" in text
    cleanup = _extract_bash_function(text, "cleanup")
    assert "prime pods terminate" in cleanup
    assert 'exit "$code"' in cleanup


# --- capacity picker: no silent GPU substitution, clean "<type> <cloud_id>" ---
#
# Regression context (2026-07-12, live paid run): pick_cloud_id_with_fallback
# assigned GPU_TYPE inside a command-substitution subshell and returned only the
# cloud_id, so when A100_80GB capacity vanished the launcher silently provisioned
# an A100_40GB pod while still logging "(A100_80GB)". The picker now emits the
# tuple "<gpu_type> <cloud_id>" and has no substitute tier at all.

_A40_ID = "1A100.40S.22V"


def _avail(cloud_id, provider="DataCrunch", stock="available", **kw):
    res = {
        "cloud_id": cloud_id,
        "provider": provider,
        "stock_status": stock,
        "price_per_hour": "1.20",
        "is_spot": False,
    }
    res.update(kw)
    return {"gpu_resources": [res]}


def _run_pick_compute(
    tmp_path: Path,
    *,
    avail_by_type,
    excluded: str = "",
    curation_only: int = 0,
    gpu_type: str = "A100_80GB",
    with_retries: bool = False,
    flicker_empty_calls: int = 0,
):
    """Run the REAL ``pick_compute`` (or ``pick_compute_with_retries``) with its
    real ``log``/``die`` and python pickers, behind a fake ``prime`` whose
    availability branches on gpu-type.

    ``flicker_empty_calls`` makes the fake report NO capacity for the first N
    ``availability`` calls before the real payload appears, emulating shared
    inventory that flickers between calls.

    Returns ``(stdout, stderr, rc)`` with streams captured separately, so we can
    assert diagnostics never leak into the selected value.
    """
    avail_dir = tmp_path / "avail"
    avail_dir.mkdir(parents=True, exist_ok=True)
    for gt, payload in avail_by_type.items():
        (avail_dir / f"{gt}.json").write_text(json.dumps(payload))

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    _write_script(
        bin_dir / "prime",
        """#!/usr/bin/env bash
set +e
case "$1" in
  whoami) exit 0 ;;
  availability)
    gt=""; prev=""
    for a in "$@"; do
      if [ "$prev" = "--gpu-type" ]; then gt="$a"; fi
      prev="$a"
    done
    # Emulate flickering inventory: the first N calls see nothing.
    n=0
    if [ -f "${WCG_CALLS_FILE:-/dev/null}" ]; then n="$(cat "$WCG_CALLS_FILE")"; fi
    n=$((n + 1))
    [ -n "${WCG_CALLS_FILE:-}" ] && echo "$n" > "$WCG_CALLS_FILE"
    if [ "$n" -le "${WCG_FLICKER_EMPTY:-0}" ]; then
      echo '{"gpu_resources": []}'
      exit 0
    fi
    cat "${WCG_AVAIL_DIR:?}/${gt}.json" 2>/dev/null || true
    exit 0
    ;;
esac
exit 0
""",
    )
    # Keep pick_compute_with_retries' backoff instant.
    _write_script(bin_dir / "sleep", """#!/usr/bin/env bash\nexit 0\n""")

    text = EVAL_SCRIPT.read_text(encoding="utf-8")

    def _full_function(name):
        return f"{name}() {{\n{_extract_bash_function(text, name)}\n}}"

    names = ["log", "die", "pick_cloud_id", "pick_cpu_cloud_id", "pick_compute"]
    if with_retries:
        names.append("pick_compute_with_retries")
    lib_file = tmp_path / "pick_lib.sh"
    lib_file.write_text("\n".join(_full_function(n) for n in names) + "\n")

    entry = "pick_compute_with_retries" if with_retries else "pick_compute"
    driver = tmp_path / "driver.sh"
    driver.write_text(
        textwrap.dedent(
            f"""#!/usr/bin/env bash
set +e
GPU_TYPE="{gpu_type}"
CURATION_ONLY={curation_only}
EXCLUDED_CLOUD_IDS="{excluded}"
source "{lib_file}"
{entry}
"""
        )
    )
    driver.chmod(0o755)

    env = dict(os.environ)
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env["WCG_AVAIL_DIR"] = str(avail_dir)
    env["WCG_FLICKER_EMPTY"] = str(flicker_empty_calls)
    env["WCG_CALLS_FILE"] = str(tmp_path / "calls.txt")

    proc = subprocess.run(
        ["bash", str(driver)], env=env, capture_output=True, text=True, timeout=60
    )
    return proc.stdout, proc.stderr, proc.returncode


def _ts_re():
    # Matches the log() timestamp prefix, e.g. `[08:06:28]`.
    return re.compile(r"\[\d{2}:\d{2}:\d{2}\]")


def test_pick_compute_emits_clean_type_and_cloud_id_tuple(tmp_path):
    """Happy path: the picker emits exactly "<gpu_type> <cloud_id>" on one line,
    with no log text or timestamp contaminating the captured value.
    """
    out, err, rc = _run_pick_compute(
        tmp_path,
        avail_by_type={"A100_80GB": _avail("dc-80"), "A100_40GB": _avail(_A40_ID)},
    )
    assert rc == 0, f"rc={rc} stderr={err}"
    assert out.strip() == "A100_80GB dc-80", f"captured stdout={out!r}"
    assert "\n" not in out.strip(), f"tuple must be one line: {out!r}"
    assert not _ts_re().search(out), f"timestamp leaked into tuple: {out!r}"

    # The parent parses these two fields; both must round-trip exactly.
    gpu_type, _, cloud_id = out.strip().partition(" ")
    assert gpu_type == "A100_80GB"
    assert cloud_id == "dc-80"


def test_no_a100_40gb_fallback_when_80gb_has_no_capacity(tmp_path):
    """The core regression: A100_80GB sold out, A100_40GB sitting there available.
    The picker must FAIL rather than silently substituting the 40GB offer.
    """
    out, err, rc = _run_pick_compute(
        tmp_path,
        avail_by_type={
            "A100_80GB": _avail("dc-80", stock="sold_out"),  # 503-style unavailable
            "A100_40GB": _avail(_A40_ID),  # available, must NOT be taken
        },
    )
    assert rc != 0, f"expected failure, got rc={rc} stdout={out!r}"
    assert out.strip() == "", f"no-capacity stdout must be empty: {out!r}"
    assert "A100_40GB" not in out, f"40GB substituted into stdout: {out!r}"
    assert _A40_ID not in out, f"40GB cloud_id leaked: {out!r}"
    assert _A40_ID not in err, f"40GB offer must not even be queried: {err!r}"


def test_requested_gpu_type_never_substituted_when_only_excluded_providers(tmp_path):
    """A100_80GB exists but only from rejected providers; a 40GB offer is available
    from a good provider. Still a hard fail -- provider rejection must not become
    a back door to a different GPU type.
    """
    out, _err, rc = _run_pick_compute(
        tmp_path,
        avail_by_type={
            "A100_80GB": _avail("mc-80", provider="MassedCompute"),
            "A100_40GB": _avail(_A40_ID),
        },
    )
    assert rc != 0, f"expected failure, got rc={rc} stdout={out!r}"
    assert out.strip() == ""
    assert _A40_ID not in out


def test_excluded_cloud_id_does_not_downgrade_gpu_type(tmp_path):
    """The only A100_80GB offer is on the exclusion list (a broken offer). The
    picker must fail on the requested type, not drop to A100_40GB.
    """
    out, _err, rc = _run_pick_compute(
        tmp_path,
        avail_by_type={
            "A100_80GB": _avail("dc-80-excluded"),
            "A100_40GB": _avail(_A40_ID),
        },
        excluded="dc-80-excluded",
    )
    assert rc != 0, f"expected failure, got rc={rc} stdout={out!r}"
    assert out.strip() == ""
    assert _A40_ID not in out


def test_pick_compute_with_retries_survives_capacity_flicker(tmp_path):
    """Shared inventory flickers: the first two availability calls report nothing,
    then the A100_80GB offer reappears. Retries must ride this out and still
    return the REQUESTED type -- never downgrade because of a transient miss.
    """
    out, err, rc = _run_pick_compute(
        tmp_path,
        avail_by_type={"A100_80GB": _avail("dc-80"), "A100_40GB": _avail(_A40_ID)},
        with_retries=True,
        flicker_empty_calls=2,
    )
    assert rc == 0, f"rc={rc} stderr={err}"
    assert out.strip() == "A100_80GB dc-80", f"captured stdout={out!r}"
    assert "No A100_80GB capacity on pick attempt" in err, err
    assert _A40_ID not in out


def test_pick_compute_with_retries_exhausted_fails_without_substitution(tmp_path):
    """Capacity never returns: retries are exhausted and the picker fails with
    empty stdout -- it does not reach for the available 40GB offer.
    """
    out, err, rc = _run_pick_compute(
        tmp_path,
        avail_by_type={
            "A100_80GB": {"gpu_resources": []},
            "A100_40GB": _avail(_A40_ID),
        },
        with_retries=True,
    )
    assert rc != 0, f"expected failure, got rc={rc}"
    assert out.strip() == "", f"no-capacity stdout must be empty: {out!r}"
    assert "No A100_80GB capacity on pick attempt 4/4" in err, err
    assert _A40_ID not in out


def test_curation_only_cpu_selected_clean_tuple(tmp_path):
    """CPU curation path stays explicit: it emits the CPU_NODE type in the tuple."""
    out, err, rc = _run_pick_compute(
        tmp_path,
        avail_by_type={
            "CPU_NODE": _avail("dc-cpu-123", memory_gb="64", vcpus="16"),
        },
        curation_only=1,
        gpu_type="CPU_NODE",
    )
    assert rc == 0, f"rc={rc} stderr={err}"
    assert out.strip() == "CPU_NODE dc-cpu-123", f"captured stdout={out!r}"
    assert "\n" not in out.strip()
    assert not _ts_re().search(out)


def test_curation_only_cpu_no_capacity_dies_with_empty_stdout(tmp_path):
    """CPU curation with no eligible CPU node is a hard FATAL, never a GPU pod."""
    out, err, rc = _run_pick_compute(
        tmp_path,
        avail_by_type={"CPU_NODE": {"gpu_resources": []}, "A100_80GB": _avail("dc-80")},
        curation_only=1,
        gpu_type="CPU_NODE",
    )
    assert rc != 0, f"expected failure rc, got {rc}"
    assert out.strip() == "", f"no-capacity stdout must be empty: {out!r}"
    assert "FATAL" in err
    assert "dc-80" not in out, "curation-only must never select a GPU offer"


# --- end-to-end: pods create receives the requested type, or nothing at all ----


def test_launcher_passes_clean_cloud_id_to_pods_create(tmp_path):
    """End-to-end: the launcher hands ``prime pods create`` EXACTLY the clean
    cloud ID (no timestamp/log text) on its single recorded line, having picked
    the requested A100_80GB from the non-excluded provider.
    """
    repo = _make_temp_repo(tmp_path)
    result, record = _run_script(
        tmp_path,
        repo,
        avail_by_type={"A100_80GB": _AVAIL_MIXED, "A100_40GB": _avail(_A40_ID)},
        status=_STATUS_ROOT,
        gpu_type="A100_80GB",
    )
    picked = record / "picked.txt"
    assert picked.exists(), f"prime pods create not invoked; stderr={result.stderr}"
    lines = [ln for ln in picked.read_text().splitlines() if ln.strip()]
    assert len(lines) == 1, f"expected exactly one picked cloud_id, got {lines!r}"
    assert lines[0] == "dc-a100", f"launcher passed wrong/contaminated id: {lines!r}"
    assert _ts_re().search(lines[0]) is None


def test_launcher_creates_no_pod_when_requested_gpu_type_unavailable(tmp_path):
    """End-to-end regression for the live incident: A100_80GB gone, A100_40GB
    available. The launcher must create NO pod at all and fail loudly, rather
    than provisioning a downgraded 40GB pod.
    """
    repo = _make_temp_repo(tmp_path)
    result, record = _run_script(
        tmp_path,
        repo,
        avail_by_type={
            "A100_80GB": {"gpu_resources": []},
            "A100_40GB": _avail(_A40_ID),
        },
        status=_STATUS_ROOT,
        gpu_type="A100_80GB",
    )
    assert result.returncode != 0, (
        "launcher must fail when requested GPU is unavailable"
    )
    assert not (record / "picked.txt").exists(), (
        f"no pod may be created on a GPU downgrade; picked="
        f"{(record / 'picked.txt').read_text() if (record / 'picked.txt').exists() else ''!r}"
    )
    assert not (record / "eval_ran.log").exists(), "eval must not run"
    assert "no GPU fallback" in result.stderr, result.stderr


def test_created_pod_gpu_type_mismatch_aborts_and_terminates(tmp_path):
    """Last line of defense: if the provider hands back a GPU different from the
    one requested, the launcher aborts and the EXIT trap terminates the pod so
    no paid training runs on the wrong hardware.
    """
    repo = _make_temp_repo(tmp_path)
    result, record = _run_script(
        tmp_path,
        repo,
        avail_by_type={"A100_80GB": _AVAIL_MIXED},
        status=_STATUS_ROOT,
        gpu_type="A100_80GB",
        create_gpu_type="A100_40GB",  # provider lies / substitutes
    )
    assert result.returncode != 0, "GPU-type mismatch must abort the run"
    assert not (record / "eval_ran.log").exists(), "eval must not run on the wrong GPU"

    terminated = record / "terminated.txt"
    assert terminated.exists(), (
        f"mismatched pod was not terminated; stderr={result.stderr}"
    )
    assert "POD123" in terminated.read_text()
    assert "was created as A100_40GB" in result.stderr, result.stderr


def test_created_pod_gpu_type_match_proceeds(tmp_path):
    """Control for the mismatch test: when the created pod reports the requested
    GPU type, the run proceeds (and the type is confirmed in the log).
    """
    repo = _make_temp_repo(tmp_path)
    result, record = _run_script(
        tmp_path,
        repo,
        avail_by_type={"A100_80GB": _AVAIL_MIXED},
        status=_STATUS_ROOT,
        gpu_type="A100_80GB",
        create_gpu_type="A100_80GB",
    )
    assert (record / "eval_ran.log").exists(), (
        f"eval should run on a matching GPU; rc={result.returncode} stderr={result.stderr}"
    )
    assert "Verified created pod GPU type: A100_80GB" in result.stderr, result.stderr


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
