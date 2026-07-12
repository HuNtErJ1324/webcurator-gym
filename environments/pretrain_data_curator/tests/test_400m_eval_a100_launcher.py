"""Deterministic tests for 400M A100 launcher root-provider safeguards."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
EVAL_SCRIPT = REPO_ROOT / "scripts" / "run_400m_eval_a100.sh"


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


def test_rsync_permanent_failure_executes_immediately(tmp_path: Path):
    """Execute the permanent-failure branch: one attempt, no retry sleep."""
    probe = tmp_path / "rsync_perm_probe.sh"
    probe.write_text(
        r"""#!/usr/bin/env bash
set -euo pipefail
attempts=0
die() { echo "DIE: $*"; exit 2; }
log() { echo "LOG: $*"; }
remote_rsync() {
  local attempt stderr
  local -r PERMANENT_RE='permission denied|cannot stat|link_stat|mkdir failed|read-only file system|read-only filesystem'
  for attempt in 1 2 3 4 5; do
    attempts=$((attempts + 1))
    stderr="$(mktemp)"
    echo 'rsync: mkdir "/root/webcurator-gym" failed: Permission denied (13)' >"$stderr"
    if false; then
      rm -f "$stderr"
      return 0
    fi
    if grep -Eqi "$PERMANENT_RE" "$stderr"; then
      local reason
      reason="$(grep -Ei "$PERMANENT_RE" "$stderr" | head -n1)"
      rm -f "$stderr"
      die "rsync permanent failure (no retry): $reason"
    fi
    rm -f "$stderr"
    log "rsync attempt $attempt failed; retrying in 20s"
    sleep 20
  done
  die "rsync failed after 5 attempts"
}
remote_rsync
echo "attempts=$attempts"
""",
        encoding="utf-8",
    )
    probe.chmod(0o755)
    proc = subprocess.run(
        ["bash", str(probe)], check=False, capture_output=True, text=True
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 2, combined
    assert "rsync permanent failure (no retry)" in combined
    assert "Permission denied" in combined
    assert "retrying in 20s" not in combined
    assert "attempts=" not in combined  # never reached after die


def test_upload_secrets_enforces_chmod_600():
    text = EVAL_SCRIPT.read_text(encoding="utf-8")
    fn = _extract_bash_function(text, "upload_secrets")
    assert "chmod 600 /root/webcurator-gym/secrets.env" in fn
    assert "Failed to chmod 600 the remote secrets file" in fn
    assert re.search(r"if ! remote chmod 600", fn)
    assert "scp -i" in fn
    assert ">/dev/null" in fn
    assert 'echo "$tmp"' not in fn
    assert 'cat "$tmp"' not in fn


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
