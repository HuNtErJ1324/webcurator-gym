"""Regression tests for A100 smoke launcher profile + result gating."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from pretrain_data_curator.smoke_result_gate import validate_smoke_results

REPO_ROOT = Path(__file__).resolve().parents[3]
ENV_DIR = Path(__file__).resolve().parents[1]
SMOKE_SCRIPT = REPO_ROOT / "scripts" / "run_25m_smoke_a100.sh"
CONFIG_10M = ENV_DIR / "configs" / "eval" / "10M-60turn-codex-smoke.toml"
CONFIG_25M = ENV_DIR / "configs" / "eval" / "25M-60turn-codex-smoke.toml"


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


def _dry_run(extra: list[str]) -> str:
    cmd = [
        str(SMOKE_SCRIPT),
        "--model",
        "deepseek/deepseek-v4-pro",
        "--gpu-type",
        "A100_80GB",
        "--dry-run",
        *extra,
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return proc.stdout + proc.stderr


def test_smoke_script_selects_10m_profile_and_suffix():
    assert SMOKE_SCRIPT.is_file()
    assert CONFIG_10M.is_file()
    out = _dry_run(["--profile", "10M"])
    assert "profile=10M" in out
    assert "base_config=configs/eval/10M-60turn-codex-smoke.toml" in out
    assert "run_suffix=10M-60turn-codex-smoke" in out
    assert "expected_token_budget=10000000" in out
    assert "allow_gpu_fallback=0" in out
    assert "run_name=deepseek-deepseek-v4-pro-10M-60turn-codex-smoke" in out
    assert "compute_type=A100_80GB" in out


def test_smoke_script_default_remains_25m():
    assert CONFIG_25M.is_file()
    out = _dry_run([])
    assert "profile=25M" in out
    assert "base_config=configs/eval/25M-60turn-codex-smoke.toml" in out
    assert "run_suffix=25M-60turn-codex-smoke" in out
    assert "expected_token_budget=25000000" in out
    assert "allow_gpu_fallback=1" in out


def test_smoke_script_config_flag_selects_10m():
    out = _dry_run(["--config", "configs/eval/10M-60turn-codex-smoke.toml"])
    assert "profile=10M" in out
    assert "run_suffix=10M-60turn-codex-smoke" in out
    assert "allow_gpu_fallback=0" in out


def test_smoke_script_preserves_cleanup_trap_and_runtime_gates():
    text = SMOKE_SCRIPT.read_text(encoding="utf-8")
    assert "trap cleanup EXIT" in text
    cleanup = _extract_bash_function(text, "cleanup")
    assert "prime pods terminate" in cleanup
    assert "KEEP_POD" in cleanup

    gpu = _extract_bash_function(text, "remote_provision_gpu")
    assert "nvidia-smi" in gpu
    assert "nvidia-ctk" in gpu or "nvidia-container-toolkit" in gpu
    assert "webcurator-runtime:latest" in gpu
    assert "Dockerfile.runtime" in gpu

    fallback = _extract_bash_function(text, "pick_cloud_id_with_fallback")
    assert 'ALLOW_GPU_FALLBACK" -eq 1' in fallback or "ALLOW_GPU_FALLBACK" in fallback
    assert "A100_40GB" in fallback

    # select_cloud_id_or_die must be invoked directly so die hits the main shell.
    assert re.search(r"(?m)^select_cloud_id_or_die$", text)
    assert 'CLOUD_ID="$(select_cloud_id_or_die)"' not in text


def test_smoke_script_site_rebuild_only_after_valid_gate():
    text = SMOKE_SCRIPT.read_text(encoding="utf-8")
    assert "--skip-site" in text
    assert "SKIP_SITE=0" in text
    validate_at = text.find("validate_downloaded_results")
    rebuild_at = text.find("rebuild_site")
    skip_at = text.find('SKIP_SITE" -eq 0')
    assert 0 <= validate_at < rebuild_at
    assert validate_at < skip_at
    # Failure path must retain artifacts and skip site rebuild.
    assert "site rebuild skipped" in text
    assert "retaining artifacts" in text


def _write_valid_results(tmp_path: Path, *, budget: int = 10_000_000) -> Path:
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.jsonl").write_text(
        json.dumps(
            {
                "id": "smoke-test",
                "is_completed": True,
                "rewards": {"reward": 0.1},
                "metrics": {"corpus_tokens": 1000},
                "errors": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (out / "config.toml").write_text(
        "\n".join(
            [
                'model = "deepseek/deepseek-v4-pro"',
                "[args]",
                f"token_budget = {budget}",
                "[args.proxy_student]",
                f"train_token_budget = {budget}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (out / "eval-stream.log").write_text(
        "results: outputs/evals/example\n",
        encoding="utf-8",
    )
    return out


def test_validate_smoke_results_accepts_completed_budget_match(tmp_path: Path):
    out = _write_valid_results(tmp_path)
    msg = validate_smoke_results(
        out,
        expected_token_budget=10_000_000,
        run_suffix="10M-60turn-codex-smoke",
    )
    assert "valid_records=1" in msg
    assert "expected_token_budget=10000000" in msg


def test_validate_smoke_results_rejects_failure_marker(tmp_path: Path):
    out = _write_valid_results(tmp_path)
    (out / "FAILED").write_text("boom\n", encoding="utf-8")
    with pytest.raises(ValueError, match="failure marker"):
        validate_smoke_results(out, expected_token_budget=10_000_000)


def test_validate_smoke_results_rejects_incomplete_and_budget_mismatch(tmp_path: Path):
    out = _write_valid_results(tmp_path)
    (out / "results.jsonl").write_text(
        json.dumps({"id": "x", "is_completed": False}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="is_completed"):
        validate_smoke_results(out, expected_token_budget=10_000_000)

    out = _write_valid_results(tmp_path / "budget", budget=25_000_000)
    with pytest.raises(ValueError, match="token_budget"):
        validate_smoke_results(out, expected_token_budget=10_000_000)


def test_validate_smoke_results_rejects_traceback_without_results(tmp_path: Path):
    out = _write_valid_results(tmp_path)
    (out / "eval-stream.log").write_text(
        "Traceback (most recent call last):\nRuntimeError: boom\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="traceback without results"):
        validate_smoke_results(out, expected_token_budget=10_000_000)


def test_smoke_script_validate_only_gate(tmp_path: Path):
    out = _write_valid_results(tmp_path)
    ok = subprocess.run(
        [
            str(SMOKE_SCRIPT),
            "--model",
            "deepseek/deepseek-v4-pro",
            "--profile",
            "10M",
            "--validate-only",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert ok.returncode == 0, ok.stderr + ok.stdout
    assert "Validation OK" in (ok.stdout + ok.stderr)

    (out / "FAILED").write_text("nope\n", encoding="utf-8")
    bad = subprocess.run(
        [
            str(SMOKE_SCRIPT),
            "--model",
            "deepseek/deepseek-v4-pro",
            "--profile",
            "10M",
            "--validate-only",
            str(out),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert bad.returncode != 0
    assert out.exists()  # invalid results retained


def test_10m_smoke_config_uses_webcurator_runtime_and_10m_budget():
    import tomllib

    cfg = tomllib.loads(CONFIG_10M.read_text(encoding="utf-8"))
    assert cfg["args"]["token_budget"] == 10_000_000
    proxy = cfg["args"]["proxy_student"]
    assert proxy["train_token_budget"] == 10_000_000
    assert proxy["runtime_backend"] == "docker"
    assert proxy["docker_image"] == "webcurator-runtime:latest"


def test_25m_smoke_config_uses_webcurator_runtime_and_25m_budget():
    import tomllib

    cfg = tomllib.loads(CONFIG_25M.read_text(encoding="utf-8"))
    assert cfg["args"]["token_budget"] == 25_000_000
    proxy = cfg["args"]["proxy_student"]
    assert proxy["train_token_budget"] == 25_000_000
    assert proxy["runtime_backend"] == "docker"
    assert proxy["docker_image"] == "webcurator-runtime:latest"
    assert "pytorch/pytorch" not in proxy["docker_image"]


def test_curation_only_cpu_miss_fails_without_retry_sleeps(tmp_path: Path):
    """CPU miss must return failure cleanly — no die-in-$() and no 4x sleep."""
    text = SMOKE_SCRIPT.read_text(encoding="utf-8")
    fallback = _extract_bash_function(text, "pick_cloud_id_with_fallback")
    retries = _extract_bash_function(text, "pick_cloud_id_with_retries")
    select = _extract_bash_function(text, "select_cloud_id_or_die")

    # Curation branch must return 1, never die inside the $()-called helper.
    curation_branch = fallback.split("local types=")[0]
    assert "CURATION_ONLY" in curation_branch
    assert "return 1" in curation_branch
    assert not re.search(r"(?m)^\s*die\s", curation_branch)

    # Retries helper short-circuits curation-only before the sleep loop.
    assert "CURATION_ONLY" in retries
    sleep_at = retries.find("sleep 10")
    curation_return = retries.find("CURATION_ONLY")
    assert 0 <= curation_return < sleep_at

    # die runs in select_cloud_id_or_die (direct call), not via echo/$().
    assert "die " in select
    assert 'echo "$picked"' not in select

    probe = tmp_path / "curation_cpu_probe.sh"
    count_file = tmp_path / "fallback_calls.txt"
    sleep_file = tmp_path / "sleep_calls.txt"
    probe.write_text(
        rf"""#!/usr/bin/env bash
set -euo pipefail
CURATION_ONLY=1
GPU_TYPE="CPU_NODE"
COUNT_FILE='{count_file}'
SLEEP_FILE='{sleep_file}'
: > "$COUNT_FILE"
: > "$SLEEP_FILE"
die() {{ echo "FATAL: $*" >&2; exit 2; }}
log() {{ :; }}
pick_cpu_cloud_id() {{ return 1; }}
pick_cloud_id_with_fallback() {{
  echo called >> "$COUNT_FILE"
  if [[ "$CURATION_ONLY" -eq 1 ]]; then
    if CLOUD_ID="$(pick_cpu_cloud_id)"; then
      GPU_TYPE="CPU_NODE"
      echo "$CLOUD_ID"
      return 0
    fi
    return 1
  fi
  return 1
}}
pick_cloud_id_with_retries() {{
  local attempt picked
  if [[ "$CURATION_ONLY" -eq 1 ]]; then
    if picked="$(pick_cloud_id_with_fallback)"; then
      echo "$picked"
      return 0
    fi
    return 1
  fi
  for attempt in 1 2 3 4; do
    if picked="$(pick_cloud_id_with_fallback)"; then
      echo "$picked"
      return 0
    fi
    echo sleep >> "$SLEEP_FILE"
  done
  return 1
}}
select_cloud_id_or_die() {{
  local picked
  if picked="$(pick_cloud_id_with_retries)"; then
    CLOUD_ID="$picked"
    return 0
  fi
  if [[ "$CURATION_ONLY" -eq 1 ]]; then
    die "No available CPU_NODE slots with >=8 vCPU and >=32 GB RAM"
  fi
  die "No available compute slots ($GPU_TYPE fallback exhausted)"
}}
select_cloud_id_or_die
echo "unexpected success"
exit 1
""",
        encoding="utf-8",
    )
    probe.chmod(0o755)
    proc = subprocess.run(
        ["bash", str(probe)],
        check=False,
        capture_output=True,
        text=True,
    )
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 2, combined
    assert "FATAL: No available CPU_NODE slots" in combined
    assert "fallback exhausted" not in combined
    assert count_file.read_text(encoding="utf-8").count("called") == 1
    assert sleep_file.read_text(encoding="utf-8").strip() == ""


def test_write_resolved_config_escapes_quotes_and_newlines(tmp_path: Path):
    import tomllib

    src = CONFIG_10M
    dest = tmp_path / "resolved.toml"
    helper = tmp_path / "resolve_model.py"
    helper.write_text(
        "\n".join(
            [
                "import json, pathlib, re, sys",
                "model, src, dest = sys.argv[1:4]",
                "text = pathlib.Path(src).read_text()",
                "model_lit = json.dumps(model, ensure_ascii=False)",
                "text = re.sub(",
                "    r'^model\\s*=.*$',",
                "    lambda _m: f'model = {model_lit}',",
                "    text,",
                "    count=1,",
                "    flags=re.M,",
                ")",
                "pathlib.Path(dest).write_text(text)",
                "",
            ]
        ),
        encoding="utf-8",
    )
    nasty = 'org/model"quote\nline\tpath\\x'
    subprocess.run(
        ["python3", str(helper), nasty, str(src), str(dest)],
        check=True,
        capture_output=True,
        text=True,
    )
    # Script must use json.dumps for the same escaping contract.
    script = SMOKE_SCRIPT.read_text(encoding="utf-8")
    assert "json.dumps(model" in script
    parsed = tomllib.loads(dest.read_text(encoding="utf-8"))
    assert parsed["model"] == nasty


def test_remote_eval_quotes_eval_config_path():
    text = SMOKE_SCRIPT.read_text(encoding="utf-8")
    assert 'uv run eval -m "${MODEL}" @ "${EVAL_CONFIG}"' in text
    assert "@ ${EVAL_CONFIG}" not in text
    # Whitespace / glob-sensitive paths must stay inside quotes.
    assert '"${EVAL_CONFIG}"' in text
    assert 'tee "${remote_log}"' in text


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

    script = SMOKE_SCRIPT.read_text(encoding="utf-8")
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
            )
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


def test_gpu_filter_excludes_massedcompute_and_crusoe_case_insensitive():
    # MassedCompute (any case) and crusoecloud must be skipped; the cheapest
    # root-capable provider must be selected instead.
    resources = [
        _gpu_resource("mc-cheap", "MassedCompute", price="0.40"),
        _gpu_resource("mc-low", "massedcompute", price="0.41"),
        _gpu_resource("cr-cheap", "crusoecloud", price="0.50"),
        _gpu_resource("rp-root", "runpod", price="0.99"),
    ]
    selected = _select_cloud_id("pick_cloud_id", resources)
    assert selected == "rp-root"
    # When only MassedCompute is available the filter must yield nothing.
    with pytest.raises(AssertionError):
        _select_cloud_id("pick_cloud_id", [_gpu_resource("mc-only", "massedcompute")])


def test_cpu_filter_excludes_massedcompute_case_insensitive():
    resources = [
        _cpu_resource("mc-cpu", "MassedCompute", price="0.40"),
        _cpu_resource("cr-cpu", "crusoecloud", price="0.50"),
        _cpu_resource("rp-cpu", "runpod", price="0.90"),
    ]
    selected = _select_cloud_id("pick_cpu_cloud_id", resources)
    assert selected == "rp-cpu"
    with pytest.raises(AssertionError):
        _select_cloud_id(
            "pick_cpu_cloud_id", [_cpu_resource("mc-only", "massedcompute")]
        )


def test_root_capable_provider_selected_over_unsupported():
    # A root-capable provider must win even when a cheaper MassedCompute offer
    # exists; this is the "root-capable provider selection" guarantee.
    resources = [
        _gpu_resource("mc-cheapest", "massedcompute", price="0.10"),
        _gpu_resource("rp-root", "runpod", price="1.20"),
    ]
    assert _select_cloud_id("pick_cloud_id", resources) == "rp-root"


def test_non_root_ssh_user_fail_fast_terminates_pod(tmp_path: Path):
    text = SMOKE_SCRIPT.read_text(encoding="utf-8")
    # The root check must sit between SSH auth and the rsync/sync step.
    auth_at = text.find('wait_for_ssh_auth\n')
    root_check_at = text.find(
        'if [[ "$SSH_USER" != "root" ]]; then'
    )
    rsync_at = text.find("\nremote_rsync\n")
    assert 0 <= auth_at < root_check_at < rsync_at, "root check misplaced"
    assert "FATAL: pod authenticated as non-root user" in text
    assert "Unsupported provider image" in text
    # The fail-fast must run under the EXIT trap so the pod is terminated.
    assert "trap cleanup EXIT" in text
    cleanup = _extract_bash_function(text, "cleanup")
    assert "prime pods terminate" in cleanup

    # Drive the exact guarded block in a controlled subshell: a non-root user
    # must die (non-zero) and never reach the sync step.
    probe = tmp_path / "non_root_probe.sh"
    probe.write_text(
        r"""#!/usr/bin/env bash
set -euo pipefail
SSH_USER="ubuntu"
POD_ID="pod-xyz"
KEEP_POD=0
terminated=""
die() { echo "FATAL: $*" >&2; exit 3; }
log() { :; }
cleanup() {
  local code=$?
  if [[ -n "$POD_ID" && "$KEEP_POD" -eq 0 ]]; then
    terminated="$POD_ID"
  fi
  exit "$code"
}
trap cleanup EXIT
# --- guarded block copied from launcher ---
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
    assert "FATAL: pod authenticated as non-root user 'ubuntu'" in (
        proc.stdout + proc.stderr
    )


def test_rsync_permanent_failure_fails_fast_no_retry():
    import re as _re

    text = SMOKE_SCRIPT.read_text(encoding="utf-8")
    rsync_fn = _extract_bash_function(text, "remote_rsync")
    m = _re.search(r"PERMANENT_RE='([^']+)'", rsync_fn)
    assert m, "could not find PERMANENT_RE pattern in remote_rsync"
    pattern = _re.compile(m.group(1), _re.IGNORECASE)

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

    # Permanent path must die before the retry sleep; retries must still exist.
    perm_die_at = rsync_fn.find('die "rsync permanent failure')
    sleep_at = rsync_fn.find("sleep 20")
    assert 0 <= perm_die_at < sleep_at, "permanent failure must die before retry"
    assert rsync_fn.count("sleep 20") >= 1, "transient retry/backoff preserved"


def test_rsync_transient_retry_preserved():
    text = SMOKE_SCRIPT.read_text(encoding="utf-8")
    rsync_fn = _extract_bash_function(text, "remote_rsync")
    # Retry loop over multiple attempts with backoff sleep must remain.
    assert re.search(r"for attempt in 1 2 3 4 5", rsync_fn)
    assert 'log "rsync attempt $attempt failed; retrying in 20s"' in rsync_fn
    assert 'die "rsync failed after 5 attempts"' in rsync_fn


def test_upload_secrets_enforces_chmod_600(tmp_path: Path):
    text = SMOKE_SCRIPT.read_text(encoding="utf-8")
    fn = _extract_bash_function(text, "upload_secrets")
    # Must set restrictive perms on the remote secrets file.
    assert "chmod 600 /root/webcurator-gym/secrets.env" in fn
    # Must fail the run if chmod fails (never leave secrets world-readable).
    assert "Failed to chmod 600 the remote secrets file" in fn
    assert re.search(r"if ! remote chmod 600", fn)
    # Secret contents must never be logged: scp output is suppressed and the
    # function must not echo the temp file or its contents.
    assert "scp -i" in fn
    assert ">/dev/null" in fn
    assert "echo \"$tmp\"" not in fn
    assert "cat \"$tmp\"" not in fn


def test_cleanup_trap_preserved_and_terminates_pod():
    text = SMOKE_SCRIPT.read_text(encoding="utf-8")
    assert "trap cleanup EXIT" in text
    cleanup = _extract_bash_function(text, "cleanup")
    assert "prime pods terminate" in cleanup
    assert "KEEP_POD" in cleanup
    # Unconditional termination: a non-zero run still hits cleanup (EXIT trap).
    assert "exit \"$code\"" in cleanup
