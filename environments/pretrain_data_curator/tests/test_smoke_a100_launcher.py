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
