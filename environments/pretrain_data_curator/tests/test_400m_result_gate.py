"""Deterministic regressions for the 400M semantic result gate."""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path

import pytest

from pretrain_data_curator.result_gate import validate_400m_results

REPO_ROOT = Path(__file__).resolve().parents[3]
EVAL_SCRIPT = REPO_ROOT / "scripts" / "run_400m_eval_a100.sh"


def _valid_row() -> dict:
    return {
        "is_completed": True,
        "stop_condition": "agent_completed",
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


def _write_result(tmp_path: Path, row: dict) -> Path:
    path = tmp_path / "results.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row) + "\n")
    return path


def test_valid_production_result_passes(tmp_path: Path):
    result = _write_result(tmp_path, _valid_row())
    assert (
        validate_400m_results(result, require_production_training=True)
        == "valid_rows=1 mode=production"
    )


def test_missing_model_manifest_fails_closed(tmp_path: Path):
    row = _valid_row()
    row["metrics"].update(
        {"finalized": 0.0, "manifest_missing": 1.0, "corpus_tokens": 0.0}
    )
    with pytest.raises(ValueError, match="finalized=1"):
        validate_400m_results(
            _write_result(tmp_path, row), require_production_training=True
        )


def test_zero_corpus_or_sources_fails_closed(tmp_path: Path):
    for metric in ("corpus_tokens", "num_sources"):
        row = _valid_row()
        row["metrics"][metric] = 0.0
        with pytest.raises(ValueError, match=rf"{metric} must be > 0"):
            validate_400m_results(
                _write_result(tmp_path / metric, row),
                require_production_training=True,
            )


@pytest.mark.parametrize(
    ("location", "metric", "value", "message"),
    [
        ("metrics", None, {}, "empty or missing metrics"),
        ("reward", None, None, "numeric reward"),
        ("metrics", "train_flops", None, "numeric train_flops"),
        ("metrics", "perf_loss", float("nan"), "perf_loss must be finite"),
        ("metrics", "train_flops", float("inf"), "train_flops must be finite"),
        ("metrics", "trainer_error_msg", float("-inf"), "must be finite"),
    ],
)
def test_empty_none_and_nonfinite_metrics_fail_closed(
    tmp_path: Path,
    location: str,
    metric: str | None,
    value: object,
    message: str,
):
    row = _valid_row()
    if location == "reward":
        row["rewards"]["reward"] = value
    elif metric is None:
        row["metrics"] = value
    else:
        row["metrics"][metric] = value
    with pytest.raises(ValueError, match=message):
        validate_400m_results(
            _write_result(tmp_path, row), require_production_training=True
        )


def test_curation_only_does_not_require_production_training_metrics(tmp_path: Path):
    row = _valid_row()
    for name in ("train_flops", "perf_loss", "trainer_error_msg"):
        row["metrics"].pop(name)
    assert "mode=curation-only" in validate_400m_results(
        _write_result(tmp_path, row), require_production_training=False
    )


def test_site_manifest_summary_is_none_and_nonfinite_safe(tmp_path: Path):
    script = EVAL_SCRIPT.read_text()
    start = script.index("import json, math, sys")
    end = script.index("\nPY", start)
    body = script[start:end]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "run_count": 3,
                "runs": [
                    {"model": None, "reward": None},
                    {"model": "nan", "reward": math.nan},
                    {"model": "valid", "reward": 0.125},
                ],
            }
        )
    )
    result = subprocess.run(
        ["python3", "-", str(manifest)],
        input=body,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("reward=n/a") == 2
    assert "reward=0.1250" in result.stdout


def test_launcher_gates_remote_and_downloaded_results_before_site_work():
    script = EVAL_SCRIPT.read_text()
    assert "SEMANTIC_INVALID=" in script
    assert "STATUS=semantic_invalid" in script
    assert "REMOTE_SEMANTIC_INVALID=1" in script

    remote_gate = script.index("python3 pretrain_data_curator/result_gate.py")
    remote_status = script.index("SEMANTIC_INVALID=", remote_gate)
    assert remote_gate < remote_status

    download = script.rindex('download_results "$RESULTS_REL"')
    local_gate = script.rindex("validate_downloaded_results")
    summary = script.rindex('log "Eval summary:"')
    site = script.rindex('if [[ "$SKIP_SITE" -eq 0 ]]')
    assert download < local_gate < summary < site
