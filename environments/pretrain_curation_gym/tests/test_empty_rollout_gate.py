"""Empty-rollout detection (fix #3) and the v1 smoke result-gate port.

The environment flags a rollout that produced no usable artifact — no valid
workspace manifest and zero self-scores — which, under an opaque agent harness,
is the only reliable signal of an infrastructure/model-endpoint failure (the
trace itself carries no turns or usage). With ``error_on_empty_rollout`` set it
is raised as a retryable ``EmptyRolloutError`` instead of a silent zero, and the
result gate rejects it (and a non-materialized corpus) in downloaded results.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import verifiers.v1 as vf
from verifiers.v1.errors import RolloutError

from pretrain_curation_gym import (
    CuratorTaskConfig,
    CuratorTaskset,
    CuratorTasksetConfig,
)
from pretrain_curation_gym.models import (
    MANIFEST_PROVENANCE_MISSING,
    MANIFEST_PROVENANCE_WORKSPACE_FILE,
)
from pretrain_curation_gym.state import CuratorState
from pretrain_curation_gym.task import CuratorTask, EmptyRolloutError


class _ZeroScorer:
    """Bypass real materialization/training: return a scored-but-zero result so
    ``score`` runs end to end without a runtime or Hugging Face access."""

    async def compute_scoring(self, state, runtime):
        return {
            "perf": 0.0,
            "leakage": {"leakage_score": 0.0, "num_contaminated_matches": 0},
            "decon_error": 0.0,
            "val_screen_skipped": 0.0,
            "loss": 0.0,
            "accuracy": 0.0,
            "flops": 0.0,
            "tokens": 0,
            "num_sources": 0,
            "budget_fill_ratio": 0.0,
            "perf_vs_baseline": 0.0,
        }


def _task(*, error_on_empty_rollout: bool) -> CuratorTask:
    [task] = CuratorTaskset(
        CuratorTasksetConfig(
            task=CuratorTaskConfig(error_on_empty_rollout=error_on_empty_rollout)
        )
    ).load()
    task._scorer = _ZeroScorer()
    return task


def _trace(task: CuratorTask, state: CuratorState) -> vf.Trace:
    return vf.Trace(
        task=vf.TraceTask(type=type(task).__name__, data=task.data), state=state
    )


def test_empty_rollout_predicate() -> None:
    task = _task(error_on_empty_rollout=False)
    assert task._empty_rollout(
        CuratorState(manifest_provenance=MANIFEST_PROVENANCE_MISSING, self_score_runs=0)
    )
    # engaged: at least one self-score ran
    assert not task._empty_rollout(
        CuratorState(manifest_provenance=MANIFEST_PROVENANCE_MISSING, self_score_runs=1)
    )
    # engaged: a valid workspace manifest exists
    assert not task._empty_rollout(
        CuratorState(
            manifest_provenance=MANIFEST_PROVENANCE_WORKSPACE_FILE, self_score_runs=0
        )
    )


def test_empty_rollout_error_is_named_retryable_rollout_error() -> None:
    # [retries.rollout] include matches by exception type NAME; keep it stable so
    # include=["EmptyRolloutError"] keeps retrying empty rollouts.
    assert issubclass(EmptyRolloutError, RolloutError)
    assert type(EmptyRolloutError("x")).__name__ == "EmptyRolloutError"


@pytest.mark.asyncio
async def test_empty_rollout_metric_recorded_without_raising_by_default() -> None:
    task = _task(error_on_empty_rollout=False)
    trace = _trace(
        task,
        CuratorState(
            manifest_provenance=MANIFEST_PROVENANCE_MISSING, self_score_runs=0
        ),
    )
    await task.score(trace)  # flag off preserves the RL silent-zero: no raise
    assert trace.metrics["empty_rollout"] == 1.0


@pytest.mark.asyncio
async def test_empty_rollout_raises_when_enabled() -> None:
    task = _task(error_on_empty_rollout=True)
    trace = _trace(
        task,
        CuratorState(
            manifest_provenance=MANIFEST_PROVENANCE_MISSING, self_score_runs=0
        ),
    )
    with pytest.raises(EmptyRolloutError):
        await task.score(trace)
    # the metric is recorded before the raise, so the trace still shows why.
    assert trace.metrics["empty_rollout"] == 1.0


@pytest.mark.asyncio
async def test_engaged_rollout_does_not_raise_when_enabled() -> None:
    task = _task(error_on_empty_rollout=True)
    trace = _trace(
        task,
        CuratorState(
            manifest_provenance=MANIFEST_PROVENANCE_WORKSPACE_FILE, self_score_runs=2
        ),
    )
    await task.score(trace)  # engaged rollout: never raises
    assert trace.metrics["empty_rollout"] == 0.0


# ---- result-gate port (scripts/smoke_result_gate.py) ----

_GATE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "smoke_result_gate.py"
_spec = importlib.util.spec_from_file_location("smoke_result_gate", _GATE_PATH)
assert _spec is not None and _spec.loader is not None
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)


def test_record_reward_sums_v1_component_dict() -> None:
    assert gate._record_reward(
        {"rewards": {"perf_reward": 0.5, "leakage_penalty": -0.1}}
    ) == pytest.approx(0.4)
    assert gate._record_reward({"rewards": {}}) is None
    assert gate._record_reward({"rewards": {"perf_reward": "x"}}) is None
    assert gate._record_reward({"rewards": 0.7}) == 0.7
    assert gate._record_reward({"reward": 0.3}) == 0.3  # legacy scalar fallback


def _write_result_dir(
    tmp_path: Path, record: dict, *, real_trainer: bool = True
) -> Path:
    (tmp_path / "traces.jsonl").write_text(json.dumps(record) + "\n")
    (tmp_path / "config.toml").write_text(
        "[taskset.task.curator]\n"
        "token_budget = 10000000\n"
        f"use_real_trainer = {'true' if real_trainer else 'false'}\n"
    )
    return tmp_path


def _healthy_record() -> dict:
    return {
        "is_completed": True,
        "stop_condition": "agent_completed",
        "errors": [],
        "rewards": {"perf_reward": 0.5, "leakage_penalty": -0.01},
        "metrics": {
            "corpus_tokens": 9_999_972.0,
            "empty_rollout": 0.0,
            "trainer_error_msg": 0.0,
            "train_flops": 1.6e16,
            "perf_loss": 5.4,
        },
    }


def test_gate_passes_healthy_v1_trace(tmp_path: Path) -> None:
    out = _write_result_dir(tmp_path, _healthy_record())
    assert "valid_records=1" in gate.validate_smoke_results(
        out, expected_token_budget=10_000_000
    )


def test_gate_rejects_empty_rollout(tmp_path: Path) -> None:
    rec = _healthy_record()
    rec["metrics"]["empty_rollout"] = 1.0
    rec["metrics"]["corpus_tokens"] = 0.0
    out = _write_result_dir(tmp_path, rec)
    with pytest.raises(ValueError, match="empty_rollout"):
        gate.validate_smoke_results(out, expected_token_budget=10_000_000)


def test_gate_rejects_zero_corpus(tmp_path: Path) -> None:
    rec = _healthy_record()
    rec["metrics"]["corpus_tokens"] = 0.0
    out = _write_result_dir(tmp_path, rec)
    with pytest.raises(ValueError, match="corpus_tokens"):
        gate.validate_smoke_results(out, expected_token_budget=10_000_000)


def test_gate_rejects_captured_error_payload(tmp_path: Path) -> None:
    rec = _healthy_record()
    rec["errors"] = [{"type": "EmptyRolloutError", "message": "..."}]
    out = _write_result_dir(tmp_path, rec)
    with pytest.raises(ValueError, match="error/failure payload"):
        gate.validate_smoke_results(out, expected_token_budget=10_000_000)


def test_gate_not_fatal_on_external_failure(tmp_path: Path) -> None:
    # run2 (a healthy real run) carried external_failure=1 from recovered fetch
    # errors; the gate must not treat that as fatal.
    rec = _healthy_record()
    rec["metrics"]["external_failure"] = 1.0
    out = _write_result_dir(tmp_path, rec)
    assert "valid_records=1" in gate.validate_smoke_results(
        out, expected_token_budget=10_000_000
    )
