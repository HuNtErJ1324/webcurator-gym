"""Agent-visible turn state and self-score iteration telemetry.

Covers the three additions around rollout visibility:

* the prompt reports the framework turn limit and points to ``turns.py``;
* a telemetry-only v1 stop hook refreshes its state before each model call;
* ``self_score.py`` appends one JSON line per run to
  ``.self_score_history.jsonl``, which finalize folds into zero-weight metrics.
"""

from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from pretrain_curation_gym import CuratorEnvConfig, load_environment
from pretrain_curation_gym.models import CuratorConfig
from pretrain_curation_gym.gpu.self_score import (
    SELF_SCORE_FILENAME,
    SELF_SCORE_HISTORY_FILENAME,
    render_self_score_script,
)
from pretrain_curation_gym.gpu.turns import (
    TURNS_FILENAME,
    TURN_STATE_FILENAME,
    render_turn_state,
    render_turns_script,
)
from pretrain_curation_gym.rollout_state import CuratorState, RolloutStore
from pretrain_curation_gym.taskset import (
    CuratorTaskset,
    CuratorTasksetConfig,
    _parse_self_score_history,
)
from pretrain_curation_gym.taskdata import build_tasks
import verifiers.v1 as vf
from verifiers.v1.session import RolloutLimits


# --- prompt contract ---------------------------------------------------------


def test_prompt_reports_the_framework_turn_limit():
    task = load_environment(CuratorEnvConfig(max_turns=37)).taskset.load()[0]
    prompt = str(task.data.prompt)
    assert "Turn limit: 37 model turns" in prompt
    assert "python turns.py" in prompt


def test_prompt_never_mentions_web_search():
    prompt = str(build_tasks("2024-12-31", 1_000_000)[0].data.prompt)
    for phrase in ("web search", "websearch", "web_search", "search the web"):
        assert not re.search(phrase, prompt, re.IGNORECASE), phrase


def test_prompt_discloses_the_self_score_history_file():
    prompt = str(build_tasks("2024-12-31", 1_000_000)[0].data.prompt)
    assert SELF_SCORE_HISTORY_FILENAME in prompt


def test_framework_limit_tracks_the_trace_turn_count():
    limits = RolloutLimits(max_turns=5)
    trace = SimpleNamespace(num_turns=2)
    assert limits.reached(trace) is None
    trace.num_turns = 5
    assert limits.reached(trace) == "max_turns"


def test_turns_script_reports_the_current_framework_state(tmp_path: Path):
    (tmp_path / TURNS_FILENAME).write_bytes(render_turns_script())
    (tmp_path / TURN_STATE_FILENAME).write_bytes(render_turn_state(4, 37))
    result = subprocess.run(
        [sys.executable, TURNS_FILENAME],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "turn 5 of 37 (32 remaining after this one)" in result.stdout
    assert json.loads(result.stdout.splitlines()[-1]) == {
        "current_turn": 5,
        "max_turns": 37,
        "turns_completed": 4,
        "turns_remaining": 32,
    }


class _RecordingRuntime:
    type = "subprocess"

    def __init__(self):
        self.writes: list[tuple[str, bytes]] = []

    async def write(self, path: str, data: bytes) -> None:
        self.writes.append((path, bytes(data)))


@pytest.mark.asyncio
async def test_setup_installs_the_agent_turn_script_and_initial_state():
    task = load_environment(CuratorEnvConfig(max_turns=37)).taskset.load()[0]
    state = CuratorState()
    trace = vf.Trace(
        task=vf.TraceTask(type=type(task).__name__, data=task.data), state=state
    )
    runtime = _RecordingRuntime()

    await task.setup(trace, runtime)

    writes = dict(runtime.writes)
    assert writes[TURNS_FILENAME] == render_turns_script()
    assert writes[TURN_STATE_FILENAME] == render_turn_state(0, 37)
    assert state._turn_runtime is runtime


def test_turn_hook_refreshes_telemetry_without_enforcing_a_second_limit():
    task = load_environment(CuratorEnvConfig(max_turns=5)).taskset.load()[0]
    runtime = _RecordingRuntime()
    state = CuratorState()
    state._turn_runtime = runtime
    trace = SimpleNamespace(num_turns=2, state=state)

    assert asyncio.run(task.refresh_turn_state(trace)) is False
    assert runtime.writes == [(TURN_STATE_FILENAME, render_turn_state(2, 5))]
    assert not hasattr(task.config.curator, "max_turns")


# --- self-score history file --------------------------------------------------


def _run_self_score(tmp_path: Path, manifest: Path) -> subprocess.CompletedProcess:
    (tmp_path / SELF_SCORE_FILENAME).write_bytes(
        render_self_score_script(
            CuratorConfig(token_budget=1_000),
            decon_evals_dir=str(tmp_path / "absent-evals"),
        )
    )
    return subprocess.run(
        [sys.executable, SELF_SCORE_FILENAME, manifest.name],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_self_score_appends_one_history_line_per_run(tmp_path: Path):
    (tmp_path / "dev.jsonl").write_text(
        "\n".join(
            json.dumps({"text": "Clean development sample " * 30}) for _ in range(4)
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "draft.json"
    manifest.write_text(
        json.dumps(
            {
                "token_budget": 1_000,
                "sources": [
                    {
                        "kind": "local",
                        "local_path": "dev.jsonl",
                        "local_format": "jsonl",
                        "weight": 1.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    first = _run_self_score(tmp_path, manifest)
    assert first.returncode == 0, first.stderr
    second = _run_self_score(tmp_path, manifest)
    assert second.returncode == 0, second.stderr

    history_lines = (
        (tmp_path / SELF_SCORE_HISTORY_FILENAME)
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert len(history_lines) == 2
    stdout_payload = json.loads(second.stdout)
    record = json.loads(history_lines[-1])
    assert record["ok"] is True
    assert record["reward"] == stdout_payload["reward"]
    assert record["manifest"] == manifest.name
    assert record["sampled_documents"] == stdout_payload["sampled_documents"]
    # stdout stays exactly one JSON object: the history write never leaks there
    assert "[self-score]" not in second.stdout


# --- finalize ingestion + metrics ----------------------------------------------


def test_parse_self_score_history_skips_garbage_and_orders_rewards():
    text = "\n".join(
        [
            json.dumps({"ok": True, "reward": 0.1}),
            "not json at all",
            json.dumps(["not", "a", "dict"]),
            json.dumps({"ok": False, "reward": None}),
            json.dumps({"ok": True, "reward": True}),  # bool is not a score
            json.dumps({"ok": True, "reward": float("nan")}),
            "",
            json.dumps({"ok": True, "reward": 0.5}),
            json.dumps({"ok": True, "reward": 0.3}),
        ]
    )
    runs, ok_rewards = _parse_self_score_history(text)
    assert runs == 6  # every parseable dict counts as a run
    assert ok_rewards == [0.1, 0.5, 0.3]


def _metric_trace(task) -> vf.Trace:
    return vf.Trace(
        task=vf.TraceTask(type=type(task).__name__, data=task.data),
        state=CuratorState(),
    )


class _MetricScorer:
    async def compute_scoring(self, state, runtime):
        return {
            "perf": 0.0,
            "leakage": {
                "leakage_score": 0.0,
                "num_contaminated_matches": 0,
            },
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


def _record_metrics(task, trace) -> dict[str, float]:
    task._scorer = _MetricScorer()
    asyncio.run(task.score(trace))
    return trace.metrics


def test_self_score_metrics_reflect_the_recorded_summary():
    task = CuratorTaskset(CuratorTasksetConfig(id="test")).load()[0]
    trace = _metric_trace(task)
    RolloutStore.set_self_score_summary(trace.state, runs=4, ok_rewards=[0.1, 0.5, 0.3])
    metrics = _record_metrics(task, trace)
    assert metrics["self_score_runs"] == 4.0
    assert metrics["self_score_ok_runs"] == 3.0
    assert metrics["self_score_best_reward"] == 0.5
    assert metrics["self_score_last_reward"] == 0.3
    assert metrics["self_score_improvement"] == pytest.approx(0.2)


def test_self_score_metrics_default_to_zero_without_any_run():
    task = CuratorTaskset(CuratorTasksetConfig(id="test")).load()[0]
    trace = _metric_trace(task)
    metrics = _record_metrics(task, trace)
    assert metrics["self_score_runs"] == 0.0
    assert metrics["self_score_ok_runs"] == 0.0
    assert metrics["self_score_best_reward"] == 0.0
    assert metrics["self_score_last_reward"] == 0.0
    assert metrics["self_score_improvement"] == 0.0


def test_num_turns_metric_reports_the_trace_turn_count():
    task = CuratorTaskset(CuratorTasksetConfig(id="test")).load()[0]
    trace = SimpleNamespace(num_turns=17)
    assert asyncio.run(task.num_turns(trace)) == 17.0


class _HistoryRuntime:
    def __init__(self, payload: bytes | None):
        self._payload = payload

    async def read(self, path: str) -> bytes:
        if self._payload is None:
            raise FileNotFoundError(path)
        assert path == SELF_SCORE_HISTORY_FILENAME
        return self._payload


def test_finalize_ingest_populates_the_state_summary():
    task = CuratorTaskset(CuratorTasksetConfig(id="test")).load()[0]
    trace = _metric_trace(task)
    history = "\n".join(
        [
            json.dumps({"ok": True, "reward": -0.2}),
            json.dumps({"ok": False, "reward": None}),
            json.dumps({"ok": True, "reward": 0.4}),
        ]
    ).encode("utf-8")
    asyncio.run(task.ingest_self_score_history(trace.state, _HistoryRuntime(history)))
    assert RolloutStore.self_score_runs(trace.state) == 3
    assert RolloutStore.self_score_ok_runs(trace.state) == 2
    assert RolloutStore.self_score_first_reward(trace.state) == -0.2
    assert RolloutStore.self_score_best_reward(trace.state) == 0.4
    assert RolloutStore.self_score_last_reward(trace.state) == 0.4


def test_finalize_ingest_tolerates_missing_history_and_no_runtime():
    task = CuratorTaskset(CuratorTasksetConfig(id="test")).load()[0]
    trace = _metric_trace(task)
    asyncio.run(task.ingest_self_score_history(trace.state, _HistoryRuntime(None)))
    asyncio.run(task.ingest_self_score_history(trace.state, None))
    assert RolloutStore.self_score_runs(trace.state) == 0
