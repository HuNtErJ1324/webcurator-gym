from __future__ import annotations

import json

import pytest
import verifiers.v1 as vf
from verifiers.v1.decorators import discover_decorated

from pretrain_curation_gym import (
    CuratorEnvConfig,
    CuratorTaskConfig,
    CuratorTaskset,
    CuratorTasksetConfig,
    load_environment,
)
from pretrain_curation_gym.corpus import CorpusBuilder
from pretrain_curation_gym.manifest import ManifestParser
from pretrain_curation_gym.models import (
    MANIFEST_PROVENANCE_WORKSPACE_FILE,
    Manifest,
    Source,
)
from pretrain_curation_gym.state import CuratorState
from pretrain_curation_gym.rewards import CuratorScorer
from pretrain_curation_gym.trainer import HeuristicProxyTrainer


def test_taskset_is_thin_v1_composition() -> None:
    taskset = CuratorTaskset(CuratorTasksetConfig())
    [task] = taskset.load()

    assert task.data.system_prompt is None
    assert "Sole curation budget: 1000000 tokens" in task.data.prompt
    assert "Turn limit: 64 model turns" in task.data.prompt
    assert "python turns.py" in task.data.prompt
    assert "to `manifest.json`" in task.data.prompt
    assert "/workspace/manifest.json" not in task.data.prompt
    assert len(discover_decorated(task, "reward")) == 1
    assert len(discover_decorated(task, "metric")) == 3
    assert [stop.__name__ for stop in discover_decorated(task, "stop")] == [
        "refresh_turn_state"
    ]


def test_max_turns_has_one_framework_owned_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "test-token")
    config = CuratorEnvConfig(
        max_turns=7,
        taskset=CuratorTasksetConfig(
            task=CuratorTaskConfig(curator={"token_budget": 12_345})
        ),
    )
    environment = load_environment(config)
    [task] = environment.taskset.load()

    assert config.max_turns == 7
    assert task.data.token_budget == 12_345
    assert task.data.max_turns == 7
    assert "Turn limit: 7 model turns" in task.data.prompt
    assert not hasattr(config.taskset.task.curator, "max_turns")
    assert not hasattr(config.taskset, "max_turns")


def test_environment_loader_composes_native_v1(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "test-token")
    environment = load_environment(CuratorEnvConfig())

    assert isinstance(environment, vf.Environment)
    assert environment.config.taskset.id == "pretrain-curation-gym"
    assert environment.config.max_turns == 64
    assert environment.taskset.load()[0].data.max_turns == 64
    assert "HF_TOKEN" in environment.config.harness.forward_env


def test_manifest_parser_preserves_tolerant_contract() -> None:
    parser = ManifestParser()
    manifest = parser.parse(
        'draft {"sources": []}\n```json\n'
        + json.dumps(
            {
                "token_budget": 10_000,
                "sources": [
                    {
                        "id": "owner/data",
                        "weight": "2.5",
                        "filters": [
                            {"kind": "min_chars", "params": {"value": "200"}},
                            {"kind": "unknown", "params": {}},
                        ],
                    }
                ],
            }
        )
        + "\n```",
        default_token_budget=1_000_000,
    )

    assert manifest is not None
    assert manifest.token_budget == 10_000
    assert manifest.sources[0].dataset_id == "owner/data"
    assert manifest.sources[0].weight == 2.5
    assert [item.kind for item in manifest.sources[0].filters] == ["min_chars"]


@pytest.mark.asyncio
async def test_one_keyed_reward_pass_records_all_diagnostics() -> None:
    [task] = CuratorTaskset(CuratorTasksetConfig()).load()
    state = CuratorState(
        manifest=Manifest(sources=[Source(dataset_id="owner/data")]).model_dump(),
        manifest_finalized=True,
        manifest_provenance=MANIFEST_PROVENANCE_WORKSPACE_FILE,
    )

    class FakeScorer:
        calls = 0

        async def compute_scoring(self, state, runtime):
            self.calls += 1
            return {
                "perf": 0.75,
                "leakage": {
                    "leakage_score": 0.1,
                    "num_contaminated_matches": 2,
                },
                "decon_error": 0.0,
                "val_screen_skipped": 0.0,
                "loss": 4.0,
                "accuracy": 0.2,
                "flops": 123.0,
                "tokens": 900,
                "num_sources": 1,
                "budget_fill_ratio": 0.9,
                "perf_vs_baseline": 0.3,
            }

    scorer = FakeScorer()
    task._scorer = scorer
    trace = vf.Trace(
        task=vf.TraceTask(type=type(task).__name__, data=task.data),
        state=state,
    )

    await task.score(trace)

    assert scorer.calls == 1
    assert trace.rewards == {"perf_reward": 0.75, "leakage_penalty": -0.1}
    assert trace.metrics["perf_loss"] == 4.0
    assert trace.metrics["corpus_tokens"] == 900.0
    assert trace.metrics["finalized"] == 1.0
    assert trace.metrics["hf_cli_calls"] == 0.0
    assert len(trace.metrics) == 29


@pytest.mark.asyncio
async def test_hf_cli_invocations_are_framework_metrics() -> None:
    [task] = CuratorTaskset(CuratorTasksetConfig()).load()
    trace = type(
        "ActivityTrace",
        (),
        {
            "num_turns": 1,
            "assistant_messages": [
                vf.AssistantMessage(
                    content="",
                    tool_calls=[
                        vf.ToolCall(
                            id="tc-1",
                            name="bash",
                            arguments=json.dumps(
                                {
                                    "command": "hf datasets info owner/data; "
                                    "hf download owner/data --repo-type dataset"
                                }
                            ),
                        )
                    ],
                )
            ],
        },
    )()

    assert await task.hf_cli_calls(trace) == 2.0


@pytest.mark.asyncio
async def test_real_materialization_and_heuristic_scoring_path() -> None:
    [task] = CuratorTaskset(
        CuratorTasksetConfig(task=CuratorTaskConfig(curator={"token_budget": 500}))
    ).load()

    class FakeDatasetClient:
        def sample_documents(self, dataset_id, config, split, text_field, n):
            return [
                "A clean, substantial encyclopedia paragraph about physics and matter."
                * 8,
                "A second diverse document about mathematics, proofs, and algorithms."
                * 8,
            ]

    task._scorer = CuratorScorer(
        task.curator,
        CorpusBuilder(FakeDatasetClient()),
        HeuristicProxyTrainer(),
        decon_detector=None,
        val_loader=None,
        screen_val_set=False,
    )
    manifest = Manifest(
        token_budget=500,
        sources=[Source(dataset_id="owner/clean-data")],
    )
    state = CuratorState(
        manifest=manifest.model_dump(),
        manifest_finalized=True,
        manifest_provenance=MANIFEST_PROVENANCE_WORKSPACE_FILE,
    )
    trace = vf.Trace(
        task=vf.TraceTask(type=type(task).__name__, data=task.data),
        state=state,
    )

    await task.score(trace)

    assert trace.metrics["corpus_tokens"] > 0
    assert trace.metrics["num_sources"] == 1
    assert trace.rewards["perf_reward"] != 0
    assert state.scratch_dir is None
