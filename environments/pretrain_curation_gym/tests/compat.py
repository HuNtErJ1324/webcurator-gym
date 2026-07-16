"""Test-only adapters for exercising predecessor regressions on the v1 rewrite.

These adapters deliberately do not ship in the environment package.  They map
the predecessor's flat construction helpers onto the rewrite's nested v1
configuration and direct ``CuratorState`` API, allowing unchanged behavioral
tests to verify the new implementation without polluting its public surface.
"""

from __future__ import annotations

import json
import math
from types import ModuleType
from typing import Any

import verifiers.v1 as vf

from pretrain_curation_gym.config import (
    CuratorEnvConfig,
    CuratorTaskConfig,
    CuratorTasksetConfig as NativeTasksetConfig,
)
from pretrain_curation_gym.environment import (
    load_environment as native_load_environment,
)
from pretrain_curation_gym.leakage import LeakageScores
from pretrain_curation_gym.models import CuratorConfig, Manifest
from pretrain_curation_gym.rewards import CuratorScorer
from pretrain_curation_gym.state import CuratorState
from pretrain_curation_gym.task import CuratorTask
from pretrain_curation_gym.tasks import CuratorTaskData


def legacy_taskset_config(id: str = "test", **values: Any) -> NativeTasksetConfig:
    """Translate the predecessor's flat TasksetConfig into v1-owned leaves."""
    task_fields = set(CuratorTaskConfig.model_fields) - {"curator", "judges"}
    curator_fields = set(CuratorConfig.model_fields)
    task_values = {key: values.pop(key) for key in list(values) if key in task_fields}
    curator_values = {
        key: values.pop(key) for key in list(values) if key in curator_fields
    }
    if values:
        unknown = ", ".join(sorted(values))
        raise TypeError(f"unknown legacy taskset option(s): {unknown}")
    return NativeTasksetConfig(
        id=id,
        task=CuratorTaskConfig(
            curator=CuratorConfig(**curator_values),
            **task_values,
        ),
    )


def legacy_load_environment(**values: Any):
    """Translate predecessor loader kwargs to the concrete v1 EnvConfig."""
    harness_id = values.pop("harness_id", "default")
    taskset = legacy_taskset_config(id="pretrain-curation-gym", **values)
    config = CuratorEnvConfig(
        taskset=taskset,
        harness=vf.HarnessConfig(id=harness_id),
        max_turns=taskset.task.curator.max_turns,
    )
    environment = native_load_environment(config)
    environment.env_args = dict(values)
    return environment


def legacy_build_tasks(
    cutoff_date: str,
    token_budget: int,
    *,
    manifest_filename: str = "manifest.json",
    allow_local_sources: bool = True,
    max_turns: int = 64,
    alpha_perf: float = 1.0,
    lambda_leakage: float = 1.0,
    perf_target_loss: float = 3.28,
    config: vf.TaskConfig | None = None,
) -> list[CuratorTask]:
    task_config = (
        config
        if isinstance(config, CuratorTaskConfig)
        else CuratorTaskConfig(
            manifest_filename=manifest_filename,
            curator=CuratorConfig(
                cutoff_date=cutoff_date,
                token_budget=token_budget,
                allow_local_sources=allow_local_sources,
                max_turns=max_turns,
                alpha_perf=alpha_perf,
                lambda_leakage=lambda_leakage,
                perf_target_loss=perf_target_loss,
            ),
        )
    )
    return [CuratorTask(CuratorTaskData.from_config(task_config), task_config)]


class RolloutStore:
    """Predecessor accessor names mapped to the rewrite's direct state API."""

    manifest = staticmethod(lambda state: state.parsed_manifest)
    is_finalized = staticmethod(lambda state: state.manifest_finalized)
    set_finalized = staticmethod(
        lambda state, value: setattr(state, "manifest_finalized", value)
    )
    manifest_provenance = staticmethod(lambda state: state.manifest_provenance)
    set_manifest_provenance = staticmethod(
        lambda state, value: setattr(state, "manifest_provenance", value)
    )
    scratch_dir = staticmethod(lambda state: state.workspace())
    cleanup = staticmethod(lambda state: state.cleanup())
    cached_docs = staticmethod(lambda state, key: state.cached_documents(key))
    store_docs = staticmethod(lambda state, key, docs: state.cache_documents(key, docs))
    tool_error_count = staticmethod(lambda state: state.tool_error_count)
    local_source_count = staticmethod(lambda state: state.local_source_count)
    local_source_bytes = staticmethod(lambda state: state.local_source_bytes)
    local_source_truncated = staticmethod(lambda state: state.local_source_truncated)
    val_set_access = staticmethod(lambda state: state.val_set_access)
    set_val_set_access = staticmethod(
        lambda state, value: setattr(state, "val_set_access", bool(value))
    )
    self_score_runs = staticmethod(lambda state: state.self_score_runs)
    self_score_ok_runs = staticmethod(lambda state: state.self_score_ok_runs)
    self_score_first_reward = staticmethod(lambda state: state.self_score_first_reward)
    self_score_best_reward = staticmethod(lambda state: state.self_score_best_reward)
    self_score_last_reward = staticmethod(lambda state: state.self_score_last_reward)
    set_external_failure = staticmethod(
        lambda state, value=True: setattr(state, "external_failure", bool(value))
    )
    has_external_failure = staticmethod(lambda state: state.external_failure)
    set_trainer_error = staticmethod(
        lambda state, value: setattr(state, "trainer_error", value)
    )
    trainer_error = staticmethod(lambda state: state.trainer_error)

    @staticmethod
    def set_manifest(state: CuratorState, manifest: Manifest) -> None:
        state.manifest = manifest.model_dump()

    @staticmethod
    def set_materialization_stats(state: CuratorState, **values: Any) -> None:
        state.set_materialization_stats(**values)

    @staticmethod
    def record_tool_error(state: CuratorState, kind: str) -> None:
        state.record_error(kind, external=False)

    @staticmethod
    def add_local_source(
        state: CuratorState, *, bytes_pulled: int, truncated: bool
    ) -> None:
        state.record_local_source(bytes_pulled=bytes_pulled, truncated=truncated)

    @staticmethod
    def set_self_score_summary(
        state: CuratorState, *, runs: int, ok_rewards: list[float]
    ) -> None:
        state.set_self_score_summary(runs=runs, rewards=ok_rewards)


class NoOpLeakageDetector:
    def score(self, docs, val_set=None) -> LeakageScores:
        return LeakageScores(0.0, 0, ())


def bind_fast_scorer(
    task: CuratorTask,
    *,
    corpus_builder,
    trainer,
    leakage_detector,
) -> CuratorScorer:
    scorer = CuratorScorer(
        task.config.curator,
        corpus_builder,
        trainer,
        leakage_detector,
        val_loader=None,
        screen_val_set=False,
    )
    task._scorer = scorer
    return scorer


def compatibility_module(name: str) -> ModuleType:
    module = ModuleType(name)
    module.CuratorState = CuratorState
    module.RolloutStore = RolloutStore
    return module


def parse_self_score_history(text: str) -> tuple[int, list[float]]:
    runs = 0
    rewards: list[float] = []
    for line in text.splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        runs += 1
        reward = record.get("reward")
        if (
            record.get("ok") is True
            and isinstance(reward, (int, float))
            and not isinstance(reward, bool)
            and math.isfinite(reward)
        ):
            rewards.append(float(reward))
    return runs, rewards
