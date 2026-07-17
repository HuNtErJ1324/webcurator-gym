"""Modal-only harness tests (shared runtime behavior lives in test_docker_harness_runtime)."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
import verifiers.v1 as vf
from verifiers.v1.runtimes.modal import ModalConfig

from pretrain_curation_gym.config import CuratorEnvConfig, CuratorTaskConfig
from pretrain_curation_gym.corpus import CuratedCorpus, SourceCorpus
from pretrain_curation_gym.environment import load_environment
from pretrain_curation_gym.models import CuratorConfig, ProxyStudentConfig
from pretrain_curation_gym.rollout_state import CuratorState
from pretrain_curation_gym.taskset import CuratorTaskset
from pretrain_curation_gym.config import CuratorTasksetConfig
from pretrain_curation_gym.trainer import RuntimeProxyTrainer, TrainerError


def _corpus() -> CuratedCorpus:
    return CuratedCorpus(
        sources=[SourceCorpus.from_iter("owner/data", None, 1.0, ["hello world " * 30])]
    )


class FakeRuntime:
    def __init__(self, result=None, runtime_type="modal") -> None:
        self.config = SimpleNamespace(type=runtime_type)
        self.result = result or SimpleNamespace(
            stdout="RESULT_JSON "
            + json.dumps(
                {
                    "loss": 2.5,
                    "accuracy": 0.4,
                    "flops": 1e9,
                    "tokens_trained": 1000,
                }
            ),
            stderr="",
            exit_code=0,
        )
        self.stop_calls = 0

    @property
    def type(self) -> str:
        return self.config.type

    async def write(self, path: str, data: bytes) -> None:
        pass

    async def run(self, argv: list[str], env: dict[str, str]):
        return self.result

    async def stop(self) -> None:
        self.stop_calls += 1


@pytest.mark.asyncio
async def test_swallowed_wait_for_cancellation_is_reraised(monkeypatch):
    real_wait_for = asyncio.wait_for

    async def swallow_at_completion(awaitable, timeout):
        result = await real_wait_for(awaitable, timeout)
        asyncio.current_task().cancel()
        return result

    monkeypatch.setattr(
        "pretrain_curation_gym.trainer.asyncio.wait_for",
        swallow_at_completion,
    )
    runtime = FakeRuntime()

    with pytest.raises(asyncio.CancelledError):
        await RuntimeProxyTrainer().train_and_eval(
            _corpus(),
            ProxyStudentConfig(),
            runtime=runtime,
        )

    assert runtime.stop_calls == 1


def test_load_environment_preserves_native_modal_config(monkeypatch):
    monkeypatch.setenv("MODAL_TOKEN_ID", "test-id")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "test-secret")

    runtime_config = ModalConfig(
        image="registry.example/curator:gpu",
        workdir="/workspace",
        gpu="A100-80GB",
        cpu=8.0,
        memory=32.0,
        disk=40.0,
    )
    env = load_environment(
        CuratorEnvConfig(
            taskset=CuratorTasksetConfig(
                task=CuratorTaskConfig(
                    curator=CuratorConfig(
                        use_real_trainer=True,
                        proxy_student=ProxyStudentConfig(timeout_minutes=45),
                    )
                )
            ),
            harness=vf.HarnessConfig(runtime=runtime_config),
            timeout=vf.TimeoutConfig(scoring=3240.0),
        )
    )

    runtime = env.harness.config.runtime
    assert env.harness.config.env == {}
    assert isinstance(runtime, ModalConfig)
    assert runtime.image == "registry.example/curator:gpu"
    assert runtime.workdir == "/workspace"
    assert runtime.gpu == "A100-80GB"
    assert runtime.cpu == 8.0
    assert runtime.memory == 32.0
    assert runtime.disk == 40.0
    assert env.config.timeout.scoring == 3240.0


def test_load_environment_requires_modal_credentials(monkeypatch):
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)

    with pytest.raises(ValueError, match="MODAL_TOKEN_ID, MODAL_TOKEN_SECRET"):
        load_environment(
            CuratorEnvConfig(
                taskset=CuratorTasksetConfig(
                    task=CuratorTaskConfig(curator=CuratorConfig(use_real_trainer=True))
                ),
                harness=vf.HarnessConfig(runtime=ModalConfig()),
            )
        )


def test_modal_timeout_cannot_exceed_runtime_lifetime(monkeypatch):
    monkeypatch.setenv("MODAL_TOKEN_ID", "test-id")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "test-secret")
    with pytest.raises(ValueError, match="Modal 24h sandbox maximum"):
        load_environment(
            CuratorEnvConfig(
                taskset=CuratorTasksetConfig(
                    task=CuratorTaskConfig(
                        curator=CuratorConfig(
                            use_real_trainer=True,
                            proxy_student=ProxyStudentConfig(timeout_minutes=1441),
                        )
                    )
                ),
                harness=vf.HarnessConfig(runtime=ModalConfig()),
            )
        )


@pytest.mark.asyncio
async def test_taskset_setup_rejects_modal_trainer_on_other_runtime(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-token")
    taskset = CuratorTaskset(
        CuratorTasksetConfig(
            id="pretrain-curation-gym",
            task=CuratorTaskConfig(curator=CuratorConfig(use_real_trainer=True)),
        )
    )

    task = taskset.load()[0]
    trace = vf.Trace(
        task=vf.TraceTask(type=type(task).__name__, data=task.data),
        state=CuratorState(),
    )
    with pytest.raises(TrainerError, match="Docker or Modal harness runtime"):
        await task.setup(trace, FakeRuntime(runtime_type="subprocess"))
