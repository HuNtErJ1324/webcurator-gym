"""Modal-only harness tests (shared runtime behavior lives in test_docker_harness_runtime)."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
import verifiers.v1 as vf
from pydantic import ValidationError
from verifiers.v1.runtimes.modal import ModalConfig

from pretrain_curation_gym.corpus import CuratedCorpus, SourceCorpus
from pretrain_curation_gym.runtime_config import _modal_gpu_for
from pretrain_curation_gym.models import ProxyStudentConfig
from pretrain_curation_gym.pretrain_curation_gym import load_environment
from pretrain_curation_gym.rollout_state import CuratorState
from pretrain_curation_gym.taskset import CuratorTaskset, CuratorTasksetConfig
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
            ProxyStudentConfig(runtime_backend="modal"),
            runtime=runtime,
        )

    assert runtime.stop_calls == 1


def test_load_environment_uses_modal_config_and_gpu_mapping(monkeypatch):
    monkeypatch.setenv("MODAL_TOKEN_ID", "test-id")
    monkeypatch.setenv("MODAL_TOKEN_SECRET", "test-secret")

    env = load_environment(
        use_real_trainer=True,
        proxy_student={
            "runtime_backend": "modal",
            "modal_gpu": "A100",
            "cpu_cores": 8,
            "memory_gb": 32,
            "disk_size_gb": 40,
            "timeout_minutes": 45,
        },
    )

    runtime = env.harness.config.runtime
    assert env.harness.config.env == {}
    assert isinstance(runtime, ModalConfig)
    assert runtime.image == "pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime"
    assert runtime.workdir == "/workspace"
    assert runtime.gpu == "A100-80GB"
    assert runtime.cpu == 8.0
    assert runtime.memory == 32.0
    assert runtime.disk == 40.0
    assert env.config.timeout.scoring == 45 * 60 + 540
    assert _modal_gpu_for("H100") == "H100"
    assert _modal_gpu_for("H200") == "H200"
    assert _modal_gpu_for("unknown") == "L4"


def test_load_environment_requires_modal_credentials(monkeypatch):
    monkeypatch.delenv("MODAL_TOKEN_ID", raising=False)
    monkeypatch.delenv("MODAL_TOKEN_SECRET", raising=False)

    with pytest.raises(ValueError, match="MODAL_TOKEN_ID, MODAL_TOKEN_SECRET"):
        load_environment(
            use_real_trainer=True,
            proxy_student={"runtime_backend": "modal"},
        )


def test_native_modal_tasks_declare_runtime_requirements_and_deadline():
    taskset = CuratorTaskset(
        CuratorTasksetConfig(
            id="pretrain-curation-gym",
            use_real_trainer=True,
            proxy_student={
                "runtime_backend": "modal",
                "docker_image": "registry.example/curator:gpu",
                "modal_gpu": "H200",
                "timeout_minutes": 45,
            },
        )
    )

    task = taskset.load()[0]

    assert task.data.image == "registry.example/curator:gpu"
    assert task.data.workdir == "/workspace"
    assert task.data.resources.gpu == "H200"
    assert task.data.resources.cpu == 4.0
    assert task.data.resources.memory == 16.0
    assert task.data.timeout.scoring == 45 * 60 + 540


def test_modal_timeout_cannot_exceed_runtime_lifetime():
    with pytest.raises(ValidationError, match="Modal 24h sandbox maximum"):
        ProxyStudentConfig(runtime_backend="modal", timeout_minutes=1441)


@pytest.mark.asyncio
async def test_taskset_setup_rejects_modal_trainer_on_other_runtime(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-token")
    taskset = CuratorTaskset(
        CuratorTasksetConfig(
            id="pretrain-curation-gym",
            use_real_trainer=True,
            proxy_student={"runtime_backend": "modal"},
        )
    )

    task = taskset.load()[0]
    trace = vf.Trace(
        task=vf.TraceTask(type=type(task).__name__, data=task.data),
        state=CuratorState(),
    )
    with pytest.raises(TrainerError, match="Docker or Modal harness runtime"):
        await task.setup(trace, FakeRuntime(runtime_type="subprocess"))
