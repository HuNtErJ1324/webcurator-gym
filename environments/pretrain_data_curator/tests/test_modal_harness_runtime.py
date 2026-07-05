from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
import verifiers.v1 as vf
from pydantic import ValidationError
from verifiers.v1.runtimes.modal import ModalConfig

from pretrain_data_curator.corpus import CuratedCorpus, SourceCorpus
from pretrain_data_curator.modal_backend import ModalProxyTrainer, _modal_gpu_for
from pretrain_data_curator.models import (
    CuratorConfig,
    Manifest,
    ProxyStudentConfig,
    Source,
)
from pretrain_data_curator.pretrain_data_curator import load_environment
from pretrain_data_curator.rewards import CuratorScorer
from pretrain_data_curator.rollout_state import CuratorState, RolloutStore
from pretrain_data_curator.tasks import build_tasks
from pretrain_data_curator.taskset import CuratorTaskset, CuratorTasksetConfig
from pretrain_data_curator.trainer import TrainResult, TrainerError


def _corpus() -> CuratedCorpus:
    return CuratedCorpus(
        sources=[SourceCorpus.from_iter("owner/data", None, 1.0, ["hello world " * 30])]
    )


def _result(
    *,
    loss: float = 2.5,
    accuracy: float = 0.4,
    flops: float = 1e9,
    tokens_trained: int = 1000,
) -> str:
    return "RESULT_JSON " + json.dumps(
        {
            "loss": loss,
            "accuracy": accuracy,
            "flops": flops,
            "tokens_trained": tokens_trained,
        }
    )


class FakeRuntime:
    def __init__(self, result=None, runtime_type="modal") -> None:
        self.config = SimpleNamespace(type=runtime_type)
        self.result = result or SimpleNamespace(
            stdout=_result(), stderr="", exit_code=0
        )
        self.files: dict[str, bytes] = {}
        self.commands: list[tuple[list[str], dict[str, str]]] = []
        self.stop_calls = 0

    @property
    def type(self) -> str:
        return self.config.type

    async def write(self, path: str, data: bytes) -> None:
        self.files[path] = data

    async def run(self, argv: list[str], env: dict[str, str]):
        self.commands.append((argv, env))
        return self.result

    async def stop(self) -> None:
        self.stop_calls += 1


@pytest.mark.asyncio
async def test_modal_trainer_writes_and_runs_on_supplied_harness_runtime():
    runtime = FakeRuntime()

    result = await ModalProxyTrainer().train_and_eval(
        _corpus(), ProxyStudentConfig(runtime_backend="modal"), runtime=runtime
    )

    assert result.backend == "modal"
    assert result.loss == 2.5
    assert set(runtime.files) == {
        "/workspace/corpus.txt",
        "/workspace/config.json",
        "/workspace/train.py",
    }
    assert runtime.commands == [(["python", "/workspace/train.py"], {})]
    assert runtime.stop_calls == 0


class _CorpusBuilder:
    async def materialize(self, manifest, state, *, runtime=None):
        return _corpus()


class _Leakage:
    def score(self, documents, val_set=None):
        return SimpleNamespace(
            as_dict=lambda: {
                "leakage_score": 0.0,
                "num_contaminated_matches": 0,
            }
        )


class _RecordingTrainer:
    def __init__(self) -> None:
        self.runtime = None

    async def train_and_eval(self, corpus, config, *, runtime=None):
        self.runtime = runtime
        return TrainResult(
            loss=2.0,
            accuracy=0.5,
            flops=10.0,
            tokens_trained=10,
            backend="modal",
        )


@pytest.mark.asyncio
async def test_modal_reward_threads_runtime_through_scoring_chain():
    config = CuratorConfig(
        use_real_trainer=True,
        proxy_student=ProxyStudentConfig(runtime_backend="modal"),
    )
    trainer = _RecordingTrainer()
    taskset = CuratorTaskset(
        CuratorTasksetConfig(
            id="test",
            use_real_trainer=True,
            proxy_student={"runtime_backend": "modal"},
        )
    )
    taskset._scorer = CuratorScorer(config, _CorpusBuilder(), trainer, _Leakage())
    state = CuratorState()
    RolloutStore.set_manifest(
        state, Manifest(sources=[Source(dataset_id="owner/data")])
    )
    RolloutStore.set_finalized(state, True)
    trace = vf.Trace(task=build_tasks("2024-12-31", 1_000_000)[0], state=state)
    runtime = FakeRuntime()

    await taskset.score(trace, runtime)

    assert trace.rewards["perf_reward"] > 0.0
    assert trainer.runtime is runtime


@pytest.mark.asyncio
async def test_modal_training_timeout_stops_runtime_without_hanging(monkeypatch):
    class BlockingRuntime(FakeRuntime):
        async def run(self, argv, env):
            await asyncio.Event().wait()

    monkeypatch.setattr(
        ProxyStudentConfig,
        "effective_timeout_minutes",
        property(lambda self: 0.0005),
    )
    runtime = BlockingRuntime()

    with pytest.raises(TrainerError, match="timed out"):
        await asyncio.wait_for(
            ModalProxyTrainer().train_and_eval(
                _corpus(),
                ProxyStudentConfig(runtime_backend="modal"),
                runtime=runtime,
            ),
            timeout=1.0,
        )

    assert runtime.stop_calls == 1


@pytest.mark.asyncio
async def test_cancelled_modal_training_stops_runtime():
    class BlockingRuntime(FakeRuntime):
        async def run(self, argv, env):
            await asyncio.Event().wait()

    runtime = BlockingRuntime()
    task = asyncio.create_task(
        ModalProxyTrainer().train_and_eval(
            _corpus(),
            ProxyStudentConfig(runtime_backend="modal"),
            runtime=runtime,
        )
    )
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert runtime.stop_calls == 1


@pytest.mark.asyncio
async def test_swallowed_wait_for_cancellation_is_reraised(monkeypatch):
    real_wait_for = asyncio.wait_for

    async def swallow_at_completion(awaitable, timeout):
        result = await real_wait_for(awaitable, timeout)
        asyncio.current_task().cancel()
        return result

    monkeypatch.setattr(
        "pretrain_data_curator.modal_backend.asyncio.wait_for",
        swallow_at_completion,
    )
    runtime = FakeRuntime()

    with pytest.raises(asyncio.CancelledError):
        await ModalProxyTrainer().train_and_eval(
            _corpus(),
            ProxyStudentConfig(runtime_backend="modal"),
            runtime=runtime,
        )

    assert runtime.stop_calls == 1
    assert runtime.commands == []


@pytest.mark.asyncio
async def test_modal_training_semaphore_bounds_runtime_commands():
    active = 0
    max_active = 0

    class ConcurrentRuntime(FakeRuntime):
        async def run(self, argv, env):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            try:
                await asyncio.sleep(0.02)
                return self.result
            finally:
                active -= 1

    config = ProxyStudentConfig(runtime_backend="modal")
    await asyncio.gather(
        ModalProxyTrainer(concurrency_limit=1).train_and_eval(
            _corpus(), config, runtime=ConcurrentRuntime()
        ),
        ModalProxyTrainer(concurrency_limit=1).train_and_eval(
            _corpus(), config, runtime=ConcurrentRuntime()
        ),
    )

    assert max_active == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("result", "message"),
    [
        (
            SimpleNamespace(stdout="", stderr="CUDA exploded", exit_code=7),
            "exited with code 7",
        ),
        (
            SimpleNamespace(
                stdout="RESULT_JSON {not-json", stderr="parse context", exit_code=0
            ),
            "malformed RESULT_JSON",
        ),
        (
            SimpleNamespace(stdout="ordinary output", stderr="missing", exit_code=0),
            "no RESULT_JSON marker",
        ),
    ],
)
async def test_modal_runtime_failures_raise_clear_trainer_errors(result, message):
    runtime = FakeRuntime(result)

    with pytest.raises(TrainerError, match=message) as excinfo:
        await ModalProxyTrainer().train_and_eval(
            _corpus(),
            ProxyStudentConfig(runtime_backend="modal"),
            runtime=runtime,
        )

    assert runtime.stop_calls == 1
    assert excinfo.value.stderr_tail in {"CUDA exploded", "parse context", "missing"}


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
            id="pretrain-data-curator",
            use_real_trainer=True,
            proxy_student={
                "runtime_backend": "modal",
                "docker_image": "registry.example/curator:gpu",
                "modal_gpu": "H200",
                "timeout_minutes": 45,
            },
        )
    )

    task = taskset.load_tasks()[0]

    assert task.image == "registry.example/curator:gpu"
    assert task.workdir == "/workspace"
    assert task.resources.gpu == "H200"
    assert task.resources.cpu == 4.0
    assert task.resources.memory == 16.0
    assert task.timeout.scoring == 45 * 60 + 540


def test_modal_timeout_cannot_exceed_runtime_lifetime():
    with pytest.raises(ValidationError, match="Modal 24h sandbox maximum"):
        ProxyStudentConfig(runtime_backend="modal", timeout_minutes=1441)


@pytest.mark.asyncio
async def test_taskset_setup_rejects_modal_trainer_on_other_runtime(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-token")
    taskset = CuratorTaskset(
        CuratorTasksetConfig(
            id="pretrain-data-curator",
            use_real_trainer=True,
            proxy_student={"runtime_backend": "modal"},
        )
    )

    with pytest.raises(TrainerError, match="Docker or Modal harness runtime"):
        await taskset.setup(
            taskset.load_tasks()[0],
            FakeRuntime(runtime_type="subprocess"),
        )
