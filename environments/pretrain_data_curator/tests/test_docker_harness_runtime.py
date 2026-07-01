from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
import verifiers.v1 as vf

from pretrain_data_curator.corpus import CuratedCorpus, SourceCorpus
from pretrain_data_curator.docker_backend import HarnessRuntimeProxyTrainer
from pretrain_data_curator.modal_backend import ModalProxyTrainer
from pretrain_data_curator.models import CuratorConfig, Manifest, ProxyStudentConfig, Source
from pretrain_data_curator.rewards import CuratorScorer
from pretrain_data_curator.rollout_state import CuratorState, RolloutStore
from pretrain_data_curator.tasks import build_tasks
from pretrain_data_curator.taskset import CuratorTaskset, CuratorTasksetConfig
from pretrain_data_curator.trainer import HeuristicProxyTrainer, TrainerError, TrainResult


def _corpus() -> CuratedCorpus:
    return CuratedCorpus(
        sources=[SourceCorpus("owner/data", None, 1.0, ["hello world " * 30])]
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
    def __init__(self, result=None, runtime_type="docker") -> None:
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
async def test_harness_runtime_trainer_writes_and_runs_on_supplied_runtime():
    runtime = FakeRuntime()
    trainer = HarnessRuntimeProxyTrainer()

    result = await trainer.train_and_eval(
        _corpus(), ProxyStudentConfig(trainer_backend="docker"), runtime=runtime
    )

    assert result.backend == "docker"
    assert result.loss == 2.5
    assert set(runtime.files) == {
        "/workspace/corpus.txt",
        "/workspace/config.json",
        "/workspace/train.py",
    }
    assert runtime.commands == [(["python", "/workspace/train.py"], {})]
    # The rollout owns the successful runtime until all scoring finishes.
    assert runtime.stop_calls == 0


class _CorpusBuilder:
    async def materialize(self, manifest, state):
        return _corpus()


class _Leakage:
    def score(self, documents):
        return SimpleNamespace(
            as_dict=lambda: {
                "exact": 0.0,
                "fuzzy": 0.0,
                "semantic": 0.0,
                "overall": 0.0,
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
            backend="docker",
        )


@pytest.mark.asyncio
async def test_reward_threads_injected_runtime_through_scoring_chain():
    config = CuratorConfig(
        use_real_trainer=True,
        proxy_student=ProxyStudentConfig(trainer_backend="docker"),
    )
    trainer = _RecordingTrainer()
    taskset = CuratorTaskset(
        CuratorTasksetConfig(
            id="test",
            use_real_trainer=True,
            proxy_student={"trainer_backend": "docker"},
        )
    )
    taskset._scorer = CuratorScorer(
        config, _CorpusBuilder(), trainer, _Leakage()
    )
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
async def test_training_timeout_stops_runtime_without_hanging(monkeypatch):
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
            HarnessRuntimeProxyTrainer().train_and_eval(
                _corpus(),
                ProxyStudentConfig(trainer_backend="docker"),
                runtime=runtime,
            ),
            timeout=1.0,
        )

    assert runtime.stop_calls == 1


@pytest.mark.asyncio
async def test_cancelled_training_stops_runtime():
    class BlockingRuntime(FakeRuntime):
        async def run(self, argv, env):
            await asyncio.Event().wait()

    runtime = BlockingRuntime()
    task = asyncio.create_task(
        HarnessRuntimeProxyTrainer().train_and_eval(
            _corpus(),
            ProxyStudentConfig(trainer_backend="docker"),
            runtime=runtime,
        )
    )
    await asyncio.sleep(0)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert runtime.stop_calls == 1


@pytest.mark.asyncio
async def test_training_semaphore_bounds_runtime_commands():
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

    config = ProxyStudentConfig(trainer_backend="docker")
    await asyncio.gather(
        HarnessRuntimeProxyTrainer(concurrency_limit=1).train_and_eval(
            _corpus(), config, runtime=ConcurrentRuntime()
        ),
        HarnessRuntimeProxyTrainer(concurrency_limit=1).train_and_eval(
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
async def test_runtime_failures_raise_clear_trainer_errors(result, message):
    runtime = FakeRuntime(result)

    with pytest.raises(TrainerError, match=message) as excinfo:
        await HarnessRuntimeProxyTrainer().train_and_eval(
            _corpus(),
            ProxyStudentConfig(trainer_backend="docker"),
            runtime=runtime,
        )

    assert runtime.stop_calls == 1
    assert excinfo.value.stderr_tail in {"CUDA exploded", "parse context", "missing"}


def test_docker_and_modal_backend_selection_remain_distinct():
    docker_taskset = CuratorTaskset(
        CuratorTasksetConfig(
            id="docker",
            use_real_trainer=True,
            proxy_student={"trainer_backend": "docker"},
        )
    )
    modal_taskset = CuratorTaskset(
        CuratorTasksetConfig(
            id="modal",
            use_real_trainer=True,
            proxy_student={"trainer_backend": "modal"},
        )
    )

    assert isinstance(
        docker_taskset._build_real_trainer(), HarnessRuntimeProxyTrainer
    )
    assert isinstance(modal_taskset._build_real_trainer(), ModalProxyTrainer)


def test_native_docker_tasks_declare_runtime_requirements_and_deadline():
    taskset = CuratorTaskset(
        CuratorTasksetConfig(
            id="pretrain-data-curator",
            use_real_trainer=True,
            proxy_student={
                "trainer_backend": "docker",
                "docker_image": "pretrain-data-curator:gpu",
                "gpu_count": 1,
                "timeout_minutes": 45,
            },
        )
    )

    task = taskset.load_tasks()[0]

    assert task.image == "pretrain-data-curator:gpu"
    assert task.workdir == "/workspace"
    assert task.resources.gpu == "1"
    assert task.resources.cpu == 4.0
    assert task.resources.memory == 16.0
    assert task.timeout.scoring == 45 * 60 + 540


def test_package_is_discoverable_as_a_native_v1_taskset():
    from verifiers.v1.loaders import taskset_class

    assert taskset_class("pretrain-data-curator") is CuratorTaskset


@pytest.mark.asyncio
async def test_taskset_setup_rejects_docker_trainer_on_subprocess_runtime():
    taskset = CuratorTaskset(
        CuratorTasksetConfig(
            id="pretrain-data-curator",
            use_real_trainer=True,
            proxy_student={"trainer_backend": "docker"},
        )
    )

    with pytest.raises(TrainerError, match="--harness.runtime.type docker"):
        await taskset.setup(
            taskset.load_tasks()[0],
            FakeRuntime(runtime_type="subprocess"),
        )


class _FailingCorpusBuilder:
    async def materialize(self, manifest, state):
        RolloutStore.record_tool_error(state, "bad_config")
        RolloutStore.set_external_failure(state, True)
        return CuratedCorpus(sources=[])


@pytest.mark.asyncio
async def test_external_failure_metrics_wait_for_materialization():
    config = CuratorConfig()
    taskset = CuratorTaskset(CuratorTasksetConfig(id="test"))
    taskset._scorer = CuratorScorer(
        config,
        _FailingCorpusBuilder(),
        HeuristicProxyTrainer(),
        _Leakage(),
    )
    state = CuratorState()
    RolloutStore.set_manifest(
        state, Manifest(sources=[Source(dataset_id="owner/data")])
    )
    RolloutStore.set_finalized(state, True)
    trace = vf.Trace(task=build_tasks("2024-12-31", 1_000_000)[0], state=state)

    await taskset.score(trace, None)

    assert trace.metrics["external_failure"] == 1.0
    assert trace.metrics["tool_error_count"] == 1.0
