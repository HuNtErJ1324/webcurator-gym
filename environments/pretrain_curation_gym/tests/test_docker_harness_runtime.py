from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
import verifiers.v1 as vf

import pretrain_curation_gym.trainer as trainer_module
from pretrain_curation_gym.corpus import CuratedCorpus, SourceCorpus
from pretrain_curation_gym.trainer import RuntimeProxyTrainer
from pretrain_curation_gym.models import (
    CuratorConfig,
    Manifest,
    ProxyStudentConfig,
    Source,
)
from pretrain_curation_gym.rewards import CuratorScorer
from pretrain_curation_gym.rollout_state import CuratorState, RolloutStore
from pretrain_curation_gym.taskset import CuratorTaskset, CuratorTasksetConfig
from pretrain_curation_gym.trainer import (
    HeuristicProxyTrainer,
    TrainerError,
    TrainResult,
)


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


TRACEBACK_STDOUT = """step 0 loss 10.2
Traceback (most recent call last):
  File "/workspace/train.py", line 217, in <module>
    main()
RuntimeError: loss is non-finite
"""
PIP_ROOT_WARNING = (
    "WARNING: Running pip as the 'root' user can result in broken permissions"
)


class FakeRuntime:
    def __init__(
        self, result=None, runtime_type="docker", *, hang_on_train: bool = False
    ) -> None:
        self.config = SimpleNamespace(type=runtime_type)
        self.result = result or SimpleNamespace(
            stdout=_result(), stderr="", exit_code=0
        )
        self.files: dict[str, bytes] = {}
        self.commands: list[tuple[list[str], dict[str, str]]] = []
        self.stop_calls = 0
        self.hang_on_train = hang_on_train

    @property
    def type(self) -> str:
        return self.config.type

    async def write(self, path: str, data: bytes) -> None:
        self.files[path] = data

    async def read(self, path: str) -> bytes:
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]

    async def run(self, argv: list[str], env: dict[str, str]):
        self.commands.append((argv, env))
        if self.hang_on_train and argv == ["python", "/workspace/train.py"]:
            await asyncio.Event().wait()
        if argv[:2] == ["python", "-c"] and "CGROUP_JSON" in (
            argv[2] if len(argv) > 2 else ""
        ):
            return SimpleNamespace(
                stdout='CGROUP_JSON {"error": null, "events": {}, "memory_max": null}\n',
                stderr="",
                exit_code=0,
            )
        return self.result

    async def stop(self) -> None:
        self.stop_calls += 1

    @property
    def train_commands(self) -> list[tuple[list[str], dict[str, str]]]:
        return [
            cmd for cmd in self.commands if cmd[0] == ["python", "/workspace/train.py"]
        ]


@pytest.mark.asyncio
async def test_runtime_trainer_offloads_corpus_join(monkeypatch):
    real_to_thread = asyncio.to_thread
    offloaded: list[tuple[str, tuple[object, ...]]] = []

    async def spy(fn, *args, **kwargs):
        offloaded.append((getattr(fn, "__name__", repr(fn)), args))
        return await real_to_thread(fn, *args, **kwargs)

    monkeypatch.setattr(trainer_module.asyncio, "to_thread", spy)

    await RuntimeProxyTrainer().train_and_eval(
        _corpus(), ProxyStudentConfig(runtime_backend="docker"), runtime=FakeRuntime()
    )

    assert any(name == "joined_text" for name, _args in offloaded)


@pytest.mark.asyncio
async def test_harness_runtime_trainer_writes_and_runs_on_supplied_runtime():
    runtime = FakeRuntime()
    trainer = RuntimeProxyTrainer()

    result = await trainer.train_and_eval(
        _corpus(), ProxyStudentConfig(runtime_backend="docker"), runtime=runtime
    )

    assert result.backend == "docker"
    assert result.loss == 2.5
    assert set(runtime.files) == {
        "/workspace/corpus.txt",
        "/workspace/config.json",
        "/workspace/train.py",
        "/workspace/train_stdout.log",
        "/workspace/train_stderr.log",
        "/workspace/train_stderr_redirect.log",
        "/workspace/train_oom_diagnostics.json",
    }
    assert runtime.train_commands == [(["python", "/workspace/train.py"], {})]
    assert any(
        cmd[0][:2] == ["python", "-c"] and "memory.events" in cmd[0][2]
        for cmd in runtime.commands
    )
    # The rollout owns the successful runtime until all scoring finishes.
    assert runtime.stop_calls == 0
    assert runtime.files["/workspace/train_stdout.log"] == _result().encode()
    assert runtime.files["/workspace/train_stderr.log"] == b""
    assert runtime.files["/workspace/train_stderr_redirect.log"] == b""
    diag = json.loads(runtime.files["/workspace/train_oom_diagnostics.json"])
    assert "kill_class" in diag
    assert "cgroup_events_before" in diag


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
            backend="docker",
        )


@pytest.mark.asyncio
async def test_reward_threads_injected_runtime_through_scoring_chain():
    config = CuratorConfig(
        use_real_trainer=True,
        proxy_student=ProxyStudentConfig(runtime_backend="docker"),
    )
    trainer = _RecordingTrainer()
    taskset = CuratorTaskset(
        CuratorTasksetConfig(
            id="test",
            use_real_trainer=True,
            proxy_student={"runtime_backend": "docker"},
        )
    )
    task = taskset.load()[0]
    task._scorer = CuratorScorer(config, _CorpusBuilder(), trainer, _Leakage())
    state = CuratorState()
    RolloutStore.set_manifest(
        state, Manifest(sources=[Source(dataset_id="owner/data")])
    )
    RolloutStore.set_finalized(state, True)
    trace = vf.Trace(
        task=vf.TraceTask(type=type(task).__name__, data=task.data), state=state
    )
    runtime = FakeRuntime()

    await task.score(trace, runtime)

    assert trace.rewards["perf_reward"] > 0.0
    assert trainer.runtime is runtime


@pytest.mark.asyncio
async def test_training_timeout_stops_runtime_without_hanging(monkeypatch):
    monkeypatch.setattr(
        ProxyStudentConfig,
        "effective_timeout_minutes",
        property(lambda self: 0.0005),
    )
    runtime = FakeRuntime(hang_on_train=True)

    with pytest.raises(TrainerError, match="timed out"):
        await asyncio.wait_for(
            RuntimeProxyTrainer().train_and_eval(
                _corpus(),
                ProxyStudentConfig(runtime_backend="docker"),
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
        RuntimeProxyTrainer().train_and_eval(
            _corpus(),
            ProxyStudentConfig(runtime_backend="docker"),
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

    config = ProxyStudentConfig(runtime_backend="docker")
    await asyncio.gather(
        RuntimeProxyTrainer(concurrency_limit=1).train_and_eval(
            _corpus(), config, runtime=ConcurrentRuntime()
        ),
        RuntimeProxyTrainer(concurrency_limit=1).train_and_eval(
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
        await RuntimeProxyTrainer().train_and_eval(
            _corpus(),
            ProxyStudentConfig(runtime_backend="docker"),
            runtime=runtime,
        )

    assert runtime.stop_calls == 1
    assert "--- stdout tail ---" in excinfo.value.stderr_tail
    assert "--- stderr tail ---" in excinfo.value.stderr_tail
    assert any(
        expected in excinfo.value.stderr_tail
        for expected in ("CUDA exploded", "parse context", "missing")
    )


@pytest.mark.asyncio
async def test_training_crash_surfaces_stdout_traceback_and_persists_full_logs():
    result = SimpleNamespace(
        stdout=f"installing dependencies\n{TRACEBACK_STDOUT}",
        stderr=PIP_ROOT_WARNING,
        exit_code=1,
    )
    runtime = FakeRuntime(result)

    with pytest.raises(TrainerError, match="exited with code 1") as excinfo:
        await RuntimeProxyTrainer().train_and_eval(
            _corpus(),
            ProxyStudentConfig(runtime_backend="docker"),
            runtime=runtime,
        )

    diagnostic = excinfo.value.stderr_tail
    assert 'File "/workspace/train.py", line 217, in <module>' in diagnostic
    assert "RuntimeError: loss is non-finite" in diagnostic
    assert PIP_ROOT_WARNING in diagnostic
    assert diagnostic.index("RuntimeError: loss is non-finite") < diagnostic.index(
        PIP_ROOT_WARNING
    )
    assert runtime.files["/workspace/train_stdout.log"] == result.stdout.encode()
    assert runtime.files["/workspace/train_stderr.log"] == result.stderr.encode()
    assert runtime.files["/workspace/train_stderr_redirect.log"] == b""


@pytest.mark.asyncio
async def test_training_crash_surfaces_redirected_stderr_file():
    """Trainer redirects sys.stderr to /workspace/stderr.txt; harness must read it."""
    result = SimpleNamespace(stdout="step 0", stderr="", exit_code=1)
    runtime = FakeRuntime(result)
    runtime.files["/workspace/stderr.txt"] = (
        b"CUDA out of memory. Tried to allocate 9.20 GiB\n"
        b"RuntimeError: CUDA error: out of memory\n"
    )

    with pytest.raises(TrainerError, match="exited with code 1") as excinfo:
        await RuntimeProxyTrainer().train_and_eval(
            _corpus(),
            ProxyStudentConfig(runtime_backend="docker"),
            runtime=runtime,
        )

    assert "CUDA out of memory" in excinfo.value.stderr_tail
    assert "out of memory" in runtime.files["/workspace/train_stderr.log"].decode()
    assert (
        b"CUDA out of memory" in runtime.files["/workspace/train_stderr_redirect.log"]
    )


@pytest.mark.asyncio
async def test_no_result_json_surfaces_stdout_tail():
    result = SimpleNamespace(
        stdout="loaded corpus\nlast progress line before crash",
        stderr=PIP_ROOT_WARNING,
        exit_code=0,
    )
    runtime = FakeRuntime(result)

    with pytest.raises(TrainerError, match="no RESULT_JSON marker") as excinfo:
        await RuntimeProxyTrainer().train_and_eval(
            _corpus(),
            ProxyStudentConfig(runtime_backend="docker"),
            runtime=runtime,
        )

    assert "last progress line before crash" in excinfo.value.stderr_tail
    assert PIP_ROOT_WARNING in excinfo.value.stderr_tail
    assert runtime.files["/workspace/train_stdout.log"] == result.stdout.encode()


@pytest.mark.asyncio
async def test_trainer_error_str_preserves_training_traceback():
    config = CuratorConfig(
        use_real_trainer=True,
        proxy_student=ProxyStudentConfig(runtime_backend="docker"),
    )
    taskset = CuratorTaskset(
        CuratorTasksetConfig(
            id="test",
            use_real_trainer=True,
            proxy_student={"runtime_backend": "docker"},
        )
    )
    task = taskset.load()[0]
    task._scorer = CuratorScorer(
        config, _CorpusBuilder(), RuntimeProxyTrainer(), _Leakage()
    )
    state = CuratorState()
    RolloutStore.set_manifest(
        state, Manifest(sources=[Source(dataset_id="owner/data")])
    )
    RolloutStore.set_finalized(state, True)
    trace = vf.Trace(
        task=vf.TraceTask(type=type(task).__name__, data=task.data), state=state
    )
    runtime = FakeRuntime(
        SimpleNamespace(
            stdout=TRACEBACK_STDOUT,
            stderr=PIP_ROOT_WARNING,
            exit_code=1,
        )
    )

    await task.score(trace, runtime)
    err = state.trainer_error or ""

    assert 'File "/workspace/train.py", line 217, in <module>' in err
    assert "RuntimeError: loss is non-finite" in err
    assert trace.metrics["trainer_error_msg"] == 1.0


@pytest.mark.asyncio
async def test_build_real_trainer_uses_one_runtime_implementation():
    # One RuntimeProxyTrainer handles both live runtime types; Docker adds its
    # cgroup diagnostics while Modal uses the common write/run/read path.
    taskset = CuratorTaskset(CuratorTasksetConfig(id="test", use_real_trainer=True))
    trainer = taskset.load()[0].scorer().trainer
    assert isinstance(trainer, RuntimeProxyTrainer)

    docker_result = await trainer.train_and_eval(
        _corpus(), ProxyStudentConfig(), runtime=FakeRuntime(runtime_type="docker")
    )
    assert docker_result.backend == "docker"

    modal_result = await trainer.train_and_eval(
        _corpus(), ProxyStudentConfig(), runtime=FakeRuntime(runtime_type="modal")
    )
    assert modal_result.backend == "modal"

    with pytest.raises(TrainerError, match="Docker or Modal harness runtime"):
        await trainer.train_and_eval(
            _corpus(),
            ProxyStudentConfig(),
            runtime=FakeRuntime(runtime_type="subprocess"),
        )


def test_native_tasks_do_not_duplicate_harness_runtime_requirements():
    taskset = CuratorTaskset(
        CuratorTasksetConfig(
            id="pretrain-curation-gym",
            use_real_trainer=True,
            proxy_student={
                "runtime_backend": "docker",
                "docker_image": "pretrain-curation-gym:gpu",
                "gpu_count": 1,
                "timeout_minutes": 45,
            },
        )
    )

    task = taskset.load()[0]

    assert task.data.image is None
    assert task.data.workdir is None
    assert task.data.resources == vf.TaskResources()
    assert task.data.timeout == vf.TaskTimeout()


def test_package_is_discoverable_as_a_native_v1_taskset():
    from verifiers.v1.loaders import taskset_class

    assert taskset_class("pretrain-curation-gym") is CuratorTaskset


@pytest.mark.asyncio
async def test_taskset_setup_rejects_docker_trainer_on_subprocess_runtime(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-token")
    taskset = CuratorTaskset(
        CuratorTasksetConfig(
            id="pretrain-curation-gym",
            use_real_trainer=True,
            proxy_student={"runtime_backend": "docker"},
        )
    )

    task = taskset.load()[0]
    trace = vf.Trace(
        task=vf.TraceTask(type=type(task).__name__, data=task.data),
        state=CuratorState(),
    )
    with pytest.raises(TrainerError, match="Docker or Modal harness runtime"):
        await task.setup(trace, FakeRuntime(runtime_type="subprocess"))


class _FailingCorpusBuilder:
    async def materialize(self, manifest, state, *, runtime=None):
        RolloutStore.record_tool_error(state, "bad_config")
        RolloutStore.set_external_failure(state, True)
        return CuratedCorpus(sources=[])


@pytest.mark.asyncio
async def test_external_failure_metrics_wait_for_materialization():
    config = CuratorConfig()
    taskset = CuratorTaskset(CuratorTasksetConfig(id="test"))
    task = taskset.load()[0]
    task._scorer = CuratorScorer(
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
    trace = vf.Trace(
        task=vf.TraceTask(type=type(task).__name__, data=task.data), state=state
    )

    await task.score(trace, None)

    assert trace.metrics["external_failure"] == 1.0
    assert trace.metrics["tool_error_count"] == 1.0
