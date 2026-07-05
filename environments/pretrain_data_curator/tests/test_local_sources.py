from __future__ import annotations

import json
import shlex
from pathlib import Path
from types import SimpleNamespace

import pytest
import verifiers.v1 as vf
from pydantic import ValidationError
from verifiers.v1 import graph

from pretrain_data_curator.corpus import (
    CorpusBuilder,
    _iter_local_documents,
)
from pretrain_data_curator.hf_access import estimate_tokens
from pretrain_data_curator.models import (
    CuratorConfig,
    FilterSpec,
    Manifest,
    Source,
)
from pretrain_data_curator.pretrain_data_curator import load_environment
from pretrain_data_curator.rollout_state import CuratorState, RolloutStore
from pretrain_data_curator.tasks import build_tasks
from pretrain_data_curator.taskset import (
    CuratorTaskset,
    CuratorTasksetConfig,
    _coerce_source,
)
from pretrain_data_curator.val_set import NANOGPT_VAL_DATASET_ID


class FakeClient:
    def __init__(self, docs: dict[str, list[str]] | None = None) -> None:
        self.docs = docs or {}
        self.calls: list[tuple[str, int]] = []

    def sample_documents(
        self,
        dataset_id: str,
        config: str | None,
        split: str,
        text_field: str | None,
        n: int,
    ) -> list[str]:
        self.calls.append((dataset_id, n))
        return self.docs.get(dataset_id, [])[:n]


class FakeRuntime:
    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files
        self.commands: list[tuple[list[str], dict[str, str]]] = []
        self.read_calls: list[str] = []

    async def run(self, argv: list[str], env: dict[str, str]):
        self.commands.append((argv, env))
        command = shlex.split(argv[2])
        path = command[-1]
        data = self.files.get(path)
        if data is None:
            return SimpleNamespace(exit_code=1, stdout="", stderr="not found")
        if command[:2] == ["wc", "-c"]:
            return SimpleNamespace(exit_code=0, stdout=f"{len(data)}\n", stderr="")
        if command[:2] == ["head", "-c"]:
            cap = int(command[2])
            return SimpleNamespace(
                exit_code=0,
                stdout=data[:cap].decode("utf-8", errors="replace"),
                stderr="",
            )
        raise AssertionError(f"unexpected command: {command}")

    async def read(self, path: str) -> bytes:
        self.read_calls.append(path)
        raise FileNotFoundError(path)


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_hf_source_schema_is_backward_compatible():
    source = Source(dataset_id="owner/data")

    assert source.kind == "hf"
    assert source.local_path is None
    assert source.local_format == "auto"
    assert source.model_dump() == {
        "dataset_id": "owner/data",
        "config": None,
        "split": "train",
        "text_field": None,
        "weight": 1.0,
        "filters": [],
        "sampling": {"max_docs": None, "max_tokens": None},
    }


def test_local_source_requires_non_empty_path():
    with pytest.raises(ValidationError, match="non-empty local_path"):
        Source(dataset_id="local:test", kind="local")


@pytest.mark.parametrize("path", ["/etc/passwd", "../secret", "data/../../secret"])
def test_local_source_rejects_absolute_and_parent_paths(path: str):
    with pytest.raises(ValidationError, match="workspace-relative"):
        Source(dataset_id="local:test", kind="local", local_path=path)


@pytest.mark.parametrize(
    "path",
    [
        "corpus.txt",
        "nested/config.json",
        "train.py",
        "outputs/val.bin",
        ".vf_hf_cost.jsonl",
    ],
)
def test_local_source_rejects_reserved_runtime_files(path: str):
    with pytest.raises(ValidationError, match="reserved file"):
        Source(dataset_id="local:test", kind="local", local_path=path)


def test_coerce_source_infers_local_kind_and_label():
    source = _coerce_source(
        {
            "local_path": "data/docs.jsonl",
            "local_format": "jsonl",
            "text_field": "body",
            "weight": 2,
        }
    )

    assert source is not None
    assert source.dataset_id == "local:data/docs.jsonl"
    assert source.kind == "local"
    assert source.local_path == "data/docs.jsonl"
    assert source.local_format == "jsonl"
    assert source.text_field == "body"
    assert source.weight == 2


@pytest.mark.parametrize(
    "raw",
    [
        {"kind": "local", "id": "bad/missing-path"},
        {"local_path": "../escape.jsonl"},
        {"local_path": "train.py"},
        {"local_path": "data/docs.jsonl", "local_format": "parquet"},
    ],
)
def test_coerce_source_drops_malformed_local_sources(raw):
    assert _coerce_source(raw) is None


def test_iter_local_jsonl_supports_fields_strings_autodetect_and_bad_lines(tmp_path):
    path = _write(
        tmp_path / "docs.jsonl",
        "\n".join(
            [
                json.dumps({"body": "explicit field"}),
                json.dumps("bare string"),
                json.dumps({"text": "auto detected"}),
                "malformed raw line",
                json.dumps({"number": 3}),
                "",
            ]
        ),
    )

    explicit = list(_iter_local_documents(path, "jsonl", "body"))
    automatic = list(_iter_local_documents(path, "auto", None))

    assert explicit == [
        "explicit field",
        "bare string",
        "auto detected",
        "malformed raw line",
    ]
    assert automatic == [
        "explicit field",
        "bare string",
        "auto detected",
        "malformed raw line",
    ]


def test_iter_local_txt_splits_only_on_blank_lines(tmp_path):
    path = _write(
        tmp_path / "docs.txt",
        "first line\ncontinues\n\n  second document  \n \nthird",
    )

    assert list(_iter_local_documents(path, "auto", None)) == [
        "first line\ncontinues",
        "second document",
        "third",
    ]


@pytest.mark.parametrize("fmt,suffix", [("jsonl", ".jsonl"), ("txt", ".txt")])
def test_iter_local_empty_file_has_no_documents(tmp_path, fmt: str, suffix: str):
    path = _write(tmp_path / f"empty{suffix}", "")
    assert list(_iter_local_documents(path, fmt, None)) == []


@pytest.mark.asyncio
async def test_fetch_local_docs_caches_and_bills_once():
    content = (
        json.dumps({"text": "alpha document"})
        + "\n"
        + json.dumps({"text": "beta document"})
        + "\n"
    ).encode()
    runtime = FakeRuntime({"data/docs.jsonl": content})
    state = CuratorState()
    source = Source(
        dataset_id="local:fixture",
        kind="local",
        local_path="data/docs.jsonl",
        local_format="jsonl",
    )
    builder = CorpusBuilder(FakeClient(), max_local_source_bytes=1024)

    docs, error = await builder.fetch_local_docs(state, source, runtime)
    cached, cached_error = await builder.fetch_local_docs(state, source, runtime)

    assert error is None and cached_error is None
    assert docs == cached == ["alpha document", "beta document"]
    assert len(runtime.commands) == 2
    assert runtime.commands[0][0][2] == "wc -c < data/docs.jsonl"
    assert runtime.commands[1][0][2] == "head -c 1024 -- data/docs.jsonl"
    assert runtime.read_calls == []
    ledger = RolloutStore.ledger(state)
    assert ledger.code_calls == 1
    assert ledger.tokens == sum(estimate_tokens(doc) for doc in docs)
    assert RolloutStore.local_source_count(state) == 1
    assert RolloutStore.local_source_bytes(state) == len(content)
    assert not RolloutStore.local_source_truncated(state)
    assert len(state.doc_cache) == 1


@pytest.mark.asyncio
async def test_fetch_local_docs_caps_transfer_before_parsing():
    content = b"first document\n\nsecond document is beyond the cap"
    runtime = FakeRuntime({"data/docs.txt": content})
    state = CuratorState()
    source = Source(
        dataset_id="local:fixture",
        kind="local",
        local_path="data/docs.txt",
        local_format="txt",
    )
    builder = CorpusBuilder(FakeClient(), max_local_source_bytes=16)

    docs, error = await builder.fetch_local_docs(state, source, runtime)

    assert error is None
    assert docs == ["first document"]
    assert runtime.commands[1][0][2] == "head -c 16 -- data/docs.txt"
    assert len(runtime.commands[1][0][2]) < len(content)
    assert RolloutStore.local_source_bytes(state) == 16
    assert RolloutStore.local_source_truncated(state)


@pytest.mark.asyncio
async def test_fetch_local_docs_without_runtime_is_typed_empty_source():
    state = CuratorState()
    source = Source(
        dataset_id="local:fixture",
        kind="local",
        local_path="data/docs.txt",
    )
    builder = CorpusBuilder(FakeClient())

    docs, error = await builder.fetch_local_docs(state, source, None)

    assert docs == []
    assert error is not None and error["error_kind"] == "local_no_runtime"
    assert state.tool_errors == {"local_no_runtime": 1}


@pytest.mark.asyncio
async def test_fetch_local_docs_respects_disabled_config():
    state = CuratorState()
    source = Source(
        dataset_id="local:fixture",
        kind="local",
        local_path="data/docs.txt",
    )
    builder = CorpusBuilder(FakeClient(), allow_local_sources=False)

    docs, error = await builder.fetch_local_docs(
        state, source, FakeRuntime({"data/docs.txt": b"text"})
    )

    assert docs == []
    assert error is not None and error["error_kind"] == "local_disabled"


@pytest.mark.asyncio
async def test_materialize_mixes_hf_and_local_with_weighted_allocation():
    doc = "a" * 25
    client = FakeClient({"hub/data": [doc] * 8})
    local_content = "\n".join(json.dumps(doc) for _ in range(4)).encode()
    runtime = FakeRuntime({"data/local.jsonl": local_content})
    builder = CorpusBuilder(client, sample_docs_per_source=50)
    manifest = Manifest(
        token_budget=3_000,
        sources=[
            Source(dataset_id="hub/data", weight=2),
            Source(
                dataset_id="local:data/local.jsonl",
                kind="local",
                local_path="data/local.jsonl",
                local_format="jsonl",
                weight=1,
            ),
        ],
    )

    corpus = await builder.materialize(
        manifest, CuratorState(), runtime=runtime
    )

    assert corpus.sources[0].doc_count == 8
    assert corpus.sources[1].doc_count == 4
    assert corpus.sources[0].tokens == 2 * corpus.sources[1].tokens


@pytest.mark.asyncio
async def test_materialize_backfills_from_cached_local_surplus():
    docs = ["a" * 1_200, "b" * 1_200]
    local_content = "\n".join(json.dumps(doc) for doc in docs).encode()
    client = FakeClient({"filtered/out": ["x" * 1_200, "y" * 1_200]})
    runtime = FakeRuntime({"data/local.jsonl": local_content})
    builder = CorpusBuilder(client, sample_docs_per_source=10)
    manifest = Manifest(
        token_budget=1_000,
        sources=[
            Source(
                dataset_id="filtered/out",
                filters=[FilterSpec(kind="min_chars", params={"value": 2_000})],
            ),
            Source(
                dataset_id="local:surplus",
                kind="local",
                local_path="data/local.jsonl",
                local_format="jsonl",
            ),
        ],
    )
    state = CuratorState()

    corpus = await builder.materialize(manifest, state, runtime=runtime)

    assert corpus.sources[0].doc_count == 0
    assert corpus.sources[1].doc_count == 2
    assert corpus.total_tokens == 600
    assert len(runtime.commands) == 2
    assert len(state.doc_cache) == 2


@pytest.mark.asyncio
async def test_materialize_runtime_none_keeps_hf_source_unchanged():
    client = FakeClient({"hub/data": ["hub document"]})
    builder = CorpusBuilder(client)
    state = CuratorState()
    manifest = Manifest(
        sources=[
            Source(dataset_id="hub/data"),
            Source(
                dataset_id="local:missing",
                kind="local",
                local_path="data/missing.txt",
            ),
        ]
    )

    corpus = await builder.materialize(manifest, state)

    assert corpus.sources[0].documents == ["hub document"]
    assert corpus.sources[1].doc_count == 0
    assert state.tool_errors == {"local_no_runtime": 1}


def _trace_with_command_and_manifest(
    task, state, command: str, final_manifest: str
) -> vf.Trace:
    trace = vf.Trace(task=task, state=state)
    conversation = [vf.SystemMessage(content="sys"), vf.UserMessage(content="go")]
    tool_call = vf.ToolCall(
        id="tc0", name="bash", arguments=json.dumps({"command": command})
    )
    graph.prepare_turn(trace, conversation).commit(
        vf.Response(
            id="r0",
            created=0,
            model="m",
            message=vf.AssistantMessage(content="", tool_calls=[tool_call]),
            finish_reason="tool_calls",
            usage=vf.Usage(prompt_tokens=1, completion_tokens=1),
        )
    )
    conversation.extend(
        [
            vf.AssistantMessage(content="", tool_calls=[tool_call]),
            vf.ToolMessage(tool_call_id="tc0", content="done"),
        ]
    )
    graph.prepare_turn(trace, conversation).commit(
        vf.Response(
            id="r1",
            created=0,
            model="m",
            message=vf.AssistantMessage(content=final_manifest),
            finish_reason="stop",
            usage=vf.Usage(prompt_tokens=1, completion_tokens=1),
        )
    )
    return trace


@pytest.mark.asyncio
async def test_finalize_detects_validation_repository_access():
    taskset = CuratorTaskset(CuratorTasksetConfig(id="test"))
    task = build_tasks("2024-12-31", 1_000)[0]
    state = CuratorState()
    trace = _trace_with_command_and_manifest(
        task,
        state,
        f"hf download {NANOGPT_VAL_DATASET_ID} fineweb_val_000000.bin",
        '```json\n{"sources":[{"id":"good/science"}]}\n```',
    )

    await taskset.finalize(task, trace, FakeRuntime({}))

    assert RolloutStore.val_set_access(state)


@pytest.mark.asyncio
async def test_local_provenance_metrics_read_rollout_state():
    taskset = CuratorTaskset(CuratorTasksetConfig(id="test"))
    task = build_tasks("2024-12-31", 1_000)[0]
    state = CuratorState()
    RolloutStore.add_local_source(state, bytes_pulled=123, truncated=True)
    RolloutStore.set_val_set_access(state, True)
    trace = vf.Trace(task=task, state=state)
    taskset._scoring_cache[trace.id] = {}

    assert await taskset.local_source_count(trace) == 1.0
    assert await taskset.local_source_bytes(trace) == 123.0
    assert await taskset.local_source_truncated(trace) == 1.0
    assert await taskset.val_set_access(trace) == 1.0


def test_local_configuration_is_validated_and_plumbed():
    config = CuratorConfig(max_local_source_bytes=1)
    assert config.allow_local_sources
    assert config.max_local_source_bytes == 1
    with pytest.raises(ValidationError):
        CuratorConfig(max_local_source_bytes=0)

    env = load_environment(
        allow_local_sources=False,
        max_local_source_bytes=4096,
    )
    assert env.taskset.config.allow_local_sources is False
    assert env.taskset.curator.allow_local_sources is False
    assert env.taskset.curator.max_local_source_bytes == 4096
    assert env.env_args["allow_local_sources"] is False
    assert env.env_args["max_local_source_bytes"] == 4096


def test_initial_prompt_discloses_local_source_safety_and_cost():
    prompt = build_tasks("2024-12-31", 1_000_000)[0].prompt
    assert '"kind": "hf"' in prompt
    assert '"kind": "local"' in prompt
    assert "workspace-relative" in prompt
    assert "no leading `/` or `..`" in prompt
    assert "increases the cost penalty" in prompt
    assert "Fabricated data" in prompt
