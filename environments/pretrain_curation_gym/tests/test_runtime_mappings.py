"""Deterministic tests for the shared runtime/resource derivation helper and the
async offload of SourceCorpus materialization."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
import verifiers.v1 as vf
from verifiers.v1.runtimes.modal import ModalConfig

from pretrain_curation_gym.corpus import (
    SourceCorpus,
    _iter_local_documents,
    _materialize_local_docs,
)
from pretrain_curation_gym.models import ProxyStudentConfig
from pretrain_curation_gym.runtime_config import (
    derive_env_harness_runtime,
    derive_task_runtime_updates,
    derive_trainer_resources,
)


# --- async offload -----------------------------------------------------------


def test_from_docs_offload_preserves_documents_and_order():
    docs = [f"doc-{i}" for i in range(7)]

    async def run() -> SourceCorpus:
        return await asyncio.to_thread(
            SourceCorpus.from_docs,
            "ds",
            "cfg",
            1.0,
            list(docs),
        )

    corpus = asyncio.run(run())
    assert corpus.documents == docs
    assert corpus.doc_count == len(docs)


def test_from_docs_offload_gather_preserves_per_source_ordering_and_results():
    inputs = {
        "a": ["a1", "a2", "a3"],
        "b": ["b1"],
        "c": ["c1", "c2"],
        "d": [],
    }

    async def run() -> list[tuple[str, list[str], int]]:
        tasks = [
            asyncio.to_thread(
                SourceCorpus.from_docs,
                name,
                None,
                1.0,
                list(docs),
            )
            for name, docs in inputs.items()
        ]
        corpora = await asyncio.gather(*tasks)
        return [
            (corpus.dataset_id, corpus.documents, corpus.doc_count)
            for corpus in corpora
        ]

    results = asyncio.run(run())
    # asyncio.gather preserves the input order, and each corpus carries exactly
    # the documents it was given (including the empty source -> 0 docs).
    assert [name for name, _, _ in results] == list(inputs.keys())
    for (name, docs, count), original in zip(results, inputs.values(), strict=True):
        assert docs == original
        assert count == len(original)


@pytest.mark.parametrize("fmt", ["txt", "jsonl"])
def test_materialize_local_docs_offload_matches_sync_parse(fmt: str, tmp_path: Path):
    if fmt == "txt":
        raw = "first doc\n\nsecond doc\n\nthird doc\n"
    else:
        raw = "\n".join(f'{{"text": "doc-{i}"}}' for i in range(3))

    raw_path = tmp_path / f"local.{fmt}"

    async def run() -> list[str]:
        return await asyncio.to_thread(
            _materialize_local_docs,
            raw_path,
            raw,
            fmt,
            "text" if fmt == "jsonl" else None,
        )

    offloaded = asyncio.run(run())
    # The offloaded parse must equal the synchronous parse of the same file.
    raw_path.write_text(raw, encoding="utf-8")
    expected = list(
        _iter_local_documents(raw_path, fmt, "text" if fmt == "jsonl" else None)
    )
    assert offloaded == expected
    if fmt == "txt":
        assert offloaded == ["first doc", "second doc", "third doc"]
    else:
        assert offloaded == ["doc-0", "doc-1", "doc-2"]


# --- runtime / resource derivation mappings ----------------------------------


def _docker_ps(**overrides: object) -> ProxyStudentConfig:
    kwargs = {
        "runtime_backend": "docker",
        "docker_image": "img:docker",
        "gpu_count": 2,
        "cpu_cores": 8,
        "memory_gb": 32,
        "disk_size_gb": 50,
    }
    kwargs.update(overrides)
    return ProxyStudentConfig(**kwargs)


def _modal_ps(**overrides: object) -> ProxyStudentConfig:
    kwargs = {
        "runtime_backend": "modal",
        "docker_image": "img:modal",
        "modal_gpu": "H100",
        "cpu_cores": 4,
        "memory_gb": 16,
        "disk_size_gb": 20,
    }
    kwargs.update(overrides)
    return ProxyStudentConfig(**kwargs)


def test_derive_trainer_resources_docker_gpu_and_modal_gpu():
    docker = derive_trainer_resources(_docker_ps(), backend="docker")
    assert docker["gpu"] == "2"
    assert docker["image"] == "img:docker"
    assert docker["workdir"] == "/workspace"
    assert docker["cpu"] == 8.0
    assert docker["memory"] == 32.0
    assert docker["disk"] == 50.0

    # Zero gpu_count maps to None on docker (no GPU requested).
    docker0 = derive_trainer_resources(_docker_ps(gpu_count=0), backend="docker")
    assert docker0["gpu"] is None

    modal = derive_trainer_resources(_modal_ps(), backend="modal")
    assert modal["gpu"] == "H100"  # mapped through _modal_gpu_for
    assert modal["cpu"] == 4.0

    # Unknown modal_gpu falls back to L4.
    modal_default = derive_trainer_resources(_modal_ps(modal_gpu="T4"), backend="modal")
    assert modal_default["gpu"] == "L4"


def test_derive_env_harness_runtime_docker_mapping():
    runtime, timeout = derive_env_harness_runtime(_docker_ps(), use_real_trainer=True)
    assert isinstance(runtime, vf.DockerConfig)
    assert runtime.image == "img:docker"
    assert runtime.gpu == "2"
    assert runtime.cpu == 8.0
    assert runtime.memory == 32.0
    assert runtime.disk == 50.0
    assert isinstance(timeout, vf.TimeoutConfig)
    assert timeout.scoring == _docker_ps().effective_scoring_timeout_seconds


def test_derive_env_harness_runtime_modal_mapping():
    runtime, timeout = derive_env_harness_runtime(_modal_ps(), use_real_trainer=True)
    # ModalConfig import is lazy; the returned object must still be a ModalConfig.
    # (vf.RuntimeConfig is a Union type, so it cannot be used with isinstance.)
    assert isinstance(runtime, ModalConfig)
    assert not isinstance(runtime, vf.DockerConfig)
    assert runtime.image == "img:modal"
    assert runtime.gpu == "H100"
    assert runtime.cpu == 4.0
    assert isinstance(timeout, vf.TimeoutConfig)
    assert timeout.scoring == _modal_ps().effective_scoring_timeout_seconds


def test_derive_env_harness_runtime_default_is_subprocess():
    runtime, timeout = derive_env_harness_runtime(
        ProxyStudentConfig(), use_real_trainer=False
    )
    assert isinstance(runtime, vf.SubprocessConfig)
    assert isinstance(timeout, vf.TimeoutConfig)
    # No backend selected: no extended scoring timeout is applied.
    assert timeout.scoring is None


def test_derive_task_runtime_updates_docker_mapping():
    updates = derive_task_runtime_updates(_docker_ps(), use_real_trainer=True)
    assert updates["image"] == "img:docker"
    assert updates["workdir"] == "/workspace"
    resources = updates["resources"]
    assert isinstance(resources, vf.TaskResources)
    assert resources.cpu == 8.0
    assert resources.memory == 32.0
    assert resources.gpu == "2"
    assert resources.disk == 50.0
    assert isinstance(updates["timeout"], vf.TaskTimeout)
    assert updates["timeout"].scoring == _docker_ps().effective_scoring_timeout_seconds


def test_derive_task_runtime_updates_modal_mapping():
    updates = derive_task_runtime_updates(_modal_ps(), use_real_trainer=True)
    assert updates["image"] == "img:modal"
    resources = updates["resources"]
    assert isinstance(resources, vf.TaskResources)
    assert resources.gpu == "H100"
    assert resources.cpu == 4.0
    assert isinstance(updates["timeout"], vf.TaskTimeout)
    assert updates["timeout"].scoring == _modal_ps().effective_scoring_timeout_seconds


def test_derive_task_runtime_updates_default_is_empty():
    assert (
        derive_task_runtime_updates(ProxyStudentConfig(), use_real_trainer=False) == {}
    )
    # A real trainer without a backend (None) also yields no updates.
    assert (
        derive_task_runtime_updates(ProxyStudentConfig(), use_real_trainer=True) == {}
    )
