"""Deterministic tests for the manifest-backed training-debug workflow.

These exercise the cache/skip logic (no re-curation on a cache hit), mismatch
rejection (stale/wrong bundles fail before training), explicit refresh, and the
exact corpus handoff to the real proxy-student trainer. All use local sources
only, so no Hugging Face network is touched.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import tiktoken
import torch

from pretrain_data_curator.debug_train import (
    ManifestMismatchError,
    TokenBudgetError,
    UnsafeOutputError,
    build_debug_config,
    load_manifest,
    materialize_bundle,
    prepare_training_data,
    resolve_corpus,
    train_debug,
    validate_output_dir,
    validate_token_budget,
)
from pretrain_data_curator.models import Manifest, Source


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


def _make_sources(base_dir, *, docs_a, docs_b):
    a_path = base_dir / "data" / "a.jsonl"
    b_path = base_dir / "data" / "b.txt"
    a_path.parent.mkdir(parents=True, exist_ok=True)
    _write(a_path, "\n".join(json.dumps({"text": d}) for d in docs_a) + "\n")
    _write(b_path, "\n\n".join(docs_b) + "\n")
    return [
        Source(
            dataset_id="local:data/a.jsonl",
            kind="local",
            local_path="data/a.jsonl",
            local_format="jsonl",
            text_field="text",
            weight=1,
        ),
        Source(
            dataset_id="local:data/b.txt",
            kind="local",
            local_path="data/b.txt",
            local_format="txt",
            weight=1,
        ),
    ]


def _make_manifest(
    base_dir, *, budget=10_000, docs_a=("alpha", "beta"), docs_b=("gamma", "delta")
):
    sources = _make_sources(base_dir, docs_a=docs_a, docs_b=docs_b)
    return Manifest(token_budget=budget, sources=sources)


class _Recorder:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.calls = 0

    def __call__(self, manifest, bundle_dir, **kwargs):
        self.calls += 1
        return materialize_bundle(
            manifest, bundle_dir, base_dir=self.base_dir, **kwargs
        )


@pytest.fixture
def base_dir(tmp_path):
    return tmp_path / "ws"


@pytest.fixture
def manifest(base_dir):
    return _make_manifest(base_dir)


def test_load_manifest_rejects_missing(tmp_path):
    with pytest.raises(Exception):
        load_manifest(tmp_path / "nope.json")


# --------------------------------------------------------------------------- #
# Hardening: explicit positive token-budget validation
# --------------------------------------------------------------------------- #
def test_validate_token_budget_rejects_non_positive():
    for bad in (0, -1, -100):
        with pytest.raises(TokenBudgetError):
            validate_token_budget(bad)


def test_validate_token_budget_accepts_positive():
    assert validate_token_budget(1) == 1
    assert validate_token_budget(20_000) == 20_000


@pytest.mark.asyncio
async def test_resolve_corpus_rejects_non_positive_budget_before_materialize(
    base_dir, tmp_path
):
    # Bypass schema gt=0 so the debug-layer gate is what rejects the budget.
    manifest = Manifest.model_construct(token_budget=0, sources=[])
    calls = {"n": 0}

    def boom(*args, **kwargs):
        calls["n"] += 1
        raise AssertionError("materialize must not run for an invalid budget")

    with pytest.raises(TokenBudgetError):
        await asyncio.to_thread(
            resolve_corpus,
            manifest,
            tmp_path / "bundle",
            materialize_fn=boom,
            base_dir=base_dir,
        )
    assert calls["n"] == 0


# --------------------------------------------------------------------------- #
# Hardening: reject unsafe/destructive output directories before any deletion
# --------------------------------------------------------------------------- #
def test_validate_output_dir_rejects_system_roots():
    for unsafe in (Path("/"), Path("/etc"), Path("/usr/bin"), Path("/var/log")):
        with pytest.raises(UnsafeOutputError):
            validate_output_dir(unsafe)


def test_validate_output_dir_rejects_cwd_and_ancestor(monkeypatch, tmp_path):
    nested = tmp_path / "proj"
    nested.mkdir()
    monkeypatch.chdir(nested)
    with pytest.raises(UnsafeOutputError):
        validate_output_dir(Path.cwd())
    # Ancestor of cwd would wipe the project tree.
    with pytest.raises(UnsafeOutputError):
        validate_output_dir(tmp_path)


def test_validate_output_dir_rejects_base_dir_collision(tmp_path):
    base = tmp_path / "ws"
    base.mkdir()
    out = base  # output == source root must be rejected
    with pytest.raises(UnsafeOutputError):
        validate_output_dir(out, base_dir=base)
    # output above the source root (ancestor) must also be rejected
    with pytest.raises(UnsafeOutputError):
        validate_output_dir(tmp_path, base_dir=base)


def test_validate_output_dir_allows_safe_subdir(tmp_path):
    out = tmp_path / "pdc-debug-out"
    assert validate_output_dir(out) == out.resolve()
    assert validate_output_dir(out, base_dir=tmp_path / "ws") == out.resolve()
    # A subdirectory under base_dir is allowed (defaults live under the workspace).
    under = tmp_path / "ws" / "pdc-debug-out"
    assert validate_output_dir(under, base_dir=tmp_path / "ws") == under.resolve()


@pytest.mark.asyncio
async def test_train_debug_rejects_unsafe_output_before_deletion(monkeypatch, tmp_path):
    corpus_path = _write(
        tmp_path / "corpus.txt",
        json.dumps({"format": "document-list-v1", "documents": ["alpha"]}),
    )
    config = build_debug_config(steps=3, block_size=16, batch_size=2)
    marker = tmp_path / "keep_me.txt"
    marker.write_text("alive", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    trained = {"n": 0}

    def fake_train(*args, **kwargs):
        trained["n"] += 1
        raise AssertionError("train_fn must not run for an unsafe output dir")

    # CWD is unsafe: rejection must happen before clearing marker or training.
    with pytest.raises(UnsafeOutputError):
        await asyncio.to_thread(
            train_debug, corpus_path, tmp_path, config, train_fn=fake_train
        )
    assert marker.exists()
    assert marker.read_text(encoding="utf-8") == "alive"
    assert trained["n"] == 0

    # Protected system path is also rejected before any side effect.
    with pytest.raises(UnsafeOutputError):
        await asyncio.to_thread(
            train_debug, corpus_path, Path("/etc"), config, train_fn=fake_train
        )
    assert trained["n"] == 0


@pytest.mark.asyncio
async def test_resolve_corpus_rejects_unsafe_bundle_before_materialize(
    base_dir, manifest, monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    marker = tmp_path / "keep_me.txt"
    marker.write_text("alive", encoding="utf-8")
    calls = {"n": 0}

    def boom(*args, **kwargs):
        calls["n"] += 1
        raise AssertionError("materialize must not run for an unsafe bundle dir")

    with pytest.raises(UnsafeOutputError):
        await asyncio.to_thread(
            resolve_corpus,
            manifest,
            tmp_path,  # cwd / base-dir ancestor hazard
            materialize_fn=boom,
            base_dir=base_dir,
        )
    assert calls["n"] == 0
    assert marker.exists()
    assert marker.read_text(encoding="utf-8") == "alive"


@pytest.mark.asyncio
async def test_train_debug_safe_output_is_cleared_then_written(
    base_dir, manifest, tmp_path
):
    bundle_dir = tmp_path / "bundle"
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "stale.json").write_text("old", encoding="utf-8")

    recorder = _Recorder(base_dir)
    cp, prov, _ = await asyncio.to_thread(
        resolve_corpus, manifest, bundle_dir, materialize_fn=recorder
    )

    def fake_train(build_model, train_data, val_data, **kwargs):
        return (3.0, 0.5, 0.0, int(len(train_data)), 1_000_000)

    result = await asyncio.to_thread(
        train_debug,
        cp,
        output_dir,
        build_debug_config(steps=3, block_size=16, batch_size=2),
        train_fn=fake_train,
    )
    # stale file cleared, fresh result written
    assert not (output_dir / "stale.json").exists()
    assert (output_dir / "result.json").is_file()
    assert result["loss"] == 3.0


@pytest.mark.asyncio
async def test_cache_hit_does_not_recurate(base_dir, manifest, tmp_path):
    bundle_dir = tmp_path / "bundle"
    recorder = _Recorder(base_dir)

    cp1, prov1, curated1 = await asyncio.to_thread(
        resolve_corpus, manifest, bundle_dir, materialize_fn=recorder
    )
    assert curated1 is True
    assert recorder.calls == 1

    cp2, prov2, curated2 = await asyncio.to_thread(
        resolve_corpus, manifest, bundle_dir, materialize_fn=recorder
    )
    assert curated2 is False
    assert recorder.calls == 1  # materialization was NOT re-run
    assert cp1.read_text() == cp2.read_text()
    assert prov1["manifest_digest"] == prov2["manifest_digest"]


@pytest.mark.asyncio
async def test_mismatch_rejected_before_training(base_dir, manifest, tmp_path):
    bundle_dir = tmp_path / "bundle"
    recorder = _Recorder(base_dir)
    await asyncio.to_thread(
        resolve_corpus, manifest, bundle_dir, materialize_fn=recorder
    )

    # A genuinely different manifest (different budget + source set) -> different
    # digest -> must be rejected before any curation/training (never silently reused).
    other = Manifest(
        token_budget=10_001,
        sources=[
            Source(
                dataset_id="local:data/a.jsonl",
                kind="local",
                local_path="data/a.jsonl",
                local_format="jsonl",
                text_field="text",
                weight=1,
            )
        ],
    )
    with pytest.raises(ManifestMismatchError):
        await asyncio.to_thread(
            resolve_corpus, other, bundle_dir, materialize_fn=_Recorder(base_dir)
        )


@pytest.mark.asyncio
async def test_expected_token_budget_mismatch_rejected(manifest, tmp_path):
    bundle_dir = tmp_path / "bundle"
    with pytest.raises(ManifestMismatchError):
        await asyncio.to_thread(
            resolve_corpus, manifest, bundle_dir, expected_token_budget=9_999
        )


@pytest.mark.asyncio
async def test_explicit_refresh_recurates(base_dir, manifest, tmp_path):
    bundle_dir = tmp_path / "bundle"
    recorder = _Recorder(base_dir)

    await asyncio.to_thread(
        resolve_corpus, manifest, bundle_dir, materialize_fn=recorder
    )
    assert recorder.calls == 1

    cp, prov, curated = await asyncio.to_thread(
        resolve_corpus, manifest, bundle_dir, refresh=True, materialize_fn=recorder
    )
    assert curated is True
    assert recorder.calls == 2  # --refresh forces re-curation

    # Changing the manifest and refreshing must succeed with the new digest.
    changed = _make_manifest(base_dir, budget=20_000, docs_a=("p", "q"), docs_b=("r",))
    cp2, prov2, curated2 = await asyncio.to_thread(
        resolve_corpus, changed, bundle_dir, refresh=True, materialize_fn=recorder
    )
    assert curated2 is True
    assert recorder.calls == 3
    assert prov2["manifest_digest"] != prov["manifest_digest"]


@pytest.mark.asyncio
async def test_corpus_handoff_to_trainer_is_exact(base_dir, manifest, tmp_path):
    bundle_dir = tmp_path / "bundle"
    output_dir = tmp_path / "out"
    recorder = _Recorder(base_dir)
    cp, prov, _ = await asyncio.to_thread(
        resolve_corpus, manifest, bundle_dir, materialize_fn=recorder
    )

    recorded = {}

    def fake_train(build_model, train_data, val_data, **kwargs):
        recorded["train"] = train_data
        recorded["val"] = val_data
        recorded["document_ranges"] = kwargs["document_ranges"]
        recorded["device"] = kwargs["device"]
        return (3.0, 0.5, 0.0, int(len(train_data)), 1_000_000)

    config = build_debug_config(steps=10, block_size=64, batch_size=4)
    result = await asyncio.to_thread(
        train_debug, cp, output_dir, config, train_fn=fake_train
    )

    # The trainer received exactly the tokens of the bundle's corpus.txt.
    expected_train, expected_val, expected_ranges = prepare_training_data(
        cp, val_fraction=config.val_fraction
    )
    assert torch.equal(recorded["train"], expected_train)
    assert torch.equal(recorded["val"], expected_val)
    assert recorded["document_ranges"] == expected_ranges
    assert recorded["device"] == "cpu"
    assert result["tokens_trained"] == int(len(expected_train))
    assert (output_dir / "result.json").is_file()


def test_prepare_training_data_uses_same_eot_stream_with_or_without_aligned_planning(
    tmp_path,
):
    documents = ["alpha", "", "line one\n\nline two", "omega"]
    corpus_path = _write(
        tmp_path / "corpus.txt",
        json.dumps({"format": "document-list-v1", "documents": documents}),
    )
    enc = tiktoken.get_encoding("gpt2")
    expected = torch.tensor(
        [
            token
            for document in documents
            for token in [50256, *enc.encode_ordinary(document)]
        ],
        dtype=torch.long,
    )

    aligned_train, aligned_val, aligned_ranges = prepare_training_data(
        corpus_path, val_fraction=0.25, eos_aligned_batches=True
    )
    flat_train, flat_val, flat_ranges = prepare_training_data(
        corpus_path, val_fraction=0.25, eos_aligned_batches=False
    )

    assert torch.equal(torch.cat((aligned_train, aligned_val)), expected)
    assert torch.equal(flat_train, aligned_train)
    assert torch.equal(flat_val, aligned_val)
    assert flat_ranges is None
    assert aligned_ranges is not None
    assert expected[0].item() == 50256
    assert aligned_train[0].item() == 50256
    assert int((expected == 50256).sum()) == len(documents)


@pytest.mark.slow
@pytest.mark.asyncio
async def test_real_recipe_runs_end_to_end(base_dir, manifest, tmp_path):
    """Exercise the actual speedrun recipe (CPU) on the curated bundle.

    Marked slow: it instantiates and trains a tiny GPT for a few steps.
    """
    bundle_dir = tmp_path / "bundle"
    output_dir = tmp_path / "out"
    recorder = _Recorder(base_dir)
    cp, prov, _ = await asyncio.to_thread(
        resolve_corpus, manifest, bundle_dir, materialize_fn=recorder
    )
    config = build_debug_config(
        steps=3, block_size=32, batch_size=2, n_layer=2, n_embd=64, n_head=4
    )
    result = await asyncio.to_thread(train_debug, cp, output_dir, config)
    assert result["loss"] is not None
    import math

    assert math.isfinite(result["loss"])
    assert result["tokens_trained"] > 0
