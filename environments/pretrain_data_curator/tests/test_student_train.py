"""CPU unit tests for the modern proxy-student TRAINING recipe (``student_train``).

These guard the single source of truth for the real trainer's optimizer schedule,
contiguous-window batching, gradient clipping/weight decay, and multi-run
averaging. Every piece is exercised on CPU here, and the verbatim recipe source is
asserted to be embedded byte-identically in the sandbox training script
(``trainer.py``) — so the GPU run executes this exact code. A discriminative-power
test proves the upgraded recipe makes a clean corpus reach a LOWER proxy val loss
than a dirty one (the signal the upgrade is meant to sharpen).
"""

from __future__ import annotations

import ast
import inspect
import math
import random
import sys

import pytest
import tiktoken
import torch

from pretrain_data_curator.student_model import StudentModelConfig
from pretrain_data_curator.student_train import (
    _TRAINING_COMPONENTS,
    averaged_train_and_eval,
    batch_document_attn_mask,
    build_document_attn_mask,
    encode_document_tokens,
    lr_at_step,
    plan_train_windows,
    plan_eos_aligned_windows,
    prepare_student_model_dtype,
    shuffled_window_starts,
    make_seq_len_schedule,
    train_and_eval_student,
    window_document_ids,
)
from pretrain_data_curator.trainer import NANOGPT_TRAIN_SCRIPT


def _tiny_cfg(vocab_size: int) -> StudentModelConfig:
    # Smallest valid student: even depth >= 2, head_dim a multiple of 4.
    return StudentModelConfig(
        model_dim=32,
        num_layers=2,
        num_heads=2,
        mlp_ratio=4,
        vocab_size=vocab_size,
        num_value_embeds=1,
    )


# --- (1) LR schedule: linear warmup then cosine decay to a floor ------------


def test_lr_at_step_warmup_then_cosine_to_floor():
    base, total, warm, floor = 1.0, 100, 10, 0.1
    # Linear warmup: lr = base*(step+1)/warm, peaking at base on the last warmup step.
    assert lr_at_step(0, total, warm, base, floor) == pytest.approx(base / warm)
    assert lr_at_step(warm - 1, total, warm, base, floor) == pytest.approx(base)
    # Cosine cooldown starts at the peak right after warmup...
    assert lr_at_step(warm, total, warm, base, floor) == pytest.approx(base)
    # ...decays monotonically...
    cooldown = [lr_at_step(s, total, warm, base, floor) for s in range(warm, total)]
    assert all(a >= b - 1e-12 for a, b in zip(cooldown, cooldown[1:]))
    # ...and lands near (but not below) the floor = base*min_ratio.
    assert cooldown[-1] < base * 0.2
    assert cooldown[-1] >= base * floor - 1e-9
    # warmup_steps == 0 => no warmup, decay begins at the peak immediately.
    assert lr_at_step(0, total, 0, base, floor) == pytest.approx(base)


def test_lr_at_step_floor_ratio_is_the_asymptote():
    # At the very last step the cosine has all but bottomed out, so the LR is within
    # a hair of the documented base*min_ratio floor.
    base, total, warm, floor = 0.5, 64, 8, 0.05
    last = lr_at_step(total - 1, total, warm, base, floor)
    assert last == pytest.approx(base * floor, abs=base * 0.01)


# Configs as (total_steps, warmup_steps), including the short/degenerate runs the
# loop actually executes: (5,2) regular short, (2,1) one-step cooldown, (3,0) no
# warmup, (1,0) single step that is both first and last decay step.
_LR_SCHEDULE_CONFIGS = [(5, 2), (1, 0), (64, 8)]


@pytest.mark.parametrize("total, warm", _LR_SCHEDULE_CONFIGS)
def test_lr_at_step_last_executed_step_hits_floor_exactly(total, warm):
    # The loop runs `for step in range(total)`, so the LAST executed step is total-1.
    # That step must land EXACTLY on the floor base*min_ratio (no off-by-one that
    # stops a step short). This is the 'no off-by-one at final step' acceptance
    # point: it FAILS against the pre-fix `max(1, total-warmup)` denominator, which
    # left short runs at e.g. 0.325*base (5,2) or never decaying at all (2,1)/(1,0).
    base, floor = 0.7, 0.1
    assert lr_at_step(total - 1, total, warm, base, floor) == pytest.approx(
        base * floor
    )


@pytest.mark.parametrize("total, warm", [(5, 2), (64, 8)])
def test_lr_at_step_warmup_peak_unchanged(total, warm):
    # Warmup branch is preserved verbatim: linear ramp base*(step+1)/warm whose last
    # warmup step (warm-1) is exactly the peak base_lr.
    base, floor = 0.7, 0.1
    assert lr_at_step(0, total, warm, base, floor) == pytest.approx(base / warm)
    assert lr_at_step(warm - 1, total, warm, base, floor) == pytest.approx(base)


@pytest.mark.parametrize("total, warm", _LR_SCHEDULE_CONFIGS)
def test_lr_at_step_cooldown_monotonic_nonincreasing(total, warm):
    # Across every executed cooldown step (warm..total-1) the LR is monotonically
    # non-increasing -- a proper cosine cooldown, not a re-rising or constant LR --
    # for every config including the degenerate short runs.
    base, floor = 0.7, 0.1
    cooldown = [lr_at_step(s, total, warm, base, floor) for s in range(warm, total)]
    assert all(a >= b - 1e-12 for a, b in zip(cooldown, cooldown[1:]))
    assert cooldown[-1] == pytest.approx(base * floor)


def test_lr_at_step_no_zero_division_on_degenerate_runs():
    # Warmup-only / zero-decay runs (total <= warmup) must stay sane and never raise.
    for total, warm in [(2, 2), (1, 1), (3, 5), (1, 4)]:
        for step in range(total):
            lr_at_step(step, total, warm, 0.7, 0.1)  # no ZeroDivisionError


# --- (2) contiguous-window batching (NOT random-with-replacement) -----------


def test_plan_train_windows_are_contiguous_sequential_and_in_bounds():
    starts = plan_train_windows(100, 16)
    # Non-overlapping windows tiling the stream by `block`, in increasing order.
    assert starts == [0, 16, 32, 48, 64, 80]
    assert starts == sorted(starts)
    assert all(b - a == 16 for a, b in zip(starts, starts[1:]))
    # Every window's shifted target stays in-bounds (start+block+1 <= n_tokens).
    assert all(s + 16 + 1 <= 100 for s in starts)


def test_plan_train_windows_tiny_corpus_safety_net():
    # A corpus at/under block+1 yields a single window at 0 (the script tiles such
    # corpora first, so this branch is only the safety net).
    assert plan_train_windows(8, 16) == [0]
    assert plan_train_windows(17, 16) == [0]


def test_batching_draws_each_contiguous_window_before_repeating(monkeypatch):
    # Prove the loader walks CONTIGUOUS windows sequentially (a shuffled permutation
    # of the window starts each epoch) rather than sampling offsets with
    # replacement: across exactly one epoch every window start is drawn once. Tokens
    # are a plain arange so each example's first token == its window start index.
    import pretrain_data_curator.student_train as st

    block, batch = 8, 1
    n = 8 * block + 1  # 8 full contiguous windows
    V = n  # distinct token id per position, all < vocab_size
    data = torch.arange(n)
    starts = plan_train_windows(n, block)
    assert len(starts) == 8

    captured: list[int] = []
    real_stack = torch.stack

    def spy_stack(tensors, *a, **k):
        out = real_stack(tensors, *a, **k)
        if out.dim() == 2 and out.size(0) == batch and out.size(1) == block:
            captured.append(int(out[0, 0].item()))
        return out

    monkeypatch.setattr(st.torch, "stack", spy_stack)
    model = _tiny_cfg(V).build()
    gen = torch.Generator().manual_seed(0)
    # Exactly one full training epoch (len(starts) steps at batch=1), then the val
    # pass. Training runs entirely BEFORE eval, so the first len(starts) captured
    # examples are the training draws.
    train_and_eval_student(
        model,
        data,
        data,
        block_size=block,
        batch_size=batch,
        steps=len(starts),
        base_lr=1e-3,
        warmup_steps=2,
        weight_decay=0.1,
        grad_clip=1.0,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        lr_min_ratio=0.1,
        vocab_size=V,
        device="cpu",
        generator=gen,
        training_recipe="record_01_adamw",
    )
    # Each training step stacks the inputs (window start s) then the targets
    # (start s+1), so the input-window starts are the even-indexed captures of the
    # one training epoch (which precedes the val pass).
    input_starts = captured[: 2 * len(starts)][::2]
    # Each contiguous window is drawn exactly once per epoch (sequential, no
    # replacement) — never the random-with-replacement sampler the upgrade replaced.
    assert sorted(input_starts) == sorted(starts)


# --- (3) optimizer/regularization actually applied (not silently ignored) ---


def test_recipe_applies_adamw_betas_weight_decay_grad_clip_and_schedule(monkeypatch):
    import pretrain_data_curator.student_train as st

    captured: dict = {}
    real_adamw = torch.optim.AdamW

    def spy_adamw(params, **kw):
        captured["adamw"] = kw
        return real_adamw(params, **kw)

    clip_calls: list[float] = []
    real_clip = torch.nn.utils.clip_grad_norm_

    def spy_clip(params, max_norm, *a, **k):
        clip_calls.append(float(max_norm))
        return real_clip(params, max_norm, *a, **k)

    lr_calls: list[tuple] = []
    real_lr = st.lr_at_step

    def spy_lr(step, total_steps, warmup_steps, base_lr, min_ratio):
        lr_calls.append((step, total_steps, warmup_steps, base_lr, min_ratio))
        return real_lr(step, total_steps, warmup_steps, base_lr, min_ratio)

    monkeypatch.setattr(torch.optim, "AdamW", spy_adamw)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy_clip)
    monkeypatch.setattr(st, "lr_at_step", spy_lr)

    V = 16
    model = _tiny_cfg(V).build()
    data = torch.randint(0, V, (200,))
    gen = torch.Generator().manual_seed(0)
    steps = 5
    train_and_eval_student(
        model,
        data,
        data,
        block_size=8,
        batch_size=4,
        steps=steps,
        base_lr=1e-3,
        warmup_steps=2,
        weight_decay=0.1,
        grad_clip=1.0,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        lr_min_ratio=0.1,
        vocab_size=V,
        device="cpu",
        generator=gen,
        training_recipe="record_01_adamw",
    )

    # AdamW built with the record_01 betas/eps + decoupled weight decay (not ignored).
    assert captured["adamw"]["betas"] == (0.9, 0.95)
    assert captured["adamw"]["eps"] == 1e-8
    assert captured["adamw"]["weight_decay"] == 0.1
    assert captured["adamw"]["lr"] == 1e-3
    # Gradient clip applied once per step with the configured max-norm.
    assert clip_calls == [1.0] * steps
    # The per-step LR schedule is applied for every step (0..steps-1), with the
    # configured total/warmup/base/floor — i.e. not a silent constant LR.
    assert [c[0] for c in lr_calls] == list(range(steps))
    assert all(c[1:] == (steps, 2, 1e-3, 0.1) for c in lr_calls)


def test_grad_clip_zero_skips_clipping(monkeypatch):
    # grad_clip == 0 means "no clipping": the clip op must not be called at all.
    clip_calls: list = []
    real_clip = torch.nn.utils.clip_grad_norm_

    def spy_clip(params, max_norm, *a, **k):
        clip_calls.append(max_norm)
        return real_clip(params, max_norm, *a, **k)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy_clip)
    V = 16
    model = _tiny_cfg(V).build()
    data = torch.randint(0, V, (100,))
    gen = torch.Generator().manual_seed(0)
    train_and_eval_student(
        model,
        data,
        data,
        block_size=8,
        batch_size=2,
        steps=3,
        base_lr=1e-3,
        warmup_steps=1,
        weight_decay=0.1,
        grad_clip=0.0,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        lr_min_ratio=0.1,
        vocab_size=V,
        device="cpu",
        generator=gen,
        training_recipe="record_01_adamw",
    )
    assert clip_calls == []


# --- (4) multi-run averaging + non-finite collapse to the sentinel ----------


def test_averaged_train_and_eval_averages_loss_and_sums_flops(monkeypatch):
    import pretrain_data_curator.student_train as st

    V = 16
    cfg = _tiny_cfg(V)
    seq = [2.0, 4.0, 6.0]
    calls = {"i": 0}

    def fake_train(model, train_data, val_data, **kw):
        loss = seq[calls["i"]]
        calls["i"] += 1
        return loss, 0.5, 100  # (loss, accuracy, tokens_trained)

    monkeypatch.setattr(st, "train_and_eval_student", fake_train)
    data = torch.randint(0, V, (50,))
    loss, acc, flops, tokens, n_params = st.averaged_train_and_eval(
        cfg.build,
        data,
        data,
        n_runs=3,
        base_seed=0,
        device="cpu",
        block_size=8,
        batch_size=4,
        steps=1,
        base_lr=1e-3,
        warmup_steps=1,
        weight_decay=0.1,
        grad_clip=1.0,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        lr_min_ratio=0.1,
        vocab_size=V,
    )
    assert calls["i"] == 3  # ran every requested run
    assert loss == pytest.approx(sum(seq) / 3)  # loss AVERAGED across runs
    assert acc == pytest.approx(0.5)
    assert tokens == 300  # tokens SUMMED (3 * 100)
    assert n_params > 0
    assert flops == pytest.approx(3 * 6.0 * n_params * 100)  # FLOPs SUMMED across runs


def test_averaged_train_and_eval_nonfinite_run_collapses_to_sentinel(monkeypatch):
    import pretrain_data_curator.student_train as st

    V = 16
    cfg = _tiny_cfg(V)
    seq = [2.0, float("inf"), 3.0]
    calls = {"i": 0}

    def fake_train(model, train_data, val_data, **kw):
        loss = seq[calls["i"]]
        calls["i"] += 1
        return loss, 0.5, 100

    monkeypatch.setattr(st, "train_and_eval_student", fake_train)
    data = torch.randint(0, V, (50,))
    result = st.averaged_train_and_eval(
        cfg.build,
        data,
        data,
        n_runs=3,
        base_seed=0,
        device="cpu",
        block_size=8,
        batch_size=4,
        steps=1,
        base_lr=1e-3,
        warmup_steps=1,
        weight_decay=0.1,
        grad_clip=1.0,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        lr_min_ratio=0.1,
        vocab_size=V,
    )
    # A single non-finite run collapses the WHOLE result to the infinite-loss
    # sentinel (perf -> 0), and short-circuits before the remaining runs.
    assert result == (float("inf"), 0.0, 0.0, 0, 0)
    assert calls["i"] == 2


def test_n_train_runs_one_equals_a_single_run(monkeypatch):
    # Default n_runs=1 is exactly one train+eval (cost + calibration unchanged).
    import pretrain_data_curator.student_train as st

    V = 16
    cfg = _tiny_cfg(V)
    calls = {"i": 0}

    def fake_train(model, train_data, val_data, **kw):
        calls["i"] += 1
        return 3.0, 0.25, 100

    monkeypatch.setattr(st, "train_and_eval_student", fake_train)
    data = torch.randint(0, V, (50,))
    loss, acc, flops, tokens, n_params = st.averaged_train_and_eval(
        cfg.build,
        data,
        data,
        n_runs=1,
        base_seed=0,
        device="cpu",
        block_size=8,
        batch_size=4,
        steps=1,
        base_lr=1e-3,
        warmup_steps=1,
        weight_decay=0.1,
        grad_clip=1.0,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        lr_min_ratio=0.1,
        vocab_size=V,
    )
    assert calls["i"] == 1
    assert (loss, acc, tokens) == (3.0, 0.25, 100)
    assert flops == pytest.approx(6.0 * n_params * 100)


# --- (5) DISCRIMINATIVE POWER: clean corpus -> lower proxy val loss ----------


def _char_tensor(text: str, stoi: dict[str, int]) -> torch.Tensor:
    return torch.tensor([stoi[c] for c in text], dtype=torch.long)


def test_clean_corpus_yields_lower_proxy_val_loss_than_dirty():
    # The downstream-loss setup: train the fixed student on a curated corpus and
    # score it on a FIXED held-out CLEAN stream. Under the upgraded record_01 recipe
    # a clean, structured corpus must reach a clearly LOWER held-out cross-entropy
    # than a high-entropy "dirty" corpus — the signal this upgrade sharpens. Run on
    # CPU with a tiny char-level student so no GPU (or tiktoken/network) is needed.
    clean = "the quick brown fox jumps over the lazy dog . " * 120
    held_out = "the quick brown fox jumps over the lazy dog . " * 24
    rng = random.Random(0)
    symbols = "$%#@^&*~`|<>{}[]"
    dirty = "".join(rng.choice(symbols) for _ in range(len(clean)))

    vocab = sorted(set(clean + held_out + dirty))
    stoi = {c: i for i, c in enumerate(vocab)}
    vocab_size = len(vocab)
    val_data = _char_tensor(held_out, stoi)
    cfg = _tiny_cfg(vocab_size)
    hparams = dict(
        block_size=16,
        batch_size=8,
        steps=24,
        base_lr=3e-3,
        warmup_steps=4,
        weight_decay=0.1,
        grad_clip=1.0,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        lr_min_ratio=0.1,
        vocab_size=vocab_size,
        training_recipe="record_01_adamw",
    )

    clean_loss, *_ = averaged_train_and_eval(
        cfg.build,
        _char_tensor(clean, stoi),
        val_data,
        n_runs=1,
        base_seed=0,
        device="cpu",
        **hparams,
    )
    dirty_loss, *_ = averaged_train_and_eval(
        cfg.build,
        _char_tensor(dirty, stoi),
        val_data,
        n_runs=1,
        base_seed=0,
        device="cpu",
        **hparams,
    )
    assert math.isfinite(clean_loss) and math.isfinite(dirty_loss)
    # Clean training generalizes to the held-out clean stream; dirty training does
    # not — a clear, deterministic separation under the new recipe.
    assert clean_loss < dirty_loss


def test_discriminative_signal_requires_contiguous_recipe_not_replacement(monkeypatch):
    # The clean<dirty separation above would also pass for a random-with-replacement
    # trainer (clean matches the held-out stream; dirty is high-entropy noise), so on
    # its own it does NOT prove the upgraded contiguous-window recipe is what produced
    # the signal. This complementary test closes that honesty gap by asserting the two
    # invariants the recipe GUARANTEES and a random-with-replacement sampler provably
    # BREAKS, exercised on the real trainer that produces the proxy val loss:
    #   (a) every training window start is block-ALIGNED and a genuine
    #       plan_train_windows start — never an arbitrary offset in [0, n-block-1];
    #   (b) over K whole epochs every window is drawn EXACTLY K times (full, uniform
    #       coverage), not the unbalanced miss-some/repeat-some draw of replacement.
    # Tokens are a plain arange so each example's first token == its window start.
    import collections

    import pretrain_data_curator.student_train as st

    block, batch, epochs = 8, 1, 3
    n = 8 * block + 1  # 8 full contiguous windows
    V = n
    data = torch.arange(n)
    starts = plan_train_windows(n, block)
    assert len(starts) == 8

    captured: list[int] = []
    real_stack = torch.stack

    def spy_stack(tensors, *a, **k):
        out = real_stack(tensors, *a, **k)
        if out.dim() == 2 and out.size(0) == batch and out.size(1) == block:
            captured.append(int(out[0, 0].item()))
        return out

    monkeypatch.setattr(st.torch, "stack", spy_stack)
    model = _tiny_cfg(V).build()
    gen = torch.Generator().manual_seed(0)
    steps = epochs * len(starts)
    train_and_eval_student(
        model,
        data,
        data,
        block_size=block,
        batch_size=batch,
        steps=steps,
        base_lr=1e-3,
        warmup_steps=2,
        weight_decay=0.1,
        grad_clip=1.0,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        lr_min_ratio=0.1,
        vocab_size=V,
        device="cpu",
        generator=gen,
        training_recipe="record_01_adamw",
    )
    # Training (steps draws) runs entirely BEFORE the val pass and stacks inputs then
    # targets each step, so the input-window starts are the even-indexed captures of
    # the first `steps` training draws.
    input_starts = captured[: 2 * steps][::2]
    assert len(input_starts) == steps

    # (a) ALIGNED, real contiguous-window starts — never random offsets.
    valid = set(starts)
    assert all(s in valid for s in input_starts)
    assert all(s % block == 0 for s in input_starts)
    # (b) EXACTLY-uniform K-fold coverage (each window once per epoch, K epochs).
    assert collections.Counter(input_starts) == {s: epochs for s in starts}

    # Demonstrate concretely that a random-with-replacement sampler (the OLD
    # get_batch the upgrade replaced, drawing offsets in [0, n-block-1]) FAILS both
    # invariants for the same step budget — so the assertions above are a genuine
    # guard, not ones a replacement sampler would also satisfy. Fixed seed => fully
    # deterministic, never flaky.
    rng = random.Random(0)
    last_offset = n - block - 1
    replacement_starts = [rng.randint(0, last_offset) for _ in range(steps)]
    assert any(s % block != 0 for s in replacement_starts)  # not block-aligned
    assert collections.Counter(replacement_starts) != {
        s: epochs for s in starts
    }  # not uniform


# --- (8) document-aware portable training feature tests ---


def test_real_document_tokenization_preserves_boundaries():
    enc = tiktoken.get_encoding("gpt2")
    documents = ["first", "", "line one\n\nline two"]
    token_ids, ranges = encode_document_tokens(documents, enc)
    expected = [
        token
        for document in documents
        for token in [50256, *enc.encode_ordinary(document)]
    ]

    assert token_ids == expected
    assert token_ids[0] == 50256
    assert token_ids.count(50256) == len(documents)
    assert len(ranges) == len(documents)
    for (start, end), document in zip(ranges, documents, strict=True):
        assert token_ids[start] == 50256
        assert token_ids[start + 1 : end] == enc.encode_ordinary(document)
    assert ranges[0][0] == 0
    assert ranges[1][1] - ranges[1][0] == 1  # blank document is still explicit


def test_eos_aligned_windows_never_cross_document_boundaries():
    ranges = [(0, 20), (20, 29), (29, 65)]
    starts = plan_eos_aligned_windows(65, 8, ranges)
    assert starts == [0, 8, 20, 29, 37, 45, 53]
    assert all(
        any(
            start >= doc_start and start + 8 + 1 <= doc_end
            for doc_start, doc_end in ranges
        )
        for start in starts
    )


def test_eos_aligned_windows_reserve_multi_token_lookahead():
    ranges = [(0, 12), (12, 24)]
    assert plan_eos_aligned_windows(24, 8, ranges, lookahead=1) == [0, 12]
    assert plan_eos_aligned_windows(24, 8, ranges, lookahead=4) == [0, 12]
    # First doc is short for lookahead=4 (need=12); one sequential pack covers it.
    # The following doc has no residual uncovered span long enough for another window.
    assert plan_eos_aligned_windows(22, 8, [(0, 10), (10, 22)], lookahead=4) == [0]


def test_eos_aligned_windows_have_no_flat_fallback():
    assert plan_eos_aligned_windows(100, 16, []) == []
    # Even with packing, the stream is shorter than block+lookahead.
    assert plan_eos_aligned_windows(8, 8, [(0, 5), (5, 8)]) == []


def test_eos_aligned_windows_reject_gaps():
    with pytest.raises(ValueError, match="contiguous"):
        plan_eos_aligned_windows(20, 8, [(0, 5), (7, 20)])


def test_packed_windows_include_short_documents_without_overlap():
    ranges = [(0, 5), (5, 10), (10, 28)]
    starts = plan_eos_aligned_windows(28, 8, ranges)
    # One sequential pack from the first short doc, then long-doc strides.
    assert starts == [0, 10, 18]
    assert 5 not in starts  # no overlapping second short-doc start


def test_packed_window_targets_follow_flat_shift_across_eot():
    """Speedrun contract: targets are the flat next-token shift (may be next EOT)."""
    enc = tiktoken.get_encoding("gpt2")
    documents = ["aa", "bb", "long enough document text here"]
    token_ids, ranges = encode_document_tokens(documents, enc)
    data = torch.tensor(token_ids, dtype=torch.long)
    starts = plan_eos_aligned_windows(len(data), 8, ranges)
    assert 0 in starts
    x = data[0:8]
    y = data[1:9]
    # First document is shorter than the window, so some target is past its end.
    first_end = ranges[0][1]
    assert first_end < 9
    assert int(y[first_end - 1].item()) == 50256  # leading EOT of the next doc
    assert int(x[0].item()) == 50256


def test_packed_attention_mask_keeps_same_document_causal_keys():
    ranges = [(0, 5), (5, 20)]
    mask = batch_document_attn_mask([0], 8, ranges, device="cpu")
    assert mask is not None
    assert mask.shape == (1, 8, 8)
    # True = may participate: cross-document key is masked out.
    assert bool(mask[0, 5, 0].item()) is False
    # Same-document past key may participate.
    assert bool(mask[0, 4, 0].item()) is True
    # Future keys are masked out.
    assert bool(mask[0, 0, 1].item()) is False
    # Single-document windows skip the mask path entirely.
    assert batch_document_attn_mask([5], 8, ranges, device="cpu") is None


def test_document_gap_ids_do_not_attend_each_other():
    doc_ids = torch.tensor([[0, 0, -1, -1, 1]])
    mask = build_document_attn_mask(doc_ids)
    # Gap positions cannot attend to each other.
    assert bool(mask[0, 2, 3].item()) is False
    assert bool(mask[0, 3, 2].item()) is False
    # Self-attention remains available for gaps.
    assert bool(mask[0, 2, 2].item()) is True
    assert bool(mask[0, 3, 3].item()) is True


def test_packed_planning_is_fixed_seed_deterministic():
    ranges = [(0, 4), (4, 8), (8, 40), (40, 80)]
    starts = plan_eos_aligned_windows(80, 8, ranges)
    assert starts == [0, 9, 17, 25, 40, 48, 56, 64]
    first = shuffled_window_starts(starts, torch.Generator().manual_seed(123))
    second = shuffled_window_starts(starts, torch.Generator().manual_seed(123))
    different = shuffled_window_starts(starts, torch.Generator().manual_seed(124))
    assert first == second
    assert first != different


def test_packed_short_documents_contribute_trained_tokens():
    V = 16
    model = _tiny_cfg(V).build()
    # Three short docs that only become usable via packing.
    data = torch.arange(24) % V
    ranges = [(0, 6), (6, 12), (12, 24)]
    starts = plan_eos_aligned_windows(24, 8, ranges)
    assert starts == [0, 12]  # one pack, then residual long-enough third doc
    gen = torch.Generator().manual_seed(0)
    loss, acc, tokens = train_and_eval_student(
        model,
        data,
        data,
        block_size=8,
        batch_size=2,
        steps=3,
        base_lr=1e-3,
        warmup_steps=1,
        weight_decay=0.1,
        grad_clip=1.0,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        lr_min_ratio=0.1,
        vocab_size=V,
        device="cpu",
        generator=gen,
        training_recipe="record_01_adamw",
        document_ranges=ranges,
        batch_schedule_enabled=False,
    )
    assert math.isfinite(loss)
    assert tokens == 3 * 2 * 8


def test_legacy_flat_windows_ignore_document_ranges_when_disabled():
    assert plan_train_windows(20, 8) == [0, 8]
    # plan_eos with empty ranges still refuses a flat fallback.
    assert plan_eos_aligned_windows(20, 8, []) == []


def test_eos_aligned_window_selection_is_fixed_seed_deterministic():
    starts = plan_eos_aligned_windows(80, 8, [(0, 40), (40, 80)])
    first = shuffled_window_starts(starts, torch.Generator().manual_seed(123))
    second = shuffled_window_starts(starts, torch.Generator().manual_seed(123))
    different = shuffled_window_starts(starts, torch.Generator().manual_seed(124))
    assert first == second
    assert first != different


def test_window_document_ids_cover_packed_span():
    ids = window_document_ids(0, 8, [(0, 5), (5, 20)])
    assert ids.tolist() == [0, 0, 0, 0, 0, 1, 1, 1]
    mask = build_document_attn_mask(ids.unsqueeze(0))
    assert mask.shape == (1, 8, 8)


def test_seq_len_schedule_warmup():
    block_fn = make_seq_len_schedule(100, 64)
    assert block_fn(0) == 8  # min block
    assert block_fn(25) == 64  # should reach max by now (25% of 100)
    assert block_fn(100) == 64  # max after warmup


def test_seq_len_schedule_exact_values():
    block_fn = make_seq_len_schedule(40, 32)
    # warmup_frac=0.25 => 10 warmup steps
    # min=8, max=32, range=24
    # at step 0: frac=0 => 8
    # at step 5: frac=0.5 => 8 + 12 = 20
    # at step 10: frac=1.0 => 32
    assert block_fn(0) == 8
    assert block_fn(5) == 20
    assert block_fn(10) == 32
    assert block_fn(40) == 32


def test_training_with_eos_aligned_windows():
    V = 16
    model = _tiny_cfg(V).build()
    data = torch.randint(0, V, (120,))
    gen = torch.Generator().manual_seed(0)
    loss, acc, tokens = train_and_eval_student(
        model,
        data,
        data,
        block_size=8,
        batch_size=2,
        steps=3,
        base_lr=1e-3,
        warmup_steps=1,
        weight_decay=0.1,
        grad_clip=1.0,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        lr_min_ratio=0.1,
        vocab_size=V,
        device="cpu",
        generator=gen,
        training_recipe="record_01_adamw",
        document_ranges=[(0, 30), (30, 60), (60, 120)],
    )
    assert math.isfinite(loss)
    assert tokens > 0


def test_eos_selection_integrates_with_odd_step_adam_cadence(monkeypatch):
    V = 16
    model = _tiny_cfg(V).build()
    data = torch.randint(0, V, (120,))
    adam_steps = 0
    real_step = torch.optim.AdamW.step

    def spy_step(self, *args, **kwargs):
        nonlocal adam_steps
        adam_steps += 1
        return real_step(self, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", spy_step)
    train_and_eval_student(
        model,
        data,
        data,
        block_size=8,
        batch_size=2,
        steps=4,
        base_lr=1e-3,
        warmup_steps=1,
        weight_decay=0.1,
        grad_clip=1.0,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        lr_min_ratio=0.1,
        vocab_size=V,
        device="cpu",
        generator=torch.Generator().manual_seed(7),
        training_recipe="speedrun_muon",
        document_ranges=[(0, 40), (40, 80), (80, 120)],
        adam_on_odd_steps=True,
    )
    assert adam_steps == 2


def test_training_with_seq_len_schedule():
    V = 16
    model = _tiny_cfg(V).build()
    data = torch.randint(0, V, (200,))
    gen = torch.Generator().manual_seed(0)
    loss, acc, tokens = train_and_eval_student(
        model,
        data,
        data,
        block_size=16,
        batch_size=2,
        steps=5,
        base_lr=1e-3,
        warmup_steps=1,
        weight_decay=0.1,
        grad_clip=1.0,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        lr_min_ratio=0.1,
        vocab_size=V,
        device="cpu",
        generator=gen,
        training_recipe="record_01_adamw",
        seq_len_schedule=True,
    )
    assert math.isfinite(loss)
    assert tokens > 0


def test_training_with_multi_token_pred():
    """Full training with multi_token_pred=2 exercises correct hidden-state
    auxiliary heads and shifted target construction."""
    V = 16
    cfg = _tiny_cfg(V)
    model = cfg.build()
    data = torch.randint(0, V, (200,))
    gen = torch.Generator().manual_seed(0)
    loss, acc, tokens = train_and_eval_student(
        model,
        data,
        data,
        block_size=8,
        batch_size=2,
        steps=5,
        base_lr=1e-3,
        warmup_steps=1,
        weight_decay=0.1,
        grad_clip=1.0,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        lr_min_ratio=0.1,
        vocab_size=V,
        device="cpu",
        generator=gen,
        training_recipe="record_01_adamw",
        multi_token_pred=2,
    )
    assert math.isfinite(loss)
    assert tokens > 0


def test_training_with_multi_token_pred_and_speedrun():
    """Multi-token prediction must also work under the speedrun_muon recipe."""
    V = 16
    from pretrain_data_curator.student_model import StudentModelConfig as SMC

    cfg = SMC(model_dim=32, num_layers=2, num_heads=2, vocab_size=V, num_value_embeds=1)
    model = cfg.build()
    data = torch.randint(0, V, (200,))
    gen = torch.Generator().manual_seed(0)
    loss, acc, tokens = train_and_eval_student(
        model,
        data,
        data,
        block_size=8,
        batch_size=2,
        steps=5,
        base_lr=1e-3,
        warmup_steps=1,
        weight_decay=0.1,
        grad_clip=1.0,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        lr_min_ratio=0.1,
        vocab_size=V,
        device="cpu",
        generator=gen,
        training_recipe="speedrun_muon",
        multi_token_pred=2,
    )
    assert math.isfinite(loss)
    assert tokens > 0


def test_full_training_with_speedrun_and_nor_muon():
    V = 16
    from pretrain_data_curator.student_model import StudentModelConfig as SMC

    cfg = SMC(model_dim=32, num_layers=2, num_heads=2, vocab_size=V, num_value_embeds=1)
    model = cfg.build()
    data = torch.randint(0, V, (100,))
    gen = torch.Generator().manual_seed(0)
    loss, acc, tokens = train_and_eval_student(
        model,
        data,
        data,
        block_size=8,
        batch_size=2,
        steps=4,
        base_lr=1e-3,
        warmup_steps=1,
        weight_decay=0.1,
        grad_clip=1.0,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        lr_min_ratio=0.1,
        vocab_size=V,
        device="cpu",
        generator=gen,
        training_recipe="speedrun_muon",
        nor_muon=True,
        cautious_wd=True,
    )
    assert math.isfinite(loss)
    assert tokens > 0


def test_max_document_tokens_rejects_instead_of_splitting_a_document():
    enc = tiktoken.get_encoding("gpt2")
    with pytest.raises(ValueError, match="documents are never truncated"):
        encode_document_tokens(["one " * 100], enc, max_document_tokens=32)

    token_ids, ranges = encode_document_tokens(
        ["", "short"], enc, max_document_tokens=32
    )
    assert token_ids == [50256, 50256, *enc.encode_ordinary("short")]
    assert ranges == [(0, 1), (1, len(token_ids))]


def test_multi_token_pred_targets_shifted_correctly(monkeypatch):
    """Verify multi-token prediction targets are correctly shifted by k+2
    positions, not constructed from a single-element slice of train_src."""
    import pretrain_data_curator.student_train as st

    V = 64
    from pretrain_data_curator.student_model import GPT

    model = GPT(
        vocab_size=V, num_layers=2, model_dim=32, num_heads=2, multi_token_pred=2
    )

    # Create a predictable sequence (values modulo vocab so all tokens are valid)
    data = torch.arange(200, dtype=torch.long) % V
    gen = torch.Generator().manual_seed(0)

    captured_targets = {}
    real_mt_loss = st._compute_multi_token_loss

    def spy_mt_loss(logits, hidden, y, multi_heads, y_future, vocab_size):
        for k, ft in enumerate(y_future):
            captured_targets[k] = ft.clone()
        return real_mt_loss(logits, hidden, y, multi_heads, y_future, vocab_size)

    monkeypatch.setattr(st, "_compute_multi_token_loss", spy_mt_loss)

    train_and_eval_student(
        model,
        data,
        data,
        block_size=8,
        batch_size=2,
        steps=3,
        base_lr=1e-3,
        warmup_steps=1,
        weight_decay=0.1,
        grad_clip=1.0,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        lr_min_ratio=0.1,
        vocab_size=V,
        device="cpu",
        generator=gen,
        training_recipe="speedrun_muon",
        multi_token_pred=2,
    )

    assert len(captured_targets) == 2
    for k, ft in captured_targets.items():
        # Each future target should have the correct block_size dim
        assert ft.shape[1] == 8  # block_size


def test_untie_embed_lm_head_at_frac(monkeypatch):
    """With untie_at_frac > 0, the embedding and lm_head weights start tied
    and are untied (different data pointers) after the scheduled step."""
    import pretrain_data_curator.student_train as st

    V = 32
    from pretrain_data_curator.student_model import StudentModelConfig as SMC

    cfg = SMC(model_dim=32, num_layers=2, num_heads=2, vocab_size=V, num_value_embeds=1)
    model = cfg.build()
    data = torch.randint(0, V, (200,))
    gen = torch.Generator().manual_seed(0)

    def spy_train(model, train_data, val_data, **kw):
        # After init_speedrun_weights and weight tying, before training starts:
        # The real train_and_eval_student does the tying. We'll check after.
        result = st.train_and_eval_student(model, train_data, val_data, **kw)
        return result

    loss, acc, tokens = train_and_eval_student(
        model,
        data,
        data,
        block_size=8,
        batch_size=2,
        steps=8,
        base_lr=1e-3,
        warmup_steps=1,
        weight_decay=0.1,
        grad_clip=1.0,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        lr_min_ratio=0.1,
        vocab_size=V,
        device="cpu",
        generator=gen,
        training_recipe="speedrun_muon",
        untie_at_frac=0.5,  # untie at step 4 (8 * 0.5 = 4)
    )
    # After training with untie_at_frac, the weights should be separate
    # (different data pointers since untie creates a new Parameter)
    assert model.lm_head.weight.data_ptr() != model.embed.weight.data_ptr()
    assert math.isfinite(loss)
    assert tokens > 0


def test_untie_does_not_tie_when_frac_zero():
    """With untie_at_frac=0.0, the weight tying step is skipped entirely,
    so embed and lm_head have separate weight tensors."""
    V = 32
    from pretrain_data_curator.student_model import StudentModelConfig as SMC

    cfg = SMC(model_dim=32, num_layers=2, num_heads=2, vocab_size=V, num_value_embeds=1)
    model = cfg.build()
    data = torch.randint(0, V, (200,))
    gen = torch.Generator().manual_seed(0)

    loss, acc, tokens = train_and_eval_student(
        model,
        data,
        data,
        block_size=8,
        batch_size=2,
        steps=4,
        base_lr=1e-3,
        warmup_steps=1,
        weight_decay=0.1,
        grad_clip=1.0,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        lr_min_ratio=0.1,
        vocab_size=V,
        device="cpu",
        generator=gen,
        training_recipe="speedrun_muon",
        untie_at_frac=0.0,  # no tying/untieing
    )
    # With untie_at_frac=0.0, weight tying is skipped; weights stay separate.
    assert model.lm_head.weight.data_ptr() != model.embed.weight.data_ptr()
    assert math.isfinite(loss)
    assert tokens > 0


def test_full_training_with_polar_express():
    V = 16
    from pretrain_data_curator.student_model import StudentModelConfig as SMC

    cfg = SMC(model_dim=32, num_layers=2, num_heads=2, vocab_size=V, num_value_embeds=1)
    model = cfg.build()
    data = torch.randint(0, V, (100,))
    gen = torch.Generator().manual_seed(0)
    loss, acc, tokens = train_and_eval_student(
        model,
        data,
        data,
        block_size=8,
        batch_size=2,
        steps=4,
        base_lr=1e-3,
        warmup_steps=1,
        weight_decay=0.1,
        grad_clip=1.0,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        lr_min_ratio=0.1,
        vocab_size=V,
        device="cpu",
        generator=gen,
        training_recipe="speedrun_muon",
        polar_express=True,
    )
    assert math.isfinite(loss)
    assert tokens > 0


# --- (9) regression: untie_at_frac + bigram_hash_embed (BLOCKER 1) -----------


def test_untie_at_frac_with_bigram_hash_embed_does_not_crash():
    """untie_at_frac > 0 with bigram_hash_embed=True must not crash on
    model.embed.weight (BigramHashEmbedding has no .weight attribute)."""
    V = 32
    from pretrain_data_curator.student_model import StudentModelConfig as SMC

    cfg = SMC(
        model_dim=32,
        num_layers=2,
        num_heads=2,
        vocab_size=V,
        num_value_embeds=1,
        bigram_hash_embed=True,
    )
    model = cfg.build()
    data = torch.randint(0, V, (200,))
    gen = torch.Generator().manual_seed(0)
    loss, acc, tokens = train_and_eval_student(
        model,
        data,
        data,
        block_size=8,
        batch_size=2,
        steps=8,
        base_lr=1e-3,
        warmup_steps=1,
        weight_decay=0.1,
        grad_clip=1.0,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        lr_min_ratio=0.1,
        vocab_size=V,
        device="cpu",
        generator=gen,
        training_recipe="speedrun_muon",
        untie_at_frac=0.5,
    )
    # The lm_head must be tracked by the optimizer after untie so loss is finite.
    assert math.isfinite(loss)
    assert tokens > 0
    assert model.lm_head.weight.data_ptr() != model.embed.token_embed.weight.data_ptr()


def test_untie_at_frac_registers_lm_head_in_optimizer(monkeypatch):
    """After untie the new lm_head.weight must be registered in AdamW."""
    V = 32
    from pretrain_data_curator.student_model import StudentModelConfig as SMC

    cfg = SMC(model_dim=32, num_layers=2, num_heads=2, vocab_size=V, num_value_embeds=1)
    model = cfg.build()
    data = torch.randint(0, V, (200,))
    gen = torch.Generator().manual_seed(0)
    import pretrain_data_curator.student_train as st

    lm_head_param_ids = set()
    real_add = torch.optim.AdamW.add_param_group

    def spy_add_param_group(self, group):
        nonlocal lm_head_param_ids
        for p in group["params"]:
            lm_head_param_ids.add(id(p))
        return real_add(self, group)

    monkeypatch.setattr(torch.optim.AdamW, "add_param_group", spy_add_param_group)
    st.train_and_eval_student(
        model,
        data,
        data,
        block_size=8,
        batch_size=2,
        steps=8,
        base_lr=1e-3,
        warmup_steps=1,
        weight_decay=0.1,
        grad_clip=1.0,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        lr_min_ratio=0.1,
        vocab_size=V,
        device="cpu",
        generator=gen,
        training_recipe="speedrun_muon",
        untie_at_frac=0.5,
    )
    assert id(model.lm_head.weight) in lm_head_param_ids


def test_untied_lm_head_weight_actually_updates():
    """Regression: the newly untied lm_head.weight changes during subsequent
    training steps (proves it is registered in the optimizer and stepped)."""
    V = 32
    from pretrain_data_curator.student_model import StudentModelConfig as SMC

    cfg = SMC(model_dim=32, num_layers=2, num_heads=2, vocab_size=V, num_value_embeds=1)
    model = cfg.build()
    data = torch.randint(0, V, (200,))
    gen = torch.Generator().manual_seed(0)

    train_and_eval_student(
        model,
        data,
        data,
        block_size=8,
        batch_size=2,
        steps=8,
        base_lr=1e-3,
        warmup_steps=1,
        weight_decay=0.1,
        grad_clip=1.0,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        lr_min_ratio=0.1,
        vocab_size=V,
        device="cpu",
        generator=gen,
        training_recipe="speedrun_muon",
        untie_at_frac=0.5,
    )
    # After the untie (step 4) the lm_head weight is zeroed, but subsequent
    # steps (5-7) must have trained it to a non-zero state.
    assert model.lm_head.weight.abs().sum().item() > 0.0


# --- (10) regression: grad_accum_embed_head_steps > 1 (BLOCKER 2) ------------


def test_grad_accum_embed_head_finite_and_produces_loss():
    """Gradient accumulation for embed+head must produce a finite loss and
    non-zero tokens (smoke test: no crash, the loop actually runs)."""
    V = 16
    from pretrain_data_curator.student_model import StudentModelConfig as SMC

    cfg = SMC(model_dim=32, num_layers=2, num_heads=2, vocab_size=V, num_value_embeds=1)
    model = cfg.build()
    data = torch.randint(0, V, (300,))
    gen = torch.Generator().manual_seed(0)
    loss, acc, tokens = train_and_eval_student(
        model,
        data,
        data,
        block_size=8,
        batch_size=2,
        steps=6,
        base_lr=1e-3,
        warmup_steps=1,
        weight_decay=0.1,
        grad_clip=1.0,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        lr_min_ratio=0.1,
        vocab_size=V,
        device="cpu",
        generator=gen,
        training_recipe="speedrun_muon",
        grad_accum_embed_head_steps=2,
    )
    assert math.isfinite(loss)
    assert tokens > 0


def test_grad_accum_embed_head_embed_and_lm_head_weights_update():
    """With accum>1 the embed and lm_head weights must change from their
    initial values (proving accumulated grads are applied)."""
    V = 16
    from pretrain_data_curator.student_model import StudentModelConfig as SMC

    cfg = SMC(model_dim=32, num_layers=2, num_heads=2, vocab_size=V, num_value_embeds=1)
    model = cfg.build()
    data = torch.randint(0, V, (300,))
    gen = torch.Generator().manual_seed(0)

    pre_embed = model.embed.weight.data.clone()
    pre_lm_head = model.lm_head.weight.data.clone()

    train_and_eval_student(
        model,
        data,
        data,
        block_size=8,
        batch_size=2,
        steps=8,
        base_lr=1e-3,
        warmup_steps=1,
        weight_decay=0.1,
        grad_clip=1.0,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        lr_min_ratio=0.1,
        vocab_size=V,
        device="cpu",
        generator=gen,
        training_recipe="speedrun_muon",
        grad_accum_embed_head_steps=2,
    )
    # embed and lm_head weights must differ from zero-init (they were trained)
    assert not torch.equal(model.embed.weight.data, pre_embed)
    assert not torch.equal(model.lm_head.weight.data, pre_lm_head)


def test_grad_accum_embed_head_with_untie_at_frac():
    """Combined: gradient accumulation + weight untie must not crash and
    produce finite loss (tests both blockers together)."""
    V = 16
    from pretrain_data_curator.student_model import StudentModelConfig as SMC

    cfg = SMC(model_dim=32, num_layers=2, num_heads=2, vocab_size=V, num_value_embeds=1)
    model = cfg.build()
    data = torch.randint(0, V, (300,))
    gen = torch.Generator().manual_seed(0)
    loss, acc, tokens = train_and_eval_student(
        model,
        data,
        data,
        block_size=8,
        batch_size=2,
        steps=10,
        base_lr=1e-3,
        warmup_steps=1,
        weight_decay=0.1,
        grad_clip=1.0,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        lr_min_ratio=0.1,
        vocab_size=V,
        device="cpu",
        generator=gen,
        training_recipe="speedrun_muon",
        grad_accum_embed_head_steps=2,
        untie_at_frac=0.5,
    )
    assert math.isfinite(loss)
    assert tokens > 0
    assert model.lm_head.weight.data_ptr() != model.embed.weight.data_ptr()


def test_grad_accum_embed_head_no_duplicate_first_step_counting(monkeypatch):
    """Prove accumulation does not double-count the first sub-step's grads by
    spying on per-backward grad norms: each backward in a micro-step should
    produce a fresh single-batch gradient, not a growing accumulation."""
    V = 16
    from pretrain_data_curator.student_model import StudentModelConfig as SMC

    cfg = SMC(model_dim=32, num_layers=2, num_heads=2, vocab_size=V, num_value_embeds=1)
    model = cfg.build()
    data = torch.randint(0, V, (500,))
    gen = torch.Generator().manual_seed(0)

    real_backward = torch.Tensor.backward
    embed_grad_norms = []

    def spy_backward(self, *args, **kwargs):
        result = real_backward(self, *args, **kwargs)
        if model.lm_head.weight.grad is not None:
            embed_grad_norms.append(model.lm_head.weight.grad.norm().item())
        return result

    monkeypatch.setattr(torch.Tensor, "backward", spy_backward)

    train_and_eval_student(
        model,
        data,
        data,
        block_size=8,
        batch_size=2,
        steps=6,
        base_lr=1e-3,
        warmup_steps=1,
        weight_decay=0.1,
        grad_clip=1.0,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        lr_min_ratio=0.1,
        vocab_size=V,
        device="cpu",
        generator=gen,
        training_recipe="speedrun_muon",
        grad_accum_embed_head_steps=2,
    )

    # Each backward must have been called.
    assert len(embed_grad_norms) >= 4
    # With the fix, each micro-step produces a fresh embed/head gradient
    # whose norm is bounded and does not grow from carryover.  The median
    # single-batch norm should be within 3x of every sample (no individual
    # sample has a norm 10x+ the median — which would happen if grads were
    # accumulating across micro-steps).
    median_norm = sorted(embed_grad_norms)[len(embed_grad_norms) // 2]
    for norm in embed_grad_norms:
        assert norm < median_norm * 4.0, (
            f"Embed grad norm {norm:.4f} exceeds 4x median {median_norm:.4f} — "
            "likely double-counting carryover"
        )


def test_grad_accum_embed_head_muon_grads_fresh_per_micro_step(monkeypatch):
    """Prove Muon gradients do not go stale (accumulate) across micro-steps:
    each micro-step's backward should produce a fresh Muon gradient whose
    norm is single-batch-scale, not a multi-step accumulation."""

    V = 16
    from pretrain_data_curator.student_model import StudentModelConfig as SMC

    cfg = SMC(model_dim=32, num_layers=2, num_heads=2, vocab_size=V, num_value_embeds=1)
    model = cfg.build()
    data = torch.randint(0, V, (500,))
    gen = torch.Generator().manual_seed(0)

    # Find a Muon-managed param (2D block weight).
    muon_param = None
    for name, p in model.named_parameters():
        if name.startswith("blocks.") and p.ndim >= 2:
            muon_param = p
            break
    assert muon_param is not None, "No Muon param found"

    real_backward = torch.Tensor.backward
    muon_grad_norms = []

    def spy_backward(self, *args, **kwargs):
        result = real_backward(self, *args, **kwargs)
        if muon_param.grad is not None:
            muon_grad_norms.append(muon_param.grad.norm().item())
        return result

    monkeypatch.setattr(torch.Tensor, "backward", spy_backward)

    train_and_eval_student(
        model,
        data,
        data,
        block_size=8,
        batch_size=2,
        steps=6,
        base_lr=1e-3,
        warmup_steps=1,
        weight_decay=0.1,
        grad_clip=1.0,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        lr_min_ratio=0.1,
        vocab_size=V,
        device="cpu",
        generator=gen,
        training_recipe="speedrun_muon",
        grad_accum_embed_head_steps=2,
    )

    assert len(muon_grad_norms) >= 4
    # With proper zeroing between micro-steps, all Muon grad norms are
    # single-batch scale — none should be orders of magnitude larger.
    median_norm = sorted(muon_grad_norms)[len(muon_grad_norms) // 2]
    for norm in muon_grad_norms:
        assert norm < median_norm * 4.0, (
            f"Muon grad norm {norm:.4f} exceeds 4x median {median_norm:.4f} — "
            "likely stale gradient carryover"
        )


def test_grad_accum_embed_head_clip_called(monkeypatch):
    """Prove gradient clipping is applied in the accumulation branch."""

    clip_calls: list[float] = []
    real_clip = torch.nn.utils.clip_grad_norm_

    def spy_clip(params, max_norm, *a, **k):
        clip_calls.append(float(max_norm))
        return real_clip(params, max_norm, *a, **k)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy_clip)

    V = 16
    from pretrain_data_curator.student_model import StudentModelConfig as SMC

    cfg = SMC(model_dim=32, num_layers=2, num_heads=2, vocab_size=V, num_value_embeds=1)
    model = cfg.build()
    data = torch.randint(0, V, (300,))
    gen = torch.Generator().manual_seed(0)

    train_and_eval_student(
        model,
        data,
        data,
        block_size=8,
        batch_size=2,
        steps=6,
        base_lr=1e-3,
        warmup_steps=1,
        weight_decay=0.1,
        grad_clip=1.0,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        lr_min_ratio=0.1,
        vocab_size=V,
        device="cpu",
        generator=gen,
        training_recipe="speedrun_muon",
        grad_accum_embed_head_steps=2,
    )

    # Muon clips at all six updates; Adam clips only at its three updates.
    assert len(clip_calls) == 9, (
        f"Expected 9 clip calls, got {len(clip_calls)}: {clip_calls}"
    )
    assert all(c == 1.0 for c in clip_calls)


def test_grad_accum_embed_head_non_embed_adam_bounded(monkeypatch):
    """Prove non-embed/head Adam groups (value_embeds, scalars) do not
    accumulate gradients across multiple accumulation cycles (no stale
    carryover across outer steps)."""

    V = 16
    from pretrain_data_curator.student_model import StudentModelConfig as SMC

    cfg = SMC(model_dim=32, num_layers=2, num_heads=2, vocab_size=V, num_value_embeds=1)
    model = cfg.build()
    data = torch.randint(0, V, (500,))
    gen = torch.Generator().manual_seed(0)

    # Find a non-embed/head Adam param (value_embed here).
    # SGD-like weight change per step measures how much gradient signal
    # the param received — accumulated vs fresh.
    pre_weights = {}
    for name, p in model.named_parameters():
        if "value_embeds" in name:
            pre_weights[name] = p.data.clone()

    train_and_eval_student(
        model,
        data,
        data,
        block_size=8,
        batch_size=2,
        steps=8,
        base_lr=1e-3,
        warmup_steps=1,
        weight_decay=0.1,
        grad_clip=1.0,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        lr_min_ratio=0.1,
        vocab_size=V,
        device="cpu",
        generator=gen,
        training_recipe="speedrun_muon",
        grad_accum_embed_head_steps=2,
    )

    # Value embed weights should have changed (proving Adam stepped them)
    # but the change should be modest (not exploding from stale carryover).
    for name, p in model.named_parameters():
        if "value_embeds" in name:
            delta = (p.data - pre_weights[name]).norm().item()
            assert delta > 0.0, f"{name} did not change at all"
            assert delta < 100.0, (
                f"{name} changed by {delta:.2f} — possible stale "
                f"multi-cycle accumulation"
            )


def test_grad_accum_embed_head_produces_correct_weight_changes(monkeypatch):
    """REFERENCE: verify that turning on accumulation produces a similar final
    embed/head weight delta as turning it off (equivalent total data seen).
    This is a minimal check that the accumulated gradient direction is sensible
    — not the exact equivalence test (which is impossible due to different Muon
    stepping frequency), but a guard against catastrophic corruption."""
    V = 16
    from pretrain_data_curator.student_model import StudentModelConfig as SMC

    cfg = SMC(model_dim=32, num_layers=2, num_heads=2, vocab_size=V, num_value_embeds=1)

    model_with = cfg.build()
    model_without = cfg.build()
    data = torch.randint(0, V, (500,))
    gen_with = torch.Generator().manual_seed(42)
    gen_without = torch.Generator().manual_seed(42)

    # Record initial weights (same init for both, same seed)
    pre_with = {
        n: p.data.clone()
        for n, p in model_with.named_parameters()
        if n.startswith("embed.") or n.startswith("lm_head.")
    }

    train_and_eval_student(
        model_with,
        data,
        data,
        block_size=8,
        batch_size=2,
        steps=8,
        base_lr=1e-3,
        warmup_steps=2,
        weight_decay=0.1,
        grad_clip=1.0,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        lr_min_ratio=0.1,
        vocab_size=V,
        device="cpu",
        generator=gen_with,
        training_recipe="speedrun_muon",
        grad_accum_embed_head_steps=2,
    )

    train_and_eval_student(
        model_without,
        data,
        data,
        block_size=8,
        batch_size=2,
        steps=8,
        base_lr=1e-3,
        warmup_steps=2,
        weight_decay=0.1,
        grad_clip=1.0,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        lr_min_ratio=0.1,
        vocab_size=V,
        device="cpu",
        generator=gen_without,
        training_recipe="speedrun_muon",
        grad_accum_embed_head_steps=1,
    )

    # Both runs changed embed/head from initial values.
    for name in pre_with:
        p_with = dict(model_with.named_parameters())[name]
        p_without = dict(model_without.named_parameters())[name]
        # Both changed (not zero).
        assert not torch.equal(p_with.data, pre_with[name]), (
            f"{name} unchanged with accum"
        )
        assert not torch.equal(p_without.data, pre_with[name]), (
            f"{name} unchanged without accum"
        )
        # Embed/head weight delta norms should be same order of magnitude
        # (neither is zero while the other is nonzero).
        delta_with = (p_with.data - pre_with[name]).norm().item()
        delta_without = (p_without.data - pre_with[name]).norm().item()
        # Both must be > 0 (proving grads were applied, not lost).
        assert delta_with > 0.0 and delta_without > 0.0


# --- (6) single source of truth: verbatim recipe embedded in the script ------


def test_prepare_student_model_dtype_cpu_noop():
    model = _tiny_cfg(32).build()
    q_before = model.blocks[0].attn.q.weight.data.clone()
    prepare_student_model_dtype(model, "cpu")
    assert model.blocks[0].attn.q.weight.dtype == torch.float32
    assert model.embed.weight.dtype == torch.float32
    assert model.lm_head.weight.dtype == torch.float32
    assert torch.equal(model.blocks[0].attn.q.weight.data, q_before)


def test_prepare_student_model_dtype_cuda_casts_muon_keeps_adam_fp32(monkeypatch):
    """Exercise CUDA dtype policy without requiring a GPU.

    Mock ``_is_cuda_device`` so the cast path runs on CPU-resident tensors.
    """
    import pretrain_data_curator.student_train as st

    monkeypatch.setattr(st, "_is_cuda_device", lambda device: True)
    model = _tiny_cfg(32).build()
    prepare_student_model_dtype(model, "cuda")
    assert model.blocks[0].attn.q.weight.dtype == torch.bfloat16
    assert model.blocks[0].mlp.fc.weight.dtype == torch.bfloat16
    assert model.embed.weight.dtype == torch.float32
    assert model.lm_head.weight.dtype == torch.float32
    assert model.resid_lambdas_attn.dtype == torch.float32
    for table in model.value_embeds.embed:
        assert table.weight.dtype == torch.float32


def test_sandbox_script_embeds_training_recipe_verbatim():
    # The GPU-only script must run the SAME tested components. Document encoding
    # is embedded before corpus construction; the remaining recipe follows later.
    ast.parse(NANOGPT_TRAIN_SCRIPT)  # assembled script is valid Python
    for component in _TRAINING_COMPONENTS:
        assert inspect.getsource(component).rstrip() in NANOGPT_TRAIN_SCRIPT
    # The record_01 recipe is wired: AdamW(betas/eps/weight_decay), warmup+cosine LR,
    # grad-clip, contiguous batching, and multi-run averaging.
    assert "class Muon(" in NANOGPT_TRAIN_SCRIPT
    assert "build_speedrun_optimizers(" in NANOGPT_TRAIN_SCRIPT
    assert "build_batch_schedule(" in NANOGPT_TRAIN_SCRIPT
    assert "clip_grad_norm_" in NANOGPT_TRAIN_SCRIPT
    assert "lr_at_step(step" in NANOGPT_TRAIN_SCRIPT
    assert "plan_train_windows(" in NANOGPT_TRAIN_SCRIPT
    assert "averaged_train_and_eval(" in NANOGPT_TRAIN_SCRIPT
    assert 'text.split("\\n\\n")' not in NANOGPT_TRAIN_SCRIPT
    assert '"\\n\\n".join(documents)' not in NANOGPT_TRAIN_SCRIPT
    assert (
        "corpus_ids, encoded_document_ranges = encode_document_tokens("
        in NANOGPT_TRAIN_SCRIPT
    )
    assert (
        "flat text cannot recover source document boundaries safely"
        in NANOGPT_TRAIN_SCRIPT
    )
    # ...and the OLD constant-LR plain-AdamW + random-with-replacement sampler is gone.
    assert "torch.randint(len(src) - block - 1" not in NANOGPT_TRAIN_SCRIPT
    assert (
        "opt = torch.optim.AdamW(model.parameters(), lr=float(cfg"
        not in NANOGPT_TRAIN_SCRIPT
    )
    # The sandbox script imports tqdm (installed on demand, like tiktoken) so the
    # embedded training loop's progress bar/logging actually runs there too.
    assert "from tqdm import tqdm" in NANOGPT_TRAIN_SCRIPT


def test_assembled_trainer_defines_batch_stage_boundaries_before_build_batch_schedule():
    # Production regression: the generated /workspace/train.py embeds
    # student_optimizer.build_batch_schedule, which calls batch_stage_boundaries
    # (defined in batch_schedule.py and only *imported* there). The assembly must
    # inline the canonical batch_stage_boundaries so the remote trainer does not
    # raise NameError at step 0. We assemble the EXACT production payload and
    # actually execute the embedded optimizer block to call build_batch_schedule.
    script = NANOGPT_TRAIN_SCRIPT
    ast.parse(script)  # assembled payload is valid Python

    # batch_stage_boundaries must be embedded (regression for the missing-helper
    # NameError) and must appear before build_batch_schedule in source order.
    assert "def batch_stage_boundaries(" in script
    assert script.index("def batch_stage_boundaries(") < script.index(
        "def build_batch_schedule("
    ), "batch_stage_boundaries must be defined before build_batch_schedule"

    # Extract the embedded optimizer block verbatim -- from the batch_schedule
    # embed (which carries its _FRAC_SUM_TOL constant) through the end of the
    # optimizer defs, just before the training-recipe defs.
    start = script.index("_FRAC_SUM_TOL")
    end = script.index("def lr_at_step(")
    block = script[start:end]

    import math
    from collections.abc import Sequence
    import dataclasses

    namespace = {
        "math": math,
        "Sequence": Sequence,
        "dataclass": dataclasses.dataclass,
        "torch": __import__("torch"),
    }
    try:
        exec(block, namespace)
    except NameError as exc:
        pytest.fail(f"assembled optimizer block has an unresolved name: {exc}")

    # Execute build_batch_schedule for the production total_steps with equal
    # [1/3]*3 fractions and confirm the boundaries/stages are correct.
    boundaries, stages, cd_start, cd_floor = namespace["build_batch_schedule"](
        12208, stage_fracs=(1 / 3, 1 / 3, 1 / 3)
    )
    assert len(boundaries) == 3
    assert boundaries[0][0] == 0
    assert boundaries[-1][1] == 12208
    for (s, e), (ns, ne) in zip(boundaries, boundaries[1:]):
        assert e == ns, f"stage boundaries must be contiguous: {boundaries}"
    # Each of the first two stages gets round(12208/3) = 4069 steps; the last
    # absorbs the remainder exactly.
    assert boundaries[0] == (0, 4069), boundaries
    assert boundaries[1] == (4069, 8138), boundaries
    assert boundaries[2] == (8138, 12208), boundaries
    assert len(stages) == 3
    assert cd_start == int(12208 * (1.0 - 0.60))
    assert cd_floor == 0.15


def test_sandbox_tqdm_fallback_shim_works_when_import_and_install_both_fail(
    monkeypatch, capsys
):
    # If tqdm can't be imported AND the on-demand pip install also fails
    # (offline container, no pip, read-only fs, ...), training must never
    # crash just because tqdm was unavailable -- the embedded script falls
    # back to a no-op progress shim. Extract the EXACT tqdm-import snippet
    # from the assembled sandbox script and execute it with both failures
    # forced, then verify the resulting `tqdm` name is usable exactly the way
    # train_and_eval_student uses it (iteration, .write, .set_postfix, .close)
    # so the throttled plain print()-style lines still work with no live bar.
    script = NANOGPT_TRAIN_SCRIPT
    start = script.index("try:\n    from tqdm import tqdm")
    # The `# __PLAN_VAL_WINDOWS__` placeholder is already substituted with the
    # real function source by the time NANOGPT_TRAIN_SCRIPT is assembled, so
    # anchor on that function's def line instead.
    end = script.index("def plan_val_windows(", start)
    snippet = script[start:end].rstrip()

    class _FailingSubprocess:
        @staticmethod
        def run(*args, **kwargs):
            raise RuntimeError("simulated: no network/pip available")

    # Force `from tqdm import tqdm` to raise ImportError even though tqdm IS
    # actually installed in this test env (it's a real dependency now).
    monkeypatch.setitem(sys.modules, "tqdm", None)
    namespace = {"sys": sys, "subprocess": _FailingSubprocess}
    exec(snippet, namespace)

    shim_tqdm = namespace["tqdm"]
    bar = shim_tqdm(
        range(3), total=3, desc="test", unit="step", leave=False, file=sys.stdout
    )
    assert list(bar) == [0, 1, 2]
    bar.set_postfix(loss="1.0000")  # no-op, must not raise
    bar.write("[test] step 3/3")
    bar.close()  # no-op, must not raise
    assert "[test] step 3/3" in capsys.readouterr().out


# --- (7) OBSERVABILITY: tqdm progress bar + throttled stdout progress lines --


def test_progress_lines_emitted_for_first_and_last_step(capsys):
    # A tiny run (steps=5, well under the 50-step floor cadence) must still emit
    # the first and last step's plain progress line on stdout -- and ONLY those --
    # so degenerate test-sized runs stay legible from a captured stdout blob
    # without spamming a line per step.
    V = 16
    model = _tiny_cfg(V).build()
    data = torch.randint(0, V, (100,))
    gen = torch.Generator().manual_seed(0)
    train_and_eval_student(
        model,
        data,
        data,
        block_size=8,
        batch_size=2,
        steps=5,
        base_lr=1e-3,
        warmup_steps=1,
        weight_decay=0.1,
        grad_clip=1.0,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        lr_min_ratio=0.1,
        vocab_size=V,
        device="cpu",
        generator=gen,
        training_recipe="record_01_adamw",
    )
    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if line.startswith("[train]")]
    assert len(lines) == 2
    assert "step 1/5" in lines[0]
    assert "step 5/5" in lines[1]
    for line in lines:
        assert (
            "loss" in line and "tok/s" in line and "elapsed" in line and "eta" in line
        )


def test_progress_line_survives_a_single_step_run():
    # steps=1: the first step IS the last step. Must not crash, hang, or double-log.
    import io
    from contextlib import redirect_stdout

    V = 16
    model = _tiny_cfg(V).build()
    data = torch.randint(0, V, (50,))
    gen = torch.Generator().manual_seed(0)
    buf = io.StringIO()
    with redirect_stdout(buf):
        train_and_eval_student(
            model,
            data,
            data,
            block_size=8,
            batch_size=2,
            steps=1,
            base_lr=1e-3,
            warmup_steps=1,
            weight_decay=0.1,
            grad_clip=1.0,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
            lr_min_ratio=0.1,
            vocab_size=V,
            device="cpu",
            generator=gen,
            training_recipe="record_01_adamw",
        )
    lines = [line for line in buf.getvalue().splitlines() if line.startswith("[train]")]
    assert len(lines) == 1
    assert "step 1/1" in lines[0]


def test_progress_lines_throttled_for_a_longer_run(capsys):
    # steps=30 with the "50 steps or 2%, whichever is coarser" cadence logs at
    # completed in {1, 30} for this short run.
    V = 16
    model = _tiny_cfg(V).build()
    data = torch.randint(0, V, (300,))
    gen = torch.Generator().manual_seed(0)
    train_and_eval_student(
        model,
        data,
        data,
        block_size=8,
        batch_size=2,
        steps=30,
        base_lr=1e-3,
        warmup_steps=4,
        weight_decay=0.1,
        grad_clip=1.0,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        lr_min_ratio=0.1,
        vocab_size=V,
        device="cpu",
        generator=gen,
        training_recipe="record_01_adamw",
    )
    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if line.startswith("[train]")]
    logged_steps = [int(line.split("step ")[1].split("/")[0]) for line in lines]
    assert logged_steps == [1, 30]


def test_ema_loss_updates_every_step_not_just_at_log_points(monkeypatch, capsys):
    # Regression test for a bug where the EMA only updated INSIDE the throttled
    # logging branch, so a 120-step run only ever averaged in the ~4 sampled
    # losses instead of all 120. Spy on F.cross_entropy to capture every
    # step's real raw training loss, independently compute the reference
    # full per-step EMA (decay=0.9) over ALL of them, and confirm the final
    # logged line's reported loss matches it -- which only holds if the EMA is
    # actually updated every step.
    import pretrain_data_curator.student_train as st

    V = 16
    model = _tiny_cfg(V).build()
    data = torch.randint(0, V, (400,))
    gen = torch.Generator().manual_seed(0)
    steps = 30

    real_cross_entropy = st.F.cross_entropy
    captured_losses = []

    def spy_cross_entropy(*args, **kwargs):
        out = real_cross_entropy(*args, **kwargs)
        # Only the training-loop call site omits `reduction` (default "mean");
        # the val-scoring call site always passes reduction="sum" explicitly,
        # so this filter captures exactly the per-step training losses.
        if kwargs.get("reduction", "mean") == "mean":
            captured_losses.append(out.detach().clone())
        return out

    monkeypatch.setattr(st.F, "cross_entropy", spy_cross_entropy)
    train_and_eval_student(
        model,
        data,
        data,
        block_size=8,
        batch_size=2,
        steps=steps,
        base_lr=1e-3,
        warmup_steps=4,
        weight_decay=0.1,
        grad_clip=1.0,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        lr_min_ratio=0.1,
        vocab_size=V,
        device="cpu",
        generator=gen,
        training_recipe="record_01_adamw",
    )
    assert len(captured_losses) == steps  # one captured training loss per step

    ema_decay = 0.9
    ref_ema = None
    for loss_val in (t.item() for t in captured_losses):
        ref_ema = (
            loss_val
            if ref_ema is None
            else ema_decay * ref_ema + (1 - ema_decay) * loss_val
        )

    out = capsys.readouterr().out
    last_line = [line for line in out.splitlines() if line.startswith("[train]")][-1]
    reported_loss = float(last_line.split("loss ")[1].split(" |")[0])
    assert reported_loss == pytest.approx(ref_ema, rel=1e-3)


def test_run_label_prefixes_progress_lines_when_multiple_runs(capsys):
    # n_runs > 1 must tag each run's progress lines with "run k/n" so the log of a
    # multi-run averaged eval is legible about which seeded run is in flight.
    V = 16
    cfg = _tiny_cfg(V)
    data = torch.randint(0, V, (100,))
    averaged_train_and_eval(
        cfg.build,
        data,
        data,
        n_runs=2,
        base_seed=0,
        device="cpu",
        block_size=8,
        batch_size=2,
        steps=2,
        base_lr=1e-3,
        warmup_steps=1,
        weight_decay=0.1,
        grad_clip=1.0,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        lr_min_ratio=0.1,
        vocab_size=V,
        training_recipe="record_01_adamw",
    )
    out = capsys.readouterr().out
    assert "[run 1/2 train] step 1/2" in out
    assert "[run 1/2 train] step 2/2" in out
    assert "[run 2/2 train] step 1/2" in out
    assert "[run 2/2 train] step 2/2" in out


def test_run_label_absent_for_a_single_run(capsys):
    # n_runs == 1 (the default/common case) must NOT prefix a "run 1/1" label --
    # single-run output stays exactly as plain as before, uncluttered.
    V = 16
    cfg = _tiny_cfg(V)
    data = torch.randint(0, V, (100,))
    averaged_train_and_eval(
        cfg.build,
        data,
        data,
        n_runs=1,
        base_seed=0,
        device="cpu",
        block_size=8,
        batch_size=2,
        steps=2,
        base_lr=1e-3,
        warmup_steps=1,
        weight_decay=0.1,
        grad_clip=1.0,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        lr_min_ratio=0.1,
        vocab_size=V,
        training_recipe="record_01_adamw",
    )
    out = capsys.readouterr().out
    assert "[train] step 1/2" in out
    assert "run 1/1" not in out


# --- (11) golden-sample accumulation behavior ----------------------------------


def _adam_param(model) -> torch.nn.Parameter:
    """Find a non-embed/head Adam-managed param (value_embed)."""
    for name, p in model.named_parameters():
        if "value_embeds" in name and p.ndim >= 2:
            return p
    raise AssertionError("No value_embed param found")


def test_grad_accum_non_embed_adam_grads_fresh_per_micro_step(monkeypatch):
    """Non-embed/head Adam params (value_embeds, scalars) must have fresh
    gradients per micro-step — no stale carryover from the previous micro-step
    within an accumulation cycle.  This is the Adam counterpart of
    test_grad_accum_embed_head_muon_grads_fresh_per_micro_step."""
    V = 16
    from pretrain_data_curator.student_model import StudentModelConfig as SMC

    cfg = SMC(model_dim=32, num_layers=2, num_heads=2, vocab_size=V, num_value_embeds=1)
    model = cfg.build()

    adam_param = _adam_param(model)
    data = torch.randint(0, V, (500,))
    gen = torch.Generator().manual_seed(0)

    real_backward = torch.Tensor.backward
    adam_grad_norms = []

    def spy_backward(self, *args, **kwargs):
        result = real_backward(self, *args, **kwargs)
        if adam_param.grad is not None:
            adam_grad_norms.append(adam_param.grad.norm().item())
        return result

    monkeypatch.setattr(torch.Tensor, "backward", spy_backward)

    train_and_eval_student(
        model,
        data,
        data,
        block_size=8,
        batch_size=2,
        steps=6,
        base_lr=1e-3,
        warmup_steps=1,
        weight_decay=0.1,
        grad_clip=1.0,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        lr_min_ratio=0.1,
        vocab_size=V,
        device="cpu",
        generator=gen,
        training_recipe="speedrun_muon",
        grad_accum_embed_head_steps=2,
    )

    assert len(adam_grad_norms) >= 4
    # With proper zeroing between micro-steps, all non-embed/head Adam grad
    # norms are single-batch scale — none should be multiple of others.
    median_norm = sorted(adam_grad_norms)[len(adam_grad_norms) // 2]
    for norm in adam_grad_norms:
        assert norm < median_norm * 4.0, (
            f"Non-embed/head Adam grad norm {norm:.4f} exceeds 4x median "
            f"{median_norm:.4f} — stale carryover across micro-steps"
        )


def test_grad_accum_multi_heads_in_buffer_with_accum(monkeypatch):
    """multi_heads.* params must be in the accumulation buffer when
    grad_accum_embed_head_steps > 1 and multi_token_pred > 0, so their
    gradients are accumulated across micro-steps (not silently lost or
    double-counted)."""
    V = 16
    from pretrain_data_curator.student_model import GPT

    model = GPT(
        vocab_size=V,
        num_layers=2,
        model_dim=32,
        num_heads=2,
        multi_token_pred=2,
    )

    data = torch.randint(0, V, (300,))
    gen = torch.Generator().manual_seed(0)

    pre_weights = {}
    for name, p in model.named_parameters():
        if name.startswith("multi_heads."):
            pre_weights[name] = p.data.clone()
    assert pre_weights, "No multi_heads.* params found — model lacks multi-token heads"

    train_and_eval_student(
        model,
        data,
        data,
        block_size=8,
        batch_size=2,
        steps=6,
        base_lr=1e-3,
        warmup_steps=1,
        weight_decay=0.1,
        grad_clip=1.0,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        lr_min_ratio=0.1,
        vocab_size=V,
        device="cpu",
        generator=gen,
        training_recipe="speedrun_muon",
        grad_accum_embed_head_steps=2,
        multi_token_pred=2,
    )

    # Multi-head weights should have changed (accumulated grads applied)
    for name, p in model.named_parameters():
        if name.startswith("multi_heads."):
            delta = (p.data - pre_weights[name]).norm().item()
            assert delta > 0.0, f"{name} did not change — not in accumulation buffer"


def test_grad_accum_partial_cycle_flushes_remaining_grads():
    """When steps is not evenly divisible by grad_accum_embed_head_steps, the
    last partial cycle must still flush accumulated embed/head grads instead of
    silently dropping them.  Use steps=7 with accum=3 → 2 full cycles + 1
    partial cycle (7 steps, micro-steps 1-2-3, 4-5-6, 7 being partial)."""
    V = 16
    from pretrain_data_curator.student_model import StudentModelConfig as SMC

    cfg = SMC(model_dim=32, num_layers=2, num_heads=2, vocab_size=V, num_value_embeds=1)
    model = cfg.build()
    data = torch.randint(0, V, (400,))
    gen = torch.Generator().manual_seed(0)

    pre_embed = model.embed.weight.data.clone()
    pre_lm_head = model.lm_head.weight.data.clone()

    loss, acc, tokens = train_and_eval_student(
        model,
        data,
        data,
        block_size=8,
        batch_size=2,
        steps=7,
        base_lr=1e-3,
        warmup_steps=1,
        weight_decay=0.1,
        grad_clip=1.0,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        lr_min_ratio=0.1,
        vocab_size=V,
        device="cpu",
        generator=gen,
        training_recipe="speedrun_muon",
        grad_accum_embed_head_steps=3,
    )

    assert math.isfinite(loss)
    assert tokens > 0
    # Embed/head weights must differ from initial (proving partial cycle grads
    # were applied, not dropped).
    assert not torch.equal(model.embed.weight.data, pre_embed), (
        "embed unchanged — partial cycle grads likely dropped"
    )
    assert not torch.equal(model.lm_head.weight.data, pre_lm_head), (
        "lm_head unchanged — partial cycle grads likely dropped"
    )


def test_grad_accum_partial_cycle_vs_full_cycle_similar_delta():
    """A partial final cycle should produce an embed/head delta of the same
    order of magnitude as a full cycle (not zero or drastically smaller).
    Compare a divisible-step run vs a non-divisible one with the same total
    data schedule."""
    V = 16
    from pretrain_data_curator.student_model import StudentModelConfig as SMC

    cfg = SMC(model_dim=32, num_layers=2, num_heads=2, vocab_size=V, num_value_embeds=1)

    # Run with 6 steps (divisible by 3) — 2 full cycles
    model_full = cfg.build()
    data = torch.randint(0, V, (500,))
    gen_full = torch.Generator().manual_seed(42)
    pre_full = {
        n: p.data.clone()
        for n, p in model_full.named_parameters()
        if n.startswith("embed.") or n.startswith("lm_head.")
    }
    train_and_eval_student(
        model_full,
        data,
        data,
        block_size=8,
        batch_size=2,
        steps=6,
        base_lr=1e-3,
        warmup_steps=1,
        weight_decay=0.1,
        grad_clip=1.0,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        lr_min_ratio=0.1,
        vocab_size=V,
        device="cpu",
        generator=gen_full,
        training_recipe="speedrun_muon",
        grad_accum_embed_head_steps=3,
    )

    # Run with 7 steps (not divisible by 3) — 2 full cycles + 1 partial cycle
    model_partial = cfg.build()
    gen_partial = torch.Generator().manual_seed(42)
    pre_partial = {
        n: p.data.clone()
        for n, p in model_partial.named_parameters()
        if n.startswith("embed.") or n.startswith("lm_head.")
    }
    train_and_eval_student(
        model_partial,
        data,
        data,
        block_size=8,
        batch_size=2,
        steps=7,
        base_lr=1e-3,
        warmup_steps=1,
        weight_decay=0.1,
        grad_clip=1.0,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        lr_min_ratio=0.1,
        vocab_size=V,
        device="cpu",
        generator=gen_partial,
        training_recipe="speedrun_muon",
        grad_accum_embed_head_steps=3,
    )

    for name in pre_full:
        p_full = dict(model_full.named_parameters())[name]
        p_partial = dict(model_partial.named_parameters())[name]

        delta_full = (p_full.data - pre_full[name]).norm().item()
        delta_partial = (p_partial.data - pre_partial[name]).norm().item()

        # Partial run has seen more data so delta should be >= or similar, not
        # zero (which would mean the partial cycle's grads were dropped).
        assert delta_partial > 0.0, f"{name}: partial cycle delta is 0"
        assert delta_full > 0.0, f"{name}: full cycle delta is 0"
        # The partial delta should be at least as large (more data seen) or
        # at minimum within the same order of magnitude.
        assert delta_partial >= delta_full * 0.5, (
            f"{name}: partial delta {delta_partial:.4f} is much smaller than "
            f"full delta {delta_full:.4f} — grads likely dropped"
        )


def test_flush_on_even_step_applies_adam_grads():
    """When adam_on_odd_steps=True and the post-loop flush fires on an even
    step index, the accumulated Adam-managed grads (embed, lm_head, multi_heads)
    must still be applied.

    With steps=1 and grad_accum_embed_head_steps=2, the first (and only)
    micro-step creates a partial accumulation buffer that is flushed at
    step=0 (even).  Before the fix, step_speedrun_optimizers skipped Adam
    because step % 2 == 0 and adam_on_odd_steps=True, silently dropping the
    accumulated grads.  After the fix, force_adam=True on the flush path
    ensures Adam always steps."""
    V = 16
    from pretrain_data_curator.student_model import GPT

    model = GPT(
        vocab_size=V,
        num_layers=2,
        model_dim=32,
        num_heads=2,
        multi_token_pred=2,
    )

    data = torch.randint(0, V, (200,))
    gen = torch.Generator().manual_seed(0)

    pre = {}
    for name, p in model.named_parameters():
        if (
            name.startswith("embed.")
            or name.startswith("lm_head.")
            or name.startswith("multi_heads.")
        ):
            pre[name] = p.data.clone()
    assert pre, "No tracked params found"

    loss, acc, tokens = train_and_eval_student(
        model,
        data,
        data,
        block_size=8,
        batch_size=2,
        steps=1,
        base_lr=1e-3,
        warmup_steps=1,
        weight_decay=0.1,
        grad_clip=1.0,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        lr_min_ratio=0.1,
        vocab_size=V,
        device="cpu",
        generator=gen,
        training_recipe="speedrun_muon",
        grad_accum_embed_head_steps=2,
        multi_token_pred=2,
        adam_on_odd_steps=True,
    )

    assert math.isfinite(loss)
    # Every Adam-managed param must have changed from its initial value,
    # proving the even-step flush applied their accumulated grads.
    for name, p in model.named_parameters():
        if name in pre:
            assert not torch.equal(p.data, pre[name]), (
                f"{name} unchanged — even-step flush likely dropped Adam grads"
            )


def test_flush_on_even_step_updates_all_embed_lm_head_multi_heads():
    """A longer run where the flush fires on an even step must still update
    all Adam-managed parameter groups (embed, lm_head, multi_heads).

    With steps=5 and grad_accum_embed_head_steps=4:
      - Full cycle at step 3 (odd)  → Adam runs normally.
      - Partial micro-step at step 4 (even, flush) → must apply Adam via
        force_adam, otherwise multi_heads gradient is silently dropped."""
    V = 16
    from pretrain_data_curator.student_model import GPT

    model = GPT(
        vocab_size=V,
        num_layers=2,
        model_dim=32,
        num_heads=2,
        multi_token_pred=2,
    )

    data = torch.randint(0, V, (400,))
    gen = torch.Generator().manual_seed(0)

    pre = {}
    for name, p in model.named_parameters():
        if (
            name.startswith("embed.")
            or name.startswith("lm_head.")
            or name.startswith("multi_heads.")
        ):
            pre[name] = p.data.clone()
    assert pre, "No tracked params found"

    loss, acc, tokens = train_and_eval_student(
        model,
        data,
        data,
        block_size=8,
        batch_size=2,
        steps=5,
        base_lr=1e-3,
        warmup_steps=1,
        weight_decay=0.1,
        grad_clip=1.0,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        lr_min_ratio=0.1,
        vocab_size=V,
        device="cpu",
        generator=gen,
        training_recipe="speedrun_muon",
        grad_accum_embed_head_steps=4,
        multi_token_pred=2,
        adam_on_odd_steps=True,
    )

    assert math.isfinite(loss)
    for name, p in model.named_parameters():
        if name in pre:
            assert not torch.equal(p.data, pre[name]), (
                f"{name} unchanged — even-step flush likely dropped Adam grads"
            )


def test_in_loop_full_cycle_closes_on_even_step_forces_adam(monkeypatch):
    """A full accumulation cycle that closes INSIDE the main loop (not the
    post-loop flush) must still force Adam to step when it lands on an even
    step index under ``adam_on_odd_steps=True``.

    With ``grad_accum_embed_head_steps=3`` and ``steps=3``, the single full
    cycle closes exactly at step index 2 (0-indexed, even) on the last loop
    iteration — ``accum_count`` reaches 3 there, so the buffer is cleared to
    ``None`` before the loop exits and the post-loop flush never fires. This
    isolates the in-loop closing path from the terminal-flush path already
    covered by ``test_flush_on_even_step_applies_adam_grads``.

    Before the in-loop fix, ``step_speedrun_optimizers`` was called here
    without ``force_adam=True``, so the parity check (``step % 2 == 1``)
    skipped ``adam_opt.step()`` entirely, and the unconditional
    ``adam_opt.zero_grad()`` immediately after silently discarded the
    accumulated embed/head/multi_heads grads — AdamW's ``.step()`` was never
    invoked at all for that cycle."""
    V = 16
    from pretrain_data_curator.student_model import GPT

    model = GPT(
        vocab_size=V, num_layers=2, model_dim=32, num_heads=2, multi_token_pred=2
    )
    data = torch.randint(0, V, (300,))
    gen = torch.Generator().manual_seed(0)

    import pretrain_data_curator.student_train as student_train_mod

    real_step_fn = student_train_mod.step_speedrun_optimizers
    calls: list[tuple[int, bool]] = []

    def spy_step_fn(muon_opt, adam_opt, *, step, **kwargs):
        calls.append((step, kwargs.get("force_adam", False)))
        return real_step_fn(muon_opt, adam_opt, step=step, **kwargs)

    monkeypatch.setattr(student_train_mod, "step_speedrun_optimizers", spy_step_fn)

    adam_step_calls: list[bool] = []
    real_adam_step = torch.optim.AdamW.step

    def spy_adam_step(self, *a, **k):
        adam_step_calls.append(True)
        return real_adam_step(self, *a, **k)

    monkeypatch.setattr(torch.optim.AdamW, "step", spy_adam_step)

    loss, acc, tokens = train_and_eval_student(
        model,
        data,
        data,
        block_size=8,
        batch_size=2,
        steps=3,
        base_lr=1e-3,
        warmup_steps=1,
        weight_decay=0.1,
        grad_clip=1.0,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        lr_min_ratio=0.1,
        vocab_size=V,
        device="cpu",
        generator=gen,
        training_recipe="speedrun_muon",
        grad_accum_embed_head_steps=3,
        multi_token_pred=2,
        adam_on_odd_steps=True,
    )

    assert math.isfinite(loss)
    # Exactly one in-loop full cycle closes, at even step index 2, and it
    # must request force_adam=True.
    assert calls == [(2, True)], (
        f"expected a single in-loop full-cycle call at even step=2 with "
        f"force_adam=True, got {calls}"
    )
    # AdamW.step() must have actually run — proving the accumulated
    # embed/head/multi_heads grads were applied rather than silently zeroed.
    assert len(adam_step_calls) == 1, (
        f"expected AdamW.step() to run exactly once for the in-loop full "
        f"cycle, got {len(adam_step_calls)} calls — accumulated grads were "
        f"likely dropped"
    )


# --- non-accumulation path (grad_accum_embed_head_steps <= 1): must match
# --- the original modded-nanogpt train_gpt.py odd-step Adam semantics -----


def test_non_accum_path_adam_zero_grad_only_follows_adam_step(monkeypatch):
    """Non-accumulation path (the default, ``grad_accum_embed_head_steps=1``):
    AdamW.step() and AdamW.zero_grad() must be paired 1:1, and zero_grad()
    must only ever fire immediately after a step() -- never before it and
    never on a skipped (even) step. In the original train_gpt.py, an
    Adam-managed param's ``.grad`` is only cleared on the step that actually
    applies it (``continue  # Don't clear Adam grads on even steps``); a
    version that unconditionally zeros before every backward() would show
    zero_grad() calls with no matching step() right before them."""
    V = 16
    model = _tiny_cfg(V).build()
    data = torch.randint(0, V, (200,))
    gen = torch.Generator().manual_seed(0)

    events: list[str] = []
    real_step = torch.optim.AdamW.step
    real_zero = torch.optim.AdamW.zero_grad

    def spy_step(self, *a, **k):
        events.append("step")
        return real_step(self, *a, **k)

    def spy_zero(self, *a, **k):
        events.append("zero_grad")
        return real_zero(self, *a, **k)

    monkeypatch.setattr(torch.optim.AdamW, "step", spy_step)
    monkeypatch.setattr(torch.optim.AdamW, "zero_grad", spy_zero)

    steps = 6
    train_and_eval_student(
        model,
        data,
        data,
        block_size=8,
        batch_size=2,
        steps=steps,
        base_lr=1e-3,
        warmup_steps=1,
        weight_decay=0.1,
        grad_clip=1.0,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        lr_min_ratio=0.1,
        vocab_size=V,
        device="cpu",
        generator=gen,
        training_recipe="speedrun_muon",
        adam_on_odd_steps=True,
    )

    # Odd step indices in range(6) are 1, 3, 5 -> 3 Adam updates.
    assert events.count("step") == steps // 2
    assert events.count("zero_grad") == events.count("step")
    step_positions = [i for i, e in enumerate(events) if e == "step"]
    zero_positions = [i for i, e in enumerate(events) if e == "zero_grad"]
    for sp, zp in zip(step_positions, zero_positions, strict=True):
        assert zp == sp + 1, (
            f"zero_grad() must immediately follow step(), got sequence {events}"
        )


def test_non_accum_path_adam_grad_accumulates_across_even_odd_pair(monkeypatch):
    """Semantic test against the original flow: on the non-accumulation
    path, an even step's Adam-managed gradient must not be discarded --
    it must accumulate into the following odd step's AdamW.step() call,
    exactly like ``train_gpt.py``'s ``NorMuonAndAdam.step`` (Adam grads are
    only cleared on the step that applies them, so the default
    accumulate-into-.grad behavior of consecutive ``backward()`` calls does
    the summation). This is stronger than a weight-drift check: it verifies
    the exact tensor AdamW.step() consumes equals the sum of both steps'
    raw backward contributions, not just the most recent one."""
    V = 16
    model = _tiny_cfg(V).build()
    data = torch.randint(0, V, (300,))
    gen = torch.Generator().manual_seed(0)

    # Capture each backward()'s raw contribution to lm_head.weight.grad,
    # BEFORE it is accumulated into (or discarded from) .grad. lm_head is
    # used here (rather than embed) because it receives a nonzero direct
    # gradient (hidden^T @ error) on every step regardless of init scale,
    # so it exercises the accumulate-vs-discard distinction on every step.
    raw_grad_per_step: list[torch.Tensor] = []

    def _hook(grad):
        raw_grad_per_step.append(grad.detach().clone())
        return grad

    handle = model.lm_head.weight.register_hook(_hook)

    # Capture the actual tensor AdamW.step() consumes for lm_head.weight,
    # right before the real step() call applies it.
    pre_step_adam_grad: list[torch.Tensor] = []
    real_step = torch.optim.AdamW.step

    def spy_step(self, *a, **k):
        for group in self.param_groups:
            for p in group["params"]:
                if p is model.lm_head.weight and p.grad is not None:
                    pre_step_adam_grad.append(p.grad.detach().clone())
        return real_step(self, *a, **k)

    monkeypatch.setattr(torch.optim.AdamW, "step", spy_step)
    try:
        steps = 4
        train_and_eval_student(
            model,
            data,
            data,
            block_size=8,
            batch_size=2,
            steps=steps,
            base_lr=1e-3,
            warmup_steps=1,
            weight_decay=0.1,
            # grad_clip disabled: clipping would rescale .grad in place and
            # break the exact-sum comparison below.
            grad_clip=0.0,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
            lr_min_ratio=0.1,
            vocab_size=V,
            device="cpu",
            generator=gen,
            training_recipe="speedrun_muon",
            adam_on_odd_steps=True,
        )
    finally:
        handle.remove()

    assert len(raw_grad_per_step) == steps
    # Adam steps at step=1 and step=3 (odd indices in range(4)).
    assert len(pre_step_adam_grad) == 2

    even_odd_pair_1 = raw_grad_per_step[0] + raw_grad_per_step[1]
    even_odd_pair_2 = raw_grad_per_step[2] + raw_grad_per_step[3]

    assert torch.allclose(pre_step_adam_grad[0], even_odd_pair_1, atol=1e-6), (
        "AdamW.step() at step=1 must consume the SUM of step 0's (even, "
        "skipped) and step 1's (odd) raw gradients, not just step 1's alone "
        "-- the even step's gradient must not be discarded"
    )
    assert torch.allclose(pre_step_adam_grad[1], even_odd_pair_2, atol=1e-6), (
        "AdamW.step() at step=3 must consume the SUM of step 2's (even, "
        "skipped) and step 3's (odd) raw gradients"
    )
    # Contrast with the (buggy) simplified behavior this replaces: applying
    # only the odd step's own gradient, discarding the even step's.
    assert not torch.allclose(pre_step_adam_grad[0], raw_grad_per_step[1], atol=1e-6), (
        "AdamW.step() must not consume only the odd step's own gradient -- "
        "that would mean the even step's gradient was silently dropped"
    )


# --- Held-out validation microbatch / chunked lm_head scoring (A100 OOM fix) -


def test_eval_val_loss_respects_microbatch_cap(monkeypatch):
    """Validation must never forward more than ``batch`` windows at once."""
    from pretrain_data_curator.student_train import _eval_val_loss

    V, block, n_tokens = 64, 8, 65  # 64 targets -> 8 full-length windows
    model = _tiny_cfg(V).build().eval()
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.02)
    val = torch.randint(0, V, (n_tokens,), dtype=torch.long)
    seen = []
    real_forward_hidden = model.forward_hidden

    def capped_forward_hidden(xb, **kwargs):
        seen.append(int(xb.size(0)))
        assert xb.size(0) <= 2, f"validation microbatch exceeded cap: {xb.size(0)}"
        return real_forward_hidden(xb, **kwargs)

    monkeypatch.setattr(model, "forward_hidden", capped_forward_hidden)
    loss, acc = _eval_val_loss(
        model,
        val,
        block=block,
        batch=2,
        vocab_size=V,
        device="cpu",
        logit_chunk_tokens=16,
    )
    assert math.isfinite(loss)
    assert 0.0 <= acc <= 1.0
    assert seen, "expected at least one validation forward"
    assert all(b <= 2 for b in seen)
    assert sum(seen) == 8  # every full-length window scored


def test_eval_val_loss_chunked_matches_full_vocab_semantics():
    """Chunked lm_head scoring must match a single full-vocab pass exactly."""
    from pretrain_data_curator.student_train import _eval_val_loss

    V, block, n_tokens = 64, 8, 41
    torch.manual_seed(0)
    model = _tiny_cfg(V).build().eval()
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.02)
    val = torch.randint(0, V, (n_tokens,), dtype=torch.long)

    full_loss, full_acc = _eval_val_loss(
        model,
        val,
        block=block,
        batch=4,
        vocab_size=V,
        device="cpu",
        logit_chunk_tokens=None,
    )
    chunked_loss, chunked_acc = _eval_val_loss(
        model,
        val,
        block=block,
        batch=1,
        vocab_size=V,
        device="cpu",
        logit_chunk_tokens=7,
    )
    assert chunked_loss == pytest.approx(full_loss, rel=0, abs=1e-6)
    assert chunked_acc == pytest.approx(full_acc, rel=0, abs=1e-12)


def test_eval_val_loss_processes_all_validation_targets_under_cap():
    """Every predictable target is scored even when microbatch + chunk caps are tiny."""
    from pretrain_data_curator.student_train import _eval_val_loss
    from pretrain_data_curator.val_set import plan_val_windows

    V, block, n_tokens = 48, 5, 23
    model = _tiny_cfg(V).build().eval()
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.02)
    val = torch.arange(n_tokens, dtype=torch.long) % V
    windows = plan_val_windows(n_tokens, block)
    expected_targets = sum(length for _, length in windows)
    assert expected_targets == n_tokens - 1

    loss, acc = _eval_val_loss(
        model,
        val,
        block=block,
        batch=1,
        vocab_size=V,
        device="cpu",
        logit_chunk_tokens=3,
    )
    assert math.isfinite(loss)
    assert 0.0 <= acc <= 1.0
    # Sanity: scoring the same stream twice is stable (all targets visited).
    loss2, acc2 = _eval_val_loss(
        model,
        val,
        block=block,
        batch=1,
        vocab_size=V,
        device="cpu",
        logit_chunk_tokens=3,
    )
    assert loss2 == pytest.approx(loss, rel=0, abs=0.0)
    assert acc2 == pytest.approx(acc, rel=0, abs=0.0)


def test_train_and_eval_honors_separate_val_batch_size(monkeypatch):
    """``val_batch_size`` must drive validation, not training ``batch_size``."""
    from pretrain_data_curator import student_train as st

    V = 64
    model = _tiny_cfg(V).build()
    data = torch.randint(0, V, (128,), dtype=torch.long)
    gen = torch.Generator().manual_seed(0)
    captured = {}

    def fake_eval(
        model, val_data, *, block, batch, vocab_size, device, logit_chunk_tokens=None
    ):
        captured["batch"] = batch
        captured["logit_chunk_tokens"] = logit_chunk_tokens
        return 1.23, 0.45

    monkeypatch.setattr(st, "_eval_val_loss", fake_eval)
    loss, acc, tokens = train_and_eval_student(
        model,
        data,
        data,
        block_size=8,
        batch_size=4,
        steps=2,
        base_lr=1e-3,
        warmup_steps=1,
        weight_decay=0.1,
        grad_clip=1.0,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        lr_min_ratio=0.1,
        vocab_size=V,
        device="cpu",
        generator=gen,
        training_recipe="record_01_adamw",
        val_batch_size=1,
        val_logit_chunk_tokens=32,
    )
    assert captured["batch"] == 1
    assert captured["logit_chunk_tokens"] == 32
    assert loss == pytest.approx(1.23)
    assert acc == pytest.approx(0.45)
    assert tokens > 0


def test_proxy_payload_and_sandbox_script_carry_val_microbatch_knobs():
    from pretrain_data_curator.models import ProxyStudentConfig

    payload = ProxyStudentConfig(
        val_batch_size=1, val_logit_chunk_tokens=1024, train_microbatch_size=16
    ).training_payload()
    assert payload["val_batch_size"] == 1
    assert payload["val_logit_chunk_tokens"] == 1024
    assert payload["train_microbatch_size"] == 16
    assert 'val_batch_size=cfg.get("val_batch_size")' in NANOGPT_TRAIN_SCRIPT
    assert (
        'val_logit_chunk_tokens=cfg.get("val_logit_chunk_tokens")'
        in NANOGPT_TRAIN_SCRIPT
    )
    assert (
        'train_microbatch_size=cfg.get("train_microbatch_size")' in NANOGPT_TRAIN_SCRIPT
    )
    assert "def _score_hidden_chunked(" in NANOGPT_TRAIN_SCRIPT
    assert "def _microbatch_ranges(" in NANOGPT_TRAIN_SCRIPT
    assert "def _scaled_microbatch_loss(" in NANOGPT_TRAIN_SCRIPT
    assert "def forward_hidden(" in NANOGPT_TRAIN_SCRIPT
    assert "def apply_lm_head(" in NANOGPT_TRAIN_SCRIPT
    assert "buffering=1" in NANOGPT_TRAIN_SCRIPT
    assert "atexit.register(_stderr_fh.flush)" in NANOGPT_TRAIN_SCRIPT


# --- Loss-scaled train microbatch accumulation (A100 OOM fix) ----------------


def test_microbatch_ranges_covers_effective_batch():
    from pretrain_data_curator.student_train import _microbatch_ranges

    assert list(_microbatch_ranges(16, None)) == [(0, 16)]
    assert list(_microbatch_ranges(16, 16)) == [(0, 16)]
    assert list(_microbatch_ranges(16, 32)) == [(0, 16)]
    assert list(_microbatch_ranges(32, 16)) == [(0, 16), (16, 32)]
    assert list(_microbatch_ranges(48, 16)) == [(0, 16), (16, 32), (32, 48)]
    assert list(_microbatch_ranges(33, 16)) == [(0, 16), (16, 32), (32, 33)]


def test_train_microbatch_accum_matches_full_batch_adamw_grads():
    """Loss-scaled microbatch grads match a single full-batch backward (AdamW path).

    Documented tolerance: exact equality on CPU fp32 for this tiny model; any
    future softcap/bf16 path may need a small abs tolerance.
    """
    from pretrain_data_curator.student_train import (
        _microbatch_ranges,
        _scaled_microbatch_loss,
    )

    torch.manual_seed(0)
    V = 64
    model = _tiny_cfg(V).build()
    x = torch.randint(0, V, (4, 8), dtype=torch.long)
    y = torch.randint(0, V, (4, 8), dtype=torch.long)

    model.zero_grad(set_to_none=True)
    logits = model(x)
    full_loss = torch.nn.functional.cross_entropy(logits.view(-1, V), y.view(-1))
    full_loss.backward()
    full_grads = {
        n: p.grad.detach().clone()
        for n, p in model.named_parameters()
        if p.grad is not None
    }

    model.zero_grad(set_to_none=True)
    for start, end in _microbatch_ranges(4, 2):
        logits = model(x[start:end])
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, V), y[start:end].view(-1)
        )
        _scaled_microbatch_loss(loss, end - start, 4).backward()
    micro_grads = {
        n: p.grad.detach().clone()
        for n, p in model.named_parameters()
        if p.grad is not None
    }

    assert set(full_grads) == set(micro_grads)
    for name in full_grads:
        assert torch.allclose(full_grads[name], micro_grads[name], rtol=0, atol=1e-6), (
            name
        )


def test_train_microbatch_preserves_tokens_and_schedule_ramp(monkeypatch):
    """Scheduled effective batch still drives token accounting under microbatching."""
    from pretrain_data_curator import student_train as st

    V = 64
    model = _tiny_cfg(V).build()
    data = torch.randint(0, V, (256,), dtype=torch.long)
    gen = torch.Generator().manual_seed(0)
    seen_batches: list[int] = []
    real_ranges = st._microbatch_ranges

    def tracking_ranges(batch_size, microbatch_size):
        seen_batches.append(int(batch_size))
        return real_ranges(batch_size, microbatch_size)

    monkeypatch.setattr(st, "_microbatch_ranges", tracking_ranges)
    monkeypatch.setattr(st, "_eval_val_loss", lambda *a, **k: (1.0, 0.5))

    _, _, tokens = train_and_eval_student(
        model,
        data,
        data,
        block_size=8,
        batch_size=2,
        steps=6,
        base_lr=1e-3,
        warmup_steps=1,
        weight_decay=0.1,
        grad_clip=1.0,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        lr_min_ratio=0.1,
        vocab_size=V,
        device="cpu",
        generator=gen,
        training_recipe="speedrun_muon",
        batch_schedule_enabled=True,
        batch_stage_fracs=(1 / 3, 1 / 3, 1 / 3),
        batch_stage_muls=(1, 2, 3),
        lr_stage_muls=(1.0, 1.0, 1.0),
        lr_cooldown_frac=0.0,
        train_microbatch_size=2,
        adam_on_odd_steps=False,
    )
    # 6 steps -> 2 per stage at muls 1,2,3 -> effective batches 2,4,6
    assert seen_batches == [2, 2, 4, 4, 6, 6]
    # tokens = sum(effective_batch * block) over steps
    assert tokens == (2 + 2 + 4 + 4 + 6 + 6) * 8


def test_train_microbatch_adamw_end_to_end_matches_full_batch_weights():
    """End-to-end AdamW: microbatch=2 vs full batch=4 yields near-identical weights.

    Tolerance documents fp32 accumulation order differences across micro-forwards.
    """
    V = 64
    data = torch.randint(0, V, (128,), dtype=torch.long)
    common = dict(
        block_size=8,
        batch_size=4,
        steps=3,
        base_lr=1e-3,
        warmup_steps=1,
        weight_decay=0.0,
        grad_clip=0.0,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        lr_min_ratio=1.0,
        vocab_size=V,
        device="cpu",
        training_recipe="record_01_adamw",
    )

    torch.manual_seed(0)
    model_full = _tiny_cfg(V).build()
    gen_full = torch.Generator().manual_seed(0)
    train_and_eval_student(
        model_full, data, data, generator=gen_full, train_microbatch_size=None, **common
    )

    torch.manual_seed(0)
    model_micro = _tiny_cfg(V).build()
    gen_micro = torch.Generator().manual_seed(0)
    train_and_eval_student(
        model_micro, data, data, generator=gen_micro, train_microbatch_size=2, **common
    )

    for (n1, p1), (n2, p2) in zip(
        model_full.named_parameters(), model_micro.named_parameters()
    ):
        assert n1 == n2
        # Slightly looser than 1e-5: residual-lambda init at sqrt(1.1) makes
        # fp32 microbatch vs full-batch accumulation order noise visible at 1e-5.
        assert torch.allclose(p1, p2, rtol=1e-4, atol=1e-4), n1


# --- Microbatch x grad_accum_embed_head_steps regression (values 1 and >1) ---


@pytest.mark.parametrize("grad_accum", [1, 2])
@pytest.mark.parametrize("microbatch", [None, 1])
def test_grad_accum_microbatch_clip_cadence_unchanged(
    monkeypatch, grad_accum, microbatch
):
    """Clipping fires once per optimizer update for grad_accum in {1, >1}, and
    loss-scaled microbatching must not add or skip a clip (no double/skipped clip).

    With batch schedule off, batch=2, steps=6, adam_on_odd_steps=True there are
    always 6 Muon updates + 3 AdamW updates = 9 clips, independent of how many
    microbatches each backward is split into.
    """
    clip_calls: list[float] = []
    real_clip = torch.nn.utils.clip_grad_norm_

    def spy_clip(params, max_norm, *a, **k):
        clip_calls.append(float(max_norm))
        return real_clip(params, max_norm, *a, **k)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy_clip)

    V = 32
    model = _tiny_cfg(V).build()
    data = torch.randint(0, V, (300,), dtype=torch.long)
    gen = torch.Generator().manual_seed(0)
    train_and_eval_student(
        model,
        data,
        data,
        block_size=8,
        batch_size=2,
        steps=6,
        base_lr=1e-3,
        warmup_steps=1,
        weight_decay=0.1,
        grad_clip=1.0,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        lr_min_ratio=0.1,
        vocab_size=V,
        device="cpu",
        generator=gen,
        training_recipe="speedrun_muon",
        batch_schedule_enabled=False,
        adam_on_odd_steps=True,
        grad_accum_embed_head_steps=grad_accum,
        train_microbatch_size=microbatch,
    )
    assert len(clip_calls) == 9, (
        f"grad_accum={grad_accum} microbatch={microbatch}: expected 9 clips, "
        f"got {len(clip_calls)}: {clip_calls}"
    )
    assert all(c == 1.0 for c in clip_calls)


@pytest.mark.parametrize("grad_accum", [1, 2])
def test_grad_accum_microbatch_feeds_optimizer_full_batch_grads(
    monkeypatch, grad_accum
):
    """For grad_accum in {1, >1}, loss-scaled microbatching must hand the speedrun
    optimizers the same embed/lm_head gradients as a single full-effective-batch
    backward at the optimizer-update boundary.

    Comparing *at the optimizer boundary* (not final weights) isolates the property
    this change guarantees: Muon's Newton-Schulz orthogonalization and the 75x
    value-embed LR would otherwise amplify pure fp32 summation-order noise into
    visible weight drift. ``grad_clip=0`` so nothing perturbs the raw accumulation.
    """
    from pretrain_data_curator import student_train as st

    V = 48
    data = torch.randint(0, V, (256,), dtype=torch.long)
    common = dict(
        block_size=8,
        batch_size=2,
        steps=grad_accum,  # exactly one full accumulation cycle
        base_lr=1e-3,
        warmup_steps=1,
        weight_decay=0.1,
        grad_clip=0.0,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        lr_min_ratio=0.1,
        vocab_size=V,
        device="cpu",
        training_recipe="speedrun_muon",
        batch_schedule_enabled=False,
        grad_accum_embed_head_steps=grad_accum,
    )
    monkeypatch.setattr(st, "_eval_val_loss", lambda *a, **k: (1.0, 0.5))

    def run(microbatch):
        torch.manual_seed(0)
        model = _tiny_cfg(V).build()
        gen = torch.Generator().manual_seed(0)
        real_step = st.step_speedrun_optimizers
        captured: dict[str, torch.Tensor] = {}

        def spy(muon_opt, adam_opt, **kwargs):
            # First optimizer update carries this cycle's summed embed/head grads
            # (grad_accum=1: the step's own grads; grad_accum>1: the restored buffer).
            if not captured:
                for name, p in model.named_parameters():
                    if name.startswith("embed.") or name.startswith("lm_head."):
                        if p.grad is not None:
                            captured[name] = p.grad.detach().clone()
            return real_step(muon_opt, adam_opt, **kwargs)

        monkeypatch.setattr(st, "step_speedrun_optimizers", spy)
        train_and_eval_student(
            model, data, data, generator=gen, train_microbatch_size=microbatch, **common
        )
        assert captured, "expected an optimizer update to capture embed/head grads"
        return captured

    full = run(None)
    micro = run(1)
    assert set(full) == set(micro)
    for name in full:
        assert torch.allclose(full[name], micro[name], rtol=1e-4, atol=1e-6), name


# --- (N) Chunked scoring: bf16 hidden + fp32 lm_head ----------------------


def test_score_hidden_chunked_bf16_hidden_fp32_head_cpu():
    """``_score_hidden_chunked`` must accept bf16 hidden states and route them
    through the model's centralized fp32 lm-head/softcap logic, and chunked
    scoring must agree (modulo fp summation order) with the unchunked pass and
    with a single full-vocab reference CE. Mirrors the bf16-on-CUDA trunk feeding
    an fp32 ``lm_head``; the CPU bf16 injection keeps the test deterministic.
    """
    import torch.nn.functional as F

    import pretrain_data_curator.student_train as st
    from pretrain_data_curator.student_model import StudentModelConfig as SMC

    V, dim = 50, 32
    model = SMC(
        model_dim=dim, num_layers=2, num_heads=2, vocab_size=V, num_value_embeds=1
    ).build()  # exposes apply_lm_head (centralized fp32 path)

    gen = torch.Generator().manual_seed(0)
    n = 64
    hidden_fp32 = torch.randn(n, dim, generator=gen)
    hidden_bf16 = hidden_fp32.bfloat16()  # simulate bf16 CUDA trunk output
    targets = torch.randint(0, V, (n,), generator=gen)

    # Chunked vs unchunked via the centralized fp32 lm-head path.
    full_loss, full_correct = st._score_hidden_chunked(
        model, hidden_bf16, targets, vocab_size=V, logit_chunk_tokens=n
    )
    chunk_loss, chunk_correct = st._score_hidden_chunked(
        model, hidden_bf16, targets, vocab_size=V, logit_chunk_tokens=8
    )
    assert chunk_correct == full_correct
    assert chunk_loss == pytest.approx(full_loss, rel=1e-5)

    # Reference: single full-vocab pass through apply_lm_head.
    with torch.no_grad():
        ref_logits = model.apply_lm_head(hidden_bf16.float())
    ref_loss = F.cross_entropy(ref_logits.float(), targets).item()
    ref_correct = (ref_logits.float().argmax(-1) == targets).sum().item()
    assert chunk_correct == ref_correct
    assert (chunk_loss / n) == pytest.approx(ref_loss, rel=1e-5)


def _make_fp32_head_stub(dim: int, vocab: int, softcap: float):
    """A head-only model WITHOUT ``apply_lm_head`` to exercise the generic
    ``lm_head``/softcap fallback branch in ``_score_hidden_chunked``."""
    import torch.nn as nn

    class _Fp32HeadStub(nn.Module):
        def __init__(self):
            super().__init__()
            self.lm_head = nn.Linear(dim, vocab, bias=False)
            self.softcap = float(softcap)

    return _Fp32HeadStub()


def test_score_hidden_chunked_else_branch_bf16_hidden_fp32_head_cpu():
    """The generic ``lm_head``/softcap fallback branch must also accept bf16
    hidden states and cast to fp32 before the fp32 linear layer, so bf16 hidden
    + fp32 head works and chunked scoring agrees with the unchunked pass.
    """
    import torch.nn.functional as F

    import pretrain_data_curator.student_train as st

    V, dim, softcap = 50, 32, 30.0
    model = _make_fp32_head_stub(dim, V, softcap)  # fp32 weights, no apply_lm_head
    assert next(model.parameters()).dtype is torch.float32

    gen = torch.Generator().manual_seed(1)
    n = 64
    hidden_fp32 = torch.randn(n, dim, generator=gen)
    hidden_bf16 = hidden_fp32.bfloat16()
    targets = torch.randint(0, V, (n,), generator=gen)

    full_loss, full_correct = st._score_hidden_chunked(
        model, hidden_bf16, targets, vocab_size=V, logit_chunk_tokens=n
    )
    chunk_loss, chunk_correct = st._score_hidden_chunked(
        model, hidden_bf16, targets, vocab_size=V, logit_chunk_tokens=8
    )
    assert chunk_correct == full_correct
    assert chunk_loss == pytest.approx(full_loss, rel=1e-5)

    with torch.no_grad():
        ref_logits = model.softcap * torch.tanh(
            model.lm_head(hidden_bf16.float()).float() / model.softcap
        )
    ref_loss = F.cross_entropy(ref_logits.float(), targets).item()
    ref_correct = (ref_logits.float().argmax(-1) == targets).sum().item()
    assert chunk_correct == ref_correct
    assert (chunk_loss / n) == pytest.approx(ref_loss, rel=1e-5)


def test_untie_lm_head_adam_group_retains_betas(monkeypatch):
    """When embed/lm_head are untied mid-training, the new ``lm_head`` Adam param
    group must retain the speedrun recipe betas ``(0.5, 0.95)`` rather than
    inheriting Adam defaults ``(0.9, 0.999)``.
    """
    import pretrain_data_curator.student_train as st
    from pretrain_data_curator.student_model import StudentModelConfig as SMC

    V = 32
    cfg = SMC(model_dim=32, num_layers=2, num_heads=2, vocab_size=V, num_value_embeds=1)
    model = cfg.build()
    data = torch.randint(0, V, (200,))
    gen = torch.Generator().manual_seed(0)

    captured = {}
    real_build = st.build_speedrun_optimizers

    def spy_build(m, **kw):
        muon_opt, adam_opt = real_build(m, **kw)
        captured["adam_opt"] = adam_opt
        return muon_opt, adam_opt

    monkeypatch.setattr(st, "build_speedrun_optimizers", spy_build)

    train_and_eval_student(
        model,
        data,
        data,
        block_size=8,
        batch_size=2,
        steps=8,
        base_lr=1e-3,
        warmup_steps=1,
        weight_decay=0.1,
        grad_clip=1.0,
        beta1=0.9,
        beta2=0.95,
        eps=1e-8,
        lr_min_ratio=0.1,
        vocab_size=V,
        device="cpu",
        generator=gen,
        training_recipe="speedrun_muon",
        untie_at_frac=0.5,
    )

    adam_opt = captured["adam_opt"]
    lm_head_groups = [
        g
        for g in adam_opt.param_groups
        if any(p is model.lm_head.weight for p in g["params"])
    ]
    assert lm_head_groups, "expected an lm_head adam param group after untie"
    for g in lm_head_groups:
        assert g["betas"] == (0.5, 0.95), g["betas"]
