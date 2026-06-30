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

import pytest
import torch

from pretrain_data_curator.student_model import StudentModelConfig
from pretrain_data_curator.student_train import (
    _TRAINING_COMPONENTS,
    averaged_train_and_eval,
    lr_at_step,
    plan_train_windows,
    train_and_eval_student,
    training_source,
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
_LR_SCHEDULE_CONFIGS = [(5, 2), (2, 1), (3, 0), (1, 0), (64, 8), (100, 10)]


@pytest.mark.parametrize("total, warm", _LR_SCHEDULE_CONFIGS)
def test_lr_at_step_last_executed_step_hits_floor_exactly(total, warm):
    # The loop runs `for step in range(total)`, so the LAST executed step is total-1.
    # That step must land EXACTLY on the floor base*min_ratio (no off-by-one that
    # stops a step short). This is the 'no off-by-one at final step' acceptance
    # point: it FAILS against the pre-fix `max(1, total-warmup)` denominator, which
    # left short runs at e.g. 0.325*base (5,2) or never decaying at all (2,1)/(1,0).
    base, floor = 0.7, 0.1
    assert lr_at_step(total - 1, total, warm, base, floor) == pytest.approx(base * floor)


@pytest.mark.parametrize("total, warm", [(5, 2), (64, 8), (100, 10)])
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
        model, data, data, block_size=block, batch_size=batch, steps=len(starts),
        base_lr=1e-3, warmup_steps=2, weight_decay=0.1, grad_clip=1.0,
        beta1=0.9, beta2=0.95, eps=1e-8, lr_min_ratio=0.1, vocab_size=V,
        device="cpu", generator=gen,
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
        model, data, data, block_size=8, batch_size=4, steps=steps,
        base_lr=1e-3, warmup_steps=2, weight_decay=0.1, grad_clip=1.0,
        beta1=0.9, beta2=0.95, eps=1e-8, lr_min_ratio=0.1, vocab_size=V,
        device="cpu", generator=gen,
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
        model, data, data, block_size=8, batch_size=2, steps=3,
        base_lr=1e-3, warmup_steps=1, weight_decay=0.1, grad_clip=0.0,
        beta1=0.9, beta2=0.95, eps=1e-8, lr_min_ratio=0.1, vocab_size=V,
        device="cpu", generator=gen,
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
        cfg.build, data, data, n_runs=3, base_seed=0, device="cpu",
        block_size=8, batch_size=4, steps=1, base_lr=1e-3, warmup_steps=1,
        weight_decay=0.1, grad_clip=1.0, beta1=0.9, beta2=0.95, eps=1e-8,
        lr_min_ratio=0.1, vocab_size=V,
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
        cfg.build, data, data, n_runs=3, base_seed=0, device="cpu",
        block_size=8, batch_size=4, steps=1, base_lr=1e-3, warmup_steps=1,
        weight_decay=0.1, grad_clip=1.0, beta1=0.9, beta2=0.95, eps=1e-8,
        lr_min_ratio=0.1, vocab_size=V,
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
        cfg.build, data, data, n_runs=1, base_seed=0, device="cpu",
        block_size=8, batch_size=4, steps=1, base_lr=1e-3, warmup_steps=1,
        weight_decay=0.1, grad_clip=1.0, beta1=0.9, beta2=0.95, eps=1e-8,
        lr_min_ratio=0.1, vocab_size=V,
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
        block_size=16, batch_size=8, steps=80, base_lr=3e-3, warmup_steps=8,
        weight_decay=0.1, grad_clip=1.0, beta1=0.9, beta2=0.95, eps=1e-8,
        lr_min_ratio=0.1, vocab_size=vocab_size,
    )

    clean_loss, *_ = averaged_train_and_eval(
        cfg.build, _char_tensor(clean, stoi), val_data,
        n_runs=1, base_seed=0, device="cpu", **hparams,
    )
    dirty_loss, *_ = averaged_train_and_eval(
        cfg.build, _char_tensor(dirty, stoi), val_data,
        n_runs=1, base_seed=0, device="cpu", **hparams,
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
        model, data, data, block_size=block, batch_size=batch, steps=steps,
        base_lr=1e-3, warmup_steps=2, weight_decay=0.1, grad_clip=1.0,
        beta1=0.9, beta2=0.95, eps=1e-8, lr_min_ratio=0.1, vocab_size=V,
        device="cpu", generator=gen,
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
    assert collections.Counter(replacement_starts) != {s: epochs for s in starts}  # not uniform


# --- (6) single source of truth: verbatim recipe embedded in the script ------


def test_sandbox_script_embeds_training_recipe_verbatim():
    # The GPU-only script must run the SAME training recipe these CPU tests exercise:
    # the exact recipe source appears byte-identically in the assembled script.
    ast.parse(NANOGPT_TRAIN_SCRIPT)  # assembled script is valid Python
    assert training_source() in NANOGPT_TRAIN_SCRIPT
    for component in _TRAINING_COMPONENTS:
        assert inspect.getsource(component).rstrip() in NANOGPT_TRAIN_SCRIPT
    # The record_01 recipe is wired: AdamW(betas/eps/weight_decay), warmup+cosine LR,
    # grad-clip, contiguous batching, and multi-run averaging.
    assert "torch.optim.AdamW(" in NANOGPT_TRAIN_SCRIPT
    assert "betas=(beta1, beta2)" in NANOGPT_TRAIN_SCRIPT
    assert "weight_decay=weight_decay" in NANOGPT_TRAIN_SCRIPT
    assert "clip_grad_norm_" in NANOGPT_TRAIN_SCRIPT
    assert "lr_at_step(step" in NANOGPT_TRAIN_SCRIPT
    assert "plan_train_windows(" in NANOGPT_TRAIN_SCRIPT
    assert "averaged_train_and_eval(" in NANOGPT_TRAIN_SCRIPT
    # ...and the OLD constant-LR plain-AdamW + random-with-replacement sampler is gone.
    assert "torch.randint(len(src) - block - 1" not in NANOGPT_TRAIN_SCRIPT
    assert "opt = torch.optim.AdamW(model.parameters(), lr=float(cfg" not in NANOGPT_TRAIN_SCRIPT
