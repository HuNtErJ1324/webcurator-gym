"""Modern proxy-student TRAINING recipe (nanogpt-speedrun record_01), CPU-runnable.

This module is the **single source of truth** for the real proxy-student trainer's
optimizer schedule, contiguous-window batching, and multi-run averaging. The same
function definitions are embedded byte-identically into the GPU sandbox training
script (``trainer.py``) via :func:`training_source`, so the CPU unit tests in this
package exercise the *exact* training code the sandbox runs -- the same discipline
``student_model.py`` uses for the architecture and ``val_set.plan_val_windows``
uses for the held-out cross-entropy windowing.

The recipe faithfully mirrors ``leloy/nanogpt-speedrun`` record_01 (the AdamW
GPT-2 reproduction baseline that the speedrun starts from), while inheriting
``student_model.py``'s deliberate EXCLUSION of the GPU-only training optimizations
(Muon/NorMuon, FP8, distributed comms, FlexAttention, ``torch.compile``, bf16) so
everything here runs fp32 on CPU:

* **AdamW** with record_01 betas ``(0.9, 0.95)``, ``eps=1e-8`` and decoupled
  ``weight_decay`` (default 0.1).
* **LR schedule** (:func:`lr_at_step`) -- linear warmup to the peak LR, then a
  cosine decay to a small floor (``lr_min_ratio`` of the peak); applied per-step
  like record_01's training loop.
* **Gradient clipping** -- global-norm clip (default 1.0) before every
  ``opt.step()``.
* **Contiguous-window batching** (:func:`plan_train_windows`) -- non-overlapping
  ``tokens[i:i+block]`` windows drawn sequentially over the tokenized train stream
  (record_01 ``get_batch`` style), with the window order shuffled per epoch and a
  cursor advanced across steps -- NOT random offsets sampled with replacement.
  (More docs/source -> a longer real token stream -> more distinct windows.)
* **Multiple averaged runs** (:func:`averaged_train_and_eval`) -- train+eval
  ``n_runs`` times with distinct seeds and average the val loss/accuracy, so the
  signal is a sharper, lower-variance measure of curated-data quality.
"""

from __future__ import annotations

import inspect
import math
import sys
import time

import torch
from torch.nn import functional as F
from tqdm import tqdm

from .val_set import plan_val_windows


def lr_at_step(step, total_steps, warmup_steps, base_lr, min_ratio):
    """record_01 learning rate for ``step``: linear warmup, then cosine cooldown.

    Linear warmup over ``warmup_steps`` (``base_lr * (step+1) / warmup_steps``),
    then a cosine decay from ``base_lr`` down to a floor of ``base_lr * min_ratio``.
    The training loop runs ``for step in range(total_steps)``, so the LAST executed
    step is ``total_steps - 1``; the cosine span is measured to that final step
    (``decay_span = total_steps - warmup_steps - 1``) so the last executed decay
    step lands EXACTLY on the floor instead of stopping a step short. When at most
    one decay step exists (``decay_span <= 0`` -- e.g. a one-step cooldown, or a
    warmup-only/zero-decay run) the span collapses and that single final decay step
    is pinned to the floor (``progress == 1.0``), never base_lr, and the division is
    skipped so there is no ZeroDivisionError. ``warmup_steps == 0`` means no warmup
    (decay starts at step 0). Kept pure (builtins + ``math`` only) so its exact
    source embeds verbatim into the sandbox training script and the CPU unit tests
    guard the schedule the GPU loop applies.
    """
    step = int(step)
    total_steps = int(total_steps)
    warmup_steps = int(warmup_steps)
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    decay_span = total_steps - warmup_steps - 1
    if decay_span <= 0:
        progress = 1.0
    else:
        progress = min(1.0, max(0.0, (step - warmup_steps) / decay_span))
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_ratio + (1.0 - min_ratio) * coeff)


def plan_train_windows(n_tokens, block):
    """Contiguous, non-overlapping training-window start indices over the stream.

    Returns the start indices of consecutive ``tokens[i:i+block]`` windows (whose
    paired next-token target is ``tokens[i+1:i+block+1]``), so batching draws
    CONTIGUOUS windows sequentially -- record_01 ``get_batch`` style -- rather than
    sampling random offsets with replacement. Every window satisfies
    ``start + block + 1 <= n_tokens``, so the shifted target never runs off the end.
    Builtins-only and dependency-free so its exact source embeds verbatim into the
    sandbox training script and is unit-tested on CPU.
    """
    block = int(block)
    n_tokens = int(n_tokens)
    last_start = n_tokens - block - 1
    if last_start < 0:
        return [0]
    return list(range(0, last_start + 1, block))


def train_and_eval_student(
    model,
    train_data,
    val_data,
    *,
    block_size,
    batch_size,
    steps,
    base_lr,
    warmup_steps,
    weight_decay,
    grad_clip,
    beta1,
    beta2,
    eps,
    lr_min_ratio,
    vocab_size,
    device,
    generator,
    run_label="",
):
    """Train ``model`` on ``train_data`` (record_01 recipe) and score held-out CE.

    Optimizes with AdamW(betas=(beta1, beta2), eps, weight_decay) under the
    :func:`lr_at_step` warmup+cosine schedule and a global-norm gradient clip
    (``grad_clip``), feeding CONTIGUOUS windows from :func:`plan_train_windows`
    whose order is reshuffled each epoch via ``generator``. Returns
    ``(val_loss, accuracy, tokens_trained)`` where ``val_loss`` is the mean
    cross-entropy (nats/token) over EVERY predictable next-token position of
    ``val_data`` (via :func:`plan_val_windows`, which raises on a degenerate
    <=1-token val set rather than scoring a bogus 0.0).

    Purely observational: a live ``tqdm`` bar (step/total, EMA loss, tok/s,
    elapsed, ETA) tracks the loop -- prefixed with ``run_label`` (e.g.
    ``"run 2/3 "``) when :func:`averaged_train_and_eval` runs >1 seeded run --
    plus coarse newline-terminated progress lines on stdout every ``log_every``
    steps (and always the first/last step) so long unattended runs stay legible
    from a captured, non-interactive stdout blob. The extra per-step ``.item()``
    GPU sync this needs is throttled to that same cadence, never every step, so
    it never touches the training math or the returned values.
    """
    block = int(block_size)
    batch = int(batch_size)
    steps = int(steps)

    train_src = train_data
    if len(train_src) <= block + 1:
        # Tiny-corpus fallback: tile so at least one full contiguous window exists
        # (the only case where windows wrap; a real token stream never tiles).
        train_src = train_src.repeat(math.ceil((block + 2) / max(len(train_src), 1)))
    starts = plan_train_windows(len(train_src), block)

    order = []
    cursor = 0

    def next_batch():
        nonlocal order, cursor
        xs = []
        ys = []
        for _ in range(batch):
            if cursor >= len(order):
                # Epoch boundary: reshuffle the CONTIGUOUS-window order (seeded by
                # ``generator`` so runs are reproducible and distinct seeds differ).
                perm = torch.randperm(len(starts), generator=generator)
                order = [starts[i] for i in perm.tolist()]
                cursor = 0
            i = order[cursor]
            cursor += 1
            xs.append(train_src[i : i + block])
            ys.append(train_src[i + 1 : i + block + 1])
        return torch.stack(xs).to(device), torch.stack(ys).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr,
        betas=(beta1, beta2),
        eps=eps,
        weight_decay=weight_decay,
    )
    model.train()
    # Coarse plain-text cadence: every ~2% of steps or every 50 steps, whichever
    # is COARSER (the larger gap), so a huge run logs a bounded number of lines
    # and a tiny test-sized run doesn't spam one per step; the first and last
    # step always log regardless, so even steps<=1 gets at least one line.
    log_every = max(50, max(1, steps // 50))
    ema_loss_tensor = None
    ema_decay = 0.9
    start_time = time.time()
    desc = f"{run_label}train" if run_label else "train"
    # ``file=sys.stdout``: in the sandbox script (trainer.py) ``sys.stderr`` is
    # redirected to a file, so a bar left on tqdm's stderr default would never
    # surface; stdout is what's actually captured, so that's where the live bar
    # (and the plain lines below, via ``pbar.write``) both go.
    pbar = tqdm(range(steps), total=steps, desc=desc, unit="step", leave=False, file=sys.stdout)
    for step in pbar:
        lr = lr_at_step(step, steps, warmup_steps, base_lr, lr_min_ratio)
        for group in opt.param_groups:
            group["lr"] = lr
        x, y = next_batch()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        # Real per-step EMA, kept on-device as a tensor so this costs no new GPU
        # sync (`.item()`) -- only converted to a Python float below, and only at
        # the throttled logging cadence.
        with torch.no_grad():
            loss_detached = loss.detach()
            ema_loss_tensor = loss_detached if ema_loss_tensor is None else (
                ema_decay * ema_loss_tensor + (1.0 - ema_decay) * loss_detached
            )

        completed = step + 1
        if step == 0 or completed == steps or completed % log_every == 0:
            # The only GPU sync (`.item()`) added for display; throttled to this
            # cadence, never on every step.
            ema_loss = ema_loss_tensor.item()
            elapsed = time.time() - start_time
            tokens_per_sec = (completed * batch * block) / elapsed if elapsed > 0 else 0.0
            eta_seconds = (elapsed / completed) * (steps - completed) if completed > 0 else 0.0
            pbar.set_postfix(loss=f"{ema_loss:.4f}", tok_s=f"{tokens_per_sec:.0f}")
            pbar.write(
                f"[{desc}] step {completed}/{steps} | loss {ema_loss:.4f} "
                f"| {tokens_per_sec:.0f} tok/s | elapsed {elapsed:.1f}s | eta {eta_seconds:.1f}s"
            )
    pbar.close()

    # Cross-entropy (nats/token, mean) over EVERY predictable next-token position of
    # the held-out val stream, batching consecutive equal-length windows so variable
    # lengths (full blocks, then the final partial window) never get stacked.
    model.eval()
    val_windows = plan_val_windows(len(val_data), block)
    total = sum(L for _, L in val_windows)
    loss_sum = 0.0
    correct = 0
    with torch.no_grad():
        wi = 0
        while wi < len(val_windows):
            L = val_windows[wi][1]
            vs = []
            while wi < len(val_windows) and val_windows[wi][1] == L and len(vs) < batch:
                vs.append(val_windows[wi][0])
                wi += 1
            xb = torch.stack([val_data[s : s + L] for s in vs]).to(device)
            yb = torch.stack([val_data[s + 1 : s + L + 1] for s in vs]).to(device)
            logits = model(xb)
            loss_sum += F.cross_entropy(
                logits.reshape(-1, vocab_size), yb.reshape(-1), reduction="sum"
            ).item()
            correct += (logits.argmax(-1) == yb).sum().item()
    val_loss = loss_sum / total
    acc = correct / total
    tokens_trained = steps * batch * block
    return val_loss, acc, tokens_trained


def averaged_train_and_eval(
    build_model, train_data, val_data, *, n_runs, base_seed, device, **hparams
):
    """Train+eval ``n_runs`` times with distinct seeds and average the val signal.

    Each run reseeds the global RNG and the window-shuffle generator to
    ``base_seed + run`` and trains a FRESH model from ``build_model()``. If ANY run
    yields a non-finite loss the whole result collapses to the infinite-loss
    sentinel ``(inf, 0.0, 0.0, 0, 0)`` -- strict completion, exactly like the
    empty-corpus path -- so a single bad run can never average into a deceptively
    finite score. Loss and accuracy are AVERAGED; FLOPs and tokens are SUMMED across
    runs (N runs really spend N x the compute, and cost accounting bills all of it).
    Returns ``(val_loss, accuracy, flops, tokens_trained, n_params)``.
    """
    n_runs = max(1, int(n_runs))
    losses = []
    accs = []
    total_flops = 0.0
    total_tokens = 0
    n_params = 0
    for run in range(n_runs):
        torch.manual_seed(base_seed + run)
        generator = torch.Generator().manual_seed(base_seed + run)
        model = build_model()
        run_label = f"run {run + 1}/{n_runs} " if n_runs > 1 else ""
        loss, acc, tokens_trained = train_and_eval_student(
            model, train_data, val_data, device=device, generator=generator,
            run_label=run_label, **hparams
        )
        if not math.isfinite(loss):
            return float("inf"), 0.0, 0.0, 0, 0
        n_params = sum(p.numel() for p in model.parameters())
        losses.append(loss)
        accs.append(acc)
        total_flops += 6.0 * n_params * tokens_trained
        total_tokens += tokens_trained
    return sum(losses) / len(losses), sum(accs) / len(accs), total_flops, total_tokens, n_params


# Source-of-truth training-recipe helpers, in dependency (definition) order. Their
# exact source is embedded into the sandbox training script in ``trainer.py``.
_TRAINING_COMPONENTS = (
    lr_at_step,
    plan_train_windows,
    train_and_eval_student,
    averaged_train_and_eval,
)


def training_source() -> str:
    """The verbatim source of the training-recipe helpers, for byte-identical embed.

    Returns the concatenated source of every recipe function in this module (in
    definition order). ``trainer.py`` injects this exact string into its sandbox
    training script -- after the embedded model and ``plan_val_windows`` -- so the
    GPU-only training run executes the same optimizer schedule, contiguous
    batching, and multi-run averaging these CPU tests exercise. (``plan_val_windows``
    is embedded separately and only *called* here, never redefined.)
    """
    return "\n\n\n".join(inspect.getsource(c).rstrip() for c in _TRAINING_COMPONENTS)
