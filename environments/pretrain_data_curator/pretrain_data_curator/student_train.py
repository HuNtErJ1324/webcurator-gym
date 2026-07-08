"""Modern proxy-student TRAINING recipe (modded-nanogpt speedrun), CPU-runnable.

This module is the **single source of truth** for the real proxy-student trainer's
optimizer schedule, contiguous-window batching, and multi-run averaging. The same
function definitions are embedded byte-identically into the GPU sandbox training
script (``trainer.py``) via :func:`training_source`, so the CPU unit tests in this
package exercise the *exact* training code the sandbox runs.

Default recipe (**``speedrun_muon``**) ports the CPU-runnable subset of
``KellerJordan/modded-nanogpt`` ``train_gpt.py``:

* **Muon** (Newton–Schulz) on 2-D block weights; **AdamW** on embeddings/scalars.
* **Heterogeneous stepping** — AdamW only on odd steps.
* **Batch-size schedule** — 1× → 2× → 3× micro-batch over three equal stages.
* **LR schedule** — stage LR multipliers + linear cooldown (BatchSizeSchedule record).
* **Muon momentum warmup/cooldown**.

Legacy recipe (**``record_01_adamw``**) remains for fast CPU tests:

* Plain **AdamW** + linear warmup + cosine cooldown + grad clip.

Portable features (all opt-in):

* **EoS-aligned batch starts** — align training windows to end-of-sequence markers.
* **Max document length handling** — split over-long documents.
* **True 2-step gradient accumulation** — accumulate embed+lm_head grads before update.
* **True max seq length schedule** — warm up sequence length from small to max.
* **Multi-token prediction** — auxiliary future-token prediction losses.
* **Untie embed/lm_head** — share weights initially, untie at 2/3 of training.
"""

from __future__ import annotations

import inspect
import math
import sys
import time

import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

from .student_optimizer import (
    build_batch_schedule,
    build_speedrun_optimizers,
    capture_initial_lrs,
    get_muon_momentum,
    init_speedrun_weights,
    lookup_batch_stage,
    schedule_lr_multiplier,
    set_optimizer_lrs,
    step_speedrun_optimizers,
)
from .val_set import plan_val_windows


def lr_at_step(step, total_steps, warmup_steps, base_lr, min_ratio):
    """record_01 learning rate for ``step``: linear warmup, then cosine cooldown."""
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
    """Contiguous, non-overlapping training-window start indices over the stream."""
    block = int(block)
    n_tokens = int(n_tokens)
    last_start = n_tokens - block - 1
    if last_start < 0:
        return [0]
    return list(range(0, last_start + 1, block))


def plan_eos_aligned_windows(n_tokens, block, eos_positions):
    """Training-window start indices aligned to end-of-sequence markers.

    Windows start at EoS positions (or 0) and proceed contiguously, so each
    window naturally begins at a document boundary when possible.
    """
    block = int(block)
    n_tokens = int(n_tokens)
    if not eos_positions:
        return plan_train_windows(n_tokens, block)
    starts = [0]
    for eos in sorted(eos_positions):
        if eos + 1 < n_tokens - block:
            starts.append(eos + 1)
    deduped = sorted(set(s for s in starts if s + block + 1 <= n_tokens))
    if not deduped:
        return [0]
    return deduped


def make_seq_len_schedule(total_steps, max_block, warmup_frac=0.25):
    """Produce a sequence-length schedule that warms up from small to ``max_block``.

    Returns a callable ``block_at_step(step) -> int``.
    """
    max_block = int(max_block)
    warmup_steps = max(1, int(total_steps * warmup_frac))
    min_block = max(8, max_block // 8)

    def block_at_step(step):
        step = int(step)
        if step >= warmup_steps:
            return max_block
        frac = step / warmup_steps
        return int(min_block + frac * (max_block - min_block))

    return block_at_step


def _enforce_max_doc_len(starts, n_tokens, max_doc_len, block):
    """Add synthetic start positions to break up segments longer than ``max_doc_len``.

    When the gap between consecutive window starts exceeds ``max_doc_len``,
    intermediate starts are inserted so that every position is within
    ``max_doc_len`` tokens of a window boundary. Out-of-bounds starts are
    dropped.
    """
    if max_doc_len is None:
        return starts
    if not starts:
        return starts
    refined = [starts[0]]
    for s in starts[1:]:
        prev = refined[-1]
        gap = s - prev
        if gap > max_doc_len:
            for mid in range(prev + max_doc_len, s, max_doc_len):
                if mid + block + 1 <= n_tokens:
                    refined.append(mid)
        refined.append(s)
    return sorted(set(r for r in refined if r + block + 1 <= n_tokens))


def _eval_val_loss(model, val_data, *, block, batch, vocab_size, device):
    model.eval()
    val_windows = plan_val_windows(len(val_data), block)
    total = sum(length for _, length in val_windows)
    loss_sum = 0.0
    correct = 0
    with torch.no_grad():
        wi = 0
        while wi < len(val_windows):
            length = val_windows[wi][1]
            starts = []
            while wi < len(val_windows) and val_windows[wi][1] == length and len(starts) < batch:
                starts.append(val_windows[wi][0])
                wi += 1
            xb = torch.stack([val_data[s : s + length] for s in starts]).to(device)
            yb = torch.stack([val_data[s + 1 : s + length + 1] for s in starts]).to(device)
            logits = model(xb)
            loss_sum += F.cross_entropy(
                logits.reshape(-1, vocab_size), yb.reshape(-1), reduction="sum"
            ).item()
            correct += (logits.argmax(-1) == yb).sum().item()
    return loss_sum / total, correct / total


def _compute_multi_token_loss(logits, hidden, y, multi_heads, y_future, vocab_size):
    """Total loss including auxiliary multi-token prediction losses.

    Multi-token heads are applied to the final hidden states (pre-lm_head),
    not to logits, since the heads are ``nn.Linear(model_dim, vocab_size)``.
    """
    main_loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    if multi_heads is None or not hasattr(multi_heads, 'heads') or not multi_heads.heads:
        return main_loss
    mt_loss = 0.0
    mt_weight = 0.3
    for k, head in enumerate(multi_heads.heads):
        if k < len(y_future) and y_future[k] is not None:
            head_logits = head(hidden)
            mt_loss = mt_loss + F.cross_entropy(
                head_logits.view(-1, vocab_size), y_future[k].view(-1)
            )
    if mt_loss > 0.0:
        return main_loss + mt_weight * mt_loss
    return main_loss


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
    training_recipe="speedrun_muon",
    muon_lr=0.023,
    muon_weight_decay=0.05,
    adam_lr=0.008,
    adam_eps=1e-10,
    adam_weight_decay=0.005,
    embed_lr_mul=1.0,
    lm_head_lr_mul=1.0,
    value_embed_lr_mul=75.0,
    scalar_lr_mul=5.0,
    embed_wd_mul=150.0,
    lm_head_wd_mul=150.0,
    value_embed_wd_mul=5.0,
    scalar_wd_mul=0.0,
    batch_schedule_enabled=True,
    batch_stage_fracs=(1 / 3, 1 / 3, 1 / 3),
    batch_stage_muls=(1, 2, 3),
    lr_stage_muls=(1.0, 1.52, 1.73),
    lr_cooldown_frac=0.60,
    lr_cooldown_floor=0.15,
    muon_momentum_min=0.85,
    muon_momentum_max=0.95,
    muon_warmup_steps=None,
    muon_cooldown_steps=None,
    adam_on_odd_steps=True,
    # --- portable feature flags ---
    eos_positions=None,
    max_doc_len=None,
    grad_accum_embed_head_steps=1,
    seq_len_schedule=False,
    multi_token_pred=0,
    untie_at_frac=0.0,
    cautious_wd=False,
    nor_muon=False,
    polar_express=False,
):
    """Train ``model`` and score held-out CE; returns ``(val_loss, accuracy, tokens_trained)``."""
    block = int(block_size)
    batch = int(batch_size)
    steps = int(steps)

    train_src = train_data
    if len(train_src) <= block + 1:
        train_src = train_src.repeat(math.ceil((block + 2) / max(len(train_src), 1)))

    # --- Portable: EoS-aligned windows ---
    if eos_positions is not None:
        starts = plan_eos_aligned_windows(len(train_src), block, eos_positions)
    else:
        starts = plan_train_windows(len(train_src), block)

    # --- Portable: max doc length enforcement ---
    if max_doc_len is not None:
        starts = _enforce_max_doc_len(starts, len(train_src), max_doc_len, block)

    # --- Portable: seq length schedule ---
    block_fn = make_seq_len_schedule(steps, block) if seq_len_schedule else None

    order = []
    cursor = 0

    def next_batch(effective_batch, current_block):
        nonlocal order, cursor
        xs = []
        ys = []
        batch_starts = []
        for _ in range(int(effective_batch)):
            if cursor >= len(order):
                perm = torch.randperm(len(starts), generator=generator)
                order = [starts[i] for i in perm.tolist()]
                cursor = 0
            i = order[cursor]
            cursor += 1
            batch_starts.append(i)
            xs.append(train_src[i : i + current_block])
            ys.append(train_src[i + 1 : i + current_block + 1])
        return torch.stack(xs).to(device), torch.stack(ys).to(device), batch_starts

    log_every = max(50, max(1, steps // 50))
    ema_loss_tensor = None
    ema_decay = 0.9
    start_time = time.time()
    desc = f"{run_label}train" if run_label else "train"
    pbar = tqdm(range(steps), total=steps, desc=desc, unit="step", leave=False, file=sys.stdout)

    if training_recipe == "record_01_adamw":
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=base_lr,
            betas=(beta1, beta2),
            eps=eps,
            weight_decay=weight_decay,
        )
        model.train()
        tokens_trained = 0
        for step in pbar:
            lr = lr_at_step(step, steps, warmup_steps, base_lr, lr_min_ratio)
            for group in opt.param_groups:
                group["lr"] = lr
            current_block = block_fn(step) if block_fn else block
            x, y, _ = next_batch(batch, current_block)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            tokens_trained += batch * current_block
            with torch.no_grad():
                loss_detached = loss.detach()
                ema_loss_tensor = loss_detached if ema_loss_tensor is None else (
                    ema_decay * ema_loss_tensor + (1.0 - ema_decay) * loss_detached
                )
            completed = step + 1
            if step == 0 or completed == steps or completed % log_every == 0:
                ema_loss = ema_loss_tensor.item()
                elapsed = time.time() - start_time
                tokens_per_sec = tokens_trained / elapsed if elapsed > 0 else 0.0
                eta_seconds = (elapsed / completed) * (steps - completed) if completed > 0 else 0.0
                pbar.set_postfix(loss=f"{ema_loss:.4f}", tok_s=f"{tokens_per_sec:.0f}")
                pbar.write(
                    f"[{desc}] step {completed}/{steps} | loss {ema_loss:.4f} "
                    f"| {tokens_per_sec:.0f} tok/s | elapsed {elapsed:.1f}s | eta {eta_seconds:.1f}s"
                )
    else:
        init_speedrun_weights(model)

        # --- Portable: tie embed/lm_head for later untie ---
        # BigramHashEmbedding does not expose a flat .weight (its learned
        # token_embed has a different shape), so weight-tying is shape-
        # incompatible; skip the tie in that case.
        if untie_at_frac > 0.0 and hasattr(model.embed, "weight"):
            model.lm_head.weight = model.embed.weight

        muon_opt, adam_opt = build_speedrun_optimizers(
            model,
            muon_lr=muon_lr,
            muon_weight_decay=muon_weight_decay,
            adam_lr=adam_lr,
            adam_eps=adam_eps,
            adam_weight_decay=adam_weight_decay,
            embed_lr_mul=embed_lr_mul,
            lm_head_lr_mul=lm_head_lr_mul,
            value_embed_lr_mul=value_embed_lr_mul,
            scalar_lr_mul=scalar_lr_mul,
            embed_wd_mul=embed_wd_mul,
            lm_head_wd_mul=lm_head_wd_mul,
            value_embed_wd_mul=value_embed_wd_mul,
            scalar_wd_mul=scalar_wd_mul,
            nor_muon=nor_muon,
            polar_express=polar_express,
        )
        initial_muon_lr, initial_adam_lrs = capture_initial_lrs(muon_opt, adam_opt)
        if batch_schedule_enabled:
            boundaries, stages, cd_start, cd_floor = build_batch_schedule(
                steps,
                stage_fracs=batch_stage_fracs,
                batch_muls=batch_stage_muls,
                lr_muls=lr_stage_muls,
                cooldown_frac=lr_cooldown_frac,
                cooldown_floor=lr_cooldown_floor,
            )
        else:
            from .student_optimizer import BatchScheduleStage

            boundaries = [(0, steps)]
            stages = [BatchScheduleStage(batch_mul=1, lr_mul=1.0)]
            cd_start, cd_floor = steps, lr_cooldown_floor
        muon_warm = muon_warmup_steps if muon_warmup_steps is not None else min(300, max(1, steps // 5))
        muon_cd = muon_cooldown_steps if muon_cooldown_steps is not None else min(50, max(1, steps // 20))
        model.train()
        tokens_trained = 0
        accum_buffer_embed_head = None
        accum_count = 0

        for step in pbar:
            if batch_schedule_enabled:
                stage = lookup_batch_stage(step, boundaries, stages)
                effective_batch = batch * stage.batch_mul
                lr_scale = schedule_lr_multiplier(
                    step,
                    stage,
                    cd_start=cd_start,
                    scheduled_steps=steps,
                    cooldown_floor=cd_floor,
                )
            else:
                effective_batch = batch
                lr_scale = 1.0
            set_optimizer_lrs(
                muon_opt,
                adam_opt,
                lr_scale=lr_scale,
                initial_muon_lr=initial_muon_lr,
                initial_adam_lrs=initial_adam_lrs,
            )
            muon_momentum = get_muon_momentum(
                step,
                steps,
                warmup_steps=muon_warm,
                cooldown_steps=muon_cd,
                momentum_min=muon_momentum_min,
                momentum_max=muon_momentum_max,
            )

            current_block = block_fn(step) if block_fn else block
            x, y, batch_starts = next_batch(effective_batch, current_block)

            # --- Portable: multi-token prediction targets ---
            use_mt = multi_token_pred > 0 and hasattr(model, 'multi_heads') and model.multi_heads is not None
            y_future = []
            if use_mt:
                for k in range(multi_token_pred):
                    shift = k + 2
                    future_targets = []
                    for sidx in batch_starts:
                        if sidx + shift + current_block <= len(train_src):
                            future_targets.append(train_src[sidx + shift : sidx + shift + current_block])
                        else:
                            n_avail = max(0, len(train_src) - (sidx + shift))
                            if n_avail > 0:
                                seg = train_src[sidx + shift:]
                                seg = torch.cat([seg, torch.zeros(current_block - n_avail, dtype=train_src.dtype)])
                            else:
                                seg = torch.zeros(current_block, dtype=train_src.dtype)
                            future_targets.append(seg)
                    y_future.append(torch.stack(future_targets).to(device))

            logits = model(x, output_hidden=use_mt)

            # --- Portable: multi-token prediction loss ---
            if use_mt:
                hidden = logits[1]
                loss = _compute_multi_token_loss(logits[0], hidden, y, model.multi_heads, y_future, vocab_size)
            else:
                loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))

            # --- Portable: true N-step gradient accumulation for embed + lm_head ---
            if grad_accum_embed_head_steps > 1:
                embed_head_params = []
                for name, p in model.named_parameters():
                    if name.startswith("embed.") or name.startswith("lm_head.") or name.startswith("multi_heads."):
                        embed_head_params.append(p)

                loss.backward()

                # Accumulate embed/head/multi-head grads into a buffer, then
                # zero p.grad so the next backward starts fresh (no double-count).
                if accum_buffer_embed_head is None:
                    accum_buffer_embed_head = {}
                    for p in embed_head_params:
                        if p.grad is not None:
                            accum_buffer_embed_head[id(p)] = p.grad.clone()
                            p.grad = None
                else:
                    for p in embed_head_params:
                        if p.grad is not None and id(p) in accum_buffer_embed_head:
                            accum_buffer_embed_head[id(p)] += p.grad
                            p.grad = None

                accum_count += 1

                if accum_count >= grad_accum_embed_head_steps:
                    # Restore accumulated buffer to embed/head/multi-head param grads.
                    for name, p in model.named_parameters():
                        if name.startswith("embed.") or name.startswith("lm_head.") or name.startswith("multi_heads."):
                            if id(p) in accum_buffer_embed_head:
                                p.grad = accum_buffer_embed_head[id(p)].clone().to(device=p.device)
                    # Grad clipping — skipped in the original accumulation branch.
                    if grad_clip and grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    # force_adam=True: a closing cycle always carries freshly
                    # restored embed/head/multi-head grads that must be applied
                    # now, regardless of step parity — otherwise the unconditional
                    # adam_opt.zero_grad() below would silently drop them when the
                    # cycle happens to close on an even step under adam_on_odd_steps.
                    step_speedrun_optimizers(
                        muon_opt,
                        adam_opt,
                        step=step,
                        muon_momentum=muon_momentum,
                        adam_on_odd_steps=adam_on_odd_steps,
                        cautious_wd=cautious_wd,
                        lr_scale=lr_scale,
                        force_adam=True,
                    )
                    accum_buffer_embed_head = None
                    accum_count = 0
                    muon_opt.zero_grad(set_to_none=True)
                    adam_opt.zero_grad(set_to_none=True)
                else:
                    muon_opt.step(momentum=muon_momentum, cautious_wd=cautious_wd, lr_scale=lr_scale)
                    # Zero all grads between micro-steps so stale gradients do
                    # not accumulate for any param group.  Embed/head/multi-head
                    # grads are safe in the buffer; everything else must be
                    # cleared so the next backward produces fresh gradients.
                    muon_opt.zero_grad(set_to_none=True)
                    adam_opt.zero_grad(set_to_none=True)
            else:
                # Original modded-nanogpt flow: Muon gets a fresh gradient and
                # is stepped + cleared every step. Adam-managed params (embed,
                # lm_head, value_embeds, scalars) are only stepped on odd
                # steps -- on a skipped (even) step their .grad is left alone
                # rather than zeroed, so the next backward() accumulates on
                # top of it (PyTorch's default add-into-.grad), matching the
                # original's "don't clear Adam grads on even steps" behavior
                # instead of silently discarding the even step's gradient.
                muon_opt.zero_grad(set_to_none=True)
                loss.backward()
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                adam_stepped = step_speedrun_optimizers(
                    muon_opt,
                    adam_opt,
                    step=step,
                    muon_momentum=muon_momentum,
                    adam_on_odd_steps=adam_on_odd_steps,
                    cautious_wd=cautious_wd,
                    lr_scale=lr_scale,
                )
                if adam_stepped:
                    adam_opt.zero_grad(set_to_none=True)

            # --- Portable: untie embed and lm_head at 2/3 of training ---
            if untie_at_frac > 0.0 and step == int(steps * untie_at_frac):
                if hasattr(model.embed, "weight") and model.lm_head.weight.data_ptr() == model.embed.weight.data_ptr():
                    model.lm_head.weight = nn.Parameter(model.lm_head.weight.data.clone())
                    model.lm_head.weight.data.zero_()
                    lr_val = adam_lr * lm_head_lr_mul
                    wd_val = adam_weight_decay * lm_head_wd_mul
                    adam_opt.add_param_group({
                        "params": [model.lm_head.weight],
                        "lr": lr_val,
                        "weight_decay": wd_val,
                    })

            tokens_trained += effective_batch * current_block
            with torch.no_grad():
                loss_detached = loss.detach()
                ema_loss_tensor = loss_detached if ema_loss_tensor is None else (
                    ema_decay * ema_loss_tensor + (1.0 - ema_decay) * loss_detached
                )
            completed = step + 1
            if step == 0 or completed == steps or completed % log_every == 0:
                ema_loss = ema_loss_tensor.item()
                elapsed = time.time() - start_time
                tokens_per_sec = tokens_trained / elapsed if elapsed > 0 else 0.0
                eta_seconds = (elapsed / completed) * (steps - completed) if completed > 0 else 0.0
                pbar.set_postfix(loss=f"{ema_loss:.4f}", tok_s=f"{tokens_per_sec:.0f}")
                pbar.write(
                    f"[{desc}] step {completed}/{steps} | loss {ema_loss:.4f} "
                    f"| bs={effective_batch} lr_scale={lr_scale:.3f} "
                    f"| {tokens_per_sec:.0f} tok/s | elapsed {elapsed:.1f}s | eta {eta_seconds:.1f}s"
                )

    # Flush any remaining partial accumulation cycle so embed/head grads
    # from the last incomplete cycle are not silently dropped.
    if training_recipe != "record_01_adamw" and grad_accum_embed_head_steps > 1 and accum_buffer_embed_head is not None:
        for name, p in model.named_parameters():
            if name.startswith("embed.") or name.startswith("lm_head.") or name.startswith("multi_heads."):
                if id(p) in accum_buffer_embed_head:
                    p.grad = accum_buffer_embed_head[id(p)].clone().to(device=p.device)
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        step_speedrun_optimizers(
            muon_opt, adam_opt,
            step=step,
            muon_momentum=muon_momentum,
            adam_on_odd_steps=adam_on_odd_steps,
            cautious_wd=cautious_wd,
            lr_scale=lr_scale,
            force_adam=True,
        )
        accum_buffer_embed_head = None
        accum_count = 0

    pbar.close()
    val_loss, acc = _eval_val_loss(
        model, val_data, block=block, batch=batch, vocab_size=vocab_size, device=device
    )
    return val_loss, acc, tokens_trained


def averaged_train_and_eval(
    build_model, train_data, val_data, *, n_runs, base_seed, device, **hparams
):
    """Train+eval ``n_runs`` times with distinct seeds and average the val signal."""
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
            model,
            train_data,
            val_data,
            device=device,
            generator=generator,
            run_label=run_label,
            **hparams,
        )
        if not math.isfinite(loss):
            return float("inf"), 0.0, 0.0, 0, 0
        n_params = sum(p.numel() for p in model.parameters())
        losses.append(loss)
        accs.append(acc)
        total_flops += 6.0 * n_params * tokens_trained
        total_tokens += tokens_trained
    return sum(losses) / len(losses), sum(accs) / len(accs), total_flops, total_tokens, n_params


_TRAINING_COMPONENTS = (
    lr_at_step,
    plan_train_windows,
    plan_eos_aligned_windows,
    make_seq_len_schedule,
    _enforce_max_doc_len,
    _eval_val_loss,
    _compute_multi_token_loss,
    train_and_eval_student,
    averaged_train_and_eval,
)


def training_source() -> str:
    """The verbatim source of the training-recipe helpers, for byte-identical embed."""
    return "\n\n\n".join(inspect.getsource(c).rstrip() for c in _TRAINING_COMPONENTS)
