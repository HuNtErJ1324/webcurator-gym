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

* **EoS-aligned packed batch starts** — EOT-prefixed documents; long docs keep
  intra-document windows, short docs pack into fixed blocks (Speedrun-style).
* **Max document length handling** — reject over-long documents explicitly.
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
    classify_speedrun_params,
    clip_optimizer_grads,
    get_muon_momentum,
    init_speedrun_weights,
    lookup_batch_stage,
    schedule_lr_multiplier,
    set_optimizer_lrs,
    step_speedrun_optimizers,
)
from .val_set import plan_val_windows


def _is_cuda_device(device) -> bool:
    if isinstance(device, torch.device):
        return device.type == "cuda"
    return isinstance(device, str) and device.startswith("cuda")


def prepare_student_model_dtype(model, device):
    """CUDA-only bfloat16 for Muon matrices; Adam groups stay float32.

    CPU behavior is unchanged (all float32). On CUDA, cast the full module to
    bfloat16 then restore Adam-managed parameters to float32 so optimizer state
    and lm_head/softcap/CE remain numerically stable.
    """
    if not _is_cuda_device(device):
        return model
    model.to(dtype=torch.bfloat16)
    _, adam = classify_speedrun_params(model)
    for params in adam.values():
        for param in params:
            param.data = param.data.to(dtype=torch.float32)
    return model


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


def encode_document_tokens(documents, encoder, max_document_tokens=None):
    """Build the official FineWeb document token stream and exact ranges.

    KellerJordan/modded-nanogpt prefixes every source document, including an empty
    one, with the GPT-2 EOT token 50256 and then appends
    ``encoder.encode_ordinary(document)``. No text delimiter represents a boundary.
    ``max_document_tokens`` is an explicit per-document validation limit (including
    EOT); over-long documents are rejected rather than truncated.
    """
    token_ids = []
    document_ranges = []
    cap = None if max_document_tokens is None else int(max_document_tokens)
    if cap is not None and cap < 1:
        raise ValueError("max_document_tokens must be positive")
    for index, document in enumerate(documents):
        start = len(token_ids)
        encoded = list(encoder.encode_ordinary(document))
        document_tokens = 1 + len(encoded)
        if cap is not None and document_tokens > cap:
            raise ValueError(
                f"document {index} has {document_tokens} tokens including EOT, "
                f"exceeding max_document_tokens={cap}; documents are never truncated"
            )
        token_ids.append(50256)
        token_ids.extend(encoded)
        document_ranges.append((start, len(token_ids)))
    return token_ids, document_ranges


def plan_eos_aligned_windows(n_tokens, block, document_ranges, lookahead=1):
    """Plan training-window starts over an EOT-prefixed document stream.

    Long documents (length >= ``block + lookahead``) keep non-overlapping
    intra-document starts with stride ``block``.

    Short documents are packed with deterministic sequential, non-overlapping
    windows (portable adaptation of modded-nanogpt ``Shard.next_batch``): each
    pack starts at the next unused short document's leading EOT/BOS, covers
    ``block + lookahead`` tokens of the concatenated stream, and advances past
    every document that overlaps that span so short docs are not oversampled.
    If a pack truncates into a long document, residual intra-document stride
    windows begin at the first uncovered position of that long document.

    Boundary contract (Speedrun-aligned as far as this SDPA port allows):

    * **Targets** use the flat next-token shift over the window, so a target may
      be the next document's leading EOT (Speedrun ``buf[:-1]`` / ``buf[1:]``).
    * **Attention** must not cross document boundaries: callers use
      :func:`batch_document_attn_mask` for packed multi-document windows
      (Speedrun uses flash-attn varlen ``cum_lengths`` for the same guarantee).
    * Document ranges must be sorted, disjoint, and contiguous (no gaps).
    * There is no flat-stream fallback that ignores ``document_ranges``.
    """
    block = int(block)
    n_tokens = int(n_tokens)
    lookahead = max(1, int(lookahead))
    need = block + lookahead
    ranges = []
    previous_end = 0
    for raw_start, raw_end in document_ranges or ():
        start, end = int(raw_start), int(raw_end)
        if start < previous_end or end < start or end > n_tokens:
            raise ValueError("document ranges must be sorted, disjoint, and in bounds")
        if ranges and start != previous_end:
            raise ValueError(
                "document ranges must be contiguous with no gaps; "
                f"found gap between {previous_end} and {start}"
            )
        previous_end = end
        ranges.append((start, end))

    starts = []
    cursor = 0
    while cursor < len(ranges):
        doc_start, doc_end = ranges[cursor]
        length = doc_end - doc_start
        if length >= need:
            starts.extend(range(doc_start, doc_end - need + 1, block))
            cursor += 1
            continue

        if doc_start + need > n_tokens:
            cursor += 1
            continue

        pack_start = doc_start
        pack_end = pack_start + need
        starts.append(pack_start)
        while cursor < len(ranges) and ranges[cursor][0] < pack_end:
            d_start, d_end = ranges[cursor]
            d_len = d_end - d_start
            if d_len >= need:
                residual_from = max(pack_end, d_start)
                if residual_from + need <= d_end:
                    starts.extend(range(residual_from, d_end - need + 1, block))
                cursor += 1
                break
            cursor += 1
    return starts


def window_document_ids(start, length, document_ranges, device=None):
    """Per-position document ids for ``train_src[start:start+length]``."""
    start = int(start)
    length = int(length)
    ids = torch.full((length,), -1, dtype=torch.long, device=device)
    end = start + length
    for doc_id, (raw_start, raw_end) in enumerate(document_ranges or ()):
        doc_start, doc_end = int(raw_start), int(raw_end)
        lo = max(doc_start, start)
        hi = min(doc_end, end)
        if lo < hi:
            ids[lo - start : hi - start] = doc_id
    return ids


def build_document_attn_mask(doc_ids):
    """Boolean SDPA mask ``(B, T, T)``; ``True`` means the key may participate.

    Keeps causal same-document positions. Gap ids (``< 0``) never attend to each
    other; they are restricted to self-attention so softmax stays well-defined.
    """
    if doc_ids.dim() != 2:
        raise ValueError(f"doc_ids must be (batch, seq), got shape {tuple(doc_ids.shape)}")
    _batch, seq_len = doc_ids.shape
    idx = torch.arange(seq_len, device=doc_ids.device)
    causal = idx[None, :] <= idx[:, None]
    valid = doc_ids >= 0
    same_document = (
        (doc_ids[:, :, None] == doc_ids[:, None, :])
        & valid[:, :, None]
        & valid[:, None, :]
    )
    keep = causal.unsqueeze(0) & same_document
    eye = torch.eye(seq_len, dtype=torch.bool, device=doc_ids.device)
    return keep | eye.unsqueeze(0)


def batch_document_attn_mask(starts, block, document_ranges, device):
    """Build a batched document keep-mask, or ``None`` when every window is single-doc."""
    if not starts:
        return None
    rows = []
    any_cross = False
    any_gap = False
    block = int(block)
    for start in starts:
        doc_ids = window_document_ids(
            start, block, document_ranges, device=device
        )
        if doc_ids.numel() > 0 and bool((doc_ids < 0).any().item()):
            any_gap = True
        if doc_ids.numel() > 0 and bool((doc_ids != doc_ids[0]).any().item()):
            any_cross = True
        rows.append(doc_ids)
    if not any_cross and not any_gap:
        return None
    return build_document_attn_mask(torch.stack(rows, dim=0))


def shuffled_window_starts(starts, generator):
    """Return one fixed-seed-deterministic permutation of planned starts."""
    permutation = torch.randperm(len(starts), generator=generator)
    return [starts[index] for index in permutation.tolist()]


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


def _score_hidden_chunked(model, hidden, targets, *, vocab_size, logit_chunk_tokens):
    """Sum CE + correct counts over ``hidden`` without a full (N, vocab) allocation.

    When ``logit_chunk_tokens`` is set, projects ``lm_head``/softcap in row chunks so
    peak activation memory stays O(chunk * vocab) instead of O(N * vocab). Mean CE
    and accuracy over the full set are identical to a single full-vocab pass.
    """
    flat_h = hidden.reshape(-1, hidden.size(-1))
    flat_y = targets.reshape(-1)
    n = int(flat_h.size(0))
    chunk = int(logit_chunk_tokens) if logit_chunk_tokens is not None else n
    if chunk < 1:
        raise ValueError(f"logit_chunk_tokens must be >= 1, got {logit_chunk_tokens}")
    loss_sum = 0.0
    correct = 0
    apply_head = getattr(model, "apply_lm_head", None)
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        h = flat_h[start:end]
        y = flat_y[start:end]
        if apply_head is not None:
            logits = apply_head(h)
        else:
            logits = model.softcap * torch.tanh(
                model.lm_head(h.float()).float() / model.softcap
            )
        loss_sum += F.cross_entropy(logits.float(), y, reduction="sum").item()
        correct += (logits.float().argmax(-1) == y).sum().item()
        del logits
    return loss_sum, correct


def _eval_val_loss(
    model,
    val_data,
    *,
    block,
    batch,
    vocab_size,
    device,
    logit_chunk_tokens=None,
):
    """Mean held-out CE / accuracy over every predictable val target.

    ``batch`` is the validation microbatch (independent of training batch size).
    When the model exposes ``forward_hidden``, the trunk runs once per microbatch
    and ``lm_head``/softcap are applied in ``logit_chunk_tokens``-sized chunks so
    oversized full-vocab logit tensors are never materialized.
    """
    model.eval()
    val_windows = plan_val_windows(len(val_data), block)
    total = sum(length for _, length in val_windows)
    loss_sum = 0.0
    correct = 0
    val_batch = max(1, int(batch))
    forward_hidden = getattr(model, "forward_hidden", None)
    with torch.no_grad():
        wi = 0
        while wi < len(val_windows):
            length = val_windows[wi][1]
            starts = []
            while (
                wi < len(val_windows)
                and val_windows[wi][1] == length
                and len(starts) < val_batch
            ):
                starts.append(val_windows[wi][0])
                wi += 1
            xb = torch.stack([val_data[s : s + length] for s in starts]).to(device)
            yb = torch.stack([val_data[s + 1 : s + length + 1] for s in starts]).to(
                device
            )
            if forward_hidden is not None:
                hidden = forward_hidden(xb)
                chunk_loss, chunk_correct = _score_hidden_chunked(
                    model,
                    hidden,
                    yb,
                    vocab_size=vocab_size,
                    logit_chunk_tokens=logit_chunk_tokens,
                )
                loss_sum += chunk_loss
                correct += chunk_correct
                del hidden
            else:
                logits = model(xb)
                if logit_chunk_tokens is None:
                    loss_sum += F.cross_entropy(
                        logits.float().reshape(-1, vocab_size),
                        yb.reshape(-1),
                        reduction="sum",
                    ).item()
                    correct += (logits.float().argmax(-1) == yb).sum().item()
                else:
                    flat_logits = logits.float().reshape(-1, vocab_size)
                    flat_y = yb.reshape(-1)
                    chunk = int(logit_chunk_tokens)
                    for start in range(0, flat_logits.size(0), chunk):
                        end = min(start + chunk, flat_logits.size(0))
                        sl = flat_logits[start:end]
                        sy = flat_y[start:end]
                        loss_sum += F.cross_entropy(sl, sy, reduction="sum").item()
                        correct += (sl.argmax(-1) == sy).sum().item()
                del logits
            del xb, yb
    return loss_sum / total, correct / total


def _compute_multi_token_loss(logits, hidden, y, multi_heads, y_future, vocab_size):
    """Total loss including auxiliary multi-token prediction losses.

    Multi-token heads are applied to the final hidden states (pre-lm_head),
    not to logits, since the heads are ``nn.Linear(model_dim, vocab_size)``.
    """
    main_loss = F.cross_entropy(logits.float().view(-1, vocab_size), y.view(-1))
    if (
        multi_heads is None
        or not hasattr(multi_heads, "heads")
        or not multi_heads.heads
    ):
        return main_loss
    mt_loss = 0.0
    mt_weight = 0.3
    for k, head in enumerate(multi_heads.heads):
        if k < len(y_future) and y_future[k] is not None:
            head_logits = head(hidden).float()
            mt_loss = mt_loss + F.cross_entropy(
                head_logits.view(-1, vocab_size), y_future[k].view(-1)
            )
    if mt_loss > 0.0:
        return main_loss + mt_weight * mt_loss
    return main_loss


def _microbatch_ranges(batch_size, microbatch_size):
    """Yield ``(start, end)`` slices that cover ``batch_size`` without exceeding ``microbatch_size``.

    ``microbatch_size is None`` (or >= batch) yields a single full-batch slice — legacy
    behavior. Used so scheduled effective batches (16→32→48) can honor the intended
    token/update semantics while never materializing full-vocab logits for the whole
    effective batch at once.
    """
    n = int(batch_size)
    if n < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if microbatch_size is None:
        yield 0, n
        return
    micro = int(microbatch_size)
    if micro < 1:
        raise ValueError(f"train_microbatch_size must be >= 1, got {microbatch_size}")
    if micro >= n:
        yield 0, n
        return
    for start in range(0, n, micro):
        yield start, min(start + micro, n)


def _scaled_microbatch_loss(loss, micro_n, total_n):
    """Scale mean-reduced microbatch loss so summed grads match full-batch mean CE."""
    return loss * (float(micro_n) / float(total_n))


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
    muon_weight_decay=1.2,
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
    document_ranges=None,
    grad_accum_embed_head_steps=1,
    seq_len_schedule=False,
    multi_token_pred=0,
    untie_at_frac=0.0,
    cautious_wd=False,
    nor_muon=True,
    polar_express=False,
    train_microbatch_size=None,
    val_batch_size=None,
    val_logit_chunk_tokens=None,
):
    """Train ``model`` and score held-out CE; returns ``(val_loss, accuracy, tokens_trained)``."""
    prepare_student_model_dtype(model, device)
    block = int(block_size)
    batch = int(batch_size)
    steps = int(steps)
    eval_batch = int(batch_size if val_batch_size is None else val_batch_size)
    if eval_batch < 1:
        raise ValueError(f"val_batch_size must be >= 1, got {val_batch_size}")
    if train_microbatch_size is not None and int(train_microbatch_size) < 1:
        raise ValueError(
            f"train_microbatch_size must be >= 1, got {train_microbatch_size}"
        )

    train_src = train_data
    if document_ranges is None and len(train_src) <= block + 1:
        train_src = train_src.repeat(math.ceil((block + 2) / max(len(train_src), 1)))

    # Document-aware training never falls back to flat windows that ignore
    # document_ranges. Short documents are packed into fixed blocks; long
    # documents keep intra-document starts. Packed multi-document windows use
    # an SDPA document mask so attention does not cross EOT boundaries.
    if document_ranges is not None:
        starts = plan_eos_aligned_windows(
            len(train_src),
            block,
            document_ranges,
            lookahead=max(1, int(multi_token_pred) + 1),
        )
        if not starts:
            raise ValueError(
                "no packed document window fits block_size + lookahead tokens; "
                "lower block_size or provide more/longer documents"
            )
    else:
        starts = plan_train_windows(len(train_src), block)

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
                order = shuffled_window_starts(starts, generator)
                cursor = 0
            i = order[cursor]
            cursor += 1
            batch_starts.append(i)
            xs.append(train_src[i : i + current_block])
            ys.append(train_src[i + 1 : i + current_block + 1])
        attn_mask = None
        if document_ranges is not None:
            attn_mask = batch_document_attn_mask(
                batch_starts, current_block, document_ranges, device
            )
        return (
            torch.stack(xs).to(device),
            torch.stack(ys).to(device),
            batch_starts,
            attn_mask,
        )

    log_every = max(50, max(1, steps // 50))
    ema_loss_tensor = None
    ema_decay = 0.9
    start_time = time.time()
    desc = f"{run_label}train" if run_label else "train"
    pbar = tqdm(
        range(steps), total=steps, desc=desc, unit="step", leave=False, file=sys.stdout
    )

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
            x, y, _batch_starts, attn_mask = next_batch(batch, current_block)
            opt.zero_grad(set_to_none=True)
            # Loss-scaled microbatch accumulation: summed microbatch grads equal a
            # single full-batch mean-CE backward while capping peak logit memory.
            loss_weighted = None
            for start, end in _microbatch_ranges(batch, train_microbatch_size):
                xb = x[start:end]
                yb = y[start:end]
                mb_mask = None if attn_mask is None else attn_mask[start:end]
                logits = model(xb, attn_mask=mb_mask)
                loss = F.cross_entropy(logits.float().view(-1, vocab_size), yb.view(-1))
                _scaled_microbatch_loss(loss, end - start, batch).backward()
                with torch.no_grad():
                    piece = loss.detach() * float(end - start)
                    loss_weighted = (
                        piece if loss_weighted is None else loss_weighted + piece
                    )
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            tokens_trained += batch * current_block
            with torch.no_grad():
                loss_detached = loss_weighted / float(batch)
                ema_loss_tensor = (
                    loss_detached
                    if ema_loss_tensor is None
                    else (
                        ema_decay * ema_loss_tensor + (1.0 - ema_decay) * loss_detached
                    )
                )
            completed = step + 1
            if step == 0 or completed == steps or completed % log_every == 0:
                ema_loss = ema_loss_tensor.item()
                elapsed = time.time() - start_time
                tokens_per_sec = tokens_trained / elapsed if elapsed > 0 else 0.0
                eta_seconds = (
                    (elapsed / completed) * (steps - completed)
                    if completed > 0
                    else 0.0
                )
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
        muon_warm = (
            muon_warmup_steps
            if muon_warmup_steps is not None
            else min(300, max(1, steps // 5))
        )
        muon_cd = (
            muon_cooldown_steps
            if muon_cooldown_steps is not None
            else min(50, max(1, steps // 20))
        )
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
            x, y, batch_starts, attn_mask = next_batch(effective_batch, current_block)

            # --- Portable: multi-token prediction targets ---
            use_mt = (
                multi_token_pred > 0
                and hasattr(model, "multi_heads")
                and model.multi_heads is not None
            )
            y_future = []
            if use_mt:
                for k in range(multi_token_pred):
                    shift = k + 2
                    future_targets = []
                    for sidx in batch_starts:
                        if sidx + shift + current_block <= len(train_src):
                            future_targets.append(
                                train_src[sidx + shift : sidx + shift + current_block]
                            )
                        else:
                            n_avail = max(0, len(train_src) - (sidx + shift))
                            if n_avail > 0:
                                seg = train_src[sidx + shift :]
                                seg = torch.cat(
                                    [
                                        seg,
                                        torch.zeros(
                                            current_block - n_avail,
                                            dtype=train_src.dtype,
                                        ),
                                    ]
                                )
                            else:
                                seg = torch.zeros(current_block, dtype=train_src.dtype)
                            future_targets.append(seg)
                    y_future.append(torch.stack(future_targets).to(device))

            # Loss-scaled microbatch accumulation: honor scheduled effective_batch
            # tokens/update while capping peak full-vocab logit memory. Grads sum
            # across microbatches so the net .grad equals a single full-effective
            # -batch mean-CE backward before any optimizer/accumulation bookkeeping.
            if grad_accum_embed_head_steps <= 1:
                muon_opt.zero_grad(set_to_none=True)
            loss_weighted = None
            for start, end in _microbatch_ranges(
                effective_batch, train_microbatch_size
            ):
                xb = x[start:end]
                yb = y[start:end]
                mb_mask = None if attn_mask is None else attn_mask[start:end]
                logits = model(xb, attn_mask=mb_mask, output_hidden=use_mt)
                if use_mt:
                    hidden = logits[1]
                    y_future_mb = [yf[start:end] for yf in y_future]
                    loss = _compute_multi_token_loss(
                        logits[0],
                        hidden,
                        yb,
                        model.multi_heads,
                        y_future_mb,
                        vocab_size,
                    )
                else:
                    loss = F.cross_entropy(
                        logits.float().view(-1, vocab_size), yb.view(-1)
                    )
                _scaled_microbatch_loss(loss, end - start, effective_batch).backward()
                with torch.no_grad():
                    piece = loss.detach() * float(end - start)
                    loss_weighted = (
                        piece if loss_weighted is None else loss_weighted + piece
                    )

            # --- Portable: true N-step gradient accumulation for embed + lm_head ---
            if grad_accum_embed_head_steps > 1:
                embed_head_params = []
                for name, p in model.named_parameters():
                    if (
                        name.startswith("embed.")
                        or name.startswith("lm_head.")
                        or name.startswith("multi_heads.")
                    ):
                        embed_head_params.append(p)

                # backward() already ran per-microbatch above; grads for embed/head
                # /multi-head params are fully summed for this effective batch.
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
                        if (
                            name.startswith("embed.")
                            or name.startswith("lm_head.")
                            or name.startswith("multi_heads.")
                        ):
                            if id(p) in accum_buffer_embed_head:
                                p.grad = (
                                    accum_buffer_embed_head[id(p)]
                                    .clone()
                                    .to(device=p.device)
                                )
                    # Per-optimizer clipping occurs inside the actual updates below.
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
                        grad_clip=grad_clip,
                    )
                    accum_buffer_embed_head = None
                    accum_count = 0
                    muon_opt.zero_grad(set_to_none=True)
                    adam_opt.zero_grad(set_to_none=True)
                else:
                    clip_optimizer_grads(muon_opt, grad_clip)
                    muon_opt.step(
                        momentum=muon_momentum,
                        cautious_wd=cautious_wd,
                        lr_scale=lr_scale,
                    )
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
                # (muon grads were zeroed before the microbatch backward loop; the
                # loss-scaled backward already ran per microbatch above.)
                adam_stepped = step_speedrun_optimizers(
                    muon_opt,
                    adam_opt,
                    step=step,
                    muon_momentum=muon_momentum,
                    adam_on_odd_steps=adam_on_odd_steps,
                    cautious_wd=cautious_wd,
                    lr_scale=lr_scale,
                    grad_clip=grad_clip,
                )
                if adam_stepped:
                    adam_opt.zero_grad(set_to_none=True)

            # --- Portable: untie embed and lm_head at 2/3 of training ---
            if untie_at_frac > 0.0 and step == int(steps * untie_at_frac):
                if (
                    hasattr(model.embed, "weight")
                    and model.lm_head.weight.data_ptr() == model.embed.weight.data_ptr()
                ):
                    model.lm_head.weight = nn.Parameter(
                        model.lm_head.weight.data.clone()
                    )
                    model.lm_head.weight.data.zero_()
                    lr_val = adam_lr * lm_head_lr_mul
                    wd_val = adam_weight_decay * lm_head_wd_mul
                    adam_opt.add_param_group(
                        {
                            "params": [model.lm_head.weight],
                            "lr": lr_val,
                            "weight_decay": wd_val,
                            "betas": (0.5, 0.95),
                        }
                    )

            tokens_trained += effective_batch * current_block
            with torch.no_grad():
                loss_detached = loss_weighted / float(effective_batch)
                ema_loss_tensor = (
                    loss_detached
                    if ema_loss_tensor is None
                    else (
                        ema_decay * ema_loss_tensor + (1.0 - ema_decay) * loss_detached
                    )
                )
            completed = step + 1
            if step == 0 or completed == steps or completed % log_every == 0:
                ema_loss = ema_loss_tensor.item()
                elapsed = time.time() - start_time
                tokens_per_sec = tokens_trained / elapsed if elapsed > 0 else 0.0
                eta_seconds = (
                    (elapsed / completed) * (steps - completed)
                    if completed > 0
                    else 0.0
                )
                pbar.set_postfix(loss=f"{ema_loss:.4f}", tok_s=f"{tokens_per_sec:.0f}")
                pbar.write(
                    f"[{desc}] step {completed}/{steps} | loss {ema_loss:.4f} "
                    f"| bs={effective_batch} lr_scale={lr_scale:.3f} "
                    f"| {tokens_per_sec:.0f} tok/s | elapsed {elapsed:.1f}s | eta {eta_seconds:.1f}s"
                )

    # Flush any remaining partial accumulation cycle so embed/head grads
    # from the last incomplete cycle are not silently dropped.
    if (
        training_recipe != "record_01_adamw"
        and grad_accum_embed_head_steps > 1
        and accum_buffer_embed_head is not None
    ):
        for name, p in model.named_parameters():
            if (
                name.startswith("embed.")
                or name.startswith("lm_head.")
                or name.startswith("multi_heads.")
            ):
                if id(p) in accum_buffer_embed_head:
                    p.grad = accum_buffer_embed_head[id(p)].clone().to(device=p.device)
        step_speedrun_optimizers(
            muon_opt,
            adam_opt,
            step=step,
            muon_momentum=muon_momentum,
            adam_on_odd_steps=adam_on_odd_steps,
            cautious_wd=cautious_wd,
            lr_scale=lr_scale,
            force_adam=True,
            grad_clip=grad_clip,
        )
        accum_buffer_embed_head = None
        accum_count = 0

    pbar.close()
    # Drop optimizer state before the held-out pass so validation activations are
    # not competing with Muon/Adam buffers for the last ~10GB on A100-80GB.
    if training_recipe == "record_01_adamw":
        del opt
    else:
        del muon_opt, adam_opt
    if torch.cuda.is_available() and (
        (hasattr(device, "type") and device.type == "cuda")
        or (isinstance(device, str) and device.startswith("cuda"))
    ):
        torch.cuda.empty_cache()
    val_loss, acc = _eval_val_loss(
        model,
        val_data,
        block=block,
        batch=eval_batch,
        vocab_size=vocab_size,
        device=device,
        logit_chunk_tokens=val_logit_chunk_tokens,
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
    return (
        sum(losses) / len(losses),
        sum(accs) / len(accs),
        total_flops,
        total_tokens,
        n_params,
    )


_TRAINING_COMPONENTS = (
    lr_at_step,
    plan_train_windows,
    encode_document_tokens,
    plan_eos_aligned_windows,
    window_document_ids,
    build_document_attn_mask,
    batch_document_attn_mask,
    shuffled_window_starts,
    make_seq_len_schedule,
    _is_cuda_device,
    prepare_student_model_dtype,
    _microbatch_ranges,
    _scaled_microbatch_loss,
    _score_hidden_chunked,
    _eval_val_loss,
    _compute_multi_token_loss,
    train_and_eval_student,
    averaged_train_and_eval,
)


def training_source() -> str:
    """The verbatim source of the training-recipe helpers, for byte-identical embed."""
    return "\n\n\n".join(inspect.getsource(c).rstrip() for c in _TRAINING_COMPONENTS)
