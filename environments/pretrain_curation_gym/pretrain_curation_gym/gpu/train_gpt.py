"""Single-file NanoGPT-speedrun proxy-student trainer."""

import atexit
import bisect
import json
import math
import os
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TRAIN_WORKDIR = "/workspace"
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

_FRAC_SUM_TOL = 1e-6
MODDED_NANOGPT_UPSTREAM_COMMIT = "edf47a05a12062d661c4cfd4eef848c5ab5bed32"


def parse_document_payload(corpus_text):
    """Return explicit train/validation document lists from a corpus payload."""
    try:
        payload = json.loads(corpus_text)
    except json.JSONDecodeError:
        return None, None
    if not isinstance(payload, dict):
        return None, None
    if payload.get("format") == "document-list-v1":
        documents = payload.get("documents")
        if not isinstance(documents, list) or not all(
            isinstance(doc, str) for doc in documents
        ):
            raise ValueError("invalid document-list-v1 corpus payload")
        return documents, None
    if payload.get("format") == "document-split-v1":
        documents = payload.get("train_documents")
        val_documents = payload.get("val_documents")
        if (
            not isinstance(documents, list)
            or not documents
            or not all(isinstance(doc, str) for doc in documents)
            or not isinstance(val_documents, list)
            or not val_documents
            or not all(isinstance(doc, str) for doc in val_documents)
        ):
            raise ValueError("invalid document-split-v1 corpus payload")
        return documents, val_documents
    return None, None


def batch_stage_boundaries(
    total_steps: int, stage_fracs: Sequence[float]
) -> list[tuple[int, int]]:
    """Half-open ``(start, end)`` stage intervals covering ``[0, total_steps)``."""
    frac_sum = sum(stage_fracs)
    if not math.isclose(frac_sum, 1.0, rel_tol=0, abs_tol=_FRAC_SUM_TOL):
        raise ValueError(f"stage_fracs must sum to 1.0, got {frac_sum}")
    scheduled = max(1, int(total_steps))
    ends = [0]
    for frac in stage_fracs[:-1]:
        ends.append(min(scheduled, ends[-1] + max(1, round(frac * scheduled))))
    ends.append(scheduled)
    return [(ends[i], ends[i + 1]) for i in range(len(stage_fracs))]


def make_seq_len_schedule(total_steps, max_block):
    """Scale the pinned upstream 896→2048 context schedule to ``max_block``."""
    max_block = int(max_block)
    short_steps = max(1, round(int(total_steps) / 3))
    short_block = max(8, round(max_block * 7 / 16))

    def block_at_step(step):
        return short_block if int(step) < short_steps else max_block

    return block_at_step


def scheduled_presentation_tokens(
    steps: int,
    *,
    batch_size: int,
    block_size: int,
    batch_stage_muls: Sequence[int],
    batch_stage_fracs: Sequence[float],
    batch_schedule_enabled: bool = True,
    seq_len_schedule: bool = False,
) -> int:
    """Tokens presented across ``steps`` under the staged batch schedule."""
    steps = max(0, int(steps))
    if steps == 0:
        return 0
    per_base = int(batch_size) * int(block_size)
    if batch_schedule_enabled:
        if len(batch_stage_fracs) != len(batch_stage_muls):
            raise ValueError(
                "batch_stage_fracs and batch_stage_muls must have equal length"
            )
        boundaries = batch_stage_boundaries(steps, batch_stage_fracs)
        muls = [int(m) for m in batch_stage_muls]
    else:
        boundaries = [(0, steps)]
        muls = [1]
    if not seq_len_schedule:
        return sum(
            (end - start) * per_base * mul
            for (start, end), mul in zip(boundaries, muls, strict=True)
        )
    block_at = make_seq_len_schedule(steps, int(block_size))
    batch = int(batch_size)
    total = 0
    for (start, end), mul in zip(boundaries, muls, strict=True):
        for step in range(start, end):
            total += batch * mul * block_at(step)
    return total


def _max_stage_deviation_bound(muls: Sequence[int]) -> int:
    """N-independent bound on schedule deviation from the linear trend."""
    k = len(muls)
    if k <= 1:
        return 0
    bound = sum((2 * i - 1) * muls[i - 1] for i in range(1, k))
    bound += (k - 1) * muls[-1]
    return bound


def steps_for_token_budget(
    budget: int,
    *,
    batch_size: int,
    block_size: int,
    batch_stage_muls: Sequence[int],
    batch_stage_fracs: Sequence[float],
    batch_schedule_enabled: bool = True,
    seq_len_schedule: bool = False,
) -> int:
    """Minimal steps whose scheduled presentations meet ``budget``."""
    if seq_len_schedule:
        return _steps_for_budget_with_seq_len_schedule(
            budget,
            batch_size=batch_size,
            block_size=block_size,
            batch_stage_muls=batch_stage_muls,
            batch_stage_fracs=batch_stage_fracs,
            batch_schedule_enabled=batch_schedule_enabled,
        )
    budget = int(budget)
    if budget < 1:
        return 1
    per_base = int(batch_size) * int(block_size)
    if per_base < 1:
        raise ValueError("batch_size * block_size must be >= 1")

    def tokens_at(n: int) -> int:
        return scheduled_presentation_tokens(
            n,
            batch_size=batch_size,
            block_size=block_size,
            batch_stage_muls=batch_stage_muls,
            batch_stage_fracs=batch_stage_fracs,
            batch_schedule_enabled=batch_schedule_enabled,
        )

    if not batch_schedule_enabled:
        return max(1, math.ceil(budget / per_base))

    if len(batch_stage_fracs) != len(batch_stage_muls):
        raise ValueError(
            "batch_stage_fracs and batch_stage_muls must have equal length"
        )

    muls = [int(m) for m in batch_stage_muls]
    if min(muls) < 1:
        raise ValueError("batch_stage_muls must be >= 1")
    max_mul = max(muls)
    min_mul = min(muls)

    n_floor = max(1, math.ceil(budget / (per_base * max_mul)))
    n_ceiling = max(1, math.ceil(budget / (per_base * min_mul)))

    mul_avg = sum(float(f) * m for f, m in zip(batch_stage_fracs, muls, strict=True))
    if max_mul == min_mul or mul_avg <= 0:
        return max(
            1, min(n_ceiling, math.ceil(budget / (per_base * max(mul_avg, min_mul))))
        )

    # Bound is step*mul units; scale by per_base for tokens.
    p_bound_tokens = per_base * _max_stage_deviation_bound(muls)
    # Integer margin for float/rounding slop.
    margin = 4
    lo = math.floor((budget - p_bound_tokens) / (per_base * mul_avg)) - margin
    hi = math.ceil((budget + p_bound_tokens) / (per_base * mul_avg)) + margin
    lo = max(n_floor, lo)
    hi = min(n_ceiling, max(hi, lo))
    lo = min(lo, hi)

    for n in range(lo, hi + 1):
        if tokens_at(n) >= budget:
            return n

    return n_ceiling


_SEQ_BUDGET_STEPS_CACHE: dict = {}


def _steps_for_budget_with_seq_len_schedule(
    budget: int,
    *,
    batch_size: int,
    block_size: int,
    batch_stage_muls: Sequence[int],
    batch_stage_fracs: Sequence[float],
    batch_schedule_enabled: bool,
) -> int:
    """Minimal steps meeting ``budget`` when the seq-len warmup shrinks windows."""
    budget = int(budget)
    if budget < 1:
        return 1
    key = (
        budget,
        int(batch_size),
        int(block_size),
        tuple(int(m) for m in batch_stage_muls),
        tuple(float(f) for f in batch_stage_fracs),
        bool(batch_schedule_enabled),
    )
    cached = _SEQ_BUDGET_STEPS_CACHE.get(key)
    if cached is not None:
        return cached

    def tokens_at(n: int) -> int:
        return scheduled_presentation_tokens(
            n,
            batch_size=batch_size,
            block_size=block_size,
            batch_stage_muls=batch_stage_muls,
            batch_stage_fracs=batch_stage_fracs,
            batch_schedule_enabled=batch_schedule_enabled,
            seq_len_schedule=True,
        )

    lo = steps_for_token_budget(
        budget,
        batch_size=batch_size,
        block_size=block_size,
        batch_stage_muls=batch_stage_muls,
        batch_stage_fracs=batch_stage_fracs,
        batch_schedule_enabled=batch_schedule_enabled,
        seq_len_schedule=False,
    )
    if tokens_at(lo) >= budget:
        # N < lo cannot meet budget even under the fixed-block upper bound.
        _SEQ_BUDGET_STEPS_CACHE[key] = lo
        return lo
    hi = lo
    while tokens_at(hi) < budget:
        hi = max(hi + 1, int(hi * 1.05))
    while lo < hi:
        mid = (lo + hi) // 2
        if tokens_at(mid) >= budget:
            hi = mid
        else:
            lo = mid + 1
    for n in range(max(1, lo - 48), lo):
        if tokens_at(n) >= budget:
            lo = n
            break
    _SEQ_BUDGET_STEPS_CACHE[key] = lo
    return lo


def plan_val_windows(n_tokens, block):
    """Non-overlapping windows covering EVERY held-out next-token target."""
    block = int(block)
    if block < 1:
        raise ValueError(f"block must be >= 1, got {block}")
    n_targets = int(n_tokens) - 1
    if n_targets < 1:
        raise ValueError(
            f"held-out val set has no predictable positions (n_tokens={n_tokens})"
        )
    windows = []
    start = 0
    while start < n_targets:
        length = min(block, n_targets - start)
        windows.append((start, length))
        start += length
    return windows


try:
    import torch
except ModuleNotFoundError:
    torch = None
else:
    import numpy as np
    import tiktoken
    import torch.nn as nn
    from torch.nn import functional as F
    from tqdm import tqdm

    class RMSNorm(nn.Module):
        """RMS normalization with a learnable per-feature gain."""

        def __init__(self, dim: int):
            super().__init__()
            self.gains = nn.Parameter(torch.ones(dim))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return F.rms_norm(x, (x.size(-1),), weight=self.gains.type_as(x))

    class Rotary(nn.Module):
        """Half-truncate, base-frequency rotary position embeddings (RoPE)."""

        def __init__(self, head_dim: int, base_inv_freq: float = 1024.0):
            super().__init__()
            if head_dim % 4 != 0:
                raise ValueError(f"head_dim must be divisible by 4, got {head_dim}")
            angular_freq = (1.0 / base_inv_freq) ** torch.linspace(
                0, 1, steps=head_dim // 4, dtype=torch.float32
            )
            angular_freq = torch.cat(
                [angular_freq, angular_freq.new_zeros(head_dim // 4)]
            )
            self.register_buffer("angular_freq", angular_freq, persistent=False)

        def forward(self, x_BTHD: torch.Tensor) -> torch.Tensor:
            pos = torch.arange(
                x_BTHD.size(1), dtype=torch.float32, device=x_BTHD.device
            )
            theta = torch.outer(pos, self.angular_freq)[None, :, None, :]
            cos, sin = theta.cos(), theta.sin()
            x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
            y1 = x1 * cos + x2 * sin
            y2 = x1 * (-sin) + x2 * cos
            return torch.cat((y1, y2), 3).type_as(x_BTHD)

    def _sliding_window_mask(
        seq_len: int, window_size: int, device: torch.device
    ) -> torch.Tensor:
        """Causal band mask for SDPA: ``True`` means the key may participate."""
        idx = torch.arange(seq_len, device=device)
        return (idx[None, :] <= idx[:, None]) & (
            idx[None, :] >= idx[:, None] - window_size + 1
        )

    def _causal_attn_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """Boolean causal mask ``(T, T)``; ``True`` means the key may participate."""
        idx = torch.arange(seq_len, device=device)
        return idx[None, :] <= idx[:, None]

    def _combine_attn_masks(
        seq_len: int,
        device: torch.device,
        *,
        window_size: int | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """AND keep-masks into one SDPA boolean mask (``True`` = participate)."""
        combined: torch.Tensor | None = None
        if window_size is not None and window_size < seq_len:
            combined = _sliding_window_mask(seq_len, window_size, device)
        if attn_mask is not None:
            mask = attn_mask
            if mask.dim() == 2:
                pass
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            elif mask.dim() == 4:
                if mask.size(1) != 1:
                    raise ValueError(
                        f"attn_mask batch head dim must be 1, got shape {tuple(mask.shape)}"
                    )
            else:
                raise ValueError(
                    f"attn_mask must be 2/3/4-D, got shape {tuple(mask.shape)}"
                )
            combined = mask if combined is None else (combined & mask)
        return combined

    class CausalSelfAttention(nn.Module):
        """Causal self-attention with QK-norm, RoPE, value-residual mix, and head gating."""

        def __init__(
            self,
            dim: int,
            num_heads: int,
            *,
            attn_scale: float = 0.12,
            xsa_enabled: bool = False,
        ):
            super().__init__()
            if dim % num_heads != 0:
                raise ValueError(
                    f"dim ({dim}) must be divisible by num_heads ({num_heads})"
                )
            self.num_heads = num_heads
            self.head_dim = dim // num_heads
            self.attn_scale = float(attn_scale)
            self.q = nn.Linear(dim, dim, bias=False)
            self.k = nn.Linear(dim, dim, bias=False)
            self.v = nn.Linear(dim, dim, bias=False)
            self.proj = nn.Linear(dim, dim, bias=False)
            self.proj.weight.data.zero_()
            self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))
            self.attn_gate = nn.Linear(12, num_heads, bias=False)
            self.rotary = Rotary(self.head_dim)
            self.xsa_alpha = (
                nn.Parameter(torch.zeros(num_heads)) if xsa_enabled else None
            )

        def forward(
            self,
            x: torch.Tensor,
            value_embed: torch.Tensor | None,
            *,
            window_size: int | None = None,
            partial_key_offset: bool = False,
            attn_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            B, T = x.size(0), x.size(1)
            q = self.q(x).view(B, T, self.num_heads, self.head_dim)
            k = self.k(x).view(B, T, self.num_heads, self.head_dim)
            v = self.v(x).view(B, T, self.num_heads, self.head_dim)
            if value_embed is None:
                v = self.lambdas[0] * v
            else:
                v = self.lambdas[0] * v + self.lambdas[1] * value_embed.view_as(v)
            q = F.rms_norm(q, (q.size(-1),))
            k = F.rms_norm(k, (k.size(-1),))
            q, k = self.rotary(q), self.rotary(k)
            if partial_key_offset:
                # Upstream (edf47a0): shift stationary K half +1 after RoPE.
                half = self.head_dim // 2
                stationary = torch.cat(
                    [k[:, :1, :, half:], k[:, :-1, :, half:]], dim=1
                )
                k = torch.cat([k[..., :half], stationary], dim=-1)
            q_t = q.transpose(1, 2)
            k_t = k.transpose(1, 2)
            v_t = v.transpose(1, 2)
            combined = _combine_attn_masks(
                T, x.device, window_size=window_size, attn_mask=attn_mask
            )
            if combined is None:
                y = F.scaled_dot_product_attention(
                    q_t, k_t, v_t, is_causal=True, scale=self.attn_scale
                ).transpose(1, 2)
            else:
                y = F.scaled_dot_product_attention(
                    q_t,
                    k_t,
                    v_t,
                    attn_mask=combined,
                    scale=self.attn_scale,
                ).transpose(1, 2)
            if self.xsa_alpha is not None:
                # Upstream XSA: subtract learned per-head proj of attn onto V.
                value_direction = F.normalize(v, dim=-1, eps=1e-4)
                projection = (y * value_direction).sum(-1, keepdim=True)
                alpha = torch.tanh(self.xsa_alpha).type_as(y).view(
                    1, 1, self.num_heads, 1
                )
                y = y - alpha * projection * value_direction
            gate = torch.sigmoid(self.attn_gate(x[..., :12])).view(
                B, T, self.num_heads, 1
            )
            y = (y * gate).contiguous().view(B, T, self.num_heads * self.head_dim)
            return self.proj(y)

    class PairedHeadAttention(nn.Module):
        """Paired-head self-attention: pairs of heads share Q/K projections."""

        def __init__(self, dim: int, num_heads: int, *, attn_scale: float = 0.12):
            super().__init__()
            if dim % num_heads != 0:
                raise ValueError(
                    f"dim ({dim}) must be divisible by num_heads ({num_heads})"
                )
            if num_heads % 2 != 0:
                raise ValueError(
                    f"PairedHeadAttention requires even num_heads, got {num_heads}"
                )
            self.num_heads = num_heads
            self.num_pairs = num_heads // 2
            self.head_dim = dim // num_heads
            self.attn_scale = float(attn_scale)
            pair_dim = self.num_pairs * self.head_dim
            self.q = nn.Linear(dim, pair_dim, bias=False)
            self.k = nn.Linear(dim, pair_dim, bias=False)
            self.v = nn.Linear(dim, dim, bias=False)
            self.proj = nn.Linear(dim, dim, bias=False)
            self.proj.weight.data.zero_()
            self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))
            self.attn_gate = nn.Linear(12, num_heads, bias=False)
            self.rotary = Rotary(self.head_dim)

        def forward(
            self,
            x: torch.Tensor,
            value_embed: torch.Tensor | None,
            *,
            window_size: int | None = None,
            partial_key_offset: bool = False,
            attn_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            B, T = x.size(0), x.size(1)
            q = self.q(x).view(B, T, self.num_pairs, self.head_dim)
            k = self.k(x).view(B, T, self.num_pairs, self.head_dim)
            v = self.v(x).view(B, T, self.num_heads, self.head_dim)
            if value_embed is None:
                v = self.lambdas[0] * v
            else:
                v = self.lambdas[0] * v + self.lambdas[1] * value_embed.view_as(v)
            q = F.rms_norm(q, (q.size(-1),))
            k = F.rms_norm(k, (k.size(-1),))
            # No stationary-K / XSA on paired-attn layers (head geometry).
            q, k = self.rotary(q), self.rotary(k)
            v_pair = v.view(B, T, self.num_pairs, 2, self.head_dim).reshape(
                B, T, self.num_pairs, 2 * self.head_dim
            )
            q_t = q.transpose(1, 2)
            k_t = k.transpose(1, 2)
            v_t = v_pair.transpose(1, 2)
            combined = _combine_attn_masks(
                T, x.device, window_size=window_size, attn_mask=attn_mask
            )
            if combined is None:
                y = F.scaled_dot_product_attention(
                    q_t, k_t, v_t, is_causal=True, scale=self.attn_scale
                ).transpose(1, 2)
            else:
                y = F.scaled_dot_product_attention(
                    q_t,
                    k_t,
                    v_t,
                    attn_mask=combined,
                    scale=self.attn_scale,
                ).transpose(1, 2)
            y = y.reshape(B, T, self.num_heads, self.head_dim)
            gate = torch.sigmoid(self.attn_gate(x[..., :12])).view(
                B, T, self.num_heads, 1
            )
            y = (y * gate).contiguous().view(B, T, self.num_heads * self.head_dim)
            return self.proj(y)

    class MLP(nn.Module):
        """ReLU² feed-forward network with a zero-init output projection."""

        def __init__(self, dim: int, mlp_ratio: int = 4):
            super().__init__()
            hidden = mlp_ratio * dim
            self.fc = nn.Linear(dim, hidden, bias=False)
            self.proj = nn.Linear(hidden, dim, bias=False)
            self.proj.weight.data.zero_()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.fc(x)
            x = F.relu(x).square()
            return self.proj(x)

    class Block(nn.Module):
        """Pre-norm transformer block (residual/post lambdas live on ``GPT``)."""

        def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: int = 4,
            *,
            attn_scale: float = 0.12,
            paired_head: bool = False,
            xsa_enabled: bool = False,
        ):
            super().__init__()
            attn_cls = PairedHeadAttention if paired_head else CausalSelfAttention
            attn_kwargs = {"attn_scale": attn_scale}
            if not paired_head:
                attn_kwargs["xsa_enabled"] = xsa_enabled
            self.attn = attn_cls(dim, num_heads, **attn_kwargs)
            self.mlp = MLP(dim, mlp_ratio)
            self.norm1 = RMSNorm(dim)
            self.norm2 = RMSNorm(dim)

        def forward(
            self,
            x: torch.Tensor,
            value_embed: torch.Tensor | None,
            *,
            window_size: int | None = None,
            partial_key_offset: bool = False,
            attn_mask: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            attn_out = self.attn(
                self.norm1(x),
                value_embed,
                window_size=window_size,
                partial_key_offset=partial_key_offset,
                attn_mask=attn_mask,
            )
            mlp_out = self.mlp(self.norm2(x))
            return attn_out, mlp_out

    class ValueEmbedding(nn.Module):
        """Sparse per-token value embeddings (SparsifyEmbeds design)."""

        def __init__(
            self, vocab_size: int, model_dim: int, num_layers: int, num_tables: int = 3
        ):
            super().__init__()
            self.num_layers = num_layers
            self.num_tables = max(1, min(num_tables, num_layers // 2))
            self.embed = nn.ModuleList(
                [nn.Embedding(vocab_size, model_dim) for _ in range(self.num_tables)]
            )
            for table in self.embed:
                nn.init.normal_(table.weight, mean=0.0, std=0.01)

        def forward(self, idx: torch.Tensor) -> list:
            tables = [emb(idx) for emb in self.embed]
            middle = self.num_layers - 2 * self.num_tables
            return tables + [None] * middle + tables

    class BigramHashEmbedding(nn.Module):
        """Bigram hash embedding on 1/4 of model_dim with sign trick."""

        def __init__(self, vocab_size: int, model_dim: int):
            super().__init__()
            if model_dim % 4 != 0:
                raise ValueError(
                    f"BigramHashEmbedding requires model_dim % 4 == 0, got {model_dim}"
                )
            self.full_dim = model_dim
            self.hash_dim = model_dim // 4
            self.token_embed = nn.Embedding(vocab_size, model_dim - self.hash_dim)
            rng = torch.Generator().manual_seed(42)
            hash_seed = torch.randint(
                0, 2**31, (vocab_size,), generator=rng, dtype=torch.long
            )
            self.register_buffer("hash_seed", hash_seed, persistent=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B, T = x.shape
            tok_part = self.token_embed(x)
            if T < 2:
                hash_part = torch.zeros(
                    B, T, self.hash_dim, device=x.device, dtype=tok_part.dtype
                )
                return torch.cat([tok_part, hash_part], dim=-1)
            prev = x[:, :-1]
            curr = x[:, 1:]
            h = (self.hash_seed[prev] * 2654435761) ^ (
                self.hash_seed[curr] * 2246822519
            )
            sign = (h.float() * (1.0 / 2**31)).fmod(2.0).abs().sub(1.0).sign()
            sign = sign.unsqueeze(-1).expand(-1, -1, self.hash_dim)
            pad_first = torch.zeros(
                B, 1, self.hash_dim, device=x.device, dtype=sign.dtype
            )
            hash_part = torch.cat([pad_first, sign], dim=1)
            return torch.cat([tok_part, hash_part], dim=-1)

    class Smear(nn.Module):
        """Learned 1-token lookback smear on the embedding stream."""

        def __init__(self, dim: int):
            super().__init__()
            self.smear_gate = nn.Parameter(torch.zeros(dim))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            prev = torch.cat([x[:, :1], x[:, :-1]], dim=1)
            gate = torch.sigmoid(self.smear_gate)
            return x + gate * prev

    class MUDD(nn.Module):
        """Multi-layer skip connections feeding residual stream and attention values."""

        def __init__(self, dim: int, num_skip_pairs: int = 2):
            super().__init__()
            self.num_skip_pairs = num_skip_pairs
            self.resid_gates = nn.Parameter(torch.zeros(num_skip_pairs))
            self.value_gates = nn.Parameter(torch.zeros(num_skip_pairs))
            self.resid_projs = nn.ModuleList(
                [nn.Linear(dim, dim, bias=False) for _ in range(num_skip_pairs)]
            )
            self.value_projs = nn.ModuleList(
                [nn.Linear(dim, dim, bias=False) for _ in range(num_skip_pairs)]
            )

        def forward(
            self, source: torch.Tensor, layer_idx: int
        ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
            if layer_idx >= self.num_skip_pairs:
                return None, None
            proj_dtype = self.resid_projs[layer_idx].weight.dtype
            src = source.to(dtype=proj_dtype)
            resid = torch.sigmoid(self.resid_gates[layer_idx]) * self.resid_projs[
                layer_idx
            ](src)
            val = torch.sigmoid(self.value_gates[layer_idx]) * self.value_projs[
                layer_idx
            ](src)
            return resid.type_as(source), val.type_as(source)

    class MultiTokenHeads(nn.Module):
        """Extra LM prediction heads for multi-token prediction (future tokens)."""

        def __init__(self, dim: int, vocab_size: int, num_extra_heads: int = 3):
            super().__init__()
            self.num_extra_heads = num_extra_heads
            self.heads = nn.ModuleList(
                [nn.Linear(dim, vocab_size, bias=False) for _ in range(num_extra_heads)]
            )
            for h in self.heads:
                h.weight.data.zero_()

        def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
            return [h(x) for h in self.heads]

    class GPT(nn.Module):
        """Proxy-student transformer read verbatim by ``trainer.py`` for the GPU sandbox."""

        def __init__(
            self,
            vocab_size: int,
            num_layers: int,
            model_dim: int,
            num_heads: int,
            mlp_ratio: int = 4,
            softcap: float = 30.0,
            num_value_embeds: int = 3,
            attn_scale: float = 0.12,
            sliding_window_size: int | None = None,
            bigram_hash_embed: bool = False,
            smear_embed: bool = False,
            partial_key_offset: bool = False,
            paired_head: bool = False,
            mudd_pairs: int = 0,
            xsa_enabled: bool = False,
            xsa_pairs: int = 0,
            single_act_last_k: int = 0,
            exp_residual_decay: float | None = None,
            multi_token_pred: int = 0,
        ):
            super().__init__()
            if num_layers < 2 or num_layers % 2 != 0:
                raise ValueError(f"num_layers must be even and >= 2, got {num_layers}")
            self.num_layers = num_layers
            self.softcap = float(softcap)
            self.sliding_window_size = sliding_window_size
            self.num_encoder_layers = num_layers // 2
            self.num_decoder_layers = num_layers - self.num_encoder_layers
            self.skip_gates = nn.Parameter(torch.ones(self.num_decoder_layers))
            self.post_lambdas = nn.Parameter(torch.ones(num_layers, 2))
            resid_init = math.sqrt(1.1)
            self.resid_lambdas_attn = nn.Parameter(
                torch.full((num_layers,), resid_init)
            )
            self.resid_lambdas_mlp = nn.Parameter(torch.full((num_layers,), resid_init))
            self.x0_lambdas = nn.Parameter(torch.zeros(num_layers))
            self.exp_residual_decay = exp_residual_decay

            if bigram_hash_embed:
                self.embed = BigramHashEmbedding(vocab_size, model_dim)
            else:
                self.embed = nn.Embedding(vocab_size, model_dim)

            if smear_embed:
                self.smear = Smear(model_dim)
            else:
                self.smear = None

            self.value_embeds = ValueEmbedding(
                vocab_size, model_dim, num_layers, num_value_embeds
            )

            self.partial_key_offset = partial_key_offset
            self.single_act_last_k = single_act_last_k
            self.multi_token_pred = multi_token_pred

            self.blocks = nn.ModuleList(
                [
                    Block(
                        model_dim,
                        num_heads,
                        mlp_ratio,
                        attn_scale=attn_scale,
                        # Upstream topology: paired {0,2,5,9}; XSA on other attn layers.
                        paired_head=paired_head and layer_idx in {0, 2, 5, 9},
                        xsa_enabled=(
                            xsa_enabled
                            and {1: 0, 3: 1, 4: 2, 7: 3, 8: 4, 10: 5}.get(
                                layer_idx, 6
                            )
                            < xsa_pairs
                        ),
                    )
                    for layer_idx in range(num_layers)
                ]
            )

            self.mudd_pairs = mudd_pairs
            if mudd_pairs > 0:
                self.mudd = MUDD(model_dim, num_skip_pairs=mudd_pairs)
            else:
                self.mudd = None

            self.xsa_enabled = xsa_enabled
            self.xsa_pairs = xsa_pairs

            if multi_token_pred > 0:
                self.multi_heads = MultiTokenHeads(
                    model_dim, vocab_size, multi_token_pred
                )
            else:
                self.multi_heads = None

            self.norm_in = RMSNorm(model_dim)
            self.norm_out = RMSNorm(model_dim)
            self.lm_head = nn.Linear(model_dim, vocab_size, bias=False)
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.005)

        def _apply_exp_residual_decay(
            self, layer_idx: int, x: torch.Tensor
        ) -> torch.Tensor:
            if self.exp_residual_decay is not None:
                alpha = self.exp_residual_decay**layer_idx
                return x * alpha
            return x

        def forward_hidden(
            self,
            idx: torch.Tensor,
            *,
            window_size: int | None = None,
            attn_mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            """Run the trunk through ``norm_out`` without materializing full-vocab logits."""
            ws = window_size if window_size is not None else self.sliding_window_size
            x = self.norm_in(self.embed(idx))
            if self.smear is not None:
                x = self.smear(x)
            trunk_dtype = self.blocks[0].attn.q.weight.dtype
            x = x.to(dtype=trunk_dtype)
            x0 = x
            ve = [
                v.to(dtype=trunk_dtype) if v is not None else None
                for v in self.value_embeds(idx)
            ]
            ve_enc, ve_dec = (
                ve[: self.num_encoder_layers],
                ve[self.num_encoder_layers :],
            )
            skip_connections: list[torch.Tensor] = []
            encoder_outputs: list[torch.Tensor] = []
            for i in range(self.num_encoder_layers):
                attn_out, mlp_out = self.blocks[i](
                    x,
                    ve_enc[i],
                    window_size=ws,
                    # Stationary-K only on the two long-window attn layers.
                    partial_key_offset=self.partial_key_offset and i in {3, 10},
                    attn_mask=attn_mask,
                )
                x = (
                    self._apply_exp_residual_decay(i, x)
                    if self.exp_residual_decay is not None
                    else x
                )
                x = (
                    self.resid_lambdas_attn[i] * x
                    + self.post_lambdas[i, 0] * attn_out
                    + self.x0_lambdas[i] * x0
                )
                x = self.resid_lambdas_mlp[i] * x + self.post_lambdas[i, 1] * mlp_out
                skip_connections.append(x)
                encoder_outputs.append(x)

            single_act = None
            if self.single_act_last_k > 0:
                s_start = max(0, self.num_encoder_layers - self.single_act_last_k)
                single_act = encoder_outputs[s_start] if encoder_outputs else x

            for i in range(self.num_decoder_layers):
                layer_idx = self.num_encoder_layers + i

                mudd_resid, mudd_val = None, None
                if self.mudd is not None:
                    src_idx = max(0, self.num_encoder_layers - 1 - i)
                    if src_idx < len(encoder_outputs):
                        mudd_resid, mudd_val = self.mudd(encoder_outputs[src_idx], i)

                x = x + torch.sigmoid(self.skip_gates[i]) * skip_connections.pop()

                attn_input = x
                if (
                    self.single_act_last_k > 0
                    and i >= self.num_decoder_layers - self.single_act_last_k
                ):
                    if single_act is not None:
                        attn_input = single_act

                ve_i = ve_dec[i]
                if mudd_val is not None:
                    if ve_i is None:
                        ve_i = mudd_val
                    else:
                        ve_i = ve_i + mudd_val

                attn_out, mlp_out = self.blocks[layer_idx](
                    attn_input,
                    ve_i,
                    window_size=ws,
                    partial_key_offset=(
                        self.partial_key_offset and layer_idx in {3, 10}
                    ),
                    attn_mask=attn_mask,
                )

                if mudd_resid is not None:
                    attn_out = attn_out + mudd_resid

                x = (
                    self._apply_exp_residual_decay(layer_idx, x)
                    if self.exp_residual_decay is not None
                    else x
                )
                x = (
                    self.resid_lambdas_attn[layer_idx] * x
                    + self.post_lambdas[layer_idx, 0] * attn_out
                    + self.x0_lambdas[layer_idx] * x0
                )
                x = (
                    self.resid_lambdas_mlp[layer_idx] * x
                    + self.post_lambdas[layer_idx, 1] * mlp_out
                )
            return self.norm_out(x)

        def apply_lm_head(self, hidden: torch.Tensor) -> torch.Tensor:
            """Project hidden states through ``lm_head`` + tanh softcap in float32."""
            logits = self.lm_head(hidden.float()).float()
            return self.softcap * torch.tanh(logits / self.softcap)

        def forward(
            self,
            idx: torch.Tensor,
            *,
            window_size: int | None = None,
            attn_mask: torch.Tensor | None = None,
            output_hidden: bool = False,
        ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
            hidden = self.forward_hidden(
                idx, window_size=window_size, attn_mask=attn_mask
            )
            out = self.apply_lm_head(hidden)
            if output_hidden:
                return out, hidden
            return out

    @dataclass(frozen=True)
    class StudentModelConfig:
        """A concrete proxy-student model configuration (dims + softcap)."""

        model_dim: int = 768
        num_layers: int = 12
        num_heads: int = 6
        mlp_ratio: int = 4
        vocab_size: int = 50304
        softcap: float = 30.0
        num_value_embeds: int = 3
        attn_scale: float = 0.12
        sliding_window_size: int | None = None
        bigram_hash_embed: bool = False
        smear_embed: bool = False
        partial_key_offset: bool = False
        paired_head: bool = False
        mudd_pairs: int = 0
        xsa_enabled: bool = False
        xsa_pairs: int = 0
        single_act_last_k: int = 0
        exp_residual_decay: float | None = None
        multi_token_pred: int = 0

        def build(self) -> GPT:
            return GPT(
                vocab_size=self.vocab_size,
                num_layers=self.num_layers,
                model_dim=self.model_dim,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                softcap=self.softcap,
                num_value_embeds=self.num_value_embeds,
                attn_scale=self.attn_scale,
                sliding_window_size=self.sliding_window_size,
                bigram_hash_embed=self.bigram_hash_embed,
                smear_embed=self.smear_embed,
                partial_key_offset=self.partial_key_offset,
                paired_head=self.paired_head,
                mudd_pairs=self.mudd_pairs,
                xsa_enabled=self.xsa_enabled,
                xsa_pairs=self.xsa_pairs,
                single_act_last_k=self.single_act_last_k,
                exp_residual_decay=self.exp_residual_decay,
                multi_token_pred=self.multi_token_pred,
            )

    GPT2_SMALL = StudentModelConfig()
    GPT2_SMALL_PARAM_COUNT = 278_122_938

    def estimate_instantiated_param_count(
        *,
        vocab_size: int = 50304,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        mlp_ratio: int = 4,
        softcap: float = 30.0,
        num_value_embeds: int = 3,
        attn_scale: float = 0.12,
        sliding_window_size: int | None = None,
        bigram_hash_embed: bool = False,
        smear_embed: bool = False,
        partial_key_offset: bool = False,
        paired_head: bool = False,
        mudd_pairs: int = 0,
        xsa_enabled: bool = False,
        xsa_pairs: int = 0,
        single_act_last_k: int = 0,
        exp_residual_decay: float | None = None,
        multi_token_pred: int = 0,
    ) -> int:
        """Return the exact parameter count ``GPT.build`` would instantiate."""
        with torch.device("meta"):
            model = GPT(
                vocab_size=vocab_size,
                num_layers=num_layers,
                model_dim=model_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                softcap=softcap,
                num_value_embeds=num_value_embeds,
                attn_scale=attn_scale,
                sliding_window_size=sliding_window_size,
                bigram_hash_embed=bigram_hash_embed,
                smear_embed=smear_embed,
                partial_key_offset=partial_key_offset,
                paired_head=paired_head,
                mudd_pairs=mudd_pairs,
                xsa_enabled=xsa_enabled,
                xsa_pairs=xsa_pairs,
                single_act_last_k=single_act_last_k,
                exp_residual_decay=exp_residual_decay,
                multi_token_pred=multi_token_pred,
            )
            return sum(p.numel() for p in model.parameters())

    def zeropower_via_newtonschulz5(g: torch.Tensor) -> torch.Tensor:
        """Orthogonalize a 2-D update via Newton–Schulz (Muon core step)."""
        assert g.ndim >= 2
        x = g.to(dtype=torch.float32)
        if g.size(-2) > g.size(-1):
            x = x.mT
        x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
        a, b, c = 2.0, -1.5, 0.5
        for _ in range(12):
            a_mat = x @ x.mT
            b_mat = b * a_mat + c * a_mat @ a_mat
            x = a * x + b_mat @ x
        if g.size(-2) > g.size(-1):
            x = x.mT
        return x.to(dtype=g.dtype)

    def zeropower_via_polar_express(g: torch.Tensor) -> torch.Tensor:
        """Polar decomposition via ONI iteration variant (Polar Express)."""
        assert g.ndim >= 2
        x = g.to(dtype=torch.float32)
        if g.size(-2) > g.size(-1):
            x = x.mT
        x = x / (x.norm(dim=(-2, -1), keepdim=True) + 1e-7)
        for _ in range(6):
            xtx = x.mT @ x
            eye = torch.eye(xtx.size(-1), device=xtx.device, dtype=xtx.dtype)
            x = 0.5 * x @ (3.0 * eye - xtx)
        if g.size(-2) > g.size(-1):
            x = x.mT
        return x.to(dtype=g.dtype)

    def muon_update(
        grad: torch.Tensor,
        momentum: torch.Tensor,
        *,
        mu: float = 0.95,
        nesterov: bool = True,
        polar_express: bool = False,
    ) -> torch.Tensor:
        """One Muon momentum + orthogonalization step (single tensor, no torch.compile)."""
        grad_fp32 = grad.float()
        momentum.lerp_(grad_fp32, 1.0 - mu)
        update = grad_fp32.lerp(momentum, mu) if nesterov else momentum
        if polar_express:
            update = zeropower_via_polar_express(update)
        else:
            update = zeropower_via_newtonschulz5(update)
        if grad.ndim >= 2:
            update = update * max(1.0, grad.size(-2) / grad.size(-1)) ** 0.5
        return update

    def muon_update_normalized(
        grad: torch.Tensor,
        momentum: torch.Tensor,
        *,
        mu: float = 0.95,
        nesterov: bool = True,
        polar_express: bool = False,
    ) -> torch.Tensor:
        """NorMuon: normalized Muon update (RMS-normalize the update before applying)."""
        update = muon_update(
            grad, momentum, mu=mu, nesterov=nesterov, polar_express=polar_express
        )
        rms = update.norm() / (update.numel() ** 0.5)
        return update / (rms + 1e-8)

    class Muon(torch.optim.Optimizer):
        """Single-device Muon optimizer (no distributed all_gather)."""

        def __init__(
            self,
            params,
            *,
            lr: float = 0.02,
            weight_decay: float = 0.0,
            momentum: float = 0.95,
            nor_muon: bool = False,
            polar_express: bool = False,
        ):
            params = [p for p in params if p.requires_grad]
            if not params:
                raise ValueError("Muon requires at least one trainable parameter")
            defaults = dict(
                lr=lr,
                weight_decay=weight_decay,
                momentum=momentum,
                nor_muon=nor_muon,
                polar_express=polar_express,
            )
            super().__init__(params, defaults)

        @torch.no_grad()
        def step(
            self,
            *,
            momentum: float | None = None,
            cautious_wd: bool = False,
            lr_scale: float = 1.0,
        ):
            for group in self.param_groups:
                mu = group["momentum"] if momentum is None else momentum
                lr = group["lr"]
                wd = group["weight_decay"]
                nor_muon = group.get("nor_muon", False)
                polar_express = group.get("polar_express", False)
                for param in group["params"]:
                    if param.grad is None:
                        continue
                    state = self.state[param]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros(
                            param.shape, dtype=torch.float32, device=param.device
                        )
                    if nor_muon:
                        update = muon_update_normalized(
                            param.grad,
                            state["momentum_buffer"],
                            mu=mu,
                            polar_express=polar_express,
                        )
                    else:
                        update = muon_update(
                            param.grad,
                            state["momentum_buffer"],
                            mu=mu,
                            polar_express=polar_express,
                        )
                    effective_wd = wd
                    if cautious_wd:
                        effective_wd = wd * lr_scale
                    if effective_wd:
                        param.mul_(1.0 - lr * effective_wd)
                    param.add_(update.to(dtype=param.dtype), alpha=-lr)

    @dataclass(frozen=True)
    class BatchScheduleStage:
        """One segment of the speedrun batch/LR schedule."""

        batch_mul: int
        lr_mul: float

    def build_batch_schedule(
        total_steps: int,
        *,
        stage_fracs: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
        batch_muls: tuple[int, int, int] = (1, 2, 3),
        lr_muls: tuple[float, float, float] = (1.0, 1.52, 1.73),
        cooldown_frac: float = 0.40,
        cooldown_floor: float = 0.15,
    ) -> tuple[list[tuple[int, int]], BatchScheduleStage, float, float]:
        """Return stage boundaries, stage lookup metadata, cooldown start, and floor."""
        if len(stage_fracs) != len(batch_muls) or len(stage_fracs) != len(lr_muls):
            raise ValueError(
                "stage_fracs, batch_muls, and lr_muls must have equal length"
            )
        total_steps = max(1, int(total_steps))
        boundaries = batch_stage_boundaries(total_steps, stage_fracs)
        stages = [
            BatchScheduleStage(batch_mul=m, lr_mul=lr)
            for m, lr in zip(batch_muls, lr_muls, strict=True)
        ]
        cd_start = int(total_steps * (1.0 - cooldown_frac))
        return boundaries, stages, cd_start, cooldown_floor

    def lookup_batch_stage(
        step: int,
        boundaries: list[tuple[int, int]],
        stages: list[BatchScheduleStage],
    ) -> BatchScheduleStage:
        for (start, end), stage in zip(boundaries, stages, strict=True):
            if start <= step < end:
                return stage
        return stages[-1]

    def schedule_lr_multiplier(
        step: int,
        stage: BatchScheduleStage,
        *,
        cd_start: int,
        scheduled_steps: int,
        cooldown_floor: float,
    ) -> float:
        """Stage LR multiplier with linear cooldown (``train_gpt.py`` ``get_lr``)."""
        lr = stage.lr_mul
        if step >= cd_start and scheduled_steps > cd_start:
            t = min(1.0, (step - cd_start) / (scheduled_steps - cd_start))
            lr = lr * (1.0 - t) + cooldown_floor * t
        return lr

    def get_muon_momentum(
        step: int,
        total_steps: int,
        *,
        warmup_steps: int,
        cooldown_steps: int,
        momentum_min: float = 0.85,
        momentum_max: float = 0.95,
    ) -> float:
        """Linear Muon momentum warmup then cooldown (``train_gpt.py``)."""
        momentum_cd_start = max(0, total_steps - cooldown_steps)
        if warmup_steps > 0 and step < warmup_steps:
            frac = step / warmup_steps
            return momentum_min + frac * (momentum_max - momentum_min)
        if cooldown_steps > 0 and step > momentum_cd_start:
            frac = (step - momentum_cd_start) / cooldown_steps
            return momentum_max - frac * (momentum_max - momentum_min)
        return momentum_max

    def classify_speedrun_params(
        model,
    ) -> tuple[list[torch.nn.Parameter], dict[str, list[torch.nn.Parameter]]]:
        """Split ``GPT`` params into Muon matrices vs AdamW groups."""
        muon_params: list[torch.nn.Parameter] = []
        adam: dict[str, list[torch.nn.Parameter]] = {
            "embed": [],
            "lm_head": [],
            "value_embeds": [],
            "scalars": [],
        }
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim >= 2 and name.startswith("blocks."):
                muon_params.append(param)
            elif name.startswith("embed."):
                adam["embed"].append(param)
            elif name.startswith("lm_head."):
                if not any(p is param for p in adam["embed"]):
                    adam["lm_head"].append(param)
            elif name.startswith("value_embeds."):
                adam["value_embeds"].append(param)
            elif name.startswith("multi_heads."):
                adam["lm_head"].append(param)
            else:
                adam["scalars"].append(param)
        return muon_params, adam

    def build_speedrun_optimizers(
        model,
        *,
        muon_lr: float = 0.023,
        muon_weight_decay: float = 1.2,
        adam_lr: float = 0.008,
        adam_eps: float = 1e-10,
        adam_weight_decay: float = 0.005,
        embed_lr_mul: float = 1.0,
        lm_head_lr_mul: float = 1.0,
        value_embed_lr_mul: float = 75.0,
        scalar_lr_mul: float = 5.0,
        embed_wd_mul: float = 150.0,
        lm_head_wd_mul: float = 150.0,
        value_embed_wd_mul: float = 5.0,
        scalar_wd_mul: float = 0.0,
        nor_muon: bool = True,
        polar_express: bool = False,
    ):
        """Build Muon + multi-group AdamW optimizers for ``GPT``."""
        muon_params, adam = classify_speedrun_params(model)
        muon_opt = Muon(
            muon_params,
            lr=muon_lr,
            weight_decay=muon_weight_decay,
            nor_muon=nor_muon,
            polar_express=polar_express,
        )
        adam_groups = []
        if adam["embed"]:
            adam_groups.append(
                {
                    "params": adam["embed"],
                    "lr": adam_lr * embed_lr_mul,
                    "weight_decay": adam_weight_decay * embed_wd_mul,
                    "betas": (0.5, 0.95),
                }
            )
        if adam["lm_head"]:
            adam_groups.append(
                {
                    "params": adam["lm_head"],
                    "lr": adam_lr * lm_head_lr_mul,
                    "weight_decay": adam_weight_decay * lm_head_wd_mul,
                    "betas": (0.5, 0.95),
                }
            )
        if adam["value_embeds"]:
            adam_groups.append(
                {
                    "params": adam["value_embeds"],
                    "lr": adam_lr * value_embed_lr_mul,
                    "weight_decay": adam_weight_decay * value_embed_wd_mul,
                    "betas": (0.75, 0.95),
                }
            )
        if adam["scalars"]:
            adam_groups.append(
                {
                    "params": adam["scalars"],
                    "lr": adam_lr * scalar_lr_mul,
                    "weight_decay": adam_weight_decay * scalar_wd_mul,
                    "betas": (0.9, 0.99),
                }
            )
        adam_opt = torch.optim.AdamW(
            adam_groups,
            eps=adam_eps,
        )
        return muon_opt, adam_opt

    def init_speedrun_weights(model) -> None:
        """Weight init aligned with modded-nanogpt block matrices (``train_gpt_simple``)."""
        for name, param in model.named_parameters():
            data = param.data
            if not name.endswith("weight"):
                continue
            if "proj" in name:
                data.zero_()
            elif "lm_head" in name:
                data.normal_(std=0.005)
            elif "value_embeds" in name:
                data.normal_(std=0.01)
            elif "embed" in name:
                if data.numel() > 0:
                    data.normal_(std=0.02)
            elif data.ndim >= 2:
                std = (0.33**0.5) / data.size(-1) ** 0.5
                data.normal_(std=std)

    def set_optimizer_lrs(
        muon_opt: Muon,
        adam_opt: torch.optim.AdamW,
        *,
        lr_scale: float,
        initial_muon_lr: float,
        initial_adam_lrs: list[float],
    ) -> None:
        for group in muon_opt.param_groups:
            group["lr"] = initial_muon_lr * lr_scale
        for group, base_lr in zip(
            adam_opt.param_groups, initial_adam_lrs, strict=False
        ):
            group["lr"] = base_lr * lr_scale

    def capture_initial_lrs(
        muon_opt: Muon, adam_opt: torch.optim.AdamW
    ) -> tuple[float, list[float]]:
        muon_lr = muon_opt.param_groups[0]["lr"]
        adam_lrs = [g["lr"] for g in adam_opt.param_groups]
        return muon_lr, adam_lrs

    def clip_optimizer_grads(optimizer: torch.optim.Optimizer, max_norm: float) -> None:
        """Clip only gradients owned by one optimizer at its update boundary."""
        if not max_norm or max_norm <= 0:
            return
        params = [
            param
            for group in optimizer.param_groups
            for param in group["params"]
            if param.grad is not None
        ]
        if params:
            torch.nn.utils.clip_grad_norm_(params, max_norm)

    def step_speedrun_optimizers(
        muon_opt: Muon,
        adam_opt: torch.optim.AdamW,
        *,
        step: int,
        muon_momentum: float,
        adam_on_odd_steps: bool = True,
        cautious_wd: bool = False,
        lr_scale: float = 1.0,
        force_adam: bool = False,
        grad_clip: float = 0.0,
    ) -> bool:
        """Muon every step; AdamW only on odd steps (heterogeneous batching record)."""
        clip_optimizer_grads(muon_opt, grad_clip)
        muon_opt.step(
            momentum=muon_momentum, cautious_wd=cautious_wd, lr_scale=lr_scale
        )
        did_adam_step = force_adam or not adam_on_odd_steps or step % 2 == 1
        if did_adam_step:
            clip_optimizer_grads(adam_opt, grad_clip)
            adam_opt.step()
        return did_adam_step

    def _is_cuda_device(device) -> bool:
        if isinstance(device, torch.device):
            return device.type == "cuda"
        return isinstance(device, str) and device.startswith("cuda")

    def prepare_student_model_dtype(model, device):
        """CUDA-only bfloat16 for Muon matrices; Adam groups stay float32."""
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
        """Build the official FineWeb document token stream and exact ranges."""
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
        """Plan training-window starts over an EOT-prefixed document stream."""
        block = int(block)
        n_tokens = int(n_tokens)
        lookahead = max(1, int(lookahead))
        need = block + lookahead
        ranges = []
        previous_end = 0
        for raw_start, raw_end in document_ranges or ():
            start, end = int(raw_start), int(raw_end)
            if start < previous_end or end < start or end > n_tokens:
                raise ValueError(
                    "document ranges must be sorted, disjoint, and in bounds"
                )
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
        ranges = document_ranges or ()
        if not ranges:
            return ids
        doc_id = bisect.bisect_right(ranges, start, key=lambda r: r[0]) - 1
        if doc_id < 0:
            doc_id = 0
        for i in range(doc_id, len(ranges)):
            doc_start, doc_end = int(ranges[i][0]), int(ranges[i][1])
            if doc_start >= end:
                break
            lo = max(doc_start, start)
            hi = min(doc_end, end)
            if lo < hi:
                ids[lo - start : hi - start] = i
        return ids

    def build_document_attn_mask(doc_ids):
        """Boolean SDPA mask ``(B, T, T)``; ``True`` means the key may participate."""
        if doc_ids.dim() != 2:
            raise ValueError(
                f"doc_ids must be (batch, seq), got shape {tuple(doc_ids.shape)}"
            )
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
            doc_ids = window_document_ids(start, block, document_ranges, device=device)
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

    def _score_hidden_chunked(
        model, hidden, targets, *, vocab_size, logit_chunk_tokens
    ):
        """Sum CE + correct counts over ``hidden`` without a full (N, vocab) allocation."""
        flat_h = hidden.reshape(-1, hidden.size(-1))
        flat_y = targets.reshape(-1)
        n = int(flat_h.size(0))
        chunk = int(logit_chunk_tokens) if logit_chunk_tokens is not None else n
        if chunk < 1:
            raise ValueError(
                f"logit_chunk_tokens must be >= 1, got {logit_chunk_tokens}"
            )
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
        """Mean held-out CE / accuracy over every predictable val target."""
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
        """Total loss including auxiliary multi-token prediction losses."""
        main_loss = F.cross_entropy(logits.float().view(-1, vocab_size), y.view(-1))
        if (
            multi_heads is None
            or not hasattr(multi_heads, "heads")
            or not multi_heads.heads
        ):
            return main_loss
        mt_loss = 0.0
        mt_weight = 0.3
        hidden_f = hidden.float()
        for k, head in enumerate(multi_heads.heads):
            if k < len(y_future) and y_future[k] is not None:
                head_logits = head(hidden_f)
                mt_loss = mt_loss + F.cross_entropy(
                    head_logits.view(-1, vocab_size), y_future[k].view(-1)
                )
        if mt_loss > 0.0:
            return main_loss + mt_weight * mt_loss
        return main_loss

    def _microbatch_ranges(batch_size, microbatch_size):
        """Yield ``(start, end)`` slices that cover ``batch_size`` without exceeding ``microbatch_size``."""
        n = int(batch_size)
        if n < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if microbatch_size is None:
            yield 0, n
            return
        micro = int(microbatch_size)
        if micro < 1:
            raise ValueError(
                f"train_microbatch_size must be >= 1, got {microbatch_size}"
            )
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
        lr_cooldown_frac=0.40,
        lr_cooldown_floor=0.15,
        muon_momentum_min=0.85,
        muon_momentum_max=0.95,
        muon_warmup_steps=None,
        muon_cooldown_steps=None,
        adam_on_odd_steps=True,
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
            train_src = train_src.repeat(
                math.ceil((block + 2) / max(len(train_src), 1))
            )

        # Never ignore document_ranges for packing.
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
            range(steps),
            total=steps,
            desc=desc,
            unit="step",
            leave=False,
            file=sys.stdout,
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
                loss_weighted = None
                for start, end in _microbatch_ranges(batch, train_microbatch_size):
                    xb = x[start:end]
                    yb = y[start:end]
                    mb_mask = None if attn_mask is None else attn_mask[start:end]
                    logits = model(xb, attn_mask=mb_mask)
                    loss = F.cross_entropy(
                        logits.float().view(-1, vocab_size), yb.view(-1)
                    )
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
                            ema_decay * ema_loss_tensor
                            + (1.0 - ema_decay) * loss_detached
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
                    pbar.set_postfix(
                        loss=f"{ema_loss:.4f}", tok_s=f"{tokens_per_sec:.0f}"
                    )
                    pbar.write(
                        f"[{desc}] step {completed}/{steps} | loss {ema_loss:.4f} "
                        f"| {tokens_per_sec:.0f} tok/s | elapsed {elapsed:.1f}s | eta {eta_seconds:.1f}s"
                    )
        else:
            init_speedrun_weights(model)

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
                x, y, batch_starts, attn_mask = next_batch(
                    effective_batch, current_block
                )

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
                                    train_src[
                                        sidx + shift : sidx + shift + current_block
                                    ]
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
                                    seg = torch.zeros(
                                        current_block, dtype=train_src.dtype
                                    )
                                future_targets.append(seg)
                        y_future.append(torch.stack(future_targets).to(device))

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
                    _scaled_microbatch_loss(
                        loss, end - start, effective_batch
                    ).backward()
                    with torch.no_grad():
                        piece = loss.detach() * float(end - start)
                        loss_weighted = (
                            piece if loss_weighted is None else loss_weighted + piece
                        )

                if grad_accum_embed_head_steps > 1:
                    embed_head_params = []
                    for name, p in model.named_parameters():
                        if (
                            name.startswith("embed.")
                            or name.startswith("lm_head.")
                            or name.startswith("multi_heads.")
                        ):
                            embed_head_params.append(p)

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
                        # Clear grads between micro-steps.
                        muon_opt.zero_grad(set_to_none=True)
                        adam_opt.zero_grad(set_to_none=True)
                else:
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

                if untie_at_frac > 0.0 and step == int(steps * untie_at_frac):
                    if (
                        hasattr(model.embed, "weight")
                        and model.lm_head.weight.data_ptr()
                        == model.embed.weight.data_ptr()
                    ):
                        model.lm_head.weight = nn.Parameter(
                            model.lm_head.weight.data.clone()
                        )
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
                            ema_decay * ema_loss_tensor
                            + (1.0 - ema_decay) * loss_detached
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
                    pbar.set_postfix(
                        loss=f"{ema_loss:.4f}", tok_s=f"{tokens_per_sec:.0f}"
                    )
                    pbar.write(
                        f"[{desc}] step {completed}/{steps} | loss {ema_loss:.4f} "
                        f"| bs={effective_batch} lr_scale={lr_scale:.3f} "
                        f"| {tokens_per_sec:.0f} tok/s | elapsed {elapsed:.1f}s | eta {eta_seconds:.1f}s"
                    )

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
                        p.grad = (
                            accum_buffer_embed_head[id(p)].clone().to(device=p.device)
                        )
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

    def model_kwargs_from_config(
        cfg: Mapping[str, Any], vocab_size: int
    ) -> dict[str, Any]:
        """Translate the serialized proxy-student config into ``GPT`` kwargs."""
        return {
            "vocab_size": vocab_size,
            "num_layers": int(cfg["n_layer"]),
            "model_dim": int(cfg["n_embd"]),
            "num_heads": int(cfg["n_head"]),
            "mlp_ratio": int(cfg["mlp_ratio"]),
            "softcap": float(cfg["lm_head_softcap"]),
            "num_value_embeds": int(cfg["num_value_embeds"]),
            "attn_scale": float(cfg.get("attn_scale", 0.12)),
            "sliding_window_size": cfg.get("sliding_window_size"),
            "bigram_hash_embed": bool(cfg.get("bigram_hash_embed", False)),
            "smear_embed": bool(cfg.get("smear_embed", False)),
            "partial_key_offset": cfg.get("partial_key_offset"),
            "paired_head": bool(cfg.get("paired_head", False)),
            "mudd_pairs": int(cfg.get("mudd_pairs", 0)),
            "xsa_enabled": bool(cfg.get("xsa_enabled", False)),
            "xsa_pairs": int(cfg.get("xsa_pairs", 0)),
            "single_act_last_k": int(cfg.get("single_act_last_k", 0)),
            "exp_residual_decay": cfg.get("exp_residual_decay"),
            "multi_token_pred": int(cfg.get("multi_token_pred", 0)),
        }

    def training_kwargs_from_config(
        cfg: Mapping[str, Any], vocab_size: int
    ) -> dict[str, Any]:
        """Translate serialized config into ``averaged_train_and_eval`` kwargs."""
        return {
            "block_size": int(cfg["block_size"]),
            "batch_size": int(cfg["batch_size"]),
            "steps": int(cfg["steps"]),
            "vocab_size": vocab_size,
            "training_recipe": str(cfg.get("training_recipe", "speedrun_muon")),
            "base_lr": float(cfg.get("learning_rate", 3e-4)),
            "warmup_steps": int(cfg.get("warmup_steps", 0)),
            "weight_decay": float(cfg.get("weight_decay", 0.1)),
            "grad_clip": float(cfg.get("grad_clip", 0.0)),
            "beta1": float(cfg.get("adam_beta1", 0.9)),
            "beta2": float(cfg.get("adam_beta2", 0.95)),
            "eps": float(cfg.get("record_adam_eps", cfg.get("adam_eps", 1e-8))),
            "lr_min_ratio": float(cfg.get("lr_min_ratio", 0.1)),
            "muon_lr": float(cfg.get("muon_lr", 0.023)),
            "muon_weight_decay": float(cfg.get("muon_weight_decay", 1.2)),
            "adam_lr": float(cfg.get("adam_lr", 0.008)),
            "adam_eps": float(cfg.get("adam_eps", 1e-10)),
            "adam_weight_decay": float(cfg.get("adam_weight_decay", 0.005)),
            "embed_lr_mul": float(cfg.get("embed_lr_mul", 1.0)),
            "lm_head_lr_mul": float(cfg.get("lm_head_lr_mul", 1.0)),
            "value_embed_lr_mul": float(cfg.get("value_embed_lr_mul", 75.0)),
            "scalar_lr_mul": float(cfg.get("scalar_lr_mul", 5.0)),
            "embed_wd_mul": float(cfg.get("embed_wd_mul", 150.0)),
            "lm_head_wd_mul": float(cfg.get("lm_head_wd_mul", 150.0)),
            "value_embed_wd_mul": float(cfg.get("value_embed_wd_mul", 5.0)),
            "scalar_wd_mul": float(cfg.get("scalar_wd_mul", 0.0)),
            "batch_schedule_enabled": bool(cfg.get("batch_schedule_enabled", True)),
            "batch_stage_fracs": tuple(cfg.get("batch_stage_fracs", (1 / 3,) * 3)),
            "batch_stage_muls": tuple(cfg.get("batch_stage_muls", (1, 2, 3))),
            "lr_stage_muls": tuple(cfg.get("lr_stage_muls", (1.0, 1.52, 1.73))),
            "lr_cooldown_frac": float(cfg.get("lr_cooldown_frac", 0.40)),
            "lr_cooldown_floor": float(cfg.get("lr_cooldown_floor", 0.15)),
            "muon_momentum_min": float(cfg.get("muon_momentum_min", 0.85)),
            "muon_momentum_max": float(cfg.get("muon_momentum_max", 0.95)),
            "muon_warmup_steps": cfg.get("muon_warmup_steps"),
            "muon_cooldown_steps": cfg.get("muon_cooldown_steps"),
            "adam_on_odd_steps": bool(cfg.get("adam_on_odd_steps", True)),
            "grad_accum_embed_head_steps": int(
                cfg.get("grad_accum_embed_head_steps", 1)
            ),
            "seq_len_schedule": bool(cfg.get("seq_len_schedule", False)),
            "multi_token_pred": int(cfg.get("multi_token_pred", 0)),
            "untie_at_frac": float(cfg.get("untie_at_frac", 0.0)),
            "cautious_wd": bool(cfg.get("cautious_wd", False)),
            "nor_muon": bool(cfg.get("nor_muon", True)),
            "polar_express": bool(cfg.get("polar_express", False)),
            "train_microbatch_size": cfg.get("train_microbatch_size"),
            "val_batch_size": cfg.get("val_batch_size"),
            "val_logit_chunk_tokens": cfg.get("val_logit_chunk_tokens"),
        }

    def run_training(workdir: str = TRAIN_WORKDIR) -> dict[str, Any]:
        """Run one configured training job and persist its result."""
        stderr_path = os.path.join(workdir, "stderr.txt")
        stderr_fh = open(stderr_path, "w", buffering=1)
        sys.stderr = stderr_fh
        atexit.register(stderr_fh.flush)

        torch.set_float32_matmul_precision("high")
        with open(os.path.join(workdir, "config.json")) as f:
            cfg = json.load(f)
        with open(os.path.join(workdir, "corpus.txt"), encoding="utf-8") as f:
            corpus_text = f.read()

        documents, explicit_val_documents = parse_document_payload(corpus_text)

        seed = int(cfg["seed"])
        torch.manual_seed(seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        encoder = tiktoken.get_encoding(str(cfg.get("tokenizer", "gpt2")))
        # Vocab pad to multiple of 128 (50257 -> 50304).
        vocab_size = ((encoder.n_vocab + 127) // 128) * 128

        eos_aligned_batches = bool(cfg.get("eos_aligned_batches", True))
        if documents is None:
            if eos_aligned_batches:
                raise ValueError(
                    "EOS-aligned training requires a document-list-v1 corpus payload; "
                    "flat text cannot recover source document boundaries safely"
                )
            documents = [corpus_text]
        corpus_ids, encoded_document_ranges = encode_document_tokens(
            documents,
            encoder,
            cfg.get("max_document_tokens", cfg.get("max_doc_len")),
        )
        document_ranges = encoded_document_ranges if eos_aligned_batches else None
        corpus = torch.tensor(corpus_ids, dtype=torch.long)

        val_path = os.path.join(workdir, "val.bin")
        if explicit_val_documents is not None:
            val_ids, _ = encode_document_tokens(
                explicit_val_documents,
                encoder,
                cfg.get("max_document_tokens", cfg.get("max_doc_len")),
            )
            train_data = corpus
            val_data = torch.tensor(val_ids, dtype=torch.long)
            val_source = "stratified_corpus_split"
        elif os.path.exists(val_path):
            val_ids = np.fromfile(val_path, dtype="<u2").astype(np.int64)
            train_data = corpus
            val_data = torch.from_numpy(val_ids)
            val_source = "held_out"
        else:
            n_val = max(1, int(len(corpus) * float(cfg["val_fraction"])))
            train_data, val_data = corpus[:-n_val], corpus[-n_val:]
            if document_ranges is not None:
                document_ranges = [
                    bounds for bounds in document_ranges if bounds[1] <= len(train_data)
                ]
            val_source = "corpus_split"

        def build_model():
            return GPT(**model_kwargs_from_config(cfg, vocab_size)).to(device)

        train_kwargs = training_kwargs_from_config(cfg, vocab_size)
        train_kwargs["document_ranges"] = document_ranges
        val_loss, acc, flops, tokens_trained, n_params = averaged_train_and_eval(
            build_model,
            train_data,
            val_data,
            n_runs=int(cfg.get("n_train_runs", 1)),
            base_seed=seed,
            device=device,
            **train_kwargs,
        )
        result = {
            "loss": val_loss,
            "accuracy": acc,
            "flops": flops,
            "tokens_trained": tokens_trained,
            "n_params": n_params,
            "vocab_size": vocab_size,
            "val_tokens": int(len(val_data)),
            "val_scored_targets": int(len(val_data) - 1),
            "val_source": val_source,
            "n_train_runs": int(cfg.get("n_train_runs", 1)),
        }
        print("RESULT_JSON " + json.dumps(result), flush=True)
        Path(workdir, "result.json").write_text(json.dumps(result))
        return result


if __name__ == "__main__":
    if torch is None:
        raise RuntimeError("train_gpt.py requires PyTorch")
    run_training()
