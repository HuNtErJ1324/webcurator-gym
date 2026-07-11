"""Modern proxy-student architecture (modded-nanogpt), pure CPU-runnable PyTorch.

This module is the **single source of truth** for the proxy-student model. The same
class definitions are embedded byte-identically into the GPU sandbox training
script (``trainer.py``) via :func:`model_source`, so the CPU unit tests in this
package exercise the exact code the sandbox runs.

The architecture follows ``KellerJordan/modded-nanogpt`` through the SparsifyEmbeds /
UNet / post-lambda / gated-attention records, while deliberately EXCLUDING GPU-only
pieces (FlexAttention kernels, FlashAttention varlen, FP8, distributed comms,
``torch.compile``/triton, and YaRN runtime extension). Training uses the
CPU-portable speedrun optimizer in ``student_optimizer.py``.

Components implemented (all CPU-runnable):

* **RoPE** — half-truncate rotary embeddings (``Rotary``).
* **RMSNorm** with learnable gains.
* **QK-norm** before attention.
* **ReLU² MLP** feed-forward.
* **SDPA attention** with ``softmax_scale=attn_scale`` (default **0.12**, matching
  the speedrun records) and optional **sliding-window** masking (SDPA ``attn_mask``,
  not FlexAttention).
* **Per-layer gated attention** — ``sigmoid(Linear(x[..., :12])`` per head.
* **Per-layer residual/post lambdas** — ``resid_lambdas_{attn,mlp}`` and
  ``post_lambdas`` matching the modern speedrun residual path.
* **x0 lambdas** — per-layer injection of the post-embed stream.
* **tanh logit softcap** (default 30).
* **U-net skips** with **sigmoid skip gates** (replacing raw skip weights).
* **Sparse value embeddings** (SparsifyEmbeds): 3 distinct tables on first/last bands.
* **Untied** ``lm_head`` with normal init (std=0.005); vocab padded to 50304.
* **Bigram hash embedding** — 1/4 of ``model_dim`` via hashed bigrams with sign trick.
* **Smear** — learned 1-token lookback on the embedding stream.
* **Partial Key Offset** — offset key RoPE positions for context-length extrapolation.
* **Paired head attention** — shared Q/K across head pairs with separate V/proj.
* **MUDD** — multi-layer skip connections to residual stream and attention values.
* **Learnable XSA** — cross-self-attention across layer pairs.
* **Single activation input** — last 3 attention layers share a single input.
* **Exponential residual decay** — alternative to per-layer learnable lambdas.
* **Multi-token prediction heads** — auxiliary future-token prediction heads.
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


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
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(head_dim // 4)])
        self.register_buffer("angular_freq", angular_freq, persistent=False)

    def forward(self, x_BTHD: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(x_BTHD.size(1), dtype=torch.float32, device=x_BTHD.device)
        theta = torch.outer(pos, self.angular_freq)[None, :, None, :]
        cos, sin = theta.cos(), theta.sin()
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)


class RotaryWithOffset(nn.Module):
    """RoPE with configurable position offset (for Partial Key Offset)."""

    def __init__(self, head_dim: int, base_inv_freq: float = 1024.0):
        super().__init__()
        if head_dim % 4 != 0:
            raise ValueError(f"head_dim must be divisible by 4, got {head_dim}")
        angular_freq = (1.0 / base_inv_freq) ** torch.linspace(
            0, 1, steps=head_dim // 4, dtype=torch.float32
        )
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(head_dim // 4)])
        self.register_buffer("angular_freq", angular_freq, persistent=False)

    def forward(self, x_BTHD: torch.Tensor, pos_offset: float = 0.0) -> torch.Tensor:
        T = x_BTHD.size(1)
        pos = torch.arange(T, dtype=torch.float32, device=x_BTHD.device)
        theta = torch.outer(pos + pos_offset, self.angular_freq)[None, :, None, :]
        cos, sin = theta.cos(), theta.sin()
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)


def _sliding_window_mask(
    seq_len: int, window_size: int, device: torch.device
) -> torch.Tensor:
    """Causal band mask for SDPA: query i attends to keys in [i-window+1, i]."""
    idx = torch.arange(seq_len, device=device)
    mask = (idx[None, :] > idx[:, None]) | (
        idx[None, :] < idx[:, None] - window_size + 1
    )
    return mask


class CausalSelfAttention(nn.Module):
    """Causal self-attention with QK-norm, RoPE, value-residual mix, and head gating."""

    def __init__(self, dim: int, num_heads: int, *, attn_scale: float = 0.12):
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

    def forward(
        self,
        x: torch.Tensor,
        value_embed: torch.Tensor | None,
        *,
        window_size: int | None = None,
        partial_key_offset: float | None = None,
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
        if partial_key_offset is not None:
            k_rot = RotaryWithOffset(self.head_dim)
            k_rot.angular_freq = self.rotary.angular_freq
            k = k_rot(k, pos_offset=partial_key_offset)
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        if window_size is not None and window_size < T:
            attn_mask = _sliding_window_mask(T, window_size, x.device)
            y = F.scaled_dot_product_attention(
                q_t,
                k_t,
                v_t,
                attn_mask=attn_mask,
                scale=self.attn_scale,
            ).transpose(1, 2)
        else:
            y = F.scaled_dot_product_attention(
                q_t, k_t, v_t, is_causal=True, scale=self.attn_scale
            ).transpose(1, 2)
        gate = torch.sigmoid(self.attn_gate(x[..., :12])).view(B, T, self.num_heads, 1)
        y = (y * gate).contiguous().view(B, T, self.num_heads * self.head_dim)
        return self.proj(y)


class PairedHeadAttention(nn.Module):
    """Paired-head self-attention: pairs of heads share Q/K projections.

    Within each pair, heads have separate V projections and output projections.
    """

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
        partial_key_offset: float | None = None,
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
        q, k = self.rotary(q), self.rotary(k)
        if partial_key_offset is not None:
            k_rot = RotaryWithOffset(self.head_dim)
            k_rot.angular_freq = self.rotary.angular_freq
            k = k_rot(k, pos_offset=partial_key_offset)
        # Reshape V to pair structure: (B,T,num_heads,head_dim) -> (B,T,num_pairs,2*head_dim)
        v_pair = v.view(B, T, self.num_pairs, 2, self.head_dim).reshape(
            B, T, self.num_pairs, 2 * self.head_dim
        )
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v_pair.transpose(1, 2)
        if window_size is not None and window_size < T:
            attn_mask = _sliding_window_mask(T, window_size, x.device)
            y = F.scaled_dot_product_attention(
                q_t,
                k_t,
                v_t,
                attn_mask=attn_mask,
                scale=self.attn_scale,
            ).transpose(1, 2)
        else:
            y = F.scaled_dot_product_attention(
                q_t, k_t, v_t, is_causal=True, scale=self.attn_scale
            ).transpose(1, 2)
        # y shape: (B, T, num_pairs, 2*head_dim) -> reshape to (B, T, num_heads, head_dim)
        y = y.reshape(B, T, self.num_heads, self.head_dim)
        gate = torch.sigmoid(self.attn_gate(x[..., :12])).view(B, T, self.num_heads, 1)
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
    ):
        super().__init__()
        attn_cls = PairedHeadAttention if paired_head else CausalSelfAttention
        self.attn = attn_cls(dim, num_heads, attn_scale=attn_scale)
        self.mlp = MLP(dim, mlp_ratio)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        value_embed: torch.Tensor | None,
        *,
        window_size: int | None = None,
        partial_key_offset: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attn_out = self.attn(
            self.norm1(x),
            value_embed,
            window_size=window_size,
            partial_key_offset=partial_key_offset,
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
    """Bigram hash embedding on 1/4 of model_dim with sign trick.

    For each consecutive pair of tokens, a hash determines whether each element
    in the embedding is +1 or -1 (sign trick). The remaining 3/4 of model_dim
    uses a standard token embedding.
    """

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
        h = (self.hash_seed[prev] * 2654435761) ^ (self.hash_seed[curr] * 2246822519)
        sign = (h.float() * (1.0 / 2**31)).fmod(2.0).abs().sub(1.0).sign()
        sign = sign.unsqueeze(-1).expand(-1, -1, self.hash_dim)
        pad_first = torch.zeros(B, 1, self.hash_dim, device=x.device, dtype=sign.dtype)
        hash_part = torch.cat([pad_first, sign], dim=1)
        return torch.cat([tok_part, hash_part], dim=-1)


class Smear(nn.Module):
    """Learned 1-token lookback smear on the embedding stream.

    Each dimension has a learnable gate controlling how much of the previous
    token's activation is added to the current token's activation.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.smear_gate = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prev = torch.cat([x[:, :1], x[:, :-1]], dim=1)
        gate = torch.sigmoid(self.smear_gate)
        return x + gate * prev


class MUDD(nn.Module):
    """Multi-layer skip connections feeding residual stream and attention values.

    MUDD connections project from early encoder layers to later decoder layers,
    contributing to both the residual stream and the attention value input.
    """

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
        resid = torch.sigmoid(self.resid_gates[layer_idx]) * self.resid_projs[
            layer_idx
        ](source)
        val = torch.sigmoid(self.value_gates[layer_idx]) * self.value_projs[layer_idx](
            source
        )
        return resid, val


class XSA(nn.Module):
    """Learnable cross-self-attention across layer pairs.

    A lightweight attention module that lets decoder layers attend to encoder
    layer outputs, with learnable interpolation between XSA and local context.
    """

    def __init__(self, dim: int, num_heads: int, *, attn_scale: float = 0.12):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.attn_scale = float(attn_scale)
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.proj.weight.data.zero_()
        self.gate = nn.Parameter(torch.tensor(0.0))
        self.norm_kv = RMSNorm(dim)
        self.rotary = Rotary(self.head_dim)

    def forward(self, x: torch.Tensor, cross_src: torch.Tensor) -> torch.Tensor:
        B, T = x.shape[0], x.shape[1]
        q = self.q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k(self.norm_kv(cross_src)).view(B, -1, self.num_heads, self.head_dim)
        v = self.v(self.norm_kv(cross_src)).view(B, -1, self.num_heads, self.head_dim)
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        q, k = self.rotary(q), self.rotary(k)
        q_t, k_t, v_t = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        y = (
            F.scaled_dot_product_attention(
                q_t, k_t, v_t, is_causal=False, scale=self.attn_scale
            )
            .transpose(1, 2)
            .contiguous()
            .view(B, T, -1)
        )
        return torch.tanh(self.gate) * self.proj(y)


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
    """Decoder-only transformer with U-net skips, sparse value embeddings, and speedrun lambdas."""

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
        # ---- portable feature flags (all off by default) ----
        bigram_hash_embed: bool = False,
        smear_embed: bool = False,
        partial_key_offset: float | None = None,
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
        # sqrt(1.1) per sublayer so cumulative per-layer residual scale is 1.1
        resid_init = math.sqrt(1.1)
        self.resid_lambdas_attn = nn.Parameter(torch.full((num_layers,), resid_init))
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
                    paired_head=paired_head,
                )
                for _ in range(num_layers)
            ]
        )

        # MUDD skip connections
        self.mudd_pairs = mudd_pairs
        if mudd_pairs > 0:
            self.mudd = MUDD(model_dim, num_skip_pairs=mudd_pairs)
        else:
            self.mudd = None

        # XSA (cross-self-attention)
        self.xsa_enabled = xsa_enabled
        self.xsa_pairs = xsa_pairs
        if xsa_enabled and xsa_pairs > 0:
            self.xsa_modules = nn.ModuleList(
                [
                    XSA(model_dim, num_heads, attn_scale=attn_scale)
                    for _ in range(xsa_pairs)
                ]
            )
            self.xsa_layer_map: list[tuple[int, int]] = []

        # Multi-token prediction heads
        if multi_token_pred > 0:
            self.multi_heads = MultiTokenHeads(model_dim, vocab_size, multi_token_pred)
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
        self, idx: torch.Tensor, *, window_size: int | None = None
    ) -> torch.Tensor:
        """Run the trunk through ``norm_out`` without materializing full-vocab logits.

        Used by held-out validation to score CE in lm_head/softcap chunks so a
        single A100 80GB pass never allocates oversized ``(B*T, vocab)`` tensors.
        """
        ws = window_size if window_size is not None else self.sliding_window_size
        x = self.norm_in(self.embed(idx))
        if self.smear is not None:
            x = self.smear(x)
        x0 = x
        ve = self.value_embeds(idx)
        ve_enc, ve_dec = ve[: self.num_encoder_layers], ve[self.num_encoder_layers :]
        skip_connections: list[torch.Tensor] = []
        encoder_outputs: list[torch.Tensor] = []
        for i in range(self.num_encoder_layers):
            attn_out, mlp_out = self.blocks[i](
                x, ve_enc[i], window_size=ws, partial_key_offset=self.partial_key_offset
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

        xsa_src: list[torch.Tensor] = list(reversed(encoder_outputs))
        for i in range(self.num_decoder_layers):
            layer_idx = self.num_encoder_layers + i

            # MUDD contribution
            mudd_resid, mudd_val = None, None
            if self.mudd is not None:
                src_idx = max(0, self.num_encoder_layers - 1 - i)
                if src_idx < len(encoder_outputs):
                    mudd_resid, mudd_val = self.mudd(encoder_outputs[src_idx], i)

            x = x + torch.sigmoid(self.skip_gates[i]) * skip_connections.pop()

            # XSA contribution
            if (
                self.xsa_enabled
                and self.xsa_pairs > 0
                and i < self.xsa_pairs
                and xsa_src
            ):
                xsa_out = self.xsa_modules[i](x, xsa_src[i])
                x = x + xsa_out

            # Determine attention input: single activation for last k layers
            attn_input = x
            if (
                self.single_act_last_k > 0
                and i >= self.num_decoder_layers - self.single_act_last_k
            ):
                if single_act is not None:
                    attn_input = single_act

            # MUDD value contribution to attention
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
                partial_key_offset=self.partial_key_offset,
            )

            # MUDD residual contribution
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
        """Project hidden states through ``lm_head`` + tanh softcap in float32.

        Softcap and CE are always computed in fp32 even when the trunk runs in
        bfloat16 on CUDA, matching speedrun numerical practice.
        """
        logits = self.lm_head(hidden.float()).float()
        return self.softcap * torch.tanh(logits / self.softcap)

    def forward(
        self,
        idx: torch.Tensor,
        *,
        window_size: int | None = None,
        output_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        hidden = self.forward_hidden(idx, window_size=window_size)
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
    # --- portable feature flags ---
    bigram_hash_embed: bool = False
    smear_embed: bool = False
    partial_key_offset: float | None = None
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
# Baseline param count without portable features. Portable features add params
# when enabled, so the pinned count is for the base config only.
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
    partial_key_offset: float | None = None,
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


_MODEL_COMPONENTS = (
    RMSNorm,
    Rotary,
    RotaryWithOffset,
    _sliding_window_mask,
    CausalSelfAttention,
    PairedHeadAttention,
    MLP,
    Block,
    ValueEmbedding,
    BigramHashEmbedding,
    Smear,
    MUDD,
    XSA,
    MultiTokenHeads,
    GPT,
)


def model_source() -> str:
    """The verbatim source of the model components, for byte-identical embedding."""
    return "\n\n\n".join(inspect.getsource(c).rstrip() for c in _MODEL_COMPONENTS)
