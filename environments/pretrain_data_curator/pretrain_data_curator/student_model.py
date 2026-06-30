"""Modern proxy-student architecture (modded-nanogpt), pure CPU-runnable PyTorch.

This module is the **single source of truth** for the proxy-student model. The same
class definitions are embedded byte-identically into the GPU sandbox training
script (``trainer.py``) via :func:`model_source`, so the CPU unit tests in this
package exercise the exact code the sandbox runs (see the verbatim-source test).

The architecture follows the *modern* design from
``KellerJordan/modded-nanogpt`` — the cleaned-up
``records/track_3_optimization/train_gpt_simple.py``, the
``2024-12-08_UNetValueEmbedsTweaks`` record, and the
``2024-12-17_SparsifyEmbeds`` record (which sparsified the value embeddings) —
while deliberately EXCLUDING the repo's GPU-specific *training* optimizations
(Muon/NorMuon, FP8, distributed comms, multi-token prediction, FlexAttention,
``torch.compile``/triton, and bf16-only assumptions). Everything runs fp32 on CPU.

Components implemented (all CPU-runnable):

* **RoPE** — half-truncate, base-frequency rotary embeddings (``Rotary``).
* **RMSNorm** with a learnable gain (``RMSNorm``), via ``F.rms_norm``.
* **QK-norm** — unweighted ``F.rms_norm`` on per-head q and k before attention.
* **ReLU² MLP** — ``F.relu(x).square()`` feed-forward.
* **SDPA causal attention** — ``F.scaled_dot_product_attention(..., is_causal=
  True)``. This REPLACES the record's GPU-only FlexAttention (the document/
  sliding-window block masking is a GPU-path feature and is intentionally
  dropped; plain causal masking is used instead).
* **tanh logit softcap** — ``softcap * tanh(logits / softcap)`` (default 30).
* **U-net encoder/decoder skip connections** with learnable ``skip_weights``.
* **Sparse value embeddings** (``ValueEmbedding``, the SparsifyEmbeds design):
  a small number of DISTINCT full-model_dim tables (``num_value_embeds``, default
  3) applied only to the FIRST and LAST ``num_value_embeds`` layers (U-net mirror)
  with ``None`` for the middle layers, fed to the per-attention value-residual
  ``lambdas`` (``v = lambdas[0]*v + lambdas[1]*value_embed``, or ``lambdas[0]*v``
  when the layer's value embedding is ``None``).
* **Block lambdas** mixing the current residual with the post-embedding ``x0``.
* **Untied** ``lm_head`` with zero-init weights; padded vocab 50304.

Linear layers are bias-free (the modern convention); the only biases-equivalent
learnable scalars are the RMSNorm gains and the residual/value ``lambdas``.

Chosen GPT-2-small-class config (see :data:`GPT2_SMALL`):
``model_dim=768, num_layers=12, num_heads=6 (head_dim=128), mlp_ratio=4,
vocab_size=50304, softcap=30, num_value_embeds=3``.

Exact instantiated parameter count: **278,122,038** (~278M), see
:data:`GPT2_SMALL_PARAM_COUNT`. (The earlier C pass used 6 full per-encoder-layer
value tables, ~394M; aligning to SparsifyEmbeds' 3 tables both raises fidelity and
drops ~116M of value-embedding params.) The remaining excess over the canonical
124M *tied-embedding* GPT-2-small is intrinsic to the modern design:

    embed (wte)              V*d                  =  38,633,472
    value_embeds (3 tables)  3 * V*d              = 115,900,416   <- dominates
    12 transformer blocks (attn+mlp+norms+lambdas)=  84,953,136
    lm_head (UNTIED)         d*V                  =  38,633,472
    skip_weights + norms + lambdas                =       1,542
    ---------------------------------------------------------------
    total                                           278,122,038

The 3 distinct sparse value-embedding tables (115.9M) and the untied head (+38.6M
over a tied embedding) account for essentially all of the gap. The model
dimensions are squarely GPT-2-small (768-wide, 12 deep); the parameter *count* is
larger by construction, and architectural fidelity to modded-nanogpt — not the
124M number — is the governing criterion. The runtime ``ProxyStudentConfig`` still
defaults to a tiny config for cheap CPU/sandbox jobs — this preset documents the
GPT-2-small target the architecture is built for.

Upstream fidelity note: modded-nanogpt's SparsifyEmbeds (``ValueEmbedding`` =
3 full-width ``nn.Embedding`` tables; ``ve = [v0,v1,v2, None*6, v0,v1,v2]``) totals
~275.6M at GPT-2-small. This student matches the value-embedding design exactly;
the ~2.5M difference is documented and intentional: attention is kept on ALL 12
layers (SparsifyEmbeds drops layer-7 attention, a config-specific micro-opt that
does not generalize to a configurable depth), the vocab is uniformly padded to
50304 (SparsifyEmbeds pads only the head), and RMSNorm keeps learnable gains.
"""

from __future__ import annotations

import inspect
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
    """Half-truncate, base-frequency rotary position embeddings (RoPE).

    The first ``head_dim // 4`` frequencies follow a geometric schedule; the
    remaining ``head_dim // 4`` are zeroed (the "half-truncate" trick), so only
    the lower half of each head's channels are rotated. ``head_dim`` must be a
    multiple of 4.
    """

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


class CausalSelfAttention(nn.Module):
    """Causal self-attention with QK-norm, RoPE, and a value-residual mix.

    Uses ``F.scaled_dot_product_attention(is_causal=True)`` (CPU-runnable) in
    place of the GPU-only FlexAttention kernel from the upstream record.
    """

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim ({dim}) must be divisible by num_heads ({num_heads})")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.proj.weight.data.zero_()  # zero-init residual projection
        # Value-residual mix: v = lambdas[0]*v + lambdas[1]*value_embed.
        self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.rotary = Rotary(self.head_dim)

    def forward(self, x: torch.Tensor, value_embed: torch.Tensor | None) -> torch.Tensor:
        B, T = x.size(0), x.size(1)
        q = self.q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v(x).view(B, T, self.num_heads, self.head_dim)
        if value_embed is None:
            v = self.lambdas[0] * v  # sparse: this layer has no value embedding
        else:
            v = self.lambdas[0] * v + self.lambdas[1] * value_embed.view_as(v)
        q = F.rms_norm(q, (q.size(-1),))  # QK-norm (unweighted)
        k = F.rms_norm(k, (k.size(-1),))
        q, k = self.rotary(q), self.rotary(k)
        y = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True
        ).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
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
    """Pre-norm transformer block with a learnable x0 residual mix."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.attn = CausalSelfAttention(dim, num_heads)
        self.mlp = MLP(dim, mlp_ratio)
        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        # Mix the running residual with the post-embedding x0 (value-residual idea).
        self.lambdas = nn.Parameter(torch.tensor([1.0, 0.0]))

    def forward(
        self, x: torch.Tensor, value_embed: torch.Tensor | None, x0: torch.Tensor
    ) -> torch.Tensor:
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        x = x + self.attn(self.norm1(x), value_embed)
        x = x + self.mlp(self.norm2(x))
        return x


class ValueEmbedding(nn.Module):
    """Sparse per-token value embeddings (modded-nanogpt SparsifyEmbeds design).

    Holds ``num_tables`` DISTINCT full-``model_dim`` embedding tables (default 3),
    applied only to the first and last ``num_tables`` layers — a U-net mirror — with
    ``None`` for the middle layers, so most layers carry no value-residual. This
    matches the 2024-12-17 SparsifyEmbeds record, whose 12-layer pattern is
    ``[v0, v1, v2, None*6, v0, v1, v2]`` from 3 tables.

    ``num_tables`` is clamped to ``num_layers // 2`` so the first/last bands never
    overlap; ``forward`` returns a list of length ``num_layers`` whose entries are
    either a ``(B, T, model_dim)`` tensor or ``None``.
    """

    def __init__(self, vocab_size: int, model_dim: int, num_layers: int, num_tables: int = 3):
        super().__init__()
        self.num_layers = num_layers
        self.num_tables = max(1, min(num_tables, num_layers // 2))
        self.embed = nn.ModuleList(
            [nn.Embedding(vocab_size, model_dim) for _ in range(self.num_tables)]
        )

    def forward(self, idx: torch.Tensor) -> list:
        tables = [emb(idx) for emb in self.embed]
        middle = self.num_layers - 2 * self.num_tables
        return tables + [None] * middle + tables


class GPT(nn.Module):
    """Decoder-only transformer with U-net skips and sparse value embeddings.

    ``forward(idx)`` returns tanh-softcapped logits of shape ``(B, T, vocab_size)``
    (the cross-entropy loss is computed by the caller, so the held-out validation
    windowing stays in the training script). ``num_layers`` must be even and >= 2
    so the encoder and decoder halves are symmetric (the U-net pushes one skip per
    encoder layer and the decoder pops exactly one per layer).
    """

    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        mlp_ratio: int = 4,
        softcap: float = 30.0,
        num_value_embeds: int = 3,
    ):
        super().__init__()
        if num_layers < 2 or num_layers % 2 != 0:
            raise ValueError(f"num_layers must be even and >= 2, got {num_layers}")
        self.num_layers = num_layers
        self.softcap = float(softcap)
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        # One learnable weight per decoder layer for the U-net skip connections.
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))
        self.embed = nn.Embedding(vocab_size, model_dim)
        # Sparse value embeddings (SparsifyEmbeds): a few distinct full-width tables
        # applied only to the first/last layers, None for the middle ones.
        self.value_embeds = ValueEmbedding(
            vocab_size, model_dim, num_layers, num_value_embeds
        )
        self.blocks = nn.ModuleList(
            [Block(model_dim, num_heads, mlp_ratio) for _ in range(num_layers)]
        )
        self.norm_in = RMSNorm(model_dim)
        self.norm_out = RMSNorm(model_dim)
        self.lm_head = nn.Linear(model_dim, vocab_size, bias=False)
        self.lm_head.weight.data.zero_()  # untied, zero-init head

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.norm_in(self.embed(idx))
        x0 = x
        ve = self.value_embeds(idx)  # length num_layers; entries are Tensor or None
        ve_enc, ve_dec = ve[: self.num_encoder_layers], ve[self.num_encoder_layers :]
        skip_connections = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, ve_enc[i], x0)
            skip_connections.append(x)
        for i in range(self.num_decoder_layers):
            x = x + self.skip_weights[i] * skip_connections.pop()
            x = self.blocks[self.num_encoder_layers + i](x, ve_dec[i], x0)
        x = self.norm_out(x)
        logits = self.lm_head(x)
        return self.softcap * torch.tanh(logits / self.softcap)


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

    def build(self) -> GPT:
        return GPT(
            vocab_size=self.vocab_size,
            num_layers=self.num_layers,
            model_dim=self.model_dim,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            softcap=self.softcap,
            num_value_embeds=self.num_value_embeds,
        )


# Chosen GPT-2-small-class preset and its EXACT instantiated parameter count.
# See the module docstring for the full per-component breakdown. The count is
# pinned by ``test_student_model.py`` (instantiated on the meta device, so no
# allocation) to guard against silent architectural drift.
GPT2_SMALL = StudentModelConfig()
GPT2_SMALL_PARAM_COUNT = 278_122_038

# Source-of-truth model components, in dependency (definition) order. Their exact
# source is embedded into the sandbox training script in ``trainer.py``.
_MODEL_COMPONENTS = (RMSNorm, Rotary, CausalSelfAttention, MLP, Block, ValueEmbedding, GPT)


def model_source() -> str:
    """The verbatim source of the model components, for byte-identical embedding.

    Returns the concatenated source of every ``nn.Module`` in this module's model
    (in definition order). ``trainer.py`` injects this exact string into its
    sandbox training script, and a unit test asserts the substring is present, so
    the GPU-only training run executes the same code these CPU tests exercise.
    """
    return "\n\n\n".join(inspect.getsource(c).rstrip() for c in _MODEL_COMPONENTS)
