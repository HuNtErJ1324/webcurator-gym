"""CPU unit tests for the modern proxy-student model (``student_model.py``).

These guard the single source of truth for the architecture: every component is
exercised on CPU here, the GPT-2-small param count is pinned, and the verbatim
model source is asserted to be embedded byte-identically in the sandbox training
script in ``trainer.py`` (so the GPU run executes this exact code).
"""

from __future__ import annotations

import ast
import inspect

import pytest
import torch

from pretrain_data_curator.student_model import (
    _MODEL_COMPONENTS,
    GPT,
    GPT2_SMALL,
    GPT2_SMALL_PARAM_COUNT,
    MLP,
    RMSNorm,
    Rotary,
    CausalSelfAttention,
    StudentModelConfig,
    ValueEmbedding,
    model_source,
)
from pretrain_data_curator.trainer import NANOGPT_TRAIN_SCRIPT


def _randomize(model: torch.nn.Module) -> None:
    """Give every parameter non-trivial values.

    A freshly-built model has zero-init output projections and a zero-init head,
    so its logits are identically zero and wiring contributions are invisible.
    Filling all parameters simulates a trained model so the residual/skip/value
    paths become observable.
    """
    with torch.no_grad():
        for p in model.parameters():
            p.normal_(0.0, 0.2)


# --- (a) instantiated param count is in the documented GPT-2-small band ----


def test_gpt2_small_param_count_is_pinned():
    # Instantiate on the meta device so the ~278M-param model costs no memory.
    with torch.device("meta"):
        model = GPT2_SMALL.build()
    n_params = sum(p.numel() for p in model.parameters())
    # Exact pin (single source of truth; guards silent architectural drift).
    assert n_params == GPT2_SMALL_PARAM_COUNT == 278_122_038
    # Documented GPT-2-small-class band: 768-wide, 12-deep, but larger by count
    # because of the untied head and the 3 sparse (SparsifyEmbeds) value tables.
    assert 270_000_000 <= n_params <= 285_000_000
    assert n_params > 124_000_000  # well above the tied-embedding canonical 124M
    assert GPT2_SMALL.model_dim == 768 and GPT2_SMALL.num_layers == 12


def test_gpt2_small_unet_halves_are_symmetric():
    with torch.device("meta"):
        model = GPT2_SMALL.build()
    assert model.num_encoder_layers == model.num_decoder_layers == 6
    assert model.skip_weights.numel() == model.num_decoder_layers
    # SparsifyEmbeds: exactly 3 DISTINCT full-model_dim value-embedding tables.
    assert model.value_embeds.num_tables == GPT2_SMALL.num_value_embeds == 3
    assert len(model.value_embeds.embed) == 3
    for table in model.value_embeds.embed:
        assert tuple(table.weight.shape) == (GPT2_SMALL.vocab_size, GPT2_SMALL.model_dim)


def test_value_embeddings_match_sparsify_embeds_pattern():
    # The 12-layer SparsifyEmbeds layout is [v0,v1,v2, None*6, v0,v1,v2]: the first
    # and last 3 layers share the SAME 3 tables, the middle 6 get no value residual.
    torch.manual_seed(0)
    model = StudentModelConfig(
        model_dim=16, num_layers=12, num_heads=2, vocab_size=40, num_value_embeds=3
    ).build()
    ve = model.value_embeds(torch.randint(0, 40, (1, 5)))
    assert len(ve) == 12
    present = [i for i, t in enumerate(ve) if t is not None]
    assert present == [0, 1, 2, 9, 10, 11]
    # First and last bands are the SAME tensors (reused, not duplicated tables).
    assert ve[0] is ve[9] and ve[1] is ve[10] and ve[2] is ve[11]


def test_value_embedding_table_count_and_clamp():
    # 3 distinct full-model_dim tables; forward returns one entry per layer.
    ve = ValueEmbedding(40, 16, num_layers=12, num_tables=3)
    assert len(ve.embed) == ve.num_tables == 3
    out = ve(torch.randint(0, 40, (1, 4)))
    assert len(out) == 12 and out[0].shape == (1, 4, 16)
    # num_tables is clamped to num_layers // 2 so the first/last bands don't overlap.
    assert ValueEmbedding(40, 16, num_layers=4, num_tables=3).num_tables == 2


# --- (b) tiny-config forward returns correctly-shaped finite logits --------


def _tiny() -> GPT:
    torch.manual_seed(0)
    return StudentModelConfig(
        model_dim=32, num_layers=4, num_heads=2, mlp_ratio=4, vocab_size=64
    ).build()


def test_tiny_forward_shape_and_finite():
    model = _tiny().eval()
    idx = torch.randint(0, 64, (3, 7))
    logits = model(idx)
    assert logits.shape == (3, 7, 64)
    assert torch.isfinite(logits).all()


# --- (c) every new component exercised -------------------------------------


def test_rmsnorm_normalizes_to_unit_rms():
    norm = RMSNorm(16)
    x = torch.randn(5, 16) * 7.0  # arbitrary scale
    y = norm(x)
    rms = y.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones(5), atol=1e-3)


def test_rotary_shape_and_rotation():
    rot = Rotary(8)
    x = torch.randn(2, 4, 3, 8)  # (B, T, H, D)
    y = rot(x)
    assert y.shape == x.shape
    # theta=0 at position 0 -> RoPE is the identity there...
    assert torch.allclose(y[:, 0], x[:, 0], atol=1e-5)
    # ...but later positions are rotated (non-trivial change).
    assert not torch.allclose(y[:, 1], x[:, 1], atol=1e-3)


def test_rotary_requires_head_dim_multiple_of_four():
    with pytest.raises(ValueError, match="divisible by 4"):
        Rotary(6)


def test_rotary_sign_convention_and_half_truncation():
    # Pin the EXACT half-truncate RoPE: the upper half of the frequencies are zero
    # (so those channels are never rotated), and the rotation uses the
    # +cos/+sin / -sin/+cos sign convention.
    head_dim = 8
    rot = Rotary(head_dim, base_inv_freq=1024.0)
    quarter, half = head_dim // 4, head_dim // 2  # 2, 4
    freq = rot.angular_freq
    assert (freq[:quarter] > 0).all()  # active (geometric) frequencies
    assert torch.allclose(freq[quarter:], torch.zeros(quarter))  # half-truncated

    x = torch.zeros(1, 3, 1, head_dim)
    x[..., :half] = 1.0  # x1 = ones
    x[..., half:] = 0.0  # x2 = zeros
    y = rot(x)
    pos = 2
    theta = pos * freq
    # y1 = x1*cos + x2*sin = cos ; y2 = -x1*sin + x2*cos = -sin  (sign convention)
    assert torch.allclose(y[0, pos, 0, :half], torch.cos(theta), atol=1e-5)
    assert torch.allclose(y[0, pos, 0, half:], -torch.sin(theta), atol=1e-5)
    # The half-truncated (zero-frequency) channels pass through unrotated: there
    # y1 == x1 == 1 and y2 == x2 == 0 at every position.
    assert torch.allclose(y[0, pos, 0, quarter:half], torch.ones(quarter), atol=1e-5)
    assert torch.allclose(y[0, pos, 0, half + quarter :], torch.zeros(quarter), atol=1e-5)


def test_qk_norm_makes_attention_invariant_to_qk_scale():
    # QK-norm rms-normalizes q and k per head, so rescaling the q/k projection
    # weights must not change the attention output at all.
    attn = CausalSelfAttention(16, num_heads=2).eval()
    # Defeat the zero-init output projection so the output is genuinely non-zero;
    # otherwise both sides are identically zero and the test is vacuous (it would
    # pass even with QK-norm deleted).
    with torch.no_grad():
        attn.proj.weight.normal_(0.0, 0.3)
    x = torch.randn(2, 5, 16)
    value_embed = torch.randn(2, 5, 16)
    with torch.no_grad():
        before = attn(x, value_embed)
        assert before.abs().max() > 1e-3  # non-vacuous: output is not all-zero
        attn.q.weight.mul_(7.0)
        attn.k.weight.mul_(7.0)  # 49x raw-score scaling iff QK-norm were absent
        after = attn(x, value_embed)
    assert torch.allclose(before, after, atol=1e-4)


def test_relu_squared_mlp_zeros_negative_preactivations():
    # ReLU**2 (not GELU/identity) must map all-negative pre-activations to exactly
    # zero; with bias-free projections the whole MLP output is then exactly zero.
    mlp = MLP(8, mlp_ratio=4)
    with torch.no_grad():
        mlp.fc.weight.fill_(-1.0)  # fc(ones) = -8 everywhere -> relu -> 0
        mlp.proj.weight.normal_()
    out = mlp(torch.ones(1, 8))
    assert torch.count_nonzero(out).item() == 0


def test_tanh_softcap_bounds_logits():
    softcap = 30.0
    model = StudentModelConfig(
        model_dim=32, num_layers=4, num_heads=2, vocab_size=64, softcap=softcap
    ).build()
    _randomize(model)
    with torch.no_grad():
        model.lm_head.weight.mul_(50.0)  # push pre-cap logits well past the cap
        logits = model(torch.randint(0, 64, (2, 6)))
    # softcap * tanh(.) is strictly bounded by softcap regardless of input scale.
    assert logits.abs().max().item() < softcap
    assert torch.isfinite(logits).all()


def test_unet_decoder_consumes_encoder_skips():
    # The decoder adds skip_weights[i] * (encoder skip); zeroing the weights must
    # change the output, proving the encoder->decoder skip path is wired in.
    model = GPT(64, num_layers=4, model_dim=32, num_heads=2).eval()
    _randomize(model)
    idx = torch.randint(0, 64, (1, 6))
    with torch.no_grad():
        with_skips = model(idx).clone()
        model.skip_weights.zero_()
        without_skips = model(idx)
    assert not torch.allclose(with_skips, without_skips, atol=1e-6)


def test_value_embedding_residual_path_is_used():
    # Zeroing the value-embedding tables must change the output, proving the sparse
    # value-residual (v = l0*v + l1*value_embed) actually flows on the layers that
    # carry one. Use L=6 so the SparsifyEmbeds pattern has live value layers.
    model = GPT(64, num_layers=6, model_dim=32, num_heads=2, num_value_embeds=3).eval()
    _randomize(model)
    idx = torch.randint(0, 64, (1, 6))
    with torch.no_grad():
        before = model(idx).clone()
        for table in model.value_embeds.embed:
            table.weight.zero_()
        after = model(idx)
    assert not torch.allclose(before, after, atol=1e-6)
    # Each attention carries the two-element value-residual lambda mix.
    assert model.blocks[0].attn.lambdas.shape == (2,)
    assert model.blocks[0].lambdas.shape == (2,)  # block x0 residual mix


def test_sparse_value_layers_get_no_residual():
    # On a layer whose value embedding is None, attention must use only lambdas[0]*v
    # (no value-residual term); so a fresh model's middle layers carry no value
    # contribution. We verify the None branch runs and yields finite output for a
    # config with genuine None layers (L=8, k=2 -> layers 2..5 are None).
    model = GPT(64, num_layers=8, model_dim=32, num_heads=2, num_value_embeds=2).eval()
    ve = model.value_embeds(torch.randint(0, 64, (1, 5)))
    assert [t is None for t in ve] == [False, False, True, True, True, True, False, False]
    _randomize(model)
    out = model(torch.randint(0, 64, (1, 6)))
    assert torch.isfinite(out).all()


def test_gpt_rejects_odd_or_too_few_layers():
    with pytest.raises(ValueError, match="even and >= 2"):
        GPT(64, num_layers=3, model_dim=32, num_heads=2)
    with pytest.raises(ValueError, match="even and >= 2"):
        GPT(64, num_layers=1, model_dim=32, num_heads=2)


# --- single source of truth: verbatim model embedded in the sandbox script --


def test_trainer_embeds_student_model_verbatim():
    # The GPU-only training script must run the SAME model these CPU tests
    # exercise: the exact model source appears byte-identically in the script
    # (a refactor of the model can't silently diverge the sandbox copy).
    ast.parse(NANOGPT_TRAIN_SCRIPT)  # assembled script is valid Python
    src = model_source()
    assert src in NANOGPT_TRAIN_SCRIPT
    # Every model component (including Block and ValueEmbedding) is present
    # verbatim — iterate the actual source-of-truth tuple so none is omitted.
    for component in _MODEL_COMPONENTS:
        assert inspect.getsource(component).rstrip() in NANOGPT_TRAIN_SCRIPT
    # Modern components are wired; the old LayerNorm/MHA/GELU model is gone, and
    # no GPU-only FlexAttention dependency leaked into the script.
    assert "F.scaled_dot_product_attention" in NANOGPT_TRAIN_SCRIPT
    assert "F.rms_norm" in NANOGPT_TRAIN_SCRIPT
    assert "torch.tanh" in NANOGPT_TRAIN_SCRIPT
    assert "nn.LayerNorm" not in NANOGPT_TRAIN_SCRIPT
    assert "MultiheadAttention" not in NANOGPT_TRAIN_SCRIPT
    assert "flex_attention" not in NANOGPT_TRAIN_SCRIPT
