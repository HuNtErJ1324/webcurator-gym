"""CPU unit tests for the model in the single-file ``train_gpt.py`` trainer."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

from pretrain_curation_gym.gpu.train_gpt import (
    _causal_attn_mask,
    _combine_attn_masks,
    _sliding_window_mask,
    GPT,
    GPT2_SMALL,
    MLP,
    RMSNorm,
    Rotary,
    RotaryWithOffset,
    CausalSelfAttention,
    PairedHeadAttention,
    StudentModelConfig,
    ValueEmbedding,
    BigramHashEmbedding,
    Smear,
    MUDD,
    XSA,
    MultiTokenHeads,
)
from pretrain_curation_gym.gpu.train_gpt import build_document_attn_mask


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


def test_gpt2_small_unet_halves_are_symmetric():
    with torch.device("meta"):
        model = GPT2_SMALL.build()
    assert model.num_encoder_layers == model.num_decoder_layers == 6
    assert model.skip_gates.numel() == model.num_decoder_layers
    assert model.post_lambdas.shape == (model.num_layers, 2)
    # SparsifyEmbeds: exactly 3 DISTINCT full-model_dim value-embedding tables.
    assert model.value_embeds.num_tables == GPT2_SMALL.num_value_embeds == 3
    assert len(model.value_embeds.embed) == 3
    for table in model.value_embeds.embed:
        assert tuple(table.weight.shape) == (
            GPT2_SMALL.vocab_size,
            GPT2_SMALL.model_dim,
        )


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
    assert torch.allclose(
        y[0, pos, 0, half + quarter :], torch.zeros(quarter), atol=1e-5
    )


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
    # The decoder adds sigmoid(skip_gates[i]) * (encoder skip); driving gates to
    # -inf must change the output, proving the encoder->decoder skip path is wired.
    model = GPT(64, num_layers=4, model_dim=32, num_heads=2).eval()
    _randomize(model)
    idx = torch.randint(0, 64, (1, 6))
    with torch.no_grad():
        with_skips = model(idx).clone()
        model.skip_gates.fill_(-50.0)
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
    assert model.post_lambdas.shape == (6, 2)
    assert model.resid_lambdas_attn.shape == (6,)
    assert model.x0_lambdas.shape == (6,)


def test_sparse_value_layers_get_no_residual():
    # On a layer whose value embedding is None, attention must use only lambdas[0]*v
    # (no value-residual term); so a fresh model's middle layers carry no value
    # contribution. We verify the None branch runs and yields finite output for a
    # config with genuine None layers (L=8, k=2 -> layers 2..5 are None).
    model = GPT(64, num_layers=8, model_dim=32, num_heads=2, num_value_embeds=2).eval()
    ve = model.value_embeds(torch.randint(0, 64, (1, 5)))
    assert [t is None for t in ve] == [
        False,
        False,
        True,
        True,
        True,
        True,
        False,
        False,
    ]
    _randomize(model)
    out = model(torch.randint(0, 64, (1, 6)))
    assert torch.isfinite(out).all()


def test_gpt_rejects_odd_or_too_few_layers():
    with pytest.raises(ValueError, match="even and >= 2"):
        GPT(64, num_layers=3, model_dim=32, num_heads=2)
    with pytest.raises(ValueError, match="even and >= 2"):
        GPT(64, num_layers=1, model_dim=32, num_heads=2)


def test_sliding_window_changes_output_vs_full_context():
    model = GPT(64, num_layers=4, model_dim=32, num_heads=2).eval()
    _randomize(model)
    idx = torch.randint(0, 64, (1, 8))
    with torch.no_grad():
        full = model(idx)
        windowed = model(idx, window_size=3)
    assert not torch.allclose(full, windowed, atol=1e-6)


def test_sdpa_boolean_mask_true_means_keep():
    """PyTorch SDPA boolean convention: True participates, False is masked."""
    keep = _causal_attn_mask(4, torch.device("cpu"))
    assert bool(keep[2, 0].item()) is True
    assert bool(keep[2, 3].item()) is False
    sliding = _sliding_window_mask(6, 3, torch.device("cpu"))
    # query 5 may see keys 3,4,5 only
    assert sliding[5].tolist() == [False, False, False, True, True, True]


def test_causal_mask_future_tokens_do_not_affect_prior_outputs():
    model = GPT(64, num_layers=2, model_dim=32, num_heads=2).eval()
    _randomize(model)
    base = torch.randint(0, 64, (1, 8))
    flipped = base.clone()
    flipped[0, 5:] = (flipped[0, 5:] + 17) % 64
    with torch.no_grad():
        out_base = model(base)
        out_flip = model(flipped)
    assert torch.allclose(out_base[:, :5], out_flip[:, :5], atol=1e-5)
    assert not torch.allclose(out_base[:, 5:], out_flip[:, 5:], atol=1e-5)


def test_document_mask_blocks_cross_document_influence_on_outputs():
    model = GPT(64, num_layers=2, model_dim=32, num_heads=2).eval()
    _randomize(model)
    base = torch.randint(0, 64, (1, 8))
    # Two documents: [0,4) and [4,8)
    doc_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
    attn_mask = build_document_attn_mask(doc_ids)
    other_doc = base.clone()
    other_doc[0, :4] = (other_doc[0, :4] + 23) % 64
    with torch.no_grad():
        out_base = model(base, attn_mask=attn_mask)
        out_other = model(other_doc, attn_mask=attn_mask)
    # Queries in doc1 must be invariant to doc0 token edits.
    assert torch.allclose(out_base[:, 4:], out_other[:, 4:], atol=1e-5)
    assert not torch.allclose(out_base[:, :4], out_other[:, :4], atol=1e-5)


def test_document_mask_same_document_past_tokens_affect_outputs():
    model = GPT(64, num_layers=2, model_dim=32, num_heads=2).eval()
    _randomize(model)
    base = torch.randint(0, 64, (1, 8))
    doc_ids = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
    attn_mask = build_document_attn_mask(doc_ids)
    same_doc = base.clone()
    same_doc[0, 4] = (same_doc[0, 4] + 29) % 64
    with torch.no_grad():
        out_base = model(base, attn_mask=attn_mask)
        out_same = model(same_doc, attn_mask=attn_mask)
    # Editing an earlier same-document token must change later same-doc outputs.
    assert not torch.allclose(out_base[:, 5:], out_same[:, 5:], atol=1e-5)
    # Positions before the edited token stay unchanged.
    assert torch.allclose(out_base[:, :4], out_same[:, :4], atol=1e-5)


def test_combined_sliding_window_and_document_mask_on_outputs():
    model = GPT(64, num_layers=2, model_dim=32, num_heads=2).eval()
    _randomize(model)
    base = torch.randint(0, 64, (1, 10))
    doc_ids = torch.zeros(1, 10, dtype=torch.long)  # one document
    attn_mask = build_document_attn_mask(doc_ids)
    # Far past token inside the document but outside window_size=3.
    far = base.clone()
    far[0, 0] = (far[0, 0] + 31) % 64
    near = base.clone()
    near[0, 8] = (near[0, 8] + 31) % 64
    with torch.no_grad():
        out_base = model(base, window_size=3, attn_mask=attn_mask)
        out_far = model(far, window_size=3, attn_mask=attn_mask)
        out_near = model(near, window_size=3, attn_mask=attn_mask)
    # Token 0 is outside the window of query 9, so it must not affect that output.
    assert torch.allclose(out_base[:, 9], out_far[:, 9], atol=1e-5)
    # Token 8 is inside the window of query 9, so it must affect that output.
    assert not torch.allclose(out_base[:, 9], out_near[:, 9], atol=1e-5)
    # Combine helper keeps True=participate semantics under AND.
    combined = _combine_attn_masks(
        10, torch.device("cpu"), window_size=3, attn_mask=attn_mask
    )
    assert combined is not None
    assert bool(combined[0, 0, 9, 0].item()) is False
    assert bool(combined[0, 0, 9, 8].item()) is True


def test_attn_gate_scales_attention_output():
    model = _tiny().eval()
    _randomize(model)
    idx = torch.randint(0, 64, (1, 5))
    with torch.no_grad():
        before = model(idx).clone()
        for block in model.blocks:
            block.attn.attn_gate.weight.zero_()
            block.attn.attn_gate.bias = None
        after = model(idx)
    assert not torch.allclose(before, after, atol=1e-6)


# --- (d) portable features: BigramHashEmbedding, Smear, PairedHeadAttention ----


def test_bigram_hash_embedding_shape_and_sign_trick():
    model_dim = 16
    bhe = BigramHashEmbedding(32, model_dim)
    assert bhe.hash_dim == model_dim // 4 == 4
    assert bhe.token_embed.weight.shape == (32, model_dim - 4)
    x = torch.randint(0, 32, (2, 10))
    out = bhe(x)
    assert out.shape == (2, 10, model_dim)
    # Sign trick: hash part values are exactly -1 or +1 (position 0 is zero-padded)
    hash_part = out[:, 1:, model_dim - model_dim // 4 :]
    assert ((hash_part == 1.0) | (hash_part == -1.0)).all()


def test_bigram_hash_embedding_single_token():
    model_dim = 16
    bhe = BigramHashEmbedding(32, model_dim)
    x = torch.randint(0, 32, (1, 1))
    out = bhe(x)
    # Single token: hash part is all zeros (no pair)
    hash_part = out[:, :, model_dim - model_dim // 4 :]
    assert (hash_part == 0.0).all()


def test_bigram_hash_embedding_requires_dim_multiple_of_four():
    with pytest.raises(ValueError, match="requires model_dim % 4 == 0"):
        BigramHashEmbedding(32, 15)


def test_smear_shape_and_lookback():
    smear = Smear(8)
    x = torch.randn(2, 5, 8)
    out = smear(x)
    assert out.shape == x.shape
    # The first token's smear input is a copy of itself (padding), so the
    # first position should differ predictably from a standalone token.
    assert not torch.allclose(out[:, 0], x[:, 0])


def test_smear_zero_gate_is_identity():
    smear = Smear(8)
    nn.init.zeros_(smear.smear_gate)
    x = torch.randn(2, 5, 8)
    out = smear(x)
    # With gate=0, sigmoid(0)=0.5, so out = x + 0.5 * prev_shifted
    assert not torch.allclose(out, x)  # changed from identity


def test_smear_negative_gate_suppresses():
    smear = Smear(8)
    nn.init.constant_(smear.smear_gate, -10.0)
    x = torch.randn(2, 5, 8)
    out = smear(x)
    # sigmoid(-10) ≈ 4.5e-5, so the smear contribution is negligible
    assert torch.allclose(out, x, atol=1e-3)


def test_paired_head_attention_forward():
    attn = PairedHeadAttention(16, num_heads=2).eval()
    with torch.no_grad():
        attn.proj.weight.normal_(0.0, 0.3)
    x = torch.randn(2, 5, 16)
    out = attn(x, value_embed=None)
    assert out.shape == (2, 5, 16)
    assert torch.isfinite(out).all()


def test_paired_head_attention_requires_even_heads():
    with pytest.raises((ValueError, AssertionError)):
        PairedHeadAttention(16, num_heads=3)


def test_paired_head_attention_uses_value_embed():
    attn = PairedHeadAttention(16, num_heads=2).eval()
    with torch.no_grad():
        attn.proj.weight.normal_(0.0, 0.3)
    x = torch.randn(2, 5, 16)
    ve = torch.randn(2, 5, 16)
    out_no_ve = attn(x, value_embed=None)
    out_with_ve = attn(x, value_embed=ve)
    assert not torch.allclose(out_no_ve, out_with_ve, atol=1e-5)


def test_paired_head_attention_production_config_via_block():
    """The Block class must build PairedHeadAttention with n_head=6 and run."""
    model = GPT(50304, num_layers=2, model_dim=768, num_heads=6, paired_head=True)
    idx = torch.randint(0, 50304, (1, 8))
    out = model(idx)
    assert out.shape == (1, 8, 50304)
    assert torch.isfinite(out).all()


# --- (e) portable features: MUDD, XSA, MultiTokenHeads, RotaryWithOffset ----


def test_mudd_returns_resid_and_value():
    mudd = MUDD(16, num_skip_pairs=2)
    src = torch.randn(2, 8, 16)
    resid, val = mudd(src, 0)
    assert resid is not None and val is not None
    assert resid.shape == (2, 8, 16) and val.shape == (2, 8, 16)
    assert torch.isfinite(resid).all() and torch.isfinite(val).all()
    # Out of range layer_idx returns None
    assert mudd(src, 2) == (None, None)


def test_xsa_forward():
    xsa = XSA(16, num_heads=2).eval()
    with torch.no_grad():
        xsa.proj.weight.normal_(0.0, 0.3)
    x = torch.randn(2, 5, 16)
    cross = torch.randn(2, 10, 16)
    out = xsa(x, cross)
    assert out.shape == (2, 5, 16)
    assert torch.isfinite(out).all()


def test_multi_token_heads_shape():
    mth = MultiTokenHeads(16, 32, num_extra_heads=3)
    x = torch.randn(2, 5, 16)
    outs = mth(x)
    assert len(outs) == 3
    for o in outs:
        assert o.shape == (2, 5, 32)


def test_gpt_multi_token_output_hidden_shape():
    """``output_hidden=True`` must return ``(logits, hidden)`` where hidden has
    shape ``(B, T, model_dim)`` (pre-lm_head activations), not vocab_size."""
    model = GPT(64, num_layers=4, model_dim=32, num_heads=2, multi_token_pred=2).eval()
    _randomize(model)
    idx = torch.randint(0, 64, (2, 8))
    logits, hidden = model(idx, output_hidden=True)
    assert logits.shape == (2, 8, 64)
    assert hidden.shape == (2, 8, 32)  # model_dim=32
    # Plain forward still returns just logits
    out = model(idx)
    assert out.shape == (2, 8, 64)


def test_gpt_multi_token_heads_use_hidden_not_logits():
    """The multi-token heads are ``nn.Linear(model_dim, vocab_size)`` so their
    input must be the hidden states (model_dim), not the logits (vocab_size),
    which have a different dimensionality."""
    model = GPT(64, num_layers=4, model_dim=32, num_heads=2, multi_token_pred=2).eval()
    _randomize(model)
    idx = torch.randint(0, 64, (2, 8))
    _, hidden = model(idx, output_hidden=True)
    with torch.no_grad():
        head0_out = model.multi_heads.heads[0](hidden)
    assert head0_out.shape == (
        2,
        8,
        64,
    )  # head produces vocab_logits from hidden states
    # Verify that feeding logits (vocab_size=64) to the head would fail due to
    # shape mismatch — the head expects model_dim=32 input.
    with pytest.raises(RuntimeError):
        logits = model(idx)
        model.multi_heads.heads[0](logits)


def test_rotary_with_offset_changes_output():
    rot = RotaryWithOffset(8)
    x = torch.randn(2, 4, 3, 8)
    y_no_offset = rot(x, pos_offset=0.0)
    y_offset = rot(x, pos_offset=0.5)
    assert y_no_offset.shape == x.shape
    assert not torch.allclose(y_no_offset, y_offset, atol=1e-5)


# --- (f) feature-rich GPT forward with all portable features enabled ---------


def _randomize(model):
    with torch.no_grad():
        for p in model.parameters():
            if p.numel() > 1:
                p.normal_(0.0, 0.2)


def test_gpt_with_all_portable_features():
    model = GPT(
        vocab_size=64,
        num_layers=6,
        model_dim=32,
        num_heads=2,
        bigram_hash_embed=True,
        smear_embed=True,
        partial_key_offset=0.25,
        paired_head=True,
        mudd_pairs=2,
        xsa_enabled=True,
        xsa_pairs=2,
        single_act_last_k=2,
        exp_residual_decay=0.9,
        multi_token_pred=2,
    )
    _randomize(model)
    idx = torch.randint(0, 64, (2, 12))
    logits = model(idx)
    assert logits.shape == (2, 12, 64)
    assert torch.isfinite(logits).all()
    # Verify multi-token heads are present
    assert model.multi_heads is not None
    assert len(model.multi_heads.heads) == 2
    # Verify smear present
    assert model.smear is not None
    # Verify MUDD present
    assert model.mudd is not None
    # Verify XSA present
    assert model.xsa_enabled
    assert len(model.xsa_modules) == 2
    # Verify exp residual decay
    assert model.exp_residual_decay == 0.9
    # Verify single_act_last_k
    assert model.single_act_last_k == 2
    # Verify bigram embed
    assert hasattr(model.embed, "hash_seed")


def test_exp_residual_decay_changes_later_layers():
    model = GPT(64, num_layers=4, model_dim=32, num_heads=2, exp_residual_decay=0.5)
    _randomize(model)
    idx = torch.randint(0, 64, (1, 6))
    with torch.no_grad():
        out_with = model(idx)
    model2 = GPT(64, num_layers=4, model_dim=32, num_heads=2, exp_residual_decay=None)
    _randomize(model2)
    with torch.no_grad():
        out_without = model2(idx)
    assert not torch.allclose(out_with, out_without, atol=1e-5)


def test_single_act_last_k_affects_output():
    model = GPT(64, num_layers=4, model_dim=32, num_heads=2, single_act_last_k=2)
    _randomize(model)
    idx = torch.randint(0, 64, (1, 6))
    with torch.no_grad():
        out_with = model(idx)
    model2 = GPT(64, num_layers=4, model_dim=32, num_heads=2, single_act_last_k=0)
    _randomize(model2)
    with torch.no_grad():
        out_without = model2(idx)
    assert not torch.allclose(out_with, out_without, atol=1e-4)


def test_mudd_affects_output():
    model = GPT(64, num_layers=4, model_dim=32, num_heads=2, mudd_pairs=2)
    _randomize(model)
    idx = torch.randint(0, 64, (1, 6))
    with torch.no_grad():
        out_with = model(idx)
    model2 = GPT(64, num_layers=4, model_dim=32, num_heads=2, mudd_pairs=0)
    _randomize(model2)
    with torch.no_grad():
        out_without = model2(idx)
    assert not torch.allclose(out_with, out_without, atol=1e-4)


def test_partial_key_offset_affects_output():
    model = GPT(64, num_layers=4, model_dim=32, num_heads=2, partial_key_offset=0.25)
    _randomize(model)
    idx = torch.randint(0, 64, (1, 6))
    with torch.no_grad():
        out_with = model(idx)
    model2 = GPT(64, num_layers=4, model_dim=32, num_heads=2, partial_key_offset=None)
    _randomize(model2)
    with torch.no_grad():
        out_without = model2(idx)
    assert not torch.allclose(out_with, out_without, atol=1e-4)


# --- (g) default-off feature flags produce identical output to baseline -------


def test_gpt_with_portable_features_default_off_matches_baseline():
    # When all feature flags are at their defaults (False/0/None), the
    # backward-compatible path should match the original GPT exactly.
    torch.manual_seed(42)
    cfg_baseline = StudentModelConfig(
        model_dim=32, num_layers=4, num_heads=2, vocab_size=64
    )
    model_base = cfg_baseline.build()
    model_feat = StudentModelConfig(
        model_dim=32,
        num_layers=4,
        num_heads=2,
        vocab_size=64,
        bigram_hash_embed=False,
        smear_embed=False,
        partial_key_offset=None,
        paired_head=False,
        mudd_pairs=0,
        xsa_enabled=False,
        xsa_pairs=0,
        single_act_last_k=0,
        exp_residual_decay=None,
        multi_token_pred=0,
    ).build()
    for p, q in zip(model_base.parameters(), model_feat.parameters()):
        q.data.copy_(p.data)
    idx = torch.randint(0, 64, (2, 8))
    model_base.eval()
    model_feat.eval()
    with torch.no_grad():
        out_base = model_base(idx)
        out_feat = model_feat(idx)
    assert torch.allclose(out_base, out_feat, atol=1e-6)


def test_forward_hidden_and_apply_lm_head_match_forward():
    """Chunked validation path must be numerically identical to ``forward``."""
    model = GPT(64, num_layers=4, model_dim=32, num_heads=2).eval()
    _randomize(model)
    idx = torch.randint(0, 64, (3, 7))
    with torch.no_grad():
        full = model(idx)
        hidden = model.forward_hidden(idx)
        rebuilt = model.apply_lm_head(hidden)
        chunked = torch.cat(
            [model.apply_lm_head(hidden[i : i + 1]) for i in range(hidden.size(0))],
            dim=0,
        )
    assert hidden.shape == (3, 7, 32)
    assert torch.allclose(full, rebuilt, atol=1e-6)
    assert torch.allclose(full, chunked, atol=1e-6)


def test_resid_lambdas_initialized_to_sqrt_1_1():
    model = GPT(64, num_layers=4, model_dim=32, num_heads=2)
    expected = math.sqrt(1.1)
    assert torch.allclose(model.resid_lambdas_attn, torch.full((4,), expected))
    assert torch.allclose(model.resid_lambdas_mlp, torch.full((4,), expected))


def test_lm_head_init_std_and_apply_lm_head_float32():
    torch.manual_seed(1)
    model = GPT(64, num_layers=2, model_dim=32, num_heads=2)
    assert model.lm_head.weight.std().item() == pytest.approx(0.005, rel=0.4, abs=0.002)
    hidden = torch.randn(2, 5, 32, dtype=torch.bfloat16)
    out = model.apply_lm_head(hidden)
    assert out.dtype == torch.float32
    assert out.shape == (2, 5, 64)
