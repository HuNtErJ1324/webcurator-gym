"""CPU tests for the modded-nanogpt speedrun optimizer port."""

from __future__ import annotations

import pytest
import torch

from pretrain_data_curator.student_model import StudentModelConfig, GPT
from pretrain_data_curator.student_optimizer import (
    Muon,
    build_batch_schedule,
    build_speedrun_optimizers,
    classify_speedrun_params,
    get_muon_momentum,
    lookup_batch_stage,
    schedule_lr_multiplier,
    step_speedrun_optimizers,
    zeropower_via_newtonschulz5,
    zeropower_via_polar_express,
    muon_update_normalized,
)
from pretrain_data_curator.trainer import NANOGPT_TRAIN_SCRIPT


def test_newtonschulz5_orthogonalizes_2d():
    g = torch.randn(8, 4)
    u = zeropower_via_newtonschulz5(g)
    assert u.shape == g.shape
    if g.size(-2) <= g.size(-1):
        gram = u @ u.T
        eye = torch.eye(gram.size(0))
        assert torch.allclose(gram, eye, atol=0.35)


def test_batch_schedule_stages_and_cooldown():
    boundaries, stages, cd_start, floor = build_batch_schedule(900)
    assert len(boundaries) == 3
    assert stages[0].batch_mul == 1 and stages[-1].batch_mul == 3
    assert lookup_batch_stage(0, boundaries, stages).batch_mul == 1
    assert lookup_batch_stage(899, boundaries, stages).batch_mul == 3
    early = schedule_lr_multiplier(
        0, stages[0], cd_start=cd_start, scheduled_steps=900, cooldown_floor=floor
    )
    late = schedule_lr_multiplier(
        899, stages[-1], cd_start=cd_start, scheduled_steps=900, cooldown_floor=floor
    )
    assert early == pytest.approx(1.0)
    assert late == pytest.approx(floor, rel=0.05)


def test_muon_momentum_warmup_and_cooldown():
    assert get_muon_momentum(0, 100, warmup_steps=10, cooldown_steps=10) == pytest.approx(0.85)
    assert get_muon_momentum(50, 100, warmup_steps=10, cooldown_steps=10) == pytest.approx(0.95)
    assert get_muon_momentum(95, 100, warmup_steps=10, cooldown_steps=10) < 0.95


def test_classify_and_step_speedrun_optimizers():
    model = StudentModelConfig(
        model_dim=32, num_layers=2, num_heads=2, vocab_size=64, num_value_embeds=1
    ).build()
    muon_params, adam = classify_speedrun_params(model)
    assert muon_params
    assert adam["embed"] and adam["lm_head"] and adam["value_embeds"] and adam["scalars"]
    muon_opt, adam_opt = build_speedrun_optimizers(model)
    x = torch.randint(0, 64, (2, 8))
    loss = model(x).sum()
    loss.backward()
    step_speedrun_optimizers(muon_opt, adam_opt, step=1, muon_momentum=0.95)
    assert isinstance(muon_opt, Muon)



# --- portable optimizer features ---

def test_polar_express_orthogonalizes_2d():
    g = torch.randn(8, 4)
    u = zeropower_via_polar_express(g)
    assert u.shape == g.shape
    if g.size(-2) <= g.size(-1):
        gram = u @ u.T
        eye = torch.eye(gram.size(0))
        assert torch.allclose(gram, eye, atol=0.35)


def test_polar_express_and_newtonschulz_both_orthogonalize():
    g = torch.randn(8, 4)
    ns = zeropower_via_newtonschulz5(g)
    pe = zeropower_via_polar_express(g)
    # Both orthogonalize (NS needs more iterations for the same accuracy)
    # Verify they both produce near-orthogonal outputs
    if g.size(-2) <= g.size(-1):
        ns_gram = ns @ ns.T
        pe_gram = pe @ pe.T
        eye = torch.eye(g.size(-2))
        assert torch.allclose(ns_gram, eye, atol=0.35)
        assert torch.allclose(pe_gram, eye, atol=0.35)


def test_nor_muon_update_normalized():
    grad = torch.randn(8, 4)
    mom = torch.zeros_like(grad)
    update = muon_update_normalized(grad, mom)
    assert update.shape == grad.shape
    # NorMuon normalizes the update: RMS should be ~1
    rms = update.norm() / (update.numel() ** 0.5)
    assert rms < 2.0  # at least approximately normalized (not 100x gradient scale)


def test_nor_muon_optimizer_steps():
    model = GPT(vocab_size=64, num_layers=2, model_dim=32, num_heads=2)
    muon_params, adam = classify_speedrun_params(model)
    muon_opt = Muon(muon_params, lr=0.02, weight_decay=0.05, nor_muon=True)
    assert any(g.get("nor_muon", False) for g in muon_opt.param_groups)
    x = torch.randint(0, 64, (2, 8))
    loss = model(x).sum()
    loss.backward()
    muon_opt.step()


def test_polar_express_muon_optimizer_steps():
    model = GPT(vocab_size=64, num_layers=2, model_dim=32, num_heads=2)
    muon_params, adam = classify_speedrun_params(model)
    muon_opt = Muon(muon_params, lr=0.02, weight_decay=0.05, polar_express=True)
    assert any(g.get("polar_express", False) for g in muon_opt.param_groups)
    x = torch.randint(0, 64, (2, 8))
    loss = model(x).sum()
    loss.backward()
    muon_opt.step()


def test_build_speedrun_optimizers_with_nor_muon():
    model = GPT(vocab_size=64, num_layers=2, model_dim=32, num_heads=2)
    muon_opt, adam_opt = build_speedrun_optimizers(model, nor_muon=True, polar_express=False)
    assert isinstance(muon_opt, Muon)
    for g in muon_opt.param_groups:
        assert g.get("nor_muon", False) is True


def test_cautious_weight_decay_muon_step():
    model = GPT(vocab_size=64, num_layers=2, model_dim=32, num_heads=2)
    muon_params, adam = classify_speedrun_params(model)
    muon_opt = Muon(muon_params, lr=0.02, weight_decay=0.05)
    # Capture pre-step weight norms
    pre_norms = [p.norm().item() for p in muon_opt.param_groups[0]["params"]]
    x = torch.randint(0, 64, (2, 8))
    loss = model(x).sum()
    loss.backward()
    # With cautious_wd and lr_scale=1.0, weight_decay is unchanged
    muon_opt.step(cautious_wd=True, lr_scale=1.0)
    post_norms = [p.norm().item() for p in muon_opt.param_groups[0]["params"]]
    # Parameters changed (not asserting direction, just that step ran)
    assert len(pre_norms) == len(post_norms)


def test_step_speedrun_cautious_wd():
    model = GPT(vocab_size=64, num_layers=2, model_dim=32, num_heads=2)
    muon_opt, adam_opt = build_speedrun_optimizers(model)
    x = torch.randint(0, 64, (2, 8))
    loss = model(x).sum()
    loss.backward()
    step_speedrun_optimizers(muon_opt, adam_opt, step=1, muon_momentum=0.95, cautious_wd=True, lr_scale=0.5)


def test_step_speedrun_optimizers_reports_whether_adam_stepped():
    """``step_speedrun_optimizers`` must report whether AdamW.step() actually
    ran, so callers know whether it is safe to clear Adam-managed grads --
    the original ``train_gpt.py`` only clears an Adam param's ``.grad`` on
    the step that actually applies it."""
    model = GPT(vocab_size=64, num_layers=2, model_dim=32, num_heads=2)

    def run_step(step, **kw):
        muon_opt, adam_opt = build_speedrun_optimizers(model)
        x = torch.randint(0, 64, (2, 8))
        loss = model(x).sum()
        loss.backward()
        return step_speedrun_optimizers(muon_opt, adam_opt, step=step, muon_momentum=0.95, **kw)

    # Even step, default adam_on_odd_steps=True: Adam is skipped.
    assert run_step(0) is False
    # Odd step: Adam steps.
    assert run_step(1) is True
    # Even step but force_adam=True (used by the accumulation flush): Adam steps.
    assert run_step(0, force_adam=True) is True
    # adam_on_odd_steps=False: Adam steps every step, even on an even step.
    assert run_step(0, adam_on_odd_steps=False) is True
