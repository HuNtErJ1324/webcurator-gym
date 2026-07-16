"""CPU tests for the modded-nanogpt speedrun optimizer port."""

from __future__ import annotations

import pytest
import torch

from pretrain_curation_gym.gpu.train_gpt import StudentModelConfig, GPT
from pretrain_curation_gym.gpu.train_gpt import (
    Muon,
    build_batch_schedule,
    build_speedrun_optimizers,
    classify_speedrun_params,
    get_muon_momentum,
    init_speedrun_weights,
    lookup_batch_stage,
    schedule_lr_multiplier,
    step_speedrun_optimizers,
    zeropower_via_newtonschulz5,
    zeropower_via_polar_express,
    muon_update_normalized,
)


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
    assert get_muon_momentum(
        0, 100, warmup_steps=10, cooldown_steps=10
    ) == pytest.approx(0.85)
    assert get_muon_momentum(
        50, 100, warmup_steps=10, cooldown_steps=10
    ) == pytest.approx(0.95)
    assert get_muon_momentum(95, 100, warmup_steps=10, cooldown_steps=10) < 0.95


def test_classify_and_step_speedrun_optimizers():
    model = StudentModelConfig(
        model_dim=32, num_layers=2, num_heads=2, vocab_size=64, num_value_embeds=1
    ).build()
    muon_params, adam = classify_speedrun_params(model)
    assert muon_params
    assert (
        adam["embed"] and adam["lm_head"] and adam["value_embeds"] and adam["scalars"]
    )
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
    step_speedrun_optimizers(
        muon_opt, adam_opt, step=1, muon_momentum=0.95, cautious_wd=True, lr_scale=0.5
    )


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
        return step_speedrun_optimizers(
            muon_opt, adam_opt, step=step, muon_momentum=0.95, **kw
        )

    # Even step, default adam_on_odd_steps=True: Adam is skipped.
    assert run_step(0) is False
    # Odd step: Adam steps.
    assert run_step(1) is True
    # Even step but force_adam=True (used by the accumulation flush): Adam steps.
    assert run_step(0, force_adam=True) is True
    # adam_on_odd_steps=False: Adam steps every step, even on an even step.
    assert run_step(0, adam_on_odd_steps=False) is True


def test_optimizer_specific_clip_cadence_preserves_raw_adam_sum(monkeypatch):
    model = GPT(vocab_size=64, num_layers=2, model_dim=32, num_heads=2)
    muon_opt, adam_opt = build_speedrun_optimizers(model)
    muon_param = muon_opt.param_groups[0]["params"][0]
    adam_param = adam_opt.param_groups[0]["params"][0]
    clips: list[tuple[set[int], torch.Tensor | None]] = []
    real_clip = torch.nn.utils.clip_grad_norm_

    def spy_clip(params, max_norm, *args, **kwargs):
        params = list(params)
        clips.append(
            (
                {id(param) for param in params},
                adam_param.grad.detach().clone()
                if any(param is adam_param for param in params)
                else None,
            )
        )
        return real_clip(params, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy_clip)
    muon_param.grad = torch.full_like(muon_param, 10.0)
    adam_param.grad = torch.full_like(adam_param, 2.0)
    assert not step_speedrun_optimizers(
        muon_opt, adam_opt, step=0, muon_momentum=0.95, grad_clip=1.0
    )
    assert len(clips) == 1 and id(adam_param) not in clips[0][0]
    assert torch.equal(adam_param.grad, torch.full_like(adam_param, 2.0))
    assert adam_param not in adam_opt.state

    muon_param.grad = torch.full_like(muon_param, 10.0)
    adam_param.grad.add_(3.0)
    assert step_speedrun_optimizers(
        muon_opt, adam_opt, step=1, muon_momentum=0.95, grad_clip=1.0
    )
    assert len(clips) == 3  # Muon on both steps, Adam only on the odd update.
    assert torch.equal(clips[-1][1], torch.full_like(adam_param, 5.0))
    assert adam_param in adam_opt.state


def test_adam_on_every_step_clips_and_updates_on_even_step(monkeypatch):
    model = GPT(vocab_size=64, num_layers=2, model_dim=32, num_heads=2)
    muon_opt, adam_opt = build_speedrun_optimizers(model)
    for optimizer in (muon_opt, adam_opt):
        for group in optimizer.param_groups:
            for param in group["params"]:
                param.grad = torch.ones_like(param)
    calls = 0
    real_clip = torch.nn.utils.clip_grad_norm_

    def spy_clip(params, max_norm, *args, **kwargs):
        nonlocal calls
        calls += 1
        return real_clip(params, max_norm, *args, **kwargs)

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", spy_clip)
    assert step_speedrun_optimizers(
        muon_opt,
        adam_opt,
        step=0,
        muon_momentum=0.95,
        adam_on_odd_steps=False,
        grad_clip=1.0,
    )
    assert calls == 2
    assert adam_opt.state


def test_build_speedrun_optimizers_defaults_and_per_group_betas():
    model = StudentModelConfig(
        model_dim=32, num_layers=2, num_heads=2, vocab_size=64, num_value_embeds=1
    ).build()
    muon_opt, adam_opt = build_speedrun_optimizers(model)
    assert muon_opt.param_groups[0]["weight_decay"] == pytest.approx(1.2)
    assert all(g.get("nor_muon", False) is True for g in muon_opt.param_groups)
    betas = {tuple(g["betas"]) for g in adam_opt.param_groups}
    assert (0.5, 0.95) in betas
    assert (0.75, 0.95) in betas
    assert (0.9, 0.99) in betas
    # Group order is embed, lm_head, value_embeds, scalars when all present.
    assert adam_opt.param_groups[0]["betas"] == (0.5, 0.95)
    assert adam_opt.param_groups[1]["betas"] == (0.5, 0.95)
    assert adam_opt.param_groups[2]["betas"] == (0.75, 0.95)
    assert adam_opt.param_groups[3]["betas"] == (0.9, 0.99)


def test_muon_momentum_buffer_is_float32_for_bfloat16_params():
    param = torch.nn.Parameter(torch.randn(8, 4, dtype=torch.bfloat16))
    opt = Muon([param], lr=0.02, weight_decay=0.0, nor_muon=True)
    param.grad = torch.randn_like(param)
    opt.step(momentum=0.95)
    state = opt.state[param]
    assert state["momentum_buffer"].dtype == torch.float32
    assert param.dtype == torch.bfloat16


def test_init_speedrun_weights_lm_head_proj_and_value_embeds():
    torch.manual_seed(0)
    model = StudentModelConfig(
        model_dim=32, num_layers=2, num_heads=2, vocab_size=64, num_value_embeds=1
    ).build()
    # Force a known pre-init so post-init asserts are meaningful.
    with torch.no_grad():
        model.lm_head.weight.fill_(1.0)
        model.blocks[0].attn.proj.weight.fill_(1.0)
        model.value_embeds.embed[0].weight.fill_(1.0)
    init_speedrun_weights(model)
    assert model.blocks[0].attn.proj.weight.abs().sum().item() == 0.0
    assert model.lm_head.weight.abs().sum().item() > 0.0
    assert model.lm_head.weight.std().item() == pytest.approx(
        0.005, rel=0.35, abs=0.002
    )
    ve_std = model.value_embeds.embed[0].weight.std().item()
    assert ve_std == pytest.approx(0.01, rel=0.35, abs=0.004)
