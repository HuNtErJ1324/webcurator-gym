"""Speedrun optimizer recipe (KellerJordan/modded-nanogpt), CPU/single-device port.

Ports the *training* optimizations from ``train_gpt.py`` that do not require custom
GPU kernels, distributed comms, FlexAttention, FP8, or ``torch.compile``:

* **Classic Muon** (Newton–Schulz orthogonalization) on 2-D block weights.
* **Polar Express** (ONI-based polar decomposition) as an alternative orthogonalizer.
* **NorMuon** — normalized Muon with RMS-normalized updates.
* **AdamW** on embeddings, the LM head, value embeddings, and scalar params.
* **Heterogeneous stepping** — Adam updates only on odd steps (Muon every step).
* **Batch-size schedule** — 1× → 2× → 3× micro-batch over three equal stages.
* **LR schedule** — stage LR multipliers tied to batch growth, plus a linear
  cooldown to a floor (defaults mirror the BatchSizeSchedule record).
* **Muon momentum warmup/cooldown** (linear ramp over the run ends).
* **Cautious weight decay** — weight decay multiplied by the current LR scale.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass

import torch

from .batch_schedule import batch_stage_boundaries


def zeropower_via_newtonschulz5(g: torch.Tensor) -> torch.Tensor:
    """Orthogonalize a 2-D update via Newton–Schulz (Muon core step).

    Runs in fp32 on CPU or GPU; the upstream record uses bf16 on GPU, but the math
    is identical. Adapted from ``records/track_3_optimization/train_gpt_simple.py``.
    """
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
    """Polar decomposition via ONI iteration variant (Polar Express).

    Uses ``X_{k+1} = 0.5 * X_k @ (3*I - X_k^T @ X_k)`` for fast convergence
    on near-orthogonal matrices. CPU-portable and operates in fp32.
    The key difference from Newton-Schulz is the transposed quadratic form:
    ``X^T @ X`` (n×n) instead of ``X @ X^T`` (m×m), making it more efficient
    when n << m.
    """
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
    """One Muon momentum + orthogonalization step (single tensor, no torch.compile).

    When ``polar_express=True``, uses ONI-based polar decomposition instead of
    Newton–Schulz iteration.

    Momentum and orthogonalization run in float32 for stability even when the
    parameter/grad tensors are bfloat16 on CUDA.
    """
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
    """NorMuon: normalized Muon update (RMS-normalize the update before applying).

    First computes the standard Muon update, then RMS-normalizes it so that the
    update magnitude is decoupled from the matrix scale.
    """
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
                    # Keep Muon momentum in float32 even when params are bf16.
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
    cooldown_frac: float = 0.60,
    cooldown_floor: float = 0.15,
) -> tuple[list[tuple[int, int]], BatchScheduleStage, float, float]:
    """Return stage boundaries, stage lookup metadata, cooldown start, and floor.

    Stage boundaries are ``(start_step, end_step)`` half-open intervals covering
    ``[0, total_steps)``, computed by ``batch_schedule.batch_stage_boundaries``
    -- the single canonical implementation shared with the token-budget
    accounting path, so the two can never drift apart. ``cooldown_frac``
    applies to the *scheduled* portion (all but the optional extension steps —
    we have none in the proxy trainer).
    """
    if len(stage_fracs) != len(batch_muls) or len(stage_fracs) != len(lr_muls):
        raise ValueError("stage_fracs, batch_muls, and lr_muls must have equal length")
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
    """Split ``student_model.GPT`` params into Muon matrices vs AdamW groups."""
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
            # Skip if same tensor as embed.weight (weight tying for untie_at_frac).
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
    # --- portable optimizer features ---
    nor_muon: bool = True,
    polar_express: bool = False,
):
    """Build Muon + multi-group AdamW optimizers for ``student_model.GPT``.

    When ``nor_muon=True``, Muon uses normalized (RMS-normalized) updates.
    When ``polar_express=True``, Muon uses ONI-based polar decomposition.

    Adam betas are set per group to match the modern speedrun recipe:
    embed/lm_head ``(0.5, 0.95)``, value embeddings ``(0.75, 0.95)``,
    scalars ``(0.9, 0.99)``.
    """
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
    """Weight init aligned with modded-nanogpt block matrices (``train_gpt_simple``).

    Projection matrices stay zero-init; ``lm_head`` uses ``N(0, 0.005)``; value
    embeddings use ``N(0, 0.01)``; token embeddings use ``N(0, 0.02)``.
    """
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
    for group, base_lr in zip(adam_opt.param_groups, initial_adam_lrs, strict=False):
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
    """Muon every step; AdamW only on odd steps (heterogeneous batching record).

    When ``cautious_wd=True``, weight decay on Muon groups is scaled by
    ``lr_scale`` (cautious weight decay tied to LR).

    When ``force_adam=True``, AdamW always steps regardless of step parity
    (used during flush to avoid dropping accumulated embed/head grads).

    Returns whether ``adam_opt.step()`` actually ran this call, so callers
    know whether it is safe to zero Adam-managed grads (the original
    ``train_gpt.py`` only clears an Adam param's ``.grad`` on the step that
    actually applies it -- on a skipped step the grad is left alone so the
    next ``backward()`` accumulates on top of it).
    """
    clip_optimizer_grads(muon_opt, grad_clip)
    muon_opt.step(momentum=muon_momentum, cautious_wd=cautious_wd, lr_scale=lr_scale)
    did_adam_step = force_adam or not adam_on_odd_steps or step % 2 == 1
    if did_adam_step:
        # The retained even-step Adam gradients remain raw until this update;
        # clipping happens once, after the odd contribution has been summed.
        clip_optimizer_grads(adam_opt, grad_clip)
        adam_opt.step()
    return did_adam_step


_OPTIMIZER_COMPONENTS = (
    zeropower_via_newtonschulz5,
    zeropower_via_polar_express,
    muon_update,
    muon_update_normalized,
    Muon,
    BatchScheduleStage,
    build_batch_schedule,
    lookup_batch_stage,
    schedule_lr_multiplier,
    get_muon_momentum,
    classify_speedrun_params,
    build_speedrun_optimizers,
    init_speedrun_weights,
    set_optimizer_lrs,
    capture_initial_lrs,
    clip_optimizer_grads,
    step_speedrun_optimizers,
)


def optimizer_source() -> str:
    """Verbatim optimizer helpers for embedding into the sandbox training script."""
    return "\n\n\n".join(inspect.getsource(c).rstrip() for c in _OPTIMIZER_COMPONENTS)
