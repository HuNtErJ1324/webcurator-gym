"""Token-aware NanoGPT-Speedrun batch schedule accounting.

``train_token_budget`` must limit *actual* scheduled token presentations under
``batch_stage_muls``, not ``steps * base_batch * block``. Stage boundaries are
the single canonical implementation shared with the runtime scheduler
(``student_optimizer.build_batch_schedule`` calls ``batch_stage_boundaries``
below directly rather than duplicating the rounding logic), so accounting and
the actual training run can never drift apart. ``train_microbatch_size`` is
intentionally ignored here — it only splits an already-scheduled effective
batch for memory.

Boundary rule
-------------
When deriving steps from a token budget, choose the **smallest** step count
``N`` such that ``scheduled_presentation_tokens(N) >= budget``. The run may
therefore overshoot the configured budget, but by construction (minimality)
the overshoot is always strictly less than the marginal tokens contributed by
going from ``N - 1`` to ``N`` steps -- i.e. at most one final scheduled step's
worth. If the budget fits exactly on a step boundary, there is no overshoot.

Non-monotonicity of ``scheduled_presentation_tokens``
------------------------------------------------------
Because ``batch_stage_boundaries`` *recomputes* stage boundaries from
scratch for every candidate ``N`` (stage lengths are rounded fractions of
``N`` itself, not a fixed running schedule extended one step at a time),
``scheduled_presentation_tokens(N)`` is **not** monotone non-decreasing in
``N``. Example: equal fracs ``(1/3, 1/3, 1/3)``, muls ``(1, 2, 4)``,
``per_base=1``: ``f(4) = 11`` but ``f(5) = 10`` -- adding a step can *reduce*
total scheduled tokens, because the extra step shifts the rounded boundary
between stages, moving whole steps from a high-mul stage to a low-mul one
faster than the +1 step itself adds. A plain binary search over ``N``
therefore is not sound (it assumes the ">= budget" predicate is sorted).

Provably-sufficient bounded search
-----------------------------------
Each individual stage boundary IS monotone non-decreasing in ``N`` (proof:
``ends[0] = 0``; inductively, ``ends[i](N) = min(N, ends[i-1](N) +
max(1, round(frac_i * N)))`` is a ``min`` of two non-decreasing functions of
``N`` -- ``N`` itself, and a sum of the non-decreasing ``ends[i-1](N)`` with
the non-decreasing ``max(1, round(frac_i * N))`` -- and the min of two
non-decreasing functions is non-decreasing). Writing ``c_i(N) = N *
sum(fracs[:i])`` for the *unrounded* ideal boundary, induction on the same
recursion gives ``|ends[i](N) - c_i(N)| <= i`` (each stage's ``round()`` and
``max(1, ...)`` floor perturbs by at most 1, and the outer ``min(N, ...)``
clamp cannot increase the deviation since ``c_i(N) <= N`` always). Hence
``len_i(N) = ends[i](N) - ends[i-1](N)`` deviates from its ideal
``frac_i * N`` by at most ``2*i - 1`` for ``i < k`` (the last stage inherits
the ``k-1`` bound of ``ends[k-1]``, since ``ends[k] = N`` exactly). Summing
``per_base * mul_i`` over that gives an exact, N-independent bound ``P``
(token units) on how far ``scheduled_presentation_tokens(N)`` can stray from
the smooth linear trend ``per_base * N * weighted_avg_mul``. (Implementation
note: ``_max_stage_deviation_bound`` returns this sum *without* the
``per_base`` factor -- step*mul units -- so callers computing the token-space
bound ``P`` must scale its result by ``per_base``; conflating the two units
was the source of a real bug, since ``budget`` is always tokens.) Combined
with minimality (``f(N) >= budget > f(N-1)``), this places the true minimal
``N`` inside a window of width ``O(P / per_base)`` steps around
``budget / (per_base * weighted_avg_mul)`` -- independent of ``budget`` --
which ``steps_for_token_budget`` scans exhaustively (never assuming the
predicate is sorted).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

_FRAC_SUM_TOL = 1e-6


def batch_stage_boundaries(
    total_steps: int, stage_fracs: Sequence[float]
) -> list[tuple[int, int]]:
    """Half-open ``(start, end)`` stage intervals covering ``[0, total_steps)``.

    Canonical shared implementation: ``student_optimizer.build_batch_schedule``
    calls this directly, so runtime and accounting boundaries can never drift
    apart. Each non-final stage gets ``max(1, round(frac * scheduled))`` steps,
    clamped so the last stage absorbs the remainder exactly.
    """
    frac_sum = sum(stage_fracs)
    if not math.isclose(frac_sum, 1.0, rel_tol=0, abs_tol=_FRAC_SUM_TOL):
        raise ValueError(f"stage_fracs must sum to 1.0, got {frac_sum}")
    scheduled = max(1, int(total_steps))
    ends = [0]
    for frac in stage_fracs[:-1]:
        ends.append(min(scheduled, ends[-1] + max(1, round(frac * scheduled))))
    ends.append(scheduled)
    return [(ends[i], ends[i + 1]) for i in range(len(stage_fracs))]


def scheduled_presentation_tokens(
    steps: int,
    *,
    batch_size: int,
    block_size: int,
    batch_stage_muls: Sequence[int],
    batch_stage_fracs: Sequence[float],
    batch_schedule_enabled: bool = True,
) -> int:
    """Tokens presented across ``steps`` under the staged batch schedule.

    Each step presents ``batch_size * stage_mul * block_size`` tokens. When the
    schedule is disabled, every step uses the base batch (mul = 1).
    """
    steps = max(0, int(steps))
    if steps == 0:
        return 0
    per_base = int(batch_size) * int(block_size)
    if not batch_schedule_enabled:
        return steps * per_base
    if len(batch_stage_fracs) != len(batch_stage_muls):
        raise ValueError("batch_stage_fracs and batch_stage_muls must have equal length")
    total = 0
    for (start, end), mul in zip(
        batch_stage_boundaries(steps, batch_stage_fracs),
        batch_stage_muls,
        strict=True,
    ):
        total += (end - start) * per_base * int(mul)
    return total


def _max_stage_deviation_bound(muls: Sequence[int]) -> int:
    """Exact, N-independent bound ``P`` (in tokens / ``per_base``) on how far
    ``scheduled_presentation_tokens(N) / per_base`` can deviate from the smooth
    linear trend ``N * weighted_avg_mul``. See module docstring for the proof.
    """
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
) -> int:
    """Minimal steps whose scheduled presentations meet ``budget``.

    See module docstring for the non-monotonicity of
    ``scheduled_presentation_tokens`` and the provably-sufficient bounded
    window this scans exhaustively (no assumption that the ">= budget"
    predicate is sorted in ``N``).
    """
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
        raise ValueError("batch_stage_fracs and batch_stage_muls must have equal length")

    muls = [int(m) for m in batch_stage_muls]
    if min(muls) < 1:
        raise ValueError("batch_stage_muls must be >= 1")
    max_mul = max(muls)
    min_mul = min(muls)

    # Exact global bounds, valid regardless of rounding/non-monotonicity: every
    # step contributes at most per_base*max_mul tokens, so no N below n_floor
    # can ever reach the budget; every step contributes at least
    # per_base*min_mul, so n_ceiling always reaches it.
    n_floor = max(1, math.ceil(budget / (per_base * max_mul)))
    n_ceiling = max(1, math.ceil(budget / (per_base * min_mul)))

    mul_avg = sum(float(f) * m for f, m in zip(batch_stage_fracs, muls, strict=True))
    if max_mul == min_mul or mul_avg <= 0:
        # Every step contributes the same tokens: exactly linear, no
        # perturbation possible regardless of stage-boundary rounding.
        return max(1, min(n_ceiling, math.ceil(budget / (per_base * max(mul_avg, min_mul)))))

    # ``_max_stage_deviation_bound`` is in step*mul units (deviation of
    # len_i(N) from its ideal frac_i*N, weighted by mul_i); it must be scaled
    # by per_base to become a token-space bound comparable to ``budget``.
    p_bound_tokens = per_base * _max_stage_deviation_bound(muls)
    # Generous integer safety margin beyond the proven bound, guarding against
    # any float/rounding slop in converting the real-valued inequality window
    # to integer steps -- cheap, since the window stays O(P) either way.
    margin = 4
    lo = math.floor((budget - p_bound_tokens) / (per_base * mul_avg)) - margin
    hi = math.ceil((budget + p_bound_tokens) / (per_base * mul_avg)) + margin
    lo = max(n_floor, lo)
    hi = min(n_ceiling, max(hi, lo))
    lo = min(lo, hi)

    for n in range(lo, hi + 1):
        if tokens_at(n) >= budget:
            return n

    # Unreachable given the proof above; n_ceiling is always valid.
    return n_ceiling
