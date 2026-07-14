"""Correctness tests for the token-budget batch-schedule accounting helpers.

Covers (1) the non-monotonicity of ``scheduled_presentation_tokens`` and the
provably-sufficient bounded search ``steps_for_token_budget`` uses instead of
a (unsound) binary search, via an exact counterexample plus brute-force
property tests, and (2) exact stage-boundary parity between the accounting
helper and the runtime scheduler (``train_gpt.build_batch_schedule``),
which now share one canonical implementation.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

import pytest

from pretrain_data_curator.train_gpt import (
    batch_stage_boundaries,
    scheduled_presentation_tokens,
    steps_for_token_budget,
)
from pretrain_data_curator.train_gpt import build_batch_schedule


def _tokens(
    n: int,
    fracs: Sequence[float],
    muls: Sequence[int],
    batch_size: int = 1,
    block_size: int = 1,
    enabled: bool = True,
) -> int:
    return scheduled_presentation_tokens(
        n,
        batch_size=batch_size,
        block_size=block_size,
        batch_stage_muls=muls,
        batch_stage_fracs=fracs,
        batch_schedule_enabled=enabled,
    )


def _min_steps(
    budget: int,
    fracs: Sequence[float],
    muls: Sequence[int],
    batch_size: int = 1,
    block_size: int = 1,
    enabled: bool = True,
) -> int:
    return steps_for_token_budget(
        budget,
        batch_size=batch_size,
        block_size=block_size,
        batch_stage_muls=muls,
        batch_stage_fracs=fracs,
        batch_schedule_enabled=enabled,
    )


def _brute_force_min_steps(
    budget: int,
    fracs: Sequence[float],
    muls: Sequence[int],
    batch_size: int = 1,
    block_size: int = 1,
    cap: int = 500_000,
) -> int:
    """Reference (slow but obviously correct) linear scan from N=1 upward."""
    for n in range(1, cap + 1):
        if _tokens(n, fracs, muls, batch_size, block_size) >= budget:
            return n
    raise AssertionError(f"brute force did not converge within cap={cap}")


# --- 1) non-monotonicity: exact counterexample ------------------------------

_CE_FRACS = (1 / 3, 1 / 3, 1 / 3)
_CE_MULS = (1, 2, 4)


def test_scheduled_presentation_tokens_documented_counterexample_is_non_monotone():
    """f(4)=11 > f(5)=10 for equal thirds / muls (1,2,4), per_base=1.

    Adding a step can *reduce* total scheduled tokens: the extra step shifts
    the rounded stage boundary, moving whole steps from the high-mul stage to
    a low-mul one faster than the +1 step itself adds. This is why a plain
    binary search over N is unsound.
    """
    f4 = _tokens(4, _CE_FRACS, _CE_MULS)
    f5 = _tokens(5, _CE_FRACS, _CE_MULS)
    assert f4 == 11
    assert f5 == 10
    assert f5 < f4


def test_steps_for_token_budget_counterexample_returns_true_minimum():
    """budget=11 under this schedule: minimal N is 4, not 6 (old binary search)."""
    n = _min_steps(11, _CE_FRACS, _CE_MULS)
    assert n == 4
    assert _tokens(n, _CE_FRACS, _CE_MULS) >= 11
    assert _tokens(n - 1, _CE_FRACS, _CE_MULS) < 11


@pytest.mark.parametrize("budget", list(range(1, 30)))
def test_steps_for_token_budget_matches_brute_force_across_counterexample_schedule(budget):
    """Exhaustively cross-check every small budget on the exact counterexample
    schedule (where non-monotonicity is known to occur) against brute force."""
    got = _min_steps(budget, _CE_FRACS, _CE_MULS)
    want = _brute_force_min_steps(budget, _CE_FRACS, _CE_MULS)
    assert got == want


def _assert_minimal_and_bounded_overshoot(n, budget, fracs, muls, batch_size, block_size):
    """Core minimality contract: N meets the budget, N-1 does not, and the
    overshoot is strictly less than the token contribution of the specific
    final scheduled step that crossed the threshold (N-1 -> N) -- a direct
    corollary of minimality, not an independent assumption."""
    tokens_n = _tokens(n, fracs, muls, batch_size, block_size)
    assert tokens_n >= budget
    if n > 1:
        tokens_prev = _tokens(n - 1, fracs, muls, batch_size, block_size)
        assert tokens_prev < budget
        final_step_tokens = tokens_n - tokens_prev
        overshoot = tokens_n - budget
        assert overshoot < final_step_tokens


def test_steps_for_token_budget_unit_mismatch_regression_skewed_fractions():
    """p_bound (step*mul units) was being combined directly with budget
    (token units) without scaling by per_base, corrupting the search window
    for large per_base. batch=4, block=67 (per_base=268), muls=(1,1,8,1),
    fracs approx (0.558, 0.113, 0.005, 0.325): true minimal N=16
    (f(16)=6164, f(15)=5896); the unit bug returned 17."""
    raw = (0.558, 0.113, 0.005, 0.325)
    fracs = tuple(f / sum(raw) for f in raw)
    muls = (1, 1, 8, 1)
    n = _min_steps(5974, fracs, muls, batch_size=4, block_size=67)
    assert n == 16
    assert _tokens(16, fracs, muls, batch_size=4, block_size=67) == 6164
    assert _tokens(15, fracs, muls, batch_size=4, block_size=67) == 5896
    _assert_minimal_and_bounded_overshoot(n, 5974, fracs, muls, 4, 67)


def test_steps_for_token_budget_unit_mismatch_regression_large_per_base():
    """Second unit-mismatch regression: batch=26, block=127 (per_base=3302),
    muls=(7,7,1), fracs constructed so the true minimal N=5
    (f(4)=52832, f(5)=56134); the unit bug returned 6."""
    fracs = (0.02, 0.08, 0.9)
    muls = (7, 7, 1)
    n = _min_steps(55351, fracs, muls, batch_size=26, block_size=127)
    assert n == 5
    assert _tokens(4, fracs, muls, batch_size=26, block_size=127) == 52832
    assert _tokens(5, fracs, muls, batch_size=26, block_size=127) == 56134
    _assert_minimal_and_bounded_overshoot(n, 55351, fracs, muls, 26, 127)


# --- 2) broad property / brute-force tests over random schedules -----------


def _random_schedule(rng: random.Random):
    """Random schedule generator. Deliberately includes large block/batch
    sizes (large per_base) and skewed multiplier spreads (occasionally very
    large max/min mul ratios) -- the unit-mismatch bug above only manifests
    when per_base is large enough that the (unscaled) p_bound is negligible
    relative to the mis-derived search window, so small per_base alone does
    not exercise this class of bug."""
    k = rng.choice([2, 3, 4, 5, 6])
    raw_fracs = [rng.random() + 0.001 for _ in range(k)]
    total = sum(raw_fracs)
    fracs = tuple(f / total for f in raw_fracs)
    if rng.random() < 0.3:
        muls = tuple(rng.choice([1, rng.randint(50, 500)]) for _ in range(k))
    else:
        muls = tuple(rng.randint(1, 20) for _ in range(k))
    batch_size = rng.choice([1, 2, 4, 8, 16, 32, 64])
    block_size = rng.choice([1, 8, 67, 127, 256, 1024, 4096])
    return fracs, muls, batch_size, block_size


@pytest.mark.parametrize("trial", range(500))
def test_steps_for_token_budget_matches_brute_force_random_schedules(trial):
    """Random small/moderate schedules + budgets: returned N must equal the
    brute-force minimum exactly (proves minimality, not just a valid budget)."""
    rng = random.Random(trial)
    fracs, muls, batch_size, block_size = _random_schedule(rng)
    budget = rng.randint(1, 5_000)
    got = _min_steps(budget, fracs, muls, batch_size, block_size)
    want = _brute_force_min_steps(budget, fracs, muls, batch_size, block_size)
    assert got == want, (fracs, muls, batch_size, block_size, budget)


@pytest.mark.parametrize("trial", range(150))
def test_steps_for_token_budget_large_budget_properties(trial):
    """Large budgets (brute force from N=1 is infeasible): verify directly
    that the returned N meets the budget, N-1 does not, overshoot is bounded
    by the final scheduled step's tokens, and that no smaller N in a generous
    local window below it is valid either (local minimality safety net)."""
    rng = random.Random(10_000 + trial)
    fracs, muls, batch_size, block_size = _random_schedule(rng)
    budget = rng.randint(10_000_000, 1_000_000_000)
    n = _min_steps(budget, fracs, muls, batch_size, block_size)
    _assert_minimal_and_bounded_overshoot(n, budget, fracs, muls, batch_size, block_size)
    window = min(n - 1, 2000)
    for m in range(max(1, n - window), n):
        assert _tokens(m, fracs, muls, batch_size, block_size) < budget


@pytest.mark.parametrize("trial", range(50))
def test_steps_for_token_budget_tiny_budgets(trial):
    rng = random.Random(20_000 + trial)
    fracs, muls, batch_size, block_size = _random_schedule(rng)
    budget = rng.randint(1, 3)
    got = _min_steps(budget, fracs, muls, batch_size, block_size)
    want = _brute_force_min_steps(budget, fracs, muls, batch_size, block_size)
    assert got == want


def test_steps_for_token_budget_no_schedule_still_ceil_division():
    n = _min_steps(11, _CE_FRACS, _CE_MULS, enabled=False)
    assert n == 11  # ceil(11 / (1*1))


# --- 3) exact stage-boundary parity with the runtime scheduler -------------


def _reference_boundaries(
    total_steps: int, fracs: tuple[float, float, float], muls: tuple[int, int, int]
) -> list[tuple[int, int]]:
    """``build_batch_schedule``'s real contract is exactly 3 stages (matching
    ``ProxyStudentConfig.batch_stage_fracs: tuple[float, float, float]``), so
    the runtime-parity comparison is scoped to k=3."""
    lr_muls = (1.0, 1.52, 1.73)
    boundaries, _, _, _ = build_batch_schedule(
        total_steps, stage_fracs=fracs, batch_muls=muls, lr_muls=lr_muls
    )
    return boundaries


@pytest.mark.parametrize(
    "fracs",
    [
        (1 / 3, 1 / 3, 1 / 3),
        (0.5, 0.25, 0.25),  # exact halves: exercises Python round-half-to-even
        (0.25, 0.5, 0.25),
        (0.1, 0.2, 0.7),  # irregular, float sum-to-1 slop
        (0.6, 0.39, 0.01),
        (0.01, 0.01, 0.98),
    ],
)
@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 50, 51, 100, 101, 12_208])
def test_batch_stage_boundaries_matches_runtime_build_batch_schedule(fracs, n):
    muls = (1, 2, 3)
    assert batch_stage_boundaries(n, fracs) == _reference_boundaries(n, fracs, muls)


@pytest.mark.parametrize("trial", range(200))
def test_batch_stage_boundaries_matches_runtime_random(trial):
    """Random irregular fractions/step counts (k=3, matching the real
    ``build_batch_schedule`` contract), including cases that produce
    zero-length (pre-clamp) intermediate stages for small N."""
    rng = random.Random(30_000 + trial)
    raw = [rng.random() + 0.001 for _ in range(3)]
    total = sum(raw)
    fracs: tuple[float, float, float] = tuple(f / total for f in raw)  # type: ignore[assignment]
    muls: tuple[int, int, int] = tuple(rng.randint(1, 5) for _ in range(3))  # type: ignore[assignment]
    n = rng.choice([1, 2, 3, 4, 5, rng.randint(1, 50), rng.randint(1, 20_000)])
    assert batch_stage_boundaries(n, fracs) == _reference_boundaries(n, fracs, muls)


@pytest.mark.parametrize("trial", range(100))
def test_batch_stage_boundaries_general_k_invariants(trial):
    """Beyond the k=3 runtime contract, ``batch_stage_boundaries`` itself must
    still exactly partition [0, N) for any number of stages (used generically
    by the accounting helpers)."""
    rng = random.Random(40_000 + trial)
    k = rng.choice([2, 3, 4, 5, 6])
    raw = [rng.random() + 0.001 for _ in range(k)]
    total = sum(raw)
    fracs = tuple(f / total for f in raw)
    n = rng.choice([1, 2, 3, 4, 5, rng.randint(1, 50), rng.randint(1, 20_000)])
    boundaries = batch_stage_boundaries(n, fracs)
    assert len(boundaries) == k
    assert boundaries[0][0] == 0
    assert boundaries[-1][1] == n
    for (_start, end), (next_start, _next_end) in zip(boundaries, boundaries[1:]):
        assert end == next_start
    assert all(end >= start for start, end in boundaries)


def test_batch_stage_boundaries_rejects_fracs_not_summing_to_one():
    with pytest.raises(ValueError, match="must sum to 1.0"):
        batch_stage_boundaries(100, (0.5, 0.6))


def test_batch_stage_boundaries_partitions_exactly_even_for_tiny_step_counts():
    """Every intermediate stage's max(1, ...) floor still leaves the total
    partition summing to exactly total_steps, even when total_steps is
    smaller than the number of stages (zero-length final overlap collapses
    via the min(scheduled, ...) clamp rather than double-counting)."""
    fracs = (1 / 3, 1 / 3, 1 / 3)
    for n in range(1, 6):
        boundaries = batch_stage_boundaries(n, fracs)
        assert boundaries[0][0] == 0
        assert boundaries[-1][1] == n
        for (_s1, e1), (s2, _e2) in zip(boundaries, boundaries[1:]):
            assert e1 == s2
        assert all(end >= start for start, end in boundaries)
