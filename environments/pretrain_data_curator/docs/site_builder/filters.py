from __future__ import annotations

from typing import Any

FULL_400M_TOKEN_BUDGET = 400_000_000


def is_full_400m_eval(
    *,
    token_budget: int | None,
    use_real_trainer: bool | None,
    config: dict[str, Any],
) -> bool:
    """True for completed 400M-token runs with the real GPU proxy trainer."""
    if token_budget != FULL_400M_TOKEN_BUDGET:
        return False
    if use_real_trainer is not True:
        return False
    args = config.get("args") or {}
    if not isinstance(args, dict):
        return False
    proxy = args.get("proxy_student") or {}
    if isinstance(proxy, dict):
        train_budget = proxy.get("train_token_budget")
        if train_budget is not None and int(train_budget) != FULL_400M_TOKEN_BUDGET:
            return False
    return True
