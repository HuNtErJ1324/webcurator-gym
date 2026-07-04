"""Curation task definitions.

Under verifiers v1 a task is a typed, frozen ``vf.Task`` rather than a row in an
HF ``Dataset``. ``CuratorTask`` carries the per-episode goal as its ``prompt``
plus the typed ``token_budget`` / ``cutoff_date`` that used to ride in the v0
``info`` dict (so per-task overrides are typed and round-trip into rollout state
without the JSON-string-vs-dict ambiguity the v0 env had to defend against).
"""

from __future__ import annotations

import verifiers.v1 as vf

_GOALS = [
    "Curate a general-purpose pretraining mixture that teaches broad world "
    "knowledge and clean prose to a small GPT-2-scale student.",
    "Curate a pretraining mixture emphasizing reasoning, math, and code while "
    "keeping enough natural language for fluent generation.",
    "Curate a diverse pretraining mixture spanning encyclopedic, scientific, and "
    "instructional text with strong deduplication and quality filtering.",
    "Curate a compact, high-quality pretraining mixture that maximizes student "
    "performance per training token under a tight budget.",
]

TASK_PROMPT = """Your goal is: {goal}

## Objective
Produce the manifest that gives the fixed proxy student the strongest held-out performance within this rollout's data and discovery budgets.

## Autonomy & Exploration
- You have complete freedom in data-source choice, mixture weights, and filters.
- Iteration is encouraged. Use the shell to gather enough evidence, then stop when another call is unlikely to improve the mixture.

## Information on the Setup
- The target mixture budget is {token_budget} tokens.
- You have {max_turns} model turns. A response that invokes the shell uses one turn even if it contains multiple tool calls; every individual `hf` call is still billed.
- A discovery round is one `hf datasets ls` call and one `hf datasets info` call. Prefer one `hf` call per turn and use at most {discovery_rounds} rounds (<={discovery_calls} shell calls).
- Local sources are {local_source_status}.
- Scoring trains the fixed student and measures held-out cross-entropy. The reward is `{alpha_perf} * performance - {lambda_cost} * discovery/data cost - {lambda_leakage} * leakage`.

## Rules
1. Use only datasets modified on or before {cutoff_date}.
2. Do not use, copy, or derive data from the environment's fixed held-out validation set or evaluation corpus; those are reserved exclusively for scoring.
3. There will be no user interaction. Operate autonomously.
4. When you have enough evidence, commit immediately; do not fill remaining turns with more discovery. Commit no later than turn {commit_by}, before the turn limit: stop running commands and return only the fenced JSON manifest as a plain response. Do not print it through the shell.

Remember: the source choices are yours. Explore economically and commit the best manifest you can support with observed evidence."""


class CuratorTask(vf.Task):
    """One curation episode: a goal plus its budget and cutoff metadata.

    ``answer`` is the cutoff date (the v0 reward never used it for scoring; it is
    kept as task-provenance metadata). ``token_budget`` / ``cutoff_date`` are the
    typed per-task overrides the toolset seeds into the manifest/cutoff.
    """

    answer: str
    token_budget: int
    cutoff_date: str


def _goal_prompt(
    goal: str,
    cutoff_date: str,
    token_budget: int,
    *,
    max_turns: int,
    discovery_rounds: int,
    discovery_calls: int,
    commit_by: int,
    allow_local_sources: bool,
    alpha_perf: float,
    lambda_cost: float,
    lambda_leakage: float,
) -> str:
    return TASK_PROMPT.format(
        goal=goal,
        cutoff_date=cutoff_date,
        token_budget=token_budget,
        max_turns=max_turns,
        discovery_rounds=discovery_rounds,
        discovery_calls=discovery_calls,
        commit_by=commit_by,
        local_source_status=(
            "enabled for workspace-relative plain-text or JSONL files"
            if allow_local_sources
            else "disabled; use only Hugging Face sources"
        ),
        alpha_perf=alpha_perf,
        lambda_cost=lambda_cost,
        lambda_leakage=lambda_leakage,
    )


def build_tasks(
    cutoff_date: str,
    token_budget: int,
    *,
    max_turns: int = 12,
    discovery_rounds: int = 2,
    discovery_calls: int = 4,
    commit_by: int = 9,
    allow_local_sources: bool = True,
    alpha_perf: float = 1.0,
    lambda_cost: float = 0.1,
    lambda_leakage: float = 1.0,
    system_prompt: str | None = None,
) -> list[CuratorTask]:
    """One :class:`CuratorTask` per curation goal, with shared budget + cutoff."""
    return [
        CuratorTask(
            idx=i,
            prompt=_goal_prompt(
                goal,
                cutoff_date,
                token_budget,
                max_turns=max_turns,
                discovery_rounds=discovery_rounds,
                discovery_calls=discovery_calls,
                commit_by=commit_by,
                allow_local_sources=allow_local_sources,
                alpha_perf=alpha_perf,
                lambda_cost=lambda_cost,
                lambda_leakage=lambda_leakage,
            ),
            system_prompt=system_prompt,
            answer=cutoff_date,
            token_budget=token_budget,
            cutoff_date=cutoff_date,
        )
        for i, goal in enumerate(_GOALS)
    ]
