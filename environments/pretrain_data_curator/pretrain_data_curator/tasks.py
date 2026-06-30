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


class CuratorTask(vf.Task):
    """One curation episode: a goal plus its budget and cutoff metadata.

    ``answer`` is the cutoff date (the v0 reward never used it for scoring; it is
    kept as task-provenance metadata). ``token_budget`` / ``cutoff_date`` are the
    typed per-task overrides the toolset seeds into the manifest/cutoff.
    """

    answer: str
    token_budget: int
    cutoff_date: str


def _goal_prompt(goal: str, cutoff_date: str, token_budget: int) -> str:
    return (
        f"{goal}\n\n"
        f"Only use Hugging Face datasets modified on or before {cutoff_date}. "
        f"Target roughly {token_budget} tokens.\n\n"
        "Discover and inspect candidate datasets by running the `hf` CLI in your "
        "own shell, e.g. `hf datasets ls --search <query> --sort downloads` to find "
        "datasets and `hf datasets info <dataset_id>` to inspect one. Each `hf` "
        "call is metered and counts against your reward, so search and inspect "
        "economically.\n\n"
        "When you are done, output your final selection as a single fenced ```json "
        "block with a non-empty `sources` list: each source an object with `id` "
        "(the Hugging Face dataset id) and `weight`, plus optional `filters`, "
        "`sampling`, `config`, `split`, and `text_field`, and an optional top-level "
        "`token_budget`."
    )


def build_tasks(cutoff_date: str, token_budget: int) -> list[CuratorTask]:
    """One :class:`CuratorTask` per curation goal, with shared budget + cutoff."""
    return [
        CuratorTask(
            idx=i,
            prompt=_goal_prompt(goal, cutoff_date, token_budget),
            answer=cutoff_date,
            token_budget=token_budget,
            cutoff_date=cutoff_date,
        )
        for i, goal in enumerate(_GOALS)
    ]
