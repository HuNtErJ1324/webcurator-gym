"""Curation task dataset construction."""

from __future__ import annotations

from datasets import Dataset

from .models import CuratorConfig

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


def build_dataset(config: CuratorConfig) -> Dataset:
    """One row per curation episode: a goal plus its budget and cutoff metadata."""
    rows = []
    for goal in _GOALS:
        info = {
            "token_budget": config.token_budget,
            "cutoff_date": config.cutoff_date,
        }
        rows.append(
            {
                "question": (
                    f"{goal}\n\n"
                    f"Only use Hugging Face datasets modified on or before "
                    f"{config.cutoff_date}. Target roughly {config.token_budget} "
                    f"tokens. Search, inspect, set weighted sources with filters, "
                    f"preview stats, then call finalize_manifest."
                ),
                "answer": config.cutoff_date,
                # Stored as a real dict so per-row overrides (e.g. token_budget)
                # round-trip into rollout state instead of arriving as an opaque
                # JSON string that the environment would ignore.
                "info": info,
            }
        )
    return Dataset.from_list(rows)
