"""Typed curation tasks and their single initial prompt."""

from __future__ import annotations

import verifiers.v1 as vf

from .models import MANIFEST_FILENAME

_GOALS = [
    "Curate the strongest general-purpose pretraining mixture for the fixed student.",
]

TASK_PROMPT = """We want to train a fixed small language model on the strongest possible pretraining mixture. You are the data-curation agent, and your goal is to {goal}

## Objective
Research and iterate autonomously, then write the final manifest JSON that defines the mixture. You have complete freedom in source choice, weights, filters, local processing, and use of the shell, internet, Hugging Face `hf` CLI, and other harness tools.
When using commands, execute them through the harness tool or shell interface; writing a command as prose does not run it.

## Deliverable
When finished, write your final manifest as a single JSON object to `{manifest_path}` with this contract:

```text
{{
  "token_budget": {token_budget},
  "sample_docs_per_source": <optional integer, 1-100000>,
  "sources": [
    {{
      "id": "<observed Hugging Face owner/name>",
      "kind": "hf",
      "weight": <nonnegative number>,
      "config": <observed config or null>,
      "split": "train",
      "text_field": <field name or null>,
      "filters": [{{"kind": "<supported filter>", "params": {{...}}}}],
      "max_docs": <optional positive integer>,
      "max_tokens": <optional positive integer>
    }}
  ]
}}
```

`kind` defaults to `"hf"`. A local source instead uses `"kind": "local"`, a workspace-relative `"local_path"`, and optional `"local_format": "auto" | "jsonl" | "txt"`. Supported filters are `min_chars`, `max_chars`, `min_tokens`, `max_symbol_ratio`, `min_alpha_ratio`, `drop_regex`, `keep_regex`, and `dedup_exact`.

## Setup
- The sole curation budget is {token_budget} tokens.
- Use only data modified on or before {cutoff_date}. Local sources are {local_source_status}.
- Scoring trains the fixed student and applies `{alpha_perf} * performance - {lambda_cost} * cost - {lambda_leakage} * leakage`.
- For cheap iteration, save a draft manifest and run `python self_score.py draft.json --limit 8`. This development-only heuristic samples candidate-source data and reports estimated proxy CE and reward components; it never uses final held-out validation data.

## Rules
1. Use exact dataset IDs and configs observed during this rollout. An invented or incompatible source materializes no data, so its cost produces no performance.
2. Never access, copy, infer, or derive data from the held-out validation or evaluation corpus. Doing so is contamination and incurs the leakage penalty.
3. Set the manifest's `token_budget` field to exactly {token_budget}, and use data, calls, and training work economically. Fetching or processing beyond what can fill that token allocation increases the cost penalty without increasing the scored corpus.
4. Use only genuine downloaded data, and keep local paths relative with no leading `/` or `..`. Fabricated data or unsafe paths are rejected and cannot improve the score.
5. Create the final manifest file through the shell. Its existence is the completion signal; without a valid non-empty manifest at `{manifest_path}`, there is no positive performance score.

There will be no user interaction. Never ask the user for feedback or clarification; operate autonomously and execute the actions that make the most sense."""


class CuratorTask(vf.Task):
    """One curation episode with typed budget and cutoff provenance."""

    answer: str
    token_budget: int
    cutoff_date: str


def build_tasks(
    cutoff_date: str,
    token_budget: int,
    *,
    manifest_filename: str = MANIFEST_FILENAME,
    allow_local_sources: bool = True,
    alpha_perf: float = 1.0,
    lambda_cost: float = 0.1,
    lambda_leakage: float = 1.0,
) -> list[CuratorTask]:
    """Build one curation task with the single goal substituted into the prompt."""
    local_source_status = (
        "enabled for workspace-relative plain-text or JSONL files"
        if allow_local_sources
        else "disabled; use only Hugging Face sources"
    )
    return [
        CuratorTask(
            idx=i,
            prompt=TASK_PROMPT.format(
                goal=goal,
                cutoff_date=cutoff_date,
                token_budget=token_budget,
                manifest_path=f"/workspace/{manifest_filename}",
                local_source_status=local_source_status,
                alpha_perf=alpha_perf,
                lambda_cost=lambda_cost,
                lambda_leakage=lambda_leakage,
            ),
            system_prompt=None,
            answer=cutoff_date,
            token_budget=token_budget,
            cutoff_date=cutoff_date,
        )
        for i, goal in enumerate(_GOALS)
    ]
