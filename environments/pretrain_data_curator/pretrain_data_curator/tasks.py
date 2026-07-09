"""Typed curation tasks and their single initial prompt."""

from __future__ import annotations

import verifiers.v1 as vf

from .models import MANIFEST_FILENAME

_GOALS = [
    "curate the strongest general-purpose pretraining mixture for the fixed student.",
]

TASK_PROMPT = """We want to train a fixed small language model on the strongest possible pretraining mixture. You are the data-curation agent, and your goal is to {goal}

## Objective
Research and iterate autonomously, then write the final manifest JSON that defines the mixture. You have complete freedom in source choice, weights, filters, local processing, and use of the shell, internet, Hugging Face `hf` CLI, and other harness tools.
When using commands, execute them through the harness tool or shell interface; writing a command as prose does not run it.

## Research
Feel free to explore broadly before you lock in a mixture. Search the web, read papers and technical writeups, and investigate modern pretraining data practice — there is no prescribed recipe. Useful directions include source discovery and vetting, quality and toxicity filtering, deduplication, domain/reasoning/code/math balancing, synthetic or rewritten corpora, multilingual tradeoffs, and mixture-weighting heuristics. Let what you learn inform your manifest design and filtering choices.

## Deliverable
When finished, write your final manifest as a single JSON object to `{manifest_path}` with this contract:

```text
{{
  "token_budget": {token_budget},
  "sample_docs_per_source": <optional integer >= 1; omit for no per-source fetch cap — fetches are sized from weights and token_budget>,
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

## Self-score (you run it)
Save a draft manifest and score it yourself with the workspace `self_score.py` script. You choose every development knob via CLI flags:

```text
python self_score.py draft.json [--limit N] [--max-steps N] [--max-corpus-chars N] [--train-timeout SEC]
```

- `--limit N` — documents sampled per source (default 8 if omitted).
- `--max-steps N` — proxy-training steps (default: production student config steps).
- `--max-corpus-chars N` — character cap on joined training text (default: all sampled text).
- `--train-timeout SEC` — proxy-training wall clock in seconds (default 900).

The script samples your draft sources, trains the fixed proxy student on that sample (corpus-split cross-entropy), runs benchmark decon leakage, and prints the same `{alpha_perf} * performance - {lambda_leakage} * leakage` reward shape as final scoring. It never uses final held-out validation data; treat it as a directional dev signal, not a full-budget score.

## Setup
- The sole curation budget is {token_budget} tokens.
- Use only data modified on or before {cutoff_date}. Local sources are {local_source_status}.
- Scoring trains the fixed student and applies `{alpha_perf} * performance - {lambda_leakage} * leakage`.
- Performance is scaled from a neutral baseline so progress near the target loss counts more: validation loss `{perf_target_loss}` scores `1.0`, worse than the neutral baseline is negative, and beating `{perf_target_loss}` exceeds `1.0`.

## Rules
1. Use exact dataset IDs and configs observed during this rollout. An invented or incompatible source materializes no data, so its cost produces no performance.
2. Your corpus is checked for data contamination against public benchmark eval sets (AGI Eval, GSM8K, MMLU) AND the held-out validation set using the decon n-gram detector. Contamination against any eval set incurs the leakage penalty in the reward.
3. Set the manifest's `token_budget` field to exactly {token_budget}, and use data, calls, and training work economically. Fetching or processing beyond what can fill that token allocation wastes cost (tracked as a telemetry metric) without increasing the scored corpus.
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
    lambda_leakage: float = 1.0,
    perf_target_loss: float = 3.28,
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
                lambda_leakage=lambda_leakage,
                perf_target_loss=perf_target_loss,
            ),
            system_prompt=None,
            answer=cutoff_date,
            token_budget=token_budget,
            cutoff_date=cutoff_date,
        )
        for i, goal in enumerate(_GOALS)
    ]
