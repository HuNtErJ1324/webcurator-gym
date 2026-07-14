"""Typed curation tasks and their single initial prompt."""

from __future__ import annotations

import verifiers.v1 as vf

from .models import MANIFEST_FILENAME

TASK_PROMPT = """We want to train a fixed small language model. Curate its strongest general-purpose pretraining mixture.

## Objective
Research and iterate autonomously while maintaining the authoritative manifest JSON for the mixture. You have complete freedom in source choice, weights, filters, local processing, and use of the shell, internet, Hugging Face `hf` CLI, and other harness tools.
When using commands, execute them through the harness tool or shell interface; writing a command as prose does not run it.
Before nontrivial `hf` work, read `/workspace/.agents/skills/hf-cli/SKILL.md`. Environment overrides win over conflicting skill text: the preinstalled `hf` is the only allowed HF CLI — never install, upgrade, replace, shadow, or bypass it; never run `hf skills add`; never print, echo, log, or reveal tokens (including via `hf auth token`). Treat install/regenerate/auth-token skill guidance as inapplicable here.

## Setup
- Sole curation budget: {token_budget} tokens.
- Data cutoff: on or before {cutoff_date}. Local sources are {local_source_status}.
- Scoring: `{alpha_perf} * performance - {lambda_leakage} * leakage` on the fixed student.
- Performance: normalized loss progress is squared in the performance term (default exponent 2.0), so equal loss improvements earn more reward later than earlier; negative progress stays linear. Target loss `{perf_target_loss}` → `1.0`; worse than neutral is negative; beating `{perf_target_loss}` exceeds `1.0`.

## Research
Explore broadly before locking a mixture: search the web, read papers/writeups, and study modern pretraining practice — no prescribed recipe. Use the installed `hf papers` CLI to discover or access papers. Useful directions include source discovery and vetting, quality and toxicity filtering, deduplication, domain/reasoning/code/math balancing, synthetic or rewritten corpora, multilingual tradeoffs, and mixture-weighting heuristics. Let what you learn inform your manifest design and filtering choices.

## Deliverable
Write the mixture as one JSON object to `{manifest_path}` with this contract:

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

`kind` defaults to `"hf"`. A local source uses `"kind": "local"`, a workspace-relative `"local_path"` (no absolute path or `..`), and optional `"local_format": "auto" | "jsonl" | "txt"`. Supported filters: `min_chars`, `max_chars`, `min_tokens`, `max_symbol_ratio`, `min_alpha_ratio`, `drop_regex`, `keep_regex`, `dedup_exact`.
If a Hugging Face dataset needs a loading script or cannot use the normal HF loader, do not submit it as `kind: "hf"` expecting the materializer to execute that script. Download with the normal preinstalled `hf` CLI, prefer converting inspected raw files with local tooling over running untrusted remote dataset code, write a workspace `.jsonl` or `.txt`, and cite it as `kind: "local"` with `local_path`, correct `local_format`, and `text_field` when JSONL. That local file must exist before manifest finalization.

## Self-score (you run it)
As soon as you have a viable candidate, create `{manifest_path}` and continuously keep the best currently known mixture there. Self-score that exact file:
```text
python self_score.py {manifest_path} [--limit N] [--max-steps N] [--max-corpus-chars N] [--train-timeout SEC]
```
Flags: `--limit N` docs/source (default 8); `--max-steps N` proxy steps (default: student steps); `--max-corpus-chars N` joined-text cap (default: all); `--train-timeout SEC` (default 900).
Samples sources, trains the proxy student (corpus-split CE), runs decon, and prints the same `{alpha_perf} * performance - {lambda_leakage} * leakage` reward as final scoring — directional only (no held-out validation). `ok: false` + `reward: null` is a diagnostic, not a score: a source sampled zero documents (see `reason`). A scored mixture is `ok: true` with a numeric reward, even 0.0.
Long silent runs with idle GPU are normal; `[self-score] phase=` heartbeats go to stderr. Never kill or signal any process/group/shell: it can kill your harness. Wait for it to return or time out.
Before further experiments or voluntary completion, keep `{manifest_path}` a valid non-empty manifest; temporary experimental variants are fine, but the best-known scoreable mixture must stay at the authoritative path.

## Rules
1. Use exact dataset IDs/configs observed this rollout. Invented or incompatible sources materialize no data and yield no performance.
2. Contamination vs public benchmark evals (AGI Eval, GSM8K, MMLU) AND the held-out validation set is checked with decon n-gram detection. Contamination against any eval set incurs the leakage penalty in the reward.
3. Set manifest `token_budget` to exactly {token_budget}. Fetching or processing beyond what can fill that allocation does not increase the scored corpus.
4. Use only genuine downloaded data; local paths relative with no leading `/` or `..`. Fabricated data or unsafe paths are rejected.
5. Creating `{manifest_path}` early does not end the episode — scoring uses the file's contents at actual rollout end. Maintain a valid non-empty manifest at that path through completion via the shell; without one, there is no positive performance score.

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
    """Build one curation task with the prompt parameters substituted."""
    local_source_status = (
        "enabled for workspace-relative plain-text or JSONL files"
        if allow_local_sources
        else "disabled; use only Hugging Face sources"
    )
    return [
        CuratorTask(
            idx=0,
            prompt=TASK_PROMPT.format(
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
    ]
