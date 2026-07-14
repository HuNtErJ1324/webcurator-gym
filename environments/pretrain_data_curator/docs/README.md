# Pretrain Data Curator Bench Site

PostTrainBench-style static site for **full 400M-token** curation evaluations.

## Layout

Full 400M eval artifacts live under:

```text
outputs/evals-400m/<run-name>/
  config.toml
  results.jsonl
```

Smoke tests and smaller budgets stay in `outputs/evals/`.

## Build

```bash
cd environments/pretrain_data_curator
python docs/build_site.py
```

By default the builder scans `outputs/evals-400m/` and keeps only runs with:

- `token_budget = 400_000_000`
- `use_real_trainer = true`
- `proxy_student.train_token_budget = 400_000_000` when configured

## View

```bash
cd docs/site
python -m http.server 8080
```

Open `http://localhost:8080`. Rollout traces live on a separate page:

```text
traces/run.html?id=<run-id>#tab=trace
```

Tabs: **Trace**, **Metrics**, **Artifacts**, **Log** (PostTrainBench-style).

A static **Codebase** page (`codebase.html`, linked from the header) explains the
environment: rollout lifecycle, repository layout, module map, and how this site
is generated. It is a plain asset in `site_builder/assets/`; edit it there and
rebuild.

## Training-token budgets

`proxy_student.train_token_budget` limits actual scheduled token presentations,
not merely `steps × base_batch × block_size`. This matters because the
NanoGPT-style batch schedule increases the effective batch during training. The
the pure schedule helpers in `train_gpt.py` compute the minimal number of optimizer steps
whose staged presentations meet the configured budget, and the runtime optimizer
uses the same stage-boundary implementation as accounting.

For the standard 400M profile (`batch_size = 16`, `block_size = 1024`, equal
stages with multipliers `1, 2, 3`), the token-aware schedule uses 12,208 steps.
The previous base-batch calculation used 24,415 steps and presented about 800M
tokens despite reporting a 400M budget. `train_microbatch_size` remains a
memory-only control and does not change budget accounting.

Debug curation snapshots under `outputs/debug/<run-name>/` are included automatically.
Rebuild with `python docs/build_site.py` (omit them with `--no-debug`).

## Trace rendering

Traces are rendered with:

- **Markdown** for prompts and long assistant replies (via `marked`)
- **Syntax highlighting** for JSON, shell, and Python (via `highlight.js`)
- **Pretty JSON** for tool arguments and structured outputs

## Regenerate after new evals

Save new full 400M runs under `outputs/evals-400m/<descriptive-name>/`, then:

```bash
python docs/build_site.py
```
