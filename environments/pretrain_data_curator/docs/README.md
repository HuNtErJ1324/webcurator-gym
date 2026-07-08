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
