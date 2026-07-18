# pretrain-curation-gym

An environment where an agent builds a pretraining data mixture from Hugging Face
datasets. The mixture is scored on how well a fixed proxy model learns from it,
minus a penalty for leaking evaluation or held-out validation text.

## Quickstart

```bash
prime env install pretrain-curation-gym
set -a; source secrets.env; set +a
uv run --project environments/pretrain_curation_gym \
  eval pretrain-curation-gym -n 1 -r 1
```

You need `HF_TOKEN`. For real Modal training, also set `MODAL_TOKEN_ID` and
`MODAL_TOKEN_SECRET`.

Use the `eval` command above (not `prime eval run`) for this environment.

## How it works

The agent gets an open-ended prompt and a shell. It searches Hugging Face,
optionally prepares local text sources, and writes a workspace `manifest.json`
that describes the final corpus.

At the end of a rollout, that corpus is trained and screened. The reward is:

```text
reward = performance - leakage
```

Performance improves when the proxy student reaches a lower loss on held-out
data. Leakage rises when the corpus overlaps AGI Eval, GSM8K, MMLU, or the
held-out validation text. Weights for each term are configurable.

By default scoring uses a fast heuristic student. For real proxy training, set
`taskset.task.curator.use_real_trainer=true` and configure the harness runtime
(Docker or Modal).

## Configuration

Minimal example:

```toml
max_turns = 64

[taskset]
id = "pretrain-curation-gym"

[taskset.task]
manifest_filename = "manifest.json"
hf_token_env = "HF_TOKEN"

[taskset.task.curator]
cutoff_date = "2024-12-31"
token_budget = 1000000
alpha_perf = 1.0
lambda_leakage = 1.0
use_real_trainer = false

[harness]
id = "default"
```

See `uv run --project environments/pretrain_curation_gym eval
pretrain-curation-gym --help` for the full option list.

### Evaluation profiles

Ready-made configs live in `configs/eval/`:

| Config | Token budget | Turns |
| --- | --- | --- |
| `deepseek-v4-pro-10M-100turn.toml` | 10M | 100 |
| `deepseek-v4-pro-100M-300turn.toml` | 100M | 300 |
| `deepseek-v4-pro-400M-300turn.toml` | 400M | 300 |

```bash
uv run --project environments/pretrain_curation_gym \
  eval @ environments/pretrain_curation_gym/configs/eval/deepseek-v4-pro-10M-100turn.toml
```

Hosted training profiles are in `configs/rl/`. Publish the package to
`hunterj/pretrain-curation-gym` before launching those runs.

## Development

```bash
uv run --project environments/pretrain_curation_gym \
  pytest environments/pretrain_curation_gym/tests -q
uv run --project environments/pretrain_curation_gym \
  ruff check environments/pretrain_curation_gym
```
