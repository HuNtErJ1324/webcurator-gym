# pretrain-curation-gym

A native Verifiers v1 environment in which an agent curates a pre-cutoff LLM
pretraining corpus. The data mixture is scored by a fixed proxy student's
cross-entropy performance minus token-weighted benchmark/validation leakage.

## Quickstart

```bash
prime env install pretrain-curation-gym
set -a; source secrets.env; set +a
uv run --project environments/pretrain_curation_gym \
  eval pretrain-curation-gym -n 1 -r 1
```

`HF_TOKEN` is required. Real Modal training additionally requires
`MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`.

The installed `prime eval run` command currently routes through Verifiers' v0
bridge and cannot execute a native `vf.Environment`; use the v1 `eval` command
above until that wrapper adopts the v1 runner.

## Task and reward

The agent receives one method-open prompt and uses the stock harness shell. No
MCP tool server is attached. It searches and inspects Hugging Face datasets,
may prepare bounded workspace-local JSONL/text sources, and maintains one
authoritative workspace-relative `manifest.json`.

The score is unchanged from `pretrain-data-curator`:

```text
reward = alpha_perf * performance - lambda_leakage * leakage
```

By default, performance is the convex target-scaled loss improvement:

```text
p = (baseline_loss - loss) / (baseline_loss - target_loss)
performance = p ** gamma if p >= 0 else p
```

The default `gamma` is `2.0`; target loss `3.28` maps to `1.0`. Leakage is the
token-weighted decon n-gram contamination score against the bundled AGI Eval,
GSM8K, and MMLU sets plus the ephemeral held-out validation text.

After the corpus is materialized once, proxy training/validation and Decon read
it concurrently. Their unchanged component results are joined before the final
reward is reduced.

The heuristic student remains the fast default. Set
`taskset.task.curator.use_real_trainer=true` and select a Docker or Modal
`proxy_student.runtime_backend` for meaningful fixed-student CE.

## v1 configuration

Configuration follows ownership rather than duplicating loader kwargs:

```toml
[taskset]
id = "pretrain-curation-gym"

[taskset.task]
manifest_filename = "manifest.json"
hf_token_env = "HF_TOKEN"

[taskset.task.curator]
cutoff_date = "2024-12-31"
token_budget = 1000000
max_turns = 64
alpha_perf = 1.0
lambda_leakage = 1.0
use_real_trainer = false

[harness]
id = "default"
```

- `CuratorEnvConfig` owns v1 taskset/harness composition and framework limits.
- `CuratorTasksetConfig` owns the typed task config only.
- `CuratorTaskConfig` owns curation, manifest, credential, and decon settings.
- `CuratorState` is the sole mutable per-rollout store.

Run `uv run --project environments/pretrain_curation_gym eval
pretrain-curation-gym --help` for the complete typed CLI surface.

Sixteen migrated evaluation profiles live in `configs/eval/`. They retain the
predecessor's model, budget, turn, runtime, and scoring choices in native v1
sections. For example:

```bash
uv run --project environments/pretrain_curation_gym \
  eval @ environments/pretrain_curation_gym/configs/eval/deepseek-v4-flash-smoke.toml
```

The two Hosted Training profiles in `configs/rl/` use the same typed
`[env.taskset]` and `[env.harness]` boundaries. They reference
`hunterj/pretrain-curation-gym`, so publish the verified package to that Hub ID
with the intended public/private visibility before launching either profile.

## Preserved behavior

- Same manifest/source/filter/sampling contracts, including local sources.
- Same pre-cutoff Hugging Face streaming, retry, timeout, cache, and
  single-flight behavior.
- Same fixed proxy-student architecture and heuristic/Docker/Modal trainers.
- Same decon detector and ephemeral held-out validation screening.
- Same strict workspace-file finalization and non-production trace candidates.
- Same self-scoring script, history diagnostics, turn counter, and 27 metrics.
- Same external-failure, trainer-error, and provenance telemetry.

The runtime workspace is now addressed relatively. This is equivalent to
`/workspace` for Docker/Modal but also works correctly in v1 subprocess and
other runtimes, whose workspace root is provider-owned.

## Architecture

- `config.py` — the three v1 config ownership boundaries.
- `taskset.py` — a small loader that constructs one typed task.
- `tasks.py` — immutable task data and prompt rendering.
- `task.py` — v1 setup/finalize/stop/scoring lifecycle.
- `state.py` — typed rollout state, scratch files, and document cache.
- `manifest.py` — manifest parsing and compatibility-only trace candidates.
- `corpus.py` — streaming materialization, filtering, sampling, and local files.
- `rewards.py` — one framework-independent heavy scoring pass.
- `trainer.py`, `gpu/` — heuristic and fixed-student training implementations.
- `leakage.py`, `val_set.py` — contamination and held-out validation inputs.

Unlike the populated predecessor, scoring is registered as one keyed
`@vf.reward`. It materializes/trains/screens exactly once, records all diagnostic
metrics on the v1 trace, and returns the two named reward contributions. This
removes the cache/lock required when dozens of metric wrappers raced the same
heavy scoring operation.

## Validation

```bash
uv run --project environments/pretrain_curation_gym \
  pytest environments/pretrain_curation_gym/tests -q
uv run --project environments/pretrain_curation_gym \
  ruff check environments/pretrain_curation_gym
uv build --project environments/pretrain_curation_gym
```
