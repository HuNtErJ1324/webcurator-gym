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

### Required config

The CLI builds the taskset directly from `[taskset.*]` and never calls this
package's `load_environment` (a programmatic-only entry point), so these must be
set in your run's TOML:

```toml
max_turns = 300                        # enforced by the framework

[taskset.task]
max_turns = 300                        # optional; what the agent is TOLD it has

[harness]
forward_env = ["HF_TOKEN"]             # or the agent's `hf` CLI is unauthenticated

[harness.runtime]                      # when type = "docker"
workdir = "/workspace"                 # must match Dockerfile.runtime's WORKDIR
```

`forward_env` is the only way `HF_TOKEN` reaches the agent's container: the
harness passes its resolved env to the agent program, and nothing else in the
image carries the token. Without it, host-side scoring still authenticates (it
reads the token from the host environment), so the failure is quiet — the agent
just can't reach gated datasets.

`[taskset.task] max_turns` is a second field because task code cannot read the
framework's limit — v1 never plumbs it to a `Task`. Leave it unset and the agent
is told nothing about its budget; `turns.py` still reports the current turn
(which comes from the framework), just no total or remaining. Set it and keep it
equal to the top-level `max_turns`. If it is ever lower than what the framework
enforces, the run logs a warning naming both values.

`[harness.runtime] workdir` defaults to `/app`, but `Dockerfile.runtime` sets
`WORKDIR /workspace` and bakes two things at absolute paths under it: the `hf`
audit wrapper's `PATH` entry (`/workspace/.agents/bin`) and the decon binary with
its bundled eval sets (`/workspace/decon/`). Leaving the default silently
detaches both — the audit log stays empty and the baked assets are not where the
agent or the rendered self-score script look for them. See
`configs/eval/deepseek-v4-pro-400M-500turn.toml` for a complete working example.

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

The shipped production profile is
`configs/eval/deepseek-v4-pro-400M-500turn.toml` (400M-token budget, 500 turns).
Other local eval/RL/smoke TOMLs under `configs/` stay git-ignored.

| Config | Token budget | Turns |
| --- | --- | --- |
| `deepseek-v4-pro-400M-500turn.toml` | 400M | 500 |

```bash
uv run --project environments/pretrain_curation_gym \
  eval @ environments/pretrain_curation_gym/configs/eval/deepseek-v4-pro-400M-500turn.toml
```

Codex and Claude Code are opaque external harnesses: Verifiers does not receive
their native conversation nodes. When such a harness returns an empty trace,
the environment adds a clearly labeled workspace-telemetry trajectory containing
the task prompt and every `.self_score_history.jsonl` record. The same records
are retained under `trace.info.self_score_history` and exposed as indexed metrics
such as `self_score_001_reward` and `self_score_001_elapsed_seconds` for plotting.
Hugging Face CLI calls are counted by a redacting workspace wrapper rather than
inferred from unavailable harness nodes.

## Speedrun fidelity

The portable proxy trainer is audited against
[`KellerJordan/modded-nanogpt@edf47a05`](https://github.com/KellerJordan/modded-nanogpt/commit/edf47a05a12062d661c4cfd4eef848c5ab5bed32).
Pinned training semantics include the staged `896 -> 2048` context ratio
(scaled to this trainer's maximum context), stationary-half one-token key
shift, projection-removal XSA, paired-layer topology, and weight-preserving
late embedding/head untie.

Intentional scale adaptations remain: steps are derived to meet each token
budget after scheduling, batch sizes are profile-specific, Muon
warmup/cooldown scale with run length, and `wd_ref_steps` preserves the
weight-decay timescale. Portability differences are explicit in
`gpu/train_gpt.py`: single-GPU SDPA with document masks, 1024 maximum context,
an unfused attention block at layer 6, compact sign-derived bigram features,
and no FP8/fused kernels.

## Development

```bash
uv run --project environments/pretrain_curation_gym \
  pytest environments/pretrain_curation_gym/tests -q
uv run --project environments/pretrain_curation_gym \
  ruff check environments/pretrain_curation_gym
```

The repository tracks this environment's tests and release profiles, but they
are not included in the wheel.

### Runtime image

Docker and Modal profiles use the versioned runtime image referenced by the
shipped TOML files. Build and publish it before running those profiles:

```bash
docker build \
  -f environments/pretrain_curation_gym/Dockerfile.runtime \
  -t ghcr.io/hunterj1324/webcurator-runtime:0.1.0 \
  environments/pretrain_curation_gym
docker push ghcr.io/hunterj1324/webcurator-runtime:0.1.0
```

Treat release tags as immutable. The image supplies CUDA PyTorch, the real `hf`
CLI, the Hugging Face `datasets` library, tokenizer dependencies, and the
vendored `decon` binary with `decon/bundled-evals`. The same tree is
force-included into the manylinux wheel (via `scripts/hatch_build.py`) so Hub
installs get a working host-side detector. Task setup fails early if any
required runtime asset is missing.

### Runtime PATH

Task setup writes a redacting `hf` audit wrapper to `<workdir>/.agents/bin/hf`,
which populates the `hf_cli_calls` metric and `trace.info.hf_cli_history`. It
records nothing unless that directory precedes the real CLI on `PATH`.
`Dockerfile.runtime` sets this up for every container; `load_environment`
re-prepends the directory when a harness config supplies its own `PATH`. A
custom runtime image must do the same. Setup verifies both the wrapper and the
underlying real CLI before the rollout begins.
