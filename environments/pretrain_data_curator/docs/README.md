# Documentation map

These pages describe the checked-in `pretrain-data-curator` implementation.
Start with the package [README](../README.md) if you only need to install and run
it.

## Read by task

| If you want to… | Read |
| --- | --- |
| Understand one rollout end to end | [Architecture](architecture.md) |
| Understand the evaluated agent's shell and prompt | [Agent workflow](agent-workflow.md) |
| Write or inspect a final manifest | [Manifest and filtering](manifest.md) |
| Choose arguments or a GPU backend | [Configuration](configuration.md) |
| Interpret a reward or metric | [Reward and metrics](reward.md) |
| Understand what the proxy model actually measures | [Proxy student](proxy-student.md) |
| Diagnose a failed or zero-reward run | [Troubleshooting](troubleshooting.md) |
| Change or validate the package | [Development](development.md) |
| Add a filter, trainer, or other extension | [Extending](extending.md) |

## Core vocabulary

- **Taskset**: the verifiers v1 object that supplies tasks, setup/finalization,
  stopping conditions, rewards, and metrics.
- **Harness**: the agent loop. Bash is the default; `harness_id` can select
  compatible bundled CLI harnesses such as Codex or mini-SWE-agent.
- **Runtime**: where harness commands execute. It is a subprocess by default,
  or a rollout-owned Docker/Modal runtime for those real-trainer backends.
- **Manifest**: the agent's final JSON decision: source slices, weights, filters,
  caps, and a token budget.
- **Materialization**: streaming selected source documents, filtering them, and
  enforcing allocation/caps.
- **Proxy student**: the fixed model and recipe used to measure the selected
  corpus. The cheap default is only a deterministic surrogate.
- **Leakage corpus**: short held-out text used to penalize contamination. It is
  separate from the tokenized validation set used by real proxy training.
- **Validation set**: a fixed GPT-2-token shard used to compute real student
  cross-entropy after training.

## Run configuration

There is one checked-in local eval run config:

- `configs/eval/deepseek-v4-flash-smoke.toml`: exhaustive DeepSeek V4 Flash
  curation smoke with real Modal proxy training. It doubles as the authoritative
  loader/student/validation field reference.

RL training configs live under `configs/rl/` (`curator-gpt20b-modal.toml` and
`curator-gpt20b-modal-1rollout.toml`) for Hosted Training with a Modal H100.
