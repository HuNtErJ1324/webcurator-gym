# pretrain-data-curator

`pretrain-data-curator` evaluates whether an agent can design a useful LLM
pretraining mixture. The agent searches the Hugging Face Hub, submits a weighted
manifest, and receives a reward based on:

```text
reward = performance - cost - leakage
```

The important experimental control is that the student model, optimizer,
training budget, and validation data are fixed. Only the agent-selected training
data changes.

## What a rollout does

1. The environment gives the agent one of four curation goals, a token budget,
   a Hugging Face cutoff date, and a shell.
2. The agent uses the `hf` CLI to search for and inspect datasets. There is no
   environment-specific MCP server and no `curator_*` command.
3. The agent ends with a JSON manifest containing dataset slices, relative
   weights, filters, and optional per-source caps.
4. The environment parses the manifest, streams documents from each source,
   applies filters and caps, and records the resulting corpus.
5. A fixed proxy student evaluates the corpus. The default is a deterministic
   CPU heuristic for smoke tests; real evaluation trains the student on a GPU.
6. The scorer subtracts discovery/materialization cost and contamination against
   a held-out leakage corpus.

The agent never uploads a prepared corpus. It selects public Hugging Face
datasets and describes how the environment should materialize them.

## Quick start

Run these commands from the workspace root:

```bash
prime env install pretrain-data-curator -p ./environments
prime eval run pretrain-data-curator -m openai/gpt-4.1-mini -n 4 -r 1
```

The default run uses `HeuristicProxyTrainer`. It is suitable for installation,
prompt, manifest, filtering, and scoring smoke tests. It does **not** train a
language model and should not be interpreted as a data-quality benchmark.

`HF_TOKEN` is required when scoring first materializes a Hugging Face dataset:

```bash
export HF_TOKEN=hf_...
```

The token is checked lazily in the environment worker. Constructing or importing
the environment does not require it.

## Minimal manifest

The final assistant message should contain a fenced JSON object:

```json
{
  "token_budget": 1000000,
  "sources": [
    {
      "id": "HuggingFaceFW/fineweb",
      "config": "sample-10BT",
      "split": "train",
      "text_field": "text",
      "weight": 1.0,
      "filters": [
        {"kind": "min_chars", "params": {"value": 200}},
        {"kind": "dedup_exact"}
      ]
    }
  ]
}
```

`id` and `weight` are the primary agent-facing fields. `config`, `split`,
`text_field`, `filters`, `max_docs`, and `max_tokens` are optional. If
`text_field` is omitted or `null`, the materializer tries common textual columns.
See [Manifest and filtering](docs/manifest.md) for exact coercion, allocation,
and filter behavior.

## Choosing a proxy backend

| Mode | Selection | Intended use | Infrastructure |
| --- | --- | --- | --- |
| Heuristic | `use_real_trainer=false` | Unit tests and cheap smoke evals | CPU only |
| Prime sandbox | `use_real_trainer=true`, `trainer_backend="prime"` | Real proxy score with an independently provisioned GPU sandbox | Prime Sandboxes |
| Docker runtime | `use_real_trainer=true`, `trainer_backend="docker"` | Local or self-hosted GPU evaluation | Docker + NVIDIA Container Toolkit |
| Modal runtime | `use_real_trainer=true`, `trainer_backend="modal"` | Hosted GPU evaluation without a local Docker daemon | Modal credentials |

Prime is the default real-trainer backend. Docker and Modal run discovery and
training inside the same rollout-owned harness runtime. That distinction affects
image requirements, credentials, networking, and billing; read
[Proxy student](docs/proxy-student.md) before a GPU run.

## Reward at a glance

With default coefficients:

```text
R = 1.0 * Perf - 0.1 * Cost - 1.0 * Leakage
```

- `Perf` is a bounded improvement over a fixed baseline loss by default.
- `Cost` prices Hub calls, command output tokens, corpus tokens, and training
  FLOPs from one ledger.
- `Leakage` is the maximum of exact, fuzzy MinHash, and character-trigram
  similarity hit rates.
- An unfinalized or empty manifest receives no positive performance reward but
  still pays discovery cost.
- External Hub or trainer failures degrade to a zero-performance sentinel and
  set diagnostics instead of crashing the entire rollout.

See [Reward and metrics](docs/reward.md) for formulas and every emitted metric.

## Documentation

| Page | Use it for |
| --- | --- |
| [Documentation map](docs/README.md) | Find the right page quickly. |
| [Architecture](docs/architecture.md) | Understand the v1 taskset, state, finalization, materialization, and scoring lifecycle. |
| [Agent workflow](docs/agent-workflow.md) | Understand what the evaluated agent sees and how discovery is metered. |
| [Manifest and filtering](docs/manifest.md) | Author or validate the final JSON contract. |
| [Configuration](docs/configuration.md) | Select environment arguments, trainer backends, and runtime resources. |
| [Reward and metrics](docs/reward.md) | Interpret scores, penalties, and failure diagnostics. |
| [Proxy student](docs/proxy-student.md) | Understand the heuristic and real training paths. |
| [Troubleshooting](docs/troubleshooting.md) | Diagnose credentials, images, Docker/WSL, Modal, timeouts, and zero rewards. |
| [Development](docs/development.md) | Install, test, audit, and safely extend the package. |

## Known boundaries

- The cutoff date is an instruction to the agent. The finalizer does not call
  the Hub to reject a manifest solely because a repository was modified later.
- Token counts in the environment are deterministic estimates:
  `max(word_count, character_count // 4)`. The real trainer tokenizes with GPT-2
  BPE inside the GPU runtime.
- A source weight controls its share of the manifest token budget, but the
  actual corpus can be smaller because of fetch limits, filtering, short
  datasets, or explicit source caps.
- The default heuristic is useful for mechanics, not scientific comparison.
- Docker real training supports a co-located daemon only. Remote
  `docker_host=ssh://...` is rejected.

## Source map

The public entry point is
`pretrain_data_curator.pretrain_data_curator:load_environment`.

| Module | Responsibility |
| --- | --- |
| `taskset.py` | Tasks, system prompt, manifest recovery, finalization, rewards, metrics, and turn limit |
| `models.py` | Pydantic manifest, ledger, and environment/trainer configuration |
| `hf_meter.py` | `hf` command shim, JSONL discovery ledger, and trace fallback |
| `hf_access.py` | Streaming Hugging Face access, retry, timeout, and error classification |
| `corpus.py` | Weight allocation, document fetches, filters, caps, caching, and billing |
| `rewards.py` | Single heavy scoring pass and composite reward components |
| `leakage.py` | Exact, MinHash, and character-trigram leakage checks |
| `trainer.py` | Heuristic trainer, Prime sandbox trainer, and generated training script |
| `docker_backend.py` / `modal_backend.py` | Training through a live v1 harness runtime |
| `student_model.py` / `student_train.py` | Embedded model architecture and training recipe |
| `val_set.py` | Fixed held-out validation-token loading and validation windows |
| `rollout_state.py` | Typed per-rollout state and validated accessors |
| `hosted_compat.py` | v1 episode to legacy Prime eval/Hosted Training bridge |
