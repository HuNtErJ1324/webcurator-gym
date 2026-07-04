# pretrain-data-curator

An environment where an agent curates an **LLM pretraining dataset** from the
pre-cutoff Hugging Face universe. The curated mixture is scored by training a
**fixed GPT-2-scale proxy-student where everything but the data is held
constant**, combined with cross-entropy performance, cost, and leakage terms.

## Overview

- **Environment ID**: `pretrain-data-curator`
- **Type**: toolless native verifiers v1 `CuratorTaskset` (`vf.Taskset`) — the agent discovers datasets using the `hf` CLI in its own shell, installing it defensively when the runtime image omits it; no MCP tool server is exposed
- **External service**: Hugging Face Hub API (search + streaming sampling)
- **Required secret**: `HF_TOKEN` by default (checked lazily in the env-server, not at load time)
- **Proxy-student training**: optional GPU training, off by default; selectable co-located harness-runtime Docker backend **or Modal** (recommended for Hosted Training)

The agent searches Hugging Face for datasets at or before a cutoff date,
inspects them, assembles a weighted and filtered **manifest** `M`, previews
statistics, and finalizes. The finalized manifest is materialized into a corpus
and scored.

## Reward

```
R(M, H) = Perf(M) - Cost(M) - Leakage(M)
```

| Term | Meaning | Default weight |
| --- | --- | --- |
| `Perf` | Proxy-student cross-entropy performance on the curated corpus | `alpha_perf=1.0` |
| `Cost` | Web queries, hub calls, tokens, training FLOPs (one ledger) | `lambda_cost=0.1` |
| `Leakage` | Exact + fuzzy (MinHash) + semantic overlap vs the held-out eval set | `lambda_leakage=1.0` |

Each term is also emitted as a metric (`perf_loss`, `perf_accuracy`,
`perf_vs_baseline`, `train_flops`, `corpus_tokens`, `num_sources`, `leakage_*`,
`cost_total`, `finalized`). Local-source provenance is exposed as
`local_source_count`, `local_source_bytes`, `local_source_truncated`, and
`val_set_access`. Two further zero-weight **diagnostics**,
`tool_error_count` and `external_failure`, separate bad curation from
external/infrastructure failure (a flaky Hub or sandbox). See
[`docs/reward.md`](docs/reward.md).

### Proxy-student backends

`Perf(M)` is computed by a `ProxyStudentTrainer`:

- **`HeuristicProxyTrainer`** (default): a deterministic CPU surrogate that
  predicts loss/accuracy from corpus scale, cleanliness, and diversity. It is a
  **dev/smoke surrogate, not a real training signal** — it lets the environment
  run and be tested without a GPU, but its `Perf` is a proxy, not a measurement.
  It trains no model, so it does **not** compute per-token CE on the held-out
  validation set. `validation_set` still supplies the default leakage reference.
- **Real proxy-student training** (`use_real_trainer=true`): the path that yields a
  **meaningful** `Perf`. It actually trains a fixed small GPT (config in
  `ProxyStudentConfig`), GPT-2-BPE-tokenized, in the live Docker or Modal harness
  runtime hosting the rollout, on the materialized corpus, and scores it against
  a **fixed held-out validation token stream** — by default the **NanoGPT
  speedrun** set (FineWeb `sample-10BT` GPT-2 val tokens, `kjj0/fineweb10B-gpt2`,
  the first `10_485_760` tokens), reporting cross-entropy in nats/token,
  next-token accuracy, and FLOPs (see `ValidationSetConfig` / `validation_set`).
  Only the training corpus varies between rollouts. Its lifecycle is hardened
  (per-step timeouts, exit-code checks, stderr-tail preservation).

Measured/estimated training FLOPs are charged back onto the cost ledger.

#### Real-trainer backends: Docker and Modal

When `use_real_trainer=true`, proxy training uses one of two **selectable**
backends, chosen by `proxy_student.runtime_backend` (required — no default).
This field is a static, pre-runtime hint only: it shapes the harness runtime and
task declarations built before any rollout exists. Which trainer actually runs
is decided entirely by the live harness runtime's type at score time, via a
`RuntimeSelectedTrainer` dispatcher — so `runtime_backend` and the harness
runtime you actually configure must agree.

- **`docker`**: declares a GPU-capable v1 `DockerConfig` on the bash harness.
  Dataset discovery, the agent loop, finalization, and proxy-student scoring all
  use the same rollout-owned container on a Docker daemon co-located with the
  eval worker. Scoring receives that live runtime and writes/runs training
  directly through `runtime.write()` and `runtime.run()`.
- **`modal`**: declares a GPU-capable v1 `ModalConfig` on the bash harness.
  Dataset discovery, the agent loop, finalization, and proxy-student scoring all
  use the same rollout-owned Modal sandbox. The CPU-only env-server needs no
  Docker daemon or GPU; it uses outbound HTTPS and the Modal SDK. Requires
  `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`. Set `modal_gpu` (default `"L4"`)
  to choose the GPU type. Because the GPU sandbox now hosts the full rollout,
  Modal billing covers discovery through scoring, not only the training phase.

Docker maps `gpu_count` → `--gpus <count>`; there is no VM-mode or named-GPU-type
config to satisfy for either real-trainer backend, and Docker has no 24h timeout
ceiling (a self-hosted host has no such cap — only Modal enforces one). The
single runtime image must ship both sets of dependencies: bash plus the `hf`
dataset-discovery CLI, and Python plus CUDA PyTorch for training. The included
[`Dockerfile.runtime`](Dockerfile.runtime) derives such an image from the
backend's existing `pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime` default:

```bash
docker build -t pretrain-data-curator:gpu \
  -f environments/pretrain_data_curator/Dockerfile.runtime \
  environments/pretrain_data_curator
```

**Docker-backend prerequisites (on the eval/GPU host):** Docker Engine + the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
(so `--gpus` works). Verify the host first with
`docker run --rm --gpus 1 pretrain-data-curator:gpu nvidia-smi`.

```bash
cd environments/pretrain_data_curator
uv run eval pretrain-data-curator \
  --harness.id bash \
  --harness.runtime.type docker \
  --harness.runtime.image pretrain-data-curator:gpu \
  --harness.runtime.workdir /workspace \
  --harness.runtime.gpu 1 \
  --taskset.use-real-trainer true \
  --taskset.token-budget 50000 \
  --taskset.sample-docs-per-source 16 \
  --taskset.max-turns 12 \
  --taskset.proxy-student \
    '{"runtime_backend":"docker","docker_image":"pretrain-data-curator:gpu","train_token_budget":8192,"gpu_count":1}' \
  -m deepseek/deepseek-v4-flash -n 1 -r 1
```

Use the native v1 taskset command above for local Docker runs. The currently
released `prime eval run` wrapper has no `--harness.runtime.*` passthrough, so it
cannot select this runtime. An owner-qualified target such as
`hunterj/pretrain-data-curator` also resolves the published Hub version rather
than this local checkout. The small budgets above are an end-to-end smoke
configuration; raise them deliberately for comparative data-quality work.

SSH-remote `docker_host` orchestration is intentionally unsupported by this
single-runtime path; `load_environment` rejects a configured `docker_host`.
Leave it unset and do not set an ambient remote `DOCKER_HOST`. A local Unix
socket is valid; Docker Desktop under WSL may require the proxy socket documented
in [Troubleshooting](docs/troubleshooting.md#docker-desktop-uses-an-invalid-npipe-context).
On Docker Desktop under WSL2, the environment automatically advertises the
interception server on the WSL interface; `PDC_DOCKER_HOST_IP` can override the
detected address if necessary.

The Docker and Modal proxy-student commands retain their budget-derived
deadline and structured exit/`RESULT_JSON` validation. Timeout or cancellation
stops the shared runtime immediately, and the rollout's `finally` performs the
idempotent teardown backstop. `max_concurrent_training` bounds training commands
independently of rollout concurrency. The framework scoring timeout is expanded
from the trainer deadline so multi-hour jobs are not cancelled by a short
reward timeout.

## Agent interface

`CuratorTaskset` deliberately exposes **no MCP tool server** (`Taskset.tools` is
not overridden). The agent's interface is the Hugging Face `hf` CLI in its own
shell, with a conditional first-command install when the image omits it — there
are no `curator_*` API commands.

| `hf` command | Purpose | Cost |
| --- | --- | --- |
| `hf datasets ls --search "<query>" --sort downloads --limit 5 \| head -c 6000` | Search and rank dataset repositories without flooding model context. | `web_queries += 1`, `hub_calls += 1`, plus stdout bytes. |
| `hf datasets ls --search "<q>" --format json --expand downloads,likes,lastModified --limit 5 \| head -c 6000` | JSON-formatted search; deliberately omit high-volume `tags`. | same as above |
| `hf datasets info <dataset_id> --expand downloads,likes,tags \| head -c 6000` | Inspect one shortlisted repository. | `hub_calls += 1`, plus stdout bytes. |
| `hf version`, `hf env`, `hf auth`, `hf cache` | Local setup/status — no network. | none |
| Final fenced ` ```json ` block | The agent's curation decision; parsed by `finalize()`. | none |

The agent must emit a non-empty manifest JSON as its **final** message. The
minimal schema:

```json
{
  "token_budget": 1000000,
  "sources": [
    {
      "id": "HuggingFaceFW/fineweb",
      "weight": 1.0,
      "config": "sample-10BT",
      "split": "train",
      "text_field": "text",
      "filters": [{"kind": "min_chars", "params": {"value": 200}}]
    }
  ]
}
```

Each source requires `id` (the Hugging Face dataset id) and `weight` (≥ 0).
For script-backed or otherwise non-streamable datasets, the agent may instead
download/derive text or JSONL in its bash workspace and use a `kind: "local"`
source. See [Manifest and filtering](docs/manifest.md#local-sources).
`config`, `split`, `text_field`, `filters`, `max_docs`, and `max_tokens` are
optional. Supported filter kinds: `min_chars`, `max_chars`, `min_tokens`,
`max_symbol_ratio`, `min_alpha_ratio`, `drop_regex`, `keep_regex`, `dedup_exact`.
When a multi-config dataset has no default and `config` is omitted, the
materializer chooses a stable English/default config (for example `en` or a
config ending in `.en`) before falling back to the first advertised config.

See [`docs/agent-workflow.md`](docs/agent-workflow.md) for the full `hf` workflow, metering
details, manifest schema, filter reference, and turn-budget guidance.

## Documentation

| Page | Use it for |
| --- | --- |
| [Documentation map](docs/README.md) | Choose the right guide. |
| [Architecture](docs/architecture.md) | Follow a rollout from task loading through cleanup. |
| [Agent workflow](docs/agent-workflow.md) | Understand shell discovery, metering, output limits, and manifest recovery. |
| [Manifest and filtering](docs/manifest.md) | Author or audit the final JSON contract. |
| [Configuration](docs/configuration.md) | Choose loader arguments, runtime resources, and eval commands. |
| [Reward and metrics](docs/reward.md) | Interpret scores, costs, leakage, and failure signals. |
| [Proxy student](docs/proxy-student.md) | Understand the heuristic and GPU training paths. |
| [Troubleshooting](docs/troubleshooting.md) | Diagnose credentials, Docker/WSL, Modal, and zero rewards. |
| [Development](docs/development.md) | Test and extend the package without duplicating execution paths. |

## Install And Eval

```bash
prime env install pretrain-data-curator -p ./environments
prime eval run pretrain-data-curator -m openai/gpt-4.1-mini -n 4 -r 1
```

Enable real GPU proxy-student training via Modal (requires `MODAL_TOKEN_ID` /
`MODAL_TOKEN_SECRET`; see [`docs/configuration.md`](docs/configuration.md) for
the Docker alternative, which needs a local GPU host instead).
`proxy_student.runtime_backend` is required (no default) whenever
`use_real_trainer=true`; set `modal_gpu` to choose the GPU type (default
`"L4"`; also `"H100"`, `"H200"`, `"A100"`) and `train_token_budget` to scale
the run (up to ~1e9 tokens):

```bash
prime eval run pretrain-data-curator -m openai/gpt-4.1-mini -n 4 -r 1 \
  -a '{"use_real_trainer": true, "max_turns": 64, "scan_limit": 200, "sample_docs_per_source": 50000, "proxy_student": {"runtime_backend": "modal", "train_token_budget": 400000000, "modal_gpu": "H200"}}'
```

## Required Environment Variables

Hugging Face credentials are validated lazily in the env-server container when
a rollout first accesses the Hub. Constructing the environment does not require
`HF_TOKEN` in the orchestrator process.

| Variable | Default | Description |
| --- | --- | --- |
| `HF_TOKEN` | yes | Token passed to `huggingface_hub.HfApi` and streaming loads. |

## Environment Args

| Arg | Type | Default | Description |
| --- | --- | --- | --- |
| `cutoff_date` | str | `"2024-12-31"` | Latest allowed Hugging Face `lastModified` date. |
| `token_budget` | int | `1000000` | Target token budget for the mixture. |
| `hf_token_env` | str | `"HF_TOKEN"` | Env var checked for the HF token at first Hub API use. |
| `candidate_limit` | int | `8` | Max candidates returned per search. |
| `scan_limit` | int | `50` | Discovery budget input; benchmark configs raise it so the prompt allows more discovery rounds. |
| `sample_docs_per_source` | int | `64` | Docs sampled per source for inspection/scoring. |
| `allow_script_datasets` | bool | `false` | Permit script-backed repositories only when the installed `datasets` runtime supports them. The current `datasets>=3` runtime still rejects them permanently. |
| `allow_local_sources` | bool | `true` | Allow capped pulls of text/JSONL files created in the live bash workspace. |
| `max_local_source_bytes` | int | `33554432` | Maximum bytes transferred per local source before parsing. |
| `max_turns` | int | `12` | Max agent turns; benchmark configs raise it for longer discovery and curation. |
| `alpha_perf` | float | `1.0` | Positive cross-entropy performance weight. |
| `lambda_cost` / `lambda_leakage` | float | `0.1` / `1.0` | Penalty weights. |
| `max_concurrent_fetches` | int | `8` | Bound on concurrent HF fetches (also the corpus-builder fetch limit). |
| `max_concurrent_training` | int | `1` | Bound on concurrent sandbox-training jobs (real trainer). |
| `fetch_timeout_seconds` | float | `30.0` | Base per-attempt timeout for external HF calls. |
| `fetch_timeout_per_doc_seconds` | float | `0.25` | Additional per-document timeout for streaming dataset fetches. |
| `fetch_max_attempts` | int | `3` | Max attempts (retry/backoff) for transient HF failures. |
| `use_real_trainer` | bool | `false` | Use the GPU sandbox proxy-student instead of the heuristic. |
| `proxy_student` | dict | `{}` | Overrides for `ProxyStudentConfig` (arch, `train_token_budget`, `gpu_count`, etc.). `train_token_budget` (≤ 1e9) scales steps/corpus-cap/timeout. Selects the real-trainer backend via `runtime_backend` (`"docker"` / `"modal"`; required, no default, whenever `use_real_trainer=true`); for `"docker"`, set `docker_image` to a combined discovery/training image and leave `docker_host` unset; for `"modal"` see `modal_gpu` (default `"L4"`; also `"H100"`/`"H200"`/`"A100"`) and `gpu_count`, and set `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET`. |
| `validation_set` | dict | NanoGPT speedrun set | Overrides for `ValidationSetConfig` (held-out downstream-CE val set and default leakage source: FineWeb GPT-2 val tokens, first `10_485_760`). |
| `eval_corpus` | list[str] | `None` | Optional explicit leakage-reference override. By default a bounded sample of the real `validation_set` is decoded; the built-in corpus is only an observable offline fallback. |

Before streaming a source, the materializer checks for the Hugging Face
`{dataset_name}.py` convention. Script-backed sources fail once with
`error_kind="script_dataset"` when `allow_script_datasets=false` or the installed
`datasets` version cannot execute scripts; they are not retried. Setting the knob
to `true` never bypasses the runtime capability check. Script execution remains
unavailable in this release because the pinned Verifiers package requires
`datasets>=3`.

When streaming is unavailable, a local source can bridge a genuine downloaded
dataset into scoring without executing repository code. Local paths are
validated, transferred through the live runtime with `head -c`, billed like
fetched tokens plus one code call, and audited with provenance metrics.

## Module Layout

- `__init__.py` — `_bootstrap_verifiers_v1()`: patches `verifiers.__path__` at import time so the real v1 is importable inside Prime's Hosted Training orchestrator (which pre-loads a `verifiers==0.0.0` stub).
- `pretrain_data_curator.py` — `load_environment` entry point; builds `CuratorTasksetConfig` and returns a `hosted_compat.Environment`.
- `hosted_compat.py` — `Environment`: multiple-inheritance v0/v1 bridge. Derives from both `vf.Environment` and `legacy_vf.Environment`; delegates rollout work to the v1 episode engine and translates `vf.Trace` to the v0 `State` / `RolloutTiming` / trajectory format consumed by `prime eval` and Hosted Training.
- `models.py` — Pydantic contracts (`Manifest`, `Source`, `FilterSpec`, `CostLedger`, `CuratorConfig`, `ProxyStudentConfig`, ...).
- `hf_access.py` — `DatasetSearchClient` Protocol, live HF client, cutoff/query helpers. Credentials checked lazily here at first Hub use, not at `load_environment` call time.
- `hf_meter.py` — PATH-shadow `hf` shim, per-rollout JSONL cost log, and trace-reconstruction fallback.
- `corpus.py` — `CorpusBuilder` + `DocumentFilter` (materialize manifest into documents).
- `leakage.py` — `LeakageDetector` (exact/fuzzy/semantic contamination).
- `val_set.py` — held-out validation token stream (`ValidationSetConfig`, `ValTokenLoader`, `.bin` parser); NanoGPT-speedrun set by default.
- `trainer.py` — `ProxyStudentTrainer` interface, the heuristic backend, and `RuntimeSelectedTrainer` (dispatches to the docker/modal backend matching the live harness runtime's type).
- `docker_backend.py` — proxy-student execution on the rollout-owned v1 Docker runtime, including training limits, timeout/cancellation teardown, and structured result parsing.
- `docker_network.py` — `DockerHostReachability`: binds the host interception server to the WSL interface (instead of localhost) so containers started by Docker Desktop's WSL2 VM can reach it; a no-op outside WSL.
- `modal_backend.py` — proxy-student execution on the rollout-owned v1 Modal runtime, including GPU mapping, training limits, timeout/cancellation teardown, and structured result parsing.
- `student_model.py` — modern modded-nanogpt proxy-student architecture embedded into the sandbox script.
- `student_train.py` — AdamW schedule, contiguous batching, and multi-run averaging, also embedded into the sandbox script.
- `rewards.py` — `CuratorScorer`, the framework-agnostic heavy scoring pass.
- `rollout_state.py` — typed `CuratorState` plus `RolloutStore` accessors.
- `taskset.py` — `CuratorTaskset`, manifest parsing/recovery, `finalize()`, decorated rewards/metrics, and `@vf.stop` turn cap. No MCP tool server.
- `tasks.py` — four typed v1 curation tasks and their user prompts.
- `eval_corpus.py` — small offline fallback corpus for the leakage term.

## Notes And Limitations

- Leakage detection is fully deterministic and reproducible across processes:
  the semantic signal is a lightweight character-trigram cosine (no neural deps),
  and the fuzzy MinHash hashes shingles with a seeded `blake2b` (`_stable_hash32`)
  rather than Python's per-process-salted `hash()`. Swap in a real embedder if
  needed.
- Token/corpus cost is billed **once per real fetch**: documents are fetched
  through a per-rollout cache keyed by `(dataset_id, config, split, text_field, n)`
  with per-key single-flight, so previews and final scoring observe identical docs
  and there is no double/triple billing.
- External (HF/sandbox) calls are wrapped with timeout + retry and a typed
  `DatasetAccessError`; on failure tools/scoring **degrade** to a defined sentinel
  (empty slice / infinite-loss `TrainResult`) and record diagnostics rather than
  crashing the rollout.
- Token counts are estimated (~4 chars/token) on the env side; the real trainer
  tokenizes inside the sandbox.
- Filtering is expressed via the structured `filters` argument.
