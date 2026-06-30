# pretrain-data-curator

An environment where an agent curates an **LLM pretraining dataset** from the
pre-cutoff Hugging Face universe. The curated mixture is scored by training a
**fixed GPT-2-scale proxy-student where everything but the data is held
constant**, combined with cross-entropy performance, cost, and leakage terms.

## Overview

- **Environment ID**: `pretrain-data-curator`
- **Type**: toolless native verifiers v1 `CuratorTaskset` (`vf.Taskset`) — the agent discovers datasets using the preinstalled `hf` CLI in its own shell; no MCP tool server is exposed
- **External service**: Hugging Face Hub API (search + streaming sampling)
- **Required secret**: `HF_TOKEN` by default (checked lazily in the env-server, not at load time)
- **Proxy-student training**: optional GPU training, off by default; selectable Prime GPU sandbox, Docker backend (local/remote `DOCKER_HOST`), **or Modal** (recommended for Hosted Training)

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
`cost_total`, `finalized`). Two further zero-weight **diagnostics**,
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
  validation set and is unaffected by `validation_set`.
- **`SandboxProxyTrainer`** (`use_real_trainer=true`): the path that yields a
  **meaningful** `Perf`. It actually trains a fixed small GPT (config in
  `ProxyStudentConfig`), GPT-2-BPE-tokenized, in a Prime GPU sandbox on the
  materialized corpus and scores it against a **fixed held-out validation token
  stream** — by default the **NanoGPT speedrun** set (FineWeb `sample-10BT` GPT-2
  val tokens, `kjj0/fineweb10B-gpt2`, the first `10_485_760` tokens), reporting
  cross-entropy in nats/token, next-token accuracy, and FLOPs (see
  `ValidationSetConfig` / `validation_set`). Only the training corpus varies
  between rollouts. Its lifecycle is hardened (per-step timeouts, exit-code
  checks, stderr-tail preservation) and `prime_sandboxes` is an optional extra
  (`sandbox`) imported only on this path.

Measured/estimated training FLOPs are charged back onto the cost ledger.

#### Real-trainer backends: Prime, Docker, and Modal

When `use_real_trainer=true`, `SandboxProxyTrainer` runs on one of three
**selectable** backends, chosen by `proxy_student.trainer_backend`:

- **`prime`** (default): provisions a Prime GPU sandbox via `prime_sandboxes`.
  GPU-request guards (`vm=true`, a non-empty `gpu_type`) and the Prime 24h
  timeout ceiling apply.
- **`docker`**: runs the identical training script in a container via verifiers'
  v1 `DockerRuntime` (the local `docker` CLI). Point it at a **remote rented
  host** with `DOCKER_HOST=ssh://user@host` — only the GPU training runs there;
  rollouts (HF discovery, scoring) stay local.
- **`modal`**: provisions a Modal GPU sandbox via `modal.Sandbox.create()` (v1
  Modal API). **Recommended for Prime Hosted Training**: the env-server is
  CPU-only with no Docker daemon, but outbound HTTPS to Modal works. Requires
  `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` in the environment; `modal>=0.73` is
  included in base dependencies. Set `modal_gpu` (default `"L4"`) to choose the
  GPU type — ~$0.011/run at the 200-step default, ~$0.055/run on H100.

The Docker backend lifts the Prime-specific config rules: `vm` and `gpu_type` are
ignored (Docker maps `gpu_count` → `--gpus <count>`), so `vm=false` is allowed,
and the 24h timeout ceiling is relaxed (a self-hosted host has no such cap). The
container image **must** ship torch + CUDA; if `docker_image` is left unset it
defaults to `pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime` for this backend (pick
a `-devel` tag if you need build tooling such as `nvcc` for `torch.compile`).

**Docker-backend prerequisites (on the remote host):** Docker Engine + the
[NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
(so `--gpus` works), and **key-based SSH** from the machine running the env (the
`ssh://` Docker transport uses your SSH agent/keys — no password prompts). Verify
the host first with `DOCKER_HOST=ssh://user@host docker run --rm --gpus 1 \
  pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime nvidia-smi`.

```bash
export DOCKER_HOST=ssh://user@gpu-host   # or set proxy_student.docker_host below
prime eval run pretrain-data-curator -m openai/gpt-4.1-mini -n 4 -r 1 \
  -a '{"use_real_trainer": true, "proxy_student": {
        "trainer_backend": "docker",
        "docker_image": "pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime",
        "gpu_count": 1,
        "docker_host": "ssh://user@gpu-host"}}'
```

`docker_host` (config) is applied to `DOCKER_HOST` only when that variable is not
already set in the environment, so an ambient `export DOCKER_HOST=...` always
wins.

## Agent interface

`CuratorTaskset` deliberately exposes **no MCP tool server** (`Taskset.tools` is
not overridden). The agent's interface is the preinstalled Hugging Face `hf` CLI
in its own shell — there are no `curator_*` API commands.

| `hf` command | Purpose | Cost |
| --- | --- | --- |
| `hf datasets ls --search "<query>" --sort downloads --limit 10` | Search and rank dataset repositories. | `web_queries += 1`, `hub_calls += 1`, plus stdout bytes. |
| `hf datasets ls --search "<q>" --format json --expand downloads,likes,lastModified,tags` | JSON-formatted search with metadata fields. | same as above |
| `hf datasets info <dataset_id> --expand downloads,likes,tags` | Inspect repository metadata. | `hub_calls += 1`, plus stdout bytes. |
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
`config`, `split`, `text_field`, `filters`, `max_docs`, and `max_tokens` are
optional. Supported filter kinds: `min_chars`, `max_chars`, `min_tokens`,
`max_symbol_ratio`, `min_alpha_ratio`, `drop_regex`, `keep_regex`, `dedup_exact`.

See [`docs/tools.md`](docs/tools.md) for the full `hf` workflow, metering
details, manifest schema, filter reference, and turn-budget guidance.

## Install And Eval

```bash
prime env install pretrain-data-curator -p ./environments
prime eval run pretrain-data-curator -m openai/gpt-4.1-mini -n 4 -r 1
```

Enable real GPU proxy-student training (requires Prime Sandboxes access). The
GPU request defaults are already valid (`gpu_count=1`, `gpu_type="H100"`,
`vm=true`), so a bare `use_real_trainer` works; override `gpu_type` for H200 and
`train_token_budget` to scale the run (up to ~1e9 tokens):

```bash
prime eval run pretrain-data-curator -m openai/gpt-4.1-mini -n 4 -r 1 \
  -a '{"use_real_trainer": true, "max_turns": 64, "scan_limit": 200, "sample_docs_per_source": 50000, "proxy_student": {"train_token_budget": 400000000, "gpu_type": "H200"}}'
```

The installed `prime_sandboxes` SDK validates `gpu_type` as a non-empty string,
not an enum, so confirm the exact accepted value (`"H100"`, `"H200"`, etc.)
against the user's Prime account before relying on a non-default GPU type.

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
| `max_turns` | int | `12` | Max agent turns; benchmark configs raise it for longer discovery and curation. |
| `alpha_perf` | float | `1.0` | Positive cross-entropy performance weight. |
| `lambda_cost` / `lambda_leakage` | float | `0.1` / `1.0` | Penalty weights. |
| `max_concurrent_fetches` | int | `8` | Bound on concurrent HF fetches (also the corpus-builder fetch limit). |
| `max_concurrent_training` | int | `1` | Bound on concurrent sandbox-training jobs (real trainer). |
| `fetch_timeout_seconds` | float | `30.0` | Per-attempt timeout for external HF calls. |
| `fetch_max_attempts` | int | `3` | Max attempts (retry/backoff) for transient HF failures. |
| `use_real_trainer` | bool | `false` | Use the GPU sandbox proxy-student instead of the heuristic. |
| `proxy_student` | dict | `{}` | Overrides for `ProxyStudentConfig` (arch, `train_token_budget`, GPU request: `gpu_count`/`gpu_type`/`vm`, etc.). GPU defaults (`gpu_count=1`, `gpu_type="H100"`, `vm=true`) form a valid Prime request; `train_token_budget` (≤ 1e9) scales steps/corpus-cap/timeout. Selects the real-trainer backend via `trainer_backend` (`"prime"` default / `"docker"` / `"modal"`); for `"docker"` see `docker_host` and the backend-aware `docker_image` default above; for `"modal"` see `modal_gpu` (default `"L4"`) and set `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET`. |
| `validation_set` | dict | NanoGPT speedrun set | Overrides for `ValidationSetConfig` (held-out downstream-CE val set: FineWeb GPT-2 val tokens, first `10_485_760`). Real-trainer only. |
| `eval_corpus` | list[str] | built-in | Held-out reference corpus for the leakage term. |

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
- `trainer.py` — `ProxyStudentTrainer` interface, heuristic + GPU-sandbox backends.
- `docker_backend.py` — selectable Docker training backend (`DockerRuntimeClient`/`DockerRunRequest`) driving verifiers' v1 `DockerRuntime`, reachable on a remote host via `DOCKER_HOST=ssh://...`.
- `modal_backend.py` — Modal GPU sandbox adapter for `trainer_backend="modal"`; uploads corpus and training script to a `modal.Sandbox` and parses `RESULT_JSON` from stdout. `modal>=0.73` is a base dependency.
- `student_model.py` — modern modded-nanogpt proxy-student architecture embedded into the sandbox script.
- `student_train.py` — AdamW schedule, contiguous batching, and multi-run averaging, also embedded into the sandbox script.
- `rewards.py` — `CuratorScorer`, the framework-agnostic heavy scoring pass.
- `rollout_state.py` — typed `CuratorState` plus `RolloutStore` accessors.
- `taskset.py` — `CuratorTaskset`, manifest parsing/recovery, `finalize()`, decorated rewards/metrics, and `@vf.stop` turn cap. No MCP tool server.
- `tasks.py` — four typed v1 curation tasks and their user prompts.
- `eval_corpus.py` — small built-in held-out reference corpus for the leakage term.

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
