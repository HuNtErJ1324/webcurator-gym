# pretrain-data-curator

An environment where an agent curates an **LLM pretraining dataset** from the
pre-cutoff Hugging Face universe. The curated mixture is scored by training a
**fixed GPT-2-scale proxy-student where everything but the data is held
constant**, combined with cross-entropy performance and leakage terms.

## Quickstart

```bash
prime env install pretrain-data-curator -p ./environments
cd environments/pretrain_data_curator
set -a; source ../../secrets.env; set +a
prime eval run configs/eval/deepseek-v4-flash-smoke.toml
```

The checked-in smoke config is the single local run configuration. It uses
DeepSeek V4 Flash for curation and a real Modal H100 proxy trainer, and explicitly
lists every loader, student, and validation option. `HF_TOKEN`,
`MODAL_TOKEN_ID`, and `MODAL_TOKEN_SECRET` must be exported. For setup and
zero-score failures, inspect rollout traces in the generated bench site under `docs/site/`.

## Overview

- **Environment ID**: `pretrain-data-curator`
- **Type**: toolless native verifiers v1 `CuratorTaskset` (`vf.Taskset`) — the agent discovers datasets using the `hf` CLI in its own shell; no MCP tool server is exposed
- **External service**: Hugging Face Hub API (search + streaming sampling)
- **Required secret**: `HF_TOKEN` by default (checked in taskset setup before each rollout)
- **Proxy-student training**: optional GPU training, off by default; selectable co-located harness-runtime Docker backend **or Modal** (recommended for Hosted Training)

The agent searches Hugging Face for datasets at or before a cutoff date,
inspects them, assembles a weighted and filtered **manifest** `M`, previews
statistics, and finalizes. The finalized manifest is materialized into a corpus
and scored.

## Reward

```text
R(M, H) = alpha_perf * Perf(M)
        - lambda_leakage * Leakage(M, H)
```


By default, `Perf(M)` uses a convex power-law scaling (`perf_scaling_exponent`,
default `2.0`) from the neutral loss baseline to the nanoGPT speedrun target:

```text
p = (perf_baseline_loss - loss) / (perf_baseline_loss - perf_target_loss)
Perf = p^γ  if p ≥ 0;  Perf = p  if p < 0   (γ = perf_scaling_exponent)
```

`perf_target_loss` defaults to `3.28`, which maps to exactly `1.0`. The value is
not clamped: it is negative when loss is worse than the baseline and greater
than `1.0` when loss beats the target. Setting `γ=1.0` recovers the previous
linear formula.

| Term | Meaning | Default weight |
| --- | --- | --- |
| `Perf` | Proxy-student cross-entropy performance on the curated corpus | `alpha_perf=1.0` |
| `Leakage` | Token-weighted [decon](https://github.com/allenai/decon) n-gram contamination vs public benchmarks **and** the held-out val set | `lambda_leakage=1.0` |

Each term is also emitted as a metric (`perf_loss`, `perf_accuracy`,
`perf_vs_baseline`, `train_flops`, `corpus_tokens`, `num_sources`,
`leakage_score`, `num_contaminated_matches`, `decon_error`,
`finalized`). Local-source provenance is exposed as
`local_source_count`, `local_source_bytes`, `local_source_truncated`, and
`val_set_access`. Two further zero-weight **diagnostics**,
`tool_error_count` and `external_failure`, separate bad curation from
external/infrastructure failure (a flaky Hub or sandbox). See
[`docs/README.md`](docs/README.md).

### Proxy-student backends

`Perf(M)` is computed by a `ProxyStudentTrainer`:

- **`HeuristicProxyTrainer`** (default): a deterministic CPU surrogate that
  predicts loss/accuracy from corpus scale, cleanliness, and diversity. It is a
  **dev/smoke surrogate, not a real training signal** — it lets the environment
  run and be tested without a GPU, but its `Perf` is a proxy, not a measurement.
  It trains no model, so it does **not** compute per-token CE on the held-out
  validation set. Leakage is computed separately by decon (see below).
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

Training FLOPs are reported as the zero-weight ``train_flops`` metric.

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
  Modal covers discovery through scoring, not only the training phase.

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
  --taskset.max-turns 64 \
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
socket is valid; Docker Desktop under WSL may require
`export DOCKER_HOST=unix:///mnt/wsl/docker-desktop/shared-sockets/guest-services/docker.proxy.sock`.

The Docker and Modal proxy-student commands retain their budget-derived
deadline and structured exit/`RESULT_JSON` validation. Timeout or cancellation
stops the shared runtime immediately, and the rollout's `finally` performs the
idempotent teardown backstop. `max_concurrent_training` bounds training commands
independently of rollout concurrency. The framework scoring timeout is expanded
from the trainer deadline so multi-hour jobs are not cancelled by a short
reward timeout.

## Agent interface

`CuratorTaskset` deliberately exposes **no MCP tool server** (`Taskset.tools` is
not overridden). The agent receives one initial user prompt and no separate
system prompt, so bash, Codex, and mini-SWE-agent harnesses see the same task.
The prompt is method-open: the shell, internet, `hf`, and other harness tools are
available without a prescribed command sequence. There are no `curator_*` API
commands.

Taskset setup also writes `self_score.py` into the rollout workspace. An agent
can run `python self_score.py draft.json` with agent-chosen flags (`--limit`,
`--max-steps`, `--max-corpus-chars`, `--train-timeout`) to sample candidate
sources, estimate proxy CE and token fill, and score a draft manifest during
development. This signal does not load the final validation shard; the configured
validation repository is blocked by a hash so its identity is not disclosed by
the script.

| `hf` command | Purpose |
|---|---|
| `hf datasets ls --search "<query>" --sort downloads --limit 5 \| head -c 6000` | Search and rank dataset repositories without flooding model context. |
| `hf datasets ls --search "<q>" --format json --expand downloads,likes,lastModified --limit 5 \| head -c 6000` | JSON-formatted search; deliberately omit high-volume `tags`. |
| `hf datasets info <dataset_id> --expand downloads,likes,tags \| head -c 6000` | Inspect one shortlisted repository. |
| `hf version`, `hf env`, `hf auth`, `hf cache` | Local setup/status — no network. |
| `/workspace/manifest.json` | The agent's curation decision; read by `finalize()`. |

The agent must write a non-empty manifest JSON object to
`/workspace/manifest.json`. File existence is the completion signal; no sentinel
token or special final-message format is required. The minimal schema:

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
source with a workspace-relative `local_path`.
`config`, `split`, `text_field`, `filters`, `max_docs`, and `max_tokens` are
optional. Supported filter kinds: `min_chars`, `max_chars`, `min_tokens`,
`max_symbol_ratio`, `min_alpha_ratio`, `drop_regex`, `keep_regex`, `dedup_exact`.
When a multi-config dataset has no default and `config` is omitted, the
materializer chooses a stable English/default config (for example `en` or a
config ending in `.en`) before falling back to the first advertised config.

See [`docs/README.md`](docs/README.md) for the eval visualization site and builder.

## Documentation

| Page | Use it for |
| --- | --- |
| [Bench site builder](docs/README.md) | Generate a PostTrainBench-style leaderboard from full 400M runs in `outputs/evals-400m/` |
| [Codebase explanation](docs/site_builder/assets/codebase.html) | Bench-site page (`codebase.html`) covering the rollout lifecycle, repository layout, and module map |
| [Manifest-backed training debug workflow](docs/debug_manifest_training.md) | Materialize a curated corpus from an explicit local manifest once, then repeatedly debug the NanoGPT/proxy training path against the same bundle without re-curating. |

## Install And Eval

```bash
prime env install pretrain-data-curator -p ./environments
cd environments/pretrain_data_curator
set -a; source ../../secrets.env; set +a
prime eval run configs/eval/deepseek-v4-flash-smoke.toml
```

See the [comprehensive smoke config](configs/eval/deepseek-v4-flash-smoke.toml)
for the complete local-run argument surface. The Docker alternative requires a
working local GPU daemon; this config uses Modal because the validated WSL host
did not expose a usable Docker socket.

## Required Environment Variables

Hugging Face credentials are validated eagerly in taskset `setup()` before the
agent starts. Constructing the environment itself does not require `HF_TOKEN`,
but every rollout does.

| Variable | Default | Description |
| --- | --- | --- |
| `HF_TOKEN` | yes | Token passed to `huggingface_hub.HfApi` and streaming loads. |

## Environment Args

| Arg | Type | Default | Description |
| --- | --- | --- | --- |
| `cutoff_date` | str | `"2024-12-31"` | Latest allowed Hugging Face `lastModified` date. |
| `token_budget` | int | `1000000` | Target token budget for the mixture. |
| `hf_token_env` | str | `"HF_TOKEN"` | Env var checked for the HF token before a rollout starts. |
| `manifest_filename` | str | `"manifest.json"` | Manifest filename under `/workspace`; must be a root-level filename. |
| `candidate_limit` | int | `8` | Maximum dataset IDs used by trace-based manifest recovery/fallback only. |
| `allow_local_sources` | bool | `true` | Allow capped pulls of text/JSONL files created in the live bash workspace. |
| `max_local_source_bytes` | int | `33554432` | Maximum bytes transferred per local source before parsing. |
| `max_turns` | int | `64` | Generous harness safety cap; absent from the prompt, reward, and metrics. |
| `harness_id` | str | `"bash"` | Bundled Verifiers harness (`bash`, `codex`, `mini_swe_agent`, etc.). |
| `alpha_perf` | float | `1.0` | Cross-entropy performance weight. |
| `lambda_leakage` | float | `1.0` | Leakage penalty weight. |
| `perf_baseline_loss` | float | `log(50304)` | Neutral CE reference for relative performance. |
| `perf_target_loss` | float | `3.28` | Target CE that maps to `Perf=1.0` under baseline-relative scoring. |
| `perf_scaling_exponent` | float | `2.0` | Power-law exponent for near-target amplification; `1.0` recovers linear. |
| `baseline_relative_perf` | bool | `true` | Use target-scaled relative loss; `false` uses `exp(-loss)`. |
| `max_concurrent_fetches` | int | `8` | Bound on concurrent HF fetches (also the corpus-builder fetch limit). |
| `max_concurrent_training` | int | `1` | Bound on concurrent sandbox-training jobs (real trainer). |
| `fetch_timeout_seconds` | float | `30.0` | Base per-attempt timeout for external HF calls. |
| `fetch_timeout_per_doc_seconds` | float | `0.25` | Additional per-document timeout for streaming dataset fetches. |
| `fetch_max_attempts` | int | `3` | Max attempts (retry/backoff) for transient HF failures. |
| `use_real_trainer` | bool | `false` | Use the GPU sandbox proxy-student instead of the heuristic. |
| `proxy_student` | dict | `{}` | Overrides for `ProxyStudentConfig` (arch, `train_token_budget`, `gpu_count`, etc.). `train_token_budget` (≤ 1e9) derives `effective_steps` so *scheduled* presentations under `batch_stage_muls` meet the budget (ceiling: at most one final step of overshoot; see `batch_schedule.py`), and scales corpus-cap/timeout from `effective_train_tokens`. `train_microbatch_size` is memory-only and does not change the budget math. Selects the real-trainer backend via `runtime_backend` (`"docker"` / `"modal"`; required, no default, whenever `use_real_trainer=true`); for `"docker"`, set `docker_image` to a combined discovery/training image and leave `docker_host` unset; for `"modal"` see `modal_gpu` (default `"L4"`; also `"H100"`/`"H200"`/`"A100"`) and `gpu_count`, and set `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET`. |
| `validation_set` | dict | NanoGPT speedrun set | Overrides for `ValidationSetConfig` (held-out downstream-CE val set: FineWeb GPT-2 val tokens, first `10_485_760`). Used for proxy-student CE scoring, and — when `screen_val_set=true` — detokenised ephemerally as an extra decon leakage reference (never exposed to the agent). |
| `decon_binary` | str | `"decon"` | Path/name of the decon binary; falls back to the vendored `decon/bin/decon`. |
| `decon_evals_dir` | str \| None | `None` | Override for the benchmark eval-set directory; defaults to the bundled `decon/bundled-evals/`. |
| `decon_threshold` | float | `0.2` | decon `--contamination-score-threshold`. |
| `screen_val_set` | bool | `true` | Also screen the corpus against the held-out val set (in addition to public benchmarks). The val text is built server-side in an ephemeral temp dir and never written to the workspace, the agent container, or `decon/bundled-evals/`. |

Before streaming a source, the materializer checks for the Hugging Face
`{dataset_name}.py` convention. Script-backed sources fail once with
`error_kind="script_dataset"` and are not retried. Script execution is
unavailable because the pinned Verifiers package requires `datasets>=3`.

When streaming is unavailable, a local source can bridge a genuine downloaded
dataset into scoring without executing repository code. Local paths are
validated, transferred through the live runtime with `head -c`, and audited
with provenance metrics.

## Module Layout

- `__init__.py` — `_bootstrap_verifiers_v1()`: patches `verifiers.__path__` at import time so the real v1 is importable inside Prime's Hosted Training orchestrator (which pre-loads a `verifiers==0.0.0` stub).
- `pretrain_data_curator.py` — `load_environment` entry point; builds `CuratorTasksetConfig` and returns a `hosted_compat.Environment`.
- `hosted_compat.py` — `Environment`: multiple-inheritance v0/v1 bridge. Derives from both `vf.Environment` and `legacy_vf.Environment`; delegates rollout work to the v1 episode engine and translates `vf.Trace` to the v0 `State` / `RolloutTiming` / trajectory format consumed by `prime eval` and Hosted Training.
- `models.py` — Pydantic contracts (`Manifest`, `Source`, `FilterSpec`, `CuratorConfig`, `ProxyStudentConfig`, ...).
- `hf_access.py` — `DatasetSearchClient` Protocol, live HF client, cutoff/query helpers. Setup checks credentials before rollout; the client checks again at first Hub use.
- `hf_cli_parse.py` — parse `hf` CLI argv from shell/command text for manifest recovery and val-set access detection.
- `corpus.py` — `CorpusBuilder` + `DocumentFilter` (materialize manifest into documents).
- `leakage.py` — `DeconLeakageDetector` (decon n-gram contamination vs bundled benchmarks + optional ephemeral held-out val screen), `LeakageScores`, `DeconError`, and the token-weighted `_reduce_report`.
- `val_set.py` — held-out validation token stream (`ValidationSetConfig`, `ValTokenLoader`, `.bin` parser); NanoGPT-speedrun set by default.
- `trainer.py` — `ProxyStudentTrainer` interface, the heuristic backend, and `RuntimeSelectedTrainer` (dispatches to the docker/modal backend matching the live harness runtime's type).
- `docker_backend.py` — proxy-student execution on the rollout-owned v1 Docker runtime, including training limits, timeout/cancellation teardown, and structured result parsing.
- `modal_backend.py` — proxy-student execution on the rollout-owned v1 Modal runtime, including GPU mapping, training limits, timeout/cancellation teardown, and structured result parsing.
- `student_model.py` — modern modded-nanogpt proxy-student architecture embedded into the sandbox script.
- `student_train.py` — AdamW schedule, contiguous batching, and multi-run averaging, also embedded into the sandbox script.
- `rewards.py` — `CuratorScorer`, the framework-agnostic heavy scoring pass.
- `rollout_state.py` — typed `CuratorState` plus `RolloutStore` accessors.
- `taskset.py` — `CuratorTaskset`, manifest parsing/recovery, `finalize()`, decorated rewards/metrics, and `@vf.stop` turn cap. No MCP tool server.
- `tasks.py` — one typed v1 curation task rendered as the initial user prompt.
- `self_score.py` — renders the standalone leakage-safe development proxy copied into each rollout workspace; runs decon against the bundled **benchmarks only** (never the held-out val set).

## Notes And Limitations

- Leakage detection is deterministic: the decon Rust binary runs as a subprocess
  over the materialized corpus against baked benchmark eval sets (and, when
  `screen_val_set=true`, an ephemeral detokenised val reference), reduced to a
  token-weighted scalar. A detector failure raises `DeconError` and is surfaced
  as `decon_error`/`external_failure` rather than a silent `0.0`.
- Documents are fetched **once per real fetch**: a per-rollout cache keyed by
  `(dataset_id, config, split, text_field, n)` with per-key single-flight so
  previews and final scoring observe identical docs.
- External (HF/sandbox) calls are wrapped with timeout + retry and a typed
  `DatasetAccessError`; on failure tools/scoring **degrade** to a defined sentinel
  (empty slice / infinite-loss `TrainResult`) and record diagnostics rather than
  crashing the rollout.
- Token counts use `max(word_count, character_count // 4)` on the env side; the
  real trainer tokenizes inside the sandbox.
- Filtering is expressed via the structured `filters` argument.
