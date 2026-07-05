# Configuration

`load_environment` accepts only declared keyword arguments. Misspelled or stale
fields raise `TypeError` instead of being silently ignored.

## Environment arguments

### Task and harness

| Argument | Type | Default | Effect |
| --- | --- | --- | --- |
| `cutoff_date` | `str` | `"2024-12-31"` | Latest repository modification date the agent is instructed to use |
| `token_budget` | `int` | `1_000_000` | Manifest default and source-weight allocation budget |
| `hf_token_env` | `str` | `"HF_TOKEN"` | Environment variable required during rollout setup |
| `manifest_filename` | `str` | `"manifest.json"` | Root-level filename read from `/workspace` during finalization |
| `candidate_limit` | `int` | `8` | Maximum IDs used by trace-based manifest recovery |
| `sample_docs_per_source` | `int` | `64` | Hard upper bound on rows requested from each source |
| `allow_local_sources` | `bool` | `true` | Permit manifests to pull workspace-local text/JSONL files at scoring time |
| `max_local_source_bytes` | `int` | `33_554_432` | Per-local-source transfer cap; valid range 1 byte through 1 GiB |
| `max_turns` | `int` | `64` | Harness safety cap; not shown to the agent or used in scoring |
| `harness_id` | `str` | `"bash"` | Bundled Verifiers harness ID, including `bash`, `codex`, and `mini_swe_agent` |

`token_budget` is the only agent-facing optimization budget. There is no
discovery-call/output allocation, and `max_turns` is only a safety backstop.

The source fetch path probes for `{dataset_name}.py` before loading. If that file
exists, the source records the permanent `script_dataset` error and is attempted
only once. The current Verifiers dependency requires `datasets>=3`, which removed
script execution; use a data-only export.

`allow_local_sources=false` makes local entries fail soft with a typed empty
source while leaving Hub entries unchanged. The byte cap is enforced by
`head -c` in the live harness runtime before stdout crosses the runtime boundary.

### Reward

| Argument | Default | Effect |
| --- | ---: | --- |
| `alpha_perf` | `1.0` | Multiplier on positive proxy performance |
| `lambda_cost` | `0.1` | Multiplier on the cost penalty |
| `lambda_leakage` | `1.0` | Multiplier on the leakage penalty |
| `perf_baseline_loss` | `log(50304)` | No-information loss used by baseline-relative performance |
| `baseline_relative_perf` | `true` | Use bounded relative loss improvement; otherwise use `exp(-loss)` |
| `decon_binary` | `"decon"` | Path to the decon Rust binary; falls back to the vendored `decon/bin/decon` |
| `decon_evals_dir` | `None` | Override directory for the bundled benchmark eval sets; defaults to `decon/bundled-evals/` |
| `decon_threshold` | `0.2` | decon `--contamination-score-threshold` |
| `screen_val_set` | `true` | Also screen the corpus against the held-out val set (detokenised ephemerally; never exposed to the agent) |

### Reliability and concurrency

| Argument | Default | Effect |
| --- | ---: | --- |
| `max_concurrent_fetches` | `8` | Loop-local bound on active Hub fetches |
| `max_concurrent_training` | `1` | Loop-local bound on active GPU training commands |
| `fetch_timeout_seconds` | `30.0` | Base timeout for each external fetch attempt |
| `fetch_timeout_per_doc_seconds` | `0.25` | Additional timeout per requested streaming document |
| `fetch_max_attempts` | `3` | Maximum attempts for transient failures |

### Trainer selection

| Argument | Default | Effect |
| --- | --- | --- |
| `use_real_trainer` | `false` | Select actual GPU training instead of the heuristic |
| `proxy_student` | `{}` | Nested `ProxyStudentConfig` overrides |
| `validation_set` | `{}` | Nested `ValidationSetConfig` overrides |

## Proxy-student fields

### Model and optimization

| Field | Default |
| --- | ---: |
| `n_layer` | `4` |
| `n_head` | `4` |
| `n_embd` | `256` |
| `mlp_ratio` | `4` |
| `lm_head_softcap` | `30.0` |
| `num_value_embeds` | `3` |
| `block_size` | `256` |
| `batch_size` | `16` |
| `steps` | `200` |
| `train_token_budget` | `null` |
| `learning_rate` | `3e-4` |
| `seed` | `0` |
| `val_fraction` | `0.1` |
| `weight_decay` | `0.1` |
| `adam_beta1` / `adam_beta2` | `0.9` / `0.95` |
| `adam_eps` | `1e-8` |
| `grad_clip` | `1.0` |
| `warmup_steps` | derived |
| `lr_min_ratio` | `0.1` |
| `n_train_runs` | `1` |

When `train_token_budget` is set:

```text
effective_steps = ceil(train_token_budget / (batch_size * block_size))
```

Otherwise `steps` is used. The effective number of trained tokens is always
`effective_steps * batch_size * block_size`.

The model validators require an even depth, `n_embd` divisible by `n_head`, and
a resulting head dimension divisible by four.

### Backend and resources

| Field | Default | Notes |
| --- | --- | --- |
| `runtime_backend` | `null` | Required as `"docker"` or `"modal"` when `use_real_trainer=true` |
| `docker_image` | `pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime` | Runtime image for Docker or Modal; production images should also include `hf`, `tiktoken`, and `tqdm` |
| `docker_host` | `null` | Retained config field; non-null is rejected for Docker |
| `modal_gpu` | `"L4"` | `"A100"` maps to `"A100-80GB"`; unknown values fall back to L4 |
| `gpu_count` | `1` | Docker maps a positive value to the runtime GPU request |
| `cpu_cores` | `4` | Runtime/sandbox CPU request |
| `memory_gb` | `16` | Runtime/sandbox memory request |
| `disk_size_gb` | `20` | Runtime/sandbox disk request |
| `max_corpus_chars` | derived | Explicit upload-text cap when set |
| `timeout_minutes` | derived | Training command/sandbox timeout |
| `upload_timeout_seconds` | `120.0` | Per-file runtime write timeout |

By default, `max_corpus_chars` grows with the effective train-token count,
floored at 5,000,000 and capped at 2,000,000,000. The derived timeout includes
15 setup minutes plus a conservative throughput estimate, with a 30-minute
floor. Modal caps derived/explicit timeouts at 24 hours; self-hosted Docker
does not.

## Validation-set fields

Real training and the default leakage reference use the first 10,485,760 tokens
in the NanoGPT speedrun FineWeb validation shard:

| Field | Default |
| --- | --- |
| `dataset_id` | `"kjj0/fineweb10B-gpt2"` |
| `filename` | `"fineweb_val_000000.bin"` |
| `repo_type` | `"dataset"` |
| `tokenizer` | `"gpt2"` |
| `val_tokens` | `10_485_760` |

The heuristic trainer ignores this configuration for its synthetic performance
signal, but leakage detection still uses it.

## Canonical smoke run

The repository keeps one exhaustive local run config. It uses DeepSeek V4 Flash
to curate a 25M-token corpus and the real proxy trainer on a local GPU Docker:

```bash
prime env install pretrain-data-curator -p ./environments
cd environments/pretrain_data_curator
set -a; source ../../secrets.env; set +a
prime eval run configs/eval/deepseek-v4-flash-smoke.toml
```

Do not add `--skip-upload`; canonical Prime eval runs save results for the
private Evaluations tab and `prime eval tui`. The config explicitly enumerates
all loader, `ProxyStudentConfig`, and `ValidationSetConfig` fields; keep its
exhaustiveness test synchronized when the source models change.

## Agent self-score

Rollout setup copies `self_score.py` into the harness workspace. It accepts a
draft manifest and a bounded candidate-sample limit:

```bash
python self_score.py draft.json --limit 8
```

This script is independent of `use_real_trainer`: it always returns a cheap
development heuristic. It samples only manifest sources, blocks the configured
validation repository by a non-reversible digest, and never loads the final
validation shard or leakage reference. See [Agent workflow](agent-workflow.md)
for output fields and limitations.

## Docker harness runtime

Docker uses one GPU-capable v1 runtime for agent discovery and training. Build
the reference image:

```bash
docker build -t pretrain-data-curator:gpu \
  -f environments/pretrain_data_curator/Dockerfile.runtime \
  environments/pretrain_data_curator
docker run --rm --gpus 1 pretrain-data-curator:gpu nvidia-smi
```

Then run the native v1 taskset command:

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

The installed `prime eval run` wrapper does not expose
`--harness.runtime.*`; use the native command for this local Docker mode.
Docker Engine, the NVIDIA Container Toolkit, and a co-located daemon are
required. Leave `proxy_student.docker_host` and any **remote** `DOCKER_HOST`
unset. A local Unix-socket `DOCKER_HOST` is acceptable; Docker Desktop under WSL
may require its local proxy socket as described in
[Troubleshooting](troubleshooting.md#docker-desktop-uses-an-invalid-npipe-context).
The values above are intentionally smoke-sized and have been exercised
end-to-end with DeepSeek V4 Flash; they are not a scientific training budget.

## Modal harness runtime

Modal also uses one GPU runtime for the full rollout. Set:

```json
{
  "use_real_trainer": true,
  "proxy_student": {
    "runtime_backend": "modal",
    "modal_gpu": "H100",
    "train_token_budget": 5000000
  }
}
```

The environment worker needs:

```bash
export MODAL_TOKEN_ID=...
export MODAL_TOKEN_SECRET=...
```

`load_environment` validates those variables immediately for the Modal path.
The registry image must support both bash/`hf` discovery and CUDA PyTorch
training. `Dockerfile.runtime` is the reference dependency contract, but Modal
must pull a published registry image; it cannot build the local Dockerfile.

## Reference config

[`../configs/eval/deepseek-v4-flash-smoke.toml`](../configs/eval/deepseek-v4-flash-smoke.toml)
is both the only checked-in local run config and the exhaustive field reference.
Keep it synchronized with `load_environment`, `ProxyStudentConfig`, and
`ValidationSetConfig`.
