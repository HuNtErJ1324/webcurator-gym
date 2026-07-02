# Configuration

`load_environment` accepts only declared keyword arguments. Misspelled or stale
fields raise `TypeError` instead of being silently ignored.

## Environment arguments

### Task and discovery

| Argument | Type | Default | Effect |
| --- | --- | --- | --- |
| `cutoff_date` | `str` | `"2024-12-31"` | Latest repository modification date the agent is instructed to use |
| `token_budget` | `int` | `1_000_000` | Manifest default and source-weight allocation budget |
| `hf_token_env` | `str` | `"HF_TOKEN"` | Environment variable read on first materialization |
| `candidate_limit` | `int` | `8` | Maximum IDs used by trace-based manifest recovery |
| `scan_limit` | `int` | `50` | Input to the prompt's suggested discovery-round count |
| `sample_docs_per_source` | `int` | `64` | Hard upper bound on rows requested from each source |
| `max_turns` | `int` | `12` | Agent-loop turn cap |

`scan_limit` must be at least `candidate_limit`. Both are configuration and
prompt/recovery controls; the agent still chooses actual `hf --limit` values.

### Reward

| Argument | Default | Effect |
| --- | ---: | --- |
| `alpha_perf` | `1.0` | Multiplier on positive proxy performance |
| `lambda_cost` | `0.1` | Multiplier on the cost penalty |
| `lambda_leakage` | `1.0` | Multiplier on the leakage penalty |
| `perf_baseline_loss` | `log(50304)` | No-information loss used by baseline-relative performance |
| `baseline_relative_perf` | `true` | Use bounded relative loss improvement; otherwise use `exp(-loss)` |
| `eval_corpus` | built-in list | Override leakage-reference text |

### Reliability and concurrency

| Argument | Default | Effect |
| --- | ---: | --- |
| `max_concurrent_fetches` | `8` | Loop-local bound on active Hub fetches |
| `max_concurrent_training` | `1` | Loop-local bound on active GPU training commands |
| `fetch_timeout_seconds` | `30.0` | Timeout for each external fetch attempt |
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

Real training defaults to the first 10,485,760 tokens in the NanoGPT speedrun
FineWeb validation shard:

| Field | Default |
| --- | --- |
| `dataset_id` | `"kjj0/fineweb10B-gpt2"` |
| `filename` | `"fineweb_val_000000.bin"` |
| `repo_type` | `"dataset"` |
| `tokenizer` | `"gpt2"` |
| `val_tokens` | `10_485_760` |

The heuristic trainer ignores this configuration.

## Heuristic smoke run

The canonical local lifecycle is:

```bash
prime env install pretrain-data-curator -p ./environments
prime eval run pretrain-data-curator -m openai/gpt-4.1-mini -n 4 -r 1
```

To override a few fields:

```bash
prime eval run pretrain-data-curator \
  -m openai/gpt-4.1-mini -n 4 -r 1 \
  -a '{"max_turns": 24, "token_budget": 2000000, "sample_docs_per_source": 256}'
```

Do not add `--skip-upload`; canonical Prime eval runs save results for the
private Evaluations tab and `prime eval tui`.

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
  --taskset.max-turns 12 \
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

[`../configs/eval/example.toml`](../configs/eval/example.toml)
lists every environment, proxy-student, and validation-set field at its current
source default. Keep it synchronized with `load_environment`,
`ProxyStudentConfig`, and `ValidationSetConfig`.
