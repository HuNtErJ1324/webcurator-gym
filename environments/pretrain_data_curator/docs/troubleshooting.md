# Troubleshooting

## The run has zero reward

Check metrics in this order:

1. `finalized`: zero means no usable manifest was recovered.
2. `external_failure`, `tool_error_count`, `trainer_error_msg`: nonzero values
   indicate infrastructure failure.
3. `num_sources` and `corpus_tokens`: zero means sources did not materialize.
4. `perf_loss`: zero may represent the metric's display value for a nonfinite
   trainer loss; use the failure diagnostics with it.
5. Reward components: cost or leakage can offset positive performance.

An unfinalized rollout still pays discovery cost. Inspect the last assistant
messages for truncated JSON, an empty `sources` list, or prose without a complete
object.

## `HF_TOKEN` is missing

Typical error:

```text
Hugging Face token environment variable 'HF_TOKEN' is required for rollouts
```

Set the variable in the **environment worker**, not only in an unrelated local
shell:

```bash
export HF_TOKEN=hf_...
```

If `hf_token_env` was changed, set that named variable. Credentials are checked
when the taskset first constructs the live Hub client, not at package import.

## A dataset produces no documents

Inspect:

- repository access and gating;
- exact config and split names;
- whether the source has a usable textual field;
- filters that may reject every row;
- very small `max_tokens` or `max_docs`;
- `tool_errors` classification in the trace.

Leaving `text_field=null` enables fallback detection. If a dataset has no default
config, the client tries a stable English/default choice, but explicit config is
safer.

Permanent missing/auth/config/split/field failures are not retried. Network,
timeout, and unknown failures retry up to `fetch_max_attempts`.

## `hf` is not found in the harness

The system prompt's first command conditionally installs
`huggingface-hub>=0.34`. If that fails:

- confirm the runtime has `pip`;
- confirm outbound package-index access;
- for Docker/Modal production, build `hf` into the image rather than relying on
  cold installation;
- verify the agent actually ran the bootstrap command instead of printing it.

The environment-side package installation does not guarantee that a remote
Docker/Modal harness image contains the CLI.

## Docker cannot access the GPU

Verify outside the environment first:

```bash
docker run --rm --gpus 1 pretrain-data-curator:gpu nvidia-smi
```

If it fails, fix the host NVIDIA driver, Docker Engine, or NVIDIA Container
Toolkit before debugging the taskset.

The Docker path requires a local/co-located daemon. `proxy_student.docker_host`
is rejected, and an ambient remote `DOCKER_HOST` is unsupported.

## Docker Desktop uses an invalid `npipe` context

Inside WSL, Docker Desktop can select a Windows `npipe://` context that the Linux
CLI cannot use. Symptoms include `protocol not available` or a Docker CLI panic.
Point the CLI at Docker Desktop's local WSL proxy socket:

```bash
export DOCKER_HOST=unix:///mnt/wsl/docker-desktop/shared-sockets/guest-services/docker.proxy.sock
docker version
```

This is still a co-located local daemon. It is distinct from unsupported remote
Docker orchestration such as `ssh://gpu-host`.

## Docker harness cannot reach interception under WSL2

Docker Desktop runs containers in a VM, so container localhost is not the WSL
host. `DockerHostReachability` detects WSL and advertises the route address.

If detection chooses the wrong interface:

```bash
export PDC_DOCKER_HOST_IP=<reachable-wsl-address>
```

Use the address reachable from the Docker container, not necessarily
`127.0.0.1`.

## Docker image starts but training dependencies are missing

One image hosts both agent discovery and proxy training. It needs:

- bash, `hf`, and basic shell utilities;
- Python and CUDA-enabled PyTorch;
- NumPy;
- `tiktoken` and `tqdm`, or package-index access for fallback installation.

Build [`../Dockerfile.runtime`](../Dockerfile.runtime) and select the resulting
tag in both the harness runtime and `proxy_student.docker_image`.

## Modal credentials fail

The Modal loader validates both:

```text
MODAL_TOKEN_ID
MODAL_TOKEN_SECRET
```

Set them in the environment worker. The worker also needs outbound HTTPS to the
Modal API. It does not need a local Docker daemon.

If sandbox creation succeeds but discovery fails, inspect the registry image for
`hf`/shell support and outbound Hub/package-index access.

## Trainer times out

`timeout_minutes` bounds the training command/sandbox lifetime.
`train_token_budget` also increases the derived timeout when no explicit value
is set.

Check:

- effective steps from batch, block, and token budget;
- `n_train_runs`, which multiplies work;
- image-pull and dependency-install time;
- corpus and validation upload size;
- `max_concurrent_training`;
- Prime/Modal's 24-hour ceiling.

For Docker/Modal, the framework scoring deadline includes a margin beyond the
command timeout. If a custom framework config imposes a shorter outer deadline,
the outer cancellation still wins.

## Trainer exits without `RESULT_JSON`

The generated script must print one final line beginning with `RESULT_JSON`.
Missing or malformed output becomes `TrainerError`. Inspect the preserved stderr
tail for:

- CUDA out-of-memory;
- invalid model dimensions;
- missing tokenizer/PyTorch dependencies;
- corrupt validation shard;
- nonfinite training;
- process termination by runtime limits.

The scorer records the error and returns zero performance rather than raising it
through the full eval.

## Prime eval and native `eval` select different code

An owner-qualified ID such as `hunterj/pretrain-data-curator` resolves a
published Hub environment. To test this checkout:

```bash
prime env install pretrain-data-curator -p ./environments
prime eval run pretrain-data-curator ...
```

The Docker harness-runtime example uses native v1 `uv run eval` because the
installed `prime eval run` wrapper does not pass through
`--harness.runtime.*`.

## The heuristic and real trainer disagree

That is expected. The heuristic is a deterministic engineering surrogate, not a
calibrated predictor of real held-out cross-entropy. Use it to validate mechanics
and use a fixed real-trainer configuration for scientific comparisons.
