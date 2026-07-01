# Configuration

## Docker harness runtime

Set `use_real_trainer=true` and
`proxy_student.trainer_backend="docker"` to place the bash harness and
proxy-student training in one verifiers v1 `DockerConfig`. The loader maps:

| Proxy-student field | Docker runtime field |
| --- | --- |
| `docker_image` | `image` |
| `gpu_count` | `gpu` (`None` when zero) |
| `cpu_cores` | `cpu` |
| `memory_gb` | `memory` |
| `disk_size_gb` | advisory `disk` |

The work directory is `/workspace`. The configured image must support both
rollout phases: bash and the Hugging Face `hf` CLI for discovery, plus Python,
CUDA PyTorch, NumPy, and tokenizer dependencies for scoring. Build the included
image and select it explicitly:

```bash
docker build -t pretrain-data-curator:gpu \
  -f environments/pretrain_data_curator/Dockerfile.runtime \
  environments/pretrain_data_curator

cd environments/pretrain_data_curator
uv run eval pretrain-data-curator \
  --harness.id bash \
  --harness.runtime.type docker \
  --harness.runtime.image pretrain-data-curator:gpu \
  --harness.runtime.workdir /workspace \
  --harness.runtime.gpu 1 \
  --taskset.use-real-trainer true \
  --taskset.token-budget 500000 \
  --taskset.max-turns 12 \
  --taskset.proxy-student \
    '{"trainer_backend":"docker","docker_image":"pretrain-data-curator:gpu","train_token_budget":5000000,"gpu_count":1}' \
  -m nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 -n 1 -r 1
```

The package exports `CuratorTaskset` through `__all__`, so
`eval pretrain-data-curator` is the native v1 taskset path; do not use legacy
`--id`. The currently released `prime eval run` wrapper cannot pass
`--harness.runtime.*`, and therefore cannot launch this local Docker mode.
Owner-qualified environment ids select the published Hub package, not local
source.

This mode assumes the rollout worker, Docker CLI, Docker daemon, and GPU are on
the same machine. `docker_host` must be unset; SSH-remote Docker orchestration is
not supported by the shared-runtime design. Prime backend configuration is
unchanged.

On Docker Desktop under WSL2, the environment automatically binds the
interception server to the WSL interface because container localhost belongs to
Docker Desktop's VM. Set `PDC_DOCKER_HOST_IP` only if automatic route detection
chooses the wrong WSL address.

## Modal harness runtime

Set `use_real_trainer=true` and
`proxy_student.trainer_backend="modal"` to place the bash harness and
proxy-student training in one Verifiers v1 `ModalConfig`. The loader maps:

| Proxy-student field | Modal runtime field |
| --- | --- |
| `docker_image` | `image` |
| `modal_gpu` | `gpu` (`H100`, `H200`, `A100-80GB`, otherwise `L4`) |
| `cpu_cores` | `cpu` |
| `memory_gb` | `memory` |
| `disk_size_gb` | advisory `disk` |

The work directory is `/workspace`. As with Docker, the registry image must
support both the bash/Hugging Face discovery phase and CUDA PyTorch scoring.
When no image is set, Modal uses
`pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime`; the training script installs
`tiktoken` on demand, and the harness bootstraps its own uv-script dependencies.
That bare image does not include the Hugging Face `hf` CLI. Prime's default
image may have the same gap.

The shared agent prompt therefore starts its first discovery command with an
idempotent `command -v hf` check and, only when needed, a quiet
`pip install 'huggingface-hub>=0.34'`. The first `hf datasets ls`
search runs in the same shell command, so a preinstalled CLI adds no extra
turn. On a bare image the fallback is functional but slower, requires outbound
package-index access, and consumes paid Modal GPU time while installing.

For production, use a registry-hosted image containing the dependencies from
`Dockerfile.runtime` and set it as `docker_image`. Modal can only pull a
registry image through `ModalConfig`; it cannot build that Dockerfile. No
prebuilt project image is currently published, so publishing and maintaining
one remains an infrastructure TODO for an operator with registry credentials.

Set `MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` on the env-server. The loader
validates both variables before constructing a Modal environment. The
env-server itself remains CPU-only and needs no Docker daemon: Verifiers calls
Modal over outbound HTTPS and runs the complete harness remotely. This changes
resource duration from the old design: the Modal GPU sandbox is allocated for
dataset discovery and finalization as well as proxy training, so billing covers
the full rollout.

## Deadlines and concurrency

`ProxyStudentConfig.effective_timeout_minutes` remains the training-command
deadline. For Docker and Modal, each native v1 task declares a framework
scoring timeout above that deadline, plus a margin for corpus writes and other
scoring work. This allows budget-derived multi-hour runs to complete without
inheriting a short reward-computation timeout. Modal's runtime lifetime is
limited to 24 hours, so larger explicit training deadlines are rejected.

`max_concurrent_training` is a loop-local semaphore around the runtime writes
and training command. It limits active GPU training jobs independently of the
number of concurrent rollouts. Each rollout still owns its own runtime
container or sandbox.
