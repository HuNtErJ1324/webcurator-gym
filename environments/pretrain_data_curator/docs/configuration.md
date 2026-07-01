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

prime eval run pretrain-data-curator \
  -a '{"use_real_trainer":true,"proxy_student":{
    "trainer_backend":"docker",
    "docker_image":"pretrain-data-curator:gpu",
    "gpu_count":1
  }}'
```

This mode assumes the rollout worker, Docker CLI, Docker daemon, and GPU are on
the same machine. `docker_host` must be unset; SSH-remote Docker orchestration is
not supported by the shared-runtime design. Prime and Modal backend
configuration is unchanged.

## Deadlines and concurrency

`ProxyStudentConfig.effective_timeout_minutes` remains the training-command
deadline. For Docker, the environment also sets the framework scoring timeout
to that deadline plus a margin for corpus writes and other scoring work. This
allows budget-derived multi-hour runs to complete without inheriting a short
reward-computation timeout.

`max_concurrent_training` is a loop-local semaphore around the runtime writes
and training command. It limits active GPU training jobs independently of the
number of concurrent rollouts. Each rollout still owns its own runtime
container.
