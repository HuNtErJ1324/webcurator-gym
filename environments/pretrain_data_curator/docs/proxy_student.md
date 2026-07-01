# Proxy student

The real proxy student trains during `Phase.SCORING`, before the rollout runtime
is torn down. The Docker and Modal backends use the same runtime that already
hosted agent setup, dataset discovery, and finalization:

1. `Taskset.score(trace, runtime)` injects the live runtime into each scoring
   callback.
2. The cached scoring pass forwards it through `perf_reward` → `_prepared` →
   `CuratorScorer.compute_scoring` → `_train`.
3. `HarnessRuntimeProxyTrainer` writes `corpus.txt`, `config.json`, `train.py`,
   and the optional held-out `val.bin` through `runtime.write`.
4. It runs `python /workspace/train.py` through `runtime.run` and parses the
   final `RESULT_JSON {...}` stdout marker.
5. The rollout tears down the runtime once after scoring.

Run this path through the native taskset command,
`uv run eval pretrain-data-curator --harness.id bash
--harness.runtime.type docker ...`. The package is discovered as a flat v1
taskset module through its exported `CuratorTaskset`; legacy `--id` is not
needed. `prime eval run` currently exposes no harness-runtime flags and cannot
launch this local Docker mode.

There is no second Docker container or Modal sandbox. A non-zero exit, missing
marker, or malformed result raises `TrainerError` with the captured stderr tail.
Training is wrapped in the budget-derived deadline. On failure, timeout, or
cancellation the trainer stops the runtime immediately; the rollout's final
teardown is an idempotent backstop. Both trainers explicitly re-raise a pending
task cancellation after each `asyncio.wait_for` call, covering the pre-3.12 race
where `wait_for` can swallow cancellation as its awaitable completes.

Docker and Modal tasks declare their image, GPU/CPU/memory/disk request, work
directory, and scoring deadline. Pairing either trainer with the wrong runtime
fails before generation with an explicit harness-runtime error.
Under WSL2, host interception is advertised on the WSL interface rather than
`127.0.0.1`, because Docker Desktop runs the container in a separate VM.

Prime continues to use `SandboxProxyTrainer` and `prime_sandboxes`; it does not
use the harness runtime for training.

The Modal runtime is remote. `ModalRuntime.start()` uses the Modal SDK from the
CPU-only env-server to create the sandbox over outbound HTTPS, and Verifiers
tunnels interception traffic to the harness. No local Docker daemon or GPU is
required on the env-server, so Hosted Training compatibility is preserved.
Unlike the previous second-sandbox implementation, the Modal GPU runtime now
exists for the entire rollout, including dataset discovery, not only scoring.
Account for that longer GPU allocation when estimating cost and ensure the
env-server can reach Modal and has `MODAL_TOKEN_ID` and
`MODAL_TOKEN_SECRET`.

## Image contract

Because one Docker or Modal container now performs both phases, its image must
combine the agent and trainer dependencies. `Dockerfile.runtime` is the
reference image definition, based on
`pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime` with `hf`, `uv`, and `tiktoken`
installed. Use a `-devel` PyTorch base instead if custom CUDA compilation is
required.

The default Modal image is the bare upstream PyTorch image, and Prime defaults
may likewise omit `hf`. The shared prompt defensively checks for `hf` in its
first discovery command and quietly installs `huggingface-hub>=0.34` when
missing before running the first search in that same turn. This fallback works
without a custom registry image, but cold installation is slower, requires
outbound package-index access, and is billed as part of the full Modal GPU
rollout.

The recommended production setup is to build `Dockerfile.runtime`, push the
result to a registry Modal can pull, and configure that image explicitly.
Verifiers' `ModalConfig` uses `modal.Image.from_registry`; it cannot build the
local Dockerfile. This repository does not currently publish that image, so
registry publication remains a known infrastructure TODO.

Remote `docker_host=ssh://...` is outside this design: it would move the entire
agent harness away from the rollout worker and reintroduce networking and
credential concerns that the co-located single-pod path deliberately avoids.
