# Proxy student

The real proxy student trains during `Phase.SCORING`, before the rollout runtime
is torn down. The Docker backend uses the same runtime that already hosted agent
setup, dataset discovery, and finalization:

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

There is no second Docker client, container, or lifecycle. A non-zero exit,
missing marker, or malformed result raises `TrainerError` with the captured
stderr tail. Training is wrapped in the budget-derived deadline. On failure,
timeout, or cancellation the trainer stops the runtime immediately; the
rollout's final teardown is an idempotent backstop.

Docker tasks declare their image, GPU/CPU/memory/disk request, work directory,
and scoring deadline. Pairing a Docker trainer with a subprocess runtime fails
before generation with an explicit `--harness.runtime.type docker` error.
Under WSL2, host interception is advertised on the WSL interface rather than
`127.0.0.1`, because Docker Desktop runs the container in a separate VM.

Prime continues to use `SandboxProxyTrainer` and `prime_sandboxes`. Modal
continues to use `ModalProxyTrainer` and `modal.Sandbox`. Neither backend uses
the harness runtime for training.

## Image contract

Because one Docker container now performs both phases, its image must combine
the agent and trainer dependencies. `Dockerfile.runtime` is the reference image
definition, based on
`pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime` with `hf`, `uv`, and `tiktoken`
installed. Use a `-devel` PyTorch base instead if custom CUDA compilation is
required.

Remote `docker_host=ssh://...` is outside this design: it would move the entire
agent harness away from the rollout worker and reintroduce networking and
credential concerns that the co-located single-pod path deliberately avoids.
