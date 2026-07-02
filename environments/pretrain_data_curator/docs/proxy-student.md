# Proxy student

The proxy student makes data quality measurable while holding the rest of the
experiment fixed. All rollouts in a comparable evaluation should use the same
student configuration, tokenizer, training recipe, training budget, and
validation set.

## Heuristic versus real training

### Heuristic trainer

`HeuristicProxyTrainer` is the default. It derives deterministic synthetic
loss/accuracy from:

- estimated corpus size;
- character cleanliness;
- source diversity;
- fixed model-size and train-token FLOP estimates.

It is fast and requires no GPU, which makes it appropriate for tests and smoke
evaluation. It does not tokenize the corpus, update model weights, or score the
held-out validation shard. Results from this backend are not evidence that one
pretraining mixture is scientifically better than another.

### Real trainer

`use_real_trainer=true` trains the architecture in `student_model.py` with the
recipe in `student_train.py`. `trainer.py` extracts those exact class/function
definitions into a generated script; CPU tests assert byte-identical embedding
so the remote GPU path cannot silently drift from tested source.

The real path:

1. joins the curated documents up to `effective_max_corpus_chars`;
2. tokenizes them with GPT-2 BPE;
3. loads a fixed GPT-2-tokenized validation shard;
4. trains one or more fresh students with deterministic seeds;
5. computes mean held-out next-token cross-entropy and accuracy;
6. returns aggregate loss, accuracy, tokens, parameter count, and FLOPs.

## Model

The configurable decoder-only model uses:

- RMSNorm and QK normalization;
- half-truncated rotary embeddings;
- causal scaled-dot-product attention;
- ReLU-squared MLPs;
- U-Net-style encoder/decoder skip connections;
- sparse value-embedding tables shared by mirrored layers;
- an untied, zero-initialized language-model head;
- tanh logit soft-capping.

The cheap runtime default is 4 layers, width 256, 4 heads, block size 256, and
batch size 16. The architecture also supports a documented GPT-2-small-class
shape, but the default is intentionally smaller.

Pydantic validators reject shapes that cannot satisfy the architecture:

- layer count must be even and at least two;
- embedding width must divide evenly by head count;
- head dimension must be a multiple of four.

## Training recipe

Each run uses AdamW with:

- betas `(0.9, 0.95)`;
- epsilon `1e-8`;
- weight decay `0.1`;
- global gradient clipping at `1.0`;
- linear warmup;
- cosine decay to `lr_min_ratio` of peak learning rate;
- contiguous training windows.

`n_train_runs > 1` rebuilds the model with distinct seeds and averages validation
loss/accuracy. Tokens and FLOPs are summed, so the cost ledger charges every run.
Any nonfinite constituent run collapses the aggregate to the failure sentinel.

## Held-out validation

The default real-trainer validation data is:

```text
dataset:  kjj0/fineweb10B-gpt2
file:     fineweb_val_000000.bin
tokens:   first 10,485,760
tokenizer: GPT-2
```

The loader validates the NanoGPT shard header, extracts little-endian token IDs,
and caches the resolved data. Validation windows cover every predictable token,
including a final partial window. A shard with fewer than two tokens raises
instead of reporting an artificial zero loss.

When no held-out validation set is supplied to a low-level trainer, the generated
script can fall back to a corpus tail split. The environment's configured real
trainer normally supplies the fixed held-out shard; benchmark comparisons should
not rely on the fallback.

## Training budget derivation

With `train_token_budget` unset:

```text
effective_steps = steps
```

With it set:

```text
effective_steps =
    ceil(train_token_budget / (batch_size * block_size))
effective_train_tokens =
    effective_steps * batch_size * block_size
```

The same budget also scales:

- the maximum uploaded corpus characters;
- the derived sandbox/command timeout;
- estimated FLOPs and therefore cost.

This keeps one primary scaling knob rather than requiring users to coordinate
steps, upload size, and timeout independently.

## Backend: Prime sandbox

`trainer_backend="prime"` leaves the agent harness in its normal runtime. During
scoring, `SandboxProxyTrainer` creates a separate Prime GPU sandbox, uploads
`corpus.txt`, `config.json`, `train.py`, and optional `val.bin`, then executes the
script.

Lifecycle steps have individual timeouts. The trainer checks exit status,
requires a structured `RESULT_JSON` marker, preserves a stderr tail on failure,
and deletes the sandbox in `finally`.

This path imports `prime_sandboxes` only when used. It requires a valid Prime GPU
request: positive GPU count, VM mode enabled, and a nonempty GPU type. The
dependency is provided by the package's optional `sandbox` extra.

## Backend: Docker harness runtime

`trainer_backend="docker"` changes the v1 harness runtime itself. Discovery,
agent generation, finalization, and scoring all happen in one rollout-owned
container.

`HarnessRuntimeProxyTrainer` receives that live runtime and writes:

```text
/workspace/corpus.txt
/workspace/config.json
/workspace/train.py
/workspace/val.bin  (when available)
```

It calls `python /workspace/train.py` through `runtime.run()`. A mismatch between
the selected trainer and runtime type fails during setup before generation.

The image must contain both halves of the workload:

- bash and the Hugging Face `hf` CLI for the agent;
- Python, CUDA PyTorch, NumPy, tokenizer dependencies, and `tqdm` for training.

[`../Dockerfile.runtime`](../Dockerfile.runtime) is the reference. Docker must be
co-located with the eval worker, and the NVIDIA Container Toolkit must make
`--gpus` functional.

## Backend: Modal harness runtime

`trainer_backend="modal"` also runs the complete rollout in one GPU runtime.
`ModalProxyTrainer` uses the same file/result contract as Docker through the live
v1 runtime.

The CPU environment worker uses the Modal SDK over HTTPS and needs
`MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET`, but no local GPU or Docker daemon.
Because the Modal GPU exists during discovery as well as training, paid runtime
duration includes the agent loop.

The loader maps `modal_gpu` values:

| Input | Modal GPU |
| --- | --- |
| `H100` | `H100` |
| `H200` | `H200` |
| `A100` | `A100-80GB` |
| anything else | `L4` |

Use a registry image that already satisfies the combined image contract for
production. The bare default can bootstrap `hf` and install tokenizer
dependencies, but cold installation consumes time and requires package-index
network access.

## Concurrency, timeout, and cancellation

`max_concurrent_training` limits active training commands independently of
rollout concurrency. The semaphore is loop-local so it works correctly across
test/event-loop lifecycles.

For Docker and Modal, the framework scoring deadline is set above the derived
training deadline to leave time for corpus materialization and uploads. A timeout
or cancellation stops the live runtime immediately; the episode teardown remains
an idempotent backstop.

Trainer errors are converted by `CuratorScorer` into an infinite-loss sentinel.
Cancellation remains cancellation and is re-raised.
