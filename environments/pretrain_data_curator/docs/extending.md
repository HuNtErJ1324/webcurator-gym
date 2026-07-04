# Extending

Common extension points, each grounded in the existing code. These are
descriptions of how the seams work; snippets are illustrative.

## Add a new filter kind

Current agent manifests are allow-listed in `taskset.py` before
`DocumentFilter` sees them. A complete extension therefore requires three
coordinated changes:

1. Add the kind to `taskset._SUPPORTED_FILTER_KINDS`.
2. Add its implementation to `DocumentFilter._apply_one_iter`
   (`pretrain_data_curator/corpus.py`).
3. Add it to `tasks.TASK_PROMPT`, `self_score.py`, and
   [Manifest and filtering](manifest.md#filters).

```python
if kind == "max_tokens":
    threshold = int(params.get("value", 10**9))
    return [d for d in docs if estimate_tokens(d) <= threshold]
```

Unknown kinds in assistant JSON are discarded by `_coerce_filters`. Unknown
kinds only reach `DocumentFilter` when a `Source` is constructed directly; its
fallback is a no-op. Keep filters deterministic and cheap. Materialization
fetches asynchronously, but the filter iterator itself runs inline.

## Extend the agent's CLI workflow

Agent behavior is defined by the single `TASK_PROMPT` in `tasks.py`; task
instances set `system_prompt=None`. Keep the self-score contract aligned with
the prompt and manifest parser. Do not reintroduce an MCP toolset unless the
intended harness compatibility changes.

If a new `hf` subcommand needs different billing, update
`hf_meter.classify_hf_argv()` and tests for both JSONL records and trace
reconstruction. The shim must continue forwarding stdout, stderr, arguments,
and exit status without changing CLI behavior. Metering failure must remain
non-fatal.

## Swap the leakage "semantic" backend

`LeakageDetector` currently uses character-trigram cosine similarity for its
semantic component. To use embeddings, preserve the contract:

- precompute held-out reference vectors;
- return the fraction of curated documents above the configured threshold;
- keep values in `[0,1]`;
- keep `overall = max(exact, fuzzy, semantic)`.

Add any dependency to `pyproject.toml`, validate required credentials in
`load_environment`, and keep heavy work off the event loop. Exact SHA-1 and
fuzzy MinHash checks can remain unchanged.

## Add a trainer backend

Any trainer implements:

```python
class ProxyStudentTrainer(Protocol):
    async def train_and_eval(
        self, corpus: CuratedCorpus, config: ProxyStudentConfig
    ) -> TrainResult: ...
```

`TrainResult` must provide loss, accuracy, FLOPs, tokens trained, and a backend
label. Lower finite loss is better; non-finite loss maps to zero Perf. FLOPs are
added to the shared cost ledger exactly once.

Which concrete trainer actually runs is decided at score time purely by the
live harness runtime's `type`, via `trainer.RuntimeSelectedTrainer` — there is
no separate backend-selector config consulted at that point (Prime GPU
sandboxes are not supported). `ProxyStudentConfig.runtime_backend`, a strict
`Literal["docker", "modal"] | None`, is a narrower, *static* hint consulted
only before any runtime exists (shaping `load_environment`'s `harness.runtime`
and `CuratorTaskset.load_tasks()`'s self-declared task image/resources). To add
another backend:

1. Extend `runtime_backend`'s literal and any backend-specific
   validators/defaults (e.g. a new timeout ceiling, mirroring
   `_check_modal_timeout_ceiling`).
2. Add an entry to the `RuntimeSelectedTrainer` dict built in
   `CuratorTaskset._build_real_trainer()`, keyed by the new runtime type
   string.
3. Build a concrete trainer that accepts a live `vf.Runtime` and validates
   `runtime.type` itself before doing any work — `HarnessRuntimeProxyTrainer`
   (`docker_backend.py`) and `ModalProxyTrainer` (`modal_backend.py`) are the
   concrete examples: each reads/writes through the runtime's `.write`/`.run`.
4. Preserve timeouts, cleanup, semaphore bounds, stderr reporting, fixed model,
   fixed training recipe, and held-out validation.
5. Wire the matching static declarations into `load_environment`
   (`pretrain_data_curator.py`) and `load_tasks()` (`taskset.py`) so the new
   `runtime_backend` value builds the right `vf.RuntimeConfig` and task
   resources ahead of any live runtime.

## Change the proxy model or training recipe

`student_model.py` and `student_train.py` are executable sources of truth, not
just host-side helpers. `trainer._nanogpt_train_script()` embeds their source
verbatim into the GPU program. Model changes must remain in `model_source()`'s
dependency order; recipe changes must remain in `training_source()`'s order.

Update CPU tests that assert byte-identical embedding, architecture contracts,
schedule behavior, batching, and averaging. Avoid editing only the template in
`trainer.py`, because that would split tested host behavior from GPU behavior.

## Inject collaborators for testing

There is no toolset to bind. Construct `CuratorTaskset` directly and inject its
lazy seams before calling scoring:

- `_client` or `_corpus_builder` for source sampling;
- `_trainer` for deterministic `TrainResult`s;
- `_leakage_detector` for controlled contamination;
- `_val_loader` for real-trainer lifecycle tests;
- `_scorer` when testing only decorated reward plumbing.

Drive finalization with a `Trace` containing assistant JSON, or construct a
fresh `CuratorState` for scorer-only tests. `DatasetSearchClient`
is a one-method Protocol: `sample_documents(dataset_id, config, split,
text_field, n)` for materialization. There is no `search_datasets` method —
discovery happens entirely through the agent's own `hf` CLI calls, outside the
environment.

Test `hf` discovery independently through `hf_meter`:

- `classify_hf_argv` for call classes;
- `ledger_from_records` / `parse_cost_log` for shim output;
- `extract_hf_commands` / `ledger_from_messages` for fallback traces;
- `meter_ledger` for runtime-log preference.

Direct native tests with injected collaborators do not need a real HF token.
`load_environment()` also constructs without one, but `taskset.setup()` requires
`hf_token_env` before the agent starts.
