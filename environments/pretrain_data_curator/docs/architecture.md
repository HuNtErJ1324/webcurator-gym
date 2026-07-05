# Architecture

## Contract

The package exposes:

```python
load_environment(...) -> vf.Environment
```

Internally, `CuratorTaskset` is a native verifiers v1 `vf.Taskset`.
`load_environment` wraps it in `hosted_compat.Environment`, which is both a v1
environment and a compatibility bridge for the legacy state shape still consumed
by the installed Prime eval and Hosted Training stack.

The taskset is deliberately **toolless** in the verifiers sense: it does not
register an MCP server. The bash harness supplies the shell, and the evaluated
agent invokes Hugging Face's `hf` executable there.

## Rollout lifecycle

```text
load_environment
  -> build CuratorTasksetConfig and harness/runtime config
  -> taskset.load_tasks() selects one curation goal
  -> taskset.setup() validates runtime/backend compatibility and writes self_score.py
  -> bash harness runs the agent and hf commands
  -> normal harness completion (with max_turns as a safety backstop)
  -> taskset.finalize() recovers a manifest and meters hf activity
  -> taskset.score() computes all metrics and rewards
  -> hosted_compat translates the v1 Trace to legacy State when needed
  -> runtime teardown
```

Finalization happens while the runtime is still alive. This is essential for two
reasons:

1. Discovery metering first tries to read `.vf_hf_cost.jsonl` from that runtime.
2. Docker and Modal proxy training later receives the same live runtime during
   scoring.

## Tasks and prompts

`tasks.py` defines one typed `CuratorTask` value with a single curation goal
and the configured token budget and cutoff date. Task data stays typed
rather than being carried in unstructured dataset metadata.

`tasks.py` renders one compact initial user prompt; `system_prompt` is `None`.
Every harness therefore receives the same framing, objective, method-open
autonomy, manifest contract, setup facts, and numbered failure-mode rules. The
prompt contains no dataset priors, first-command requirement, command recipe
wall, discovery allowance, or turn allocation.

The only agent optimization budget is `token_budget`. `max_turns` remains a
generous harness safety cap and is absent from the prompt, reward, and metrics.

During setup, `taskset.py` writes a standalone `self_score.py` into the runtime
workspace. It samples candidate-source development rows and applies the same
scale/cleanliness/diversity shape as the heuristic trainer. It never loads final
validation tokens or decoded leakage references, and rejects the configured
validation repository using a SHA-256 digest rather than exposing its identity.

## Manifest recovery

The primary output is the configured manifest file
(`/workspace/manifest.json` by default). `finalize()` reads it through the live
runtime and briefly polls for it to handle ordering between the shell write and
finalization. File existence is the only completion signal; there is no sentinel
token.

For backward compatibility, if the file is absent or unusable:

1. Finalization scans assistant messages newest first.
2. The compatibility parser scans fenced blocks and then the whole message.
3. A string-aware balanced-brace scanner finds JSON objects.
4. The last parseable object with a non-empty `sources` list wins.
5. A last-resort trace recovery synthesizes sources
   from explicitly inspected `hf datasets info` IDs. Raw search-output IDs are
   used only when the agent inspected nothing.

Trace recovery is capped by `candidate_limit`. It exists to salvage a rollout
truncated near its turn limit; it is not equivalent to a deliberate weighted,
filtered manifest.

If recovery still finds nothing, `manifest_finalized` remains false. Scoring
does not fetch a corpus or run a trainer in that case.

## Per-rollout state

`CuratorState` is the single v1 state object shared through the trace. It stores:

| Field | Purpose |
| --- | --- |
| `cutoff_date` | Task provenance used during the rollout |
| `manifest` | JSON-serializable Pydantic manifest dump |
| `cost_ledger` | Discovery, materialization, token, and FLOP counters |
| `doc_cache` | `FetchKey` to scratch-file name map; raw document text is not stored in state |
| `scratch_dir` | Lazily created rollout directory for cached and materialized JSONL |
| `tool_errors` | Counts by classified external-error kind |
| `external_failure` | Whether Hub or trainer infrastructure failed |
| `manifest_finalized` | Whether finalization recovered a usable manifest |
| `trainer_error` | Truncated trainer failure detail for diagnostics |

`RolloutStore` owns conversions between state dictionaries and the strict
Pydantic models. Locks and semaphores are not stored in state because rollout
state must remain serializable.

## Discovery metering

At worker initialization, `hf_meter.install_shim()` places a wrapper named `hf`
earlier on `PATH` when a real CLI exists. The wrapper:

- executes the real CLI unchanged;
- classifies the command;
- records stdout/stderr byte counts and exit status;
- appends one JSON record to `.vf_hf_cost.jsonl`.

During finalization, `meter_ledger()` reads that file through the live runtime.
If it cannot, it reconstructs recognizable `hf` commands and output sizes from
the trace. The runtime log is more complete; trace reconstruction is a fallback.

Discovery metering and corpus materialization are separate. Search output tokens
are charged during finalization. Documents fetched for scoring are charged when
`CorpusBuilder` fetches them.

## Materialization

For each source, `CorpusBuilder.materialize()`:

1. computes the source's token target from normalized positive weights;
2. estimates a fetch count at 250 tokens per document, bounded by
   `sample_docs_per_source`;
3. constructs a `FetchKey(dataset_id, config, split, text_field, n)`;
4. checks the per-rollout cache;
5. streams documents through `datasets.load_dataset(..., streaming=True)`;
6. applies filters in manifest order;
7. enforces the tightest of the weight target and explicit `max_tokens`, plus
   `max_docs`;
8. writes retained documents to a per-source JSONL file and returns a
   file-backed `SourceCorpus`.

Blocking Hugging Face work runs in a worker thread. A loop-local semaphore bounds
concurrent fetches. A per-rollout, per-key lock coalesces concurrent identical
fetches, so the Hub call and token charge occur once. Cached raw rows and
materialized rows live under the rollout scratch directory; this prevents
corpus memory from growing with the total number of selected sources. Scoring
removes the scratch directory deterministically, with a weak-reference finalizer
as a test/direct-call backstop.

If a dataset has no default configuration and the manifest omitted `config`, the
client tries `default`, `en`, `english`, `plain_text`, an English-suffixed
configuration, then the first advertised configuration.

## Scoring once

All decorated metrics and rewards may be invoked concurrently by verifiers.
`CuratorTaskset._prepared()` therefore guards the heavy scoring pass with a
per-trace lock and caches its result in `self._scoring_cache`. Corpus materialization,
proxy training, leakage detection, and final cost calculation execute once per
rollout.

The scorer:

1. short-circuits an unfinalized/empty manifest;
2. materializes the corpus;
3. invokes the configured trainer;
4. adds training FLOPs to the same cost ledger;
5. checks every materialized document for leakage;
6. derives performance, cost, leakage, and diagnostic values.

## Failure semantics

Hub access uses per-attempt timeout, bounded exponential backoff, and classified
`DatasetAccessError` values. Permanent errors such as authentication, missing
datasets, script-backed datasets that cannot be executed, invalid configs/splits,
and bad fields are not retried.

A source fetch failure becomes an empty source and records error telemetry. A
trainer exception becomes an infinite-loss `TrainResult` with backend `error`.
Both lead to zero positive performance while preserving the rollout, its costs,
and diagnostic metrics. Cancellation is not converted into an ordinary failure;
runtime trainers re-raise pending cancellation and stop the runtime.
