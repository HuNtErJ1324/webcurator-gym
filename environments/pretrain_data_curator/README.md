# pretrain-data-curator

An environment where an agent curates an **LLM pretraining dataset** from the
pre-cutoff Hugging Face universe. The curated mixture is scored by training a
**fixed GPT-2-scale proxy-student where everything but the data is held
constant**, combined with deterministic quality, diversity, cost, and leakage
terms.

## Overview

- **Environment ID**: `pretrain-data-curator`
- **Type**: multi-turn `StatefulToolEnv`
- **External service**: Hugging Face Hub API (search + streaming sampling)
- **Required secret**: `HF_TOKEN` by default
- **Proxy-student training**: optional Prime GPU sandbox (off by default)

The agent searches Hugging Face for datasets at or before a cutoff date,
inspects them, assembles a weighted and filtered **manifest** `M`, previews
statistics, and finalizes. The finalized manifest is materialized into a corpus
and scored.

## Reward

```
R(M, H) = a1*Perf(M) + a2*Quality(M, H) + a3*Diversity(M) - l1*Cost(M) - l2*Leakage(M)
```

| Term | Meaning | Default weight |
| --- | --- | --- |
| `Perf` | Proxy-student val loss/accuracy after training on the curated corpus | `alpha_perf=1.0` |
| `Quality` | Document-level cleanliness + provenance coverage | `alpha_quality=0.3` |
| `Diversity` | Source count, weight entropy, tag coverage | `alpha_diversity=0.2` |
| `Cost` | Web queries, hub calls, code calls, tokens, training FLOPs (one ledger) | `lambda_cost=0.1` |
| `Leakage` | Exact + fuzzy (MinHash) + semantic overlap vs the held-out eval set | `lambda_leakage=1.0` |

Each term is also emitted as a metric (`perf_loss`, `perf_accuracy`,
`train_flops`, `corpus_tokens`, `num_sources`, `leakage_*`, `cost_total`,
`finalized`, `viable`). Two further zero-weight **diagnostics**,
`tool_error_count` and `external_failure`, separate bad curation from
external/infrastructure failure (a flaky Hub or sandbox).

The quality and diversity **bonuses are gated on a minimum-viability predicate**
(manifest finalized, non-empty corpus, no severe leakage, training succeeded);
the `Perf`, `Cost`, and `Leakage` terms are always applied. See
[`docs/reward.md`](docs/reward.md).

### Proxy-student backends

`Perf(M)` is computed by a `ProxyStudentTrainer`:

- **`HeuristicProxyTrainer`** (default): a deterministic CPU surrogate that
  predicts loss/accuracy from corpus scale, cleanliness, and diversity. It is a
  **dev/smoke surrogate, not a real training signal** — it lets the environment
  run and be tested without a GPU, but its `Perf` is a proxy, not a measurement.
  It trains no model, so it does **not** compute per-token CE on the held-out
  validation set and is unaffected by `validation_set`.
- **`SandboxProxyTrainer`** (`use_real_trainer=true`): the path that yields a
  **meaningful** `Perf`. It actually trains a fixed small GPT (config in
  `ProxyStudentConfig`), GPT-2-BPE-tokenized, in a Prime GPU sandbox on the
  materialized corpus and scores it against a **fixed held-out validation token
  stream** — by default the **NanoGPT speedrun** set (FineWeb `sample-10BT` GPT-2
  val tokens, `kjj0/fineweb10B-gpt2`, the first `10_485_760` tokens), reporting
  cross-entropy in nats/token, next-token accuracy, and FLOPs (see
  `ValidationSetConfig` / `validation_set`). Only the training corpus varies
  between rollouts. Its lifecycle is hardened (per-step timeouts, exit-code
  checks, stderr-tail preservation) and `prime_sandboxes` is an optional extra
  (`sandbox`) imported only on this path.

Measured/estimated training FLOPs are charged back onto the cost ledger.

## Tools

| Tool | Purpose |
| --- | --- |
| `search_datasets(query, limit)` | Search the pre-cutoff HF universe; caches cutoff-valid candidates. |
| `inspect_dataset(dataset_id)` | Sample documents and report quick statistics. |
| `set_source(dataset_id, weight, text_field, split, config, filters, max_docs, max_tokens)` | Add/update a weighted, filtered source. `filters` is a JSON array string. |
| `remove_source(dataset_id, config)` | Remove a source. |
| `run_code(code)` | Stub; only registered/advertised when `enable_run_code=true` (off by default). Use structured `filters` instead. |
| `compute_manifest_stats()` | Preview quality/diversity/leakage/cost without training. |
| `finalize_manifest()` | Lock the manifest for end-of-rollout scoring. |

Supported `filters` kinds: `min_chars`, `max_chars`, `min_tokens`,
`max_symbol_ratio`, `min_alpha_ratio`, `drop_regex`, `keep_regex`, `dedup_exact`.

Example filter argument:

```json
[{"kind": "min_chars", "params": {"value": 200}}, {"kind": "dedup_exact"}]
```

## Install And Eval

```bash
prime env install pretrain-data-curator -p ./environments
prime eval run pretrain-data-curator -m openai/gpt-4.1-mini -n 4 -r 1
```

Enable real GPU proxy-student training (requires Prime Sandboxes access):

```bash
prime eval run pretrain-data-curator -m openai/gpt-4.1-mini -n 4 -r 1 \
  -a '{"use_real_trainer": true, "proxy_student": {"steps": 300, "gpu_count": 1}}'
```

## Required Environment Variables

`load_environment()` validates Hugging Face credentials with `vf.ensure_keys(...)`.

| Variable | Default | Description |
| --- | --- | --- |
| `HF_TOKEN` | yes | Token passed to `huggingface_hub.HfApi` and streaming loads. |

## Environment Args

| Arg | Type | Default | Description |
| --- | --- | --- | --- |
| `cutoff_date` | str | `"2024-12-31"` | Latest allowed Hugging Face `lastModified` date. |
| `token_budget` | int | `1000000` | Target token budget for the mixture. |
| `hf_token_env` | str | `"HF_TOKEN"` | Env var validated for the HF token. |
| `candidate_limit` | int | `8` | Max candidates returned per search. |
| `scan_limit` | int | `50` | Max raw HF results scanned before filtering. |
| `sample_docs_per_source` | int | `64` | Docs sampled per source for inspection/scoring. |
| `max_turns` | int | `12` | Max agent turns. |
| `alpha_perf` / `alpha_quality` / `alpha_diversity` | float | `1.0` / `0.3` / `0.2` | Positive reward weights. |
| `lambda_cost` / `lambda_leakage` | float | `0.1` / `1.0` | Penalty weights. |
| `leakage_severe_threshold` | float | `0.5` | Leakage at/above this suppresses the quality/diversity bonuses (viability gate). |
| `max_concurrent_fetches` | int | `8` | Bound on concurrent HF fetches (also the corpus-builder fetch limit). |
| `max_concurrent_training` | int | `1` | Bound on concurrent sandbox-training jobs (real trainer). |
| `fetch_timeout_seconds` | float | `30.0` | Per-attempt timeout for external HF calls. |
| `fetch_max_attempts` | int | `3` | Max attempts (retry/backoff) for transient HF failures. |
| `enable_run_code` | bool | `false` | Register/advertise the `run_code` stub tool. |
| `use_real_trainer` | bool | `false` | Use the GPU sandbox proxy-student instead of the heuristic. |
| `proxy_student` | dict | `{}` | Overrides for `ProxyStudentConfig` (arch, steps, GPU, etc.). |
| `validation_set` | dict | NanoGPT speedrun set | Overrides for `ValidationSetConfig` (held-out downstream-CE val set: FineWeb GPT-2 val tokens, first `10_485_760`). Real-trainer only. |
| `eval_corpus` | list[str] | built-in | Held-out reference corpus for the leakage term. |

## Module Layout

- `models.py` — Pydantic contracts (`Manifest`, `Source`, `FilterSpec`, `CostLedger`, `CuratorConfig`, ...).
- `hf_access.py` — `DatasetSearchClient` Protocol, live HF client, cutoff/query helpers.
- `corpus.py` — `CorpusBuilder` + `DocumentFilter` (materialize manifest into documents).
- `leakage.py` — `LeakageDetector` (exact/fuzzy/semantic contamination).
- `val_set.py` — held-out validation token stream (`ValidationSetConfig`, `ValTokenLoader`, `.bin` parser); NanoGPT-speedrun set by default.
- `trainer.py` — `ProxyStudentTrainer` interface, heuristic + GPU-sandbox backends.
- `rewards.py` — `CuratorRubric` composing the full reward.
- `environment.py` — `PretrainDataCuratorEnv` (`StatefulToolEnv`) and its tools.
- `tasks.py` — curation task dataset.
- `pretrain_data_curator.py` — `load_environment` entrypoint.

## Notes And Limitations

- Leakage detection is fully deterministic and reproducible across processes:
  the semantic signal is a lightweight character-trigram cosine (no neural deps),
  and the fuzzy MinHash hashes shingles with a seeded `blake2b` (`_stable_hash32`)
  rather than Python's per-process-salted `hash()`. Swap in a real embedder if
  needed.
- Token/corpus cost is billed **once per real fetch**: documents are fetched
  through a per-rollout cache keyed by `(dataset_id, config, split, text_field, n)`
  with per-key single-flight, so previews and final scoring observe identical docs
  and there is no double/triple billing.
- External (HF/sandbox) calls are wrapped with timeout + retry and a typed
  `DatasetAccessError`; on failure tools/scoring **degrade** to a defined sentinel
  (empty slice / infinite-loss `TrainResult`) and record diagnostics rather than
  crashing the rollout.
- Token counts are estimated (~4 chars/token) on the env side; the real trainer
  tokenizes inside the sandbox.
- `run_code` is intentionally a stub and is opt-in (`enable_run_code`); filtering
  is expressed via the structured `filters` argument.
