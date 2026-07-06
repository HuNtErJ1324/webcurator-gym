# Reward and metrics

`CuratorScorer` computes one cached scoring dictionary per rollout. Decorated v1
reward and metric methods read from that dictionary, so concurrent signal
evaluation does not repeat corpus fetches or training.

## Composite reward

```text
R(M, H) = alpha_perf * Perf(M) - lambda_leakage * Leakage(M, H)
```

Default coefficients are:

| Component | Config | Default |
| --- | --- | ---: |
| Performance | `alpha_perf` | `1.0` |
| Leakage | `lambda_leakage` | `1.0` |

Each decorated reward is registered with framework weight `1.0`; the method
itself applies its configured coefficient and sign.

## Performance

The trainer returns validation loss, next-token accuracy, trained tokens, FLOPs,
and a backend name. Lower loss is better.

With the default `baseline_relative_perf=true`:

```text
Perf = (perf_baseline_loss - loss) / (perf_baseline_loss - perf_target_loss)
```

`perf_baseline_loss` defaults to `log(50304)`, the cross-entropy of a uniform
distribution over the padded vocabulary. It is a constant; the environment does
not train a second baseline model.

`perf_target_loss` defaults to `3.28`, the nanoGPT speedrun validation-loss
target. A loss equal to `perf_baseline_loss` maps to `0.0`; a loss equal to
`perf_target_loss` maps to exactly `1.0`. The mapping is not clamped: worse than
baseline is negative, and beating the target exceeds `1.0`. Configuration
validation rejects `perf_baseline_loss <= perf_target_loss`.

With `baseline_relative_perf=false`:

```text
Perf = exp(-loss)
```

The absolute mapping is mainly useful for toy losses below one. For realistic
language-model losses near 9 nats/token, it collapses close to zero.

Nonfinite loss always maps to zero performance. This includes empty training
corpora and trainer-failure sentinels.

`perf_vs_baseline` reports the raw old diagnostic,
`(perf_baseline_loss - loss) / perf_baseline_loss`. It can be negative when the
student is worse than the no-information baseline.

## Cost

One `CostLedger` covers discovery, materialization, and training:

```text
Cost =
    web_queries * web_query_price
  + hub_calls * hub_call_price
  + code_calls * code_call_price
  + (tokens / 1000) * per_1k_tokens_price
  + (train_flops / 1e9) * per_gflop_price
```

Default unit prices:

| Unit | Price |
| --- | ---: |
| Web query | `0.0` |
| Hub call | `0.01` |
| Code call | `0.02` |
| 1,000 estimated tokens | `0.001` |
| GFLOP | `1e-6` |

The defaults mean a search adds both a zero-priced web-query count and a priced
Hub call. Each unique successful local-source pull adds one `code_call`.

Token accounting includes recognized `hf` output and documents fetched for
materialization. Parsed local-source documents are billed by the same token
estimator. Real/heuristic trainer FLOPs are added after training.

Cost total is recorded as a telemetry-only metric (``cost_total``) and no longer
enters the reward. The ``cost_total`` metric reports the priced ledger sum,
allowing runs to remain cost-observable.

## Leakage

Contamination is detected by the [allenai/decon](https://github.com/allenai/decon)
Rust n-gram detector, run at scoring time against the materialized corpus. Decon
replaces the previous exact/fuzzy/semantic (SHA-1 / MinHash / character-trigram)
detectors. It compares the corpus against two reference sets:

| Reference set | Source | Exposure |
| --- | --- | --- |
| Public benchmarks | Bundled eval sets under `decon/bundled-evals/` (e.g. MMLU, GSM8K, AGIEval) | Baked into the runtime image; offline at scoring time |
| Held-out validation set | The FineWeb-10B GPT-2 val shard, detokenised from BPE token IDs back to text via `tiktoken` (enabled by `screen_val_set`, default `true`) | Built ephemerally in a server-side temp dir and deleted after scoring; **never** written to `decon/bundled-evals/`, the workspace, or any container image |

Decon emits a JSONL report with one record per (training document x eval
instance) match, each carrying a `contamination_score` in `[0, 1]`. The scorer
reduces the report to a single token-weighted scalar via `_reduce_report`:

```text
Leakage = min(1.0, sum(score * matched_tokens) / corpus_tokens)
```

Matches are deduplicated per training document (only the highest
`score * matched_tokens` per document counts), so `Leakage` is in `[0, 1]`. Decon
runs deterministically as a subprocess off the event loop.

Because decon's IDF weighting needs many eval records, the held-out val screen is
meaningful only at production scale (~9,700 records from the 10M-token shard); a
tiny synthetic val set can score `0` even on verbatim matches.

### Fail-loud on detector failure

A decon failure (missing binary, non-zero exit, timeout, or crash) raises
`DeconError`, which the scorer catches and records as `decon_error=1.0` and
`external_failure=True`, keeping `leakage_score=0.0`. This makes a broken detector
**unambiguously distinguishable** from a genuinely clean corpus (clean =
`decon_error=0`, `external_failure=False`), so a detector failure is never a
silent free pass on the leakage penalty.

The development self-scoring script (`self_score.py`) runs decon against the
**bundled benchmarks only** — the held-out val set is never exposed inside the
agent container.

Local-source audit telemetry supplements, but does not replace, decon.
`val_set_access` flags bash commands containing the configured validation
repository ID. It is intentionally a conservative command-provenance signal; it
does not prove which bytes entered the corpus.

## Empty and unfinalized rollouts

If no non-empty manifest was read from the workspace file or recovered by a
compatibility fallback, scoring returns:

- zero performance;
- zero corpus/training/leakage diagnostics;
- the discovery cost already accumulated;
- `finalized=0`.

No dataset materialization or trainer call occurs. This makes failure to write a
manifest strictly worse than writing a minimally valid one after the same
discovery activity.

## External failure

Hub or trainer failures do not automatically invalidate the trace:

- a failed source fetch becomes an empty document slice;
- a trainer exception becomes `loss=inf`, zero accuracy/FLOPs/tokens, and
  backend `"error"`;
- performance becomes zero;
- classified error counts and `external_failure` are set;
- trainer detail is retained in state and summarized by `trainer_error_msg`.

This separation matters when analyzing low rewards: an agent can choose poor
data, or the infrastructure can fail before that data is evaluated.

## Emitted signals

### Reward components

| Name | Value |
| --- | --- |
| `perf_reward` | `alpha_perf * Perf` |
| `leakage_penalty` | `-lambda_leakage * Leakage` |

### Metrics

| Name | Meaning |
| --- | --- |
| `perf_loss` | Raw trainer loss; `0.0` when nonfinite |
| `perf_accuracy` | Raw next-token accuracy |
| `perf_vs_baseline` | Raw relative loss improvement vs `perf_baseline_loss` |
| `train_flops` | Estimated/measured training FLOPs |
| `corpus_tokens` | Estimated materialized corpus tokens |
| `budget_fill_ratio` | `corpus_tokens / manifest.token_budget`; values below `1` indicate the selected/fetched data did not fill the allocation |
| `num_sources` | Sources with at least one retained document |
| `local_source_count` | Unique successful local pulls |
| `local_source_bytes` | Bytes transferred by capped local pulls |
| `local_source_truncated` | `1.0` when any local source exceeded its cap |
| `val_set_access` | `1.0` when a bash command named the validation repository |
| `leakage_score` | Token-weighted decon contamination scalar in `[0, 1]` |
| `num_contaminated_matches` | Count of deduplicated contaminated training documents |
| `decon_error` | `1.0` when the decon detector failed (paired with `external_failure`); `leakage_score` is then `0.0` but not trustworthy |
| `cost_total` | Priced ledger total (telemetry-only; zero weight on reward) |
| `finalized` | `1.0` when a usable manifest was recovered |
| `tool_error_count` | Total classified Hub/trainer errors |
| `external_failure` | `1.0` when external infrastructure failed |
| `trainer_error_msg` | `1.0` when trainer detail was recorded |

The trainer error string itself is logged in truncated form rather than emitted
as a numeric metric. Always read `decon_error` alongside `leakage_score`: a
`leakage_score` of `0.0` means "clean corpus" only when `decon_error=0`.

## Reading a result

A practical order is:

1. Check `finalized`.
2. Check `external_failure`, `tool_error_count`, and `trainer_error_msg`.
3. Check `num_sources`, `corpus_tokens`, and local-source provenance metrics.
4. Compare `perf_loss` and `perf_vs_baseline`.
5. Inspect `leakage_score` and `num_contaminated_matches`, and confirm
   `decon_error=0` before trusting a low leakage score.
6. Decompose the final reward into the two named reward components (perf_reward and leakage_penalty); cost_total is telemetry-only.

This avoids attributing a zero performance score to data quality when the
manifest never materialized or the trainer failed.
