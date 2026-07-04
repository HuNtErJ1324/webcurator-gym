# Reward and metrics

`CuratorScorer` computes one cached scoring dictionary per rollout. Decorated v1
reward and metric methods read from that dictionary, so concurrent signal
evaluation does not repeat corpus fetches or training.

## Composite reward

```text
R(M, H) =
    alpha_perf * Perf(M)
  - lambda_cost * Cost(M)
  - lambda_leakage * Leakage(M, H)
```

Default coefficients are:

| Component | Config | Default |
| --- | --- | ---: |
| Performance | `alpha_perf` | `1.0` |
| Cost | `lambda_cost` | `0.1` |
| Leakage | `lambda_leakage` | `1.0` |

Each decorated reward is registered with framework weight `1.0`; the method
itself applies its configured coefficient and sign.

## Performance

The trainer returns validation loss, next-token accuracy, trained tokens, FLOPs,
and a backend name. Lower loss is better.

With the default `baseline_relative_perf=true`:

```text
raw_relative = (perf_baseline_loss - loss) / perf_baseline_loss
Perf = clamp(raw_relative, 0, 1)
```

`perf_baseline_loss` defaults to `log(50304)`, the cross-entropy of a uniform
distribution over the padded vocabulary. It is a constant; the environment does
not train a second baseline model.

With `baseline_relative_perf=false`:

```text
Perf = exp(-loss)
```

The absolute mapping is mainly useful for toy losses below one. For realistic
language-model losses near 9 nats/token, it collapses close to zero.

Nonfinite loss always maps to zero performance. This includes empty training
corpora and trainer-failure sentinels.

`perf_vs_baseline` reports `raw_relative` without clipping. It can be negative
when the student is worse than the no-information baseline.

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

Cost is not normalized. The configured `lambda_cost` directly scales the priced
sum.

## Leakage

Every materialized document is compared with a bounded, decoded sample of the
same held-out validation token stream used for proxy-student scoring, using three
deterministic detectors:

| Metric | Method | Default match condition |
| --- | --- | --- |
| Exact | SHA-1 of normalized lowercase whitespace | identical normalized text |
| Fuzzy | seeded MinHash of word shingles | estimated Jaccard at least `0.5` |
| Semantic | cosine over normalized character-trigram counts | cosine at least `0.8` |

Each detector reports the fraction of curated documents that match at least one
held-out document. The penalty is:

```text
Leakage = max(exact_fraction, fuzzy_fraction, semantic_fraction)
```

All three fractions and the maximum are in `[0, 1]`. Fuzzy hashing uses seeded
`blake2b`, not Python's process-randomized `hash`, so results are stable across
processes.

The "semantic" detector is lexical character-trigram similarity, not a neural
embedding model. It is fast and reproducible but should not be interpreted as a
general semantic-contamination detector.

<<<<<<< HEAD
The reference samples 64 deterministic strata with windows of at most 1,024
GPT-2 tokens, caps each decoded window at 8,192 characters and the semantic
vocabulary at 32,768 trigrams, and is cached after construction. If the
validation shard cannot be loaded or decoded, scoring logs
`leakage_reference=stub` and exposes `stub` on
`CuratorState.leakage_reference` while using the built-in offline fallback.
=======
Local-source audit telemetry supplements, but does not replace, these leakage
detectors. `val_set_access` flags bash commands containing the configured
validation repository ID. It is intentionally a conservative command-provenance
signal; it does not prove which bytes entered the corpus.
>>>>>>> feat/agent-bash-datasets

## Empty and unfinalized rollouts

If no non-empty manifest was finalized, scoring returns:

- zero performance;
- zero corpus/training/leakage diagnostics;
- the discovery cost already accumulated;
- `finalized=0`.

No dataset materialization or trainer call occurs. This makes failure to commit
a manifest strictly worse than submitting a minimally valid one after the same
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
| `cost_penalty` | `-lambda_cost * Cost` |
| `leakage_penalty` | `-lambda_leakage * Leakage` |

### Metrics

| Name | Meaning |
| --- | --- |
| `perf_loss` | Raw trainer loss; `0.0` when nonfinite |
| `perf_accuracy` | Raw next-token accuracy |
| `perf_vs_baseline` | Unclipped relative loss improvement |
| `train_flops` | Estimated/measured training FLOPs |
| `corpus_tokens` | Estimated materialized corpus tokens |
| `num_sources` | Sources with at least one retained document |
| `local_source_count` | Unique successful local pulls |
| `local_source_bytes` | Bytes transferred by capped local pulls |
| `local_source_truncated` | `1.0` when any local source exceeded its cap |
| `val_set_access` | `1.0` when a bash command named the validation repository |
| `leakage_exact` | Exact-match document fraction |
| `leakage_fuzzy` | MinHash-match document fraction |
| `leakage_semantic` | Character-trigram-match document fraction |
| `cost_total` | Priced ledger total before `lambda_cost` |
| `finalized` | `1.0` when a usable manifest was recovered |
| `tool_error_count` | Total classified Hub/trainer errors |
| `external_failure` | `1.0` when external infrastructure failed |
| `trainer_error_msg` | `1.0` when trainer detail was recorded |

The trainer error string itself is logged in truncated form rather than emitted
as a numeric metric.

## Reading a result

A practical order is:

1. Check `finalized`.
2. Check `external_failure`, `tool_error_count`, and `trainer_error_msg`.
3. Check `num_sources`, `corpus_tokens`, and local-source provenance metrics.
4. Compare `perf_loss` and `perf_vs_baseline`.
5. Inspect all three leakage metrics.
6. Decompose the final reward into the three named reward components.

This avoids attributing a zero performance score to data quality when the
manifest never materialized or the trainer failed.
