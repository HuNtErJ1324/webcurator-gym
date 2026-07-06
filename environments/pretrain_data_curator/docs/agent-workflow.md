# Agent workflow

This page describes the interface presented to the evaluated model.

## One initial prompt

The task uses a single initial user prompt and no separate system prompt.
`bash`, `codex`, and `mini_swe_agent` therefore receive the same contract:

- frame the fixed-student data-curation problem;
- require a final manifest JSON file at `/workspace/manifest.json`;
- grant broad autonomy over source choice, weights, filtering, shell, internet,
  Hugging Face `hf`, and local processing;
- state the cutoff, token allocation, reward coefficients, and self-score command;
- explain each rule through its failure mode;
- require autonomous operation without user questions.

There are no starter-dataset priors, mandatory first command, command recipes,
discovery rounds, or commit-by turn. The agent chooses its own method.

## Available interface

The taskset registers no MCP tools. The selected harness supplies a shell, and
the runtime image supplies the Hugging Face `hf` CLI. Agents may use `hf`, shell
programs, downloads, and workspace files freely within the harness.

For a script-backed or otherwise non-streamable source, genuine downloaded data
can be transformed into text or JSONL in the workspace and referenced with
`kind: "local"`. Paths must be relative; absolute paths, `..`, and reserved
trainer files are rejected. Local files are capped during runtime transfer,
billed like fetched documents, and included in provenance metrics.

## Sole budget

`token_budget` is the only budget exposed to or optimized by the agent. It is the
manifest allocation used for source-weight targets and corpus truncation.

`max_turns` is a generous harness safety limit (default `64`). It is not shown in
the prompt and does not enter reward or metrics. There is no discovery-call or
discovery-output budget. Harness/provider context limits still apply normally,
so agents should keep command output practical, but the environment does not
translate those limits into a curation budget.

## Leakage-safe self-scoring

Before the harness starts, setup writes `self_score.py` to the rollout workspace:

```bash
python self_score.py draft.json --limit 8
```

The script:

1. reads a draft manifest and requires the task's exact `token_budget`;
2. samples at most `--limit` rows from each named candidate source through the
   public datasets-server API, or from a bounded local file;
3. applies supported filters to those development samples;
4. estimates materialized tokens, fill ratio, cleanliness, source diversity,
   heuristic proxy CE, performance, scoring cost, and reward excluding leakage
   and discovery already incurred;
5. reports per-source sampling errors so the agent can revise the draft.

This is a development proxy, not final scoring. It uses no validation shard,
validation tokens, decoded leakage reference, or final trainer. The configured
validation repository is represented only by a SHA-256 digest and is rejected
before any network request, so the script neither reveals nor consumes the
held-out source. `leakage_estimate` reports a development-only contamination
estimate from the vendored decon binary against the **bundled benchmarks only**
(never the held-out val set); it is `null` only when decon or its evals are
unavailable.

## Cutoff and contamination

Every task states the latest allowed Hugging Face `lastModified` date. The agent
must verify it; finalization does not make a hidden metadata request.

The held-out validation/evaluation data is reserved for final scoring. Attempts
to access its configured repository are recorded by `val_set_access`, and
materialized overlap with public benchmarks and the held-out val set is penalized
by the decon n-gram contamination detector (token-weighted `leakage_score`).

## Cost metering

The PATH shim classifies `hf` activity:

| Command type | Ledger effect |
| --- | --- |
| `hf datasets ls ...` | one web query, one Hub call, output tokens |
| `hf datasets info ...` | one Hub call, output tokens |
| Other networked `hf` operations | one Hub call, output tokens |
| `hf version`, `env`, `auth`, `cache`, `completion` | no network-call charge |

Output tokens use `max(word_count, character_count // 4)`. A recognized call is
counted even if it exits nonzero. Materialization adds one Hub call per unique
fetch plus fetched document tokens. A successful local pull adds one code call
plus parsed document tokens. Training FLOPs are added after proxy training.

These are priced resources tracked in the cost_total telemetry metric (no longer a reward penalty), not separate budgets.

## Final manifest and recovery

The intended completion is a single JSON object with a non-empty `sources` list
written to `/workspace/manifest.json` (or the configured filename). File
existence is the completion signal, with no sentinel token or required
final-message content. See [Manifest and filtering](manifest.md) for the full
contract. Without a valid manifest file, positive performance is zero.

Finalization polls the runtime file briefly to handle a shell-write race. For
backward compatibility, it then scans assistant messages newest-first. If no
manifest exists, it can synthesize an equal-weight fallback from dataset IDs
explicitly inspected in the trace, capped by `candidate_limit`; raw search IDs
are used only when no inspection occurred. Recovery preserves a scoreable rollout
but loses deliberate weights, filters, configs, and fetch-cap choices.

## Common failure modes

- An invented ID/config materializes no tokens.
- A wrong text field or unsupported source format yields an empty slice.
- Too-small fetch caps produce a low `budget_fill_ratio`.
- Excess calls, fetched tokens, or training work increase cost.
- Held-out overlap increases leakage and can contaminate the evaluation.
- Missing or malformed manifest file yields no positive performance score unless
  a compatibility fallback recovers a usable manifest.
