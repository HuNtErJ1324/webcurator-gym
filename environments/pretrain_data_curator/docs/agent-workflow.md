# Agent workflow

This page describes the interface presented to the model being evaluated. It is
not a replacement for Prime CLI setup instructions.

## Available interface

The agent receives a normal bash shell from the bash harness. The environment
registers no MCP tools. Dataset discovery happens with Hugging Face's current
`hf` CLI:

```bash
hf datasets ls --search "wikipedia" --sort downloads --limit 5 | head -c 6000
hf datasets ls --search "scientific text" --format json \
  --expand downloads,likes,lastModified --limit 5 | head -c 6000
hf datasets info HuggingFaceFW/fineweb --expand downloads,likes,tags \
  | head -c 6000
```

Some GPU runtime images do not contain `hf`. The system prompt requires the
first shell command to install it only when absent and continue to a search in
the same turn:

```bash
if ! command -v hf >/dev/null 2>&1; then
  pip install -q 'huggingface-hub>=0.34'
fi
hf datasets ls --search "wikipedia" --sort downloads --limit 5 | head -c 6000
```

Every `hf` command should cap displayed output with `head -c 6000`. Dataset
searches should not expand `tags`: multilingual repositories can emit enough
language tags to exhaust the model context. Tags remain useful when inspecting
one shortlisted repository.

The agent should not create a virtual environment, write a Python Hub client, or
search for legacy `huggingface-cli`/`curator_*` commands. Those paths consume a
finite turn without improving the manifest.

## Recommended decision loop

1. Search two or three distinct corpus categories with small result limits.
2. Compare repository popularity, license/tags, likely text structure, and
   `lastModified`.
3. Inspect only the most plausible candidates with `hf datasets info`.
4. Choose complementary sources rather than several near-duplicates.
5. Assign weights according to expected utility and the task's emphasis.
6. Add conservative quality filters whose parameters are supported.
7. Submit a non-empty manifest before the turn cap.

The prompt prioritizes encyclopedic, scientific, instructional, and broad web
text for general pretraining. It warns against narrow task datasets and
code-only mixtures unless the task goal justifies them.

## Cutoff date

Every task states a latest allowed Hugging Face `lastModified` date. The agent is
responsible for checking and respecting it during discovery.

This is a behavioral constraint, not an environment-side allow-list. The
manifest finalizer does not make an additional metadata call to reject
post-cutoff repositories. That design avoids hidden calls and keeps discovery
cost attributable to the evaluated agent, but it also means cutoff compliance
must be assessed from the trace when auditing a result.

## What is charged

The shim classifies `hf` commands as follows:

| Command type | Ledger effect |
| --- | --- |
| `hf datasets ls ...` | one web query, one Hub call, output bytes |
| `hf datasets info ...` | one Hub call, output bytes |
| Other networked `hf` operations | one Hub call, output bytes |
| `hf version`, `env`, `auth`, `cache`, `completion` | no network-call charge |

Output bytes are converted to estimated tokens with integer division by four.
A recognized call is counted even when it exits nonzero; failed discovery still
consumes resources.

After finalization, each unique scoring fetch adds one Hub call and the estimated
tokens in the returned documents. That charge is independent of CLI discovery
output.

## Turn budget

`max_turns` is the hard harness limit. The rendered prompt also gives an
approximate discovery allowance and commit target:

```text
discovery_rounds = max(2, min(12, max_turns // 6, scan_limit // 10))
commit_by = max(1, max_turns - max(3, max_turns // 8))
```

These are instructions, not additional framework stops. `max_turns_reached`
enforces model turns, while `discovery_output_budget_reached` protects the
provider context from oversized tool results. One assistant response can contain
multiple bash calls; it still consumes one model turn, but every `hf` call is
metered separately.

| `max_turns` | `scan_limit` | Suggested discovery rounds | Suggested commit by |
| ---: | ---: | ---: | ---: |
| 12 | 50 | 2 | 9 |
| 25 | 10 | 2 | 22 |
| 64 | 200 | 10 | 56 |

Earlier assistant messages are scanned for a valid manifest if the final turn is
truncated. Submitting a manifest early and then continuing discovery is
recoverable, but ending cleanly on the final JSON is more reliable.

The taskset also stops generation when accumulated tool-result text reaches
24,000 characters. Finalization then synthesizes a fallback manifest from
observed dataset IDs. This converts runaway discovery into a scored rollout
instead of allowing the next provider request to exceed its context window.

## Common agent mistakes

- **Describing commands instead of running them.** The harness needs actual shell
  calls to produce discovery evidence.
- **Spending turns installing tools.** Only the conditional `hf` bootstrap is
  expected.
- **Using an unknown text field.** Prefer `null` and let materialization
  auto-detect unless metadata makes the field certain.
- **Copying the schema placeholder.** Values must describe a real dataset and
  deliberate mixture.
- **Using unsupported filters.** Unknown filter kinds are discarded during
  manifest coercion.
- **Waiting until the last token to submit.** No usable manifest means zero
  positive performance reward.
