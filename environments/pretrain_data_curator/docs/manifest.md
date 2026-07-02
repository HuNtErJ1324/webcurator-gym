# Manifest and filtering

The manifest is the agent's only curation deliverable. It describes what to
stream from Hugging Face and how to turn those rows into a bounded training
corpus.

## Canonical shape

```json
{
  "token_budget": 1000000,
  "sources": [
    {
      "id": "wikimedia/wikipedia",
      "config": "20231101.en",
      "split": "train",
      "text_field": "text",
      "weight": 2.0,
      "filters": [
        {"kind": "min_chars", "params": {"value": 200}},
        {"kind": "max_symbol_ratio", "params": {"value": 0.2}},
        {"kind": "dedup_exact"}
      ],
      "max_docs": 10000,
      "max_tokens": 500000
    },
    {
      "id": "HuggingFaceFW/fineweb",
      "config": "sample-10BT",
      "weight": 1.0,
      "text_field": null
    }
  ]
}
```

At least one source must survive parsing and Pydantic validation.

## Top-level fields

| Field | Default | Meaning |
| --- | --- | --- |
| `token_budget` | task token budget | Target used to divide tokens across source weights |
| `sources` | required, non-empty | Dataset slices included in the mixture |

The token budget is an allocation target, not a guarantee that the corpus will
contain that many tokens. Fetch limits, source length, filters, and explicit caps
can all reduce the result.

## Source fields

| Field | Default | Behavior |
| --- | --- | --- |
| `id` | required | Hugging Face dataset repository ID |
| `weight` | `1.0` | Nonnegative relative allocation weight |
| `config` | `null` | Dataset configuration/name |
| `split` | `"train"` | Dataset split |
| `text_field` | `null` | Row field to read; `null` enables auto-detection |
| `filters` | `[]` | Ordered document filters |
| `max_docs` | `null` | Maximum retained documents for this source |
| `max_tokens` | `null` | Maximum estimated retained tokens for this source |

For tolerant model-output parsing, `dataset_id`, `dataset`, `repo_id`, and
`name` are accepted aliases for `id`; a source may also be a bare ID string.
`sampling.max_docs` and `sampling.max_tokens` are accepted aliases for their
top-level forms.

Invalid weights fall back to `1.0`; negative weights clamp to zero. Invalid or
nonpositive caps become `null`. Unsupported filter entries are dropped.

## Weight allocation

Let `B` be `token_budget`, `w_i` a source weight, and `W` the sum of all weights.
For `W > 0` and `w_i > 0`:

```text
source_target_i = floor(B * w_i / W)
```

The materializer estimates the number of rows to request as:

```text
fetch_docs_i = clamp(floor(source_target_i / 250), 1, sample_docs_per_source)
```

After filtering, it enforces the tighter of `source_target_i` and
source-level `max_tokens`. `max_docs` is also applied.

If all weights are zero, no weight-derived token cap is applied; each source can
use the configured fetch limit subject to explicit caps. A zero weight therefore
does not currently mean "exclude this source." Agents should omit unwanted
sources rather than relying on zero weight.

Token caps consume documents in stream order. If the next document would exceed
the remaining cap, materialization stops instead of skipping forward.

## Text-field auto-detection

When `text_field` is absent, `null`, missing in a row, or not a nonempty string,
the client tries common fields including:

```text
text, content, passage, document, abstract, body, article, sentence,
query, answer, response, output, instruction, input, context
```

For query/response rows, it also tries a concatenated `"query response"` string.
Rows with no usable string are skipped.

Auto-detection is a robustness feature, not schema inference. Explicitly setting
a verified field is more predictable for datasets with several textual columns.

## Filters

Filters run in listed order.

| Kind | Parameters | Keeps |
| --- | --- | --- |
| `min_chars` | `{"value": int}` | documents at least `value` characters |
| `max_chars` | `{"value": int}` | documents at most `value` characters |
| `min_tokens` | `{"value": int}` | documents with at least the estimated token count |
| `max_symbol_ratio` | `{"value": float}` | documents whose non-alphanumeric, non-space fraction is at most `value` |
| `min_alpha_ratio` | `{"value": float}` | documents whose alphabetic-character fraction is at least `value` |
| `drop_regex` | `{"pattern": str}` | documents that do not match the regex |
| `keep_regex` | `{"pattern": str}` | documents that match the regex |
| `dedup_exact` | `{}` | first instance of each stripped document |

The token estimate used by `min_tokens`, caps, and billing is:

```text
max(number of whitespace-separated words, number of characters // 4)
```

`dedup_exact` is local to one source after preceding filters. It compares
stripped text but retains the original first document.

Regex patterns are passed to Python's `re` engine during scoring. An invalid
pattern can fail materialization, so agents should keep expressions simple and
valid.

## Parsing and recovery

The parser accepts prose around JSON and scans multiple fenced blocks. The last
parseable object containing a non-empty `sources` list wins. Finalization then
searches older assistant messages if needed.

Duplicates are not merged. Repeating the same dataset/config produces multiple
source entries and can cause redundant logical weighting, although identical
fetch keys can share the per-rollout document cache. Prefer one explicit entry
per intended dataset slice.

## Worked allocation example

For a 3,000-token manifest with weights `2.0` and `1.0`:

```text
source A target = 2,000
source B target = 1,000
```

With the 250-token/document estimate, the materializer requests up to 8 rows
from A and 4 from B, subject to `sample_docs_per_source`. If A also sets
`max_tokens=500`, only 500 estimated tokens can survive even though its weight
target is 2,000.
