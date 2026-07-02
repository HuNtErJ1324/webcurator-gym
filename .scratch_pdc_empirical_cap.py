"""Throwaway empirical repro for the pretrain_data_curator corpus-size cap.

Constraints (per user request):
- Read-only w.r.t. the repo's tracked files. This file is written to the repo
  root with a ".scratch_" prefix and should be removed after use.
- Uses the REAL Hugging Face client and the real CorpusBuilder/hf_access code.

Run:
  cd environments/pretrain_data_curator
  set -a; source ../../secrets.env; set +a
  PYTHONPATH= ./.venv/bin/python ../../.scratch_pdc_empirical_cap.py

Key hypothesis we are testing empirically:
- The agent's manifest says sample_docs_per_source=100000, but the taskset may
  be *ignoring* it and falling back to its configured default (64) due to
  manifest parsing failing or being overwritten.

We validate by:
1) Loading the agent's final assistant message from the saved run5 results.jsonl
   and feeding it through the real parse_manifest() function.
2) Materializing corpora under both:
   (A) parsed manifest (if parse succeeds)
   (B) a simulated fallback manifest (sample_docs_per_source=None), which forces
       the CorpusBuilder default cap to apply.

For each corpus materialization, we print per-source:
- requested n (FetchKey.n)
- raw docs returned by HF streaming
- docs remaining after each filter (min_chars, dedup_exact) including dup count
- estimated tokens

This should definitively locate where the token count collapses.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Iterable

from pretrain_data_curator.corpus import (
    CorpusBuilder,
    DocumentFilter,
    _est_fetch_docs,
    _weight_token_target,
)
from pretrain_data_curator.hf_access import (
    FetchKey,
    HuggingFaceDatasetClient,
    RetryPolicy,
    estimate_tokens,
)
from pretrain_data_curator.models import FilterSpec, Manifest, Sampling, Source
from pretrain_data_curator.rollout_state import CuratorState, RolloutStore
from pretrain_data_curator.taskset import parse_manifest


RUN5_RESULTS = (
    "environments/pretrain_data_curator/outputs/"
    "pretrain-data-curator--deepseek--deepseek-v4-flash--codex/"
    "412d56ff-0257-4328-a4b8-b40cc2c62a75/results.jsonl"
)

_HF_ID_RE = re.compile(
    r"(?<![:/\w])([A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]+)(?![/\w])"
)


def _msg_text(msg: dict[str, Any]) -> str:
    c = msg.get("content")
    if c is None:
        return ""
    if isinstance(c, list):
        return "".join(p.get("text", "") for p in c if isinstance(p, dict))
    return str(c)


def _read_results_first_line(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.loads(f.readline())


def _read_run5_last_assistant_message(results_path: str) -> str:
    obj = _read_results_first_line(results_path)
    last_text = ""
    for n in obj.get("nodes", []):
        msg = n.get("message") or {}
        if msg.get("role") != "assistant":
            continue
        text = _msg_text(msg)
        if text.strip():
            last_text = text
    return last_text


def _extract_observed_ids_from_results_jsonl(results_path: str) -> list[str]:
    """Best-effort stand-in for taskset._ids_from_trace using only results.jsonl.

    We scan tool-output nodes for "owner/name" substrings and return first-seen
    order.
    """
    obj = _read_results_first_line(results_path)
    seen: dict[str, None] = {}
    for n in obj.get("nodes", []):
        msg = n.get("message") or {}
        if msg.get("role") != "tool":
            continue
        text = _msg_text(msg)
        for m in _HF_ID_RE.finditer(text):
            did = m.group(1)
            # keep it permissive; CorpusBuilder/HF will error if invalid
            if "/" in did and not did.startswith("-"):
                seen.setdefault(did, None)
    return list(seen.keys())


@dataclass
class FilterStat:
    kind: str
    before: int
    after: int
    dropped: int
    extra: dict[str, Any]


def _apply_filters_with_stats(
    docs: list[str], filters: list[FilterSpec]
) -> tuple[list[str], list[FilterStat]]:
    """Apply real filter semantics (matching DocumentFilter) but record deltas."""
    filt = DocumentFilter()
    stats: list[FilterStat] = []
    cur = list(docs)
    for spec in filters:
        before = len(cur)
        extra: dict[str, Any] = {}
        if spec.kind == "dedup_exact":
            seen: set[str] = set()
            out: list[str] = []
            dup = 0
            for doc in cur:
                key = doc.strip()
                if key in seen:
                    dup += 1
                    continue
                seen.add(key)
                out.append(doc)
            after_docs = out
            extra["duplicates"] = dup
        else:
            after_docs = filt._apply_one(cur, spec)  # noqa: SLF001
        after = len(after_docs)
        stats.append(
            FilterStat(
                kind=spec.kind,
                before=before,
                after=after,
                dropped=before - after,
                extra=extra,
            )
        )
        cur = after_docs
    return cur, stats


def _sum_tokens(docs: Iterable[str]) -> int:
    return sum(estimate_tokens(d) for d in docs)


async def _materialize_with_breakdown(
    *, builder: CorpusBuilder, manifest: Manifest, label: str
) -> None:
    state = CuratorState()
    RolloutStore.init(state, manifest, RolloutStore.ledger(state))

    total_weight = sum(s.weight for s in manifest.sources)
    cap = manifest.sample_docs_per_source or builder._sample_docs_per_source  # noqa: SLF001

    print(f"\n\n================ {label} ================")
    print(f"manifest.sample_docs_per_source={manifest.sample_docs_per_source!r}")
    print(f"builder.default_sample_docs_per_source={builder._sample_docs_per_source}")  # noqa: SLF001
    print(f"effective_cap={cap}")
    print(f"sources={len(manifest.sources)} token_budget={manifest.token_budget}")

    grand_docs = 0
    grand_tokens = 0

    for idx, source in enumerate(manifest.sources):
        print("\n---")
        print(
            f"[{idx}] {source.dataset_id} config={source.config!r} split={source.split} "
            f"text_field={source.text_field!r} weight={source.weight}"
        )

        weight_target = _weight_token_target(source, manifest.token_budget, total_weight)
        if weight_target is None:
            n = cap
        else:
            n = _est_fetch_docs(weight_target, cap)

        key = FetchKey(
            dataset_id=source.dataset_id,
            config=source.config,
            split=source.split,
            text_field=source.text_field,
            n=n,
        )

        t0 = time.perf_counter()
        try:
            raw, err = await builder.fetch_source_docs(state, key)
        except Exception as exc:  # noqa: BLE001
            dt = time.perf_counter() - t0
            print(f"FETCH EXCEPTION after {dt:.2f}s: {type(exc).__name__}: {exc}")
            continue
        dt = time.perf_counter() - t0

        if err is not None:
            print(f"FETCH ERROR after {dt:.2f}s: {err}")
            continue

        print(f"requested_n={n} weight_target={weight_target} fetch_time_s={dt:.2f}")
        print(f"raw_docs_returned={len(raw)} raw_tokens_est={_sum_tokens(raw)}")

        filtered, fstats = _apply_filters_with_stats(raw, source.filters)
        for st in fstats:
            extra = (" " + json.dumps(st.extra, sort_keys=True)) if st.extra else ""
            print(
                f"  filter={st.kind:12s} before={st.before:6d} after={st.after:6d} "
                f"dropped={st.dropped:6d}{extra}"
            )

        sampled = builder._apply_sampling(filtered, source, weight_target)  # noqa: SLF001
        tok = _sum_tokens(sampled)
        print(f"post_sampling_docs={len(sampled)} post_sampling_tokens_est={tok}")

        grand_docs += len(sampled)
        grand_tokens += tok

    print("\n=== Totals (sum of per-source post-sampling) ===")
    print(f"total_docs={grand_docs}")
    print(f"total_tokens_est={grand_tokens}")

    corpus = await builder.materialize(manifest, state)
    print("\n=== Cross-check (CorpusBuilder.materialize) ===")
    print(f"materialize.total_docs={len(corpus.documents)}")
    print(f"materialize.total_tokens_est={corpus.total_tokens}")


async def main() -> None:
    if not os.environ.get("HF_TOKEN"):
        raise SystemExit(
            "HF_TOKEN is not set in environment. Run: set -a; source ../../secrets.env; set +a"
        )

    last_msg = _read_run5_last_assistant_message(RUN5_RESULTS)
    parsed = parse_manifest(last_msg, default_token_budget=400_000_000)

    print("=== Run5 manifest parse check (real parse_manifest) ===")
    print(f"results_path={RUN5_RESULTS}")
    print(f"last_assistant_msg_chars={len(last_msg)}")
    print(f"parse_manifest_is_none={parsed is None}")
    if parsed is not None:
        print(f"parsed.token_budget={parsed.token_budget}")
        print(f"parsed.sample_docs_per_source={parsed.sample_docs_per_source!r}")
        print(f"parsed.sources={len(parsed.sources)}")
        for s in parsed.sources:
            print(
                f"  - {s.dataset_id} config={s.config!r} split={s.split} "
                f"text_field={s.text_field!r} weight={s.weight}"
            )

    observed = _extract_observed_ids_from_results_jsonl(RUN5_RESULTS)
    print("\n=== Observed ids from tool outputs (best-effort) ===")
    print(f"observed_count={len(observed)}")
    for did in observed[:30]:
        print(f"  {did}")

    # Prepare HF client + builder with generous timeout.
    client = HuggingFaceDatasetClient(token_env="HF_TOKEN")
    policy = RetryPolicy(attempts=5, timeout=300.0)
    builder = CorpusBuilder(
        client=client,
        sample_docs_per_source=64,  # matches the taskset config default in eval.log
        retry_policy=policy,
        fetch_limit=1,  # keep it serial for clearer timing output
    )

    if parsed is not None:
        await _materialize_with_breakdown(builder=builder, manifest=parsed, label="PARSED MANIFEST")

    # Construct a simulated fallback manifest (sample_docs_per_source=None) to
    # observe the hard cap behavior.
    preferred = [
        "wikimedia/wikipedia",
        "allenai/c4",
        "Salesforce/wikitext",
        "roneneldan/TinyStories",
        "HuggingFaceFW/fineweb",
        "Skylion007/openwebtext",
    ]
    picked: list[str] = []
    for did in preferred:
        if did in observed and did not in picked:
            picked.append(did)
    for did in observed:
        if did not in picked:
            picked.append(did)
        if len(picked) >= 6:
            break

    filters = [
        FilterSpec(kind="min_chars", params={"value": 200}),
        FilterSpec(kind="dedup_exact", params={}),
    ]
    fallback_sources = [
        Source(
            dataset_id=did,
            config=None,
            split="train",
            text_field=None,
            weight=1.0,
            filters=filters,
            sampling=Sampling(max_docs=None, max_tokens=None),
        )
        for did in picked
    ]
    fallback_manifest = Manifest(
        token_budget=400_000_000,
        sample_docs_per_source=None,
        sources=fallback_sources,
    )
    await _materialize_with_breakdown(builder=builder, manifest=fallback_manifest, label="SIMULATED FALLBACK MANIFEST (sample_docs_per_source=None)")


if __name__ == "__main__":
    asyncio.run(main())
