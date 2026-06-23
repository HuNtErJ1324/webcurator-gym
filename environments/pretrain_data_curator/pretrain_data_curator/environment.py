"""The multi-turn curation environment.

`PretrainDataCuratorEnv` is a `StatefulToolEnv`: the agent searches the
pre-cutoff Hugging Face universe, inspects datasets, edits a weighted/filtered
manifest, previews stats, and finalizes. All tools share the rollout manifest and
a single cost ledger via `RolloutStore`.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import verifiers as vf

from .corpus import CorpusBuilder
from .hf_access import (
    DatasetAccessError,
    DatasetSearchClient,
    FetchKey,
    RetryPolicy,
    candidate_from_info,
    estimate_tokens,
    hf_fetch_semaphore,
    parse_cutoff,
    query_variants,
    search_with_retry,
)
from .leakage import LeakageDetector
from .models import CostLedger, CuratorConfig, FilterSpec, Manifest, Sampling, Source
from .rollout_state import RolloutStore


class PretrainDataCuratorEnv(vf.StatefulToolEnv):
    def __init__(
        self,
        *,
        client: DatasetSearchClient,
        config: CuratorConfig,
        corpus_builder: CorpusBuilder,
        leakage_detector: LeakageDetector,
        **kwargs: Any,
    ) -> None:
        self.client = client
        self.config = config
        self.corpus_builder = corpus_builder
        self.leakage_detector = leakage_detector
        self._cutoff = parse_cutoff(config.cutoff_date)
        self._fetch_policy = RetryPolicy(
            attempts=config.fetch_max_attempts, timeout=config.fetch_timeout_seconds
        )
        self.curator_rubric = kwargs.get("rubric")
        super().__init__(tools=[], max_turns=config.max_turns, **kwargs)
        self.add_tool(self.search_datasets, args_to_skip=["state"])
        self.add_tool(self.inspect_dataset, args_to_skip=["state"])
        self.add_tool(self.set_source, args_to_skip=["state"])
        self.add_tool(self.remove_source, args_to_skip=["state"])
        # Config-driven tool availability: a disabled tool is never advertised
        # to the model (no schema, no tool metric).
        if config.enable_run_code:
            self.add_tool(self.run_code, args_to_skip=["state"])
        self.add_tool(self.compute_manifest_stats, args_to_skip=["state"])
        self.add_tool(self.finalize_manifest, args_to_skip=["state"])

    async def setup_state(self, state: vf.State) -> None:
        await super().setup_state(state)
        info = self._coerce_info(state.get("info"))
        token_budget = int(info.get("token_budget", self.config.token_budget))
        manifest = Manifest(token_budget=token_budget)
        RolloutStore.init(state, manifest, CostLedger())
        state["cutoff_date"] = self._cutoff.date().isoformat()

    @staticmethod
    def _coerce_info(info: Any) -> dict[str, Any]:
        """Read per-row `info` as a dict, parsing a JSON string if needed.

        Datasets sometimes serialize `info` as a JSON string; reading it as a
        dict unconditionally would silently drop per-row overrides (e.g.
        `token_budget`). Parsing on read keeps the override live either way.
        """
        if isinstance(info, str):
            try:
                info = json.loads(info)
            except (json.JSONDecodeError, ValueError):
                return {}
        return info if isinstance(info, dict) else {}

    def update_tool_args(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
        messages: vf.Messages,
        state: vf.State,
        **kwargs: Any,
    ) -> dict[str, Any]:
        tool_args["state"] = state
        return tool_args

    async def search_datasets(
        self, query: str, state: vf.State, limit: int | None = None
    ) -> str:
        """Search the pre-cutoff Hugging Face dataset universe.

        Args:
            query: Free-text search query for Hugging Face datasets.
            limit: Maximum candidates to return (defaults to the configured limit).
        """
        result_limit = min(max(limit or self.config.candidate_limit, 1), self.config.candidate_limit)
        scan_limit = max(self.config.scan_limit, result_limit)
        variants = query_variants(query)
        ledger = RolloutStore.ledger(state)
        candidates_cache = RolloutStore.candidates(state)

        found: dict[str, dict[str, Any]] = {}
        excluded_after_cutoff = 0
        used_variants: list[str] = []
        last_error: dict[str, Any] | None = None
        semaphore = hf_fetch_semaphore(self.config.max_concurrent_fetches)
        for variant in variants:
            used_variants.append(variant)
            ledger.web_queries += 1
            ledger.hub_calls += 1
            try:
                raw = await search_with_retry(
                    self.client.search_datasets,
                    variant,
                    scan_limit,
                    policy=self._fetch_policy,
                    semaphore=semaphore,
                )
            except DatasetAccessError as exc:
                # A failed query variant must not crash the tool call: record
                # telemetry and try the next (broader) variant.
                RolloutStore.record_tool_error(state, exc.kind)
                RolloutStore.set_external_failure(state, True)
                last_error = exc.as_dict()
                continue
            for info in raw:
                candidate = candidate_from_info(info)
                if candidate is None:
                    continue
                if candidate.modified_at > self._cutoff:
                    excluded_after_cutoff += 1
                    continue
                found[candidate.dataset_id] = candidate.as_dict()
            if found:
                break

        ranked = sorted(
            found.values(),
            key=lambda c: (-c["downloads"], -c["likes"], c["dataset_id"]),
        )[:result_limit]
        for entry in ranked:
            candidates_cache[entry["dataset_id"]] = entry
        RolloutStore.set_ledger(state, ledger)
        payload: dict[str, Any] = {
            "cutoff_date": state["cutoff_date"],
            "query": query,
            "attempted_queries": used_variants,
            "excluded_after_cutoff": excluded_after_cutoff,
            "candidates": ranked,
        }
        if not ranked and last_error is not None:
            payload["error"] = last_error["error"]
            payload["error_kind"] = last_error["error_kind"]
        return self._json(payload)

    async def inspect_dataset(self, dataset_id: str, state: vf.State) -> str:
        """Sample documents from a candidate dataset and report quick statistics.

        Args:
            dataset_id: A dataset id previously returned by search_datasets.
        """
        candidates = RolloutStore.candidates(state)
        if dataset_id not in candidates:
            return self._json(
                {
                    "error": "dataset must be discovered via search_datasets first",
                    "known_dataset_ids": sorted(candidates.keys()),
                }
            )
        # Fetch through the shared per-rollout cache so this preview and the
        # final corpus build observe identical docs and cost is charged once.
        key = FetchKey(
            dataset_id=dataset_id,
            config=None,
            split="train",
            text_field="text",
            n=self.config.sample_docs_per_source,
        )
        docs, error = await self.corpus_builder.fetch_source_docs(state, key)
        if error is not None:
            return self._json({"dataset_id": dataset_id, **error})
        stats = {
            "dataset_id": dataset_id,
            "sampled_docs": len(docs),
            "avg_chars": round(sum(len(d) for d in docs) / len(docs), 1) if docs else 0.0,
            "avg_tokens": round(
                sum(estimate_tokens(d) for d in docs) / len(docs), 1
            ) if docs else 0.0,
            "preview": docs[0][:300] if docs else "",
        }
        RolloutStore.inspected(state)[dataset_id] = stats
        return self._json(stats)

    async def set_source(
        self,
        dataset_id: str,
        state: vf.State,
        weight: float = 1.0,
        text_field: str = "text",
        split: str = "train",
        config: str | None = None,
        filters: str | None = None,
        max_docs: int | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Add or update a source in the curation manifest.

        Args:
            dataset_id: A dataset id previously returned by search_datasets.
            weight: Relative mixing weight (>= 0) for this source.
            text_field: Name of the text column to read.
            split: Dataset split to draw from.
            config: Optional dataset config/subset name.
            filters: JSON array string of document filters, each
                {"kind": str, "params": {...}}. Example:
                '[{"kind": "min_chars", "params": {"value": 200}}]'.
            max_docs: Optional cap on number of documents after filtering.
            max_tokens: Optional cap on estimated tokens after filtering.
        """
        candidates = RolloutStore.candidates(state)
        if dataset_id not in candidates:
            return self._json(
                {
                    "error": "dataset must be discovered via search_datasets first",
                    "known_dataset_ids": sorted(candidates.keys()),
                }
            )
        try:
            filter_specs = self._parse_filters(filters)
        except (KeyError, TypeError, ValueError) as exc:
            return self._json({"error": f"invalid filters: {exc}"})

        source = Source(
            dataset_id=dataset_id,
            config=config,
            split=split,
            text_field=text_field,
            weight=max(weight, 0.0),
            filters=filter_specs,
            sampling=Sampling(max_docs=max_docs, max_tokens=max_tokens),
        )
        manifest = RolloutStore.manifest(state)
        manifest.upsert_source(source)
        RolloutStore.set_manifest(state, manifest)
        RolloutStore.set_finalized(state, False)
        return self._json(self._manifest_summary(manifest))

    async def remove_source(
        self, dataset_id: str, state: vf.State, config: str | None = None
    ) -> str:
        """Remove a source from the manifest.

        Args:
            dataset_id: The dataset id to remove.
            config: Optional config/subset name that identifies the source.
        """
        manifest = RolloutStore.manifest(state)
        removed = manifest.remove_source(dataset_id, config)
        RolloutStore.set_manifest(state, manifest)
        RolloutStore.set_finalized(state, False)
        return self._json({"removed": removed, **self._manifest_summary(manifest)})

    async def run_code(self, code: str, state: vf.State) -> str:
        """Run a Python snippet to design a filter (stub; opt-in via config).

        Only advertised to the model when ``CuratorConfig.enable_run_code`` is
        true; sandboxed execution is not implemented here, so it returns guidance
        toward structured `filters` on set_source.

        Args:
            code: Python source. Sandbox code execution is not enabled here; use
                structured `filters` on set_source instead.
        """
        ledger = RolloutStore.ledger(state)
        ledger.code_calls += 1
        RolloutStore.set_ledger(state, ledger)
        return self._json(
            {
                "error": "code execution is not enabled in this build",
                "hint": "express filtering via the `filters` argument of set_source",
                "supported_filter_kinds": [
                    "min_chars",
                    "max_chars",
                    "min_tokens",
                    "max_symbol_ratio",
                    "min_alpha_ratio",
                    "drop_regex",
                    "keep_regex",
                    "dedup_exact",
                ],
            }
        )

    async def compute_manifest_stats(self, state: vf.State) -> str:
        """Preview quality, diversity, leakage, and cost without training.

        Materialization goes through the shared per-rollout document cache, so
        the preview observes the same documents the final scoring will, and the
        token/corpus cost is charged exactly once across preview and scoring.
        """
        manifest = RolloutStore.manifest(state)
        if not manifest.sources:
            return self._json({"error": "manifest is empty; add a source first"})
        corpus = await self.corpus_builder.materialize(manifest, state)
        ledger = RolloutStore.ledger(state)
        leakage = await asyncio.to_thread(self.leakage_detector.score, corpus.documents)
        return self._json(
            {
                **self._manifest_summary(manifest),
                "materialized_docs": len(corpus.documents),
                "materialized_tokens": corpus.total_tokens,
                "leakage": leakage.as_dict(),
                "estimated_cost": round(ledger.total(self.config.prices), 4),
                "external_failure": RolloutStore.has_external_failure(state),
            }
        )

    async def finalize_manifest(self, state: vf.State) -> str:
        """Finalize the manifest so it is scored at the end of the rollout."""
        manifest = RolloutStore.manifest(state)
        if not manifest.sources:
            return self._json({"error": "cannot finalize an empty manifest"})
        RolloutStore.set_finalized(state, True)
        return self._json(
            {
                "finalized": True,
                "state_schema_version": RolloutStore.schema_version(state),
                "state_hash": RolloutStore.canonical_hash(state),
                **self._manifest_summary(manifest),
            }
        )

    @staticmethod
    def _parse_filters(filters: str | None) -> list[FilterSpec]:
        if not filters:
            return []
        parsed = json.loads(filters)
        if not isinstance(parsed, list):
            raise ValueError("filters must be a JSON array")
        return [
            FilterSpec(kind=str(f["kind"]), params=dict(f.get("params", {})))
            for f in parsed
        ]

    def _manifest_summary(self, manifest: Manifest) -> dict[str, Any]:
        weights = manifest.normalized_weights()
        return {
            "token_budget": manifest.token_budget,
            "num_sources": len(manifest.sources),
            "sources": [
                {
                    "dataset_id": s.dataset_id,
                    "weight": round(weights.get(s.dataset_id, 0.0), 4),
                    "num_filters": len(s.filters),
                }
                for s in manifest.sources
            ],
        }

    @staticmethod
    def _json(payload: dict[str, Any]) -> str:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))
