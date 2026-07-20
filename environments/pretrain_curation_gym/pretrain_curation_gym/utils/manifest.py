"""Manifest parsing and non-production trace recovery."""

from __future__ import annotations

import json
import math
import re
import shlex
from pathlib import PurePosixPath
from typing import Any, Iterable

import verifiers.v1 as vf
from verifiers.v1.types import content_text
from pydantic import ValidationError

from .models import FilterSpec, Manifest, Sampling, Source
class ManifestParser:
    """Tolerant parser for the agent-authored manifest contract."""

    FILTER_KINDS = frozenset(
        {
            "min_chars",
            "max_chars",
            "min_tokens",
            "max_symbol_ratio",
            "min_alpha_ratio",
            "drop_regex",
            "keep_regex",
            "dedup_exact",
        }
    )
    FENCE = re.compile(r"```(?:json|jsonc|JSON)?\s*(.*?)```", re.DOTALL)

    def parse(
        self,
        text: str,
        *,
        default_token_budget: int,
        reserved_local_filename: str | None = None,
    ) -> Manifest | None:
        data = self.extract_object(text)
        if data is None:
            return None
        return self.parse_object(
            data,
            default_token_budget=default_token_budget,
            reserved_local_filename=reserved_local_filename,
        )

    def parse_object(
        self,
        data: dict[str, Any],
        *,
        default_token_budget: int,
        reserved_local_filename: str | None = None,
    ) -> Manifest | None:
        """Parse an already-decoded JSON object (no re-serialization round trip)."""
        raw_sources = data.get("sources")
        if not isinstance(raw_sources, list) or not raw_sources:
            return None

        sources = [source for raw in raw_sources if (source := self.source(raw))]
        if reserved_local_filename:
            sources = [
                source
                for source in sources
                if not (
                    source.kind == "local"
                    and source.local_path
                    and PurePosixPath(source.local_path).name == reserved_local_filename
                )
            ]
        if not sources:
            return None

        token_budget = self.integer(data.get("token_budget")) or default_token_budget
        sample_docs = self.positive_integer(data.get("sample_docs_per_source"))
        try:
            return Manifest(
                token_budget=token_budget,
                sources=sources,
                sample_docs_per_source=sample_docs,
            )
        except ValidationError:
            return None

    def extract_object(self, text: str) -> dict[str, Any] | None:
        """Prefer the last JSON object containing a non-empty sources list."""
        if not text:
            return None
        candidates = [match.group(1) for match in self.FENCE.finditer(text)] + [text]
        last_any: dict[str, Any] | None = None
        last_manifest: dict[str, Any] | None = None
        for candidate in candidates:
            for value in self.json_objects(candidate):
                last_any = value
                if isinstance(value.get("sources"), list) and value["sources"]:
                    last_manifest = value
        return last_manifest or last_any

    @staticmethod
    def json_objects(text: str) -> Iterable[dict[str, Any]]:
        """Yield every top-level JSON object embedded in ``text``."""
        decoder = json.JSONDecoder()
        index = 0
        while (start := text.find("{", index)) >= 0:
            try:
                value, end = decoder.raw_decode(text, start)
            except json.JSONDecodeError:
                index = start + 1
                continue
            if isinstance(value, dict):
                yield value
                index = end
            else:
                index = start + 1

    def source(self, raw: Any) -> Source | None:
        if isinstance(raw, str):
            return Source(dataset_id=raw.strip()) if raw.strip() else None
        if not isinstance(raw, dict):
            return None

        local_path = raw.get("local_path")
        is_local = raw.get("kind") == "local" or local_path is not None
        dataset_id = next(
            (
                raw.get(key)
                for key in ("dataset_id", "id", "dataset", "repo_id", "name")
                if raw.get(key)
            ),
            None,
        )
        if not dataset_id and is_local and isinstance(local_path, str):
            dataset_id = f"local:{local_path.strip()}"
        if not isinstance(dataset_id, str) or not dataset_id.strip():
            return None

        kwargs: dict[str, Any] = {
            "dataset_id": dataset_id.strip(),
            "filters": self.filters(raw.get("filters")),
        }
        if is_local:
            kwargs.update(kind="local", local_path=local_path)
            if raw.get("local_format") is not None:
                kwargs["local_format"] = raw["local_format"]
        for key in ("config", "split", "text_field"):
            if raw.get(key):
                kwargs[key] = str(raw[key])
        if "weight" in raw:
            try:
                kwargs["weight"] = max(0.0, float(raw["weight"]))
            except (TypeError, ValueError):
                pass

        raw_sampling = raw.get("sampling")
        sampling = raw_sampling if isinstance(raw_sampling, dict) else {}
        kwargs["sampling"] = Sampling(
            max_docs=self.positive_integer(
                raw.get("max_docs", sampling.get("max_docs"))
            ),
            max_tokens=self.positive_integer(
                raw.get("max_tokens", sampling.get("max_tokens"))
            ),
        )
        try:
            return Source(**kwargs)
        except ValidationError:
            return None

    def filters(self, raw: Any) -> list[FilterSpec]:
        if not isinstance(raw, list):
            return []
        filters: list[FilterSpec] = []
        for candidate in raw:
            if (
                not isinstance(candidate, dict)
                or candidate.get("kind") not in self.FILTER_KINDS
            ):
                continue
            kind = candidate["kind"]
            params = dict(candidate.get("params") or {})
            try:
                if (
                    kind in {"min_chars", "max_chars", "min_tokens"}
                    and "value" in params
                ):
                    params["value"] = int(params["value"])
                elif (
                    kind in {"max_symbol_ratio", "min_alpha_ratio"}
                    and "value" in params
                ):
                    value = float(params["value"])
                    if not math.isfinite(value):
                        continue
                    params["value"] = value
                elif kind in {"drop_regex", "keep_regex"}:
                    pattern = str(params.get("pattern", ""))
                    re.compile(pattern)
                    params["pattern"] = pattern
            except (TypeError, ValueError, OverflowError, re.error):
                continue
            filters.append(FilterSpec(kind=kind, params=params))
        return filters

    @staticmethod
    def integer(value: Any) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return None

    @classmethod
    def positive_integer(cls, value: Any) -> int | None:
        number = cls.integer(value)
        return number if number is not None and number >= 1 else None


class TraceManifestCandidates:
    """Extract compatibility-only manifest candidates from a v1 trace."""

    HF_ID = re.compile(
        r"(?<![:/\w])"
        r"([A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*)"
        r"(?:/[A-Za-z0-9]+(?:[._-][A-Za-z0-9]+)*)?"
        r"(?![A-Za-z0-9_./-])"
    )
    HF_COMMAND = re.compile(r"(?<![\w./-])hf\s+([^\n;&|`<>]+)")
    INVALID_NAMESPACES = frozenset(
        {
            "http",
            "https",
            "hf",
            "file",
            "s3",
            "gs",
            "az",
            "usr",
            "var",
            "etc",
            "bin",
            "tmp",
            "opt",
            "home",
            "root",
            "datasets",
            "models",
            "spaces",
            "api",
            "v1",
            "v2",
        }
    )
    INVALID_NAMES = frozenset({"train", "test", "validation", "valid", "dev", "split"})
    FILE_EXTENSIONS = frozenset(
        {
            "py",
            "json",
            "jsonl",
            "txt",
            "csv",
            "yaml",
            "yml",
            "toml",
            "sh",
            "md",
            "rst",
            "log",
            "parquet",
            "arrow",
            "gz",
            "zip",
        }
    )

    def __init__(self, parser: ManifestParser) -> None:
        self.parser = parser

    def from_messages(
        self, trace: vf.Trace, *, token_budget: int, reserved_filename: str
    ) -> Manifest | None:
        for message in reversed(trace.assistant_messages):
            manifest = self.parser.parse(
                message.content or "",
                default_token_budget=token_budget,
                reserved_local_filename=reserved_filename,
            )
            if manifest and manifest.sources:
                return manifest
        return None

    def from_dataset_ids(
        self, trace: vf.Trace, *, token_budget: int, limit: int
    ) -> Manifest | None:
        ids = self.dataset_ids(trace)[:limit]
        if not ids:
            return None
        return Manifest(
            token_budget=token_budget,
            sources=[Source(dataset_id=dataset_id) for dataset_id in ids],
        )

    def dataset_ids(self, trace: vf.Trace) -> list[str]:
        inspected: dict[str, None] = {}
        for message in trace.assistant_messages:
            for call in message.tool_calls or []:
                for argv in self.hf_commands(self.shell_command(call.arguments)):
                    if (
                        len(argv) >= 3
                        and argv[:2] == ["datasets", "info"]
                        and "/" in argv[2]
                        and not argv[2].startswith("-")
                    ):
                        inspected[argv[2]] = None
        if inspected:
            return list(inspected)

        observed: dict[str, None] = {}
        for message in trace.tool_messages:
            for match in self.HF_ID.finditer(self.message_text(message.content)):
                dataset_id = match.group(1)
                if self.looks_like_dataset_id(dataset_id):
                    observed.setdefault(dataset_id, None)
        return list(observed)

    @staticmethod
    def shell_command(arguments: Any) -> str:
        raw = arguments if isinstance(arguments, str) else str(arguments or "")
        try:
            payload = json.loads(raw or "{}")
        except (json.JSONDecodeError, TypeError):
            return raw
        if isinstance(payload, dict):
            return next(
                (
                    value
                    for key in ("command", "cmd")
                    if isinstance((value := payload.get(key)), str) and value.strip()
                ),
                raw,
            )
        return raw

    message_text = staticmethod(content_text)

    @classmethod
    def hf_commands(cls, text: str) -> list[list[str]]:
        """Extract observed ``hf`` CLI invocations as argument vectors."""
        commands: list[list[str]] = []
        for match in cls.HF_COMMAND.finditer(text or ""):
            arguments = match.group(1).strip()
            if not arguments:
                continue
            try:
                argv = shlex.split(arguments)
            except ValueError:
                argv = arguments.split()
            if argv:
                commands.append(argv)
        return commands

    def looks_like_dataset_id(self, value: str) -> bool:
        if "/" not in value:
            return False
        namespace, name = value.split("/", 1)
        if not namespace or not name:
            return False
        if (
            namespace.lower() in self.INVALID_NAMESPACES
            or name.lower() in self.INVALID_NAMES
        ):
            return False
        return not (
            "." in name and name.rsplit(".", 1)[-1].lower() in self.FILE_EXTENSIONS
        )


__all__ = ["ManifestParser", "TraceManifestCandidates"]
