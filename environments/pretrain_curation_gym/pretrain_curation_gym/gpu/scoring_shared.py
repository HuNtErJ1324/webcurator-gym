"""Stdlib-only scoring helpers shared by host and rendered self-score code."""

import hashlib
import re
from collections.abc import Iterable, Iterator, Mapping

CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    return max(len(text.split()), len(text) // CHARS_PER_TOKEN)


def weighted_token_target(weight: float, total_weight: float, token_budget: int):
    if total_weight <= 0:
        return None
    return int(token_budget * weight / total_weight)


def build_decon_detect_command(
    binary,
    *,
    training_dir,
    evals_dir,
    report_output_dir,
    threshold,
    content_key="text",
    tokenizer="cl100k",
    ngram_size=5,
    extra_args=(),
):
    return [
        binary,
        "detect",
        "--training-dir",
        training_dir,
        "--content-key",
        content_key,
        "--evals-dir",
        evals_dir,
        "--report-output-dir",
        report_output_dir,
        "--tokenizer",
        tokenizer,
        "--ngram-size",
        str(ngram_size),
        "--contamination-score-threshold",
        str(threshold),
        *extra_args,
    ]


def _parts(spec):
    if isinstance(spec, Mapping):
        return spec.get("kind"), spec.get("params") or {}
    return getattr(spec, "kind", None), getattr(spec, "params", {}) or {}


def _dedup_exact(docs: Iterable[str]) -> Iterator[str]:
    seen = set()
    for doc in docs:
        digest = hashlib.blake2b(doc.strip().encode("utf-8"), digest_size=16).digest()
        if digest in seen:
            continue
        seen.add(digest)
        yield doc


def apply_filters_iter(docs: Iterable[str], filters) -> Iterator[str]:
    """Apply supported filters in order, skipping malformed specifications."""
    stream = iter(docs)
    for spec in filters or ():
        kind, params = _parts(spec)
        try:
            if kind == "min_chars":
                threshold = int(params.get("value", 0))
                stream = filter(
                    lambda doc, threshold=threshold: len(doc) >= threshold, stream
                )
            elif kind == "max_chars":
                threshold = int(params.get("value", 10**9))
                stream = filter(
                    lambda doc, threshold=threshold: len(doc) <= threshold, stream
                )
            elif kind == "min_tokens":
                threshold = int(params.get("value", 0))
                stream = filter(
                    lambda doc, threshold=threshold: estimate_tokens(doc) >= threshold,
                    stream,
                )
            elif kind == "max_symbol_ratio":
                threshold = float(params.get("value", 1.0))
                stream = filter(
                    lambda doc, threshold=threshold: (
                        sum(
                            not (char.isalnum() or char.isspace()) for char in doc
                        )
                        / len(doc)
                        if doc
                        else 1.0
                    )
                    <= threshold,
                    stream,
                )
            elif kind == "min_alpha_ratio":
                threshold = float(params.get("value", 0.0))
                stream = filter(
                    lambda doc, threshold=threshold: (
                        sum(char.isalpha() for char in doc) / len(doc)
                        if doc
                        else 0.0
                    )
                    >= threshold,
                    stream,
                )
            elif kind in {"drop_regex", "keep_regex"}:
                pattern = re.compile(str(params.get("pattern", "")))
                keep_match = kind == "keep_regex"
                stream = filter(
                    lambda doc, pattern=pattern, keep_match=keep_match: bool(
                        pattern.search(doc)
                    )
                    is keep_match,
                    stream,
                )
            elif kind == "dedup_exact":
                stream = _dedup_exact(stream)
        except (TypeError, ValueError, OverflowError, re.error):
            continue
    return iter(stream)


def apply_filters(docs: Iterable[str], filters) -> list[str]:
    """Materializing adapter used by the standalone self-score program."""
    return list(apply_filters_iter(docs, filters))


__all__ = [
    "CHARS_PER_TOKEN",
    "apply_filters",
    "apply_filters_iter",
    "build_decon_detect_command",
    "estimate_tokens",
    "weighted_token_target",
]
