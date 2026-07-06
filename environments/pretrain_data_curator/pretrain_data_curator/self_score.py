"""Render the leakage-safe development self-scoring script for a rollout.

The rendered script is intentionally standalone and standard-library-only so it
works in subprocess, Docker, and Modal harness workspaces. It samples only
candidate training sources named in the agent's draft manifest. The configured
final-validation repository is represented only by a SHA-256 digest and rejected
before any network request; the script contains no validation filename, tokens,
decoded leakage reference, or final-scoring implementation.
"""

from __future__ import annotations

import hashlib
from textwrap import dedent

from .models import CuratorConfig
from .trainer import estimate_param_count

SELF_SCORE_FILENAME = "self_score.py"

_SCRIPT = r'''
#!/usr/bin/env python3
"""Development-only manifest proxy; does not use final validation data."""
import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

EXPECTED_TOKEN_BUDGET = __EXPECTED_TOKEN_BUDGET__
DEFAULT_FETCH_CAP = __DEFAULT_FETCH_CAP__
TARGET_TRAIN_TOKENS = __TARGET_TRAIN_TOKENS__
PERF_BASELINE_LOSS = __PERF_BASELINE_LOSS__
PERF_TARGET_LOSS = __PERF_TARGET_LOSS__
BASELINE_RELATIVE_PERF = __BASELINE_RELATIVE_PERF__
ALPHA_PERF = __ALPHA_PERF__
LAMBDA_LEAKAGE = __LAMBDA_LEAKAGE__
HUB_CALL_PRICE = __HUB_CALL_PRICE__
PER_1K_TOKENS_PRICE = __PER_1K_TOKENS_PRICE__
PER_GFLOP_PRICE = __PER_GFLOP_PRICE__
PARAM_COUNT = __PARAM_COUNT__
FORBIDDEN_SOURCE_SHA256 = "__FORBIDDEN_SOURCE_SHA256__"
HF_TOKEN_ENV = __HF_TOKEN_ENV__
DECON_BINARY = __DECON_BINARY__
DECON_EVALS_DIR = __DECON_EVALS_DIR__
DECON_THRESHOLD = __DECON_THRESHOLD__
DATASETS_SERVER = "https://datasets-server.huggingface.co"
TEXT_FIELDS = ("text", "content", "passage", "abstract")
REDACTED_SOURCE_LABEL = "[withheld validation repository]"


def fail(message):
    print(json.dumps({"ok": False, "error": message}, sort_keys=True))
    raise SystemExit(2)


def request_json(path, params):
    url = DATASETS_SERVER + path + "?" + urllib.parse.urlencode(params)
    headers = {"User-Agent": "pretrain-data-curator-self-score/1"}
    token = os.environ.get(HF_TOKEN_ENV)
    if token:
        headers["Authorization"] = "Bearer " + token
    with urllib.request.urlopen(
        urllib.request.Request(url, headers=headers), timeout=10
    ) as response:
        return json.load(response)


def text_from_row(row, requested):
    if requested and requested in row:
        value = row[requested]
    else:
        value = next((row[k] for k in TEXT_FIELDS if k in row), None)
        if value is None:
            pairs = [
                (row.get("query"), row.get("response")),
                (row.get("prompt"), row.get("completion")),
                (row.get("instruction"), row.get("output")),
            ]
            value = next(
                ("\n".join(str(x) for x in pair if x is not None) for pair in pairs
                 if all(x is not None for x in pair)),
                None,
            )
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return "" if value is None else str(value)


def local_docs(source, limit):
    path = Path(str(source.get("local_path") or ""))
    if not path.is_file() or path.is_absolute() or ".." in path.parts:
        raise ValueError("local_path is missing or unsafe")
    raw = path.read_bytes()[:1_048_576].decode("utf-8", "replace")
    fmt = source.get("local_format", "auto")
    if fmt == "jsonl" or (fmt == "auto" and path.suffix.lower() == ".jsonl"):
        docs = []
        for line in raw.splitlines():
            if len(docs) >= limit:
                break
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, str):
                docs.append(value)
            elif isinstance(value, dict):
                docs.append(text_from_row(value, source.get("text_field")))
        return docs
    return [part.strip() for part in re.split(r"\n\s*\n", raw) if part.strip()][:limit]


def source_dataset_id(source):
    return (
        source.get("dataset_id")
        or source.get("id")
        or source.get("dataset")
        or source.get("repo_id")
        or source.get("name")
        or ""
    )


def is_forbidden_source(dataset_id):
    return hashlib.sha256(str(dataset_id).encode()).hexdigest() == (
        FORBIDDEN_SOURCE_SHA256
    )


def remote_docs(source, limit):
    dataset_id = str(source_dataset_id(source))
    if is_forbidden_source(dataset_id):
        raise ValueError("source is reserved for final validation")
    split = str(source.get("split") or "train")
    config = source.get("config")
    if not config:
        split_rows = request_json("/splits", {"dataset": dataset_id}).get("splits", [])
        match = next((x for x in split_rows if x.get("split") == split), None)
        if match is None and split_rows:
            match = split_rows[0]
            split = str(match.get("split") or split)
        if match is None:
            raise ValueError("datasets-server returned no usable split")
        config = match.get("config")
    payload = request_json(
        "/first-rows",
        {"dataset": dataset_id, "config": config, "split": split},
    )
    return [
        text_from_row(item.get("row") or {}, source.get("text_field"))
        for item in payload.get("rows", [])[:limit]
    ]


def estimate_tokens(text):
    return max(len(text.split()), len(text) // 4)


def apply_filters(docs, filters):
    result = list(docs)
    for spec in filters or []:
        kind = spec.get("kind")
        params = spec.get("params") or {}
        value = params.get("value")
        if kind == "min_chars":
            result = [x for x in result if len(x) >= int(value or 0)]
        elif kind == "max_chars":
            result = [x for x in result if len(x) <= int(value or 0)]
        elif kind == "min_tokens":
            result = [x for x in result if estimate_tokens(x) >= int(value or 0)]
        elif kind == "max_symbol_ratio":
            result = [
                x for x in result
                if not x or sum(not (c.isalnum() or c.isspace()) for c in x) / len(x)
                <= float(value)
            ]
        elif kind == "min_alpha_ratio":
            result = [
                x for x in result
                if x and sum(c.isalpha() for c in x) / len(x) >= float(value)
            ]
        elif kind in ("drop_regex", "keep_regex"):
            pattern = re.compile(str(params.get("pattern") or value or ""))
            result = [
                x for x in result
                if (not pattern.search(x)) == (kind == "drop_regex")
            ]
        elif kind == "dedup_exact":
            result = list(dict.fromkeys(result))
    return result


def _reduce_report(report_lines, total_tokens):
    """Shared reducer: token-weighted contamination from decon report JSONL.

    Mirrors leakage._reduce_report exactly so the dev self-score estimate
    matches the production scorer.  Dedup per (training_file, training_line).
    Token weight prefers cluster_token_length, falling back to span chars // 4.
    """
    best_per_doc = {}
    for line in report_lines:
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue

        doc_key = (r.get("training_file", ""), r.get("training_line", 0))
        score = float(r.get("contamination_score", 0.0))

        cluster_tok = r.get("cluster_token_length")
        if cluster_tok is not None and int(cluster_tok) > 0:
            est_tokens = int(cluster_tok)
        else:
            ans_start = r.get("answer_start_idx")
            ans_end = r.get("answer_end_idx")
            q_start = r.get("question_start_idx")
            q_end = r.get("question_end_idx")

            start = min(
                ans_start if ans_start is not None else q_start or 0,
                q_start if q_start is not None else ans_start or 0,
            )
            end = max(
                ans_end if ans_end is not None else q_end or 0,
                q_end if q_end is not None else ans_end or 0,
            )
            span_chars = max(int(end) - int(start), 1)
            est_tokens = max(1, span_chars // 4)

        contribution = score * est_tokens
        if doc_key not in best_per_doc or contribution > best_per_doc[doc_key]:
            best_per_doc[doc_key] = contribution

    if not best_per_doc:
        return 0.0, 0

    total_weighted = sum(best_per_doc.values())
    leakage = min(1.0, total_weighted / total_tokens)
    return leakage, len(best_per_doc)


def decon_score(docs):
    """Run decon on sampled documents, return (leakage_score, num_matches) or (None, None)."""
    binary = DECON_BINARY
    if not binary or binary == "decon" or not os.path.isfile(binary):
        paths_to_try = [
            os.path.join(os.path.dirname(__file__), "..", "decon", "bin", "decon"),
        ]
        for p in paths_to_try:
            if os.path.isfile(p):
                binary = p
                break
        else:
            print("[self-score] WARNING: decon binary not found, skipping leakage check", file=sys.stderr)
            return None, None
    if not DECON_EVALS_DIR or not os.path.isdir(DECON_EVALS_DIR):
        print("[self-score] WARNING: decon evals dir not found, skipping leakage check", file=sys.stderr)
        return None, None
    tmp = tempfile.mkdtemp(prefix="decon_selfscore_")
    try:
        corpus_path = os.path.join(tmp, "corpus.jsonl")
        with open(corpus_path, "w") as fh:
            for doc in docs:
                if doc:
                    fh.write(json.dumps({"text": doc}, ensure_ascii=False) + "\n")
        total_chars = sum(len(d) for d in docs if d)
        total_tok = max(1, total_chars // 4)

        report_dir = os.path.join(tmp, "report")
        os.makedirs(report_dir, exist_ok=True)
        result = subprocess.run(
            [
                binary, "detect",
                "--training-dir", tmp,
                "--content-key", "text",
                "--evals-dir", DECON_EVALS_DIR,
                "--report-output-dir", report_dir,
                "--contamination-score-threshold", str(DECON_THRESHOLD),
            ],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            print("[self-score] WARNING: decon exited %d: %s" % (
                result.returncode, result.stderr[:200],
            ), file=sys.stderr)
            return None, None

        report_lines = []
        for fname in os.listdir(report_dir):
            if not fname.endswith(".jsonl"):
                continue
            with open(os.path.join(report_dir, fname)) as fh:
                report_lines.extend(fh.readlines())

        if not report_lines:
            return 0.0, 0

        return _reduce_report(report_lines, total_tok)
    except subprocess.TimeoutExpired:
        print("[self-score] WARNING: decon timed out after 600s", file=sys.stderr)
        return None, None
    except Exception as exc:
        print("[self-score] WARNING: decon failed: %s" % exc, file=sys.stderr)
        return None, None
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="Leakage-safe development proxy for a draft curator manifest."
    )
    parser.add_argument("manifest", help="draft manifest JSON file")
    parser.add_argument("--limit", type=int, default=8, choices=range(1, 65),
                        metavar="N", help="candidate rows sampled per source (1-64)")
    args = parser.parse_args()
    try:
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail("cannot read manifest: " + str(exc))
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        fail("manifest must contain a non-empty sources list")
    try:
        token_budget = int(manifest.get("token_budget", EXPECTED_TOKEN_BUDGET))
    except (TypeError, ValueError):
        fail("token_budget must be an integer")
    if token_budget != EXPECTED_TOKEN_BUDGET:
        fail("token_budget must equal the task allocation")

    cap = manifest.get("sample_docs_per_source") or DEFAULT_FETCH_CAP
    cap = max(1, min(int(cap), 100_000))
    weights = [max(0.0, float(source.get("weight", 1.0))) for source in sources]
    total_weight = sum(weights)
    if total_weight <= 0:
        fail("at least one source weight must be positive")

    source_stats = []
    estimated_total = 0
    clean_sum = 0.0
    clean_count = 0
    hub_calls = 0
    all_docs: list[str] = []
    for source, weight in zip(sources, weights):
        kind = source.get("kind", "hf")
        dataset_id = str(source_dataset_id(source))
        if kind == "local":
            label = source.get("local_path")
        elif is_forbidden_source(dataset_id):
            label = REDACTED_SOURCE_LABEL
        else:
            label = dataset_id
        try:
            docs = (
                local_docs(source, args.limit)
                if kind == "local"
                else remote_docs(source, args.limit)
            )
            if kind != "local":
                hub_calls += 2 if not source.get("config") else 1
            docs = [x for x in apply_filters(docs, source.get("filters")) if x]
            all_docs.extend(docs)
            sample_tokens = sum(estimate_tokens(x) for x in docs)
            average_tokens = sample_tokens / len(docs) if docs else 0.0
            target = int(token_budget * weight / total_weight)
            requested = min(max(target // 250, 1), cap) if weight > 0 else 0
            estimated_tokens = min(target, int(average_tokens * requested))
            error = None
            for doc in docs:
                clean_sum += sum(c.isalpha() or c.isspace() for c in doc) / len(doc)
                clean_count += 1
        except Exception as exc:
            docs, sample_tokens, estimated_tokens = [], 0, 0
            error = f"{type(exc).__name__}: {exc}"
        estimated_total += estimated_tokens
        source_stats.append({
            "source": label,
            "sampled_documents": len(docs),
            "sampled_tokens": sample_tokens,
            "estimated_materialized_tokens": estimated_tokens,
            "error": error,
        })

    fill = min(1.0, estimated_total / token_budget)
    nonzero = [x["estimated_materialized_tokens"] for x in source_stats
               if x["estimated_materialized_tokens"] > 0]
    if len(nonzero) <= 1:
        diversity = 0.0
    else:
        total = sum(nonzero)
        proportions = [x / total for x in nonzero]
        diversity = -sum(p * math.log(p) for p in proportions) / math.log(len(nonzero))
    cleanliness = clean_sum / clean_count if clean_count else 0.0
    trained_tokens = min(estimated_total, TARGET_TRAIN_TOKENS)
    scale = math.log1p(trained_tokens) / math.log1p(max(TARGET_TRAIN_TOKENS, 1))
    quality_gain = 0.6 * scale + 0.25 * cleanliness + 0.15 * diversity
    proxy_ce = max(0.2, 5.0 * (1.0 - 0.85 * quality_gain)) if nonzero else None
    if proxy_ce is None:
        perf = 0.0
        flops = 0.0
    else:
        perf = (
            (PERF_BASELINE_LOSS - proxy_ce) / (PERF_BASELINE_LOSS - PERF_TARGET_LOSS)
            if BASELINE_RELATIVE_PERF else math.exp(-proxy_ce)
        )
        flops = 6.0 * PARAM_COUNT * trained_tokens
    scoring_cost = (
        hub_calls * HUB_CALL_PRICE
        + estimated_total / 1000.0 * PER_1K_TOKENS_PRICE
        + flops / 1e9 * PER_GFLOP_PRICE
    )

    leakage_score, num_matches = decon_score(all_docs) if all_docs else (None, None)
    reward = ALPHA_PERF * perf
    if leakage_score is not None:
        reward -= LAMBDA_LEAKAGE * leakage_score

    print(json.dumps({
        "ok": True,
        "signal": "development-only heuristic; not the final score",
        "validation_data_used": False,
        "estimated_proxy_ce": proxy_ce,
        "estimated_performance": perf,
        "estimated_budget_fill_ratio": fill,
        "estimated_scoring_cost": scoring_cost,
        "estimated_reward": reward,
        "leakage_estimate": leakage_score,
        "num_contaminated_matches": num_matches,
        "sources": source_stats,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
'''


def render_self_score_script(
    config: CuratorConfig,
    *,
    hf_token_env: str = "HF_TOKEN",
    decon_binary: str = "decon",
    decon_evals_dir: str | None = None,
    decon_threshold: float = 0.2,
) -> bytes:
    """Return a configured self-score script without exposing held-out data."""
    import os as _os
    from .leakage import DEFAULT_EVAL_SETS_DIR as _DEFAULT_EVAL_SETS_DIR
    evals_dir = decon_evals_dir or _DEFAULT_EVAL_SETS_DIR
    replacements: dict[str, object] = {
        "__EXPECTED_TOKEN_BUDGET__": config.token_budget,
        "__DEFAULT_FETCH_CAP__": config.sample_docs_per_source,
        "__TARGET_TRAIN_TOKENS__": config.proxy_student.effective_train_tokens,
        "__PERF_BASELINE_LOSS__": repr(config.perf_baseline_loss),
        "__PERF_TARGET_LOSS__": repr(config.perf_target_loss),
        "__BASELINE_RELATIVE_PERF__": repr(config.baseline_relative_perf),
        "__ALPHA_PERF__": repr(config.alpha_perf),
        "__LAMBDA_LEAKAGE__": repr(config.lambda_leakage),
        "__HUB_CALL_PRICE__": repr(config.prices.hub_call),
        "__PER_1K_TOKENS_PRICE__": repr(config.prices.per_1k_tokens),
        "__PER_GFLOP_PRICE__": repr(config.prices.per_gflop),
        "__PARAM_COUNT__": estimate_param_count(config.proxy_student),
        "__FORBIDDEN_SOURCE_SHA256__": hashlib.sha256(
            config.validation_set.dataset_id.encode()
        ).hexdigest(),
        "__HF_TOKEN_ENV__": repr(hf_token_env),
        "__DECON_BINARY__": repr(decon_binary),
        "__DECON_EVALS_DIR__": repr(_os.path.abspath(evals_dir) if evals_dir else ""),
        "__DECON_THRESHOLD__": repr(decon_threshold),
    }
    script = dedent(_SCRIPT).lstrip()
    for marker, value in replacements.items():
        script = script.replace(marker, str(value))
    return script.encode()


__all__ = ["SELF_SCORE_FILENAME", "render_self_score_script"]
