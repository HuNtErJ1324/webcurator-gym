"""Render the leakage-safe development self-scoring script for a rollout.

The rendered script samples candidate training sources named in the agent's draft
manifest. The configured final-validation repository is represented only by a
SHA-256 digest and rejected before any network request; the script contains no
validation filename, tokens, decoded leakage reference, or final-scoring
implementation.

When ``use_real_trainer`` is enabled, setup also writes ``self_score_train.py``,
which runs the same proxy-student training recipe as production scoring (minus
the held-out validation shard). The dev script scores corpus-split cross-entropy
plus benchmark decon leakage — the same two reward terms as final scoring.
"""

from __future__ import annotations

import hashlib
from textwrap import dedent

from .models import CuratorConfig
from .trainer import _nanogpt_train_script

SELF_SCORE_FILENAME = "self_score.py"
SELF_SCORE_TRAIN_FILENAME = "self_score_train.py"
SELF_SCORE_TRAIN_TIMEOUT_SECONDS = 900

_SCRIPT = r'''
#!/usr/bin/env python3
"""Development sample scorer: corpus-split CE + benchmark decon only."""
import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

EXPECTED_TOKEN_BUDGET = __EXPECTED_TOKEN_BUDGET__
PERF_BASELINE_LOSS = __PERF_BASELINE_LOSS__
PERF_TARGET_LOSS = __PERF_TARGET_LOSS__
PERF_SCALING_EXPONENT = __PERF_SCALING_EXPONENT__
BASELINE_RELATIVE_PERF = __BASELINE_RELATIVE_PERF__
ALPHA_PERF = __ALPHA_PERF__
LAMBDA_LEAKAGE = __LAMBDA_LEAKAGE__
USE_REAL_TRAINER = __USE_REAL_TRAINER__
TRAIN_SCRIPT_NAME = __TRAIN_SCRIPT_NAME__
STUDENT_CONFIG = __STUDENT_CONFIG__
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


def joined_corpus(docs, cap):
    """Streaming-truncated ``\\n\\n`` join, matching corpus materialization."""
    if cap is None:
        return "\n\n".join(doc for doc in docs if doc)
    parts = []
    total = 0
    sep = "\n\n"
    for doc in docs:
        if not doc:
            continue
        piece = doc if not parts else sep + doc
        if total + len(piece) > cap:
            remaining = cap - total
            if remaining > 0:
                parts.append(piece[:remaining])
            break
        parts.append(piece)
        total += len(piece)
    return "".join(parts)


def scaled_perf(loss):
    if loss is None or not math.isfinite(loss):
        return 0.0
    if BASELINE_RELATIVE_PERF:
        p = (PERF_BASELINE_LOSS - loss) / (PERF_BASELINE_LOSS - PERF_TARGET_LOSS)
        return p ** PERF_SCALING_EXPONENT if p >= 0 else p
    return max(0.0, min(1.0, math.exp(-loss)))


def _reduce_report(report_lines, total_tokens):
    """Shared reducer: token-weighted contamination from decon report JSONL."""
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


def _read_trainer_stderr_tail(workdir, *, max_chars=8000):
    """Read trainer-redirected stderr before the temp workdir is deleted.

    ``self_score_train.py`` redirects ``sys.stderr`` into ``WORKDIR/stderr.txt``
    (line-buffered). Captured subprocess stderr is therefore often empty on
    crash; surface the file tail so CUDA OOM / traceback lines survive cleanup.
    """
    path = os.path.join(workdir, "stderr.txt")
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return ""
    text = text.strip()
    if not text:
        return ""
    return text[-max_chars:]


def decon_score(docs):
    """Run decon on sampled documents, return (leakage_score, num_matches) or (None, None)."""
    # ``DECON_BINARY``/``DECON_EVALS_DIR`` are baked as host absolute paths at
    # render time, but this script runs inside the agent's ``/workspace`` docker
    # harness runtime where those host paths don't exist. The webcurator-runtime
    # image bakes decon at ``<workspace>/decon`` (``COPY decon/ decon/``), so also
    # probe the script-relative, ``/workspace`` and PATH locations before giving up.
    here = os.path.dirname(os.path.abspath(__file__))
    binary = next(
        (
            p
            for p in (
                DECON_BINARY,
                os.path.join(here, "decon", "bin", "decon"),
                os.path.join(here, "..", "decon", "bin", "decon"),
                "/workspace/decon/bin/decon",
                shutil.which("decon") or "",
            )
            if p and os.path.isfile(p)
        ),
        None,
    )
    if binary is None:
        print("[self-score] WARNING: decon binary not found, skipping leakage check", file=sys.stderr)
        return None, None
    evals_dir = next(
        (
            d
            for d in (
                DECON_EVALS_DIR,
                os.path.join(here, "decon", "bundled-evals"),
                os.path.join(here, "..", "decon", "bundled-evals"),
                "/workspace/decon/bundled-evals",
            )
            if d and os.path.isdir(d)
        ),
        None,
    )
    if evals_dir is None:
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
                "--evals-dir", evals_dir,
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
        shutil.rmtree(tmp, ignore_errors=True)


def train_perf(docs, *, max_corpus_chars, max_steps, train_timeout):
    """Train the fixed proxy student on sampled docs; return (loss, perf, backend)."""
    if not USE_REAL_TRAINER or not docs:
        return None, None, None
    train_script = Path(__file__).with_name(TRAIN_SCRIPT_NAME)
    if not train_script.is_file():
        print(
            "[self-score] WARNING: %s not found, skipping CE training" % TRAIN_SCRIPT_NAME,
            file=sys.stderr,
        )
        return None, None, None
    text = joined_corpus(docs, max_corpus_chars)
    if not text.strip():
        return None, None, None

    tmp = tempfile.mkdtemp(prefix="selfscore_train_")
    try:
        steps = int(max_steps if max_steps is not None else STUDENT_CONFIG["steps"])
        warmup_steps = min(int(STUDENT_CONFIG["warmup_steps"]), steps)
        payload = dict(STUDENT_CONFIG)
        payload["steps"] = steps
        payload["warmup_steps"] = warmup_steps
        payload["n_train_runs"] = 1
        with open(os.path.join(tmp, "config.json"), "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        with open(os.path.join(tmp, "corpus.txt"), "w", encoding="utf-8") as fh:
            fh.write(text)
        result = subprocess.run(
            [sys.executable, str(train_script), tmp],
            capture_output=True,
            text=True,
            timeout=train_timeout,
        )
        if result.returncode != 0:
            # Trainer redirects its own stderr into WORKDIR/stderr.txt (line-buffered).
            # Read that tail BEFORE cleanup — captured process stderr is often empty.
            file_stderr = _read_trainer_stderr_tail(tmp)
            detail = file_stderr or result.stderr or result.stdout or ""
            print(
                "[self-score] WARNING: proxy training exited %d: %s"
                % (result.returncode, detail[:2000]),
                file=sys.stderr,
            )
            return None, None, None
        marker = "RESULT_JSON "
        line = next(
            (x for x in reversed((result.stdout or "").splitlines()) if x.startswith(marker)),
            None,
        )
        if line is None:
            result_path = os.path.join(tmp, "result.json")
            if not os.path.isfile(result_path):
                file_stderr = _read_trainer_stderr_tail(tmp)
                detail = file_stderr or result.stderr or result.stdout or ""
                print(
                    "[self-score] WARNING: training produced no result JSON: %s"
                    % detail[:2000],
                    file=sys.stderr,
                )
                return None, None, None
            parsed = json.loads(Path(result_path).read_text(encoding="utf-8"))
        else:
            parsed = json.loads(line[len(marker):])
        loss = float(parsed.get("loss", float("inf")))
        backend = str(parsed.get("val_source") or "sample_ce")
        return loss, scaled_perf(loss), backend
    except subprocess.TimeoutExpired as exc:
        file_stderr = _read_trainer_stderr_tail(tmp)
        detail = file_stderr or getattr(exc, "stderr", None) or getattr(exc, "stdout", None) or ""
        print(
            "[self-score] WARNING: proxy training timed out after %ds: %s"
            % (train_timeout, (detail or "")[:2000]),
            file=sys.stderr,
        )
        return None, None, None
    except Exception as exc:
        file_stderr = _read_trainer_stderr_tail(tmp)
        detail = file_stderr or str(exc)
        print(
            "[self-score] WARNING: proxy training failed: %s" % detail[:2000],
            file=sys.stderr,
        )
        return None, None, None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="Leakage-safe development proxy for a draft curator manifest."
    )
    parser.add_argument("manifest", help="draft manifest JSON file")
    parser.add_argument(
        "--limit",
        type=int,
        default=8,
        metavar="N",
        help="documents sampled per source (agent-chosen; default 8)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        metavar="N",
        help="proxy training steps (default: production student config steps)",
    )
    parser.add_argument(
        "--max-corpus-chars",
        type=int,
        default=None,
        metavar="N",
        help="joined corpus character cap for proxy training (default: all sampled text)",
    )
    parser.add_argument(
        "--train-timeout",
        type=int,
        default=900,
        metavar="SEC",
        help="proxy training wall-clock timeout in seconds (default 900)",
    )
    args = parser.parse_args()
    if args.limit < 1:
        fail("--limit must be >= 1")
    if args.max_steps is not None and args.max_steps < 1:
        fail("--max-steps must be >= 1")
    if args.max_corpus_chars is not None and args.max_corpus_chars < 1:
        fail("--max-corpus-chars must be >= 1")
    if args.train_timeout < 1:
        fail("--train-timeout must be >= 1")
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

    raw_cap = manifest.get("sample_docs_per_source")
    cap = None
    if raw_cap is not None:
        cap = max(1, int(raw_cap))
    weights = [max(0.0, float(source.get("weight", 1.0))) for source in sources]
    total_weight = sum(weights)
    if total_weight <= 0:
        fail("at least one source weight must be positive")

    source_stats = []
    estimated_total = 0
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
            docs = [x for x in apply_filters(docs, source.get("filters")) if x]
            all_docs.extend(docs)
            sample_tokens = sum(estimate_tokens(x) for x in docs)
            average_tokens = sample_tokens / len(docs) if docs else 0.0
            target = int(token_budget * weight / total_weight)
            if weight > 0:
                requested = max(target // 250, 1)
                if cap is not None:
                    requested = min(requested, cap)
            else:
                requested = 0
            estimated_tokens = min(target, int(average_tokens * requested))
            error = None
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
    perf_loss, perf, train_backend = train_perf(
        all_docs,
        max_corpus_chars=args.max_corpus_chars,
        max_steps=args.max_steps,
        train_timeout=args.train_timeout,
    )
    leakage_score, num_matches = decon_score(all_docs) if all_docs else (None, None)

    perf_reward = ALPHA_PERF * (perf or 0.0)
    leakage_penalty = (
        -LAMBDA_LEAKAGE * leakage_score if leakage_score is not None else 0.0
    )
    reward = perf_reward + leakage_penalty

    print(json.dumps({
        "ok": True,
        "signal": (
            "development sample; corpus-split cross-entropy + benchmark decon only; "
            "not the final held-out validation score"
        ),
        "validation_data_used": False,
        "self_score_settings": {
            "limit": args.limit,
            "max_steps": args.max_steps,
            "max_corpus_chars": args.max_corpus_chars,
            "train_timeout": args.train_timeout,
        },
        "perf_loss": perf_loss,
        "perf": perf,
        "perf_reward": perf_reward,
        "leakage_score": leakage_score,
        "leakage_penalty": leakage_penalty,
        "reward": reward,
        "train_backend": train_backend,
        "budget_fill_ratio": fill,
        "num_contaminated_matches": num_matches,
        "sources": source_stats,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
'''


def _student_train_payload(config: CuratorConfig) -> dict[str, object]:
    ps = config.proxy_student
    return ps.training_payload()


def render_self_score_train_script() -> bytes:
    """Return the workspace-local proxy-student trainer used by ``self_score.py``."""
    body = _nanogpt_train_script()
    replacements = {
        '_stderr_path = "/workspace/stderr.txt"': (
            '_stderr_path = os.path.join(WORKDIR, "stderr.txt")'
        ),
        'open("/workspace/config.json")': 'open(os.path.join(WORKDIR, "config.json"))',
        'open("/workspace/corpus.txt", encoding="utf-8")': (
            'open(os.path.join(WORKDIR, "corpus.txt"), encoding="utf-8")'
        ),
        'val_path = "/workspace/val.bin"': 'val_path = os.path.join(WORKDIR, "val.bin")',
        'pathlib.Path("/workspace/result.json").write_text(json.dumps(result))': (
            'pathlib.Path(os.path.join(WORKDIR, "result.json")).write_text(json.dumps(result))'
        ),
    }
    for old, new in replacements.items():
        body = body.replace(old, new)
    wrapper = (
        "#!/usr/bin/env python3\n"
        '"""Workspace-local proxy-student trainer for self_score.py."""\n'
        "import os\n"
        "import sys\n\n"
        'WORKDIR = sys.argv[1] if len(sys.argv) > 1 else "."\n\n'
    )
    return (wrapper + body.lstrip()).encode()


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

    from .leakage import resolve_decon_binary, resolve_decon_evals_dir

    replacements: dict[str, object] = {
        "__EXPECTED_TOKEN_BUDGET__": config.token_budget,
        "__PERF_BASELINE_LOSS__": repr(config.perf_baseline_loss),
        "__PERF_TARGET_LOSS__": repr(config.perf_target_loss),
        "__PERF_SCALING_EXPONENT__": repr(config.perf_scaling_exponent),
        "__BASELINE_RELATIVE_PERF__": repr(config.baseline_relative_perf),
        "__ALPHA_PERF__": repr(config.alpha_perf),
        "__LAMBDA_LEAKAGE__": repr(config.lambda_leakage),
        "__USE_REAL_TRAINER__": repr(config.use_real_trainer),
        "__TRAIN_SCRIPT_NAME__": repr(SELF_SCORE_TRAIN_FILENAME),
        "__STUDENT_CONFIG__": repr(_student_train_payload(config)),
        "__FORBIDDEN_SOURCE_SHA256__": hashlib.sha256(
            config.validation_set.dataset_id.encode()
        ).hexdigest(),
        "__HF_TOKEN_ENV__": repr(hf_token_env),
        "__DECON_BINARY__": repr(resolve_decon_binary(decon_binary)),
        "__DECON_EVALS_DIR__": repr(resolve_decon_evals_dir(decon_evals_dir)),
        "__DECON_THRESHOLD__": repr(decon_threshold),
    }
    script = dedent(_SCRIPT).lstrip()
    for marker, value in replacements.items():
        script = script.replace(marker, str(value))
    return script.encode()


__all__ = [
    "SELF_SCORE_FILENAME",
    "SELF_SCORE_TRAIN_FILENAME",
    "render_self_score_script",
    "render_self_score_train_script",
]
