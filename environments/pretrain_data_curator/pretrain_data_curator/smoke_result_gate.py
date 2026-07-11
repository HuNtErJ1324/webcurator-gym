"""Deterministic gates for A100 smoke launcher downloads."""

from __future__ import annotations

import json
import re
from pathlib import Path

TRACEBACK_MARKER = "Traceback (most recent call last)"


def validate_smoke_results(
    out_dir: Path | str,
    *,
    expected_token_budget: int,
    run_suffix: str = "",
) -> str:
    """Validate a downloaded smoke results directory.

    Requires a nonempty ``results.jsonl`` with at least one ``is_completed``
    record, no hard failure markers, no record-level traceback payloads, and
    (when ``config.toml`` is present) matching token budgets.
    """
    out = Path(out_dir)
    results = out / "results.jsonl"
    if not results.is_file() or results.stat().st_size <= 0:
        raise ValueError(f"missing or empty results.jsonl under {out}")

    hard_markers = [
        p.name
        for p in out.iterdir()
        if p.is_file()
        and (
            p.name.upper() in {"FAILED", "FAILURE", "EVAL_FAILED"}
            or p.name.endswith(".failed")
        )
    ]
    if hard_markers:
        raise ValueError(f"failure marker present: {', '.join(sorted(hard_markers))}")

    records: list[dict] = []
    for lineno, line in enumerate(results.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"results.jsonl line {lineno}: invalid JSON ({exc})") from exc
        if not isinstance(row, dict):
            raise ValueError(f"results.jsonl line {lineno}: expected JSON object")
        records.append(row)

    if not records:
        raise ValueError("results.jsonl has no JSON records")

    completed = [r for r in records if r.get("is_completed")]
    if not completed:
        raise ValueError("results.jsonl has no is_completed=true records")

    for idx, row in enumerate(completed):
        err = row.get("error")
        if err:
            raise ValueError(f"completed record {idx} has error={err!r}")
        errors = row.get("errors")
        if errors:
            blob = errors if isinstance(errors, str) else json.dumps(errors)
            if TRACEBACK_MARKER in blob:
                raise ValueError(f"completed record {idx} errors contain traceback")

    log_path = out / "eval-stream.log"
    if log_path.is_file():
        log_text = log_path.read_text(errors="ignore")
        if TRACEBACK_MARKER in log_text and "results:" not in log_text:
            raise ValueError("eval-stream.log has traceback without results: line")

    config_path = out / "config.toml"
    if config_path.is_file():
        cfg = config_path.read_text()
        m = re.search(r"(?m)^token_budget\s*=\s*([0-9_]+)\s*$", cfg)
        if m:
            got = int(m.group(1).replace("_", ""))
            if got != expected_token_budget:
                raise ValueError(
                    f"config.toml token_budget={got} != expected {expected_token_budget}"
                )
        tm = re.search(
            r"(?ms)^\[args\.proxy_student\].*?^train_token_budget\s*=\s*([0-9_]+)\s*$",
            cfg,
        )
        if tm:
            got_train = int(tm.group(1).replace("_", ""))
            if got_train != expected_token_budget:
                raise ValueError(
                    "config.toml train_token_budget="
                    f"{got_train} != expected {expected_token_budget}"
                )

    return (
        f"valid_records={len(completed)} "
        f"expected_token_budget={expected_token_budget} "
        f"run_suffix={run_suffix}"
    )


def main(argv: list[str] | None = None) -> int:
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) < 2:
        print(
            "usage: smoke_result_gate.py OUT_DIR EXPECTED_TOKEN_BUDGET [RUN_SUFFIX]",
            file=sys.stderr,
        )
        return 2
    out_dir = args[0]
    expected = int(args[1])
    suffix = args[2] if len(args) > 2 else ""
    try:
        print(validate_smoke_results(out_dir, expected_token_budget=expected, run_suffix=suffix))
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
