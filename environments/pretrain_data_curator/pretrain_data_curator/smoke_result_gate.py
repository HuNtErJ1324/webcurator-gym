"""Deterministic gates for A100 smoke launcher downloads."""

from __future__ import annotations

import json
import re
from pathlib import Path

TRACEBACK_MARKER = "Traceback (most recent call last)"


def _config_use_real_trainer(out: Path) -> bool:
    """Return True when config.toml requests the real GPU trainer."""
    config_path = out / "config.toml"
    if not config_path.is_file():
        return False
    cfg = config_path.read_text(errors="ignore")
    m = re.search(r"(?m)^use_real_trainer\s*=\s*(true|false)\s*$", cfg)
    if not m:
        return False
    return m.group(1) == "true"


def validate_smoke_results(
    out_dir: Path | str,
    *,
    expected_token_budget: int,
    run_suffix: str = "",
) -> str:
    """Validate a downloaded smoke results directory.

    ``is_completed=true`` alone is insufficient: reject ``stop_condition=error``,
    any error/failure payload, empty metrics, hard failure markers, record-level
    traceback payloads, and (when ``config.toml`` is present) mismatched token
    budgets. At least one completed record must carry real metrics (including
    numeric ``corpus_tokens``).
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
        stop = row.get("stop_condition")
        if isinstance(stop, str) and stop.strip().lower() == "error":
            raise ValueError(f"completed record {idx} has stop_condition=error")

        err = row.get("error")
        if err:
            raise ValueError(f"completed record {idx} has error={err!r}")

        errors = row.get("errors")
        if errors:
            if isinstance(errors, list) and len(errors) == 0:
                pass
            else:
                blob = errors if isinstance(errors, str) else json.dumps(errors)
                raise ValueError(f"completed record {idx} has error/failure payload: {blob}")

        failure = row.get("failure")
        if failure:
            raise ValueError(f"completed record {idx} has failure={failure!r}")

        metrics = row.get("metrics")
        if not isinstance(metrics, dict) or not metrics:
            raise ValueError(f"completed record {idx} missing nonempty metrics")
        corpus_tokens = metrics.get("corpus_tokens")
        if not isinstance(corpus_tokens, (int, float)):
            raise ValueError(
                f"completed record {idx} metrics missing numeric corpus_tokens"
            )

        # Real-trainer smokes must actually train — is_completed + corpus fill is
        # not enough when use_real_trainer=true (trainer_error_msg=1 / flops=0).
        if _config_use_real_trainer(out):
            trainer_err = metrics.get("trainer_error_msg", 0.0)
            if isinstance(trainer_err, (int, float)) and float(trainer_err) != 0.0:
                raise ValueError(
                    f"completed record {idx} trainer_error_msg={trainer_err} "
                    "(real trainer failed)"
                )
            train_flops = metrics.get("train_flops")
            if not isinstance(train_flops, (int, float)) or float(train_flops) <= 0.0:
                raise ValueError(
                    f"completed record {idx} missing positive train_flops "
                    f"(got {train_flops!r})"
                )
            perf_loss = metrics.get("perf_loss")
            if not isinstance(perf_loss, (int, float)) or not (float(perf_loss) > 0.0):
                raise ValueError(
                    f"completed record {idx} missing positive perf_loss "
                    f"(got {perf_loss!r})"
                )

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
