"""Fail-closed semantic validation for downloaded 400M eval results."""

from __future__ import annotations

import json
import math
import sys
import tomllib
from pathlib import Path

EXPECTED_TOKEN_BUDGET = 400_000_000


def _require_number(
    value: object,
    *,
    name: str,
    row_index: int,
    positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"row {row_index} missing numeric {name} (got {value!r})"
        )
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"row {row_index} {name} must be finite (got {value!r})")
    if positive and number <= 0.0:
        raise ValueError(f"row {row_index} {name} must be > 0 (got {value!r})")
    return number


def _require_metric_flag(
    metrics: dict[str, object],
    name: str,
    expected: float,
    row_index: int,
) -> None:
    actual = _require_number(metrics.get(name), name=name, row_index=row_index)
    if actual != expected:
        raise ValueError(
            f"row {row_index} requires {name}={expected:g} (got {actual:g})"
        )


def _reward(row: dict[str, object]) -> object:
    rewards = row.get("rewards")
    if isinstance(rewards, dict) and "reward" in rewards:
        return rewards.get("reward")
    return row.get("reward")


def validate_400m_results(
    results_path: Path | str,
    *,
    require_production_training: bool,
) -> str:
    """Validate every result row as a semantically complete 400M rollout."""
    path = Path(results_path)
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing or empty results file: {path}")

    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"results line {line_number} is invalid JSON: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise ValueError(f"results line {line_number} is not a JSON object")
        rows.append(row)

    if not rows:
        raise ValueError("results file has no rows")

    for index, row in enumerate(rows):
        if row.get("is_completed") is not True:
            raise ValueError(f"row {index} is not completed")

        stop = row.get("stop_condition")
        if isinstance(stop, str) and stop.strip().lower() in {"error", "truncation"}:
            raise ValueError(f"row {index} stop_condition={stop}")
        for field in ("error", "errors", "failure"):
            if row.get(field):
                raise ValueError(f"row {index} has {field}={row[field]!r}")

        metrics = row.get("metrics")
        if not isinstance(metrics, dict) or not metrics:
            raise ValueError(f"row {index} has empty or missing metrics")

        _require_number(_reward(row), name="reward", row_index=index)

        # A production-valid manifest must have been explicitly written by the
        # model to the workspace and accepted by finalize(). Missing flags fail
        # closed instead of inheriting permissive metric defaults.
        _require_metric_flag(metrics, "finalized", 1.0, index)
        _require_metric_flag(metrics, "manifest_missing", 0.0, index)
        _require_metric_flag(metrics, "manifest_invalid", 0.0, index)
        _require_number(
            metrics.get("corpus_tokens"),
            name="corpus_tokens",
            row_index=index,
            positive=True,
        )
        _require_number(
            metrics.get("num_sources"),
            name="num_sources",
            row_index=index,
            positive=True,
        )

        if require_production_training:
            _require_number(
                metrics.get("train_flops"),
                name="train_flops",
                row_index=index,
                positive=True,
            )
            _require_number(
                metrics.get("perf_loss"),
                name="perf_loss",
                row_index=index,
                positive=True,
            )
            _require_metric_flag(metrics, "trainer_error_msg", 0.0, index)

    mode = "production" if require_production_training else "curation-only"
    return f"valid_rows={len(rows)} mode={mode}"


def production_mode_from_config(config_path: Path | str) -> bool:
    """Read the resolved eval config and verify its 400M contract."""
    path = Path(config_path)
    if not path.is_file():
        raise ValueError(f"missing resolved config: {path}")
    try:
        config = tomllib.loads(path.read_text())
    except (tomllib.TOMLDecodeError, OSError) as exc:
        raise ValueError(f"cannot read resolved config {path}: {exc}") from exc

    args = config.get("args")
    if not isinstance(args, dict):
        raise ValueError("resolved config missing [args]")
    if args.get("token_budget") != EXPECTED_TOKEN_BUDGET:
        raise ValueError(
            "resolved config token_budget must be "
            f"{EXPECTED_TOKEN_BUDGET} (got {args.get('token_budget')!r})"
        )
    use_real_trainer = args.get("use_real_trainer")
    if not isinstance(use_real_trainer, bool):
        raise ValueError("resolved config missing boolean use_real_trainer")

    proxy = args.get("proxy_student")
    if not isinstance(proxy, dict):
        raise ValueError("resolved config missing [args.proxy_student]")
    if proxy.get("train_token_budget") != EXPECTED_TOKEN_BUDGET:
        raise ValueError(
            "resolved config train_token_budget must be "
            f"{EXPECTED_TOKEN_BUDGET} (got {proxy.get('train_token_budget')!r})"
        )
    return use_real_trainer


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        print("usage: result_gate.py RESULTS_JSONL RESOLVED_CONFIG", file=sys.stderr)
        return 2
    try:
        production = production_mode_from_config(args[1])
        print(
            validate_400m_results(
                args[0], require_production_training=production
            )
        )
    except ValueError as exc:
        print(f"semantic result validation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
