from __future__ import annotations

import hashlib
import json
import re
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .filters import is_full_400m_eval
from .traces import extract_trace

RUN_DIR_RE = re.compile(
    r"^pretrain-data-curator--(?P<provider>[^-]+(?:-[^-]+)*)--(?P<harness>[^/]+)$"
)


@dataclass
class RunRecord:
    id: str
    rel_path: str
    model: str
    harness: str
    token_budget: int | None
    use_real_trainer: bool | None
    reward: float | None
    metrics: dict[str, float | int | bool | None]
    timing: dict[str, Any]
    is_completed: bool
    stop_condition: str | None
    example_id: int | str | None
    trace_steps: int
    has_trace: bool
    source: str
    run_group: str
    config_summary: dict[str, Any] = field(default_factory=dict)


def _slug(*parts: str) -> str:
    raw = "-".join(p for p in parts if p)
    digest = hashlib.sha1(raw.encode()).hexdigest()[:10]
    clean = re.sub(r"[^a-zA-Z0-9._-]+", "-", raw).strip("-").lower()
    return f"{clean[:48]}-{digest}" if clean else digest


def _parse_dir_name(name: str) -> tuple[str, str]:
    match = RUN_DIR_RE.match(name)
    if match:
        provider = match.group("provider")
        harness = match.group("harness")
        return provider.replace("--", "/"), harness
    if name.startswith("pretrain-data-curator--"):
        tail = name.removeprefix("pretrain-data-curator--")
        if "--" in tail:
            model, harness = tail.rsplit("--", 1)
            return model.replace("--", "/"), harness
    return name, "unknown"


def _load_toml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return tomllib.loads(path.read_text())


def _reward_from_row(row: dict[str, Any]) -> float | None:
    rewards = row.get("rewards")
    if isinstance(rewards, dict) and rewards.get("reward") is not None:
        return float(rewards["reward"])
    if row.get("reward") is not None:
        return float(row["reward"])
    return None


def _metrics_from_row(row: dict[str, Any]) -> dict[str, float | int | bool | None]:
    metrics = dict(row.get("metrics") or {})
    for key in (
        "corpus_tokens",
        "external_failure",
        "finalized",
        "leakage_score",
        "num_sources",
        "perf_accuracy",
        "perf_loss",
        "perf_vs_baseline",
        "budget_fill_ratio",
        "tool_error_count",
        "train_flops",
    ):
        if key in row and key not in metrics:
            metrics[key] = row[key]
    return metrics


def _timing_seconds(timing: dict[str, Any] | None) -> dict[str, float]:
    timing = timing or {}
    out: dict[str, float] = {}
    for phase in ("generation", "scoring", "setup", "finalize"):
        block = timing.get(phase) or {}
        end = block.get("end")
        if isinstance(end, (int, float)):
            out[phase] = float(end)
    if out:
        out["total"] = sum(out.values())
    return out


def _infer_model(meta: dict[str, Any], config: dict[str, Any], parent_name: str) -> str:
    if meta.get("model"):
        return str(meta["model"])
    if config.get("model"):
        return str(config["model"])
    model, _ = _parse_dir_name(parent_name)
    return model


def _infer_harness(config: dict[str, Any], parent_name: str) -> str:
    args = config.get("args") or {}
    if isinstance(args, dict) and args.get("harness_id"):
        return str(args["harness_id"])
    harness = (config.get("harness") or {}).get("id")
    if harness and harness != "default":
        return str(harness)
    _, harness_from_name = _parse_dir_name(parent_name)
    return harness_from_name


def discover_runs(outputs_dir: Path, *, full_400m_only: bool = False) -> list[RunRecord]:
    runs: list[RunRecord] = []
    for results_path in sorted(outputs_dir.rglob("results.jsonl")):
        rel_parent = results_path.parent.relative_to(outputs_dir)
        parent_name = rel_parent.parts[0] if rel_parent.parts else results_path.parent.name
        run_group = str(rel_parent)
        meta_path = results_path.parent / "metadata.json"
        config_path = results_path.parent / "config.toml"
        meta = json_load(meta_path)
        config = _load_toml(config_path)
        args = config.get("args") or {}
        model = _infer_model(meta, config, parent_name)
        harness = _infer_harness(config, parent_name)
        token_budget = args.get("token_budget") if isinstance(args, dict) else None
        use_real_trainer = args.get("use_real_trainer") if isinstance(args, dict) else None
        if full_400m_only and not is_full_400m_eval(
            token_budget=int(token_budget) if token_budget is not None else None,
            use_real_trainer=bool(use_real_trainer) if use_real_trainer is not None else None,
            config=config,
        ):
            continue
        source = "evals-400m" if "evals-400m" in results_path.parts else "outputs"

        for line_no, line in enumerate(results_path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            row = json_loads(line)
            trace = extract_trace(row)
            run_id = _slug(run_group, str(line_no), row.get("id") or "")
            runs.append(
                RunRecord(
                    id=run_id,
                    rel_path=str(results_path.relative_to(outputs_dir)),
                    model=model,
                    harness=harness,
                    token_budget=int(token_budget) if token_budget is not None else None,
                    use_real_trainer=bool(use_real_trainer)
                    if use_real_trainer is not None
                    else None,
                    reward=_reward_from_row(row),
                    metrics=_metrics_from_row(row),
                    timing=_timing_seconds(row.get("timing")),
                    is_completed=bool(row.get("is_completed")),
                    stop_condition=row.get("stop_condition"),
                    example_id=row.get("example_id") or row.get("task"),
                    trace_steps=len(trace),
                    has_trace=bool(trace),
                    source=source,
                    run_group=run_group,
                    config_summary={
                        "model": model,
                        "harness": harness,
                        "token_budget": token_budget,
                        "use_real_trainer": use_real_trainer,
                        "max_turns": args.get("max_turns") if isinstance(args, dict) else config.get("max_turns"),
                    },
                )
            )
    runs.sort(
        key=lambda r: (
            -(r.reward if r.reward is not None else -1e9),
            r.model,
            r.harness,
        )
    )
    return runs


def load_run_trace(outputs_dir: Path, run: RunRecord) -> list[dict[str, Any]]:
    results_path = outputs_dir / run.rel_path
    for line_no, line in enumerate(results_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        row = json_loads(line)
        trace = extract_trace(row)
        candidate = _slug(run.run_group, str(line_no), row.get("id") or "")
        if candidate == run.id:
            return trace
    return []


def json_load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def json_loads(line: str) -> dict[str, Any]:
    return json.loads(line)


def run_to_manifest_entry(run: RunRecord) -> dict[str, Any]:
    data = asdict(run)
    data.pop("config_summary", None)
    data["config"] = run.config_summary
    return data
