from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .collect import RunRecord, _load_toml, _metrics_from_row, _reward_from_row, _slug, _timing_seconds
from .traces import extract_trace


def discover_debug_runs(debug_dir: Path) -> list[RunRecord]:
    """Build leaderboard rows from ``outputs/debug/<run-name>/`` snapshots."""
    if not debug_dir.is_dir():
        return []
    runs: list[RunRecord] = []
    for run_dir in sorted(debug_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        payload = build_debug_trace_payload(run_dir)
        if payload is None:
            continue
        model = str(payload.get("model") or "unknown")
        harness = str(payload.get("harness") or "codex")
        timing = payload.get("timing") or {}
        runs.append(
            RunRecord(
                id=str(payload["id"]),
                rel_path=str(run_dir.relative_to(debug_dir.parent)),
                model=model,
                harness=harness,
                token_budget=int(payload.get("token_budget") or 400_000_000),
                use_real_trainer=None,
                reward=payload.get("reward"),
                metrics=dict(payload.get("metrics") or {}),
                timing=_timing_seconds(timing) if isinstance(timing, dict) else {},
                is_completed=bool(payload.get("is_completed")),
                stop_condition=payload.get("stop_condition"),
                example_id=None,
                trace_steps=len(payload.get("trace") or []),
                has_trace=True,
                source="debug",
                run_group=run_dir.name,
                config_summary=dict(payload.get("config") or {}),
            )
        )
    runs.sort(key=lambda r: (r.run_group, r.model), reverse=True)
    return runs


def trace_from_eval_log(
    log_text: str,
    *,
    draft_content: str | None = None,
) -> list[dict[str, Any]]:
    """Reconstruct a readable timeline when no ``results.jsonl`` nodes were saved."""
    steps: list[dict[str, Any]] = []
    turn_num = 0

    for line in log_text.splitlines():
        turn_match = re.search(r"(\d{2}:\d{2}:\d{2}).*intercept stream turn:", line)
        if turn_match:
            turn_num += 1
            steps.append(
                {
                    "role": "system",
                    "content": (
                        f"**Turn {turn_num}** · `{turn_match.group(1)}` — "
                        "Codex turn completed (message content not captured in debug snapshot)."
                    ),
                }
            )
            continue

        handled = False
        for label, pattern in (
            ("Rollout started", r"rollout start:"),
            ("Codex interception online", r"interception up:"),
            ("Agent phase ended", r"interception down:"),
        ):
            if re.search(pattern, line):
                ts = _extract_line_timestamp(line)
                prefix = f"`{ts}` · " if ts else ""
                steps.append({"role": "system", "content": f"{prefix}**{label}**"})
                handled = True
                break
        if handled:
            continue

        if "HF access failed" in line:
            steps.append({"role": "tool", "content": line.strip()})
            continue

        if "source materialized empty" in line:
            match = re.search(r"dataset_id=([^\s]+)", line)
            dataset_id = match.group(1) if match else "unknown"
            steps.append(
                {
                    "role": "tool",
                    "content": f"Empty source: `{dataset_id}`",
                    "tool_calls": [
                        {"name": "corpus_build", "arguments": {"dataset_id": dataset_id}}
                    ],
                }
            )

    if draft_content:
        try:
            draft = json.loads(draft_content)
            source_count = len(draft.get("sources") or [])
        except json.JSONDecodeError:
            source_count = 0
        steps.append(
            {
                "role": "assistant",
                "content": (
                    f"Recovered agent draft manifest with **{source_count}** sources "
                    "(see **Artifacts** tab for `draft9.json`)."
                ),
            }
        )

    return steps


def _extract_line_timestamp(line: str) -> str | None:
    bracket = re.search(r"\[(\d{2}:\d{2}:\d{2})\]", line)
    if bracket:
        return bracket.group(1)
    plain = re.match(r"(\d{2}:\d{2}:\d{2})", line)
    return plain.group(1) if plain else None


def build_debug_trace_payload(run_dir: Path) -> dict[str, Any] | None:
    readme = run_dir / "README.txt"
    log_path = run_dir / "eval-stream.log"
    failed_path = run_dir / "failed_sources.txt"
    results_path = run_dir / "results.jsonl"
    config = _load_toml(run_dir / "config.toml")
    args = config.get("args") or {}

    has_snapshot = (
        log_path.exists()
        or results_path.exists()
        or any(run_dir.glob("draft*.json"))
    )
    if not has_snapshot:
        return None

    model = str(config.get("model") or "")
    if not model:
        if "deepseek" in run_dir.name.lower():
            model = "deepseek/deepseek-v4-pro"
        elif "glm" in run_dir.name.lower():
            model = "z-ai/glm-5.2"
        else:
            model = "unknown"

    log_text = log_path.read_text() if log_path.exists() else ""
    notes = readme.read_text().strip() if readme.exists() else ""

    artifacts: list[dict[str, str]] = []
    if readme.exists():
        artifacts.append(
            {
                "path": "README.txt",
                "label": "Run notes",
                "content": notes,
                "language": "plaintext",
            }
        )
    for draft in sorted(run_dir.glob("draft*.json")):
        artifacts.append(
            {
                "path": draft.name,
                "label": f"Agent draft ({draft.stem})",
                "content": draft.read_text(),
                "language": "json",
            }
        )
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        artifacts.append(
            {
                "path": "manifest.json",
                "label": "Final manifest",
                "content": manifest_path.read_text(),
                "language": "json",
            }
        )
    if failed_path.exists():
        artifacts.append(
            {
                "path": "failed_sources.txt",
                "label": "Failed source IDs",
                "content": failed_path.read_text(),
                "language": "plaintext",
            }
        )

    run_id = _slug(run_dir.name, "debug")
    token_budget = int(args.get("token_budget") or 400_000_000)
    use_real_trainer = args.get("use_real_trainer")
    harness = str(args.get("harness_id") or "codex")

    if results_path.exists() and results_path.stat().st_size > 0:
        row = json.loads(results_path.read_text().splitlines()[0])
        trace = extract_trace(row)
        timing = row.get("timing") or {}
        return {
            "id": run_id,
            "model": model,
            "harness": harness,
            "token_budget": token_budget,
            "reward": _reward_from_row(row),
            "metrics": _metrics_from_row(row),
            "timing": timing if isinstance(timing, dict) else {},
            "is_completed": bool(row.get("is_completed")),
            "stop_condition": row.get("stop_condition") or "curation_heuristic",
            "config": {
                "model": model,
                "harness": harness,
                "token_budget": token_budget,
                "use_real_trainer": use_real_trainer,
                "source": "debug",
            },
            "trace": trace,
            "artifacts": artifacts,
            "log": log_text,
            "notes": notes,
            "rel_path": str(run_dir.name),
            "run_group": run_dir.name,
            "debug": True,
        }

    turns = len(re.findall(r"intercept stream turn:", log_text))
    generation_seconds = _seconds_between_log_markers(
        log_text,
        r"rollout start:",
        r"interception down:",
    )
    draft_content = None
    latest_draft = sorted(run_dir.glob("draft*.json"))
    if latest_draft:
        draft_content = latest_draft[-1].read_text()

    trace = trace_from_eval_log(log_text, draft_content=draft_content)
    return {
        "id": run_id,
        "model": model,
        "harness": harness,
        "token_budget": token_budget,
        "reward": None,
        "metrics": {
            "agent_turns": float(turns),
            "failed_sources": float(
                len([ln for ln in failed_path.read_text().splitlines() if ln.strip()])
            )
            if failed_path.exists()
            else 0.0,
            "finalized": 0.0,
        },
        "timing": {
            "generation": {"end": generation_seconds or 0.0},
            "scoring": {"end": 0.0},
        },
        "is_completed": False,
        "stop_condition": "curation_debug_snapshot",
        "config": {
            "model": model,
            "harness": harness,
            "token_budget": token_budget,
            "source": "debug",
        },
        "trace": trace,
        "trace_kind": "log_reconstruction",
        "trace_note": (
            "Full conversation messages were not saved because the pod was terminated "
            "before results.jsonl was synced. Timeline reconstructed from eval-stream.log."
        ),
        "artifacts": artifacts,
        "log": log_text,
        "notes": notes,
        "rel_path": str(run_dir.name),
        "run_group": run_dir.name,
        "debug": True,
    }


def _seconds_between_log_markers(text: str, start_pat: str, end_pat: str) -> float | None:
    start = re.search(start_pat, text)
    end = re.search(end_pat, text)
    if not start or not end:
        return None
    start_ts = _parse_log_timestamp(text, start.start())
    end_ts = _parse_log_timestamp(text, end.start())
    if start_ts is None or end_ts is None:
        return None
    return max(0.0, end_ts - start_ts)


def _parse_log_timestamp(text: str, pos: int) -> float | None:
    line_start = text.rfind("\n", 0, pos) + 1
    line = text[line_start : text.find("\n", pos)]
    match = re.match(r"(\d{2}):(\d{2}):(\d{2})", line)
    if not match:
        return None
    h, m, s = (int(match.group(i)) for i in range(1, 4))
    return float(h * 3600 + m * 60 + s)
