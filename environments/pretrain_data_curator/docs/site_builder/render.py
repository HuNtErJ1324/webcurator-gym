from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .collect import RunRecord, discover_runs, load_run_trace, run_to_manifest_entry
from .debug_runs import build_debug_trace_payload, discover_debug_runs

SITE_TITLE = "Pretrain Data Curator Bench"
ASSETS_DIR = Path(__file__).resolve().parent / "assets"


def build_site(
    outputs_dir: Path,
    site_dir: Path,
    *,
    full_400m_only: bool = True,
    debug_dir: Path | None = None,
) -> dict[str, Any]:
    site_dir.mkdir(parents=True, exist_ok=True)
    data_dir = site_dir / "data"
    traces_dir = data_dir / "traces"
    if data_dir.exists():
        shutil.rmtree(data_dir)
    traces_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(outputs_dir, full_400m_only=full_400m_only)
    if debug_dir is not None:
        runs.extend(discover_debug_runs(debug_dir))
    runs.sort(
        key=lambda r: (
            r.source != "debug",
            -(r.reward if r.reward is not None else -1e9),
            r.model,
        )
    )

    manifest_runs = [run_to_manifest_entry(run) for run in runs]
    manifest = {
        "title": SITE_TITLE,
        "generated_at": datetime.now(UTC).isoformat(),
        "run_count": len(manifest_runs),
        "filter": "full_400m" if full_400m_only else "all",
        "runs": manifest_runs,
        "metric_columns": [
            {"key": "reward", "label": "Reward", "higher_is_better": True},
            {"key": "perf_loss", "label": "Perf Loss", "higher_is_better": False},
            {"key": "perf_vs_baseline", "label": "Perf vs Baseline", "higher_is_better": True},
            {"key": "leakage_score", "label": "Leakage", "higher_is_better": False},
            {"key": "budget_fill_ratio", "label": "Budget Fill", "higher_is_better": True},
            {"key": "corpus_tokens", "label": "Corpus Tokens", "higher_is_better": True},
            {"key": "num_sources", "label": "Sources", "higher_is_better": True},
            {"key": "cost_total", "label": "Cost", "higher_is_better": False},
        ],
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    debug_payloads: dict[str, dict[str, Any]] = {}
    if debug_dir is not None and debug_dir.is_dir():
        for run_dir in sorted(debug_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            payload = build_debug_trace_payload(run_dir)
            if payload is not None:
                debug_payloads[str(payload["id"])] = payload

    for run in runs:
        if not run.has_trace:
            continue
        if run.id in debug_payloads:
            payload = debug_payloads[run.id]
        else:
            trace = load_run_trace(outputs_dir, run)
            payload = {
                "id": run.id,
                "model": run.model,
                "harness": run.harness,
                "reward": run.reward,
                "metrics": run.metrics,
                "timing": run.timing,
                "stop_condition": run.stop_condition,
                "config": run.config_summary,
                "trace": trace,
                "artifacts": [],
                "log": "",
                "rel_path": run.rel_path,
                "is_completed": run.is_completed,
            }
        (traces_dir / f"{run.id}.json").write_text(json.dumps(payload, indent=2))

    _copy_assets(site_dir)
    return {
        "runs": len(runs),
        "traces": sum(1 for _ in traces_dir.glob("*.json")),
        "site_dir": str(site_dir),
    }


def _copy_assets(site_dir: Path) -> None:
    for name in ("index.html", "styles.css", "app.js", "utils.js", "renderer.js"):
        shutil.copy2(ASSETS_DIR / name, site_dir / name)
    traces_out = site_dir / "traces"
    traces_out.mkdir(parents=True, exist_ok=True)
    for name in ("run.html", "trace.js"):
        shutil.copy2(ASSETS_DIR / "traces" / name, traces_out / name)
