from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .collect import RunRecord, discover_runs, load_run_trace, run_to_manifest_entry

SITE_TITLE = "Pretrain Data Curator Bench"
ASSETS_DIR = Path(__file__).resolve().parent / "assets"


def build_site(
    outputs_dir: Path,
    site_dir: Path,
    *,
    full_400m_only: bool = True,
) -> dict[str, Any]:
    site_dir.mkdir(parents=True, exist_ok=True)
    data_dir = site_dir / "data"
    traces_dir = data_dir / "traces"
    if data_dir.exists():
        shutil.rmtree(data_dir)
    traces_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(outputs_dir, full_400m_only=full_400m_only)
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

    for run in runs:
        if not run.has_trace:
            continue
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
        }
        (traces_dir / f"{run.id}.json").write_text(json.dumps(payload, indent=2))

    _copy_assets(site_dir)
    return {
        "runs": len(runs),
        "traces": sum(1 for _ in traces_dir.glob("*.json")),
        "site_dir": str(site_dir),
    }


def _copy_assets(site_dir: Path) -> None:
    for name in ("index.html", "styles.css", "app.js", "renderer.js"):
        shutil.copy2(ASSETS_DIR / name, site_dir / name)
