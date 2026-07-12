"""Regression: shipped Codex/Responses eval configs must not set ``top_k``.

The Codex/Responses inference path rejects unknown sampling parameters, so
``sampling.top_k`` must be absent from the production eval configs. Harness and
endpoint configs that legitimately support ``top_k`` are intentionally NOT
scanned here -- the check is scoped to the known Codex/Responses eval TOMLs.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs" / "eval"

# Shipped Codex/Responses eval configs that must serialize without top_k.
CODEX_EVAL_CONFIGS = (
    "400M-300turn-codex.toml",
    "400M-300turn-codex-curation.toml",
    "deepseek-v4-pro-400M-300turn-codex.toml",
    "glm5.2-400M-300turn-codex.toml",
)


def codex_eval_config_paths(configs_dir: Path) -> list[Path]:
    """Return only the shipped Codex/Responses eval TOMLs to be scanned.

    Scoped by explicit name so harness/endpoint configs (which may use top_k)
    are never rejected by this regression.
    """
    return sorted(
        p for p in configs_dir.glob("*.toml") if p.name in CODEX_EVAL_CONFIGS
    )


def test_codex_eval_configs_parse_and_have_no_top_k():
    """The exact production configs parse and serialize without top_k."""
    assert len(CODEX_EVAL_CONFIGS) == 4
    for name in CODEX_EVAL_CONFIGS:
        cfg = tomllib.loads((CONFIGS_DIR / name).read_text(encoding="utf-8"))
        sampling = cfg.get("sampling", {})
        assert "top_k" not in sampling, f"{name} must not set sampling.top_k"
        # temperature/top_p semantics must be preserved (not dropped).
        assert sampling.get("temperature") is not None, f"{name} missing temperature"
        assert sampling.get("top_p") is not None, f"{name} missing top_p"


def test_scan_shipped_codex_eval_tomls_rejects_top_k():
    """Scanner over the shipped Codex/Responses eval TOMLs rejects top_k."""
    offenders = []
    for path in codex_eval_config_paths(CONFIGS_DIR):
        cfg = tomllib.loads(path.read_text(encoding="utf-8"))
        if "top_k" in cfg.get("sampling", {}):
            offenders.append(path.name)
    assert not offenders, f"Codex eval configs still set top_k: {offenders}"


def test_harness_endpoint_configs_may_keep_top_k(tmp_path: Path):
    """top_k is allowed for harness/endpoint configs: the scanner ignores them.

    A harness-style config that legitimately supports top_k must not be flagged
    by the Codex-eval scanner, proving top_k is not globally banned.
    """
    harness = tmp_path / "my-harness-config.toml"
    harness.write_text(
        "[sampling]\ntemperature = 0.7\ntop_p = 0.9\ntop_k = 50\n",
        encoding="utf-8",
    )
    # The harness config is out of scope: the scanner returns no paths here.
    assert codex_eval_config_paths(tmp_path) == []
    # And when parsed directly it is a perfectly valid config (top_k accepted).
    cfg = tomllib.loads(harness.read_text(encoding="utf-8"))
    assert cfg["sampling"]["top_k"] == 50
