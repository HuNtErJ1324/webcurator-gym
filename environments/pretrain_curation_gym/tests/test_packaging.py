"""Distribution invariants for runtime assets that code alone cannot supply."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path


ENV_ROOT = Path(__file__).resolve().parents[1]


def test_wheel_maps_runtime_assets_inside_package() -> None:
    project = tomllib.loads((ENV_ROOT / "pyproject.toml").read_text())
    force_include = project["tool"]["hatch"]["build"]["targets"]["wheel"][
        "force-include"
    ]

    assert force_include == {
        "decon/bin/decon": "pretrain_curation_gym/decon/bin/decon",
        "decon/bundled-evals": "pretrain_curation_gym/decon/bundled-evals",
        "manifests": "pretrain_curation_gym/manifests",
    }


def test_mapped_runtime_assets_are_complete() -> None:
    binary = ENV_ROOT / "decon" / "bin" / "decon"
    evals = ENV_ROOT / "decon" / "bundled-evals"

    assert binary.is_file()
    assert os.access(binary, os.X_OK)
    assert len(list(evals.glob("*.jsonl.gz"))) == 20
    assert (ENV_ROOT / "manifests" / "canonical-400m-fineweb-local.json").is_file()
