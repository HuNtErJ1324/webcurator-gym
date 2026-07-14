#!/usr/bin/env python3
"""Generate a PostTrainBench-style site from full 400M eval outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from site_builder.render import build_site

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_DIR = REPO_ROOT / "environments" / "pretrain_data_curator"

FULL_400M_OUTPUTS = ENV_DIR / "outputs" / "evals-400m"
DEBUG_OUTPUTS = ENV_DIR / "outputs" / "debug"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outputs",
        type=Path,
        default=FULL_400M_OUTPUTS,
        help=(
            "Directory containing full 400M eval artifacts "
            "(default: environments/pretrain_data_curator/outputs/evals-400m)"
        ),
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=DEBUG_OUTPUTS,
        help=(
            "Directory containing curation debug snapshots "
            "(default: environments/pretrain_data_curator/outputs/debug)"
        ),
    )
    parser.add_argument(
        "--no-debug",
        action="store_true",
        help="Exclude the debug snapshots directory from the site",
    )
    parser.add_argument(
        "--site-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "site",
        help="Directory to write the static site",
    )
    parser.add_argument(
        "--include-all",
        action="store_true",
        help="Include non-400M runs (off by default)",
    )
    args = parser.parse_args()
    summary = build_site(
        args.outputs.resolve(),
        args.site_dir.resolve(),
        full_400m_only=not args.include_all,
        debug_dir=None if args.no_debug else args.debug_dir.resolve(),
    )
    print(f"Wrote {summary['runs']} runs ({summary['traces']} traces) to {summary['site_dir']}")
    print(f"Open: file://{args.site_dir.resolve() / 'index.html'}")


if __name__ == "__main__":
    main()
