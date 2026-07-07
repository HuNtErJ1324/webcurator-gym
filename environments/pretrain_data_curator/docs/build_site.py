#!/usr/bin/env python3
"""Generate a PostTrainBench-style site from full 400M eval outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from site_builder.render import build_site

FULL_400M_OUTPUTS = Path(__file__).resolve().parent.parent / "outputs" / "evals-400m"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outputs",
        type=Path,
        default=FULL_400M_OUTPUTS,
        help="Directory containing full 400M eval artifacts (default: outputs/evals-400m)",
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
    )
    print(f"Wrote {summary['runs']} runs ({summary['traces']} traces) to {summary['site_dir']}")
    print(f"Open: file://{args.site_dir.resolve() / 'index.html'}")


if __name__ == "__main__":
    main()
