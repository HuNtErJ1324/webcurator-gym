"""Vendored decon benchmark eval set reference directory.

Decon ships its bundled benchmark eval sets under ``decon/bundled-evals/``.
These are PUBLIC BENCHMARK eval sets only — NOT our held-out validation set:

  - AGI Eval (10 shards, train split)
  - GSM8K (test-1, train-1)
  - MMLU (dev-1, test-1 through test-6, validation-1)

The held-out validation set (``val_set.py``) is used exclusively for the
proxy-student cross-entropy (Perf) signal and NEVER as a decon reference.
"""

from __future__ import annotations

import os as _os

DECON_EVALS_DIR = _os.path.join(
    _os.path.dirname(__file__), "..", "decon", "bundled-evals"
)
