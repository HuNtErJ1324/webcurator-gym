"""Render the leakage-safe development self-scoring script for a rollout.

The rendered script samples candidate training sources named in the agent's
draft manifest. Its implementation lives in ``self_score_payload.py`` — a real,
lint-checked, directly importable module — and rendering substitutes that
module's scoring-constant assignments with the task's configured values, so
the shipped file stays standalone (stdlib-only) with everything baked in.

The configured final-validation repository is represented only by a SHA-256
digest and rejected before any network request; the script contains no
validation filename, tokens, decoded leakage reference, or final-scoring
implementation.

When ``use_real_trainer`` is enabled, setup also writes ``self_score_train.py``,
which runs the same proxy-student training recipe as production scoring (minus
the held-out validation shard). The dev script scores corpus-split cross-entropy
plus benchmark decon leakage — the same two reward terms as final scoring.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from ..utils.models import CuratorConfig
from ..utils.trainer import _nanogpt_train_script

SELF_SCORE_FILENAME = "self_score.py"
SELF_SCORE_TRAIN_FILENAME = "self_score_train.py"
# One JSON line per completed self_score run, appended by the rendered script in
# the runtime workspace; ingested by `CuratorTask.finalize` for iteration metrics.
SELF_SCORE_HISTORY_FILENAME = ".self_score_history.jsonl"
SELF_SCORE_TRAIN_TIMEOUT_SECONDS = 900

_PAYLOAD_PATH = Path(__file__).resolve().with_name("self_score_payload.py")


def _payload_source() -> str:
    """The self-score implementation, read verbatim from its module file."""
    return _PAYLOAD_PATH.read_text(encoding="utf-8")


def __getattr__(name: str) -> str:
    # Historical alias for the template text; the payload module now IS the
    # template, so expose its source under the tested name without keeping a
    # second copy in memory at import time.
    if name == "_SCRIPT":
        return _payload_source()
    raise AttributeError(name)


def _substitute_constants(source: str, values: dict[str, object]) -> str:
    """Replace each module-level ``NAME = <default>`` with the configured value.

    Every constant must match exactly once, so a renamed or removed payload
    constant fails loudly at render time instead of silently shipping a
    default.
    """
    for name, value in values.items():
        pattern = re.compile(rf"^{name} = .*$", re.MULTILINE)
        replacement = f"{name} = {value!r}"
        source, count = pattern.subn(lambda _: replacement, source, count=1)
        if count != 1:
            raise RuntimeError(
                f"self-score payload constant {name!r} not found for substitution"
            )
    return source


def render_self_score_script(
    config: CuratorConfig,
    *,
    hf_token_env: str = "HF_TOKEN",
    decon_binary: str = "decon",
    decon_evals_dir: str | None = None,
    decon_threshold: float = 0.8,
) -> bytes:
    """Return a configured self-score script without exposing held-out data."""
    from ..utils.corpus import EST_TOKENS_PER_DOC
    from ..utils.hf_access import CHARS_PER_TOKEN
    from ..utils.leakage import resolve_decon_binary, resolve_decon_evals_dir

    values: dict[str, object] = {
        "EXPECTED_TOKEN_BUDGET": config.token_budget,
        "PERF_BASELINE_LOSS": config.perf_baseline_loss,
        "PERF_TARGET_LOSS": config.perf_target_loss,
        "PERF_SCALING_EXPONENT": config.perf_scaling_exponent,
        "BASELINE_RELATIVE_PERF": config.baseline_relative_perf,
        "ALPHA_PERF": config.alpha_perf,
        "LAMBDA_LEAKAGE": config.lambda_leakage,
        "USE_REAL_TRAINER": config.use_real_trainer,
        "TRAIN_SCRIPT_NAME": SELF_SCORE_TRAIN_FILENAME,
        "STUDENT_CONFIG": config.proxy_student.training_payload(),
        "FORBIDDEN_SOURCE_SHA256": hashlib.sha256(
            config.validation_set.dataset_id.encode()
        ).hexdigest(),
        "HF_TOKEN_ENV": hf_token_env,
        "DECON_BINARY": resolve_decon_binary(decon_binary),
        "DECON_EVALS_DIR": resolve_decon_evals_dir(decon_evals_dir),
        "DECON_THRESHOLD": decon_threshold,
        "CHARS_PER_TOKEN": CHARS_PER_TOKEN,
        "EST_TOKENS_PER_DOC": EST_TOKENS_PER_DOC,
    }
    return _substitute_constants(_payload_source(), values).encode()


def render_self_score_train_script() -> bytes:
    """Return the workspace-local proxy-student trainer used by ``self_score.py``."""
    body = _nanogpt_train_script()
    body = body.replace('TRAIN_WORKDIR = "/workspace"', "TRAIN_WORKDIR = WORKDIR")
    wrapper = (
        "#!/usr/bin/env python3\n"
        '"""Workspace-local proxy-student trainer for self_score.py."""\n'
        "import os\n"
        "import sys\n\n"
        'WORKDIR = sys.argv[1] if len(sys.argv) > 1 else "."\n\n'
    )
    return (wrapper + body.lstrip()).encode()


__all__ = [
    "SELF_SCORE_FILENAME",
    "SELF_SCORE_HISTORY_FILENAME",
    "SELF_SCORE_TRAIN_FILENAME",
    "SELF_SCORE_TRAIN_TIMEOUT_SECONDS",
    "render_self_score_script",
    "render_self_score_train_script",
]
