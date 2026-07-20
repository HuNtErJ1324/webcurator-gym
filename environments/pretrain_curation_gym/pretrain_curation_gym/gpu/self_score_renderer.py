"""Render the leakage-safe development self-scoring script for a rollout."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from ..utils.models import CuratorConfig
from ..utils.trainer import _nanogpt_train_script

SELF_SCORE_FILENAME = "self_score.py"
SELF_SCORE_TRAIN_FILENAME = "self_score_train.py"
SELF_SCORE_HISTORY_FILENAME = ".self_score_history.jsonl"
SELF_SCORE_TRAIN_TIMEOUT_SECONDS = 900

_RUNTIME_PATH = Path(__file__).resolve().with_name("self_score_runtime.py")
_SHARED_PATH = Path(__file__).resolve().with_name("scoring_shared.py")
_SHARED_IMPORT = """from .scoring_shared import (
    CHARS_PER_TOKEN,
    apply_filters,
    build_decon_detect_command,
    estimate_tokens,
    weighted_token_target,
)
"""
_RUNTIME_SOURCE = _RUNTIME_PATH.read_text(encoding="utf-8")
_SHARED_SOURCE = _SHARED_PATH.read_text(encoding="utf-8")


def _runtime_source() -> str:
    """Build standalone source from the cached runtime and shared helpers."""
    source, count = re.subn(
        re.escape(_SHARED_IMPORT),
        lambda _: _SHARED_SOURCE,
        _RUNTIME_SOURCE,
        count=1,
    )
    if count != 1:
        raise RuntimeError("self-score shared-helper import not found")
    return source


def __getattr__(name: str) -> str:
    if name == "_SCRIPT":
        return _runtime_source()
    raise AttributeError(name)


def _substitute_constants(source: str, values: dict[str, object]) -> str:
    """Replace each module-level ``NAME = <default>`` with the configured value."""
    for name, value in values.items():
        pattern = re.compile(rf"^{name} = .*$", re.MULTILINE)
        replacement = f"{name} = {value!r}"
        source, count = pattern.subn(lambda _: replacement, source, count=1)
        if count != 1:
            raise RuntimeError(
                f"self-score runtime constant {name!r} not found for substitution"
            )
    return source


def render_self_score_script(
    config: CuratorConfig,
    *,
    hf_token_env: str = "HF_TOKEN",
    runtime_decon_binary: str = "decon/bin/decon",
    runtime_decon_evals_dir: str = "decon/bundled-evals",
    decon_threshold: float = 0.8,
) -> bytes:
    """Return a configured self-score script without exposing held-out data."""
    from ..utils.corpus import EST_TOKENS_PER_DOC
    from ..utils.hf_access import CHARS_PER_TOKEN
    from ..utils.leakage import (
        DEFAULT_DECON_NGRAM_SIZE,
        DEFAULT_DECON_TOKENIZER,
        OLMO3_DETECT_ARGS,
    )

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
        "DECON_BINARY": runtime_decon_binary,
        "DECON_EVALS_DIR": runtime_decon_evals_dir,
        "DECON_THRESHOLD": decon_threshold,
        "DECON_TOKENIZER": DEFAULT_DECON_TOKENIZER,
        "DECON_NGRAM_SIZE": DEFAULT_DECON_NGRAM_SIZE,
        "DECON_DETECT_EXTRA_ARGS": OLMO3_DETECT_ARGS,
        "CHARS_PER_TOKEN": CHARS_PER_TOKEN,
        "EST_TOKENS_PER_DOC": EST_TOKENS_PER_DOC,
    }
    return _substitute_constants(_runtime_source(), values).encode()


def render_self_score_train_script() -> bytes:
    """Return the workspace-local proxy-student trainer used by ``self_score.py``."""
    body = _nanogpt_train_script()
    # Train against argv[1] workspace, not baked /workspace.
    body, substitutions = re.subn(
        r"^TRAIN_WORKDIR = .*$",
        lambda _: "TRAIN_WORKDIR = WORKDIR",
        body,
        count=1,
        flags=re.MULTILINE,
    )
    if substitutions != 1:
        raise RuntimeError(
            "train_gpt constant 'TRAIN_WORKDIR' not found for substitution"
        )
    wrapper = (
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
