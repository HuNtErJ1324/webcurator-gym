"""Benchmark and held-out val-set contamination detection via decon."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import weakref
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import orjson

from ..gpu.scoring_shared import (
    build_decon_detect_command as _build_decon_detect_command,
)
from .hf_access import CHARS_PER_TOKEN

if TYPE_CHECKING:
    from .val_set import HeldOutValSet

logger = logging.getLogger(__name__)

# Avoid div-by-zero.
_MIN_TOKENS = 1

DEFAULT_DECON_BINARY = "decon"


def _env_tree_decon(*parts: str) -> str | None:
    """Locate decon assets in a source checkout (decon/, then scripts/decon)."""
    # utils/ -> package -> environments/pretrain_curation_gym/
    env_root = Path(__file__).resolve().parent.parent.parent
    for base in ("decon", "scripts/decon"):
        candidate = env_root.joinpath(base, *parts)
        if candidate.exists():
            return str(candidate)
    return None


def _installed_decon(*parts: str) -> str | None:
    """Locate decon assets shipped inside an installed wheel."""
    pkg_root = Path(__file__).resolve().parent.parent
    candidate = pkg_root.joinpath("decon", *parts)
    return str(candidate) if candidate.exists() else None


DEFAULT_EVAL_SETS_DIR = (
    _env_tree_decon("bundled-evals")
    or _installed_decon("bundled-evals")
    or "decon/bundled-evals"
)
_LOCAL_DECON_BINARY = _env_tree_decon("bin", "decon") or _installed_decon("bin", "decon")


def resolve_decon_binary(decon_binary: str = DEFAULT_DECON_BINARY) -> str:
    """Resolve decon: explicit path, checkout/wheel binary, or PATH name."""
    if (
        decon_binary
        and decon_binary != DEFAULT_DECON_BINARY
        and os.path.isfile(decon_binary)
    ):
        return os.path.abspath(decon_binary)
    if _LOCAL_DECON_BINARY and os.path.isfile(_LOCAL_DECON_BINARY):
        return os.path.abspath(_LOCAL_DECON_BINARY)
    found = shutil.which(decon_binary) if decon_binary else None
    return found or decon_binary


def resolve_decon_evals_dir(evals_dir: str | None = None) -> str:
    """Resolve benchmark eval sets from config, checkout, or installed wheel."""
    path = evals_dir or DEFAULT_EVAL_SETS_DIR
    return os.path.abspath(path)


_VAL_EVAL_KEY = "heldout_val"
_VAL_EVAL_CHUNK_TOKENS = 1024

# OLMo 3 / decon production defaults (Appendix A.5).
DEFAULT_DECON_TOKENIZER = "cl100k"
DEFAULT_DECON_NGRAM_SIZE = 5
OLMO3_DETECT_ARGS = (
    "--sample-every-m-tokens", "1",
    "--question-max-consecutive-misses", "11",
    "--answer-ngram-size", "3",
    "--passage-ngram-size", "4",
    "--perfect-match-decay-start", "20",
    "--perfect-match-decay-end", "50",
    "--eval-min-token-length", "20",
    "--eval-min-unique-word-count", "4",
)
_DECON_TIMEOUT_SECONDS = 1800


def build_decon_detect_command(
    binary: str,
    *,
    training_dir: str,
    evals_dir: str,
    report_output_dir: str,
    threshold: float,
    content_key: str = "text",
    tokenizer: str = DEFAULT_DECON_TOKENIZER,
    ngram_size: int = DEFAULT_DECON_NGRAM_SIZE,
) -> list[str]:
    """Build the pinned ``decon detect`` argv used by final scoring and self-score."""
    return _build_decon_detect_command(
        binary,
        training_dir=training_dir,
        content_key=content_key,
        evals_dir=evals_dir,
        report_output_dir=report_output_dir,
        tokenizer=tokenizer,
        ngram_size=ngram_size,
        threshold=threshold,
        extra_args=OLMO3_DETECT_ARGS,
    )


class DeconError(RuntimeError):
    """Raised when the decon detector fails (binary missing, timeout, crash)."""


@dataclass
class LeakageScores:
    leakage_score: float
    num_contaminated_matches: int


class DeconLeakageDetector:
    """Runs decon via subprocess on a materialized corpus JSONL directory."""

    def __init__(
        self,
        decon_binary: str = DEFAULT_DECON_BINARY,
        evals_dir: str | None = None,
        threshold: float = 0.8,
        ngram_size: int = DEFAULT_DECON_NGRAM_SIZE,
        tokenizer: str = DEFAULT_DECON_TOKENIZER,
        screen_val_set: bool = True,
    ) -> None:
        self._binary = decon_binary
        self._evals_dir = evals_dir or DEFAULT_EVAL_SETS_DIR
        self._threshold = threshold
        self._ngram_size = ngram_size
        self._tokenizer = tokenizer
        self._screen_val_set = screen_val_set
        self._cache_lock = threading.Lock()
        self._probed_binary: str | None = None
        self._combined_dirs: dict[tuple[str, str, int], str] = {}

    @staticmethod
    def _build_val_eval(
        val_set: HeldOutValSet,
        output_path: str,
        chunk_tokens: int = _VAL_EVAL_CHUNK_TOKENS,
    ) -> None:
        """Detokenize the held-out val tokens and write decon eval JSONL records."""
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
        tokens = val_set.tokens

        with open(output_path, "wb") as fh:
            idx = 0
            for i in range(0, int(tokens.shape[0]), chunk_tokens):
                text = enc.decode(tokens[i : i + chunk_tokens].tolist())
                if not text.strip():
                    continue
                record = {
                    "eval_key": _VAL_EVAL_KEY,
                    "eval_instance_index": idx,
                    "split": "val",
                    "question": text,
                    "answer": "",
                    "doc_id": idx + 1,
                    "fingerprint": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                }
                fh.write(orjson.dumps(record))
                fh.write(b"\n")
                idx += 1

    def _combined_evals_dir(self, val_set: HeldOutValSet) -> str:
        """Bundled benchmarks + val eval file, built once per val identity."""
        key = (val_set.dataset_id, val_set.filename, val_set.n_tokens)
        with self._cache_lock:
            cached = self._combined_dirs.get(key)
            if cached is not None:
                return cached
            parent = tempfile.mkdtemp(prefix="decon_evals_")
            try:
                combined = os.path.join(parent, "combined_evals")
                if os.path.isdir(self._evals_dir):
                    shutil.copytree(self._evals_dir, combined, dirs_exist_ok=True)
                else:
                    os.makedirs(combined, exist_ok=True)
                self._build_val_eval(
                    val_set, os.path.join(combined, "heldout_val.jsonl")
                )
            except BaseException:
                shutil.rmtree(parent, ignore_errors=True)
                raise
            weakref.finalize(self, shutil.rmtree, parent, ignore_errors=True)
            self._combined_dirs[key] = combined
            return combined

    def _check_binary(self) -> str:
        """Resolve and smoke-test the decon binary once; reuse the result."""
        with self._cache_lock:
            if self._probed_binary is not None:
                return self._probed_binary
            binary = self._binary
            if not os.path.isfile(binary):
                resolved = _BUNDLED_DECON_BINARY
                if os.path.isfile(resolved):
                    binary = resolved
            if not os.path.isfile(binary):
                return self._binary
            try:
                probe = subprocess.run(
                    [binary, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
            except OSError as exc:
                raise DeconError(
                    f"decon binary not executable at {binary}: {exc}"
                ) from exc
            if probe.returncode != 0:
                detail = (probe.stderr or probe.stdout or "").strip()
                if "GLIBC_" in detail:
                    raise DeconError(
                        f"decon at {binary} is incompatible with this host's glibc "
                        f"({detail[:300]}). Rebuild with "
                        "environments/pretrain_curation_gym/decon/build_from_source.sh "
                        "on the target machine, or run build_static.sh for a portable binary."
                    )
                raise DeconError(f"decon at {binary} failed smoke test: {detail[:300]}")
            self._probed_binary = binary
            return binary

    def score(
        self,
        docs: Iterable[str],
        val_set: HeldOutValSet | None = None,
    ) -> LeakageScores:
        """Run decon on ``docs`` against bundled benchmarks and optional val set."""
        total_chars = 0
        temp_dir = tempfile.mkdtemp(prefix="decon_corpus_")
        corpus_path = os.path.join(temp_dir, "corpus.jsonl")
        binary = self._binary
        try:
            with open(corpus_path, "wb") as fh:
                for doc in docs:
                    total_chars += len(doc)
                    fh.write(orjson.dumps({"text": doc}))
                    fh.write(b"\n")

            if total_chars == 0:
                return LeakageScores(0.0, 0)

            if self._screen_val_set and val_set is not None:
                evals_dir = self._combined_evals_dir(val_set)
            else:
                evals_dir = self._evals_dir

            total_tokens = max(_MIN_TOKENS, total_chars // CHARS_PER_TOKEN)
            report_dir = os.path.join(temp_dir, "report")
            os.makedirs(report_dir, exist_ok=True)

            binary = self._check_binary()
            cmd = build_decon_detect_command(
                binary,
                training_dir=temp_dir,
                evals_dir=evals_dir,
                report_output_dir=report_dir,
                threshold=self._threshold,
                tokenizer=self._tokenizer,
                ngram_size=self._ngram_size,
            )
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=_DECON_TIMEOUT_SECONDS,
            )
            if result.returncode != 0:
                raise DeconError(
                    f"decon exited with code {result.returncode}: {result.stderr[:500]}"
                )

            report_lines: list[str] = []
            for fname in os.listdir(report_dir):
                if fname.endswith(".jsonl"):
                    path = os.path.join(report_dir, fname)
                    with open(path) as fh:
                        report_lines.extend(fh.readlines())
        except FileNotFoundError:
            raise DeconError(f"decon binary not found at {binary}")
        except subprocess.TimeoutExpired:
            raise DeconError(f"decon timed out after {_DECON_TIMEOUT_SECONDS}s")
        except DeconError:
            raise
        except Exception as exc:
            raise DeconError(f"decon failed: {exc}")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        if not report_lines:
            return LeakageScores(0.0, 0)

        return self._reduce_report(report_lines, total_tokens)

    def _reduce_report(
        self, report_lines: list[str], total_tokens: int
    ) -> LeakageScores:
        """Reduce the decon report JSONL to a token-weighted leakage scalar."""
        best_per_doc: dict[tuple[str, int], float] = {}

        for line in report_lines:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue

            doc_key = (r.get("training_file", ""), r.get("training_line", 0))
            score = float(r.get("contamination_score", 0.0))

            cluster_tok = r.get("cluster_token_length")
            if cluster_tok is not None and int(cluster_tok) > 0:
                est_tokens = int(cluster_tok)
            else:
                ans_start = r.get("answer_start_idx")
                ans_end = r.get("answer_end_idx")
                q_start = r.get("question_start_idx")
                q_end = r.get("question_end_idx")

                start = min(
                    ans_start if ans_start is not None else q_start or 0,
                    q_start if q_start is not None else ans_start or 0,
                )
                end = max(
                    ans_end if ans_end is not None else q_end or 0,
                    q_end if q_end is not None else ans_end or 0,
                )
                span_chars = max(int(end) - int(start), 1)
                est_tokens = max(_MIN_TOKENS, span_chars // CHARS_PER_TOKEN)

            contribution = score * est_tokens
            if doc_key not in best_per_doc or contribution > best_per_doc[doc_key]:
                best_per_doc[doc_key] = contribution

        if not best_per_doc:
            return LeakageScores(0.0, 0)

        total_weighted = sum(best_per_doc.values())
        leakage = total_weighted / total_tokens
        return LeakageScores(
            leakage_score=min(1.0, leakage),
            num_contaminated_matches=len(best_per_doc),
        )
