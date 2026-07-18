"""Benchmark and held-out val-set contamination detection via decon.

Replaces the previous exact/fuzzy/semantic detectors. Decon runs offline against
*public benchmark eval sets* (bundled under ``decon/bundled-evals/``) AND,
optionally, the held-out validation set (detokenised from GPT-2-BPE token IDs
back to text via tiktoken). The val-derived eval set is built once per
validation-set identity into a detector-owned temp directory that is reused by
every scoring pass and removed when the detector is garbage-collected — it is
NEVER written into ``decon/bundled-evals/``, the workspace, or any container
image.

PRODUCTION path (leakage.py): reads eval sets from the baked
decon/bundled-evals directory combined with the cached val eval file.
DEV path (rendered self_score script) uses the same pinned detect argv as
production, but only against the baked benchmark eval sets — the val set is
NEVER exposed inside the agent container.

The report JSONL schema (observed from decon detect):
  contamination_score: float [0,1] — combined score for this match
  training_file:       str — the training JSONL filename
  training_line:       int — 0-based line in that file
  eval_dataset:        str — name of the matched eval set
  question_start_idx   / question_end_idx   — char offsets of question match
  answer_start_idx     / answer_end_idx     — char offsets of answer match
  cluster_token_length — token length of the n-gram cluster
"""

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

from .hf_access import CHARS_PER_TOKEN

if TYPE_CHECKING:
    from .val_set import HeldOutValSet

logger = logging.getLogger(__name__)

# Minimum tokens to avoid division by zero.
_MIN_TOKENS = 1

DEFAULT_DECON_BINARY = "decon"


def _bundled_asset(*parts: str) -> str:
    """Locate an asset in an installed wheel or the editable source tree."""
    utils_dir = Path(__file__).resolve().parent
    for root in (utils_dir, utils_dir.parent, utils_dir.parent.parent):
        candidate = root.joinpath(*parts)
        if candidate.exists():
            return str(candidate)
    return str(utils_dir.parent.joinpath(*parts))


DEFAULT_EVAL_SETS_DIR = _bundled_asset("decon", "bundled-evals")
_BUNDLED_DECON_BINARY = _bundled_asset("decon", "bin", "decon")


def resolve_decon_binary(decon_binary: str = DEFAULT_DECON_BINARY) -> str:
    """Resolve the vendored decon binary to an absolute path when possible."""
    if (
        decon_binary
        and decon_binary != DEFAULT_DECON_BINARY
        and os.path.isfile(decon_binary)
    ):
        return os.path.abspath(decon_binary)
    if os.path.isfile(_BUNDLED_DECON_BINARY):
        return os.path.abspath(_BUNDLED_DECON_BINARY)
    return decon_binary


def resolve_decon_evals_dir(evals_dir: str | None = None) -> str:
    """Resolve bundled benchmark eval sets to an absolute directory."""
    path = evals_dir or DEFAULT_EVAL_SETS_DIR
    return os.path.abspath(path)


# Decon eval-set constants for the cached val-set eval file.
_VAL_EVAL_KEY = "heldout_val"
_VAL_EVAL_CHUNK_TOKENS = 1024  # GPT-2 BPE tokens per val eval record

# OLMo 3 decontamination parameters (paper Appendix A.5; decon's production
# defaults). Pinned explicitly on the detect command so a vendored-binary
# upgrade can never silently change final-scoring semantics. The one deliberate
# deviation is ``--sample-every-m-tokens 1``: OLMo 3 strides its sampling for
# throughput over trillions of tokens, while our final pass scores one corpus
# of at most a few hundred million tokens, so exhaustive sampling is
# affordable and maximizes recall. Shared with the rendered self-score script.
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
_DECON_TIMEOUT_SECONDS = 1800  # exhaustive stride at 400M-token corpora


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
    return [
        binary,
        "detect",
        "--training-dir",
        training_dir,
        "--content-key",
        content_key,
        "--evals-dir",
        evals_dir,
        "--report-output-dir",
        report_output_dir,
        "--tokenizer",
        tokenizer,
        "--ngram-size",
        str(ngram_size),
        "--contamination-score-threshold",
        str(threshold),
        *OLMO3_DETECT_ARGS,
    ]


class DeconError(RuntimeError):
    """Raised when the decon detector fails (binary missing, timeout, crash).

    The scorer records a diagnostic error metric AND withholds the reward:
    awarding zero leakage here would score an unscreened corpus strictly higher
    than the same corpus with a working detector, which is exploitable in RL.
    See ``CuratorTask.score_manifest`` and ``DeconUnavailableError``.
    """


@dataclass
class LeakageScores:
    leakage_score: float
    num_contaminated_matches: int


class DeconLeakageDetector:
    """Runs decon via subprocess on a materialized corpus JSONL directory.

    When ``screen_val_set`` is ``True`` and a ``HeldOutValSet`` is passed to
    ``score()``, the detector points decon at a combined evals directory
    (bundled benchmarks + detokenized held-out val set). That directory is
    deterministic per validation-set identity, so it is built once, cached on
    the detector, and reused by every scoring pass; it is removed when the
    detector is garbage-collected.
    """

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
        # ``score`` runs on worker threads; the caches below are shared across
        # them, so their population is serialized.
        self._cache_lock = threading.Lock()
        self._probed_binary: str | None = None
        self._combined_dirs: dict[tuple[str, str, int], str] = {}

    @staticmethod
    def _build_val_eval(
        val_set: HeldOutValSet,
        output_path: str,
        chunk_tokens: int = _VAL_EVAL_CHUNK_TOKENS,
    ) -> None:
        """Detokenize the held-out val tokens and write decon eval JSONL records.

        Each record contains one chunk of detokenized text in the ``question``
        field (``answer`` is left empty since this is raw text, not Q&A).
        The ``eval_key`` is ``"heldout_val"`` so decon report lines are
        distinguishable from benchmark matches. Tokens are decoded chunk by
        chunk so the (10M-token) stream is never materialized as one Python
        list.
        """
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
        """Bundled benchmarks + val eval file, built once per val identity.

        Copied (no symlinks, to avoid filesystem-boundary issues) into a
        detector-owned temp directory OUTSIDE any run's ``--training-dir`` so
        decon's directory walk cannot ingest eval files as training input.
        """
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
                # Not cached: a later call may find a newly-installed binary,
                # and the caller surfaces this as a typed DeconError anyway.
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
        """Run decon on ``docs`` against bundled benchmarks and optional val set.

        Writes the document stream to a temporary JSONL file, invokes the decon
        subprocess against the (cached) combined eval sets, parses its report,
        and reduces to a token-weighted scalar.

        Raises:
            DeconError: if the decon binary is missing, returns a non-zero exit
                code, times out, or any other unexpected error occurs. The caller
                is expected to catch this and record an error-state diagnostic,
                NOT silently treat it as a clean corpus.
        """
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
        """Reduce the decon report JSONL to a token-weighted leakage scalar.

        Dedup rule: for each unique ``(training_file, training_line)`` pair we
        keep the match with the highest ``contamination_score * estimated_tokens``
        product, so one training document's tokens are not double-counted across
        multiple eval-match records.

        Token weight is estimated in order of preference:
          1. ``cluster_token_length`` (the decon-reported token length of the
             matched n-gram cluster) when present and > 0.
          2. Character span derived from answer/question end-start offsets
             divided by ``CHARS_PER_TOKEN``.
        """
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

            # Prefer cluster_token_length when available.
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
