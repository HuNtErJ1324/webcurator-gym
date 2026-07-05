"""Benchmark + held-out val-set contamination detection via decon.

Replaces the previous exact/fuzzy/semantic detectors. Decon runs offline against
*public benchmark eval sets* (bundled under ``decon/bundled-evals/``) AND,
optionally, the held-out validation set (detokenised from GPT-2-BPE token IDs
back to text via tiktoken). The val-derived eval set is built ephemerally at
scoring time into a temp directory and cleaned up immediately afterwards — it
is NEVER written into ``decon/bundled-evals/``, the workspace, or any
container image.

PRODUCTION path (leakage.py): reads eval sets from the baked
decon/bundled-evals directory combined with an ephemeral val eval file.
DEV path (self_score.py) runs decon against only the baked benchmark eval
sets — the val set is NEVER exposed inside the agent container.

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
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .val_set import HeldOutValSet

logger = logging.getLogger(__name__)

# Character-to-token divisor used to estimate token counts from character spans
# when decon does not report a resolved token count for the full span.
_CHARS_PER_TOKEN = 4
# Minimum tokens to avoid division by zero.
_MIN_TOKENS = 1

DEFAULT_DECON_BINARY = "decon"
DEFAULT_EVAL_SETS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "decon", "bundled-evals"
)

# Decon eval-set constants for the ephemeral val-set eval file.
_VAL_EVAL_KEY = "heldout_val"
_VAL_EVAL_CHUNK_TOKENS = 1024  # GPT-2 BPE tokens per val eval record


class DeconError(RuntimeError):
    """Raised when the decon detector fails (binary missing, timeout, crash).

    The scorer catches this and records a diagnostic error metric rather than
    silently awarding zero leakage (which would be exploitable in RL).
    """


@dataclass
class LeakageScores:
    leakage_score: float
    num_contaminated_matches: int
    contamination_details: tuple[dict[str, object], ...]

    def as_dict(self) -> dict[str, float | int]:
        return {
            "leakage_score": round(self.leakage_score, 6),
            "num_contaminated_matches": self.num_contaminated_matches,
        }

    def overall(self) -> float:
        return self.leakage_score


class DeconLeakageDetector:
    """Runs decon via subprocess on a materialized corpus JSONL directory.

    When ``screen_val_set`` is ``True`` and a ``HeldOutValSet`` is passed to
    ``score()``, the detector builds an ephemeral combined evals directory
    (bundled benchmarks + held-out val set) inside the temp workspace and
    points decon at it.  The val eval file is cleaned up with the temp dir
    after scoring.
    """

    def __init__(
        self,
        decon_binary: str = DEFAULT_DECON_BINARY,
        evals_dir: str | None = None,
        threshold: float = 0.2,
        ngram_size: int = 5,
        tokenizer: str = "cl100k",
        screen_val_set: bool = True,
    ) -> None:
        self._binary = decon_binary
        self._evals_dir = evals_dir or DEFAULT_EVAL_SETS_DIR
        self._threshold = threshold
        self._ngram_size = ngram_size
        self._tokenizer = tokenizer
        self._screen_val_set = screen_val_set

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
        distinguishable from benchmark matches.
        """
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
        tokens: list[int] = val_set.tokens.tolist()

        with open(output_path, "w", encoding="utf-8") as fh:
            idx = 0
            for i in range(0, len(tokens), chunk_tokens):
                chunk = tokens[i : i + chunk_tokens]
                text = enc.decode(chunk)
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
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                idx += 1

    def _build_combined_evals_dir(self, parent_dir: str) -> str:
        """Copy bundled evals into a temp subdir (no symlinks to avoid
        filesystem-boundary issues) and return the path."""
        combined = os.path.join(parent_dir, "combined_evals")
        if os.path.isdir(self._evals_dir):
            shutil.copytree(self._evals_dir, combined, dirs_exist_ok=True)
        else:
            os.makedirs(combined, exist_ok=True)
        return combined

    def _check_binary(self) -> str:
        binary = self._binary
        if not os.path.isfile(binary):
            resolved = os.path.join(
                os.path.dirname(__file__), "..", "decon", "bin", "decon"
            )
            if os.path.isfile(resolved):
                return resolved
        return binary

    def score(
        self,
        docs: Iterable[str],
        val_set: HeldOutValSet | None = None,
    ) -> LeakageScores:
        """Run decon on ``docs`` against bundled benchmarks and optional val set.

        Writes the document stream to a temporary JSONL file, optionally
        detokenizes the held-out val tokens into an ephemeral decon eval file,
        invokes the decon subprocess against the combined eval sets, parses its
        report, and reduces to a token-weighted scalar.

        The val eval file lives inside the temp workspace and is cleaned up
        when this method returns.  It is NEVER written to
        ``decon/bundled-evals/`` or any persistent path.

        Raises:
            DeconError: if the decon binary is missing, returns a non-zero exit
                code, times out, or any other unexpected error occurs. The caller
                is expected to catch this and record an error-state diagnostic,
                NOT silently treat it as a clean corpus.
        """
        total_chars = 0
        temp_dir = tempfile.mkdtemp(prefix="decon_corpus_")
        corpus_path = os.path.join(temp_dir, "corpus.jsonl")
        try:
            with open(corpus_path, "w") as fh:
                for doc in docs:
                    total_chars += len(doc)
                    fh.write(json.dumps({"text": doc}, ensure_ascii=False) + "\n")

            if total_chars == 0:
                return LeakageScores(0.0, 0, ())

            # --- determine evals dir: benchmarks only or combined with val set ----
            if self._screen_val_set and val_set is not None:
                evals_dir = self._build_combined_evals_dir(temp_dir)
                self._build_val_eval(val_set, os.path.join(evals_dir, "heldout_val.jsonl"))
            else:
                evals_dir = self._evals_dir

            total_tokens = max(_MIN_TOKENS, total_chars // _CHARS_PER_TOKEN)
            report_dir = os.path.join(temp_dir, "report")
            os.makedirs(report_dir, exist_ok=True)

            binary = self._check_binary()
            cmd = [
                binary,
                "detect",
                "--training-dir",
                temp_dir,
                "--content-key",
                "text",
                "--evals-dir",
                evals_dir,
                "--report-output-dir",
                report_dir,
                "--tokenizer",
                self._tokenizer,
                "--ngram-size",
                str(self._ngram_size),
                "--contamination-score-threshold",
                str(self._threshold),
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
            )
            if result.returncode != 0:
                raise DeconError(
                    f"decon exited with code {result.returncode}: "
                    f"{result.stderr[:500]}"
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
            raise DeconError("decon timed out after 600s")
        except DeconError:
            raise
        except Exception as exc:
            raise DeconError(f"decon failed: {exc}")
        finally:
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

        if not report_lines:
            return LeakageScores(0.0, 0, ())

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
             divided by ``_CHARS_PER_TOKEN``.
        """
        best_per_doc: dict[tuple[str, int], float] = {}
        details: list[dict[str, object]] = []

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
                est_tokens = max(_MIN_TOKENS, span_chars // _CHARS_PER_TOKEN)

            contribution = score * est_tokens
            if (
                doc_key not in best_per_doc
                or contribution > best_per_doc[doc_key]
            ):
                best_per_doc[doc_key] = contribution

            details.append(
                {
                    "eval_dataset": r.get("eval_dataset", ""),
                    "contamination_score": score,
                    "span_tokens": est_tokens,
                }
            )

        if not best_per_doc:
            return LeakageScores(0.0, 0, ())

        total_weighted = sum(best_per_doc.values())
        leakage = total_weighted / total_tokens
        return LeakageScores(
            leakage_score=min(1.0, leakage),
            num_contaminated_matches=len(best_per_doc),
            contamination_details=tuple(details),
        )
