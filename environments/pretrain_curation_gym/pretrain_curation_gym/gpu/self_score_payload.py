#!/usr/bin/env python3
"""Development sample scorer: corpus-split CE + benchmark decon only.

This module IS the ``self_score.py`` file written into each rollout's runtime
workspace: ``gpu/self_score.py``'s renderer substitutes the scoring-constant
assignments below with the task's configured values and ships the result
verbatim. It runs standalone on the container's Python with only the standard
library, so nothing here may import the package or third-party code.

The configured final-validation repository is represented only by a SHA-256
digest and rejected before any network request; this file contains no
validation filename, tokens, decoded leakage reference, or the final-scoring
implementation.
"""

import argparse
import atexit
import concurrent.futures
import fcntl
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import NoReturn

# --- scoring constants -------------------------------------------------------
# The defaults keep this module importable/testable; the renderer replaces each
# assignment with the task's configured value (see render_self_score_script).
EXPECTED_TOKEN_BUDGET = 0
PERF_BASELINE_LOSS = 10.825839875788878  # ln(50304)
PERF_TARGET_LOSS = 3.28
PERF_SCALING_EXPONENT = 2.0
BASELINE_RELATIVE_PERF = True
ALPHA_PERF = 1.0
LAMBDA_LEAKAGE = 1.0
USE_REAL_TRAINER = False
TRAIN_SCRIPT_NAME = "self_score_train.py"
STUDENT_CONFIG = {}
FORBIDDEN_SOURCE_SHA256 = ""
HF_TOKEN_ENV = "HF_TOKEN"
DECON_BINARY = "decon"
DECON_EVALS_DIR = ""
DECON_THRESHOLD = 0.8
CHARS_PER_TOKEN = 4
EST_TOKENS_PER_DOC = 250

DATASETS_SERVER = "https://datasets-server.huggingface.co"
TEXT_FIELDS = ("text", "content", "passage", "abstract")
REDACTED_SOURCE_LABEL = "[withheld validation repository]"
HISTORY_FILENAME = ".self_score_history.jsonl"


def append_history(record):
    """Best-effort one-line iteration log next to this script; never breaks scoring."""
    try:
        path = Path(__file__).resolve().parent / HISTORY_FILENAME
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
    except OSError:
        pass


def fail(message) -> NoReturn:
    print(json.dumps({"ok": False, "error": message}, sort_keys=True))
    raise SystemExit(2)


def text_field_of(row, requested):
    """Return the row key `text_from_row` would read, or None when none matches."""
    if requested and requested in row:
        return requested
    return next((k for k in TEXT_FIELDS if k in row), None)


def request_json(path, params):
    url = DATASETS_SERVER + path + "?" + urllib.parse.urlencode(params)
    headers = {"User-Agent": "pretrain-curation-gym-self-score/1"}
    token = os.environ.get(HF_TOKEN_ENV)
    if token:
        headers["Authorization"] = "Bearer " + token
    with urllib.request.urlopen(
        urllib.request.Request(url, headers=headers), timeout=10
    ) as response:
        return json.load(response)


def text_from_row(row, requested):
    if requested and requested in row:
        value = row[requested]
    else:
        value = next((row[k] for k in TEXT_FIELDS if k in row), None)
        if value is None:
            pairs = [
                (row.get("query"), row.get("response")),
                (row.get("prompt"), row.get("completion")),
                (row.get("instruction"), row.get("output")),
            ]
            value = next(
                ("\n".join(str(x) for x in pair if x is not None) for pair in pairs
                 if all(x is not None for x in pair)),
                None,
            )
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return "" if value is None else str(value)


def local_docs(source, limit):
    """Return (docs, meta); meta records what was actually observed on disk."""
    path = Path(str(source.get("local_path") or ""))
    if not path.is_file() or path.is_absolute() or ".." in path.parts:
        raise ValueError("local_path is missing or unsafe")
    raw = path.read_bytes()[:1_048_576].decode("utf-8", "replace")
    fmt = source.get("local_format", "auto")
    meta = {
        "read_kind": "local",
        "bytes_read": len(raw.encode("utf-8", "replace")),
        "records_read": 0,
        "records_parsed": 0,
        "empty_text_records": 0,
        "observed_fields": [],
        "matched_text_field": None,
    }
    if fmt == "jsonl" or (fmt == "auto" and path.suffix.lower() == ".jsonl"):
        meta["read_kind"] = "local_jsonl"
        docs = []
        for line in raw.splitlines():
            if len(docs) >= limit:
                break
            meta["records_read"] += 1
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            meta["records_parsed"] += 1
            if isinstance(value, str):
                docs.append(value)
            elif isinstance(value, dict):
                if not meta["observed_fields"]:
                    meta["observed_fields"] = sorted(value)[:20]
                matched = text_field_of(value, source.get("text_field"))
                if matched and not meta["matched_text_field"]:
                    meta["matched_text_field"] = matched
                docs.append(text_from_row(value, source.get("text_field")))
        meta["empty_text_records"] = sum(1 for x in docs if not x)
        return docs, meta
    meta["read_kind"] = "local_text"
    blocks = [part.strip() for part in re.split(r"\n\s*\n", raw) if part.strip()]
    meta["records_read"] = len(blocks)
    meta["records_parsed"] = len(blocks)
    return blocks[:limit], meta


def source_dataset_id(source):
    return (
        source.get("dataset_id")
        or source.get("id")
        or source.get("dataset")
        or source.get("repo_id")
        or source.get("name")
        or ""
    )


def is_forbidden_source(dataset_id):
    return hashlib.sha256(str(dataset_id).encode()).hexdigest() == (
        FORBIDDEN_SOURCE_SHA256
    )


def remote_docs(source, limit):
    """Return (docs, meta); meta records the rows datasets-server actually served."""
    dataset_id = str(source_dataset_id(source))
    if is_forbidden_source(dataset_id):
        raise ValueError("source is reserved for final validation")
    split = str(source.get("split") or "train")
    config = source.get("config")
    if not config:
        split_rows = request_json("/splits", {"dataset": dataset_id}).get("splits", [])
        match = next((x for x in split_rows if x.get("split") == split), None)
        if match is None and split_rows:
            match = split_rows[0]
            split = str(match.get("split") or split)
        if match is None:
            raise ValueError("datasets-server returned no usable split")
        config = match.get("config")
    payload = request_json(
        "/first-rows",
        {"dataset": dataset_id, "config": config, "split": split},
    )
    rows = [item.get("row") or {} for item in payload.get("rows", [])[:limit]]
    meta = {
        "read_kind": "hf_first_rows",
        "config": config,
        "split": split,
        "records_read": len(rows),
        "records_parsed": len(rows),
        "observed_fields": sorted(rows[0])[:20] if rows else [],
        "matched_text_field": next(
            (
                field
                for field in (text_field_of(row, source.get("text_field")) for row in rows)
                if field
            ),
            None,
        ),
    }
    docs = [text_from_row(row, source.get("text_field")) for row in rows]
    meta["empty_text_records"] = sum(1 for x in docs if not x)
    return docs, meta


def estimate_tokens(text):
    return max(len(text.split()), len(text) // CHARS_PER_TOKEN)


def apply_filters(docs, filters):
    """Apply manifest filters with production-parity semantics.

    Defaults for missing params mirror ``DocumentFilter`` exactly (e.g. a
    ``max_chars`` filter without a value keeps everything), ratio filters use
    the production empty-document conventions, and a filter whose params are
    invalid is skipped — matching the production parser dropping it at
    manifest-parse time. Guarded by the parity tests against the production
    implementations.
    """
    result = list(docs)
    for spec in filters or []:
        if not isinstance(spec, dict):
            continue
        kind = spec.get("kind")
        params = spec.get("params") or {}
        try:
            if kind == "min_chars":
                threshold = int(params.get("value", 0))
                result = [x for x in result if len(x) >= threshold]
            elif kind == "max_chars":
                threshold = int(params.get("value", 10**9))
                result = [x for x in result if len(x) <= threshold]
            elif kind == "min_tokens":
                threshold = int(params.get("value", 0))
                result = [x for x in result if estimate_tokens(x) >= threshold]
            elif kind == "max_symbol_ratio":
                threshold = float(params.get("value", 1.0))
                result = [
                    x for x in result
                    if (
                        sum(not (c.isalnum() or c.isspace()) for c in x) / len(x)
                        if x
                        else 1.0
                    )
                    <= threshold
                ]
            elif kind == "min_alpha_ratio":
                threshold = float(params.get("value", 0.0))
                result = [
                    x for x in result
                    if (sum(c.isalpha() for c in x) / len(x) if x else 0.0)
                    >= threshold
                ]
            elif kind in ("drop_regex", "keep_regex"):
                pattern = re.compile(str(params.get("pattern", "")))
                result = [
                    x for x in result
                    if (not pattern.search(x)) == (kind == "drop_regex")
                ]
            elif kind == "dedup_exact":
                seen = set()
                kept = []
                for x in result:
                    key = x.strip()
                    if key in seen:
                        continue
                    seen.add(key)
                    kept.append(x)
                result = kept
        except (TypeError, ValueError, OverflowError, re.error):
            continue
    return result


def joined_corpus(docs, cap):
    """Serialize a capped document-list-v1 payload for proxy training.

    Cap semantics match production ``CuratedCorpus.joined_text``: whole
    documents only — the first document that would cross the cap stops the
    stream, and documents are never truncated. ``cap=None`` keeps everything.
    """
    documents = []
    remaining = None if cap is None else max(0, int(cap))
    for doc in docs:
        if remaining is not None:
            if len(doc) > remaining:
                break
            remaining -= len(doc)
        documents.append(doc)
    return json.dumps(
        {"format": "document-list-v1", "documents": documents},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def scaled_perf(loss):
    if loss is None or not math.isfinite(loss):
        return 0.0
    if BASELINE_RELATIVE_PERF:
        p = (PERF_BASELINE_LOSS - loss) / (PERF_BASELINE_LOSS - PERF_TARGET_LOSS)
        return p ** PERF_SCALING_EXPONENT if p >= 0 else p
    return max(0.0, min(1.0, math.exp(-loss)))


def _reduce_report(report_lines, total_tokens):
    """Shared reducer: token-weighted contamination from decon report JSONL."""
    best_per_doc = {}
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
            # Literal 4 (== CHARS_PER_TOKEN): the reducer must stay
            # self-contained so tests can extract and exec it standalone.
            est_tokens = max(1, span_chars // 4)

        contribution = score * est_tokens
        if doc_key not in best_per_doc or contribution > best_per_doc[doc_key]:
            best_per_doc[doc_key] = contribution

    if not best_per_doc:
        return 0.0, 0

    total_weighted = sum(best_per_doc.values())
    leakage = min(1.0, total_weighted / total_tokens)
    return leakage, len(best_per_doc)


# --- Progress heartbeats ----------------------------------------------------
# A healthy run is silent for many minutes (corpus sampling, materialization,
# decon, trainer startup). Without progress output an agent cannot tell a slow
# run from a hang and may try to kill it -- which can take down its own harness.
# Heartbeats go to stderr, flushed, at a bounded interval; stdout stays reserved
# for the single machine-readable JSON result.
_START_TIME = time.monotonic()


def _heartbeat_seconds():
    raw = os.environ.get("PDC_SELF_SCORE_HEARTBEAT_SECONDS", "30")
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 30.0
    return min(600.0, max(1.0, value))


def progress(phase, **fields):
    """Emit one flushed stderr heartbeat: `[self-score] phase=... elapsed=...`."""
    parts = [
        "[self-score] phase=%s elapsed=%ds"
        % (phase, int(time.monotonic() - _START_TIME))
    ]
    parts.extend("%s=%s" % (key, fields[key]) for key in sorted(fields))
    sys.stderr.write(" ".join(parts) + "\n")
    sys.stderr.flush()


def _decon_timeout_seconds():
    raw = os.environ.get("PDC_SELF_SCORE_DECON_TIMEOUT", "600")
    try:
        value = int(float(raw))
    except (TypeError, ValueError):
        value = 600
    return max(1, value)


def _communicate_with_heartbeat(proc, timeout, phase, **fields):
    """``proc.communicate(timeout=...)`` with bounded periodic heartbeats.

    Waits in heartbeat-sized slices so a long child (proxy trainer, decon) keeps
    reporting liveness, and still raises ``TimeoutExpired`` exactly at the
    caller's overall timeout.
    """
    interval = _heartbeat_seconds()
    deadline = None if timeout is None else time.monotonic() + float(timeout)
    budget = "none" if timeout is None else "%ds" % int(timeout)
    while True:
        if deadline is None:
            slice_seconds = interval
        else:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(proc.args, float(timeout or 0.0))
            slice_seconds = min(interval, remaining)
        try:
            return proc.communicate(timeout=slice_seconds)
        except subprocess.TimeoutExpired:
            if deadline is not None and time.monotonic() >= deadline:
                raise
            progress(phase, pid=proc.pid, timeout=budget, **fields)


def _read_trainer_stderr_tail(workdir, *, max_chars=8000):
    """Read trainer-redirected stderr before the temp workdir is deleted.

    ``self_score_train.py`` redirects ``sys.stderr`` into ``WORKDIR/stderr.txt``
    (line-buffered). Captured subprocess stderr is therefore often empty on
    crash; surface the file tail so CUDA OOM / traceback lines survive cleanup.
    """
    path = os.path.join(workdir, "stderr.txt")
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            text = fh.read()
    except OSError:
        return ""
    text = text.strip()
    if not text:
        return ""
    return text[-max_chars:]


# Single-flight GPU trainer lock for this evaluation container. Concurrent
# `python self_score.py` invocations serialize here so at most one proxy trainer
# occupies the GPU. The lock fd is passed into the trainer process group so an
# orphaned group continues to hold the flock until it exits; the lock file also
# records the trainer pgid for stale-group recovery after a crash.
_TRAIN_LOCK_PATH = os.environ.get(
    "PDC_SELF_SCORE_LOCK", "/tmp/pdc_self_score_train.lock"
)
_DECON_LOCK_PATH = os.environ.get(
    "PDC_SELF_SCORE_DECON_LOCK", "/tmp/pdc_self_score_decon.lock"
)
_ACTIVE_TRAIN_PROC = None
_ACTIVE_TRAIN_IDENTITY = None
_ACTIVE_DECON_PROC = None
_ACTIVE_DECON_IDENTITY = None
_ACTIVE_DECON_LOCK_FH = None
_DECON_STARTING = False
_ACTIVE_LOCK_FH = None
_SIGNAL_HANDLERS_INSTALLED = False


def _clear_fd_cloexec(fd):
    flags = fcntl.fcntl(fd, fcntl.F_GETFD)
    fcntl.fcntl(fd, fcntl.F_SETFD, flags & ~fcntl.FD_CLOEXEC)


def _proc_stat_fields(pid):
    """Return /proc/<pid>/stat fields after comm (index 0 == state), or None."""
    if pid is None:
        return None
    try:
        with open("/proc/%s/stat" % int(pid), "r", encoding="utf-8") as fh:
            stat = fh.read()
    except (OSError, ValueError):
        return None
    close = stat.rfind(")")
    if close < 0:
        return None
    fields = stat[close + 2 :].split()
    if len(fields) < 20:
        return None
    return fields


def _pgid_starttime(pgid):
    """Return /proc/<pid>/stat starttime, or None if unavailable."""
    fields = _proc_stat_fields(pgid)
    return None if fields is None else fields[19]


def _process_pgid(pid):
    """Return the process group id recorded in /proc/<pid>/stat, or None."""
    fields = _proc_stat_fields(pid)
    if fields is None:
        return None
    try:
        return int(fields[2])
    except ValueError:
        return None


def _pgid_identity(pgid):
    starttime = _pgid_starttime(pgid)
    if starttime is None:
        return None
    return "%s:%s" % (int(pgid), starttime)


def _write_lock_pgid(fh, pgid):
    fh.seek(0)
    fh.truncate()
    if pgid is not None:
        identity = _pgid_identity(pgid)
        if identity is not None:
            fh.write("%s\n" % identity)
        else:
            fh.write("%s\n" % int(pgid))
    fh.flush()
    try:
        os.fsync(fh.fileno())
    except OSError:
        pass


def _read_lock_holder(fh):
    """Return (pgid, starttime_or_None) recorded in the lock file."""
    fh.seek(0)
    text = (fh.read() or "").strip()
    if not text:
        return None, None
    raw = text.splitlines()[0].strip()
    if ":" in raw:
        left, right = raw.split(":", 1)
        try:
            return int(left), right
        except ValueError:
            return None, None
    try:
        return int(raw), None
    except ValueError:
        return None, None


def _pgid_alive(pgid):
    if pgid is None:
        return False
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _group_signal_guard(pgid, expected_starttime=None, allow_missing_leader=False):
    """Return a skip reason when signalling this group would be unsafe, else None.

    ``killpg`` hits every member of a group, so this fails closed. Without a
    recorded starttime there is nothing to prove the pid is still the process we
    recorded (after PID reuse it can name an unrelated leader -- in the eval
    container, plausibly the harness's own session), so a missing identity is
    never signalled. Otherwise /proc must still show the pid as its own group
    leader (pid == pgid) with exactly the recorded starttime.

    ``allow_missing_leader`` covers a group we created and still own whose leader
    has already been reaped. A signal there can only reach surviving members of
    the original group or raise ESRCH: a *reused* pgid requires a live process
    with pid == pgid, and there is none (no /proc entry) -- the kernel also
    refuses ``setsid`` for a pid that still names a live process group.
    """
    if pgid is None:
        return "no_pgid"
    if expected_starttime is None:
        return "no_recorded_identity"
    fields = _proc_stat_fields(pgid)
    if fields is None:
        return None if allow_missing_leader else "leader_gone"
    try:
        leader_pgid = int(fields[2])
    except ValueError:
        return "unreadable_stat"
    if leader_pgid != int(pgid):
        return "not_session_leader"
    if fields[19] != str(expected_starttime):
        return "identity_mismatch"
    return None


def _poll_reap_child(proc, timeout):
    """Reap a direct child if possible; return True when wait completed."""
    if proc is None:
        return False
    if proc.poll() is not None:
        return True
    try:
        proc.wait(timeout=timeout)
        return True
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return proc.poll() is not None


def _terminate_pgid(
    pgid,
    *,
    grace_seconds=5.0,
    child_proc=None,
    expected_starttime=None,
    allow_missing_leader=False,
):
    details = {
        "pgid": pgid,
        "terminated": False,
        "killed": False,
        "reaped": False,
        "error": None,
    }
    if pgid is None:
        details["reaped"] = True
        return details

    def _guard():
        """Re-prove ownership immediately before a signal; never killpg blind.

        Called again before every ``killpg`` (not once up front) so a leader that
        exits and has its PID reused mid-cleanup cannot inherit our signal.
        """
        reason = _group_signal_guard(
            pgid,
            expected_starttime=expected_starttime,
            allow_missing_leader=allow_missing_leader,
        )
        if reason is None:
            return False
        details["skipped"] = True
        details["reason"] = reason
        details["reaped"] = (
            child_proc.poll() is not None
            if child_proc is not None
            else not _pgid_alive(pgid)
        )
        return True

    if _guard():
        return details
    if child_proc is not None and child_proc.poll() is not None and not _pgid_alive(pgid):
        details["reaped"] = True
        return details
    if child_proc is None and not _pgid_alive(pgid):
        details["reaped"] = True
        return details
    if _guard():  # revalidate: the checks above are not free of wall-clock time
        return details
    try:
        os.killpg(pgid, signal.SIGTERM)
        details["terminated"] = True
    except ProcessLookupError:
        if child_proc is not None:
            _poll_reap_child(child_proc, 0.05)
        details["reaped"] = not _pgid_alive(pgid) and (
            child_proc is None or child_proc.poll() is not None
        )
        return details
    except Exception as exc:
        details["error"] = "term:%s" % exc

    def _grace_wait():
        deadline = time.monotonic() + grace_seconds
        while time.monotonic() < deadline:
            child_done = _poll_reap_child(child_proc, 0.05) if child_proc is not None else False
            alive = _pgid_alive(pgid)
            if child_proc is not None:
                if child_done and not alive:
                    details["reaped"] = True
                    return True
                # Leader reaped but other members remain, or leader still running:
                # keep polling until grace expires.
                time.sleep(0.05)
                continue
            if not alive:
                details["reaped"] = True
                return True
            time.sleep(0.05)
        return False

    if _grace_wait():
        return details
    if _guard():
        return details
    try:
        os.killpg(pgid, signal.SIGKILL)
        details["killed"] = True
    except ProcessLookupError:
        pass
    except Exception as exc:
        details["error"] = "kill:%s" % exc
    if _grace_wait():
        return details
    if child_proc is not None:
        _poll_reap_child(child_proc, 0.05)
    details["reaped"] = (child_proc is None or child_proc.poll() is not None) and (
        not _pgid_alive(pgid)
    )
    return details


def _reap_stale_lock_holder(fh):
    """After acquiring the exclusive lock, terminate any recorded stale trainer pgid.

    The recorded pid is only signalled when it is still the live leader of its own
    group with the recorded starttime; on any mismatch (exited, PID reused, no
    longer a group leader) the record is cleared and nothing is signalled.
    """
    stale, expected_start = _read_lock_holder(fh)
    if stale is None:
        return None
    reason = _group_signal_guard(stale, expected_starttime=expected_start)
    if reason is not None:
        _write_lock_pgid(fh, None)
        return {
            "pgid": stale,
            "skipped": True,
            "reason": reason,
            "expected_starttime": expected_start,
            "current_identity": _pgid_identity(stale),
            "terminated": False,
            "killed": False,
            "reaped": True,
            "error": None,
        }
    details = _terminate_pgid(stale, expected_starttime=expected_start)
    _write_lock_pgid(fh, None)
    return details


def _exclusive_process_lock(
    path, *, timeout, timeout_env, default_timeout, label, wait_phase
):
    """Acquire a bounded cross-process flock and reap a recorded stale group."""
    if timeout is None:
        raw = os.environ.get(timeout_env, str(default_timeout))
        try:
            timeout = float(raw)
        except (TypeError, ValueError):
            timeout = float(default_timeout)
    timeout = max(0.05, float(timeout))
    fh = open(path, "a+", encoding="utf-8")
    started = time.monotonic()
    deadline = started + timeout
    next_heartbeat = started + _heartbeat_seconds()
    while True:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            now = time.monotonic()
            if now >= deadline:
                fh.close()
                raise RuntimeError(
                    "another %s at %s (waited %.1ss)"
                    % (label, path, timeout)
                )
            if now >= next_heartbeat:
                progress(
                    wait_phase,
                    lock=path,
                    waited="%ds" % int(now - started),
                )
                next_heartbeat = now + _heartbeat_seconds()
            time.sleep(0.05)
    _reap_stale_lock_holder(fh)
    return fh


def _train_lock(lock_path=None, timeout=None):
    """Exclusive GPU lock so only one self_score trainer runs at a time."""
    return _exclusive_process_lock(
        lock_path or _TRAIN_LOCK_PATH,
        timeout=timeout,
        timeout_env="PDC_SELF_SCORE_LOCK_TIMEOUT",
        default_timeout=120.0,
        label="trainer holds the GPU lock",
        wait_phase="train_lock_wait",
    )


def _decon_lock(lock_path=None, timeout=None):
    """Independent admission lock so at most one Decon process runs at a time."""
    return _exclusive_process_lock(
        lock_path or _DECON_LOCK_PATH,
        timeout=timeout,
        timeout_env="PDC_SELF_SCORE_DECON_LOCK_TIMEOUT",
        default_timeout=float(_decon_timeout_seconds() + 30),
        label="Decon process holds the admission lock",
        wait_phase="decon_lock_wait",
    )


def _release_process_lock(fh):
    if fh is None:
        return
    try:
        try:
            _write_lock_pgid(fh, None)
        except Exception:
            pass
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    finally:
        fh.close()


def _release_train_lock(fh):
    _release_process_lock(fh)


def _release_decon_lock(fh):
    _release_process_lock(fh)


def _terminate_process_group(proc, *, grace_seconds=5.0, expected_starttime=None):
    """SIGTERM then SIGKILL an entire session/process group and reap it.

    The group was created here (``start_new_session=True``), so its leader is our
    own child. While that child is unreaped its PID cannot be reused, so when no
    starttime was recorded we may still read one from /proc and prove the leader
    is the same child, in its own group. With the child already reaped and no
    recorded identity there is nothing left to prove: the guard refuses to signal.
    """
    details = {
        "pid": getattr(proc, "pid", None),
        "pgid": getattr(proc, "pid", None),
        "returncode": proc.poll(),
        "terminated": False,
        "killed": False,
        "reaped": False,
        "error": None,
        "grandchild_pids": [],
    }
    if proc.poll() is not None and not _pgid_alive(proc.pid):
        details["reaped"] = True
        details["returncode"] = proc.returncode
        return details
    if expected_starttime is None:
        if proc.poll() is None:
            # Unreaped child: /proc still describes exactly this process.
            expected_starttime = _pgid_starttime(proc.pid)
        else:
            # A reaped leader with live descendants needs its recorded identity
            # to prove group ownership. Without one, fail closed.
            details["skipped"] = True
            details["reason"] = "no_recorded_identity"
            details["reaped"] = False
            details["returncode"] = proc.returncode
            return details
    pg = _terminate_pgid(
        proc.pid,
        grace_seconds=grace_seconds,
        child_proc=proc,
        expected_starttime=expected_starttime,
        allow_missing_leader=True,
    )
    details.update(
        {
            k: pg[k]
            for k in ("terminated", "killed", "reaped", "error", "skipped", "reason")
            if k in pg
        }
    )
    if proc.poll() is None:
        _poll_reap_child(proc, min(grace_seconds, 1.0))
    details["returncode"] = proc.poll()
    details["reaped"] = details["returncode"] is not None and not _pgid_alive(proc.pid)
    return details


def _cleanup_active_train_proc():
    """Terminate active scorer subprocess groups during cancellation or exit.

    The historical name remains part of the standalone script's tested helper
    surface. It now covers both concurrent scoring branches so neither the GPU
    trainer nor Decon can outlive an interrupted ``self_score.py`` process.
    """
    global _ACTIVE_TRAIN_PROC, _ACTIVE_TRAIN_IDENTITY
    global _ACTIVE_DECON_PROC, _ACTIVE_DECON_IDENTITY
    global _ACTIVE_DECON_LOCK_FH, _DECON_STARTING, _ACTIVE_LOCK_FH
    # Decon launches in a worker thread. If a signal lands after Popen created
    # the child but before that thread publishes it below, briefly let the
    # registration finish so cleanup cannot miss a live process group. This is
    # bounded so a worker wedged inside Popen cannot wedge signal handling.
    registration_deadline = time.monotonic() + 1.0
    while (
        _DECON_STARTING
        and _ACTIVE_DECON_PROC is None
        and time.monotonic() < registration_deadline
    ):
        time.sleep(0.005)
    train_proc = _ACTIVE_TRAIN_PROC
    train_identity = _ACTIVE_TRAIN_IDENTITY
    decon_proc = _ACTIVE_DECON_PROC
    decon_identity = _ACTIVE_DECON_IDENTITY
    decon_lock_fh = _ACTIVE_DECON_LOCK_FH
    lock_fh = _ACTIVE_LOCK_FH
    _ACTIVE_TRAIN_PROC = None
    _ACTIVE_TRAIN_IDENTITY = None
    _ACTIVE_DECON_PROC = None
    _ACTIVE_DECON_IDENTITY = None
    _ACTIVE_DECON_LOCK_FH = None
    _DECON_STARTING = False
    if train_proc is not None and (
        train_proc.poll() is None or _pgid_alive(train_proc.pid)
    ):
        details = _terminate_process_group(
            train_proc, expected_starttime=train_identity
        )
        details["cleanup_reason"] = "signal"
    if decon_proc is not None and (
        decon_proc.poll() is None or _pgid_alive(decon_proc.pid)
    ):
        details = _terminate_process_group(
            decon_proc, expected_starttime=decon_identity
        )
        details["cleanup_reason"] = "signal"
    if lock_fh is not None:
        try:
            _write_lock_pgid(lock_fh, None)
        except Exception:
            pass
    if decon_lock_fh is not None:
        try:
            _write_lock_pgid(decon_lock_fh, None)
        except Exception:
            pass


def _self_score_signal_handler(signum, frame):
    """Clean the active trainer group, restore default disposition, re-raise."""
    _cleanup_active_train_proc()
    signal.signal(signum, signal.SIG_DFL)
    try:
        os.kill(os.getpid(), signum)
    except Exception:
        raise SystemExit(128 + int(signum))


def _install_train_signal_handlers():
    global _SIGNAL_HANDLERS_INSTALLED
    if _SIGNAL_HANDLERS_INSTALLED:
        return
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            signal.signal(sig, _self_score_signal_handler)
        except Exception:
            pass
    _SIGNAL_HANDLERS_INSTALLED = True


atexit.register(_cleanup_active_train_proc)


def _run_in_process_group(argv, *, timeout, lock_fh=None):
    """Run argv in a new session; on timeout/error kill the whole process group."""
    global _ACTIVE_TRAIN_PROC, _ACTIVE_TRAIN_IDENTITY, _ACTIVE_LOCK_FH
    pass_fds = ()
    if lock_fh is not None:
        _clear_fd_cloexec(lock_fh.fileno())
        pass_fds = (lock_fh.fileno(),)
    proc = subprocess.Popen(
        argv,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
        pass_fds=pass_fds,
    )
    starttime = _pgid_starttime(proc.pid)
    _ACTIVE_TRAIN_PROC = proc
    _ACTIVE_TRAIN_IDENTITY = starttime
    _ACTIVE_LOCK_FH = lock_fh
    if lock_fh is not None:
        _write_lock_pgid(lock_fh, proc.pid)
    pg_details = {
        "pid": proc.pid,
        "pgid": proc.pid,
        "returncode": None,
        "timed_out": False,
        "cleanup_reason": None,
        "grandchild_pids": [],
    }
    progress("train_started", pid=proc.pid, timeout="none" if timeout is None else "%ds" % int(timeout))
    try:
        stdout, stderr = _communicate_with_heartbeat(proc, timeout, "train_running")
        pg_details["returncode"] = proc.returncode
        progress("train_exited", pid=proc.pid, returncode=proc.returncode)
        return subprocess.CompletedProcess(argv, proc.returncode, stdout, stderr), pg_details
    except subprocess.TimeoutExpired:
        pg_details["timed_out"] = True
        progress("train_timeout", pid=proc.pid, timeout="%ss" % timeout)
        term = _terminate_process_group(proc, expected_starttime=starttime)
        term["cleanup_reason"] = "timeout"
        pg_details.update(term)
        stdout = stderr = ""
        try:
            out, err = proc.communicate(timeout=1)
            stdout, stderr = out or "", err or ""
        except Exception:
            pass
        exc = subprocess.TimeoutExpired(
            argv, float(timeout or 0.0), output=stdout, stderr=stderr
        )
        exc.process_group = pg_details  # type: ignore[attr-defined]
        raise exc
    except Exception:
        term = _terminate_process_group(proc, expected_starttime=starttime)
        term["cleanup_reason"] = "error"
        pg_details.update(term)
        raise
    finally:
        # Hold the flock until group cleanup is complete, then clear pgid.
        if proc.poll() is None or _pgid_alive(proc.pid):
            term = _terminate_process_group(proc, expected_starttime=starttime)
            if pg_details.get("cleanup_reason") is None:
                term["cleanup_reason"] = (
                    "error" if proc.poll() is None else "orphan_cleanup"
                )
            pg_details.update(term)
        if lock_fh is not None:
            try:
                _write_lock_pgid(lock_fh, None)
            except Exception:
                pass
        if _ACTIVE_TRAIN_PROC is proc:
            _ACTIVE_TRAIN_PROC = None
            _ACTIVE_TRAIN_IDENTITY = None
        if _ACTIVE_LOCK_FH is lock_fh:
            _ACTIVE_LOCK_FH = None


def _snapshot_cgroup_memory():
    """Read cgroup v2 memory.events / memory.max; never raise."""
    events = {}
    memory_max = None
    events_error = None
    max_error = None
    try:
        with open("/sys/fs/cgroup/memory.events", "r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) != 2:
                    continue
                try:
                    events[parts[0]] = int(parts[1])
                except ValueError:
                    continue
    except OSError as exc:
        events_error = str(exc)
    try:
        with open("/sys/fs/cgroup/memory.max", "r", encoding="utf-8") as fh:
            raw = fh.read().strip()
        if raw and raw != "max":
            memory_max = int(raw)
    except (OSError, ValueError) as exc:
        max_error = str(exc)
    return {
        "events": events,
        "memory_max": memory_max,
        "events_error": events_error,
        "max_error": max_error,
    }


def _classify_trainer_kill(
    *,
    returncode=None,
    stderr=None,
    timed_out=False,
    events_before=None,
    events_after=None,
    docker_oom_killed=None,
    process_group=None,
):
    """Deterministic kill classification from cgroup/Docker/stderr/timeout evidence.

    Priority: timeout → cgroup/container OOM → CUDA OOM → external SIGKILL → unknown.
    Local timeout/error/signal cleanup is never labeled external_sigkill.
    """
    pg = process_group or {}
    if timed_out or pg.get("timed_out"):
        return "timeout"

    before = (events_before or {}).get("events") or events_before or {}
    after = (events_after or {}).get("events") or events_after or {}
    oom_delta = 0
    for key in ("oom", "oom_kill", "oom_group"):
        try:
            oom_delta += max(0, int(after.get(key, 0)) - int(before.get(key, 0)))
        except (TypeError, ValueError):
            continue
    if oom_delta > 0:
        return "cgroup_oom"
    if docker_oom_killed:
        # Docker State.OOMKilled is the container cgroup OOM killer.
        return "cgroup_oom"

    text = stderr or ""
    if "CUDA out of memory" in text or (
        re.search(r"cuda\s+out\s+of\s+memory", text, re.I)
        and re.search(r"cuda", text, re.I)
    ):
        return "cuda_oom"
    if re.search(r"cuda\s+out\s+of\s+memory", text, re.I):
        return "cuda_oom"

    if pg.get("cleanup_reason") in ("timeout", "error", "signal"):
        return "unknown"

    sig = None
    if returncode is not None:
        if returncode < 0:
            sig = -returncode
        elif returncode >= 128:
            sig = returncode - 128
    if sig == 9:
        return "external_sigkill"
    return "unknown"


def decon_score(docs):
    """Run decon on sampled documents, return (leakage_score, num_matches) or (None, None)."""
    global _ACTIVE_DECON_PROC, _ACTIVE_DECON_IDENTITY
    global _ACTIVE_DECON_LOCK_FH, _DECON_STARTING
    # ``DECON_BINARY``/``DECON_EVALS_DIR`` are rendered as host absolute paths at
    # setup time, but this script runs inside the agent's ``/workspace`` docker
    # harness runtime where those host paths don't exist. The webcurator-runtime
    # image bakes decon at ``<workspace>/decon`` (``COPY decon/ decon/``), so also
    # probe the script-relative, ``/workspace`` and PATH locations before giving up.
    here = os.path.dirname(os.path.abspath(__file__))
    binary = next(
        (
            p
            for p in (
                DECON_BINARY,
                os.path.join(here, "decon", "bin", "decon"),
                os.path.join(here, "..", "decon", "bin", "decon"),
                "/workspace/decon/bin/decon",
                shutil.which("decon") or "",
            )
            if p and os.path.isfile(p)
        ),
        None,
    )
    if binary is None:
        print("[self-score] WARNING: decon binary not found, skipping leakage check", file=sys.stderr)
        return None, None
    evals_dir = next(
        (
            d
            for d in (
                DECON_EVALS_DIR,
                os.path.join(here, "decon", "bundled-evals"),
                os.path.join(here, "..", "decon", "bundled-evals"),
                "/workspace/decon/bundled-evals",
            )
            if d and os.path.isdir(d)
        ),
        None,
    )
    if evals_dir is None:
        print("[self-score] WARNING: decon evals dir not found, skipping leakage check", file=sys.stderr)
        return None, None
    tmp = tempfile.mkdtemp(prefix="decon_selfscore_")
    proc = None
    starttime = None
    lock_fh = None
    try:
        corpus_path = os.path.join(tmp, "corpus.jsonl")
        with open(corpus_path, "w") as fh:
            for doc in docs:
                if doc:
                    fh.write(json.dumps({"text": doc}, ensure_ascii=False) + "\n")
        total_chars = sum(len(d) for d in docs if d)
        total_tok = max(1, total_chars // CHARS_PER_TOKEN)

        report_dir = os.path.join(tmp, "report")
        os.makedirs(report_dir, exist_ok=True)
        decon_timeout = _decon_timeout_seconds()
        progress("decon_lock_wait", documents=len(docs))
        lock_fh = _decon_lock()
        _ACTIVE_DECON_LOCK_FH = lock_fh
        _clear_fd_cloexec(lock_fh.fileno())
        progress("decon_started", documents=len(docs), timeout="%ds" % decon_timeout)
        _DECON_STARTING = True
        try:
            proc = subprocess.Popen(
                [
                    binary, "detect",
                    "--training-dir", tmp,
                    "--content-key", "text",
                    "--evals-dir", evals_dir,
                    "--report-output-dir", report_dir,
                    "--contamination-score-threshold", str(DECON_THRESHOLD),
                ],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                start_new_session=True,
                pass_fds=(lock_fh.fileno(),),
            )
            # Publish the Popen object before slower identity/fsync work. The
            # signal cleanup path waits through this tiny registration window.
            _ACTIVE_DECON_PROC = proc
            starttime = _pgid_starttime(proc.pid)
            _ACTIVE_DECON_IDENTITY = starttime
            _write_lock_pgid(lock_fh, proc.pid)
        finally:
            _DECON_STARTING = False
        try:
            _stdout, decon_stderr = _communicate_with_heartbeat(
                proc, decon_timeout, "decon_running"
            )
        except subprocess.TimeoutExpired:
            # A slow decon must never cost the run its JSON result: kill and reap
            # the detector, report it, and score with leakage unavailable.
            _terminate_process_group(proc, expected_starttime=starttime)
            progress("decon_timeout", documents=len(docs), timeout="%ds" % decon_timeout)
            print(
                "[self-score] WARNING: decon timed out after %ds on %d sampled "
                "documents; leakage_score is null for this run (reward omits the "
                "leakage term). Re-run with fewer docs (--limit) to get a leakage "
                "reading." % (decon_timeout, len(docs)),
                file=sys.stderr,
            )
            return None, None
        if proc.returncode != 0:
            print("[self-score] WARNING: decon exited %d: %s" % (
                proc.returncode, (decon_stderr or "")[:200],
            ), file=sys.stderr)
            return None, None

        report_lines = []
        for fname in os.listdir(report_dir):
            if not fname.endswith(".jsonl"):
                continue
            with open(os.path.join(report_dir, fname)) as fh:
                report_lines.extend(fh.readlines())

        if not report_lines:
            return 0.0, 0

        return _reduce_report(report_lines, total_tok)
    except Exception as exc:
        print("[self-score] WARNING: decon failed: %s" % exc, file=sys.stderr)
        return None, None
    finally:
        if proc is not None and (proc.poll() is None or _pgid_alive(proc.pid)):
            _terminate_process_group(proc, expected_starttime=starttime)
        if _ACTIVE_DECON_PROC is proc:
            _ACTIVE_DECON_PROC = None
            _ACTIVE_DECON_IDENTITY = None
        if _ACTIVE_DECON_LOCK_FH is lock_fh:
            _ACTIVE_DECON_LOCK_FH = None
        _release_decon_lock(lock_fh)
        shutil.rmtree(tmp, ignore_errors=True)


def train_perf(docs, *, max_corpus_chars, max_steps, train_timeout):
    """Train the fixed proxy student on sampled docs; return (loss, perf, backend)."""
    if not USE_REAL_TRAINER or not docs:
        return None, None, None
    train_script = Path(__file__).with_name(TRAIN_SCRIPT_NAME)
    if not train_script.is_file():
        print(
            "[self-score] WARNING: %s not found, skipping CE training" % TRAIN_SCRIPT_NAME,
            file=sys.stderr,
        )
        return None, None, None
    text = joined_corpus(docs, max_corpus_chars)
    if not text.strip():
        return None, None, None

    tmp = tempfile.mkdtemp(prefix="selfscore_train_")
    lock_fh = None
    pg_details = None
    try:
        _install_train_signal_handlers()
        # Serialize GPU trainers across concurrent self_score invocations.
        progress("train_lock_wait", documents=len(docs), corpus_chars=len(text))
        lock_fh = _train_lock()
        steps = int(max_steps if max_steps is not None else STUDENT_CONFIG["steps"])
        warmup_steps = min(int(STUDENT_CONFIG["warmup_steps"]), steps)
        payload = dict(STUDENT_CONFIG)
        payload["steps"] = steps
        payload["warmup_steps"] = warmup_steps
        payload["n_train_runs"] = 1
        with open(os.path.join(tmp, "config.json"), "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        with open(os.path.join(tmp, "corpus.txt"), "w", encoding="utf-8") as fh:
            fh.write(text)
        progress("train_materialized", corpus_chars=len(text), steps=steps)
        events_before = _snapshot_cgroup_memory()
        result, pg_details = _run_in_process_group(
            [sys.executable, str(train_script), tmp],
            timeout=train_timeout,
            lock_fh=lock_fh,
        )
        events_after = _snapshot_cgroup_memory()
        if result.returncode != 0:
            # Trainer redirects its own stderr into WORKDIR/stderr.txt (line-buffered).
            # Read that tail BEFORE cleanup — captured process stderr is often empty.
            file_stderr = _read_trainer_stderr_tail(tmp)
            detail = file_stderr or result.stderr or result.stdout or ""
            kill_class = _classify_trainer_kill(
                returncode=result.returncode,
                stderr=detail,
                timed_out=False,
                events_before=events_before,
                events_after=events_after,
                process_group=pg_details,
            )
            print(
                "[self-score] WARNING: proxy training exited %d: %s"
                % (result.returncode, detail[:2000]),
                file=sys.stderr,
            )
            print(
                "[self-score] kill_class=%s" % kill_class,
                file=sys.stderr,
            )
            if pg_details is not None:
                print(
                    "[self-score] process_group=%s"
                    % json.dumps(pg_details, sort_keys=True),
                    file=sys.stderr,
                )
            print(
                "[self-score] cgroup_events=%s"
                % json.dumps(
                    {"before": events_before, "after": events_after},
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
            return None, None, None
        marker = "RESULT_JSON "
        line = next(
            (x for x in reversed((result.stdout or "").splitlines()) if x.startswith(marker)),
            None,
        )
        if line is None:
            result_path = os.path.join(tmp, "result.json")
            if not os.path.isfile(result_path):
                file_stderr = _read_trainer_stderr_tail(tmp)
                detail = file_stderr or result.stderr or result.stdout or ""
                print(
                    "[self-score] WARNING: training produced no result JSON: %s"
                    % detail[:2000],
                    file=sys.stderr,
                )
                return None, None, None
            parsed = json.loads(Path(result_path).read_text(encoding="utf-8"))
        else:
            parsed = json.loads(line[len(marker):])
        loss = float(parsed.get("loss", float("inf")))
        backend = str(parsed.get("val_source") or "sample_ce")
        progress("train_scored", loss="%.4f" % loss, backend=backend)
        return loss, scaled_perf(loss), backend
    except subprocess.TimeoutExpired as exc:
        file_stderr = _read_trainer_stderr_tail(tmp)
        detail = file_stderr or getattr(exc, "stderr", None) or getattr(exc, "stdout", None) or ""
        pg = getattr(exc, "process_group", None) or pg_details
        print(
            "[self-score] WARNING: proxy training timed out after %ds: %s"
            % (train_timeout, (detail or "")[:2000]),
            file=sys.stderr,
        )
        print("[self-score] kill_class=timeout", file=sys.stderr)
        if pg is not None:
            print(
                "[self-score] process_group=%s" % json.dumps(pg, sort_keys=True),
                file=sys.stderr,
            )
        return None, None, None
    except Exception as exc:
        file_stderr = _read_trainer_stderr_tail(tmp)
        detail = file_stderr or str(exc)
        print(
            "[self-score] WARNING: proxy training failed: %s" % detail[:2000],
            file=sys.stderr,
        )
        if pg_details is not None:
            print(
                "[self-score] process_group=%s"
                % json.dumps(pg_details, sort_keys=True),
                file=sys.stderr,
            )
        return None, None, None
    finally:
        _release_train_lock(lock_fh)
        shutil.rmtree(tmp, ignore_errors=True)


def score_components(train_call, decon_call):
    """Run proxy training and Decon concurrently, preserving result semantics.

    Results are joined in their historical sequential order (training first,
    Decon second), so exception precedence stays stable. On cancellation or an
    unexpected worker exception, both futures are cancelled when possible and
    every registered subprocess group is terminated before the exception is
    re-raised. ``decon_call=None`` preserves the empty-corpus fast path.
    """
    if decon_call is None:
        # Preserve the historical empty-corpus behavior: train_perf performs its
        # normal no-op checks, while Decon is not launched at all.
        return train_call(), (None, None)

    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=2, thread_name_prefix="self-score"
    )
    train_future = None
    decon_future = None
    try:
        train_future = executor.submit(train_call)
        decon_future = executor.submit(decon_call)
        train_result = train_future.result()
        decon_result = decon_future.result()
        return train_result, decon_result
    except BaseException:
        if train_future is not None:
            train_future.cancel()
        if decon_future is not None:
            decon_future.cancel()
        _cleanup_active_train_proc()
        raise
    finally:
        # Worker functions have bounded subprocess timeouts, and the exception
        # path above terminates their active children before waiting for
        # thread teardown.
        executor.shutdown(wait=True)


def sample_reason(source, meta, sampled_texts, kept_docs):
    """Explain why a source yielded no usable documents, or None when it did.

    Reports what was actually observed (line/row counts, the fields present in
    the rows, whether filters removed everything) so a zero-document source is
    actionable instead of silently scoring zero.
    """
    if kept_docs:
        return None
    kind = source.get("kind", "hf")
    records = int(meta.get("records_read", 0) or 0)
    parsed = int(meta.get("records_parsed", 0) or 0)
    fields = meta.get("observed_fields") or []
    unit = "lines" if meta.get("read_kind") == "local_jsonl" else "rows"
    if records == 0:
        if kind == "local":
            return (
                "read 0 %s from local_path=%r (bytes_read=%s): the file is empty, "
                "or local_path/local_format is wrong"
                % (unit, source.get("local_path"), meta.get("bytes_read"))
            )
        return (
            "datasets-server returned 0 rows for config=%r split=%r: check the "
            "dataset id, config and split" % (meta.get("config"), meta.get("split"))
        )
    if parsed == 0:
        return (
            "read %d %s but parsed 0 records: local_format=%r does not match the "
            "file contents"
            % (records, unit, source.get("local_format", "auto"))
        )
    nonempty = [x for x in sampled_texts if x]
    if not nonempty:
        return (
            "all %d sampled %s produced empty text: text_field=%r matched no field; "
            "observed fields: %s"
            % (parsed, unit, source.get("text_field"), ", ".join(fields) or "<none>")
        )
    kinds = [str((spec or {}).get("kind")) for spec in (source.get("filters") or [])]
    return (
        "filters removed all %d sampled documents (filters: %s): loosen or remove them"
        % (len(nonempty), ", ".join(kinds) or "<none>")
    )


def main():
    _install_train_signal_handlers()
    parser = argparse.ArgumentParser(
        description="Leakage-safe development proxy for a draft curator manifest."
    )
    parser.add_argument("manifest", help="draft manifest JSON file")
    parser.add_argument(
        "--limit",
        type=int,
        default=8,
        metavar="N",
        help="documents sampled per source (agent-chosen; default 8)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        metavar="N",
        help="proxy training steps (default: production student config steps)",
    )
    parser.add_argument(
        "--max-corpus-chars",
        type=int,
        default=None,
        metavar="N",
        help="joined corpus character cap for proxy training (default: all sampled text)",
    )
    parser.add_argument(
        "--train-timeout",
        type=int,
        default=900,
        metavar="SEC",
        help="proxy training wall-clock timeout in seconds (default 900)",
    )
    args = parser.parse_args()
    if args.limit < 1:
        fail("--limit must be >= 1")
    if args.max_steps is not None and args.max_steps < 1:
        fail("--max-steps must be >= 1")
    if args.max_corpus_chars is not None and args.max_corpus_chars < 1:
        fail("--max-corpus-chars must be >= 1")
    if args.train_timeout < 1:
        fail("--train-timeout must be >= 1")
    try:
        manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail("cannot read manifest: " + str(exc))
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        fail("manifest must contain a non-empty sources list")
    try:
        token_budget = int(manifest.get("token_budget", EXPECTED_TOKEN_BUDGET))
    except (TypeError, ValueError):
        fail("token_budget must be an integer")
    if token_budget != EXPECTED_TOKEN_BUDGET:
        fail("token_budget must equal the task allocation")

    raw_cap = manifest.get("sample_docs_per_source")
    cap = None
    if raw_cap is not None:
        cap = max(1, int(raw_cap))
    weights = [max(0.0, float(source.get("weight", 1.0))) for source in sources]
    total_weight = sum(weights)
    if total_weight <= 0:
        fail("at least one source weight must be positive")

    progress("start", manifest=args.manifest, sources=len(sources), limit=args.limit)

    source_stats = []
    estimated_total = 0
    all_docs: list = []
    for index, (source, weight) in enumerate(zip(sources, weights), start=1):
        kind = source.get("kind", "hf")
        dataset_id = str(source_dataset_id(source))
        if kind == "local":
            label = source.get("local_path")
        elif is_forbidden_source(dataset_id):
            label = REDACTED_SOURCE_LABEL
        else:
            label = dataset_id
        progress("sampling", source=label, index="%d/%d" % (index, len(sources)))
        meta = {}
        try:
            sampled, meta = (
                local_docs(source, args.limit)
                if kind == "local"
                else remote_docs(source, args.limit)
            )
            docs = [x for x in apply_filters(sampled, source.get("filters")) if x]
            all_docs.extend(docs)
            sample_tokens = sum(estimate_tokens(x) for x in docs)
            average_tokens = sample_tokens / len(docs) if docs else 0.0
            target = int(token_budget * weight / total_weight)
            if weight > 0:
                requested = max(target // EST_TOKENS_PER_DOC, 1)
                if cap is not None:
                    requested = min(requested, cap)
            else:
                requested = 0
            estimated_tokens = min(target, int(average_tokens * requested))
            error = None
            reason = sample_reason(source, meta, sampled, docs)
        except Exception as exc:
            docs, sample_tokens, estimated_tokens = [], 0, 0
            error = "%s: %s" % (type(exc).__name__, exc)
            reason = error
        estimated_total += estimated_tokens
        source_stats.append({
            "source": label,
            "ok": bool(docs),
            "sampled_documents": len(docs),
            "sampled_tokens": sample_tokens,
            "estimated_materialized_tokens": estimated_tokens,
            "error": error,
            "reason": reason,
            "observed": meta,
        })
        progress(
            "sampled",
            source=label,
            documents=len(docs),
            tokens=sample_tokens,
        )

    failed = [stat for stat in source_stats if not stat["ok"]]
    sampled_documents = len(all_docs)
    sampled_tokens = sum(stat["sampled_tokens"] for stat in source_stats)
    progress(
        "corpus_complete",
        documents=sampled_documents,
        tokens=sampled_tokens,
        failed_sources=len(failed),
    )

    fill = min(1.0, estimated_total / token_budget)

    def run_train_score():
        return train_perf(
            all_docs,
            max_corpus_chars=args.max_corpus_chars,
            max_steps=args.max_steps,
            train_timeout=args.train_timeout,
        )

    def run_decon_score():
        return decon_score(all_docs)

    (perf_loss, perf, train_backend), (leakage_score, num_matches) = score_components(
        run_train_score,
        run_decon_score if all_docs else None,
    )
    progress("scoring", perf_loss=perf_loss, leakage_score=leakage_score)

    ok = not failed and sampled_documents > 0 and sampled_tokens > 0
    if ok:
        perf_reward = ALPHA_PERF * (perf or 0.0)
        leakage_penalty = (
            -LAMBDA_LEAKAGE * leakage_score if leakage_score is not None else 0.0
        )
        reward = perf_reward + leakage_penalty
    else:
        # A candidate with a dead source was never scored as written: part of the
        # mixture contributed nothing. Reporting a number here would be read as a
        # score, and 0.0 would be indistinguishable from a trained candidate that
        # genuinely scored 0.0. Only a fully sampled candidate gets a reward.
        perf_reward = None
        leakage_penalty = None
        reward = None

    error = None
    if failed:
        error = "%d of %d sources sampled zero documents (reward not scored) -- %s" % (
            len(failed),
            len(source_stats),
            "; ".join(
                "%s: %s" % (stat["source"], stat["reason"]) for stat in failed
            ),
        )

    progress("complete", ok=ok, reward=reward, documents=sampled_documents)
    append_history({
        "ts": time.time(),
        "manifest": args.manifest,
        "ok": ok,
        "error": error,
        "reward": reward,
        "perf_reward": perf_reward,
        "leakage_penalty": leakage_penalty,
        "perf_loss": perf_loss,
        "leakage_score": leakage_score,
        "sampled_documents": sampled_documents,
        "sampled_tokens": sampled_tokens,
        "budget_fill_ratio": fill,
        "settings": {
            "limit": args.limit,
            "max_steps": args.max_steps,
            "max_corpus_chars": args.max_corpus_chars,
            "train_timeout": args.train_timeout,
        },
    })
    print(json.dumps({
        "ok": ok,
        "error": error,
        "sampled_documents": sampled_documents,
        "sampled_tokens": sampled_tokens,
        "signal": (
            "development sample; corpus-split cross-entropy + benchmark decon only; "
            "not the final held-out validation score"
        ),
        "validation_data_used": False,
        "self_score_settings": {
            "limit": args.limit,
            "max_steps": args.max_steps,
            "max_corpus_chars": args.max_corpus_chars,
            "train_timeout": args.train_timeout,
        },
        "perf_loss": perf_loss,
        "perf": perf,
        "perf_reward": perf_reward,
        "leakage_score": leakage_score,
        "leakage_penalty": leakage_penalty,
        "reward": reward,
        "train_backend": train_backend,
        "budget_fill_ratio": fill,
        "num_contaminated_matches": num_matches,
        "sources": source_stats,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
