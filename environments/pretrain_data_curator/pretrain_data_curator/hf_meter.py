"""Live-discovery cost metering for the `hf`-CLI curation agent.

The agent does its dataset discovery by running the Hugging Face ``hf`` CLI in its
own shell (no MCP tools, no env-provided candidate pool). This module meters that
live discovery cost into the existing :class:`CostLedger`, by two complementary
paths that ``CuratorTaskset.finalize`` chooses between:

1. A **PATH-shadow shim** (`install_shim`). Once per worker we drop a tiny wrapper
   at ``/tmp/vf-hf-shim/bin/hf`` and prepend its dir to ``PATH``. The wrapper execs
   the *real* ``hf`` (resolved by absolute path at install time), passes args /
   stdout / stderr / exit code straight through, and appends one JSONL cost record
   per invocation to ``./.vf_hf_cost.jsonl`` — **relative to the cwd**, which the
   subprocess runtime sets to the per-rollout workspace ``/tmp/<trace.id>``. So each
   rollout's hf calls are isolated to its own log even though the shim is shared.
   ``parse_cost_log`` folds that log into a :class:`CostLedger`.

2. A **trace-reconstruction fallback** (`ledger_from_trace`). When the shim log is
   absent (e.g. a harness whose child shell resets ``PATH``, or a bash /
   mini_swe_agent run), we reconstruct the ``hf`` invocations from the rollout's
   own messages (tool-call command strings and assistant text), and apply the same
   ledger mapping.

Ledger mapping (mirrors the retired MCP ``search_datasets`` accounting at the old
``toolset.py:193-194``, which charged a search as both a web query and a hub call):

* a **search** (``hf datasets ls`` / ``--search`` / ``hf models ls``)
  → ``web_queries += 1`` and ``hub_calls += 1``
* an **info / download / other hub** call → ``hub_calls += 1``
* **bytes pulled** (an hf call's stdout size) → ``tokens += bytes // 4``
  (~4 bytes/token, the ``hf_access.estimate_tokens`` chars/token ratio)

``train_flops`` is **not** touched here — the scorer still adds it after training.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import sys
from collections.abc import Iterable
from typing import Any

from .models import CostLedger

# Worker-shared shim location and the per-rollout cost log (relative to cwd).
SHIM_DIR = "/tmp/vf-hf-shim"
SHIM_BIN_DIR = f"{SHIM_DIR}/bin"
SHIM_HF = f"{SHIM_BIN_DIR}/hf"
COST_LOG_NAME = ".vf_hf_cost.jsonl"

# ~4 bytes/token, matching hf_access.estimate_tokens' chars/token ratio.
_BYTES_PER_TOKEN = 4

_installed = False


# --------------------------------------------------------------------------- #
# shim install (once per worker, idempotent)
# --------------------------------------------------------------------------- #


def _resolve_real_hf() -> str | None:
    """The absolute path of the real ``hf`` on PATH, skipping our own shim dir."""
    shim_dir = os.path.abspath(SHIM_BIN_DIR)
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if not entry or os.path.abspath(entry) == shim_dir:
            continue
        cand = os.path.join(entry, "hf")
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return os.path.abspath(cand)
    return None


def _prepend_path() -> None:
    """Put the shim dir first on ``PATH`` (idempotent)."""
    parts = os.environ.get("PATH", "").split(os.pathsep)
    if parts and parts[0] == SHIM_BIN_DIR:
        return
    os.environ["PATH"] = os.pathsep.join(
        [SHIM_BIN_DIR] + [p for p in parts if p != SHIM_BIN_DIR]
    )


def _render_shim(real_hf: str, python: str) -> str:
    """The shim shell script, with the real hf + a known-good python embedded."""
    real_q = shlex.quote(real_hf)
    py_q = shlex.quote(python)
    log_q = COST_LOG_NAME
    # bash for PIPESTATUS (so hf's exit survives the `| tee` that sizes stdout).
    return f"""#!/usr/bin/env bash
# vf hf cost-metering shim (auto-generated; idempotent). Runs the REAL hf and
# appends one JSONL cost record per call to ./{log_q} (cwd == per-rollout
# workspace). stdout/stderr/exit are preserved; only stdout is tee'd to size it.
set -o pipefail 2>/dev/null || true
__VF_REAL_HF={real_q}
__VF_LOG="${{VF_HF_COST_LOG:-./{log_q}}}"
__VF_PY="${{VF_HF_SHIM_PY:-{py_q}}}"
__VF_T0="$( {{ date +%s.%N ; }} 2>/dev/null || echo 0 )"
__VF_TMP="$( {{ mktemp ; }} 2>/dev/null || echo '' )"
if [ -n "$__VF_TMP" ]; then
  "$__VF_REAL_HF" "$@" | tee "$__VF_TMP"
  __VF_EC=${{PIPESTATUS[0]}}
  __VF_BYTES="$( wc -c < "$__VF_TMP" 2>/dev/null | tr -d ' ' )"
  rm -f "$__VF_TMP"
else
  "$__VF_REAL_HF" "$@"
  __VF_EC=$?
  __VF_BYTES=0
fi
__VF_T1="$( {{ date +%s.%N ; }} 2>/dev/null || echo 0 )"
"$__VF_PY" - "$__VF_EC" "${{__VF_BYTES:-0}}" "$__VF_T0" "$__VF_T1" "$__VF_LOG" "$@" \
  >/dev/null 2>&1 <<'__VF_PYEOF' || true
import json, sys, time
try:
    ec = int(sys.argv[1])
except Exception:
    ec = 0
try:
    nbytes = int(sys.argv[2] or 0)
except Exception:
    nbytes = 0
t0, t1, logp, argv = sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6:]
def _f(x):
    try:
        return float(x)
    except Exception:
        return 0.0
rec = {{"argv": argv, "exit": ec, "bytes": nbytes,
        "duration": max(0.0, _f(t1) - _f(t0)), "ts": time.time()}}
try:
    with open(logp, "a") as fh:
        fh.write(json.dumps(rec) + "\\n")
except Exception:
    pass
__VF_PYEOF
exit $__VF_EC
"""


def install_shim() -> str | None:
    """Install the PATH-shadow ``hf`` shim once per worker and prepend it to PATH.

    Idempotent and best-effort: returns the shim path, or ``None`` if no real ``hf``
    is on PATH (in which case metering falls back to trace reconstruction). Never
    raises — a metering failure must not break taskset construction.
    """
    global _installed
    try:
        if _installed:
            _prepend_path()
            return SHIM_HF
        real = _resolve_real_hf()
        if real is None:
            return None
        os.makedirs(SHIM_BIN_DIR, exist_ok=True)
        script = _render_shim(real, sys.executable or "python3")
        tmp = f"{SHIM_HF}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(script)
        os.chmod(tmp, 0o755)
        os.replace(tmp, SHIM_HF)  # atomic publish; concurrent workers are fine
        _prepend_path()
        _installed = True
        return SHIM_HF
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# classification + ledger mapping
# --------------------------------------------------------------------------- #

# Pure-local hf subcommands that never hit the Hub (not charged).
_LOCAL_COMMANDS = {"version", "env", "help", "auth", "cache", "completion"}


def classify_hf_argv(argv: list[str]) -> str:
    """Classify the args following ``hf`` into a cost class.

    Returns one of ``"search"``, ``"info"``, ``"download"``, ``"other"`` (all hub
    calls), or ``"local"`` (no network, not charged).
    """
    # Leading positional tokens (the command verbs), before any flag.
    leading: list[str] = []
    for tok in argv:
        if tok.startswith("-"):
            break
        leading.append(tok)
    cmd = leading[0] if leading else ""
    sub = leading[1] if len(leading) > 1 else ""
    has_search = "--search" in argv

    if has_search or sub in ("ls", "search") or cmd == "search":
        return "search"
    if sub == "info" or cmd == "info":
        return "info"
    if cmd == "download":
        return "download"
    if cmd in _LOCAL_COMMANDS or cmd == "" or cmd.startswith("-"):
        return "local"
    return "other"


def _apply_argv(ledger: CostLedger, argv: list[str], nbytes: int) -> None:
    """Apply one hf invocation's cost to ``ledger`` in place."""
    kind = classify_hf_argv(argv)
    if kind == "local":
        return
    if kind == "search":
        ledger.web_queries += 1
        ledger.hub_calls += 1
    else:  # info / download / other
        ledger.hub_calls += 1
    if nbytes > 0:
        ledger.tokens += nbytes // _BYTES_PER_TOKEN


def ledger_from_records(records: Iterable[dict[str, Any]]) -> CostLedger:
    """Fold shim JSONL records (each ``{argv, exit, bytes, ...}``) into a ledger."""
    ledger = CostLedger()
    for rec in records:
        if not isinstance(rec, dict):
            continue
        argv = rec.get("argv")
        if not isinstance(argv, list):
            continue
        argv = [str(a) for a in argv]
        try:
            nbytes = int(rec.get("bytes") or 0)
        except (TypeError, ValueError):
            nbytes = 0
        _apply_argv(ledger, argv, nbytes)
    return ledger


def parse_cost_log(text: str) -> CostLedger:
    """Parse the shim's JSONL cost log (tolerant of blank / corrupt lines)."""
    records: list[dict[str, Any]] = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            records.append(obj)
    return ledger_from_records(records)


# --------------------------------------------------------------------------- #
# trace-reconstruction fallback
# --------------------------------------------------------------------------- #

# An `hf ...` invocation inside a shell command string. `hf` must be a standalone
# token (not preceded by a word char, dot, slash, or dash — so a quoted JSON
# `"hf ...` matches, while `path/hf` or `xhf` do not). Stops at the usual shell
# separators / redirections so only this one command's args are captured.
_HF_RE = re.compile(r"(?<![\w./-])hf\s+([^\n;&|`<>]+)")


def _content_text(content: Any) -> str:
    """Flatten a message body (str or list of content parts) to text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for p in content:
        text = getattr(p, "text", None)
        if text is None and isinstance(p, dict):
            text = p.get("text")
        if isinstance(text, str):
            parts.append(text)
    return "\n".join(parts)


def extract_hf_commands(text: str) -> list[list[str]]:
    """Every ``hf <args...>`` invocation found in a shell/command string, as argv."""
    cmds: list[list[str]] = []
    for m in _HF_RE.finditer(text or ""):
        argstr = m.group(1).strip()
        if not argstr:
            continue
        try:
            argv = shlex.split(argstr)
        except ValueError:
            argv = argstr.split()
        if argv:
            cmds.append(argv)
    return cmds


def ledger_from_messages(messages: Iterable[Any]) -> CostLedger:
    """Reconstruct the discovery ledger from a rollout's messages.

    Handles both tool-call harnesses (the ``hf`` command is in an assistant
    tool-call's ``arguments`` JSON; its result's byte size is the paired
    ``ToolMessage`` content) and text-action harnesses (the command is fenced in
    the assistant ``content``; no paired output, so 0 bytes).

    When the bash harness caps tool results, those already-truncated contents are
    what get metered here — no second accounting-only pass is applied.
    """
    messages = list(messages)
    # tool_call_id -> result byte size, for byte attribution on tool-call harnesses.
    out_bytes: dict[str, int] = {}
    for m in messages:
        if getattr(m, "role", None) == "tool":
            tcid = getattr(m, "tool_call_id", None)
            if tcid is not None:
                out_bytes[tcid] = len(_content_text(getattr(m, "content", "")))

    ledger = CostLedger()
    for m in messages:
        if getattr(m, "role", None) != "assistant":
            continue
        # 1) commands in the assistant's own text (e.g. fenced bash actions).
        for argv in extract_hf_commands(_content_text(getattr(m, "content", ""))):
            _apply_argv(ledger, argv, 0)
        # 2) commands carried in tool-call arguments, with paired output bytes.
        for tc in getattr(m, "tool_calls", None) or []:
            args_text = getattr(tc, "arguments", "") or ""
            nbytes = out_bytes.get(getattr(tc, "id", None), 0)
            for i, argv in enumerate(extract_hf_commands(args_text)):
                _apply_argv(ledger, argv, nbytes if i == 0 else 0)
    return ledger


def ledger_from_trace(trace: Any) -> CostLedger:
    """Trace-reconstruction fallback: meter the latest branch's messages."""
    branches = getattr(trace, "branches", None) or []
    messages = branches[-1].messages if branches else []
    return ledger_from_messages(messages)


# --------------------------------------------------------------------------- #
# entry point used by finalize()
# --------------------------------------------------------------------------- #


async def meter_ledger(trace: Any, runtime: Any) -> CostLedger:
    """The discovery cost ledger for a rollout.

    Prefers the shim's runtime cost log (the metering path that worked); falls back
    to reconstructing hf calls from the trace when no log is present.
    """
    if runtime is not None:
        try:
            raw = await runtime.read(COST_LOG_NAME)
        except Exception:
            raw = None
        if raw is not None:
            return parse_cost_log(raw.decode("utf-8", "replace"))
    return ledger_from_trace(trace)


__all__ = [
    "COST_LOG_NAME",
    "SHIM_BIN_DIR",
    "SHIM_DIR",
    "SHIM_HF",
    "classify_hf_argv",
    "extract_hf_commands",
    "install_shim",
    "ledger_from_messages",
    "ledger_from_records",
    "ledger_from_trace",
    "meter_ledger",
    "parse_cost_log",
]
