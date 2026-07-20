"""Render the workspace-local Hugging Face CLI audit wrapper."""

from __future__ import annotations

HF_CLI_AUDIT_FILENAME = ".hf_cli_history.jsonl"
HF_CLI_WRAPPER_FILENAME = ".agents/bin/hf"


def render_hf_cli_wrapper() -> bytes:
    """Return a transparent ``hf`` wrapper that logs redacted invocations."""
    # Escape sequences must stay literal in generated source.
    return br'''
import fcntl
import json
import os
import shutil
import sys
import time
from pathlib import Path

wrapper = Path(__file__).resolve()
workspace = wrapper.parents[2]
history = workspace / ".hf_cli_history.jsonl"
search_path = os.pathsep.join(
    part
    for part in os.environ.get("PATH", "").split(os.pathsep)
    if Path(part).resolve() != wrapper.parent
)
real_hf = shutil.which("hf", path=search_path)
if real_hf is None:
    print("hf audit wrapper: real hf executable not found", file=sys.stderr)
    raise SystemExit(127)

redacted = []
hide_next = False
for arg in sys.argv[1:]:
    if hide_next:
        redacted.append("[REDACTED]")
        hide_next = False
    elif arg in {"--token", "--api-key"}:
        redacted.append(arg)
        hide_next = True
    elif arg.startswith("hf_"):
        redacted.append("[REDACTED]")
    else:
        redacted.append(arg)

try:
    with history.open("a", encoding="utf-8") as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        file.write(json.dumps({"ts": time.time(), "argv": redacted}) + "\n")
        file.flush()
finally:
    os.execv(real_hf, [real_hf, *sys.argv[1:]])
'''


__all__ = [
    "HF_CLI_AUDIT_FILENAME",
    "HF_CLI_WRAPPER_FILENAME",
    "render_hf_cli_wrapper",
]
