"""Small helpers for the curator environment (pure-Python, import-safe)."""

from __future__ import annotations

import re
import shlex
from typing import Any

# An `hf ...` invocation inside a shell command string. `hf` must be a standalone
# token (not preceded by a word char, dot, slash, or dash — so a quoted JSON
# `"hf ...` matches, while `path/hf` or `xhf` do not). Stops at the usual shell
# separators / redirections so only this one command's args are captured.
_HF_RE = re.compile(r"(?<![\w./-])hf\s+([^\n;&|`<>]+)")


def content_text(content: Any) -> str:
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


__all__ = [
    "content_text",
    "extract_hf_commands",
]
