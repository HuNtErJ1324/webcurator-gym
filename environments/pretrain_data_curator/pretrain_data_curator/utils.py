"""Small helpers for curator environment (pure-Python, import-safe)."""

from __future__ import annotations

from typing import Any


def truncate_tool_output(text: str | None, cap: int) -> tuple[str, bool]:
    """Cap tool/bash output at ``cap`` characters (strict, notice included).

    - ``cap <= 0`` disables truncation (returns the original text, ``False``).
    - When truncated, appends a short notice; the entire returned string
      (body + notice) is always ``<= cap``.
    """
    if text is None:
        s = ""
    else:
        try:
            s = str(text)
        except Exception:
            s = ""
    if cap is None or cap <= 0:
        return s, False
    if len(s) <= cap:
        return s, False
    notice = f"\n\n[TRUNCATED: original length={len(s)} chars]"
    if len(notice) >= cap:
        return notice[:cap], True
    return s[: cap - len(notice)] + notice, True


def _wire_output_text(output: Any) -> str:
    """Normalize Responses/Chat tool-result payloads to plain text."""
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        parts: list[str] = []
        for part in output:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                if part.get("type") in {"input_text", "output_text", "text"}:
                    parts.append(str(part.get("text") or ""))
                elif "text" in part:
                    parts.append(str(part.get("text") or ""))
                else:
                    parts.append(str(part))
            else:
                parts.append(str(part))
        return "".join(parts)
    return str(output)
