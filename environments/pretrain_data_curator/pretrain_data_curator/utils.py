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


def truncate_wire_tool_outputs(body: dict[str, Any], cap: int) -> dict[str, Any]:
    """Cap agent-visible tool results on a Chat/Responses interception body.

    Codex posts Responses ``function_call_output`` items; chat harnesses post
    ``role=tool`` messages. Mutates ``body`` in place and returns it. ``cap <= 0``
    is a no-op.
    """
    if cap is None or cap <= 0 or not isinstance(body, dict):
        return body

    raw_input = body.get("input")
    if isinstance(raw_input, list):
        for item in raw_input:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "function_call_output":
                continue
            text = _wire_output_text(item.get("output"))
            capped, _ = truncate_tool_output(text, cap)
            item["output"] = capped

    messages = body.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            if message.get("role") != "tool":
                continue
            text = _wire_output_text(message.get("content"))
            capped, _ = truncate_tool_output(text, cap)
            message["content"] = capped

    return body
