from __future__ import annotations

import json
from typing import Any


def _text_from_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text") or block.get("content")
                if text:
                    parts.append(str(text))
            elif block:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def _tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    calls = message.get("tool_calls") or []
    out: list[dict[str, Any]] = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        fn = call.get("function") or {}
        args = fn.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                pass
        out.append(
            {
                "id": call.get("id"),
                "name": fn.get("name") or call.get("name"),
                "arguments": args,
            }
        )
    return out


def trace_from_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for node in nodes:
        message = node.get("message")
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if not role:
            continue
        content = _text_from_content(message.get("content"))
        reasoning = message.get("reasoning_content")
        tools = _tool_calls(message)
        if not content and not reasoning and not tools and role not in {"tool"}:
            continue
        step = {
            "role": role,
            "content": content,
            "tool_calls": tools,
        }
        if reasoning:
            step["reasoning"] = str(reasoning)
        if role == "tool":
            step["tool_call_id"] = message.get("tool_call_id")
        steps.append(step)
    return steps


def trace_from_legacy(row: dict[str, Any]) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for label, key in (("prompt", "prompt"), ("completion", "completion")):
        messages = row.get(key)
        if not isinstance(messages, list):
            continue
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = message.get("role") or label
            content = _text_from_content(message.get("content"))
            tools = _tool_calls(message)
            if not content and not tools:
                continue
            steps.append(
                {
                    "role": role,
                    "content": content,
                    "tool_calls": tools,
                }
            )
    return steps


def extract_trace(row: dict[str, Any]) -> list[dict[str, Any]]:
    nodes = row.get("nodes")
    if isinstance(nodes, list) and nodes:
        return trace_from_nodes(nodes)
    return trace_from_legacy(row)
