"""Bash harness variant that caps agent-visible tool output.

The stock Verifiers ``bash`` harness returns unbounded stdout/stderr from its
``bash`` tool. Large Hub dumps (e.g. ``hf datasets info ...``) can overflow the
model context and kill the rollout. This module stages a patched uv program that
reads ``MAX_TOOL_OUTPUT_CHARS`` from the harness env (set by
``load_environment``) and truncates every tool result before it is appended to
the chat.

Imports of the builtin ``DefaultHarness`` are deferred so constructing this
module never fails when the harness package is only partially installed.
when a stub/partial Verifiers install is on ``sys.path``.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .utils import truncate_tool_output

# Injected into the stock bash program after its imports. Kept in sync with
# ``truncate_tool_output`` (strict: body + notice <= cap).
_TRUNCATE_INJECT = """
import os

_MAX_TOOL_OUTPUT_CHARS = int(os.environ.get("MAX_TOOL_OUTPUT_CHARS", "0") or "0")


def _maybe_truncate(text):
    if text is None:
        text = ""
    elif not isinstance(text, str):
        text = str(text)
    if _MAX_TOOL_OUTPUT_CHARS <= 0 or len(text) <= _MAX_TOOL_OUTPUT_CHARS:
        return text
    notice = f"\\n\\n[TRUNCATED: original length={len(text)} chars]"
    if len(notice) >= _MAX_TOOL_OUTPUT_CHARS:
        return notice[:_MAX_TOOL_OUTPUT_CHARS]
    return text[: _MAX_TOOL_OUTPUT_CHARS - len(notice)] + notice

"""

_RUN_BASH_RE = re.compile(
    r"def run_bash\(command: str\) -> str:\n"
    r"    try:\n"
    r"        result = subprocess\.run\(\n"
    r"            \[\"bash\", \"-c\", command\], capture_output=True, text=True, timeout=3600\n"
    r"        \)\n"
    r"        return result\.stdout \+ result\.stderr\n"
    r"    except Exception as e:\n"
    r"        return f\"error: \{e\}\"\n"
)

_RUN_BASH_REPLACEMENT = """def run_bash(command: str) -> str:
    try:
        result = subprocess.run(
            ["bash", "-c", command], capture_output=True, text=True, timeout=3600
        )
        return _maybe_truncate(result.stdout + result.stderr)
    except Exception as e:
        return _maybe_truncate(f"error: {e}")
"""

_TOOL_APPEND_OLD = """                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": content}
                )"""

_TOOL_APPEND_NEW = """                if isinstance(content, str):
                    content = _maybe_truncate(content)
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": content}
                )"""

_TruncatingBashHarness: type[Any] | None = None
_PROGRAM_SOURCE: str | None = None


def build_truncating_program_source(stock: str) -> str:
    """Return stock bash ``program.py`` with tool-result truncation wired in."""
    if "from openai import AsyncOpenAI\n" not in stock:
        raise RuntimeError(
            "stock bash program missing expected openai import; cannot patch truncation"
        )
    patched = stock.replace(
        "from openai import AsyncOpenAI\n",
        "from openai import AsyncOpenAI\n" + _TRUNCATE_INJECT,
        1,
    )
    patched, n = _RUN_BASH_RE.subn(_RUN_BASH_REPLACEMENT, patched, count=1)
    if n != 1:
        raise RuntimeError(
            "stock bash program run_bash body changed; cannot patch truncation"
        )
    if _TOOL_APPEND_OLD not in patched:
        raise RuntimeError(
            "stock bash program tool-append site changed; cannot patch truncation"
        )
    return patched.replace(_TOOL_APPEND_OLD, _TOOL_APPEND_NEW, 1)


def truncating_program_source() -> str:
    """Lazy stock-program patch (cached)."""
    global _PROGRAM_SOURCE
    if _PROGRAM_SOURCE is None:
        from verifiers.v1.harnesses.default.harness import PROGRAM_SOURCE as stock

        _PROGRAM_SOURCE = build_truncating_program_source(stock)
    return _PROGRAM_SOURCE


def get_truncating_bash_harness_class() -> type[Any]:
    """Return the default harness with the patched bash program."""
    global _TruncatingBashHarness
    if _TruncatingBashHarness is not None:
        return _TruncatingBashHarness

    from verifiers.v1.clients import ModelContext
    from verifiers.v1.dialects.chat import message_to_wire
    from verifiers.v1.harnesses.default.harness import (
        BASH_SYSTEM_PROMPT,
        DefaultHarness,
    )
    from verifiers.v1.runtimes import ProgramResult, Runtime
    from verifiers.v1.trace import Trace

    program_source = truncating_program_source()

    class TruncatingBashHarness(DefaultHarness):
        """Default harness that caps tool results in its builtin bash program."""

        async def setup(self, runtime: Runtime) -> None:
            await runtime.prepare_uv_script(program_source, self.config.env)

        async def launch(
            self,
            ctx: ModelContext,
            trace: Trace,
            runtime: Runtime,
            endpoint: str,
            secret: str,
            mcp_urls: dict[str, str],
        ) -> ProgramResult:
            task = trace.task.data
            system_prompt, prompt = self.resolve_prompt(task)
            fragments = [BASH_SYSTEM_PROMPT]
            if getattr(self.config, "edit", True):
                fragments.append(
                    "You also have an edit tool for single-occurrence string replacement in a file."
                )
            system_prompt = "\n\n".join(
                p for p in ("\n\n".join(fragments), system_prompt) if p
            )
            env = {**self.config.resolved_env}
            args = [
                f"--base-url={endpoint}",
                f"--api-key={secret}",
                f"--model={ctx.model}",
                f"--system-prompt={system_prompt}",
            ]
            if getattr(self.config, "edit", True):
                args.append("--edit")
            if mcp_urls:
                args.append(
                    "--mcp-config="
                    + json.dumps(
                        {
                            "mcpServers": {
                                name: {"url": url} for name, url in mcp_urls.items()
                            }
                        }
                    )
                )
            if isinstance(prompt, str):
                args.append(f"--prompt={prompt}")
            elif prompt is not None:
                path = f".vf-initial-messages-{trace.id}.json"
                await runtime.write(
                    path,
                    json.dumps([message_to_wire(m) for m in prompt]).encode(),
                )
                args.append(f"--initial-messages-file={path}")
            program = await runtime.prepare_uv_script(
                program_source, self.config.resolved_env
            )
            return await runtime.run_program([*program, *args], env)

    _TruncatingBashHarness = TruncatingBashHarness
    return TruncatingBashHarness


def wrap_bash_harness(config: Any) -> Any:
    """Build a TruncatingBashHarness for an existing bash HarnessConfig."""
    return get_truncating_bash_harness_class()(config)


def load_program_run_bash(*, max_tool_output_chars: int):
    """Exec the patched program and return its ``run_bash`` (integration tests)."""
    import os

    src = truncating_program_source()
    ns: dict[str, Any] = {"__name__": "truncating_bash_program"}
    previous = os.environ.get("MAX_TOOL_OUTPUT_CHARS")
    os.environ["MAX_TOOL_OUTPUT_CHARS"] = str(max_tool_output_chars)
    try:
        exec(compile(src, "<truncating-bash-program>", "exec"), ns, ns)
    finally:
        if previous is None:
            os.environ.pop("MAX_TOOL_OUTPUT_CHARS", None)
        else:
            os.environ["MAX_TOOL_OUTPUT_CHARS"] = previous
    run_bash = ns.get("run_bash")
    if not callable(run_bash):
        raise RuntimeError("patched bash program did not define run_bash")
    return run_bash


__all__ = [
    "build_truncating_program_source",
    "get_truncating_bash_harness_class",
    "load_program_run_bash",
    "truncate_tool_output",
    "truncating_program_source",
    "wrap_bash_harness",
]
