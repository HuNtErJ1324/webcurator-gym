"""Unit + harness-boundary tests for tool/bash/Codex output truncation."""

from __future__ import annotations

import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from pretrain_data_curator.bash_harness import (
    load_program_run_bash,
    truncating_program_source,
    wrap_bash_harness,
)
from pretrain_data_curator.hosted_compat import Environment
from pretrain_data_curator.models import CuratorConfig
from pretrain_data_curator.pretrain_data_curator import load_environment
from pretrain_data_curator.taskset import CuratorTaskset, CuratorTasksetConfig
from pretrain_data_curator.truncating_client import TruncatingClient, wrap_client
from pretrain_data_curator.utils import truncate_tool_output, truncate_wire_tool_outputs

EVAL_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "eval"


def test_truncate_tool_output_boundaries():
    s = "x" * 10
    out, truncated = truncate_tool_output(s, cap=0)
    assert out == s and truncated is False

    out_neg, t_neg = truncate_tool_output(s, cap=-1)
    assert out_neg == s and t_neg is False

    exact = "y" * 20
    out_exact, t_exact = truncate_tool_output(exact, cap=20)
    assert out_exact == exact and t_exact is False

    over = "a" * 101
    out2, t2 = truncate_tool_output(over, cap=100)
    assert t2 is True
    assert "TRUNCATED" in out2
    assert len(out2) <= 100
    assert out2.startswith("a")


def test_taskset_config_roundtrip():
    tsc = CuratorTasksetConfig(id="t", max_tool_output_chars=12345)
    ts = CuratorTaskset(tsc)
    assert isinstance(ts.curator, CuratorConfig)
    assert ts.curator.max_tool_output_chars == 12345


def test_load_environment_wires_cap_into_bash_harness(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-token")
    env = load_environment()
    assert env.harness.config.env.get("MAX_TOOL_OUTPUT_CHARS") == "20000"
    assert env.taskset.curator.max_tool_output_chars == 20_000
    assert env.env_args["max_tool_output_chars"] == 20_000
    assert type(env.harness).__name__ == "TruncatingBashHarness"
    assert env.max_tool_output_chars == 20_000

    env2 = load_environment(max_tool_output_chars=5_000)
    assert env2.harness.config.env.get("MAX_TOOL_OUTPUT_CHARS") == "5000"
    assert env2.taskset.curator.max_tool_output_chars == 5_000


def test_truncating_program_source_patches_stock_run_bash():
    src = truncating_program_source()
    assert "_maybe_truncate" in src
    assert "MAX_TOOL_OUTPUT_CHARS" in src
    assert "return _maybe_truncate(result.stdout + result.stderr)" in src


def test_program_run_bash_caps_over_20k_agent_visible_output():
    """Integration: >20k bash stdout through the patched harness program boundary."""
    run_bash = load_program_run_bash(max_tool_output_chars=20_000)
    out = run_bash("python3 -c \"print('A' * 25000, end='')\"")
    assert len(out) <= 20_000
    assert "TRUNCATED" in out
    assert out.startswith("A")

    # <=0 disables truncation inside the program.
    run_bash_off = load_program_run_bash(max_tool_output_chars=0)
    raw = run_bash_off("python3 -c \"print('B' * 25000, end='')\"")
    assert len(raw) == 25_000
    assert "TRUNCATED" not in raw


def test_wrap_bash_harness_preserves_config_id_and_runtime(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "tok")
    env = load_environment(harness_id="bash")
    wrapped = wrap_bash_harness(env.harness.config)
    assert wrapped.config.id == "bash"
    assert wrapped.config.env["MAX_TOOL_OUTPUT_CHARS"] == "20000"


def test_codex_responses_wire_boundary_caps_function_call_output():
    """Codex agent-visible boundary: Responses function_call_output.output."""
    body = {
        "model": "test",
        "input": [
            {
                "type": "function_call",
                "call_id": "c1",
                "name": "shell",
                "arguments": "{}",
            },
            {
                "type": "function_call_output",
                "call_id": "c1",
                "output": "A" * 25_000,
            },
        ],
    }
    under = {
        "input": [
            {
                "type": "function_call_output",
                "call_id": "c2",
                "output": "short",
            }
        ]
    }

    truncate_wire_tool_outputs(body, 20_000)
    out = body["input"][1]["output"]
    assert isinstance(out, str)
    assert len(out) <= 20_000
    assert "TRUNCATED" in out
    assert out.startswith("A")
    # Non-tool items untouched.
    assert body["input"][0]["type"] == "function_call"

    truncate_wire_tool_outputs(under, 20_000)
    assert under["input"][0]["output"] == "short"

    disabled = {
        "input": [
            {
                "type": "function_call_output",
                "call_id": "c3",
                "output": "C" * 25_000,
            }
        ]
    }
    truncate_wire_tool_outputs(disabled, 0)
    assert len(disabled["input"][0]["output"]) == 25_000


@pytest.mark.asyncio
async def test_truncating_client_caps_body_before_provider_forward():
    """Integration: TruncatingClient mutates the body EvalClient would forward."""
    captured: dict[str, Any] = {}

    class _FakeClient:
        async def get_response(self, dialect, body, model, sampling_args, **kwargs):
            captured["body"] = body
            return SimpleNamespace(raw={"ok": True})

        async def close(self) -> None:
            return None

    client = TruncatingClient(_FakeClient(), max_tool_output_chars=20_000)
    body = {
        "input": [
            {
                "type": "function_call_output",
                "call_id": "c1",
                "output": "Z" * 25_000,
            }
        ]
    }
    await client.get_response(
        dialect=SimpleNamespace(),
        body=body,
        model="m",
        sampling_args=SimpleNamespace(),
    )
    forwarded = captured["body"]["input"][0]["output"]
    assert len(forwarded) <= 20_000
    assert "TRUNCATED" in forwarded
    # In-place mutation of the interception body.
    assert body["input"][0]["output"] == forwarded


def test_load_environment_codex_installs_client_cap(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-token")
    env = load_environment(harness_id="codex")
    assert isinstance(env, Environment)
    assert env.harness.config.id == "codex"
    assert env.max_tool_output_chars == 20_000
    assert env.harness.config.env.get("MAX_TOOL_OUTPUT_CHARS") == "20000"

    class _Inner:
        pass

    wrapped = env._capping_client(_Inner())  # type: ignore[arg-type]
    assert isinstance(wrapped, TruncatingClient)
    assert wrapped.max_tool_output_chars == 20_000

    disabled = load_environment(harness_id="codex", max_tool_output_chars=0)
    assert wrap_client(_Inner(), disabled.max_tool_output_chars) is not None
    assert not isinstance(
        disabled._capping_client(_Inner()),
        TruncatingClient,  # type: ignore[arg-type]
    )


def test_shipped_codex_eval_configs_receive_default_cap(monkeypatch):
    """Shipped production configs keep harness_id=codex and inherit the default cap."""
    monkeypatch.setenv("HF_TOKEN", "test-token")
    configs = sorted(EVAL_CONFIG_DIR.glob("*codex*.toml"))
    assert configs, "expected shipped *codex*.toml eval configs"

    for path in configs:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        eval_rows = data.get("eval") or []
        if eval_rows:
            args = eval_rows[0].get("args") or {}
        else:
            args = data.get("args") or {}
        assert args, path.name
        assert args.get("harness_id") == "codex", path.name
        # Configs omit max_tool_output_chars so load_environment's default applies.
        assert "max_tool_output_chars" not in args, path.name

        env = load_environment(harness_id="codex")
        assert env.max_tool_output_chars == 20_000
        assert env.env_args["max_tool_output_chars"] == 20_000
        assert isinstance(env._capping_client(object()), TruncatingClient)  # type: ignore[arg-type]
        # Must not silently switch production harness to bash.
        assert env.harness.config.id == "codex"
