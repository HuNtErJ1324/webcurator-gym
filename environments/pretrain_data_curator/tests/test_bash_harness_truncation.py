"""Bash-harness (in-runtime) tool-output truncation tests.

The codex/wire-level ``TruncatingClient`` capping was removed (externalized to
the runner); these tests cover only the bash harness path, which patches the
stock uv program so ``run_bash``/tool appends are capped inside the agent shell.
"""

from __future__ import annotations

import pytest

from pretrain_data_curator.bash_harness import (
    load_program_run_bash,
    wrap_bash_harness,
)
from pretrain_data_curator.models import CuratorConfig
from pretrain_data_curator.pretrain_data_curator import load_environment
from pretrain_data_curator.taskset import CuratorTaskset, CuratorTasksetConfig
from pretrain_data_curator.utils import truncate_tool_output


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


def test_load_environment_wires_cap_into_bash_harness(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-token")
    env = load_environment()
    assert env.harness.config.env.get("MAX_TOOL_OUTPUT_CHARS") == "20000"
    assert env.taskset.curator.max_tool_output_chars == 20_000
    assert type(env.harness).__name__ == "TruncatingBashHarness"

    env2 = load_environment(max_tool_output_chars=5_000)
    assert env2.harness.config.env.get("MAX_TOOL_OUTPUT_CHARS") == "5000"
    assert env2.taskset.curator.max_tool_output_chars == 5_000


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


def test_taskset_config_roundtrip():
    tsc = CuratorTasksetConfig(id="t", max_tool_output_chars=12345)
    ts = CuratorTaskset(tsc)
    assert isinstance(ts.curator, CuratorConfig)
    assert ts.curator.max_tool_output_chars == 12345
