"""Tests for the bounded Codex missing-manifest continuation guard.

The stock codex harness ends a rollout on any clean exit, so an empty model turn
at turn 4/300 scored zero with the turn budget untouched. These tests pin the
guard's contract: continue only on a clean exit with no manifest and turns left,
bounded by a retry cap, and never at the expense of strict scoring.

Everything here is deterministic — fake runtime, fake trace, scripted exit codes
and manifest visibility. No codex binary, no network, no sandbox.
"""

from __future__ import annotations

from typing import Any

import pytest
from verifiers.v1.errors import HarnessError
from verifiers.v1.harnesses.codex.harness import CodexHarness, CodexHarnessConfig
from verifiers.v1.runtimes import ProgramResult

from pretrain_data_curator.codex_harness import (
    DEFAULT_MAX_CONTINUATIONS,
    INFO_KEY,
    METRIC_CONTINUATIONS,
    METRIC_MANIFEST_PRESENT,
    OUTCOME_FRAMEWORK_STOP,
    OUTCOME_MANIFEST_PRESENT,
    OUTCOME_MANIFEST_RECOVERED,
    OUTCOME_NONZERO_EXIT,
    OUTCOME_RETRY_CAP,
    OUTCOME_TURN_BUDGET,
    build_continuation_argv,
    continuation_nudge,
    get_manifest_continuation_codex_harness_class,
    has_turn_budget,
    manifest_is_present,
    wrap_codex_harness,
)

MANIFEST = "manifest.json"


class FakeRuntime:
    """Runtime that scripts codex's exit codes and when the manifest appears.

    ``manifest_after`` is the number of completed ``run_program`` calls after
    which the file exists — ``None`` means the agent never writes it.
    """

    def __init__(
        self,
        results: list[ProgramResult],
        *,
        manifest_after: int | None = None,
        manifest_body: bytes = b'{"sources": [{"id": "x"}]}',
    ) -> None:
        self._results = list(results)
        self.manifest_after = manifest_after
        self.manifest_body = manifest_body
        self.calls: list[tuple[list[str], dict[str, str]]] = []
        self.reads: list[str] = []

    async def run_program(self, argv: list[str], env: dict[str, str]) -> ProgramResult:
        self.calls.append((list(argv), dict(env)))
        if not self._results:
            raise AssertionError("codex launched more times than the test scripted")
        return self._results.pop(0)

    async def read(self, path: str) -> bytes:
        self.reads.append(path)
        if self.manifest_after is None or len(self.calls) < self.manifest_after:
            raise FileNotFoundError(path)  # backends raise their own not-found
        return self.manifest_body


class FakeTrace:
    """The slice of `Trace` the guard touches."""

    def __init__(
        self, *, num_turns: int = 4, stop_condition: str | None = None
    ) -> None:
        self.num_turns = num_turns
        self.stop_condition = stop_condition
        self.is_completed = False
        self.metrics: dict[str, float] = {}
        self.info: dict[str, Any] = {}
        self.task = _FakeTask()

    def record_metric(self, name: str, value: float) -> None:
        assert name not in self.metrics, f"metric {name!r} recorded twice"
        self.metrics[name] = float(value)

    def stop(self, condition: str = "done") -> None:
        self.is_completed = True
        if self.stop_condition is None:
            self.stop_condition = condition


class _FakeTask:
    prompt = "Curate a pretraining corpus."
    system_prompt = None


class FakeCtx:
    model = "test-model"


OK = ProgramResult(exit_code=0, stdout="", stderr="")
CRASH = ProgramResult(exit_code=1, stdout="", stderr="codex: boom\nTraceback: bad")


def make_harness(
    *,
    max_turns: int | None = 300,
    max_continuations: int = DEFAULT_MAX_CONTINUATIONS,
    resume_context: bool = True,
) -> Any:
    return wrap_codex_harness(
        CodexHarnessConfig(id="codex"),
        manifest_filename=MANIFEST,
        max_turns=max_turns,
        max_continuations=max_continuations,
        resume_context=resume_context,
    )


async def run_harness(harness: Any, runtime: FakeRuntime, trace: FakeTrace) -> None:
    await harness.run(FakeCtx(), trace, runtime, "http://endpoint", "secret", {})


# --- the guard subclasses the stock harness (no forked argv) ---------------


def test_guard_is_a_codex_harness_subclass():
    """It must remain a CodexHarness, so install/setup stay upstream's."""
    cls = get_manifest_continuation_codex_harness_class()
    assert issubclass(cls, CodexHarness)


def test_negative_retry_cap_is_rejected():
    with pytest.raises(ValueError, match="max_continuations"):
        make_harness(max_continuations=-1)


# --- missing-manifest recovery --------------------------------------------


@pytest.mark.asyncio
async def test_missing_manifest_is_continued_and_recovered():
    """Clean exit, no manifest, turns left -> relaunch; manifest lands -> done."""
    runtime = FakeRuntime([OK, OK], manifest_after=2)
    trace = FakeTrace(num_turns=4)
    await run_harness(make_harness(), runtime, trace)

    assert len(runtime.calls) == 2, "guard should have continued codex exactly once"
    assert trace.stop_condition == "agent_completed"
    assert trace.metrics[METRIC_CONTINUATIONS] == 1.0
    assert trace.metrics[METRIC_MANIFEST_PRESENT] == 1.0
    assert trace.info[INFO_KEY]["outcome"] == OUTCOME_MANIFEST_RECOVERED


@pytest.mark.asyncio
async def test_continuation_preserves_context_and_reuses_stock_options():
    """The relaunch resumes the recorded session and reuses the launch options."""
    runtime = FakeRuntime([OK, OK], manifest_after=2)
    await run_harness(make_harness(), runtime, FakeTrace())

    first_argv, first_env = runtime.calls[0]
    second_argv, second_env = runtime.calls[1]

    # Context preservation: `codex exec <options> resume --last <nudge>`.
    assert second_argv[-3:-1] == ["resume", "--last"]
    assert second_argv[-1] == continuation_nudge(MANIFEST)
    # Codex's exec options are global to the parent command, so they must
    # precede the `resume` subcommand.
    assert second_argv.index("resume") == len(first_argv) - 1

    # Every option the stock harness passed (model, provider overrides, the
    # intercept endpoint) is carried over verbatim; only the prompt changes.
    assert second_argv[: len(first_argv) - 1] == first_argv[:-1]
    assert second_env == first_env, "continuation must keep the intercept key/env"


@pytest.mark.asyncio
async def test_continuation_nudge_does_not_invite_fabrication():
    """The nudge buys turns; it must not lower the bar."""
    nudge = continuation_nudge(MANIFEST)
    assert MANIFEST in nudge
    assert "not complete" in nudge.lower()
    assert "fabricate" in nudge.lower()


# --- manifest present: no retry -------------------------------------------


@pytest.mark.asyncio
async def test_manifest_present_does_not_retry():
    """A clean exit that produced a manifest is a normal success."""
    runtime = FakeRuntime([OK], manifest_after=1)
    trace = FakeTrace()
    await run_harness(make_harness(), runtime, trace)

    assert len(runtime.calls) == 1
    assert trace.stop_condition == "agent_completed"
    assert trace.metrics[METRIC_CONTINUATIONS] == 0.0
    assert trace.metrics[METRIC_MANIFEST_PRESENT] == 1.0
    assert trace.info[INFO_KEY]["outcome"] == OUTCOME_MANIFEST_PRESENT


@pytest.mark.asyncio
async def test_empty_manifest_file_counts_as_absent():
    """A zero-byte file is not a manifest; the guard still nudges."""
    runtime = FakeRuntime([OK, OK], manifest_after=1, manifest_body=b"")
    await run_harness(make_harness(max_continuations=1), runtime, FakeTrace())
    assert len(runtime.calls) == 2


@pytest.mark.asyncio
async def test_present_but_malformed_manifest_is_not_retried():
    """Presence is file-existence: a malformed manifest is the scorer's call.

    The guard must not coach the agent past the parser — that would be a
    scoring concern, and it stays strictly the taskset's.
    """
    runtime = FakeRuntime([OK], manifest_after=1, manifest_body=b"not json at all")
    trace = FakeTrace()
    await run_harness(make_harness(), runtime, trace)

    assert len(runtime.calls) == 1, "malformed-but-present must stop normally"
    assert trace.info[INFO_KEY]["outcome"] == OUTCOME_MANIFEST_PRESENT


# --- nonzero exit: no retry ------------------------------------------------


@pytest.mark.asyncio
async def test_nonzero_exit_raises_and_is_never_retried():
    """A crashed codex is a harness error, not a missing-manifest recovery."""
    runtime = FakeRuntime([CRASH], manifest_after=None)
    trace = FakeTrace()

    with pytest.raises(HarnessError) as excinfo:
        await run_harness(make_harness(), runtime, trace)

    assert len(runtime.calls) == 1, "a nonzero exit must never be relaunched"
    assert "exited 1" in str(excinfo.value)
    assert "Traceback: bad" in str(excinfo.value), "keep the tail of the real cause"
    assert trace.info[INFO_KEY]["outcome"] == OUTCOME_NONZERO_EXIT


@pytest.mark.asyncio
async def test_nonzero_exit_during_a_continuation_still_raises():
    """The no-retry-on-crash rule holds on the recovery path too."""
    runtime = FakeRuntime([OK, CRASH], manifest_after=None)
    trace = FakeTrace()

    with pytest.raises(HarnessError):
        await run_harness(make_harness(), runtime, trace)

    assert len(runtime.calls) == 2, "one continuation, then the crash stops it"
    assert trace.info[INFO_KEY]["outcome"] == OUTCOME_NONZERO_EXIT


# --- retry cap -------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("cap", [0, 2])
async def test_retry_cap_bounds_the_relaunches(cap: int):
    """An agent that never writes a manifest is nudged at most `cap` times."""
    runtime = FakeRuntime([OK] * (cap + 1), manifest_after=None)
    trace = FakeTrace(num_turns=4)
    await run_harness(make_harness(max_continuations=cap), runtime, trace)

    assert len(runtime.calls) == cap + 1, "initial launch + at most `cap` retries"
    assert trace.metrics[METRIC_CONTINUATIONS] == float(cap)
    assert trace.metrics[METRIC_MANIFEST_PRESENT] == 0.0
    assert trace.info[INFO_KEY]["outcome"] == OUTCOME_RETRY_CAP
    # Still a normal stop: the rollout is scored (strictly, as manifest-missing),
    # not failed.
    assert trace.stop_condition == "agent_completed"


def test_default_cap_is_bounded_and_small():
    assert DEFAULT_MAX_CONTINUATIONS in (2, 3)


# --- turn budget -----------------------------------------------------------


@pytest.mark.asyncio
async def test_no_continuation_when_turn_budget_is_exhausted():
    """With no turns left, a relaunch could not do any work; don't burn one."""
    runtime = FakeRuntime([OK], manifest_after=None)
    trace = FakeTrace(num_turns=300)
    await run_harness(make_harness(max_turns=300), runtime, trace)

    assert len(runtime.calls) == 1
    assert trace.info[INFO_KEY]["outcome"] == OUTCOME_TURN_BUDGET
    assert trace.stop_condition == "agent_completed"


@pytest.mark.asyncio
async def test_continuation_happens_with_a_single_turn_left():
    """One turn of headroom is enough to justify a nudge."""
    runtime = FakeRuntime([OK, OK], manifest_after=None)
    trace = FakeTrace(num_turns=299)
    await run_harness(make_harness(max_turns=300, max_continuations=1), runtime, trace)
    assert len(runtime.calls) == 2


@pytest.mark.asyncio
async def test_framework_stop_is_not_retried():
    """max_turns / a @stop already halted the rollout: its exit is expected."""
    runtime = FakeRuntime([OK], manifest_after=None)
    trace = FakeTrace(num_turns=300, stop_condition="max_turns")
    await run_harness(make_harness(), runtime, trace)

    assert len(runtime.calls) == 1
    assert trace.stop_condition == "max_turns", "the guard must not overwrite it"
    assert trace.info[INFO_KEY]["outcome"] == OUTCOME_FRAMEWORK_STOP


def test_has_turn_budget():
    assert has_turn_budget(4, 300) is True
    assert has_turn_budget(299, 300) is True
    assert has_turn_budget(300, 300) is False
    assert has_turn_budget(301, 300) is False
    assert has_turn_budget(10_000, None) is True, "no cap means turns always remain"


# --- observability ---------------------------------------------------------


@pytest.mark.asyncio
async def test_observability_payload_is_complete():
    """The rollout record explains what the guard did and why."""
    runtime = FakeRuntime([OK, OK], manifest_after=2)
    trace = FakeTrace(num_turns=4)
    await run_harness(make_harness(max_turns=300), runtime, trace)

    info = trace.info[INFO_KEY]
    assert info == {
        "attempts": 1,
        "max_continuations": DEFAULT_MAX_CONTINUATIONS,
        "outcome": OUTCOME_MANIFEST_RECOVERED,
        "manifest_present": True,
        "manifest_filename": MANIFEST,
        "turns_used": 4,
        "max_turns": 300,
        "resume_context": True,
    }
    assert set(trace.metrics) == {METRIC_CONTINUATIONS, METRIC_MANIFEST_PRESENT}


@pytest.mark.asyncio
async def test_guard_reads_the_configured_manifest_filename():
    """A non-default manifest_filename is what gets probed."""
    harness = wrap_codex_harness(
        CodexHarnessConfig(id="codex"),
        manifest_filename="curated.json",
        max_turns=300,
        max_continuations=0,
    )
    runtime = FakeRuntime([OK], manifest_after=None)
    await run_harness(harness, runtime, FakeTrace())
    assert runtime.reads == ["curated.json"]


# --- wiring: load_environment registers the guard for harness_id=codex -----


def test_load_environment_registers_the_guard_for_codex():
    """`--harness.id codex` must get the guard, with the env's manifest/turn config."""
    from pretrain_data_curator.pretrain_data_curator import load_environment

    harness: Any = load_environment(
        harness_id="codex", max_turns=300, manifest_filename="manifest.json"
    ).harness

    assert isinstance(harness, get_manifest_continuation_codex_harness_class())
    assert harness.manifest_filename == "manifest.json"
    assert harness.max_turns == 300
    assert harness.max_continuations == DEFAULT_MAX_CONTINUATIONS
    # The config must stay a real CodexHarnessConfig, or upstream's install
    # (which reads `version`) breaks.
    assert isinstance(harness.config, CodexHarnessConfig)
    assert harness.config.version


def test_codex_max_continuations_is_configurable():
    from pretrain_data_curator.pretrain_data_curator import load_environment

    env: Any = load_environment(harness_id="codex", codex_max_continuations=3)
    assert env.harness.max_continuations == 3
    assert env.env_args["codex_max_continuations"] == 3


def test_other_harnesses_are_not_wrapped_by_the_codex_guard():
    """The guard is codex-specific; bash keeps its own wrapper."""
    from pretrain_data_curator.pretrain_data_curator import load_environment

    harness = load_environment(harness_id="bash").harness
    assert not isinstance(harness, get_manifest_continuation_codex_harness_class())


# --- unit: helpers ---------------------------------------------------------


@pytest.mark.asyncio
async def test_manifest_is_present_maps_read_failure_to_absent():
    runtime = FakeRuntime([], manifest_after=None)
    assert await manifest_is_present(runtime, MANIFEST) is False
    assert await manifest_is_present(None, MANIFEST) is False

    present = FakeRuntime([], manifest_after=0)
    assert await manifest_is_present(present, MANIFEST) is True


def test_build_continuation_argv_without_resume_is_context_free():
    base = ["codex", "exec", "-m", "gpt", "do the task"]
    assert build_continuation_argv(base, "nudge", resume=False) == [
        "codex",
        "exec",
        "-m",
        "gpt",
        "nudge",
    ]


def test_build_continuation_argv_rejects_a_degenerate_argv():
    with pytest.raises(ValueError, match="cannot continue"):
        build_continuation_argv(["codex"], "nudge")
