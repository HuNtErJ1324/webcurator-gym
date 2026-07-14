"""Codex harness variant that guards against a premature ``agent_completed``.

The stock Verifiers ``codex`` harness treats *any* clean exit as
``agent_completed``. Codex exits 0 after an empty model turn, so a rollout can
"succeed" at turn 4/300 with no ``manifest.json`` in the workspace and ~290
turns of budget left unspent — the agent simply stopped talking, and nothing
asked it to continue.

This subclass re-implements :meth:`Harness.run` with a *bounded* continuation
guard. On a clean exit only, if the required manifest file is absent and turn
budget remains, it relaunches Codex with an explicit nudge, up to
``max_continuations`` times. Context is preserved across a continuation via
``codex exec ... resume --last`` (see :func:`build_continuation_argv`).

Deliberate non-goals, so the guard can never launder a bad rollout into a good
score:

* It never writes, repairs, or synthesizes a manifest — it only tests for the
  file's presence and asks the agent to do the work.
* It never touches scoring. A recovered rollout is scored by the taskset on
  exactly the same strict terms as any other.
* It never retries a nonzero exit. A crashed Codex is a harness error, not a
  missing-manifest case, and still raises.
* Presence is *file existence*, not validity. A present-but-malformed manifest
  stops normally and is scored (strictly) as invalid — the guard exists to fix
  "the agent never wrote anything", not to coach it past the parser.

Imports of ``CodexHarness`` are deferred so importing this module never fails
against a stub/partial Verifiers install (mirrors ``bash_harness``).
"""

from __future__ import annotations

import logging
from typing import Any, cast

logger = logging.getLogger(__name__)

#: Recovery attempts allowed after the initial launch. Bounded on purpose: a
#: model that ignores two explicit nudges is not going to be talked into the
#: task by a third, and each attempt burns real turns and wall-clock.
DEFAULT_MAX_CONTINUATIONS = 2

#: Metric/info namespace, prefixed so it cannot collide with taskset metrics
#: (the taskset already reports ``manifest_missing`` / ``manifest_invalid``).
METRIC_CONTINUATIONS = "codex_continuations"
METRIC_MANIFEST_PRESENT = "codex_manifest_present"
INFO_KEY = "codex_continuation"

# Outcomes recorded in ``trace.info[INFO_KEY]["outcome"]``.
OUTCOME_MANIFEST_PRESENT = "manifest_present"  # clean first pass, no nudge needed
OUTCOME_MANIFEST_RECOVERED = "manifest_recovered"  # nudge worked
OUTCOME_RETRY_CAP = "retry_cap_reached"  # nudged to the cap, still nothing
OUTCOME_TURN_BUDGET = "turn_budget_exhausted"  # no turns left to nudge into
OUTCOME_FRAMEWORK_STOP = "framework_stop"  # max_turns / @stop halted the rollout
OUTCOME_NONZERO_EXIT = "nonzero_exit"  # codex crashed; not a recovery case


def continuation_nudge(manifest_filename: str) -> str:
    """The prompt used to continue a Codex run that stopped without a manifest.

    States the missing artifact and the fact that the task is unfinished, and
    explicitly rules out inventing sources — the guard buys the agent more
    turns, it does not lower the bar.
    """
    return (
        f"You stopped without writing the required manifest file "
        f"{manifest_filename!r} to your workspace, so the task is NOT complete "
        f"and your work so far will score zero.\n\n"
        f"Continue the curation task now. Do not fabricate sources or "
        f"statistics: list only datasets you have actually inspected and "
        f"verified against the task's constraints. When you are done, write "
        f"the manifest to {manifest_filename!r} in your workspace before you "
        f"finish."
    )


def build_continuation_argv(
    base_argv: list[str], nudge: str, *, resume: bool = True
) -> list[str]:
    """Build the continuation argv from the argv the stock harness launched.

    The stock Codex argv is ``[codex, exec, <options...>, <prompt>]`` — the
    prompt is the trailing positional. A continuation reuses every option
    verbatim (model, provider overrides, disabled tools) and swaps the prompt
    for ``nudge``.

    With ``resume`` (the default) the continuation is
    ``codex exec <options...> resume --last <nudge>``, which reattaches to the
    session Codex just recorded, so the agent keeps the context it built up.
    Codex's exec options are declared on the parent command, so they must
    precede the ``resume`` subcommand; ``--last`` with no session id makes the
    trailing positional the prompt (``codex exec [OPTIONS] resume [OPTS]
    [SESSION_ID] [PROMPT]``).

    ``resume=False`` yields a context-free relaunch, for a runtime where no
    session was recorded to resume.
    """
    if len(base_argv) < 2:
        raise ValueError(f"unexpected codex argv, cannot continue: {base_argv!r}")
    options = list(base_argv[:-1])  # everything but the trailing prompt positional
    if resume:
        return [*options, "resume", "--last", nudge]
    return [*options, nudge]


async def manifest_is_present(runtime: Any, manifest_filename: str) -> bool:
    """Whether the agent left a non-empty manifest file in the runtime workspace.

    Mirrors the taskset's own read (``runtime.read`` on the configured filename,
    resolved relative to the runtime workdir), and deliberately answers only
    *does the file exist*: parsing is the scorer's job, not the guard's. Runtime
    backends raise different not-found errors, so any read failure means absent.
    """
    if runtime is None:
        return False
    try:
        raw = await runtime.read(manifest_filename)
    except Exception:  # noqa: BLE001 - backends use different not-found errors
        return False
    return bool(raw)


def has_turn_budget(num_turns: int, max_turns: int | None) -> bool:
    """Whether at least one model turn remains to spend on a continuation.

    ``max_turns`` of ``None`` means the framework caps nothing, so a turn is
    always available. Relaunching with no budget left would just burn a Codex
    process against an interception server that refuses its first call.
    """
    if max_turns is None:
        return True
    return num_turns < max_turns


class _ArgvRecordingRuntime:
    """Transparent ``Runtime`` proxy that records the argv/env of ``run_program``.

    Lets the guard reuse the *stock* harness's argv for continuations instead of
    rebuilding it here — so an upstream change to Codex's flags (a renamed
    provider override, a new default) is picked up automatically rather than
    silently drifting away from a hand-copied duplicate.
    """

    def __init__(self, runtime: Any) -> None:
        self._runtime = runtime
        self.argv: list[str] | None = None
        self.env: dict[str, str] | None = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self._runtime, name)

    async def run_program(self, argv: list[str], env: dict[str, str]) -> Any:
        self.argv = list(argv)
        self.env = dict(env)
        return await self._runtime.run_program(argv, env)


_ManifestContinuationCodexHarness: type[Any] | None = None


def get_manifest_continuation_codex_harness_class() -> type[Any]:
    """Return the guard class, importing ``CodexHarness`` lazily."""
    global _ManifestContinuationCodexHarness
    if _ManifestContinuationCodexHarness is not None:
        return _ManifestContinuationCodexHarness

    from verifiers.v1.clients import RolloutContext
    from verifiers.v1.errors import HarnessError, boundary
    from verifiers.v1.harnesses.codex.harness import CodexHarness
    from verifiers.v1.runtimes import Runtime
    from verifiers.v1.trace import Trace

    class ManifestContinuationCodexHarness(CodexHarness):
        """Codex harness that will not call a manifest-less clean exit a success."""

        def __init__(
            self,
            config: Any,
            *,
            manifest_filename: str,
            max_turns: int | None = None,
            max_continuations: int = DEFAULT_MAX_CONTINUATIONS,
            resume_context: bool = True,
        ) -> None:
            super().__init__(config)
            if max_continuations < 0:
                raise ValueError(
                    f"max_continuations must be >= 0, got {max_continuations}"
                )
            self.manifest_filename = manifest_filename
            self.max_turns = max_turns
            self.max_continuations = max_continuations
            self.resume_context = resume_context

        async def run(
            self,
            ctx: RolloutContext,
            trace: Trace,
            runtime: Runtime,
            endpoint: str,
            secret: str,
            mcp_urls: dict[str, str],
        ) -> None:
            """Stock ``Harness.run``, plus a bounded missing-manifest continuation.

            Per-rollout state stays on the stack / the trace: one harness
            instance is shared across concurrent rollouts, so it must never
            accumulate anything itself.
            """
            label = f"harness {self.config.id!r}"
            recorder = _ArgvRecordingRuntime(runtime)
            attempts = 0

            async with boundary(HarnessError, label):
                result = await self.launch(
                    ctx,
                    trace,
                    cast("Runtime", recorder),
                    endpoint,
                    secret,
                    mcp_urls,
                )

            while True:
                if trace.stop_condition is not None:
                    # A @stop / max_turns refused a turn mid-rollout. The exit is
                    # expected and the budget is gone; nudging would be pointless.
                    self._record(trace, attempts, OUTCOME_FRAMEWORK_STOP, False)
                    return

                if result.exit_code != 0:
                    # Codex itself failed. Never a recovery case: retrying a
                    # crash just crashes again, and hides a real harness bug.
                    self._record(trace, attempts, OUTCOME_NONZERO_EXIT, False)
                    # The real cause is at the END of a traceback, so keep the tail.
                    detail = (result.stderr or result.stdout).strip()[
                        -2000:
                    ] or "<no output>"
                    raise HarnessError(f"{label} exited {result.exit_code}: {detail}")

                if await manifest_is_present(runtime, self.manifest_filename):
                    outcome = (
                        OUTCOME_MANIFEST_RECOVERED
                        if attempts
                        else OUTCOME_MANIFEST_PRESENT
                    )
                    self._record(trace, attempts, outcome, True)
                    break

                if attempts >= self.max_continuations:
                    logger.warning(
                        "%s: no %r after %d continuation(s); giving up (retry cap)",
                        label,
                        self.manifest_filename,
                        attempts,
                    )
                    self._record(trace, attempts, OUTCOME_RETRY_CAP, False)
                    break

                if not has_turn_budget(trace.num_turns, self.max_turns):
                    logger.warning(
                        "%s: no %r and no turn budget left (%s/%s); giving up",
                        label,
                        self.manifest_filename,
                        trace.num_turns,
                        self.max_turns,
                    )
                    self._record(trace, attempts, OUTCOME_TURN_BUDGET, False)
                    break

                if recorder.argv is None or recorder.env is None:
                    # The stock launch never reached run_program; nothing to reuse.
                    logger.warning("%s: no recorded codex argv; cannot continue", label)
                    self._record(trace, attempts, OUTCOME_RETRY_CAP, False)
                    break

                attempts += 1
                logger.warning(
                    "%s: clean exit at turn %s/%s with no %r; continuing codex "
                    "(attempt %d/%d)",
                    label,
                    trace.num_turns,
                    self.max_turns,
                    self.manifest_filename,
                    attempts,
                    self.max_continuations,
                )
                argv = build_continuation_argv(
                    recorder.argv,
                    continuation_nudge(self.manifest_filename),
                    resume=self.resume_context,
                )
                async with boundary(HarnessError, label):
                    result = await runtime.run_program(argv, recorder.env)

            trace.stop("agent_completed")

        def _record(
            self,
            trace: Trace,
            attempts: int,
            outcome: str,
            manifest_present: bool,
        ) -> None:
            """Record continuation telemetry once, at the end of the run.

            ``record_metric`` warns on override, so this is called exactly once
            per rollout on every exit path (including the raising one).
            """
            trace.record_metric(METRIC_CONTINUATIONS, float(attempts))
            trace.record_metric(
                METRIC_MANIFEST_PRESENT, 1.0 if manifest_present else 0.0
            )
            trace.info[INFO_KEY] = {
                "attempts": attempts,
                "max_continuations": self.max_continuations,
                "outcome": outcome,
                "manifest_present": manifest_present,
                "manifest_filename": self.manifest_filename,
                "turns_used": trace.num_turns,
                "max_turns": self.max_turns,
                "resume_context": self.resume_context,
            }

    _ManifestContinuationCodexHarness = ManifestContinuationCodexHarness
    return ManifestContinuationCodexHarness


def wrap_codex_harness(
    config: Any,
    *,
    manifest_filename: str,
    max_turns: int | None = None,
    max_continuations: int = DEFAULT_MAX_CONTINUATIONS,
    resume_context: bool = True,
) -> Any:
    """Build a ManifestContinuationCodexHarness for an existing codex HarnessConfig."""
    return get_manifest_continuation_codex_harness_class()(
        config,
        manifest_filename=manifest_filename,
        max_turns=max_turns,
        max_continuations=max_continuations,
        resume_context=resume_context,
    )


__all__ = [
    "DEFAULT_MAX_CONTINUATIONS",
    "INFO_KEY",
    "METRIC_CONTINUATIONS",
    "METRIC_MANIFEST_PRESENT",
    "OUTCOME_FRAMEWORK_STOP",
    "OUTCOME_MANIFEST_PRESENT",
    "OUTCOME_MANIFEST_RECOVERED",
    "OUTCOME_NONZERO_EXIT",
    "OUTCOME_RETRY_CAP",
    "OUTCOME_TURN_BUDGET",
    "build_continuation_argv",
    "continuation_nudge",
    "get_manifest_continuation_codex_harness_class",
    "has_turn_budget",
    "manifest_is_present",
    "wrap_codex_harness",
]
