"""Native verifiers v1 Environment that caps tool outputs at the wire boundary.

Codex runs as a compiled CLI speaking the Responses API. Its tool
``function_call_output`` items arrive on the interception server and are
forwarded 1:1 by ``EvalClient`` via ``dialect.apply_overrides`` — they never
pass through a Python harness program. Wrapping the rollout ``Client`` here
truncates those items (and chat ``role=tool`` messages) immediately before the
provider call — the actual agent-visible boundary — without touching the Codex
binary. This is the only behavior carried over from the old v0 compatibility
façade; everything else (the synthesized HF dataset, run_rollout/run_group
shims, Trace→State conversion, legacy client-config translation) is gone.
"""

from __future__ import annotations

from typing import Any

import verifiers.v1 as vf
from verifiers.v1.clients.client import RolloutContext
from verifiers.v1.episode import Episode
from verifiers.v1.task import Task

from .truncating_client import wrap_client


class Environment(vf.Environment):
    """v1 Environment that caps Codex tool results at the interception→provider boundary."""

    def __init__(
        self,
        config: vf.EnvConfig,
        *,
        max_tool_output_chars: int = 20_000,
        env_args: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(config)
        self.max_tool_output_chars = int(max_tool_output_chars)
        self.env_args: dict[str, Any] = dict(env_args or {})

    def _capping_client(self, client: vf.Client) -> vf.Client:
        """Cap Codex/chat tool results at the interception→provider wire boundary."""
        return wrap_client(client, self.max_tool_output_chars)

    def episode(self, task: Task, ctx: RolloutContext, n: int = 1) -> Episode:
        # prime eval / env-server inject the raw client here; wrap so Codex
        # function_call_output items are capped before EvalClient forwards them.
        capped_ctx = RolloutContext(
            model=ctx.model,
            client=self._capping_client(ctx.client),
            sampling=ctx.sampling,
        )
        return super().episode(task, capped_ctx, n)


__all__ = ["Environment"]
