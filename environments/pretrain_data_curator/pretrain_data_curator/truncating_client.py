"""Client wrapper that caps agent-visible tool results on the wire body.

Codex runs as a compiled CLI and speaks the Responses API. Tool stdout/stderr
never pass through a Python harness program; they arrive on the interception
server as ``function_call_output`` items and are forwarded 1:1 by ``EvalClient``
via ``dialect.apply_overrides``. Wrapping the rollout ``Client`` truncates those
items (and chat ``role=tool`` messages) immediately before the provider call —
the actual agent-visible boundary — without touching the Codex binary.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from verifiers.v1.clients.client import Client, RelayReply
from verifiers.v1.dialects import Dialect
from verifiers.v1.graph import PendingTurn
from verifiers.v1.types import Response, SamplingConfig

from .utils import truncate_wire_tool_outputs


class TruncatingClient(Client):
    """Delegate to an inner client after capping tool outputs on ``body``."""

    def __init__(self, inner: Client, max_tool_output_chars: int) -> None:
        self.inner = inner
        self.max_tool_output_chars = int(max_tool_output_chars)

    def _cap(self, body: dict[str, Any]) -> dict[str, Any]:
        return truncate_wire_tool_outputs(body, self.max_tool_output_chars)

    async def get_response(
        self,
        dialect: Dialect,
        body: dict,
        model: str,
        sampling_args: SamplingConfig,
        session_id: str | None = None,
        turn: PendingTurn | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> Response:
        return await self.inner.get_response(
            dialect,
            self._cap(body),
            model,
            sampling_args,
            session_id=session_id,
            turn=turn,
            headers=headers,
        )

    async def relay(
        self,
        dialect: Dialect,
        body: dict,
        model: str,
        sampling_args: SamplingConfig,
        session_id: str | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> RelayReply:
        return await self.inner.relay(
            dialect,
            self._cap(body),
            model,
            sampling_args,
            session_id=session_id,
            headers=headers,
        )

    async def relay_aux(self, dialect: Dialect, route: str, body: dict) -> dict:
        return await self.inner.relay_aux(dialect, route, body)

    async def close(self) -> None:
        await self.inner.close()


def wrap_client(client: Client, max_tool_output_chars: int) -> Client:
    """Return ``client`` capped at ``max_tool_output_chars`` (``<=0`` disables)."""
    if max_tool_output_chars <= 0:
        return client
    if isinstance(client, TruncatingClient):
        return client
    return TruncatingClient(client, max_tool_output_chars)


__all__ = ["TruncatingClient", "wrap_client"]
