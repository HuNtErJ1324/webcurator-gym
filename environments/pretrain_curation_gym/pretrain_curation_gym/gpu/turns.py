"""Workspace turn tracking: the agent-visible turn counter script and its state.

The environment writes ``TURN_STATE_FILENAME`` into the runtime workspace at
setup and refreshes it before every model turn (from the ``@vf.stop`` check,
which is the only per-turn hook the framework runs). ``TURN_COUNT_FILENAME`` is
a tiny read-only script the agent can run to see the current turn, the
configured turn budget, and how many turns remain.
"""

from __future__ import annotations

import json

TURN_COUNT_FILENAME = "turn_count.py"
TURN_STATE_FILENAME = ".turn_state.json"

_TURN_COUNT_SCRIPT = '''#!/usr/bin/env python3
"""Show the current rollout turn (the environment refreshes the state file each turn)."""
import json
import sys
from pathlib import Path

STATE_PATH = Path(__file__).resolve().parent / ".turn_state.json"


def main() -> int:
    try:
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        print("turn state unavailable: cannot read %s" % STATE_PATH)
        return 1
    print(
        "turn %s of %s (%s remaining after this one)"
        % (
            state.get("current_turn"),
            state.get("max_turns"),
            state.get("turns_remaining"),
        )
    )
    print(json.dumps(state, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


def render_turn_count_script() -> bytes:
    """The workspace `turn_count.py` the agent runs to see its current turn."""
    return _TURN_COUNT_SCRIPT.encode()


def render_turn_state(turns_completed: int, max_turns: int) -> bytes:
    """JSON state consumed by `turn_count.py`.

    ``turns_completed`` is ``trace.num_turns`` at the pre-turn stop check, so the
    turn the agent is currently acting in is ``turns_completed + 1`` (clamped to
    the budget).
    """
    current = min(turns_completed + 1, max_turns)
    return json.dumps(
        {
            "turns_completed": turns_completed,
            "current_turn": current,
            "max_turns": max_turns,
            "turns_remaining": max(max_turns - current, 0),
        },
        sort_keys=True,
    ).encode()


__all__ = [
    "TURN_COUNT_FILENAME",
    "TURN_STATE_FILENAME",
    "render_turn_count_script",
    "render_turn_state",
]
