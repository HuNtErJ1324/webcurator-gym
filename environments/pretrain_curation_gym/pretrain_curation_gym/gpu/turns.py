"""Render the agent-callable turn counter installed in each runtime workspace."""

from __future__ import annotations

import json

TURNS_FILENAME = "turns.py"
TURN_STATE_FILENAME = ".turn_state.json"

_TURNS_SCRIPT = '''#!/usr/bin/env python3
"""Print the current rollout turn and remaining framework turn budget."""
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
            state["current_turn"],
            state["max_turns"],
            state["turns_remaining"],
        )
    )
    print(json.dumps(state, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


def render_turns_script() -> bytes:
    return _TURNS_SCRIPT.encode()


def render_turn_state(turns_completed: int, max_turns: int) -> bytes:
    """Render the pre-turn view consumed by ``turns.py``."""
    current_turn = min(turns_completed + 1, max_turns)
    return json.dumps(
        {
            "turns_completed": turns_completed,
            "current_turn": current_turn,
            "max_turns": max_turns,
            "turns_remaining": max(max_turns - current_turn, 0),
        },
        sort_keys=True,
    ).encode()


__all__ = [
    "TURNS_FILENAME",
    "TURN_STATE_FILENAME",
    "render_turn_state",
    "render_turns_script",
]
