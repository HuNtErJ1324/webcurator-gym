from __future__ import annotations

from types import SimpleNamespace

from pretrain_data_curator.hosted_compat import (
    Environment,
    _branch_completion_len,
    _branch_prompt_len,
)


class _Message:
    def __init__(self, role: str, content: str) -> None:
        self.role = role
        self.content = content

    def model_dump(self, **_: object) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


def _node(
    role: str,
    content: str,
    token_ids: list[int],
    mask: list[bool],
    *,
    sampled: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        message=_Message(role, content),
        token_ids=token_ids,
        mask=mask,
        sampled=sampled,
    )


def _branch(nodes: list[SimpleNamespace]) -> SimpleNamespace:
    token_ids = [token_id for node in nodes for token_id in node.token_ids]
    sampled_mask = [sampled for node in nodes for sampled in node.mask]
    return SimpleNamespace(
        nodes=nodes,
        token_ids=token_ids,
        sampled_mask=sampled_mask,
        logprobs=[0.0] * len(token_ids),
    )


def test_branch_prompt_len_matches_final_turn_prompt_semantics():
    branch = _branch(
        [
            _node("user", "prompt", [1, 2, 3], [False, False, False]),
            _node("assistant", "first", [4, 5], [True, True], sampled=True),
            _node("user", "tool result", [6, 7], [False, False]),
            _node("assistant", "final", [8, 9, 10], [False, True, True], sampled=True),
        ]
    )

    assert not hasattr(branch, "prompt_len")
    assert not hasattr(branch, "completion_len")
    assert _branch_prompt_len(branch) == 8
    assert _branch_completion_len(branch) == 4


def test_state_token_usage_does_not_require_hosted_branch_prompt_len():
    first_branch = _branch(
        [
            _node("user", "one", [1, 2], [False, False]),
            _node("assistant", "two", [3], [True], sampled=True),
        ]
    )
    final_branch = _branch(
        [
            _node("user", "three", [4, 5, 6], [False, False, False]),
            _node("assistant", "four", [7, 8], [True, True], sampled=True),
        ]
    )
    timing_stage = SimpleNamespace(start=None, end=None)
    trace = SimpleNamespace(
        branches=[first_branch, final_branch],
        usage=None,
        assistant_messages=[],
        error=None,
        info={},
        reward=0.0,
        timing=SimpleNamespace(
            start=1.0,
            setup=SimpleNamespace(start=None, end=None),
            generation=SimpleNamespace(start=None, end=None),
            finalize=timing_stage,
            scoring=SimpleNamespace(start=None, end=None),
        ),
        is_completed=True,
        is_truncated=False,
        stop_condition="done",
        metrics={},
    )

    state = Environment._state({"example_id": 0, "info": {}}, trace)

    assert state["token_usage"]["input_tokens"] == 5
    assert state["token_usage"]["output_tokens"] == 3
    assert state["token_usage"]["final_input_tokens"] == 3
    assert state["token_usage"]["final_output_tokens"] == 2
