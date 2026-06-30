"""Legacy Hosted Training bridge for the native verifiers v1 environment."""

from __future__ import annotations

from typing import Any

import verifiers as legacy_vf
import verifiers.v1 as vf
from datasets import Dataset
from verifiers.v1.clients.config import EvalClientConfig, TrainClientConfig


def _branch_prompt_len(branch: Any) -> int:
    """Final-turn prompt length without relying on Branch convenience properties."""
    last_completion = next(
        (sum(node.mask) for node in reversed(branch.nodes) if any(node.mask)),
        0,
    )
    return len(branch.token_ids) - last_completion


def _branch_completion_len(branch: Any) -> int:
    """Completion length without relying on Branch convenience properties."""
    return sum(sum(node.mask) for node in branch.nodes)


class Environment(vf.Environment, legacy_vf.Environment):
    """Run v1 episodes behind the v0 ``Environment`` API used by Prime CLI 0.6.15."""

    def __init__(self, config: vf.EnvConfig, env_args: dict[str, Any]) -> None:
        vf.Environment.__init__(self, config)
        self._compat_tasks = self.taskset.load_tasks()
        self._compat_clients: dict[str, vf.Client] = {}
        rows = [
            {
                "example_id": index,
                "prompt": [
                    {
                        "role": "user",
                        "content": "Complete the pretraining-data curation task.",
                    }
                ],
            }
            for index in range(len(self._compat_tasks))
        ]
        dataset = Dataset.from_list(rows)
        legacy_vf.Environment.__init__(
            self,
            dataset=dataset,
            eval_dataset=dataset,
            env_id=config.taskset.id,
            env_args=env_args,
            score_rollouts=False,
        )

    @staticmethod
    def _message(message: Any) -> dict[str, Any]:
        return message.model_dump(mode="json", exclude_none=True)

    def _task(self, input: legacy_vf.RolloutInput) -> vf.Task:
        index = int(input["example_id"])
        try:
            return self._compat_tasks[index]
        except IndexError as exc:
            raise ValueError(f"unknown task example_id {index}") from exc

    def _client(self, client: Any) -> vf.Client:
        config = getattr(client, "config", None)
        if config is None:
            raise TypeError(
                "the Hosted Training compatibility bridge requires a configured client"
            )
        common = {
            "base_url": config.api_base_url,
            "api_key_var": config.api_key_var,
            "headers": dict(config.extra_headers),
        }
        if config.client_type == "renderer":
            v1_config = TrainClientConfig(
                **common,
                renderer=config.renderer_config,
                pool_size=config.renderer_pool_size or 1,
                renderer_model_name=config.renderer_model_name,
            )
        else:
            v1_config = EvalClientConfig(**common)
        key = v1_config.model_dump_json()
        if key not in self._compat_clients:
            self._compat_clients[key] = vf.resolve_client(v1_config)
        return self._compat_clients[key]

    @staticmethod
    def _sampling(sampling_args: dict[str, Any] | None) -> vf.SamplingConfig:
        allowed = vf.SamplingConfig.model_fields
        return vf.SamplingConfig.model_validate(
            {
                key: value
                for key, value in (sampling_args or {}).items()
                if key in allowed
            }
        )

    @staticmethod
    def _timing(trace: vf.Trace) -> legacy_vf.RolloutTiming:
        timing = legacy_vf.RolloutTiming(start_time=trace.timing.start)
        timing.setup.start = trace.timing.setup.start
        timing.setup.end = trace.timing.setup.end
        timing.generation.start = trace.timing.generation.start
        timing.generation.end = trace.timing.generation.end
        timing.scoring.start = (
            trace.timing.finalize.start or trace.timing.scoring.start
        )
        timing.scoring.end = trace.timing.scoring.end
        return timing

    @classmethod
    def _trajectory(cls, trace: vf.Trace) -> list[dict[str, Any]]:
        steps: list[dict[str, Any]] = []
        for branch in trace.branches:
            token_ids = branch.token_ids
            sampled_mask = branch.sampled_mask
            logprobs = branch.logprobs
            split = next(
                (index for index, sampled in enumerate(sampled_mask) if sampled),
                len(token_ids),
            )
            steps.append(
                {
                    "prompt": [
                        cls._message(node.message)
                        for node in branch.nodes
                        if not node.sampled
                    ],
                    "completion": [
                        cls._message(node.message)
                        for node in branch.nodes
                        if node.sampled
                    ],
                    "tokens": {
                        "prompt_ids": token_ids[:split],
                        "prompt_mask": [False] * split,
                        "completion_ids": token_ids[split:],
                        "completion_mask": sampled_mask[split:],
                        "completion_logprobs": logprobs[split:],
                        "routed_experts": None,
                        "multi_modal_data": None,
                    },
                }
            )
        return steps

    @classmethod
    def _state(
        cls, input: legacy_vf.RolloutInput, trace: vf.Trace
    ) -> legacy_vf.State:
        error = trace.error
        branches = trace.branches
        final_branch = branches[-1] if branches else None
        usage = trace.usage
        assistant_messages = [
            cls._message(message) for message in trace.assistant_messages
        ]
        state: legacy_vf.State = {
            "example_id": int(input["example_id"]),
            "prompt": input.get("prompt"),
            "completion": assistant_messages,
            "answer": input.get("answer", ""),
            "info": {**dict(input.get("info") or {}), **trace.info},
            "reward": trace.reward,
            "error": (
                None
                if error is None
                else {
                    "error": error.type,
                    "message": error.message,
                    "error_chain_repr": f"{error.type}: {error.message}",
                    "error_chain_str": f"{error.type}: {error.message}",
                }
            ),
            "timing": cls._timing(trace),
            "is_completed": trace.is_completed,
            "is_truncated": trace.is_truncated,
            "stop_condition": trace.stop_condition,
            "metrics": dict(trace.metrics),
            "tool_defs": [],
            "trajectory": cls._trajectory(trace),
            "token_usage": {
                "input_tokens": (
                    usage.input_tokens
                    if usage is not None
                    else sum(_branch_prompt_len(branch) for branch in branches)
                ),
                "output_tokens": (
                    usage.completion_tokens
                    if usage is not None
                    else sum(_branch_completion_len(branch) for branch in branches)
                ),
                "final_input_tokens": (
                    _branch_prompt_len(final_branch)
                    if final_branch is not None
                    else 0
                ),
                "final_output_tokens": (
                    _branch_completion_len(final_branch)
                    if final_branch is not None
                    else 0
                ),
            },
        }
        return state

    async def rollout(
        self,
        input: legacy_vf.RolloutInput,
        client: Any,
        model: str,
        sampling_args: dict[str, Any] | None = None,
    ) -> legacy_vf.State:
        context = vf.RolloutContext(
            model=model,
            client=self._client(client),
            sampling=self._sampling(sampling_args),
        )
        trace = (await self.episode(self._task(input), context).run())[0]
        return self._state(input, trace)

    async def _run_rollout_state(
        self,
        input: legacy_vf.RolloutInput,
        client: Any,
        model: str,
        sampling_args: dict[str, Any],
    ) -> legacy_vf.State:
        return await self.rollout(input, client, model, sampling_args)

    async def _run_group_states(
        self,
        group_inputs: list[legacy_vf.RolloutInput],
        client: Any,
        model: str,
        sampling_args: dict[str, Any],
    ) -> list[legacy_vf.State]:
        if not group_inputs:
            return []
        context = vf.RolloutContext(
            model=model,
            client=self._client(client),
            sampling=self._sampling(sampling_args),
        )
        traces = await self.episode(
            self._task(group_inputs[0]), context, n=len(group_inputs)
        ).run()
        return [
            self._state(input, trace)
            for input, trace in zip(group_inputs, traces, strict=True)
        ]


__all__ = ["Environment"]
