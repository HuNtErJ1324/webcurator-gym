"""Proxy-student training inside the rollout's declarative Modal runtime.

The v1 rollout owns exactly one runtime for its full lifecycle. When the
``modal`` trainer backend is selected, that runtime is also the training
sandbox: this trainer writes the corpus and script through :class:`vf.Runtime`
and executes training on the same live Modal sandbox. Runtime provisioning and
normal teardown remain the rollout's responsibility.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import verifiers.v1 as vf

from .corpus import CuratedCorpus
from .models import ProxyStudentConfig
from .trainer import (
    TrainResult,
    TrainerError,
    _nanogpt_train_script,
    training_semaphore,
)
from .val_set import NANOGPT_VAL_TOKENIZER, HeldOutValSet, ValTokenLoader

logger = logging.getLogger(__name__)

# Maps ProxyStudentConfig.modal_gpu to Modal's GPU specifier. Unknown values
# preserve the historical L4 fallback.
_MODAL_GPU_MAP: dict[str, str] = {
    "H100": "H100",
    "H200": "H200",
    "A100": "A100-80GB",
}
_DEFAULT_MODAL_GPU = "L4"


def _modal_gpu_for(modal_gpu: str) -> str:
    return _MODAL_GPU_MAP.get(modal_gpu, _DEFAULT_MODAL_GPU)


class ModalProxyTrainer:
    """Train on the live Modal runtime supplied to taskset scoring."""

    STDERR_TAIL = 2000

    def __init__(
        self,
        max_corpus_chars: int | None = None,
        concurrency_limit: int = 1,
        val_loader: ValTokenLoader | None = None,
    ) -> None:
        self._max_corpus_chars = max_corpus_chars
        self._concurrency_limit = concurrency_limit
        self._val_loader = val_loader

    async def train_and_eval(
        self,
        corpus: CuratedCorpus,
        config: ProxyStudentConfig,
        *,
        runtime: vf.Runtime | None = None,
    ) -> TrainResult:
        if runtime is None:
            raise TrainerError(
                "modal proxy-student training requires the live harness runtime"
            )
        if runtime.type != "modal":
            raise TrainerError(
                f"modal proxy-student training requires a Modal runtime, got "
                f"{runtime.type!r}"
            )

        cap = (
            self._max_corpus_chars
            if self._max_corpus_chars is not None
            else config.effective_max_corpus_chars
        )
        text = "\n\n".join(corpus.documents)[:cap]
        if not text.strip():
            return TrainResult(
                loss=float("inf"),
                accuracy=0.0,
                flops=0.0,
                tokens_trained=0,
                backend="modal",
            )

        val_set = await self._resolve_val_set()
        payload = self._payload(config, val_set)

        # Rollout concurrency can exceed the intended number of simultaneous
        # paid GPU training jobs.
        async with training_semaphore(self._concurrency_limit):
            try:
                await self._write_inputs(runtime, text, payload, config, val_set)
                self._raise_if_cancelling()
                result = await self._run_training(runtime, config)
                self._raise_if_cancelling()
                return self._parse_result(result.stdout, result.stderr)
            except BaseException:
                # A failed or cancelled exec can continue consuming a Modal GPU.
                # Stop immediately; Rollout.run() also calls the idempotent stop.
                await self._stop_cancel_safe(runtime)
                raise

    @staticmethod
    def _raise_if_cancelling() -> None:
        """Recover cancellation swallowed by pre-3.12 asyncio.wait_for races."""
        task = asyncio.current_task()
        if task is not None and task.cancelling():
            raise asyncio.CancelledError()

    async def _resolve_val_set(self) -> HeldOutValSet | None:
        if self._val_loader is None:
            return None
        return await self._val_loader.load()

    @staticmethod
    def _payload(
        config: ProxyStudentConfig, val_set: HeldOutValSet | None
    ) -> dict[str, Any]:
        return {
            "n_layer": config.n_layer,
            "n_head": config.n_head,
            "n_embd": config.n_embd,
            "mlp_ratio": config.mlp_ratio,
            "lm_head_softcap": config.lm_head_softcap,
            "num_value_embeds": config.num_value_embeds,
            "block_size": config.block_size,
            "batch_size": config.batch_size,
            "steps": config.effective_steps,
            "learning_rate": config.learning_rate,
            "seed": config.seed,
            "val_fraction": config.val_fraction,
            "tokenizer": val_set.tokenizer if val_set else NANOGPT_VAL_TOKENIZER,
            "weight_decay": config.weight_decay,
            "adam_beta1": config.adam_beta1,
            "adam_beta2": config.adam_beta2,
            "adam_eps": config.adam_eps,
            "grad_clip": config.grad_clip,
            "warmup_steps": config.effective_warmup_steps,
            "lr_min_ratio": config.lr_min_ratio,
            "n_train_runs": config.n_train_runs,
        }

    async def _write_inputs(
        self,
        runtime: vf.Runtime,
        text: str,
        payload: dict[str, Any],
        config: ProxyStudentConfig,
        val_set: HeldOutValSet | None,
    ) -> None:
        files = [
            ("/workspace/corpus.txt", text.encode("utf-8")),
            ("/workspace/config.json", json.dumps(payload).encode("utf-8")),
            ("/workspace/train.py", _nanogpt_train_script().encode("utf-8")),
        ]
        if val_set is not None:
            files.append(("/workspace/val.bin", val_set.to_uint16_bytes()))
        for path, data in files:
            try:
                await asyncio.wait_for(
                    runtime.write(path, data),
                    timeout=config.upload_timeout_seconds,
                )
                # A cancellation delivered as write completes can be swallowed
                # by asyncio.wait_for on Python <3.12.
                self._raise_if_cancelling()
            except (asyncio.TimeoutError, TimeoutError) as exc:
                raise TrainerError(
                    f"timed out writing proxy-student input {path!r}"
                ) from exc

    async def _run_training(
        self, runtime: vf.Runtime, config: ProxyStudentConfig
    ) -> Any:
        timeout = config.effective_timeout_minutes * 60
        try:
            result = await asyncio.wait_for(
                runtime.run(["python", "/workspace/train.py"], {}),
                timeout=timeout,
            )
            # Apply the same pre-3.12 wait_for cancellation-race guard before
            # inspecting a result or returning to the caller.
            self._raise_if_cancelling()
        except (asyncio.TimeoutError, TimeoutError) as exc:
            raise TrainerError(
                f"proxy-student training timed out after {timeout}s"
            ) from exc
        if result.exit_code not in (0, None):
            raise TrainerError(
                f"proxy-student training exited with code {result.exit_code}",
                stderr_tail=(result.stderr or "")[-self.STDERR_TAIL :],
            )
        return result

    def _parse_result(self, stdout: str, stderr: str) -> TrainResult:
        marker = next(
            (
                line[len("RESULT_JSON ") :]
                for line in reversed((stdout or "").splitlines())
                if line.startswith("RESULT_JSON ")
            ),
            None,
        )
        if marker is None:
            raise TrainerError(
                "proxy-student training produced no RESULT_JSON marker",
                stderr_tail=(stderr or "")[-self.STDERR_TAIL :],
            )
        try:
            data = json.loads(marker)
            return TrainResult(
                loss=float(data["loss"]),
                accuracy=float(data.get("accuracy", 0.0)),
                flops=float(data.get("flops", 0.0)),
                tokens_trained=int(data.get("tokens_trained", 0)),
                backend="modal",
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise TrainerError(
                f"proxy-student training produced malformed RESULT_JSON: {exc}",
                stderr_tail=(stderr or "")[-self.STDERR_TAIL :],
            ) from exc

    @staticmethod
    async def _stop_cancel_safe(runtime: vf.Runtime) -> None:
        """Finish paid-resource teardown even while the caller is cancelled."""
        teardown = asyncio.create_task(runtime.stop())
        try:
            await asyncio.shield(teardown)
        except asyncio.CancelledError:
            await asyncio.shield(teardown)
            raise
        except Exception:
            logger.warning("modal harness runtime teardown failed", exc_info=True)
