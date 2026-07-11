"""Proxy-student training inside the rollout's declarative Docker runtime.

The v1 rollout owns exactly one runtime for its full lifecycle.  When the
``docker`` trainer backend is selected, that runtime is also the training
sandbox: this trainer writes the corpus and script through :class:`vf.Runtime`
and executes training on the same live container.  Runtime provisioning and
normal teardown remain the rollout's responsibility.
"""

from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace
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
from .val_set import HeldOutValSet, ValTokenLoader

logger = logging.getLogger(__name__)


class HarnessRuntimeProxyTrainer:
    """Train on the live Docker runtime supplied to taskset scoring."""

    STREAM_TAIL = 8000
    TRACEBACK_MARKER = "Traceback (most recent call last)"

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
                "docker proxy-student training requires the live harness runtime"
            )
        if runtime.type != "docker":
            raise TrainerError(
                f"docker proxy-student training requires a Docker runtime, got "
                f"{runtime.type!r}"
            )

        cap = (
            self._max_corpus_chars
            if self._max_corpus_chars is not None
            else config.effective_max_corpus_chars
        )
        # Streams docs off disk and stops once `cap` is reached, instead of
        # joining the (potentially huge) full corpus into memory first (see
        # `CuratedCorpus.joined_text`).
        text = corpus.joined_text(cap)
        if not text.strip():
            return TrainResult(
                loss=float("inf"),
                accuracy=0.0,
                flops=0.0,
                tokens_trained=0,
                backend="docker",
            )

        val_set = await self._resolve_val_set()
        payload = self._payload(config, val_set)

        # The rollout may provision more runtime containers than there are GPUs.
        # Bound the expensive commands independently of rollout concurrency.
        async with training_semaphore(self._concurrency_limit):
            try:
                await self._write_inputs(runtime, text, payload, config, val_set)
                # asyncio.wait_for() (pre-3.12) can race: a cancellation delivered
                # while the wrapped write is *also* completing gets silently
                # absorbed by wait_for's `if fut.done(): return fut.result()`
                # shortcut instead of propagating. That leaves the task's
                # cancellation request pending but undelivered, so re-raise it
                # explicitly here rather than falling through into the
                # potentially-unbounded `_run_training` await with no cancellation
                # left to interrupt it.
                self._raise_if_cancelling()
                result = await self._run_training(runtime, config)
                self._raise_if_cancelling()
                return self._parse_result(result.stdout, result.stderr)
            except BaseException:
                # A failed/cancelled docker exec may leave the process running in
                # the container. Stop the runtime now; Rollout.run() also stops it
                # in its finally, and DockerRuntime.stop() is idempotent.
                await self._stop_cancel_safe(runtime)
                raise

    @staticmethod
    def _raise_if_cancelling() -> None:
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
        payload = config.training_payload(
            tokenizer=val_set.tokenizer if val_set else "gpt2"
        )
        return payload

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
        except (asyncio.TimeoutError, TimeoutError) as exc:
            raise TrainerError(
                f"proxy-student training timed out after {timeout}s"
            ) from exc
        file_stderr = await self._read_redirected_stderr(runtime, config)
        merged_stderr = self._merge_stderr(getattr(result, "stderr", None), file_stderr)
        await self._persist_training_logs(
            runtime, result, config, file_stderr=file_stderr, merged_stderr=merged_stderr
        )
        if result.exit_code not in (0, None):
            raise TrainerError(
                f"proxy-student training exited with code {result.exit_code}",
                stderr_tail=self._training_diagnostic(result.stdout, merged_stderr),
            )
        return SimpleNamespace(
            stdout=getattr(result, "stdout", "") or "",
            stderr=merged_stderr,
            exit_code=getattr(result, "exit_code", None),
        )

    async def _read_redirected_stderr(
        self, runtime: vf.Runtime, config: ProxyStudentConfig
    ) -> str:
        """Read ``/workspace/stderr.txt`` written by the trainer's line-buffered redirect."""
        try:
            data = await asyncio.wait_for(
                runtime.read("/workspace/stderr.txt"),
                timeout=config.upload_timeout_seconds,
            )
        except Exception:
            return ""
        if data is None:
            return ""
        if isinstance(data, bytes):
            return data.decode("utf-8", errors="replace")
        return str(data)

    @staticmethod
    def _merge_stderr(captured: str | None, file_stderr: str | None) -> str:
        captured = (captured or "").strip()
        file_stderr = (file_stderr or "").strip()
        if file_stderr and captured:
            if file_stderr in captured:
                return captured
            if captured in file_stderr:
                return file_stderr
            return f"{captured}\n--- redirected stderr.txt ---\n{file_stderr}"
        return file_stderr or captured

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
                stderr_tail=self._training_diagnostic(stdout, stderr),
            )
        try:
            data = json.loads(marker)
            return TrainResult(
                loss=float(data["loss"]),
                accuracy=float(data.get("accuracy", 0.0)),
                flops=float(data.get("flops", 0.0)),
                tokens_trained=int(data.get("tokens_trained", 0)),
                backend="docker",
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise TrainerError(
                f"proxy-student training produced malformed RESULT_JSON: {exc}",
                stderr_tail=self._training_diagnostic(stdout, stderr),
            ) from exc

    async def _persist_training_logs(
        self,
        runtime: vf.Runtime,
        result: Any,
        config: ProxyStudentConfig,
        *,
        file_stderr: str = "",
        merged_stderr: str = "",
    ) -> None:
        logs = {
            "/workspace/train_stdout.log": (getattr(result, "stdout", "") or ""),
            "/workspace/train_stderr.log": merged_stderr
            or (getattr(result, "stderr", "") or ""),
            "/workspace/train_stderr_redirect.log": file_stderr,
        }
        try:
            for path, text in logs.items():
                await asyncio.wait_for(
                    runtime.write(path, text.encode("utf-8", errors="replace")),
                    timeout=config.upload_timeout_seconds,
                )
        except Exception:
            logger.warning("failed to persist docker training logs", exc_info=True)

    def _training_diagnostic(self, stdout: str | None, stderr: str | None) -> str:
        return "\n".join(
            [
                "--- stdout tail ---",
                self._diagnostic_stream(stdout or ""),
                "--- stderr tail ---",
                self._diagnostic_stream(stderr or ""),
            ]
        )

    def _diagnostic_stream(self, text: str) -> str:
        marker_at = text.find(self.TRACEBACK_MARKER)
        if marker_at >= 0:
            return text[marker_at:]
        return text[-self.STREAM_TAIL :]

    @staticmethod
    async def _stop_cancel_safe(runtime: vf.Runtime) -> None:
        """Finish teardown even when the caller is already being cancelled."""
        teardown = asyncio.create_task(runtime.stop())
        try:
            await asyncio.shield(teardown)
        except asyncio.CancelledError:
            # Shield keeps teardown alive. Wait for it before propagating the
            # cancellation so the Docker container cannot outlive the rollout.
            await asyncio.shield(teardown)
            raise
        except Exception:
            logger.warning("docker harness runtime teardown failed", exc_info=True)
