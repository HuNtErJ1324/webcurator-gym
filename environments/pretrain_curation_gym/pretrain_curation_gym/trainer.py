"""Proxy-student training: the Perf(M) term.

`ProxyStudentTrainer` is the contract the reward calls. Backends implementing it:

  - `HeuristicProxyTrainer`: deterministic, CPU-only stand-in that predicts
    loss/accuracy from corpus statistics. Used in tests and as the default so the
    environment is usable without GPU.
  - `RuntimeProxyTrainer`: trains a fixed small GPT-2-scale model through the
    live v1 Runtime, with Docker-only OOM diagnostics layered around the common
    launch path.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import weakref
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol

import verifiers.v1 as vf
from pydantic import BaseModel

from .util.container_memory import (
    collect_oom_diagnostics,
    format_oom_diagnostics,
    inspect_container_memory,
    resolve_container_memory_gb,
)
from .corpus import CuratedCorpus
from .util.async_utils import run_blocking_drained
from .util.hf_access import loop_local_semaphore
from .models import ProxyStudentConfig

logger = logging.getLogger(__name__)

# Loop-local bound on concurrent sandbox-training jobs, so a rollout group with
# the real trainer never spawns more GPU sandboxes than configured at once.
_TRAIN_SEMAPHORES: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore]" = weakref.WeakKeyDictionary()


def training_semaphore(limit: int) -> asyncio.Semaphore:
    return loop_local_semaphore(_TRAIN_SEMAPHORES, limit)


class TrainerError(RuntimeError):
    """A surfaced sandbox-training failure, preserving log diagnostics."""

    def __init__(self, message: str, *, stderr_tail: str = "") -> None:
        super().__init__(message)
        self.stderr_tail = stderr_tail


class TrainResult(BaseModel):
    loss: float
    accuracy: float | None
    flops: float
    tokens_trained: int
    backend: str
    # Set when the training run succeeded but post-run sandbox cleanup did not.
    cleanup_error: str | None = None


class ProxyStudentTrainer(Protocol):
    async def train_and_eval(
        self,
        corpus: CuratedCorpus,
        config: ProxyStudentConfig,
        *,
        runtime: vf.Runtime | None = None,
    ) -> TrainResult: ...


def estimate_param_count(config: ProxyStudentConfig) -> int:
    """Exact instantiated parameter count for the modded-nanogpt student.

    ``train_gpt`` model (and therefore ``torch``) is imported lazily so the
    package can load for Hub integration / heuristic scoring without a
    runtime torch dependency. Real GPU training embeds the model source into
    the sandbox script and never needs this import path.
    """
    from .gpu.train_gpt import estimate_instantiated_param_count

    return estimate_instantiated_param_count(
        num_layers=config.n_layer,
        model_dim=config.n_embd,
        num_heads=config.n_head,
        mlp_ratio=config.mlp_ratio,
        softcap=config.lm_head_softcap,
        num_value_embeds=config.num_value_embeds,
        attn_scale=config.attn_scale,
        sliding_window_size=config.sliding_window_size,
    )


def estimate_train_flops(config: ProxyStudentConfig, tokens_trained: int) -> float:
    """Standard 6 * N * D forward+backward FLOP estimate."""
    return 6.0 * estimate_param_count(config) * max(tokens_trained, 0)


class HeuristicProxyTrainer:
    """Deterministic surrogate: lower loss for larger, cleaner, more diverse data.

    This is NOT a trained model; it is a reproducible proxy used when no GPU
    sandbox is available, and as the default backend for fast iteration/tests.

    It does NOT compute a per-token cross-entropy over a held-out token stream, so
    the held-out validation set (the NanoGPT-speedrun FineWeb val tokens) does not
    apply to this backend — it is consumed only by the real (Docker/Modal)
    harness-runtime trainers. Its ``loss`` is a synthetic statistic, not a
    nats/token cross-entropy.
    """

    def __init__(self, reference_loss: float = 5.0) -> None:
        self._reference_loss = reference_loss

    async def train_and_eval(
        self,
        corpus: CuratedCorpus,
        config: ProxyStudentConfig,
        *,
        runtime: vf.Runtime | None = None,
    ) -> TrainResult:
        del runtime
        # The per-document cleanliness/diversity scan is CPU work over the whole
        # corpus; keep it off the event loop.
        return await asyncio.to_thread(self._train_and_eval_sync, corpus, config)

    def _train_and_eval_sync(
        self, corpus: CuratedCorpus, config: ProxyStudentConfig
    ) -> TrainResult:
        if corpus.is_empty():
            # Nothing to train on (e.g. every source failed to fetch); report the
            # same infinite-loss sentinel the sandbox backend uses so perf is 0.
            return TrainResult(
                loss=float("inf"),
                accuracy=0.0,
                flops=0.0,
                tokens_trained=0,
                backend="heuristic",
            )
        tokens = corpus.total_tokens
        # ``effective_train_tokens`` folds in ``train_token_budget`` (steps derived
        # so scheduled presentations under batch_stage_muls meet the budget when
        # set), so a larger budget raises the data the schedule would consume;
        # ``tokens_trained`` is still capped at the corpus's tokens, so the
        # heuristic never bills for data it does not have and stays cheap.
        target_tokens = max(config.effective_train_tokens, 1)
        tokens_trained = min(tokens, target_tokens)

        # Data-scale term: more (effective) tokens -> lower loss, with diminishing
        # returns. Cleanliness and diversity nudge it further down.
        scale = math.log1p(tokens_trained) / math.log1p(target_tokens)
        cleanliness = _avg_cleanliness(corpus)
        diversity = _source_diversity(corpus)
        quality_gain = 0.6 * scale + 0.25 * cleanliness + 0.15 * diversity

        loss = max(0.2, self._reference_loss * (1.0 - 0.85 * quality_gain))
        accuracy = max(0.0, min(1.0, 0.15 + 0.7 * quality_gain))
        flops = estimate_train_flops(config, tokens_trained)
        return TrainResult(
            loss=loss,
            accuracy=accuracy,
            flops=flops,
            tokens_trained=tokens_trained,
            backend="heuristic",
        )


def _avg_cleanliness(corpus: CuratedCorpus) -> float:
    # Streams from disk (`iter_documents()`) rather than materializing the full
    # corpus text, accumulating just a running sum/count of the per-doc ratio.
    total_ratio = 0.0
    count = 0
    for doc in corpus.iter_documents():
        if not doc:
            continue
        alpha = sum(1 for c in doc if c.isalpha() or c.isspace()) / len(doc)
        total_ratio += alpha
        count += 1
    return total_ratio / count if count else 0.0


def _source_diversity(corpus: CuratedCorpus) -> float:
    non_empty = [s for s in corpus.sources if s.doc_count]
    if len(non_empty) <= 1:
        return 0.0
    total = sum(s.tokens for s in non_empty)
    if total <= 0:
        return 0.0
    weights = [s.tokens / total for s in non_empty]
    entropy = -sum(w * math.log(w) for w in weights if w > 0)
    return entropy / math.log(len(non_empty))


TRAIN_GPT_PATH = Path(__file__).resolve().parent / "gpu" / "train_gpt.py"


def _nanogpt_train_script() -> str:
    """Return the single-file trainer copied into the sandbox workspace."""
    return TRAIN_GPT_PATH.read_text(encoding="utf-8")


def __getattr__(name: str) -> Any:
    if name == "NANOGPT_TRAIN_SCRIPT":
        return _nanogpt_train_script()
    raise AttributeError(name)


class RuntimeProxyTrainer:
    """Train through the live v1 runtime selected by the harness config.

    Docker and Modal share one write/run/read implementation. Docker-only
    cgroup and daemon inspection is an optional diagnostic layer around that
    common launch path.
    """

    STREAM_TAIL = 8000
    TRACEBACK_MARKER = "Traceback (most recent call last)"

    def __init__(
        self,
        max_corpus_chars: int | None = None,
        concurrency_limit: int = 1,
        val_loader: Any = None,
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
        runtime_type = getattr(runtime, "type", None)
        if runtime is None or runtime_type not in {"docker", "modal"}:
            raise TrainerError(
                "real proxy-student training requires a live Docker or Modal "
                f"harness runtime, got {runtime_type!r}"
            )

        cap = (
            self._max_corpus_chars
            if self._max_corpus_chars is not None
            else config.effective_max_corpus_chars
        )
        # Joining the on-disk corpus is a blocking filesystem pass. Yield here
        # so Decon can begin its independent read while this input is prepared,
        # and drain the worker before a cancelled rollout can clean up the files.
        try:
            text = await run_blocking_drained(corpus.joined_text, cap)
        except BaseException:
            await self._stop_cancel_safe(runtime)
            raise
        if not text.strip():
            return TrainResult(
                loss=float("inf"),
                accuracy=0.0,
                flops=0.0,
                tokens_trained=0,
                backend=runtime_type,
            )

        val_set = await self._resolve_val_set()
        payload = config.training_payload(
            tokenizer=val_set.tokenizer if val_set else "gpt2"
        )
        async with training_semaphore(self._concurrency_limit):
            try:
                await self._write_inputs(runtime, text, payload, config, val_set)
                self._raise_if_cancelling()
                result = await self._run_training(runtime, config)
                self._raise_if_cancelling()
                return self._parse_result(
                    result.stdout, result.stderr, backend=runtime_type
                )
            except BaseException:
                await self._stop_cancel_safe(runtime)
                raise

    @staticmethod
    def _raise_if_cancelling() -> None:
        task = asyncio.current_task()
        if task is not None and task.cancelling():
            raise asyncio.CancelledError()

    async def _resolve_val_set(self) -> Any:
        if self._val_loader is None:
            return None
        return await self._val_loader.load()

    async def _write_inputs(
        self,
        runtime: vf.Runtime,
        text: str,
        payload: dict[str, Any],
        config: ProxyStudentConfig,
        val_set: Any,
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
                    runtime.write(path, data), timeout=config.upload_timeout_seconds
                )
                self._raise_if_cancelling()
            except (asyncio.TimeoutError, TimeoutError) as exc:
                raise TrainerError(
                    f"timed out writing proxy-student input {path!r}"
                ) from exc

    async def _run_training(
        self, runtime: vf.Runtime, config: ProxyStudentConfig
    ) -> Any:
        docker = runtime.type == "docker"
        before = await self._read_cgroup(runtime, config) if docker else None
        timeout = config.effective_timeout_minutes * 60
        try:
            result = await asyncio.wait_for(
                runtime.run(["python", "/workspace/train.py"], {}), timeout=timeout
            )
            self._raise_if_cancelling()
        except (asyncio.TimeoutError, TimeoutError) as exc:
            after = await self._read_cgroup(runtime, config) if docker else None
            raise TrainerError(
                f"proxy-student training timed out after {timeout}s",
                stderr_tail=await self._training_diagnostic_async(
                    "",
                    "",
                    runtime=runtime,
                    config=config,
                    events_before=before,
                    events_after=after,
                    timed_out=True,
                ),
            ) from exc

        redirected = (
            await self._read_redirected_stderr(runtime, config) if docker else ""
        )
        stderr = self._merge_stderr(getattr(result, "stderr", None), redirected)
        after = await self._read_cgroup(runtime, config) if docker else None
        await self._persist_logs(
            runtime,
            result,
            config,
            redirected=redirected,
            stderr=stderr,
            events_before=before,
            events_after=after,
        )
        if result.exit_code not in (0, None):
            raise TrainerError(
                f"proxy-student training exited with code {result.exit_code}",
                stderr_tail=await self._training_diagnostic_async(
                    getattr(result, "stdout", ""),
                    stderr,
                    runtime=runtime,
                    config=config,
                    events_before=before,
                    events_after=after,
                    returncode=result.exit_code,
                ),
            )
        return SimpleNamespace(
            stdout=getattr(result, "stdout", "") or "",
            stderr=stderr,
            exit_code=getattr(result, "exit_code", None),
        )

    async def _read_cgroup(
        self, runtime: vf.Runtime, config: ProxyStudentConfig
    ) -> dict[str, Any]:
        script = (
            "from pathlib import Path\n"
            "import json\n"
            "events=Path('/sys/fs/cgroup/memory.events')\n"
            "mmax=Path('/sys/fs/cgroup/memory.max')\n"
            "payload={'events':{}, 'memory_max': None, 'error': None}\n"
            "try:\n"
            "    if events.is_file():\n"
            "        for line in events.read_text().splitlines():\n"
            "            parts=line.split()\n"
            "            if len(parts)==2 and parts[1].lstrip('-').isdigit():\n"
            "                payload['events'][parts[0]]=int(parts[1])\n"
            "    if mmax.is_file():\n"
            "        raw=mmax.read_text().strip()\n"
            "        payload['memory_max']=None if raw=='max' else int(raw)\n"
            "except Exception as exc:\n"
            "    payload['error']=str(exc)\n"
            "print('CGROUP_JSON '+json.dumps(payload, sort_keys=True))\n"
        )
        try:
            result = await asyncio.wait_for(
                runtime.run(["python", "-c", script], {}),
                timeout=min(30.0, config.upload_timeout_seconds),
            )
        except Exception as exc:  # diagnostics must not raise
            return {"events": {}, "memory_max": None, "error": str(exc)}
        for line in reversed((getattr(result, "stdout", "") or "").splitlines()):
            if line.startswith("CGROUP_JSON "):
                try:
                    return json.loads(line[len("CGROUP_JSON ") :])
                except json.JSONDecodeError:
                    break
        return {"events": {}, "memory_max": None, "error": "missing CGROUP_JSON"}

    async def _read_redirected_stderr(
        self, runtime: vf.Runtime, config: ProxyStudentConfig
    ) -> str:
        try:
            data = await asyncio.wait_for(
                runtime.read("/workspace/stderr.txt"),
                timeout=config.upload_timeout_seconds,
            )
        except Exception:
            return ""
        return (data or b"").decode("utf-8", errors="replace")

    @staticmethod
    def _merge_stderr(captured: str | None, redirected: str | None) -> str:
        captured = (captured or "").strip()
        redirected = (redirected or "").strip()
        if (
            captured
            and redirected
            and captured not in redirected
            and redirected not in captured
        ):
            return f"{captured}\n--- redirected stderr.txt ---\n{redirected}"
        return redirected or captured

    async def _persist_logs(
        self,
        runtime: vf.Runtime,
        result: Any,
        config: ProxyStudentConfig,
        *,
        redirected: str,
        stderr: str,
        events_before: dict[str, Any] | None,
        events_after: dict[str, Any] | None,
    ) -> None:
        logs = {
            "/workspace/train_stdout.log": getattr(result, "stdout", "") or "",
            "/workspace/train_stderr.log": stderr,
        }
        if runtime.type == "docker":
            logs["/workspace/train_stderr_redirect.log"] = redirected
            diagnostics = await self._oom_diagnostics_async(
                runtime,
                config,
                events_before=events_before,
                events_after=events_after,
                returncode=getattr(result, "exit_code", None),
                stderr=stderr,
            )
            logs["/workspace/train_oom_diagnostics.json"] = json.dumps(
                diagnostics, sort_keys=True, default=str
            )
        try:
            for path, text in logs.items():
                await asyncio.wait_for(
                    runtime.write(path, text.encode("utf-8", errors="replace")),
                    timeout=config.upload_timeout_seconds,
                )
        except Exception:
            logger.warning(
                "failed to persist %s training logs", runtime.type, exc_info=True
            )

    def _parse_result(self, stdout: str, stderr: str, *, backend: str) -> TrainResult:
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
                backend=backend,
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise TrainerError(
                f"proxy-student training produced malformed RESULT_JSON: {exc}",
                stderr_tail=self._training_diagnostic(stdout, stderr),
            ) from exc

    def _training_diagnostic(self, stdout: str, stderr: str, **kwargs: Any) -> str:
        parts = [
            "--- stdout tail ---",
            self._diagnostic_stream(stdout or ""),
            "--- stderr tail ---",
            self._diagnostic_stream(stderr or ""),
        ]
        runtime = kwargs.get("runtime")
        if runtime is not None and runtime.type == "docker":
            parts.append(
                kwargs.get("memory_diagnostic")
                or self._memory_diagnostic(
                    runtime,
                    kwargs.get("config"),
                    events_before=kwargs.get("events_before"),
                    events_after=kwargs.get("events_after"),
                    returncode=kwargs.get("returncode"),
                    stderr=stderr,
                    timed_out=bool(kwargs.get("timed_out")),
                )
            )
        return "\n".join(parts)

    async def _training_diagnostic_async(
        self, stdout: str, stderr: str, **kwargs: Any
    ) -> str:
        runtime = kwargs.get("runtime")
        if runtime is not None and runtime.type == "docker":
            kwargs["memory_diagnostic"] = await self._memory_diagnostic_async(
                runtime,
                kwargs.get("config"),
                events_before=kwargs.get("events_before"),
                events_after=kwargs.get("events_after"),
                returncode=kwargs.get("returncode"),
                stderr=stderr,
                timed_out=bool(kwargs.get("timed_out")),
            )
        return self._training_diagnostic(stdout, stderr, **kwargs)

    def _diagnostic_stream(self, text: str) -> str:
        marker_at = text.find(self.TRACEBACK_MARKER)
        return text[marker_at:] if marker_at >= 0 else text[-self.STREAM_TAIL :]

    def _oom_diagnostics(
        self,
        runtime: vf.Runtime,
        config: ProxyStudentConfig | None,
        *,
        events_before: dict[str, Any] | None = None,
        events_after: dict[str, Any] | None = None,
        returncode: int | None = None,
        stderr: str | None = None,
        timed_out: bool = False,
        inspect_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        container = getattr(runtime, "_container", None) or getattr(
            runtime, "name", None
        )
        if container and inspect_info is None:
            try:
                inspect_info = inspect_container_memory(str(container))
            except Exception as exc:
                logger.debug("oom inspect failed: %s", exc)
        return collect_oom_diagnostics(
            configured_gb=(
                resolve_container_memory_gb(config.memory_gb, backend="docker")
                if config is not None
                else None
            ),
            effective_memory_bytes=(inspect_info or {}).get("memory_bytes"),
            oom_killed=(inspect_info or {}).get("oom_killed"),
            container=str(container) if container else None,
            events_before=events_before,
            events_after=events_after,
            returncode=returncode,
            stderr=stderr,
            timed_out=timed_out,
        )

    async def _oom_diagnostics_async(
        self,
        runtime: vf.Runtime,
        config: ProxyStudentConfig | None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        container = getattr(runtime, "_container", None) or getattr(runtime, "name", "")
        try:
            inspect_info = await asyncio.to_thread(
                inspect_container_memory, str(container)
            )
        except Exception:
            inspect_info = None
        return self._oom_diagnostics(
            runtime, config, inspect_info=inspect_info, **kwargs
        )

    def _memory_diagnostic(
        self, runtime: vf.Runtime, config: ProxyStudentConfig | None, **kwargs: Any
    ) -> str:
        return format_oom_diagnostics(self._oom_diagnostics(runtime, config, **kwargs))

    async def _memory_diagnostic_async(
        self, runtime: vf.Runtime, config: ProxyStudentConfig | None, **kwargs: Any
    ) -> str:
        return format_oom_diagnostics(
            await self._oom_diagnostics_async(runtime, config, **kwargs)
        )

    @staticmethod
    async def _stop_cancel_safe(runtime: vf.Runtime) -> None:
        teardown = asyncio.create_task(runtime.stop())
        try:
            await asyncio.shield(teardown)
        except asyncio.CancelledError:
            await asyncio.shield(teardown)
            raise
        except Exception:
            logger.warning("runtime teardown failed", exc_info=True)


__all__ = [
    "HeuristicProxyTrainer",
    "ProxyStudentTrainer",
    "RuntimeProxyTrainer",
    "TrainResult",
    "TrainerError",
    "estimate_param_count",
    "estimate_train_flops",
]
