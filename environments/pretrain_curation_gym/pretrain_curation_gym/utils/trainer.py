"""Proxy-student training: the Perf(M) term."""

from __future__ import annotations

import asyncio
import json
import logging
import math
from pathlib import Path
from typing import Any, Protocol

import verifiers.v1 as vf
from pydantic import BaseModel

from .corpus import CuratedCorpus
from .async_utils import run_blocking_drained, run_shielded, training_semaphore
from .models import ProxyStudentConfig

logger = logging.getLogger(__name__)

TRAIN_GPT_PATH = Path(__file__).resolve().parent.parent / "gpu" / "train_gpt.py"
TRAIN_GPT_SOURCE = TRAIN_GPT_PATH.read_text(encoding="utf-8")


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
    """Exact instantiated parameter count for the modded-nanogpt student."""
    from ..gpu.train_gpt import estimate_instantiated_param_count

    arch = config.arch
    features = config.features
    return estimate_instantiated_param_count(
        num_layers=arch.n_layer,
        model_dim=arch.n_embd,
        num_heads=arch.n_head,
        mlp_ratio=arch.mlp_ratio,
        softcap=arch.lm_head_softcap,
        num_value_embeds=arch.num_value_embeds,
        attn_scale=arch.attn_scale,
        sliding_window_size=arch.sliding_window_size,
        bigram_hash_embed=features.bigram_hash_embed,
        smear_embed=features.smear_embed,
        partial_key_offset=features.partial_key_offset,
        paired_head=features.paired_head,
        mudd_pairs=features.mudd_pairs,
        xsa_enabled=features.xsa_enabled,
        xsa_pairs=features.xsa_pairs,
        single_act_last_k=features.single_act_last_k,
        exp_residual_decay=features.exp_residual_decay,
        multi_token_pred=features.multi_token_pred,
    )


def estimate_train_flops(config: ProxyStudentConfig, tokens_trained: int) -> float:
    """Standard 6 * N * D forward+backward FLOP estimate."""
    return 6.0 * estimate_param_count(config) * max(tokens_trained, 0)


class HeuristicProxyTrainer:
    """Deterministic surrogate: lower loss for larger, cleaner, more diverse data."""

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
        return await run_blocking_drained(self._train_and_eval_sync, corpus, config)

    def _train_and_eval_sync(
        self, corpus: CuratedCorpus, config: ProxyStudentConfig
    ) -> TrainResult:
        if corpus.is_empty():
            return TrainResult(
                loss=float("inf"),
                accuracy=0.0,
                flops=0.0,
                tokens_trained=0,
                backend="heuristic",
            )
        tokens = corpus.total_tokens
        # Caps on effective_train_tokens (includes train_token_budget).
        target_tokens = max(config.effective_train_tokens, 1)
        tokens_trained = min(tokens, target_tokens)

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


def _nanogpt_train_script() -> str:
    """Return the single-file trainer copied into the sandbox workspace."""
    return TRAIN_GPT_SOURCE


def __getattr__(name: str) -> Any:
    if name == "NANOGPT_TRAIN_SCRIPT":
        return _nanogpt_train_script()
    raise AttributeError(name)


class RuntimeProxyTrainer:
    """Train through the live v1 runtime selected by the harness config."""

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
                    runtime.write(path, data), timeout=config.sandbox.upload_timeout_seconds
                )
                self._raise_if_cancelling()
            except (asyncio.TimeoutError, TimeoutError) as exc:
                raise TrainerError(
                    f"timed out writing proxy-student input {path!r}"
                ) from exc

    async def _run_training(
        self, runtime: vf.Runtime, config: ProxyStudentConfig
    ) -> vf.ProgramResult:
        docker = runtime.type == "docker"
        timeout = config.effective_timeout_minutes * 60
        try:
            result = await asyncio.wait_for(
                runtime.run(["python", "/workspace/train.py"], {}), timeout=timeout
            )
            self._raise_if_cancelling()
        except (asyncio.TimeoutError, TimeoutError) as exc:
            raise TrainerError(
                f"proxy-student training timed out after {timeout}s",
                stderr_tail=self._training_diagnostic("", ""),
            ) from exc

        redirected = (
            await self._read_redirected_stderr(runtime, config) if docker else ""
        )
        stderr = self._merge_stderr(result.stderr, redirected)
        await self._persist_logs(
            runtime,
            result,
            config,
            redirected=redirected,
            stderr=stderr,
        )
        if result.exit_code not in (0, None):
            raise TrainerError(
                f"proxy-student training exited with code {result.exit_code}",
                stderr_tail=self._training_diagnostic(result.stdout, stderr),
            )
        return vf.ProgramResult(
            exit_code=result.exit_code,
            stdout=result.stdout or "",
            stderr=stderr,
        )

    async def _read_redirected_stderr(
        self, runtime: vf.Runtime, config: ProxyStudentConfig
    ) -> str:
        try:
            data = await asyncio.wait_for(
                runtime.read("/workspace/stderr.txt"),
                timeout=config.sandbox.upload_timeout_seconds,
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
        result: vf.ProgramResult,
        config: ProxyStudentConfig,
        *,
        redirected: str,
        stderr: str,
    ) -> None:
        logs = {
            "/workspace/train_stdout.log": result.stdout or "",
            "/workspace/train_stderr.log": stderr,
        }
        if runtime.type == "docker":
            logs["/workspace/train_stderr_redirect.log"] = redirected
        try:
            for path, text in logs.items():
                await asyncio.wait_for(
                    runtime.write(path, text.encode("utf-8", errors="replace")),
                    timeout=config.sandbox.upload_timeout_seconds,
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

    def _training_diagnostic(self, stdout: str, stderr: str) -> str:
        parts = [
            "--- stdout tail ---",
            self._diagnostic_stream(stdout or ""),
            "--- stderr tail ---",
            self._diagnostic_stream(stderr or ""),
        ]
        return "\n".join(parts)

    def _diagnostic_stream(self, text: str) -> str:
        marker_at = text.find(self.TRACEBACK_MARKER)
        return text[marker_at:] if marker_at >= 0 else text[-self.STREAM_TAIL :]

    @staticmethod
    async def _stop_cancel_safe(runtime: vf.Runtime) -> None:
        """Drain runtime teardown to completion; cancellation wins, errors log."""
        try:
            await run_shielded(runtime.stop())
        except asyncio.CancelledError:
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
