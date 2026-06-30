"""Modal GPU training backend for the proxy-student trainer.

Implements ModalProxyTrainer as a drop-in third backend alongside
HeuristicProxyTrainer (CPU heuristic) and SandboxProxyTrainer (Prime/Docker).
Calls modal.Sandbox.create(...) so it works from a CPU-only env-server in
Hosted Training.

``modal`` is imported lazily inside ``_run_in_modal`` so the default heuristic
and the other real-training backends do not initialize the Modal SDK.

Credentials: ``MODAL_TOKEN_ID`` + ``MODAL_TOKEN_SECRET`` env vars.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from .corpus import CuratedCorpus
from .models import ProxyStudentConfig
from .trainer import TrainResult, TrainerError, _nanogpt_train_script, training_semaphore
from .val_set import NANOGPT_VAL_TOKENIZER, HeldOutValSet, ValTokenLoader

logger = logging.getLogger(__name__)

# Maps ProxyStudentConfig.modal_gpu → Modal GPU string. Anything not in the
# map falls through to the L4 default (cheapest GPU available on Modal).
_MODAL_GPU_MAP: dict[str, str] = {
    "H100": "H100",
    "H200": "H200",
    "A100": "A100-80GB",
}
_DEFAULT_MODAL_GPU = "L4"


def _modal_gpu_for(modal_gpu: str) -> str:
    """Map a ProxyStudentConfig.modal_gpu value to a Modal GPU specifier."""
    return _MODAL_GPU_MAP.get(modal_gpu, _DEFAULT_MODAL_GPU)


class ModalProxyTrainer:
    """Trains the fixed proxy-student in a Modal GPU sandbox on the curated data.

    Uses ``modal.Sandbox.create(...)`` (v1 Modal API) — works from a CPU-only
    env-server in Hosted Training. The GPU type comes from
    ``ProxyStudentConfig.modal_gpu``; the training script, corpus, and config
    are uploaded as workspace files; the result is polled from
    ``/workspace/result.json`` once training writes it.

    ``modal`` is imported lazily inside ``_run_in_modal`` so this module is
    importable without modal installed.
    """

    STDERR_TAIL = 2000

    def __init__(
        self,
        max_corpus_chars: int | None = None,
        concurrency_limit: int = 1,
        val_loader: ValTokenLoader | None = None,
    ) -> None:
        # ``None`` derives the cap per-run from ``ProxyStudentConfig.effective_max_corpus_chars``.
        self._max_corpus_chars = max_corpus_chars
        self._concurrency_limit = concurrency_limit
        self._val_loader = val_loader

    async def _resolve_val_set(self) -> HeldOutValSet | None:
        if self._val_loader is None:
            return None
        return await self._val_loader.load()

    async def train_and_eval(
        self, corpus: CuratedCorpus, config: ProxyStudentConfig
    ) -> TrainResult:
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

        payload: dict[str, Any] = {
            "n_layer": config.n_layer,
            "n_head": config.n_head,
            "n_embd": config.n_embd,
            "mlp_ratio": config.mlp_ratio,
            "lm_head_softcap": config.lm_head_softcap,
            "num_value_embeds": config.num_value_embeds,
            "block_size": config.block_size,
            "batch_size": config.batch_size,
            # Budget-derived step count: tokens_trained = steps * batch * block
            # so a larger train_token_budget scales tokens and FLOPs together.
            "steps": config.effective_steps,
            "learning_rate": config.learning_rate,
            "seed": config.seed,
            "val_fraction": config.val_fraction,
            "tokenizer": val_set.tokenizer if val_set else NANOGPT_VAL_TOKENIZER,
            # record_01 optimizer schedule + regularization + averaging.
            "weight_decay": config.weight_decay,
            "adam_beta1": config.adam_beta1,
            "adam_beta2": config.adam_beta2,
            "adam_eps": config.adam_eps,
            "grad_clip": config.grad_clip,
            "warmup_steps": config.effective_warmup_steps,
            "lr_min_ratio": config.lr_min_ratio,
            "n_train_runs": config.n_train_runs,
        }

        async with training_semaphore(self._concurrency_limit):
            try:
                stdout, stderr = await self._run_in_modal(text, payload, config, val_set)
                return self._parse_result(stdout, stderr)
            except TrainerError:
                raise
            except (asyncio.TimeoutError, TimeoutError) as exc:
                raise TrainerError(
                    f"modal training timed out after "
                    f"{config.effective_timeout_minutes} minutes"
                ) from exc
            except Exception as exc:
                logger.error("modal training failed: %s", exc)
                raise TrainerError(
                    f"modal training failed: {type(exc).__name__}: {exc}"
                ) from exc

    async def _run_in_modal(
        self,
        text: str,
        payload: dict[str, Any],
        config: ProxyStudentConfig,
        val_set: HeldOutValSet | None,
    ) -> tuple[str, str]:
        import modal  # lazy: modal is an optional dependency

        gpu_str = _modal_gpu_for(config.modal_gpu)
        timeout_sec = config.effective_timeout_minutes * 60
        image = modal.Image.debian_slim().pip_install(["torch", "tiktoken", "numpy"])
        app = await modal.App.lookup.aio(
            "pretrain-data-curator-trainer", create_if_missing=True
        )

        sb = await asyncio.wait_for(
            modal.Sandbox.create.aio(
                image=image,
                app=app,
                gpu=gpu_str,
                timeout=timeout_sec,
                workdir="/workspace",
            ),
            timeout=config.create_timeout_seconds,
        )
        try:
            await _write_file(sb, "/workspace/train.py", _nanogpt_train_script().encode())
            await _write_file(sb, "/workspace/corpus.txt", text.encode("utf-8"))
            await _write_file(sb, "/workspace/config.json", json.dumps(payload).encode())
            if val_set is not None:
                await _write_file(sb, "/workspace/val.bin", val_set.to_uint16_bytes())

            await sb.exec.aio("python", "/workspace/train.py", text=False)

            # Poll for result.json rather than waiting on the process (exec_wait RPC hangs
            # in Modal SDK v1.5.1 even after the process exits).
            deadline = asyncio.get_event_loop().time() + timeout_sec + 60.0
            result_json_text: str | None = None
            while asyncio.get_event_loop().time() < deadline:
                await asyncio.sleep(5)
                try:
                    candidate = await asyncio.to_thread(
                        lambda: sb.open("/workspace/result.json", "r").read()
                    )
                    if candidate and candidate.strip():
                        json.loads(candidate)  # validate complete JSON before accepting
                        result_json_text = candidate
                        break
                except Exception:
                    pass  # file not yet written or incomplete — keep polling

            if result_json_text is None:
                # Training timed out. Try to read stderr for diagnostics.
                stderr_text = ""
                try:
                    stderr_text = await asyncio.to_thread(
                        lambda: sb.open("/workspace/stderr.txt", "r").read()
                    )
                except Exception:
                    pass
                raise TrainerError(
                    f"modal training timed out after {timeout_sec + 60:.0f}s — no result.json appeared",
                    stderr_tail=stderr_text[-self.STDERR_TAIL:],
                )

            return f"RESULT_JSON {result_json_text}\n", ""
        finally:
            try:
                await sb.terminate.aio()
            except Exception as exc:  # noqa: BLE001 - cleanup must not mask run errors
                logger.warning(
                    "modal sandbox terminate failed: %s: %s", type(exc).__name__, exc
                )

    def _parse_result(self, stdout: str, stderr: str) -> TrainResult:
        for line in reversed((stdout or "").splitlines()):
            if line.startswith("RESULT_JSON "):
                data = json.loads(line[len("RESULT_JSON ") :])
                return TrainResult(
                    loss=float(data["loss"]),
                    accuracy=float(data.get("accuracy", 0.0)),
                    flops=float(data.get("flops", 0.0)),
                    tokens_trained=int(data.get("tokens_trained", 0)),
                    backend="modal",
                )
        logger.error(
            "modal training produced no RESULT_JSON. stderr: %s", stderr[-500:]
        )
        raise TrainerError(
            "modal training produced no RESULT_JSON",
            stderr_tail=(stderr or "")[-self.STDERR_TAIL :],
        )

async def _write_file(sb: Any, path: str, data: bytes) -> None:
    """Write bytes to an absolute path inside the sandbox.

    Uses the synchronous ``Sandbox.open`` API (same pattern as the
    nanogpt_speedrun env, which is confirmed working) wrapped in
    ``asyncio.to_thread`` so we don't block the event loop.
    """
    def _sync_write() -> None:
        with sb.open(path, "wb") as f:
            f.write(data)

    await asyncio.to_thread(_sync_write)
