"""Proxy-student training: the Perf(M) term.

`ProxyStudentTrainer` is the contract the reward calls. Backends implementing it:

  - `HeuristicProxyTrainer`: deterministic, CPU-only stand-in that predicts
    loss/accuracy from corpus statistics. Used in tests and as the default so the
    environment is usable without GPU.
  - `HarnessRuntimeProxyTrainer` / `ModalProxyTrainer` (see `docker_backend.py` /
    `modal_backend.py`): actually train a fixed small GPT-2-scale model on the
    curated corpus, inside the live Docker or Modal harness runtime that hosts
    the rollout, and report measured val loss, next-token accuracy, and FLOPs.
  - `RuntimeSelectedTrainer`: dispatches to whichever of the above matches the
    live harness runtime's ``type`` at score time -- trainer selection is driven
    entirely by the runtime actually provisioned, never by a config field.
"""

from __future__ import annotations

import asyncio
import logging
import math
import weakref
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel

from .corpus import CuratedCorpus
from .hf_access import loop_local_semaphore
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
        self, corpus: CuratedCorpus, config: ProxyStudentConfig
    ) -> TrainResult: ...


def estimate_param_count(config: ProxyStudentConfig) -> int:
    """Exact instantiated parameter count for the modded-nanogpt student.

    ``train_gpt`` model (and therefore ``torch``) is imported lazily so the
    package can load for Hub integration / heuristic scoring without a
    runtime torch dependency. Real GPU training embeds the model source into
    the sandbox script and never needs this import path.
    """
    from .train_gpt import estimate_instantiated_param_count

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
        self, corpus: CuratedCorpus, config: ProxyStudentConfig
    ) -> TrainResult:
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


class RuntimeSelectedTrainer:
    """Dispatches ``train_and_eval`` to the real trainer matching the live
    harness runtime's ``type``.

    There is no separate backend-selector config consulted here: the harness
    runtime actually provisioned for the rollout is the ONLY signal used to
    pick a concrete trainer, so a rollout always trains with whichever trainer
    matches the runtime it is really running on. Prime GPU sandboxes are not
    supported -- only Docker and Modal harness runtimes have a real trainer.
    """

    def __init__(
        self, trainers_by_runtime_type: dict[str, "ProxyStudentTrainer"]
    ) -> None:
        self._trainers_by_runtime_type = trainers_by_runtime_type

    async def train_and_eval(
        self,
        corpus: CuratedCorpus,
        config: ProxyStudentConfig,
        *,
        runtime: Any = None,
    ) -> TrainResult:
        runtime_type = getattr(runtime, "type", None)
        trainer = self._trainers_by_runtime_type.get(runtime_type)
        if trainer is None:
            supported = " or ".join(sorted(self._trainers_by_runtime_type))
            raise TrainerError(
                f"use_real_trainer requires a {supported} harness runtime; got "
                f"{runtime_type!r}. Pass --harness.runtime.type {supported} (or the "
                "matching load_environment runtime args)."
            )
        return await trainer.train_and_eval(corpus, config, runtime=runtime)


TRAIN_GPT_PATH = Path(__file__).with_name("train_gpt.py")


def _nanogpt_train_script() -> str:
    """Return the single-file trainer copied into the sandbox workspace."""
    return TRAIN_GPT_PATH.read_text(encoding="utf-8")


def __getattr__(name: str) -> Any:
    if name == "NANOGPT_TRAIN_SCRIPT":
        return _nanogpt_train_script()
    raise AttributeError(name)
