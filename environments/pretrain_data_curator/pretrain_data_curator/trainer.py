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
import ast
import inspect
import logging
import math
import weakref
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel

from .corpus import CuratedCorpus
from .hf_access import loop_local_semaphore
from .models import ProxyStudentConfig
from .val_set import plan_val_windows

logger = logging.getLogger(__name__)

# Loop-local bound on concurrent sandbox-training jobs, so a rollout group with
# the real trainer never spawns more GPU sandboxes than configured at once.
_TRAIN_SEMAPHORES: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore]" = (
    weakref.WeakKeyDictionary()
)


def training_semaphore(limit: int) -> asyncio.Semaphore:
    return loop_local_semaphore(_TRAIN_SEMAPHORES, limit)


class TrainerError(RuntimeError):
    """A surfaced sandbox-training failure, preserving the stderr tail."""

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
    """Rough decoder-only transformer parameter count."""
    attn = 4 * config.n_embd * config.n_embd
    mlp = 8 * config.n_embd * config.n_embd
    per_layer = attn + mlp
    return config.n_layer * per_layer + 2 * config.n_embd


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
        # from the budget when set), so a larger budget raises the data the schedule
        # would consume; ``tokens_trained`` is still capped at the corpus's tokens,
        # so the heuristic never bills for data it does not have and stays cheap.
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

    def __init__(self, trainers_by_runtime_type: dict[str, "ProxyStudentTrainer"]) -> None:
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


_NANOGPT_TRAIN_SCRIPT_TEMPLATE = r'''
import sys
sys.stderr = open('/workspace/stderr.txt', 'w')

import json, math, os, subprocess, time
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

# tqdm drives the training progress bar/log lines (student_train.py); installed
# on demand like tiktoken below, since the base image may not ship it. Training
# must never fail just because tqdm couldn't be installed (offline container,
# no pip, read-only fs, ...), so a failed install falls back to a minimal
# tqdm-compatible shim: no live bar, but the throttled plain print()-style
# progress lines (via `.write`) still work since they don't depend on a real
# bar.
try:
    from tqdm import tqdm
except ImportError:
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tqdm"], check=True)
        from tqdm import tqdm
    except Exception:
        class _NoOpProgressBar:
            def __init__(self, iterable=None, total=None, **kwargs):
                self._iterable = range(total) if iterable is None else iterable

            def __iter__(self):
                return iter(self._iterable)

            def set_postfix(self, *args, **kwargs):
                pass

            def write(self, s, *args, **kwargs):
                print(s)

            def close(self):
                pass

        tqdm = _NoOpProgressBar

# __PLAN_VAL_WINDOWS__  (replaced with the tested plan_val_windows source)

torch.set_float32_matmul_precision("high")
with open("/workspace/config.json") as f:
    cfg = json.load(f)
with open("/workspace/corpus.txt", encoding="utf-8") as f:
    text = f.read()

seed = int(cfg["seed"])
torch.manual_seed(seed)
device = "cuda" if torch.cuda.is_available() else "cpu"

# GPT-2 BPE: the held-out NanoGPT-speedrun val tokens are GPT-2-tokenized, so the
# student must share that tokenizer/vocab. tiktoken is installed on demand (the
# base image does not ship it).
try:
    import tiktoken
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tiktoken"], check=True)
    import tiktoken
enc = tiktoken.get_encoding(str(cfg.get("tokenizer", "gpt2")))
vocab_size = enc.n_vocab

corpus_ids = enc.encode_ordinary(text)
if len(corpus_ids) < 64:
    corpus_ids = (corpus_ids * math.ceil(64 / max(len(corpus_ids), 1)))[:64] or [0] * 64
corpus = torch.tensor(corpus_ids, dtype=torch.long)

val_path = "/workspace/val.bin"
if os.path.exists(val_path):
    # Held-out validation: exactly the first cfg["val_tokens"] GPT-2 tokens of the
    # speedrun FineWeb val shard (uploaded header-free as little-endian uint16).
    val_ids = np.fromfile(val_path, dtype="<u2").astype(np.int64)
    train_data = corpus
    val_data = torch.from_numpy(val_ids)
    val_source = "held_out"
else:
    # Fallback (no external val supplied): a tail split of the curated corpus.
    n_val = max(1, int(len(corpus) * float(cfg["val_fraction"])))
    train_data, val_data = corpus[:-n_val], corpus[-n_val:]
    val_source = "corpus_split"

block = int(cfg["block_size"]); batch = int(cfg["batch_size"])

# __STUDENT_MODEL__  (replaced with the verbatim student_model.py model source)

# __STUDENT_TRAINING__  (replaced with the verbatim student_train.py recipe source)

def build_model():
    # Fixed architecture; everything is fixed but the curated training data, so a
    # fresh model is rebuilt per averaged run (after that run's seed is set).
    return GPT(
        vocab_size=vocab_size,
        num_layers=int(cfg["n_layer"]),
        model_dim=int(cfg["n_embd"]),
        num_heads=int(cfg["n_head"]),
        mlp_ratio=int(cfg["mlp_ratio"]),
        softcap=float(cfg["lm_head_softcap"]),
        num_value_embeds=int(cfg["num_value_embeds"]),
    ).to(device)

# record_01 recipe (single source of truth in student_train.py, embedded above):
# AdamW(betas, eps, weight_decay) + linear warmup + cosine-to-floor LR + grad-clip,
# CONTIGUOUS-window batching, AVERAGED over n_train_runs distinct seeds. Any
# non-finite run collapses to the infinite-loss sentinel (perf -> 0); FLOPs/tokens
# are summed across runs so cost accounting bills every run.
val_loss, acc, flops, tokens_trained, n_params = averaged_train_and_eval(
    build_model,
    train_data,
    val_data,
    n_runs=int(cfg.get("n_train_runs", 1)),
    base_seed=seed,
    device=device,
    block_size=block,
    batch_size=batch,
    steps=int(cfg["steps"]),
    base_lr=float(cfg["learning_rate"]),
    warmup_steps=int(cfg["warmup_steps"]),
    weight_decay=float(cfg["weight_decay"]),
    grad_clip=float(cfg["grad_clip"]),
    beta1=float(cfg["adam_beta1"]),
    beta2=float(cfg["adam_beta2"]),
    eps=float(cfg["adam_eps"]),
    lr_min_ratio=float(cfg["lr_min_ratio"]),
    vocab_size=vocab_size,
)
result = {
    "loss": val_loss, "accuracy": acc, "flops": flops,
    "tokens_trained": tokens_trained, "n_params": n_params, "vocab_size": vocab_size,
    "val_tokens": int(len(val_data)), "val_scored_targets": int(len(val_data) - 1),
    "val_source": val_source, "n_train_runs": int(cfg.get("n_train_runs", 1)),
}
print("RESULT_JSON " + json.dumps(result), flush=True)
import pathlib
pathlib.Path("/workspace/result.json").write_text(json.dumps(result))
'''


# Embed the exact, CPU-tested ``plan_val_windows`` source, the verbatim
# ``student_model`` architecture, AND the verbatim ``student_train`` recipe
# (optimizer schedule, contiguous batching, multi-run averaging) into the sandbox
# script, so the validation windowing, the model, and the training recipe that no
# GPU test can reach are ALL guarded by this package's CPU unit tests (the script
# runs the identical code).
def _module_definitions_source(module_filename: str, names: tuple[str, ...]) -> str:
    source = (Path(__file__).with_name(module_filename)).read_text()
    tree = ast.parse(source)
    by_name = {
        node.name: ast.get_source_segment(source, node)
        for node in tree.body
        if isinstance(node, ast.ClassDef | ast.FunctionDef)
    }
    missing = [name for name in names if not by_name.get(name)]
    if missing:
        raise RuntimeError(
            f"could not embed {module_filename} definitions: {', '.join(missing)}"
        )
    return "\n\n\n".join(by_name[name].rstrip() for name in names)


def _nanogpt_train_script() -> str:
    return (
        _NANOGPT_TRAIN_SCRIPT_TEMPLATE.replace(
            "# __PLAN_VAL_WINDOWS__  (replaced with the tested plan_val_windows source)",
            inspect.getsource(plan_val_windows).rstrip(),
        )
        .replace(
            "# __STUDENT_MODEL__  (replaced with the verbatim student_model.py model source)",
            _module_definitions_source(
                "student_model.py",
                (
                    "RMSNorm",
                    "Rotary",
                    "CausalSelfAttention",
                    "MLP",
                    "Block",
                    "ValueEmbedding",
                    "GPT",
                ),
            ),
        )
        .replace(
            "# __STUDENT_TRAINING__  (replaced with the verbatim student_train.py recipe source)",
            _module_definitions_source(
                "student_train.py",
                (
                    "lr_at_step",
                    "plan_train_windows",
                    "train_and_eval_student",
                    "averaged_train_and_eval",
                ),
            ),
        )
    )


def __getattr__(name: str) -> Any:
    if name == "NANOGPT_TRAIN_SCRIPT":
        return _nanogpt_train_script()
    raise AttributeError(name)

