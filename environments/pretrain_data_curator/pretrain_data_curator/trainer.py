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

    ``student_model`` (and therefore ``torch``) is imported lazily so the
    package can load for Hub integration / heuristic scoring without a
    runtime torch dependency. Real GPU training embeds the model source into
    the sandbox script and never needs this import path.
    """
    from .student_model import estimate_instantiated_param_count

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


_NANOGPT_TRAIN_SCRIPT_TEMPLATE = r"""
import atexit
import sys

# Redirect stderr to a durable workspace file. Use line buffering so traceback /
# CUDA OOM lines are visible to the harness before process death, and flush on
# exit so a final partial line is not lost when the interpreter aborts.
_stderr_path = "/workspace/stderr.txt"
_stderr_fh = open(_stderr_path, "w", buffering=1)
sys.stderr = _stderr_fh
atexit.register(_stderr_fh.flush)

import json, math, os, subprocess, time
from dataclasses import dataclass
# The full-vocab logits + tanh-softcap chain (out = softcap * tanh(logits/softcap))
# allocates several (B*T, 50304) fp32 temporaries; at the largest batch-schedule
# stage this sits near the 80GB A100 ceiling, where CUDA allocator fragmentation
# ("reserved but unallocated") can tip an otherwise-fitting run into OOM. Opt into
# expandable segments (set before torch initializes CUDA) to reclaim that reserve.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
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
    corpus_text = f.read()
try:
    corpus_payload = json.loads(corpus_text)
except json.JSONDecodeError:
    corpus_payload = None
if isinstance(corpus_payload, dict) and corpus_payload.get("format") == "document-list-v1":
    documents = corpus_payload.get("documents")
    if not isinstance(documents, list) or not all(isinstance(doc, str) for doc in documents):
        raise ValueError("invalid document-list-v1 corpus payload")
else:
    documents = None

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

# __DOCUMENT_ENCODING__  (replaced with tested encode_document_tokens source)

eos_aligned_batches = bool(cfg.get("eos_aligned_batches", True))
if documents is None:
    if eos_aligned_batches:
        raise ValueError(
            "EOS-aligned training requires a document-list-v1 corpus payload; "
            "flat text cannot recover source document boundaries safely"
        )
    documents = [corpus_text]
corpus_ids, encoded_document_ranges = encode_document_tokens(
    documents,
    enc,
    cfg.get("max_document_tokens", cfg.get("max_doc_len")),
)
document_ranges = encoded_document_ranges if eos_aligned_batches else None
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
    if document_ranges is not None:
        document_ranges = [bounds for bounds in document_ranges if bounds[1] <= len(train_data)]
    val_source = "corpus_split"

block = int(cfg["block_size"]); batch = int(cfg["batch_size"])

# __STUDENT_MODEL__  (replaced with the verbatim student_model.py model source)

# __STUDENT_OPTIMIZER__  (replaced with the verbatim student_optimizer.py source)

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
        attn_scale=float(cfg.get("attn_scale", 0.12)),
        sliding_window_size=cfg.get("sliding_window_size"),
        # portable feature flags (all default-off)
        bigram_hash_embed=bool(cfg.get("bigram_hash_embed", False)),
        smear_embed=bool(cfg.get("smear_embed", False)),
        partial_key_offset=cfg.get("partial_key_offset"),
        paired_head=bool(cfg.get("paired_head", False)),
        mudd_pairs=int(cfg.get("mudd_pairs", 0)),
        xsa_enabled=bool(cfg.get("xsa_enabled", False)),
        xsa_pairs=int(cfg.get("xsa_pairs", 0)),
        single_act_last_k=int(cfg.get("single_act_last_k", 0)),
        exp_residual_decay=cfg.get("exp_residual_decay"),
        multi_token_pred=int(cfg.get("multi_token_pred", 0)),
    ).to(device)

# modded-nanogpt speedrun recipe (student_train.py + student_optimizer.py):
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
    vocab_size=vocab_size,
    training_recipe=str(cfg.get("training_recipe", "speedrun_muon")),
    base_lr=float(cfg.get("learning_rate", 3e-4)),
    warmup_steps=int(cfg.get("warmup_steps", 0)),
    weight_decay=float(cfg.get("weight_decay", 0.1)),
    grad_clip=float(cfg.get("grad_clip", 0.0)),
    beta1=float(cfg.get("adam_beta1", 0.9)),
    beta2=float(cfg.get("adam_beta2", 0.95)),
    eps=float(cfg.get("record_adam_eps", cfg.get("adam_eps", 1e-8))),
    lr_min_ratio=float(cfg.get("lr_min_ratio", 0.1)),
    muon_lr=float(cfg.get("muon_lr", 0.023)),
    muon_weight_decay=float(cfg.get("muon_weight_decay", 1.2)),
    adam_lr=float(cfg.get("adam_lr", 0.008)),
    adam_eps=float(cfg.get("adam_eps", 1e-10)),
    adam_weight_decay=float(cfg.get("adam_weight_decay", 0.005)),
    embed_lr_mul=float(cfg.get("embed_lr_mul", 1.0)),
    lm_head_lr_mul=float(cfg.get("lm_head_lr_mul", 1.0)),
    value_embed_lr_mul=float(cfg.get("value_embed_lr_mul", 75.0)),
    scalar_lr_mul=float(cfg.get("scalar_lr_mul", 5.0)),
    embed_wd_mul=float(cfg.get("embed_wd_mul", 150.0)),
    lm_head_wd_mul=float(cfg.get("lm_head_wd_mul", 150.0)),
    value_embed_wd_mul=float(cfg.get("value_embed_wd_mul", 5.0)),
    scalar_wd_mul=float(cfg.get("scalar_wd_mul", 0.0)),
    batch_schedule_enabled=bool(cfg.get("batch_schedule_enabled", True)),
    batch_stage_fracs=tuple(cfg.get("batch_stage_fracs", (1/3, 1/3, 1/3))),
    batch_stage_muls=tuple(cfg.get("batch_stage_muls", (1, 2, 3))),
    lr_stage_muls=tuple(cfg.get("lr_stage_muls", (1.0, 1.52, 1.73))),
    lr_cooldown_frac=float(cfg.get("lr_cooldown_frac", 0.60)),
    lr_cooldown_floor=float(cfg.get("lr_cooldown_floor", 0.15)),
    muon_momentum_min=float(cfg.get("muon_momentum_min", 0.85)),
    muon_momentum_max=float(cfg.get("muon_momentum_max", 0.95)),
    muon_warmup_steps=cfg.get("muon_warmup_steps"),
    muon_cooldown_steps=cfg.get("muon_cooldown_steps"),
    adam_on_odd_steps=bool(cfg.get("adam_on_odd_steps", True)),
    # portable feature hparams (all default-off)
    document_ranges=document_ranges,
    grad_accum_embed_head_steps=int(cfg.get("grad_accum_embed_head_steps", 1)),
    seq_len_schedule=bool(cfg.get("seq_len_schedule", False)),
    multi_token_pred=int(cfg.get("multi_token_pred", 0)),
    untie_at_frac=float(cfg.get("untie_at_frac", 0.0)),
    cautious_wd=bool(cfg.get("cautious_wd", False)),
    nor_muon=bool(cfg.get("nor_muon", True)),
    polar_express=bool(cfg.get("polar_express", False)),
    train_microbatch_size=cfg.get("train_microbatch_size"),
    val_batch_size=cfg.get("val_batch_size"),
    val_logit_chunk_tokens=cfg.get("val_logit_chunk_tokens"),
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
"""


# Embed the exact, CPU-tested ``plan_val_windows`` source, the verbatim
# ``student_model`` architecture, AND the verbatim ``student_train`` recipe
# (optimizer schedule, contiguous batching, multi-run averaging) into the sandbox
# script, so the validation windowing, the model, and the training recipe that no
# GPU test can reach are ALL guarded by this package's CPU unit tests (the script
# runs the identical code).
def _module_definitions_source(module_filename: str, names: tuple[str, ...]) -> str:
    source = (Path(__file__).with_name(module_filename)).read_text()
    tree = ast.parse(source)
    lines = source.splitlines()
    by_name: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef | ast.FunctionDef):
            continue
        # ``node.lineno`` starts at the ``def``/``class`` keyword and excludes
        # any decorators, so a plain ``ast.get_source_segment`` silently drops
        # them (e.g. ``@dataclass`` on ``BatchScheduleStage``). Extend the
        # segment back to the first decorator's line when present.
        start_line = node.lineno
        if node.decorator_list:
            start_line = min(start_line, node.decorator_list[0].lineno)
        by_name[node.name] = "\n".join(lines[start_line - 1 : node.end_lineno])
    missing = [name for name in names if not by_name.get(name)]
    if missing:
        raise RuntimeError(
            f"could not embed {module_filename} definitions: {', '.join(missing)}"
        )
    return "\n\n\n".join(by_name[name].rstrip() for name in names)


def _nanogpt_train_script() -> str:
    from .student_train import encode_document_tokens

    return (
        _NANOGPT_TRAIN_SCRIPT_TEMPLATE.replace(
            "# __PLAN_VAL_WINDOWS__  (replaced with the tested plan_val_windows source)",
            inspect.getsource(plan_val_windows).rstrip(),
        )
        .replace(
            "# __DOCUMENT_ENCODING__  (replaced with tested encode_document_tokens source)",
            inspect.getsource(encode_document_tokens).rstrip(),
        )
        .replace(
            "# __STUDENT_MODEL__  (replaced with the verbatim student_model.py model source)",
            _module_definitions_source(
                "student_model.py",
                (
                    "RMSNorm",
                    "Rotary",
                    "RotaryWithOffset",
                    "_sliding_window_mask",
                    "_causal_attn_mask",
                    "_combine_attn_masks",
                    "CausalSelfAttention",
                    "PairedHeadAttention",
                    "MLP",
                    "Block",
                    "ValueEmbedding",
                    "BigramHashEmbedding",
                    "Smear",
                    "MUDD",
                    "XSA",
                    "MultiTokenHeads",
                    "GPT",
                ),
            ),
        )
        .replace(
            "# __STUDENT_OPTIMIZER__  (replaced with the verbatim student_optimizer.py source)",
            _module_definitions_source(
                "student_optimizer.py",
                (
                    "zeropower_via_newtonschulz5",
                    "zeropower_via_polar_express",
                    "muon_update",
                    "muon_update_normalized",
                    "Muon",
                    "BatchScheduleStage",
                    "build_batch_schedule",
                    "lookup_batch_stage",
                    "schedule_lr_multiplier",
                    "get_muon_momentum",
                    "classify_speedrun_params",
                    "build_speedrun_optimizers",
                    "init_speedrun_weights",
                    "set_optimizer_lrs",
                    "capture_initial_lrs",
                    "clip_optimizer_grads",
                    "step_speedrun_optimizers",
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
                    "plan_eos_aligned_windows",
                    "window_document_ids",
                    "build_document_attn_mask",
                    "batch_document_attn_mask",
                    "shuffled_window_starts",
                    "make_seq_len_schedule",
                    "_is_cuda_device",
                    "prepare_student_model_dtype",
                    "_microbatch_ranges",
                    "_scaled_microbatch_loss",
                    "_score_hidden_chunked",
                    "_eval_val_loss",
                    "_compute_multi_token_loss",
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
