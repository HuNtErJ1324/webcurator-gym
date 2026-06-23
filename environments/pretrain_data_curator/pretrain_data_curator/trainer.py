"""Proxy-student training: the Perf(M) term.

`ProxyStudentTrainer` is the contract the reward calls. Two backends implement it:

  - `HeuristicProxyTrainer`: deterministic, CPU-only stand-in that predicts
    loss/accuracy from corpus statistics. Used in tests and as the default so the
    environment is usable without GPU.
  - `SandboxProxyTrainer`: actually trains a fixed small GPT-2-scale model in a
    Prime GPU sandbox on the curated corpus (everything fixed but the data) and
    reports measured val loss, next-token accuracy, and FLOPs.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import weakref
from typing import Any, Callable, Protocol

from pydantic import BaseModel

from .corpus import CuratedCorpus
from .hf_access import loop_local_semaphore
from .models import ProxyStudentConfig
from .val_set import (
    NANOGPT_VAL_TOKENIZER,
    HeldOutValSet,
    ValTokenLoader,
    plan_val_windows,
)

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
    apply to this backend — it is consumed only by ``SandboxProxyTrainer``. Its
    ``loss`` is a synthetic statistic, not a nats/token cross-entropy.
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
        if not corpus.documents:
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
        target_tokens = max(config.steps * config.batch_size * config.block_size, 1)
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
    docs = corpus.documents
    if not docs:
        return 0.0
    ratios = []
    for doc in docs:
        if not doc:
            continue
        alpha = sum(1 for c in doc if c.isalpha() or c.isspace()) / len(doc)
        ratios.append(alpha)
    return sum(ratios) / len(ratios) if ratios else 0.0


def _source_diversity(corpus: CuratedCorpus) -> float:
    non_empty = [s for s in corpus.sources if s.documents]
    if len(non_empty) <= 1:
        return 0.0
    total = sum(s.tokens for s in non_empty)
    if total <= 0:
        return 0.0
    weights = [s.tokens / total for s in non_empty]
    entropy = -sum(w * math.log(w) for w in weights if w > 0)
    return entropy / math.log(len(non_empty))


_NANOGPT_TRAIN_SCRIPT_TEMPLATE = r'''
import json, math, os, subprocess, sys
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

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

def get_train_batch():
    src = train_data
    if len(src) <= block + 1:
        src = src.repeat(math.ceil((block + 2) / max(len(src), 1)))
    ix = torch.randint(len(src) - block - 1, (batch,))
    x = torch.stack([src[i:i+block] for i in ix])
    y = torch.stack([src[i+1:i+block+1] for i in ix])
    return x.to(device), y.to(device)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(n_embd, n_head, batch_first=True)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = nn.Sequential(nn.Linear(n_embd, 4*n_embd), nn.GELU(), nn.Linear(4*n_embd, n_embd))
    def forward(self, x):
        h = self.ln1(x)
        mask = torch.triu(torch.ones(h.size(1), h.size(1), device=x.device), diagonal=1).bool()
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self, vocab, n_embd, n_head, n_layer, block):
        super().__init__()
        self.tok = nn.Embedding(vocab, n_embd)
        self.pos = nn.Embedding(block, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab)
        self.block = block
    def forward(self, idx):
        pos = torch.arange(idx.size(1), device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None]
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln(x))

model = GPT(vocab_size, int(cfg["n_embd"]), int(cfg["n_head"]), int(cfg["n_layer"]), block).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["learning_rate"]))
steps = int(cfg["steps"])
model.train()
for step in range(steps):
    x, y = get_train_batch()
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

# Cross-entropy (nats/token, mean reduction) over EVERY predictable next-token
# position of the held-out val stream: all len(val_data)-1 targets, including the
# final partial window. plan_val_windows raises on a degenerate (<=1 token) val
# set, so an empty score set degrades to the sentinel rather than reporting a
# bogus perfect 0.0. The denominator is the actual number of scored targets.
model.eval()
val_windows = plan_val_windows(len(val_data), block)
total = sum(L for _, L in val_windows)  # == len(val_data) - 1, always > 0 here
loss_sum = 0.0; correct = 0
with torch.no_grad():
    wi = 0
    while wi < len(val_windows):
        # Batch consecutive equal-length windows (all full blocks, then the final
        # partial window alone) so variable lengths never get stacked together.
        L = val_windows[wi][1]
        starts = []
        while wi < len(val_windows) and val_windows[wi][1] == L and len(starts) < batch:
            starts.append(val_windows[wi][0]); wi += 1
        x = torch.stack([val_data[s:s+L] for s in starts]).to(device)
        y = torch.stack([val_data[s+1:s+L+1] for s in starts]).to(device)
        logits = model(x)
        loss_sum += F.cross_entropy(
            logits.reshape(-1, vocab_size), y.reshape(-1), reduction="sum"
        ).item()
        correct += (logits.argmax(-1) == y).sum().item()
val_loss = loss_sum / total
acc = correct / total
n_params = sum(p.numel() for p in model.parameters())
tokens_trained = steps * batch * block
flops = 6.0 * n_params * tokens_trained
print("RESULT_JSON " + json.dumps({
    "loss": val_loss, "accuracy": acc, "flops": flops,
    "tokens_trained": tokens_trained, "n_params": n_params, "vocab_size": vocab_size,
    "val_tokens": int(len(val_data)), "val_scored_targets": int(total),
    "val_source": val_source,
}))
'''


# Embed the exact, CPU-tested ``plan_val_windows`` source into the sandbox script
# so the validation windowing/coverage that no GPU test can reach is guarded by
# this module's unit tests (the script runs the identical function).
NANOGPT_TRAIN_SCRIPT = _NANOGPT_TRAIN_SCRIPT_TEMPLATE.replace(
    "# __PLAN_VAL_WINDOWS__  (replaced with the tested plan_val_windows source)",
    inspect.getsource(plan_val_windows).rstrip(),
)


class SandboxProxyTrainer:
    """Trains the fixed proxy-student in a Prime GPU sandbox on the curated data.

    The student is GPT-2-BPE-tokenized and scored as cross-entropy (nats/token)
    over a **held-out** validation token stream — by default the NanoGPT-speedrun
    FineWeb val set (the first ``val_tokens`` GPT-2 tokens; see ``val_set.py``).
    The held-out tokens are loaded host-side through the same robustness machinery
    as every other external fetch (off-loop, timeout, retry, semaphore, typed
    ``DatasetAccessError``) and uploaded into the sandbox, so the validation set is
    fixed across rollouts and independent of the curated corpus.

    Hardened lifecycle: every sandbox step is wrapped with a timeout, the
    training command's exit code is checked, a nonzero exit (or missing result)
    surfaces a `TrainerError` carrying the stderr tail, the whole lifecycle is
    bounded by a loop-local semaphore, and post-run cleanup failures are surfaced
    (logged and attached to the result) rather than silently swallowed.

    The sandbox client and request are built through injectable factories so the
    lifecycle is testable without a real sandbox (and `prime_sandboxes` is only
    imported on the live path, where it is an optional dependency). The held-out
    val set is loaded via an injectable `ValTokenLoader`; a failure to fetch it
    raises a `DatasetAccessError` *before* any sandbox is created, which the
    rubric degrades to the infinite-loss sentinel like any other external failure.
    """

    STDERR_TAIL = 2000

    def __init__(
        self,
        max_corpus_chars: int = 5_000_000,
        concurrency_limit: int = 1,
        client_factory: Callable[[], Any] | None = None,
        request_factory: Callable[[ProxyStudentConfig, str], Any] | None = None,
        val_loader: ValTokenLoader | None = None,
    ) -> None:
        self._max_corpus_chars = max_corpus_chars
        self._concurrency_limit = concurrency_limit
        self._client_factory = client_factory
        self._request_factory = request_factory
        self._val_loader = val_loader

    def _make_client(self) -> Any:
        if self._client_factory is not None:
            return self._client_factory()
        from prime_sandboxes import AsyncSandboxClient

        return AsyncSandboxClient()

    def _make_request(self, config: ProxyStudentConfig, name: str) -> Any:
        if self._request_factory is not None:
            return self._request_factory(config, name)
        from prime_sandboxes import CreateSandboxRequest

        return CreateSandboxRequest(
            name=name,
            docker_image=config.docker_image,
            start_command="tail -f /dev/null",
            cpu_cores=config.cpu_cores,
            memory_gb=config.memory_gb,
            disk_size_gb=config.disk_size_gb,
            gpu_count=config.gpu_count,
            gpu_type=config.gpu_type,
            timeout_minutes=config.timeout_minutes,
        )

    async def train_and_eval(
        self, corpus: CuratedCorpus, config: ProxyStudentConfig
    ) -> TrainResult:
        text = "\n\n".join(corpus.documents)[: self._max_corpus_chars]
        if not text.strip():
            return TrainResult(
                loss=float("inf"),
                accuracy=0.0,
                flops=0.0,
                tokens_trained=0,
                backend="sandbox",
            )

        # Resolve the held-out val set BEFORE provisioning a GPU sandbox: there is
        # no point training a student we cannot score, and a fetch failure here
        # raises a typed DatasetAccessError the rubric degrades to the sentinel.
        val_set = await self._resolve_val_set()

        payload = {
            "n_layer": config.n_layer,
            "n_head": config.n_head,
            "n_embd": config.n_embd,
            "block_size": config.block_size,
            "batch_size": config.batch_size,
            "steps": config.steps,
            "learning_rate": config.learning_rate,
            "seed": config.seed,
            "val_fraction": config.val_fraction,
            "tokenizer": val_set.tokenizer if val_set else NANOGPT_VAL_TOKENIZER,
        }

        async with training_semaphore(self._concurrency_limit):
            client = self._make_client()
            request = self._make_request(config, "proxy-student-trainer")
            sandbox = await asyncio.wait_for(
                client.create(request), timeout=config.create_timeout_seconds
            )
            cleanup_error: str | None = None
            try:
                await asyncio.wait_for(
                    client.wait_for_creation(sandbox.id),
                    timeout=config.create_timeout_seconds,
                )
                await self._upload_all(client, sandbox.id, text, payload, config, val_set)
                result = await self._run_training(client, sandbox.id, config)
                train_result = self._parse_result(result.stdout, result.stderr)
            finally:
                cleanup_error = await self._cleanup(client, sandbox.id, config)
            if cleanup_error is not None:
                train_result = train_result.model_copy(
                    update={"cleanup_error": cleanup_error}
                )
            return train_result

    async def _resolve_val_set(self) -> HeldOutValSet | None:
        """Load the held-out validation token stream, or ``None`` if unconfigured.

        Propagates ``DatasetAccessError`` on a fetch/parse failure so the rubric's
        ``_train`` degrades the whole run to the infinite-loss sentinel.
        """
        if self._val_loader is None:
            return None
        return await self._val_loader.load()

    async def _upload_all(
        self,
        client: Any,
        sandbox_id: str,
        text: str,
        payload: dict[str, Any],
        config: ProxyStudentConfig,
        val_set: HeldOutValSet | None = None,
    ) -> None:
        files = [
            ("/workspace/corpus.txt", text.encode("utf-8"), "corpus.txt"),
            (
                "/workspace/config.json",
                json.dumps(payload).encode("utf-8"),
                "config.json",
            ),
            ("/workspace/train.py", NANOGPT_TRAIN_SCRIPT.encode("utf-8"), "train.py"),
        ]
        if val_set is not None:
            # Header-free little-endian uint16 GPT-2 token ids: exactly the first
            # val_tokens tokens of the held-out shard, scored as CE in the sandbox.
            files.append(("/workspace/val.bin", val_set.to_uint16_bytes(), "val.bin"))
        for path, data, name in files:
            await asyncio.wait_for(
                client.upload_bytes(sandbox_id, path, data, name),
                timeout=config.upload_timeout_seconds,
            )

    async def _run_training(
        self, client: Any, sandbox_id: str, config: ProxyStudentConfig
    ) -> Any:
        execute_timeout = config.timeout_minutes * 60
        try:
            result = await asyncio.wait_for(
                client.execute_command(
                    sandbox_id,
                    "python /workspace/train.py",
                    timeout=execute_timeout,
                ),
                # Hard wall-clock bound above the command's own timeout.
                timeout=execute_timeout + 30,
            )
        except (asyncio.TimeoutError, TimeoutError) as exc:
            raise TrainerError(
                f"proxy-student training timed out after {execute_timeout}s"
            ) from exc
        exit_code = getattr(result, "exit_code", 0)
        if exit_code not in (0, None):
            raise TrainerError(
                f"proxy-student training exited with code {exit_code}",
                stderr_tail=(result.stderr or "")[-self.STDERR_TAIL :],
            )
        return result

    async def _cleanup(
        self, client: Any, sandbox_id: str, config: ProxyStudentConfig
    ) -> str | None:
        """Delete the sandbox; surface (don't swallow) any cleanup failure."""
        try:
            await asyncio.wait_for(
                client.delete(sandbox_id), timeout=config.create_timeout_seconds
            )
            return None
        except Exception as exc:  # noqa: BLE001 - surfaced below, not swallowed
            message = f"{type(exc).__name__}: {exc}"
            logger.warning("sandbox cleanup failed for %s: %s", sandbox_id, message)
            return message

    def _parse_result(self, stdout: str, stderr: str) -> TrainResult:
        for line in reversed((stdout or "").splitlines()):
            if line.startswith("RESULT_JSON "):
                data = json.loads(line[len("RESULT_JSON ") :])
                return TrainResult(
                    loss=float(data["loss"]),
                    accuracy=float(data.get("accuracy", 0.0)),
                    flops=float(data.get("flops", 0.0)),
                    tokens_trained=int(data.get("tokens_trained", 0)),
                    backend="sandbox",
                )
        raise TrainerError(
            "proxy-student training produced no RESULT_JSON",
            stderr_tail=(stderr or "")[-self.STDERR_TAIL :],
        )
