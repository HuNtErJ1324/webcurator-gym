"""Pydantic contracts for the pretraining-data curation environment.

Everything the agent produces and everything the reward consumes is expressed
here so the manifest, cost ledger, and configuration have one strict home.
"""

from __future__ import annotations

import math
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, Field, model_serializer, model_validator

from .val_set import ValidationSetConfig

MANIFEST_FILENAME = "manifest.json"

_RESERVED_WORKSPACE_FILES = frozenset(
    {
        MANIFEST_FILENAME,
        "corpus.txt",
        "config.json",
        "train.py",
        "val.bin",
        ".vf_hf_cost.jsonl",
    }
)

# --- proxy-student budget / sandbox derivation constants -------------------
# The real (sandbox) trainer's training length, corpus cap, and sandbox lifetime
# all derive from ``train_token_budget`` so a single token knob scales the whole
# run from a cheap default up to an H100/H200-scale few-hundred-million-token run.
_MAX_TRAIN_TOKEN_BUDGET = 1_000_000_000  # generous H100/H200 upper bound
_CHARS_PER_TOKEN = 4  # matches hf_access.estimate_tokens (chars // 4)
_MIN_CORPUS_CHARS = 5_000_000  # historical default cap; floor for small budgets
_MAX_CORPUS_CHARS = 2_000_000_000  # absolute ceiling on the uploaded corpus blob
# Sandbox lifetime derivation. Modal v1 runtimes cap a remote sandbox at 24 hours.
_MAX_SANDBOX_TIMEOUT_MINUTES = 1440
_MIN_SANDBOX_TIMEOUT_MINUTES = (
    30  # historical default; floor keeps small budgets unchanged
)
_SANDBOX_SETUP_MINUTES = 15  # image pull + pip(tiktoken) + uploads + val download
_SANDBOX_TOKENS_PER_MINUTE = 500_000  # conservative floor for benchmark-sized runs

# Default container image for the real-trainer backends (docker or modal). It
# MUST ship torch + CUDA; this is the runtime image — switch to the matching
# ``-devel`` tag if a run needs build tooling (e.g. nvcc for torch.compile/custom
# kernels). torch 2.7 / CUDA 12.6 matches recent H100/H200 driver stacks.
_DEFAULT_DOCKER_TRAINER_IMAGE = "pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime"


class FilterSpec(BaseModel):
    """A single document-level filter applied to a source.

    `kind` selects a filter implemented by `DocumentFilter`; `params` carries its
    arguments. Filters are agent-supplied, so params stays an open mapping at this
    boundary by design.
    """

    kind: str
    params: dict[str, object] = Field(default_factory=dict)


class Sampling(BaseModel):
    """Per-source caps applied after filtering."""

    max_docs: int | None = Field(default=None, ge=1)
    max_tokens: int | None = Field(default=None, ge=1)


class Source(BaseModel):
    """One Hugging Face or runtime-local dataset slice in the mixture."""

    dataset_id: str
    kind: Literal["hf", "local"] = "hf"
    local_path: str | None = None
    local_format: Literal["auto", "jsonl", "txt"] = "auto"
    config: str | None = None
    split: str = "train"
    text_field: str | None = None
    weight: float = Field(default=1.0, ge=0.0)
    filters: list[FilterSpec] = Field(default_factory=list)
    sampling: Sampling = Field(default_factory=Sampling)

    @model_validator(mode="after")
    def _check_local(self) -> "Source":
        if self.kind != "local":
            return self
        if not self.local_path or not self.local_path.strip():
            raise ValueError("local source requires a non-empty local_path")
        path = PurePosixPath(self.local_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("local_path must be workspace-relative, no '..'")
        if path.name in _RESERVED_WORKSPACE_FILES:
            raise ValueError(
                f"local_path may not reference reserved file {path.name!r}"
            )
        return self

    @model_serializer(mode="wrap")
    def _serialize(self, handler):
        data = handler(self)
        if self.kind == "hf":
            data.pop("kind", None)
            data.pop("local_path", None)
            data.pop("local_format", None)
        return data


class Manifest(BaseModel):
    """The agent's deliverable: a weighted, filtered mixture of sources."""

    token_budget: int = Field(default=1_000_000, gt=0)
    sources: list[Source] = Field(default_factory=list)
    # Agent-chosen optional cap on how many documents are fetched per source from
    # the Hub (the pre-filter fetch cap, distinct from the post-fetch
    # `Sampling.max_docs`/`max_tokens` truncation on `Source`). Omit for no cap;
    # fetches are then sized from each source's weight-proportional token target.
    sample_docs_per_source: int | None = Field(default=None, ge=1)


class CostPrices(BaseModel):
    """Per-unit prices charged on the single cost ledger."""

    web_query: float = 0.0
    hub_call: float = 0.01
    code_call: float = 0.02
    per_1k_tokens: float = 0.001
    per_gflop: float = 1e-6


class CostLedger(BaseModel):
    """Running tally of resources spent during a rollout."""

    web_queries: int = 0
    hub_calls: int = 0
    code_calls: int = 0
    tokens: int = 0
    train_flops: float = 0.0

    def total(self, prices: CostPrices) -> float:
        return (
            self.web_queries * prices.web_query
            + self.hub_calls * prices.hub_call
            + self.code_calls * prices.code_call
            + (self.tokens / 1000.0) * prices.per_1k_tokens
            + (self.train_flops / 1e9) * prices.per_gflop
        )


class ProxyStudentConfig(BaseModel):
    """Fixed GPT-2-scale student config. Only the training corpus varies.

    Every field carries bounds so an out-of-range proxy spec fails fast with a
    clear ``ValidationError`` instead of producing a degenerate (or unschedulable)
    sandbox training job.
    """

    n_layer: int = Field(default=4, ge=2, le=64)
    n_head: int = Field(default=4, ge=1, le=64)
    n_embd: int = Field(default=256, ge=8, le=4096)
    # Modern (modded-nanogpt) student knobs: ReLU**2 MLP width ratio, the tanh
    # logit-softcap constant, and the number of distinct sparse value-embedding
    # tables (SparsifyEmbeds; clamped to n_layer//2 by the model). The model itself
    # lives in ``student_model.py``.
    mlp_ratio: int = Field(default=4, ge=1, le=16)
    lm_head_softcap: float = Field(default=30.0, gt=0.0, le=1000.0)
    num_value_embeds: int = Field(default=3, ge=1, le=32)
    block_size: int = Field(default=256, ge=8, le=8192)
    batch_size: int = Field(default=16, ge=1, le=4096)
    steps: int = Field(default=200, ge=1, le=100_000)
    # Token-oriented training budget. When set it OVERRIDES ``steps`` (the real
    # training length becomes ``effective_steps`` = ceil(budget / (batch*block))),
    # letting a run scale up to ~1e9 tokens for H100/H200. ``None`` (default) keeps
    # the historical ``steps``-driven behavior exactly — so default/CPU/heuristic
    # runs stay cheap and unchanged.
    train_token_budget: int | None = Field(
        default=None, ge=1, le=_MAX_TRAIN_TOKEN_BUDGET
    )
    learning_rate: float = Field(default=3e-4, gt=0.0, le=1.0)
    seed: int = Field(default=0, ge=0)
    val_fraction: float = Field(default=0.1, gt=0.0, lt=1.0)
    # --- record_01 (nanogpt-speedrun) optimizer schedule + regularization -----
    # These upgrade the REAL (sandbox) trainer's training recipe to the
    # ``leloy/nanogpt-speedrun`` record_01 baseline — AdamW + LR warmup + cosine
    # cooldown + decoupled weight decay + grad clipping + contiguous-window
    # batching + averaged runs. They are consumed only by the real trainer's
    # sandbox script; the default heuristic backend ignores them entirely (so its
    # synthetic loss/cost calibration is unchanged). Each knob keeps a faithful
    # record_01 default, so a default-config real run differs from the old
    # constant-LR plain-AdamW path ONLY by the improved schedule/batching.
    weight_decay: float = Field(default=0.1, ge=0.0, le=1.0)
    # record_01 AdamW moments + epsilon (the Karpathy GPT-2 reproduction values the
    # first speedrun record inherits): betas (0.9, 0.95), eps 1e-8.
    adam_beta1: float = Field(default=0.9, gt=0.0, lt=1.0)
    adam_beta2: float = Field(default=0.95, gt=0.0, lt=1.0)
    adam_eps: float = Field(default=1e-8, gt=0.0, le=1e-1)
    # Global-norm gradient clip applied before every ``opt.step()`` (record_01: 1.0).
    grad_clip: float = Field(default=1.0, ge=0.0)
    # LR warmup length (steps). ``None`` (default) derives a sensible fraction of the
    # run, ``min(256, max(1, effective_steps // 10))``; an explicit value is clamped
    # to the run length. See ``effective_warmup_steps``.
    warmup_steps: int | None = Field(default=None, ge=0)
    # Cosine cooldown floor as a fraction of the peak LR (record_01 decays to ~10%).
    lr_min_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    # Number of independent train+eval runs (distinct seeds) whose val loss/accuracy
    # are AVERAGED into a lower-variance signal. Default 1 keeps the historical cost
    # AND calibration unchanged; >1 multiplies compute (FLOPs/tokens summed across
    # runs, so cost accounting bills every run).
    n_train_runs: int = Field(default=1, ge=1, le=64)
    # --- real-trainer backend selection (only used when use_real_trainer) -----
    # Static, pre-runtime hint only: shapes ``load_environment``'s harness.runtime
    # and ``load_tasks()``'s task image/resources/timeout declarations, and gates
    # the Modal timeout ceiling check below. It is NEVER read when selecting which
    # trainer actually runs at score time -- that is driven purely by the live
    # harness runtime's ``type`` via ``trainer.RuntimeSelectedTrainer``. No default:
    # a real-trainer run must explicitly pick ``'docker'`` or ``'modal'``.
    runtime_backend: Literal["docker", "modal"] | None = None
    # Retained for config compatibility, but remote Docker is not supported by
    # the shared harness-runtime path. ``load_environment`` rejects a non-None
    # value for the docker backend. Ignored by modal.
    docker_host: str | None = None
    # Modal GPU type for the modal backend. Maps to a Modal GPU specifier string:
    # ``"H100"`` → ``"H100"``, ``"H200"`` → ``"H200"``, ``"A100"`` → ``"A100-80GB"``,
    # anything else → ``"L4"`` (default, cheapest; adequate for the 55M-param default
    # student). L4 billing is ~$0.80/hr per-second; the default 200-step run takes
    # ~50s ≈ $0.011. Ignored by the docker backend.
    modal_gpu: str = Field(default="L4", min_length=1)
    # Sandbox/container backend settings (used by both real-trainer backends).
    docker_image: str = Field(default=_DEFAULT_DOCKER_TRAINER_IMAGE, min_length=1)
    gpu_count: int = Field(default=1, ge=0, le=8)
    cpu_cores: int = Field(default=4, ge=1, le=256)
    memory_gb: int = Field(default=16, ge=1, le=2048)
    disk_size_gb: int = Field(default=20, ge=1, le=10_000)
    # Upper char cap on the corpus blob uploaded to the sandbox. ``None`` (default)
    # derives it from the training budget (so a large run is not silently capped at
    # the historical ~1.25M-unique-token corpus); an explicit value overrides.
    max_corpus_chars: int | None = Field(default=None, ge=1, le=_MAX_CORPUS_CHARS)
    # Sandbox/container lifetime / command timeout (minutes). ``None`` (default)
    # derives a budget-sized timeout (floored at the historical 30). The Modal 24h
    # ceiling is enforced in ``_check_modal_timeout_ceiling`` (the self-hosted
    # docker backend has no such cap), so the static field bound is just the lower
    # bound.
    timeout_minutes: int | None = Field(default=None, ge=1)
    upload_timeout_seconds: float = Field(default=120.0, gt=0.0)

    @property
    def effective_steps(self) -> int:
        """Training steps actually run: derived from ``train_token_budget`` when
        set (ceil(budget / (batch_size * block_size))), else the explicit ``steps``."""
        if self.train_token_budget is None:
            return self.steps
        per_step = self.batch_size * self.block_size
        return max(1, math.ceil(self.train_token_budget / per_step))

    @property
    def effective_warmup_steps(self) -> int:
        """LR warmup steps actually used by the real trainer.

        An explicit ``warmup_steps`` wins (clamped to the run length so warmup never
        exceeds ``effective_steps``); otherwise a sensible fraction of the run,
        ``min(256, max(1, effective_steps // 10))``. Mirrors record_01's short
        warmup-then-cooldown schedule and stays >= 1 so the linear ramp never
        divides by zero.
        """
        if self.warmup_steps is not None:
            return min(self.warmup_steps, self.effective_steps)
        return min(256, max(1, self.effective_steps // 10))

    @property
    def effective_train_tokens(self) -> int:
        """Tokens the fixed schedule consumes: ``effective_steps * batch * block``.

        This is the token count the sandbox script trains on and bills FLOPs for
        (``6 * n_params * effective_train_tokens``), so FLOP cost scales with the
        budget without any separate accounting change.
        """
        return self.effective_steps * self.batch_size * self.block_size

    @property
    def effective_max_corpus_chars(self) -> int:
        """Char cap on the uploaded corpus blob.

        An explicit ``max_corpus_chars`` wins; otherwise it grows with the training
        budget (``_CHARS_PER_TOKEN * effective_train_tokens``), floored at the
        historical ``_MIN_CORPUS_CHARS`` and ceilinged at ``_MAX_CORPUS_CHARS``.
        """
        if self.max_corpus_chars is not None:
            return self.max_corpus_chars
        derived = _CHARS_PER_TOKEN * self.effective_train_tokens
        return max(_MIN_CORPUS_CHARS, min(derived, _MAX_CORPUS_CHARS))

    @property
    def effective_timeout_minutes(self) -> int:
        """Sandbox lifetime / command timeout (minutes).

        An explicit ``timeout_minutes`` wins; otherwise derived from the budget
        (setup overhead + tokens / throughput), floored at the historical 30.
        Modal additionally clamps the derived value to the
        ``_MAX_SANDBOX_TIMEOUT_MINUTES`` (24h) platform max via this property;
        Docker has no such ceiling, so its derived timeout is left uncapped.
        """
        if self.timeout_minutes is not None:
            return self.timeout_minutes
        derived = max(
            _MIN_SANDBOX_TIMEOUT_MINUTES,
            _SANDBOX_SETUP_MINUTES
            + math.ceil(self.effective_train_tokens / _SANDBOX_TOKENS_PER_MINUTE),
        )
        if self.runtime_backend == "modal":
            return min(_MAX_SANDBOX_TIMEOUT_MINUTES, derived)
        return derived

    @property
    def effective_scoring_timeout_seconds(self) -> float:
        """Framework deadline for the full harness-runtime scoring phase."""
        margin = max(300.0, self.upload_timeout_seconds * 4 + 60.0)
        return self.effective_timeout_minutes * 60 + margin

    @model_validator(mode="after")
    def _check_student_dims(self) -> "ProxyStudentConfig":
        # The modern student (student_model.GPT) requires: n_embd divisible by
        # n_head, a head_dim that is a multiple of 4 (half-truncate RoPE), and an
        # even depth >= 2 (symmetric U-net encoder/decoder skips). Reject anything
        # unbuildable here so it fails fast instead of inside the GPU sandbox.
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )
        head_dim = self.n_embd // self.n_head
        if head_dim % 4 != 0:
            raise ValueError(
                f"head_dim (n_embd/n_head = {head_dim}) must be a multiple of 4 "
                "for half-truncate RoPE"
            )
        if self.n_layer % 2 != 0:
            raise ValueError(
                f"n_layer ({self.n_layer}) must be even for the U-net encoder/"
                "decoder skip structure"
            )
        return self

    @model_validator(mode="after")
    def _check_modal_timeout_ceiling(self) -> "ProxyStudentConfig":
        if (
            self.runtime_backend == "modal"
            and self.timeout_minutes is not None
            and self.timeout_minutes > _MAX_SANDBOX_TIMEOUT_MINUTES
        ):
            raise ValueError(
                f"timeout_minutes ({self.timeout_minutes}) exceeds the Modal 24h "
                f"sandbox maximum ({_MAX_SANDBOX_TIMEOUT_MINUTES}); lower it"
            )
        return self


class CuratorConfig(BaseModel):
    """Central, validated configuration for the environment and reward."""

    cutoff_date: str = "2024-12-31"
    token_budget: int = Field(default=1_000_000, gt=0)
    # Safety-only harness cap. It is deliberately absent from prompts, rewards,
    # and metrics: token_budget is the sole optimization budget.
    max_turns: int = Field(default=64, ge=1, le=1000)

    candidate_limit: int = Field(
        default=8,
        ge=1,
        le=1000,
        description=(
            "Maximum dataset IDs used by trace-based manifest recovery/fallback only."
        ),
    )
    allow_local_sources: bool = True
    max_local_source_bytes: int = Field(
        default=33_554_432, ge=1, le=1_073_741_824
    )

    # Reward coefficients: R = a1*Perf_scaled_to_target - l1*Leakage
    alpha_perf: float = Field(default=1.0, ge=0.0)
    lambda_leakage: float = Field(default=1.0, ge=0.0)

    # --- baseline-relative Perf signal (additive; default-on) -----------------
    # ``perf_baseline_loss`` is the reference val cross-entropy of a neutral
    # (untrained) student. It is a CHEAP CONSTANT — no second training run is ever
    # performed, so default runs do not pay for it. Default: the CE of a uniform
    # student over the padded GPT-2 vocab (ln(50304) ~= 10.83 nats/token), the
    # no-information reference for the real trainer's nats/token CE. (The default
    # HeuristicProxyTrainer uses a smaller synthetic ``reference_loss`` of 5.0.)
    perf_baseline_loss: float = Field(default=math.log(50304), gt=0.0)
    # Target val cross-entropy that maps to Perf=1.0 under the default
    # baseline-relative path. Default: the nanoGPT speedrun target.
    perf_target_loss: float = Field(default=3.28, gt=0.0)
    # When True (the default), the Perf REWARD term is the linear improvement
    # from ``perf_baseline_loss`` to ``perf_target_loss`` instead of
    # ``exp(-loss)``.  Set to False only when the absolute loss is meaningful
    # (e.g. a tiny toy model with loss < 1): for real LMs (loss ~ 9 nats/token,
    # 50K vocab) exp(-loss) ≈ 0 and collapses the reward to zero.  The raw
    # relative-improvement diagnostic (perf_vs_baseline) is always surfaced
    # regardless of this flag.
    baseline_relative_perf: bool = True

    # Bounded concurrency and robustness knobs for external (HF/sandbox) access.
    max_concurrent_fetches: int = Field(default=8, ge=1, le=1024)
    max_concurrent_training: int = Field(default=1, ge=1, le=256)
    fetch_timeout_seconds: float = Field(default=30.0, gt=0.0)
    fetch_timeout_per_doc_seconds: float = Field(default=0.25, ge=0.0)
    fetch_max_attempts: int = Field(default=3, ge=1, le=20)

    prices: CostPrices = Field(default_factory=CostPrices)
    proxy_student: ProxyStudentConfig = Field(default_factory=ProxyStudentConfig)
    # Held-out validation set for the downstream cross-entropy (Perf) signal.
    # Defaults to the NanoGPT speedrun set (FineWeb sample-10BT GPT-2-BPE val
    # tokens). Consumed by the real (sandbox) proxy-student trainer.
    validation_set: ValidationSetConfig = Field(default_factory=ValidationSetConfig)
    use_real_trainer: bool = False

    @model_validator(mode="after")
    def _check_perf_target_below_baseline(self) -> "CuratorConfig":
        if self.perf_baseline_loss <= self.perf_target_loss:
            raise ValueError(
                "perf_baseline_loss must be greater than perf_target_loss "
                f"(got baseline={self.perf_baseline_loss}, "
                f"target={self.perf_target_loss})"
            )
        return self
