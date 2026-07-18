"""Pydantic contracts for the pretraining-data curation environment.

Everything the agent produces and everything the reward consumes is expressed
here so the manifest, configuration, and scoring result have one strict home.
"""

from __future__ import annotations

import math
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, Field, model_serializer, model_validator

from ..gpu.train_gpt import scheduled_presentation_tokens, steps_for_token_budget
from .hf_access import CHARS_PER_TOKEN
from .val_set import ValidationSetConfig

MANIFEST_FILENAME = "manifest.json"

# Where finalize() obtained the production workspace outcome. Only
# ``workspace_file`` sets ``finalized=1``. Absent vs present-but-invalid are
# distinct failure reasons; assistant/trace candidates never override them.
ManifestProvenance = Literal[
    "workspace_file",
    "invalid_workspace_file",
    "assistant_message",
    "trace_fallback",
    "missing",
]
MANIFEST_PROVENANCE_WORKSPACE_FILE: ManifestProvenance = "workspace_file"
MANIFEST_PROVENANCE_INVALID_WORKSPACE_FILE: ManifestProvenance = (
    "invalid_workspace_file"
)
MANIFEST_PROVENANCE_ASSISTANT_MESSAGE: ManifestProvenance = "assistant_message"
MANIFEST_PROVENANCE_TRACE_FALLBACK: ManifestProvenance = "trace_fallback"
MANIFEST_PROVENANCE_MISSING: ManifestProvenance = "missing"

# Non-production telemetry candidate when the workspace file did not finalize.
ManifestCandidate = Literal["assistant_message", "trace_fallback"]
MANIFEST_CANDIDATE_ASSISTANT_MESSAGE: ManifestCandidate = "assistant_message"
MANIFEST_CANDIDATE_TRACE_FALLBACK: ManifestCandidate = "trace_fallback"

_RESERVED_WORKSPACE_FILES = frozenset(
    {
        MANIFEST_FILENAME,
        "corpus.txt",
        "config.json",
        "train.py",
        "val.bin",
    }
)

# --- proxy-student budget / sandbox derivation constants -------------------
# The real (sandbox) trainer's training length, corpus cap, and sandbox lifetime
# all derive from ``train_token_budget`` so a single token knob scales the whole
# run from a cheap default up to an H100/H200-scale few-hundred-million-token run.
# When set, steps are chosen so *scheduled* presentations under ``batch_stage_muls``
# meet the budget (not base-batch alone); see ``train_gpt.py``.
_MAX_TRAIN_TOKEN_BUDGET = 1_000_000_000  # generous H100/H200 upper bound
_MIN_CORPUS_CHARS = 5_000_000  # historical default cap; floor for small budgets
_MAX_CORPUS_CHARS = 2_000_000_000  # absolute ceiling on the uploaded corpus blob
# Sandbox lifetime derivation. Modal v1 runtimes cap a remote sandbox at 24 hours.
_MIN_SANDBOX_TIMEOUT_MINUTES = (
    30  # historical default; floor keeps small budgets unchanged
)
_SANDBOX_SETUP_MINUTES = 15  # image pull + pip(tiktoken) + uploads + val download
_SANDBOX_TOKENS_PER_MINUTE = 500_000  # conservative floor for benchmark-sized runs


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


class StudentArchConfig(BaseModel):
    """Fixed GPT-2-scale student architecture. Only the training corpus varies.

    Defaults mirror ``train_gpt.GPT2_SMALL`` (modded-nanogpt / speedrun record_01
    architecture): 12 layers, 768-wide, 6 heads, ~278M instantiated params.
    Every field carries bounds so an out-of-range spec fails fast with a clear
    ``ValidationError`` instead of producing a degenerate sandbox training job.
    """

    n_layer: int = Field(default=12, ge=2, le=64)
    n_head: int = Field(default=6, ge=1, le=64)
    n_embd: int = Field(default=768, ge=8, le=4096)
    # Modern (modded-nanogpt) student knobs: ReLU**2 MLP width ratio, the tanh
    # logit-softcap constant, and the number of distinct sparse value-embedding
    # tables (SparsifyEmbeds; clamped to n_layer//2 by the model). The model
    # itself lives in ``train_gpt.py``.
    mlp_ratio: int = Field(default=4, ge=1, le=16)
    lm_head_softcap: float = Field(default=30.0, gt=0.0, le=1000.0)
    num_value_embeds: int = Field(default=3, ge=1, le=32)
    # SDPA softmax scale (speedrun records use 0.12, not 1/sqrt(head_dim)).
    attn_scale: float = Field(default=0.12, gt=0.0, le=10.0)
    # Optional causal sliding-window attention (SDPA mask). ``None`` = full context.
    sliding_window_size: int | None = Field(default=None, ge=2)
    block_size: int = Field(default=1024, ge=8, le=8192)

    @model_validator(mode="after")
    def _check_student_dims(self) -> "StudentArchConfig":
        # The modern student (train_gpt.GPT) requires: n_embd divisible by
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


class StudentRunConfig(BaseModel):
    """Training-run shape: batching, length/budget, seeding, and validation."""

    batch_size: int = Field(default=16, ge=1, le=4096)
    # Cap peak training activation memory without changing scheduled effective-batch
    # token accounting. When set (e.g. 16 on A100-80GB), each optimizer step still
    # consumes ``effective_batch = batch_size * stage_mul`` windows, but forward /
    # backward run in loss-scaled microbatches so full-vocab logits never materialize
    # for the 2×/3× schedule stages at once. ``None`` (default) = one full effective
    # batch per step (legacy behavior).
    train_microbatch_size: int | None = Field(default=None, ge=1, le=4096)
    # Held-out validation microbatch. Separate from training ``batch_size`` so the
    # final CE pass can stay under the A100-80GB ceiling while Muon/Adam state is
    # still resident. ``None`` (default) reuses ``batch_size`` for backwards
    # compatibility; set explicitly (e.g. 1) for large single-GPU runs.
    val_batch_size: int | None = Field(default=None, ge=1, le=4096)
    # Max tokens scored per lm_head/softcap chunk during validation. Caps the
    # (N, vocab) fp32 temporaries without changing mean CE / accuracy semantics.
    # ``None`` scores a whole microbatch at once (legacy behavior).
    val_logit_chunk_tokens: int | None = Field(default=None, ge=1, le=1_048_576)
    steps: int = Field(default=200, ge=1, le=100_000)
    # Token-oriented training budget. When set it OVERRIDES ``steps``: the real
    # training length becomes the minimal ``effective_steps`` whose *scheduled*
    # token presentations (base batch × ``batch_stage_muls`` over stage fracs)
    # meet the budget — not ``ceil(budget / (batch*block))``. See
    # ``train_gpt.steps_for_token_budget`` for the single-step overshoot
    # boundary rule. ``train_microbatch_size`` does not affect this (memory-only).
    # ``None`` (default) keeps the historical ``steps``-driven behavior exactly —
    # so default/CPU/heuristic runs stay cheap and unchanged.
    train_token_budget: int | None = Field(
        default=None, ge=1, le=_MAX_TRAIN_TOKEN_BUDGET
    )
    seed: int = Field(default=0, ge=0)
    val_fraction: float = Field(default=0.1, gt=0.0, lt=1.0)
    # Number of independent train+eval runs (distinct seeds) whose val loss/accuracy
    # are AVERAGED into a lower-variance signal. Default 1 keeps the historical cost
    # AND calibration unchanged; >1 multiplies compute (FLOPs/tokens summed across
    # runs, so cost accounting bills every run).
    n_train_runs: int = Field(default=1, ge=1, le=64)


class StudentOptimizerConfig(BaseModel):
    """Training recipe: modded-nanogpt speedrun Muon vs legacy record_01 AdamW."""

    training_recipe: Literal["speedrun_muon", "record_01_adamw"] = "speedrun_muon"
    # Classic Muon (2-D block weights)
    muon_lr: float = Field(default=0.023, gt=0.0, le=1.0)
    muon_weight_decay: float = Field(default=1.2, ge=0.0, le=10.0)
    muon_momentum_min: float = Field(default=0.85, gt=0.0, lt=1.0)
    muon_momentum_max: float = Field(default=0.95, gt=0.0, lt=1.0)
    muon_warmup_steps: int | None = Field(default=None, ge=0)
    muon_cooldown_steps: int | None = Field(default=None, ge=0)
    # AdamW groups (embed / head / value embeds / scalars). ``adam_eps`` is the
    # canonical Speedrun epsilon; the legacy record_01 optimizer has its own
    # explicitly named ``record_adam_eps`` below so this value cannot be shadowed.
    adam_lr: float = Field(default=0.008, gt=0.0, le=1.0)
    adam_eps: float = Field(default=1e-10, gt=0.0, le=1e-1)
    adam_weight_decay: float = Field(default=0.005, ge=0.0, le=1.0)
    embed_lr_mul: float = Field(default=1.0, gt=0.0, le=1000.0)
    lm_head_lr_mul: float = Field(default=1.0, gt=0.0, le=1000.0)
    value_embed_lr_mul: float = Field(default=75.0, gt=0.0, le=1000.0)
    scalar_lr_mul: float = Field(default=5.0, gt=0.0, le=1000.0)
    embed_wd_mul: float = Field(default=150.0, ge=0.0, le=1000.0)
    lm_head_wd_mul: float = Field(default=150.0, ge=0.0, le=1000.0)
    value_embed_wd_mul: float = Field(default=5.0, ge=0.0, le=1000.0)
    scalar_wd_mul: float = Field(default=0.0, ge=0.0, le=1000.0)
    adam_on_odd_steps: bool = True
    # --- legacy record_01 (nanogpt-speedrun) AdamW path ----------------------
    # Used when training_recipe='record_01_adamw': AdamW + LR warmup + cosine
    # cooldown + decoupled weight decay + grad clipping + contiguous-window
    # batching. Each knob keeps a faithful record_01 default; the default
    # heuristic backend ignores them entirely (so its synthetic loss/cost
    # calibration is unchanged).
    learning_rate: float = Field(default=3e-4, gt=0.0, le=1.0)
    weight_decay: float = Field(default=0.1, ge=0.0, le=1.0)
    # record_01 AdamW moments + epsilon (the Karpathy GPT-2 reproduction values the
    # first speedrun record inherits): betas (0.9, 0.95), eps 1e-8.
    adam_beta1: float = Field(default=0.9, gt=0.0, lt=1.0)
    adam_beta2: float = Field(default=0.95, gt=0.0, lt=1.0)
    record_adam_eps: float = Field(default=1e-8, gt=0.0, le=1e-1)
    # Global-norm gradient clip applied before every optimizer update.
    # Default 0.0 disables clipping (modern speedrun recipe); set >0 to enable.
    grad_clip: float = Field(default=0.0, ge=0.0)
    # LR warmup length (steps). ``None`` (default) derives a sensible fraction of the
    # run, ``min(256, max(1, effective_steps // 10))``; an explicit value is clamped
    # to the run length. See ``ProxyStudentConfig.effective_warmup_steps``.
    warmup_steps: int | None = Field(default=None, ge=0)
    # Cosine cooldown floor as a fraction of the peak LR (record_01 decays to ~10%).
    lr_min_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    # --- scale referencing ----------------------------------------------------
    # Weight-decay timescale reference. Decoupled weight decay compounds once
    # per optimizer step (cumulative shrinkage ~ lr * wd * steps), so a recipe
    # tuned at the speedrun record's ~1,390 steps over-decays at 12k steps and
    # under-decays at 150. When set (1390 = the record's step count), the
    # emitted Muon/Adam weight decay are scaled by ``wd_ref_steps /
    # effective_steps`` so cumulative decay matches the reference run length.
    # ``None`` (default) emits the raw configured values unchanged.
    wd_ref_steps: int | None = Field(default=None, ge=1)


class StudentScheduleConfig(BaseModel):
    """Batch-size + LR schedule (BatchSizeSchedule record defaults)."""

    batch_schedule_enabled: bool = True
    batch_stage_fracs: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)
    batch_stage_muls: tuple[int, int, int] = (1, 2, 3)
    lr_stage_muls: tuple[float, float, float] = (1.0, 1.52, 1.73)
    # Fraction of the run over which the LR decays linearly to the floor.
    # 0.40 matches the current speedrun record (decay spans the final 40%).
    lr_cooldown_frac: float = Field(default=0.40, ge=0.0, le=1.0)
    lr_cooldown_floor: float = Field(default=0.15, ge=0.0, le=1.0)


class StudentFeaturesConfig(BaseModel):
    """Portable proxy-student features (all off by default except EOS packing)."""

    # Bigram hash embedding on 1/4 of model_dim with sign trick
    bigram_hash_embed: bool = False
    # Learned 1-token lookback smear on the embedding stream
    smear_embed: bool = False
    # Upstream stationary-key one-token shift (disabled by default)
    partial_key_offset: bool = False
    # Paired head attention (even num_heads required)
    paired_head: bool = False
    # MUDD multi-layer skip connection pairs
    mudd_pairs: int = Field(default=0, ge=0, le=16)
    # Learnable per-head XSA projection removal on eligible attention layers
    xsa_enabled: bool = False
    xsa_pairs: int = Field(default=0, ge=0, le=16)
    # Single activation input for the last K attention layers
    single_act_last_k: int = Field(default=0, ge=0, le=16)
    # Exponential decay factor for residual stream (None = disabled)
    exp_residual_decay: float | None = Field(default=None, gt=0.0, le=1.0)
    # Multi-token prediction (number of extra future-token heads)
    multi_token_pred: int = Field(default=0, ge=0, le=8)
    # Encode each source document with GPT-2 EOT/BOS and plan packed windows:
    # long docs keep intra-document starts; short docs pack into fixed blocks.
    # Disable only for legacy flat-stream compatibility.
    eos_aligned_batches: bool = True
    # Canonical per-document token cap, including the leading EOT/BOS token.
    # Historical spellings remain accepted by the compatibility validator on
    # ``ProxyStudentConfig``.
    max_document_tokens: int | None = Field(default=None, ge=8)
    # True 2-step gradient accumulation for embed+lm_head before update
    grad_accum_embed_head_steps: int = Field(default=1, ge=1, le=8)
    # True max sequence length schedule (warm up block size)
    seq_len_schedule: bool = False
    # Fraction of training at which to untie embed and lm_head (0 = never)
    untie_at_frac: float = Field(default=0.0, ge=0.0, le=1.0)
    # Cautious weight decay tied to LR
    cautious_wd: bool = False
    # NorMuon (normalized Muon updates); on by default to match modern speedrun.
    nor_muon: bool = True
    # Polar Express (ONI-based orthogonalization in Muon)
    polar_express: bool = False


class StudentSandboxConfig(BaseModel):
    """Sandbox sizing; derived from the training budget unless overridden.

    Runtime placement and resources belong to the native v1 harness runtime.
    """

    # Upper char cap on the corpus blob uploaded to the sandbox. ``None`` (default)
    # derives it from the training budget (so a large run is not silently capped at
    # the historical ~1.25M-unique-token corpus); an explicit value overrides.
    max_corpus_chars: int | None = Field(default=None, ge=1, le=_MAX_CORPUS_CHARS)
    # Sandbox/container lifetime / command timeout (minutes). ``None`` (default)
    # derives a budget-sized timeout (floored at the historical 30). The Modal 24h
    # ceiling is enforced against the selected native harness runtime by
    # ``load_environment``; the static field bound is just the lower bound.
    timeout_minutes: int | None = Field(default=None, ge=1)
    upload_timeout_seconds: float = Field(default=120.0, gt=0.0)


# Owning submodel for each flat (legacy) proxy-student field name; drives the
# compatibility routing in ``ProxyStudentConfig._accept_legacy_flat_fields``.
_STUDENT_SUBMODEL_FIELDS: dict[str, frozenset[str]] = {
    "arch": frozenset(StudentArchConfig.model_fields),
    "run": frozenset(StudentRunConfig.model_fields),
    "optimizer": frozenset(StudentOptimizerConfig.model_fields),
    "schedule": frozenset(StudentScheduleConfig.model_fields),
    "features": frozenset(StudentFeaturesConfig.model_fields),
    "sandbox": frozenset(StudentSandboxConfig.model_fields),
}


class ProxyStudentConfig(BaseModel):
    """Fixed proxy-student specification, grouped by ownership.

    ``arch`` (model shape), ``run`` (batching/length/seeding), ``optimizer``
    (speedrun Muon + legacy record_01 recipes), ``schedule`` (batch/LR stages),
    ``features`` (portable architecture features), and ``sandbox`` (upload and
    lifetime sizing). Legacy flat field names are accepted at construction and
    routed into their owning submodel, so predecessor configs keep loading.
    """

    arch: StudentArchConfig = Field(default_factory=StudentArchConfig)
    run: StudentRunConfig = Field(default_factory=StudentRunConfig)
    optimizer: StudentOptimizerConfig = Field(default_factory=StudentOptimizerConfig)
    schedule: StudentScheduleConfig = Field(default_factory=StudentScheduleConfig)
    features: StudentFeaturesConfig = Field(default_factory=StudentFeaturesConfig)
    sandbox: StudentSandboxConfig = Field(default_factory=StudentSandboxConfig)

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_flat_fields(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        values = dict(data)
        if "eos_positions" in values:
            raise ValueError(
                "eos_positions is no longer accepted: document boundaries are "
                "derived from the preserved source-document list"
            )
        aliases = [name for name in ("max_doc_len", "max_doc_length") if name in values]
        if aliases and "max_document_tokens" in values:
            if any(values[name] != values["max_document_tokens"] for name in aliases):
                raise ValueError("conflicting max-document token settings")
        elif aliases:
            values["max_document_tokens"] = values[aliases[0]]
        for name in aliases:
            values.pop(name, None)
        # Before the duplicate-field fix, legacy record_01 configs used
        # ``adam_eps`` for their optimizer. Preserve that input meaning while
        # keeping ``adam_eps`` canonical for Speedrun configs.
        if (
            values.get("training_recipe") == "record_01_adamw"
            and "adam_eps" in values
            and "record_adam_eps" not in values
        ):
            values["record_adam_eps"] = values["adam_eps"]
        # Route remaining flat keys into their owning submodel. An explicit
        # nested value for the same field is a contradiction, not a tiebreak.
        for group, fields in _STUDENT_SUBMODEL_FIELDS.items():
            flat = [key for key in values if key in fields]
            if not flat:
                continue
            sub = values.setdefault(group, {})
            if not isinstance(sub, dict):
                raise ValueError(
                    f"cannot combine flat proxy-student field(s) {sorted(flat)} "
                    f"with a pre-built {group!r} submodel"
                )
            for key in flat:
                if key in sub and sub[key] != values[key]:
                    raise ValueError(
                        f"{key!r} set both flat and under {group!r} with "
                        "different values"
                    )
                sub[key] = values.pop(key)
        return values

    def __getattr__(self, name: str) -> object:
        # Deprecated flat access (``config.steps`` for ``config.run.steps``):
        # predecessor code and tests read proxy-student fields without the
        # owning submodel, so route those names transparently.
        if not name.startswith("_"):
            for group, fields in _STUDENT_SUBMODEL_FIELDS.items():
                if name in fields:
                    return getattr(getattr(self, group), name)
        return super().__getattr__(name)  # pyright: ignore[reportAttributeAccessIssue]

    @property
    def effective_steps(self) -> int:
        """Training steps actually run.

        With ``run.train_token_budget`` set: minimal N whose staged presentations
        meet the budget (see ``train_gpt.steps_for_token_budget``).
        Otherwise: the explicit ``run.steps`` field (unchanged historical path).
        """
        if self.run.train_token_budget is None:
            return self.run.steps
        return steps_for_token_budget(
            self.run.train_token_budget,
            batch_size=self.run.batch_size,
            block_size=self.arch.block_size,
            batch_stage_muls=self.schedule.batch_stage_muls,
            batch_stage_fracs=self.schedule.batch_stage_fracs,
            batch_schedule_enabled=self.schedule.batch_schedule_enabled,
            seq_len_schedule=self.features.seq_len_schedule,
        )

    @property
    def effective_warmup_steps(self) -> int:
        """LR warmup steps for the legacy ``record_01_adamw`` recipe only."""
        if self.optimizer.warmup_steps is not None:
            return min(self.optimizer.warmup_steps, self.effective_steps)
        return min(256, max(1, self.effective_steps // 10))

    @property
    def effective_train_tokens(self) -> int:
        """Tokens billed / consumed for the configured training length.

        * No ``train_token_budget``: historical ``effective_steps * batch * block``
          (unchanged even when the speedrun batch schedule is enabled).
        * With ``train_token_budget``: actual scheduled presentations under
          ``batch_stage_muls`` / fracs (see ``batch_schedule``), matching sandbox
          ``tokens_trained``. ``train_microbatch_size`` does not change the total.

        FLOP billing uses ``6 * n_params * effective_train_tokens``.
        """
        if self.run.train_token_budget is None:
            return self.effective_steps * self.run.batch_size * self.arch.block_size
        return scheduled_presentation_tokens(
            self.effective_steps,
            batch_size=self.run.batch_size,
            block_size=self.arch.block_size,
            batch_stage_muls=self.schedule.batch_stage_muls,
            batch_stage_fracs=self.schedule.batch_stage_fracs,
            batch_schedule_enabled=self.schedule.batch_schedule_enabled,
            seq_len_schedule=self.features.seq_len_schedule,
        )

    @property
    def effective_max_corpus_chars(self) -> int:
        """Char cap on the uploaded corpus blob.

        An explicit ``sandbox.max_corpus_chars`` wins; otherwise it grows with the
        training budget (``CHARS_PER_TOKEN * effective_train_tokens``), floored at
        the historical ``_MIN_CORPUS_CHARS`` and ceilinged at ``_MAX_CORPUS_CHARS``.
        """
        if self.sandbox.max_corpus_chars is not None:
            return self.sandbox.max_corpus_chars
        derived = CHARS_PER_TOKEN * self.effective_train_tokens
        return max(_MIN_CORPUS_CHARS, min(derived, _MAX_CORPUS_CHARS))

    @property
    def effective_timeout_minutes(self) -> int:
        """Sandbox lifetime / command timeout (minutes).

        An explicit ``sandbox.timeout_minutes`` wins; otherwise derived from the
        budget (setup overhead + tokens / throughput), floored at the historical
        30. Backend-specific ceilings are validated against the selected native
        harness runtime by ``load_environment``.
        """
        if self.sandbox.timeout_minutes is not None:
            return self.sandbox.timeout_minutes
        return max(
            _MIN_SANDBOX_TIMEOUT_MINUTES,
            _SANDBOX_SETUP_MINUTES
            + math.ceil(self.effective_train_tokens / _SANDBOX_TOKENS_PER_MINUTE),
        )

    @property
    def max_doc_len(self) -> int | None:
        """Deprecated attribute alias for ``features.max_document_tokens``."""
        return self.features.max_document_tokens

    def training_payload(self, *, tokenizer: str = "gpt2") -> dict[str, object]:
        """Flat JSON config blob consumed by the sandbox / self-score trainers.

        Assembled from the submodel dumps so a newly added field cannot be
        forgotten here; only the derived overrides are spelled out. The sandbox
        group is deliberately absent — it sizes the upload, not the training.
        """
        payload: dict[str, object] = {
            **self.arch.model_dump(mode="json"),
            **self.run.model_dump(
                mode="json", exclude={"steps", "train_token_budget"}
            ),
            **self.optimizer.model_dump(
                mode="json", exclude={"warmup_steps", "wd_ref_steps"}
            ),
            **self.schedule.model_dump(mode="json"),
            **self.features.model_dump(mode="json"),
            "steps": self.effective_steps,
            "warmup_steps": self.effective_warmup_steps,
            "tokenizer": tokenizer,
        }
        if self.optimizer.wd_ref_steps is not None:
            # Hold cumulative decay constant across run lengths (see field doc).
            ratio = self.optimizer.wd_ref_steps / self.effective_steps
            payload["muon_weight_decay"] = self.optimizer.muon_weight_decay * ratio
            payload["adam_weight_decay"] = self.optimizer.adam_weight_decay * ratio
        return payload


class CuratorConfig(BaseModel):
    """Central, validated configuration for the environment and reward."""

    cutoff_date: str = "2024-12-31"
    token_budget: int = Field(default=1_000_000, gt=0)

    candidate_limit: int = Field(
        default=8,
        ge=1,
        le=1000,
        description=(
            "Maximum dataset IDs used by trace-based manifest recovery/fallback only."
        ),
    )
    # Opt-in debug/compat path. When False (default), production finalize never
    # synthesizes a manifest from regex-scraped trace IDs. Even when True, that
    # path records provenance ``trace_fallback`` with ``finalized=0`` so it cannot
    # pass production success gates or trigger materialize/train.
    allow_trace_id_manifest_fallback: bool = False
    allow_local_sources: bool = True
    max_local_source_bytes: int = Field(default=33_554_432, ge=1, le=1_073_741_824)

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
    # Convex power-law scaling exponent γ for the baseline-relative perf signal.
    # When gamma > 1, progress near the target loss is amplified (p >= 0 → p^γ)
    # while the negative (below-baseline) branch stays linear (p < 0 → p).
    # γ=1.0 recovers the linear behavior exactly. Must be finite and > 0.
    perf_scaling_exponent: float = Field(default=2.0)
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

    @model_validator(mode="after")
    def _check_perf_scaling_exponent(self) -> "CuratorConfig":
        exp = self.perf_scaling_exponent
        if not math.isfinite(exp) or exp <= 0:
            raise ValueError(
                f"perf_scaling_exponent must be finite and > 0 (got {exp})"
            )
        return self


class ScoringResult(BaseModel, frozen=True):
    """One rollout's heavy scoring pass, shared by the reward and diagnostics.

    Produced exactly once per rollout by ``CuratorScorer.compute_scoring`` and
    retained on rollout state, so the keyed reward and the metric surface read
    the same materialize/train/screen outcome without recomputing it. The
    defaults are the empty-rollout sentinels; only the perf-curve constants are
    config-dependent and therefore required.
    """

    perf: float = 0.0
    leakage_score: float = 0.0
    num_contaminated_matches: int = 0
    decon_error: bool = False
    val_screen_skipped: bool = False
    loss: float = 0.0
    accuracy: float = 0.0
    flops: float = 0.0
    tokens: int = 0
    num_sources: int = 0
    budget_fill_ratio: float = 0.0
    perf_vs_baseline: float = 0.0
    perf_baseline_loss: float
    perf_target_loss: float
    perf_scaling_exponent: float
