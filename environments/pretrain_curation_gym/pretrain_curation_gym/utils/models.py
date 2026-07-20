"""Pydantic contracts for the pretraining-data curation environment."""

from __future__ import annotations

import math
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, Field, model_serializer, model_validator

from ..gpu.train_gpt import scheduled_presentation_tokens, steps_for_token_budget
from .hf_access import CHARS_PER_TOKEN
from .val_set import ValidationSetConfig

MANIFEST_FILENAME = "manifest.json"

# Provenance of finalize() workspace outcome.
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

_MAX_TRAIN_TOKEN_BUDGET = 1_000_000_000
_MIN_CORPUS_CHARS = 5_000_000
_MAX_CORPUS_CHARS = 2_000_000_000
_MIN_SANDBOX_TIMEOUT_MINUTES = (
    30
)
_SANDBOX_SETUP_MINUTES = 15
_SANDBOX_TOKENS_PER_MINUTE = 500_000


class FilterSpec(BaseModel):
    """A single document-level filter applied to a source."""

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
    sample_docs_per_source: int | None = Field(default=None, ge=1)


class StudentArchConfig(BaseModel):
    """Fixed GPT-2-scale student architecture."""

    n_layer: int = Field(default=12, ge=2, le=64)
    n_head: int = Field(default=6, ge=1, le=64)
    n_embd: int = Field(default=768, ge=8, le=4096)
    mlp_ratio: int = Field(default=4, ge=1, le=16)
    lm_head_softcap: float = Field(default=30.0, gt=0.0, le=1000.0)
    num_value_embeds: int = Field(default=3, ge=1, le=32)
    attn_scale: float = Field(default=0.12, gt=0.0, le=10.0)
    sliding_window_size: int | None = Field(default=None, ge=2)
    block_size: int = Field(default=1024, ge=8, le=8192)

    @model_validator(mode="after")
    def _check_student_dims(self) -> "StudentArchConfig":
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
    # Microbatch cap; does not change scheduled token accounting.
    train_microbatch_size: int | None = Field(default=None, ge=1, le=4096)
    val_batch_size: int | None = Field(default=None, ge=1, le=4096)
    val_logit_chunk_tokens: int | None = Field(default=None, ge=1, le=1_048_576)
    steps: int = Field(default=200, ge=1, le=100_000)
    train_token_budget: int | None = Field(
        default=None, ge=1, le=_MAX_TRAIN_TOKEN_BUDGET
    )
    seed: int = Field(default=0, ge=0)
    val_fraction: float = Field(default=0.1, gt=0.0, lt=1.0)
    n_train_runs: int = Field(default=1, ge=1, le=64)


class StudentOptimizerConfig(BaseModel):
    """Training recipe: modded-nanogpt speedrun Muon vs legacy record_01 AdamW."""

    training_recipe: Literal["speedrun_muon", "record_01_adamw"] = "speedrun_muon"
    muon_lr: float = Field(default=0.023, gt=0.0, le=1.0)
    muon_weight_decay: float = Field(default=1.2, ge=0.0, le=10.0)
    muon_momentum_min: float = Field(default=0.85, gt=0.0, lt=1.0)
    muon_momentum_max: float = Field(default=0.95, gt=0.0, lt=1.0)
    muon_warmup_steps: int | None = Field(default=None, ge=0)
    muon_cooldown_steps: int | None = Field(default=None, ge=0)
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
    learning_rate: float = Field(default=3e-4, gt=0.0, le=1.0)
    weight_decay: float = Field(default=0.1, ge=0.0, le=1.0)
    adam_beta1: float = Field(default=0.9, gt=0.0, lt=1.0)
    adam_beta2: float = Field(default=0.95, gt=0.0, lt=1.0)
    record_adam_eps: float = Field(default=1e-8, gt=0.0, le=1e-1)
    grad_clip: float = Field(default=0.0, ge=0.0)
    warmup_steps: int | None = Field(default=None, ge=0)
    lr_min_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    wd_ref_steps: int | None = Field(default=None, ge=1)


class StudentScheduleConfig(BaseModel):
    """Batch-size + LR schedule (BatchSizeSchedule record defaults)."""

    batch_schedule_enabled: bool = True
    batch_stage_fracs: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)
    batch_stage_muls: tuple[int, int, int] = (1, 2, 3)
    lr_stage_muls: tuple[float, float, float] = (1.0, 1.52, 1.73)
    lr_cooldown_frac: float = Field(default=0.40, ge=0.0, le=1.0)
    lr_cooldown_floor: float = Field(default=0.15, ge=0.0, le=1.0)


class StudentFeaturesConfig(BaseModel):
    """Portable proxy-student features (all off by default except EOS packing)."""

    bigram_hash_embed: bool = False
    smear_embed: bool = False
    partial_key_offset: bool = False
    paired_head: bool = False
    mudd_pairs: int = Field(default=0, ge=0, le=16)
    xsa_enabled: bool = False
    xsa_pairs: int = Field(default=0, ge=0, le=16)
    single_act_last_k: int = Field(default=0, ge=0, le=16)
    exp_residual_decay: float | None = Field(default=None, gt=0.0, le=1.0)
    multi_token_pred: int = Field(default=0, ge=0, le=8)
    eos_aligned_batches: bool = True
    max_document_tokens: int | None = Field(default=None, ge=8)
    grad_accum_embed_head_steps: int = Field(default=1, ge=1, le=8)
    seq_len_schedule: bool = False
    untie_at_frac: float = Field(default=0.0, ge=0.0, le=1.0)
    cautious_wd: bool = False
    nor_muon: bool = True
    polar_express: bool = False


class StudentSandboxConfig(BaseModel):
    """Sandbox sizing; derived from the training budget unless overridden."""

    max_corpus_chars: int | None = Field(default=None, ge=1, le=_MAX_CORPUS_CHARS)
    timeout_minutes: int | None = Field(default=None, ge=1)
    upload_timeout_seconds: float = Field(default=120.0, gt=0.0)


_STUDENT_SUBMODEL_FIELDS: dict[str, frozenset[str]] = {
    "arch": frozenset(StudentArchConfig.model_fields),
    "run": frozenset(StudentRunConfig.model_fields),
    "optimizer": frozenset(StudentOptimizerConfig.model_fields),
    "schedule": frozenset(StudentScheduleConfig.model_fields),
    "features": frozenset(StudentFeaturesConfig.model_fields),
    "sandbox": frozenset(StudentSandboxConfig.model_fields),
}


class ProxyStudentConfig(BaseModel):
    """Fixed proxy-student specification, grouped by ownership."""

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
        if (
            values.get("training_recipe") == "record_01_adamw"
            and "adam_eps" in values
            and "record_adam_eps" not in values
        ):
            values["record_adam_eps"] = values["adam_eps"]
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
        if not name.startswith("_"):
            for group, fields in _STUDENT_SUBMODEL_FIELDS.items():
                if name in fields:
                    return getattr(getattr(self, group), name)
        return super().__getattr__(name)

    @property
    def effective_steps(self) -> int:
        """Training steps actually run."""
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
        """Tokens billed / consumed for the configured training length."""
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
        """Char cap on the uploaded corpus blob."""
        if self.sandbox.max_corpus_chars is not None:
            return self.sandbox.max_corpus_chars
        derived = CHARS_PER_TOKEN * self.effective_train_tokens
        return max(_MIN_CORPUS_CHARS, min(derived, _MAX_CORPUS_CHARS))

    @property
    def effective_timeout_minutes(self) -> int:
        """Sandbox lifetime / command timeout (minutes)."""
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
        """Flat JSON config blob consumed by the sandbox / self-score trainers."""
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
    allow_trace_id_manifest_fallback: bool = False
    allow_local_sources: bool = True
    max_local_source_bytes: int = Field(default=33_554_432, ge=1, le=1_073_741_824)

    alpha_perf: float = Field(default=1.0, ge=0.0)
    lambda_leakage: float = Field(default=1.0, ge=0.0)

    perf_baseline_loss: float = Field(default=math.log(50304), gt=0.0)
    perf_target_loss: float = Field(default=3.28, gt=0.0)
    # γ for baseline-relative perf.
    perf_scaling_exponent: float = Field(default=2.0)
    baseline_relative_perf: bool = True

    max_concurrent_fetches: int = Field(default=8, ge=1, le=1024)
    max_concurrent_training: int = Field(default=1, ge=1, le=256)
    fetch_timeout_seconds: float = Field(default=30.0, gt=0.0)
    fetch_timeout_per_doc_seconds: float = Field(default=0.25, ge=0.0)
    fetch_max_attempts: int = Field(default=3, ge=1, le=20)

    proxy_student: ProxyStudentConfig = Field(default_factory=ProxyStudentConfig)
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
    """One rollout's heavy scoring pass, shared by the reward and diagnostics."""

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
