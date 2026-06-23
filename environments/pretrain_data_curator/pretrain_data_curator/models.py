"""Pydantic contracts for the pretraining-data curation environment.

Everything the agent produces and everything the reward consumes is expressed
here so the manifest, cost ledger, and configuration have one strict home.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, Field, model_validator

from .val_set import ValidationSetConfig


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
    """One Hugging Face dataset slice contributing to the mixture."""

    dataset_id: str
    config: str | None = None
    split: str = "train"
    text_field: str = "text"
    weight: float = Field(default=1.0, ge=0.0)
    filters: list[FilterSpec] = Field(default_factory=list)
    sampling: Sampling = Field(default_factory=Sampling)

    def key(self) -> tuple[str, str | None]:
        return (self.dataset_id, self.config)


class Manifest(BaseModel):
    """The agent's deliverable: a weighted, filtered mixture of sources."""

    token_budget: int = Field(default=1_000_000, gt=0)
    sources: list[Source] = Field(default_factory=list)

    def upsert_source(self, source: Source) -> None:
        for i, existing in enumerate(self.sources):
            if existing.key() == source.key():
                self.sources[i] = source
                return
        self.sources.append(source)

    def remove_source(self, dataset_id: str, config: str | None = None) -> bool:
        before = len(self.sources)
        self.sources = [s for s in self.sources if s.key() != (dataset_id, config)]
        return len(self.sources) < before

    def normalized_weights(self) -> dict[str, float]:
        total = sum(s.weight for s in self.sources)
        if total <= 0:
            return {s.dataset_id: 0.0 for s in self.sources}
        return {s.dataset_id: s.weight / total for s in self.sources}

    def weight_entropy(self) -> float:
        """Shannon entropy of normalized weights, in [0, 1] (1 == perfectly even)."""
        weights = [w for w in self.normalized_weights().values() if w > 0]
        if len(weights) <= 1:
            return 0.0
        entropy = -sum(w * math.log(w) for w in weights)
        return entropy / math.log(len(weights))


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

    n_layer: int = Field(default=4, ge=1, le=64)
    n_head: int = Field(default=4, ge=1, le=64)
    n_embd: int = Field(default=256, ge=8, le=4096)
    block_size: int = Field(default=256, ge=8, le=8192)
    batch_size: int = Field(default=16, ge=1, le=4096)
    steps: int = Field(default=200, ge=1, le=100_000)
    learning_rate: float = Field(default=3e-4, gt=0.0, le=1.0)
    seed: int = Field(default=0, ge=0)
    val_fraction: float = Field(default=0.1, gt=0.0, lt=1.0)
    # Sandbox backend settings (only used by the real trainer).
    docker_image: str = Field(
        default="pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime", min_length=1
    )
    gpu_count: int = Field(default=1, ge=0, le=8)
    gpu_type: str | None = None
    cpu_cores: int = Field(default=4, ge=1, le=256)
    memory_gb: int = Field(default=16, ge=1, le=2048)
    disk_size_gb: int = Field(default=20, ge=1, le=10_000)
    timeout_minutes: int = Field(default=30, ge=1, le=1440)
    # Hard wall-clock bounds (seconds) on individual sandbox lifecycle steps.
    create_timeout_seconds: float = Field(default=300.0, gt=0.0)
    upload_timeout_seconds: float = Field(default=120.0, gt=0.0)

    @model_validator(mode="after")
    def _check_head_divides_embedding(self) -> "ProxyStudentConfig":
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )
        return self


class CuratorConfig(BaseModel):
    """Central, validated configuration for the environment and reward.

    Bounds and a cross-field check (``scan_limit >= candidate_limit``) keep
    nonsensical configurations from silently producing degenerate rollouts.
    """

    cutoff_date: str = "2024-12-31"
    token_budget: int = Field(default=1_000_000, gt=0)
    max_turns: int = Field(default=12, ge=1, le=1000)

    candidate_limit: int = Field(default=8, ge=1, le=1000)
    scan_limit: int = Field(default=50, ge=1, le=100_000)
    sample_docs_per_source: int = Field(default=64, ge=1, le=100_000)

    # Reward coefficients: R = a1*Perf + a2*Quality + a3*Diversity - l1*Cost - l2*Leakage
    alpha_perf: float = Field(default=1.0, ge=0.0)
    alpha_quality: float = Field(default=0.3, ge=0.0)
    alpha_diversity: float = Field(default=0.2, ge=0.0)
    lambda_cost: float = Field(default=0.1, ge=0.0)
    lambda_leakage: float = Field(default=1.0, ge=0.0)

    # Minimum-viability gate: leakage at or above this fraction is "severe" and
    # suppresses the quality/diversity bonuses (see CuratorRubric).
    leakage_severe_threshold: float = Field(default=0.5, ge=0.0, le=1.0)

    # Bounded concurrency and robustness knobs for external (HF/sandbox) access.
    max_concurrent_fetches: int = Field(default=8, ge=1, le=1024)
    max_concurrent_training: int = Field(default=1, ge=1, le=256)
    fetch_timeout_seconds: float = Field(default=30.0, gt=0.0)
    fetch_max_attempts: int = Field(default=3, ge=1, le=20)

    # Tool availability. Code execution is a stub in this build and is therefore
    # not advertised to the model unless explicitly enabled.
    enable_run_code: bool = False

    prices: CostPrices = Field(default_factory=CostPrices)
    proxy_student: ProxyStudentConfig = Field(default_factory=ProxyStudentConfig)
    # Held-out validation set for the downstream cross-entropy (Perf) signal.
    # Defaults to the NanoGPT speedrun set (FineWeb sample-10BT GPT-2-BPE val
    # tokens). Consumed by the real (sandbox) proxy-student trainer.
    validation_set: ValidationSetConfig = Field(default_factory=ValidationSetConfig)
    use_real_trainer: bool = False

    @model_validator(mode="after")
    def _check_scan_covers_candidates(self) -> "CuratorConfig":
        if self.scan_limit < self.candidate_limit:
            raise ValueError(
                f"scan_limit ({self.scan_limit}) must be >= candidate_limit "
                f"({self.candidate_limit})"
            )
        return self
