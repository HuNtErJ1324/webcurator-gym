"""Pydantic contracts for the pretraining-data curation environment.

Everything the agent produces and everything the reward consumes is expressed
here so the manifest, cost ledger, and configuration have one strict home.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from .val_set import ValidationSetConfig

# --- proxy-student budget / sandbox derivation constants -------------------
# The real (sandbox) trainer's training length, corpus cap, and sandbox lifetime
# all derive from ``train_token_budget`` so a single token knob scales the whole
# run from a cheap default up to an H100/H200-scale few-hundred-million-token run.
_MAX_TRAIN_TOKEN_BUDGET = 1_000_000_000  # generous H100/H200 upper bound
_CHARS_PER_TOKEN = 4  # matches hf_access.estimate_tokens (chars // 4)
_MIN_CORPUS_CHARS = 5_000_000  # historical default cap; floor for small budgets
_MAX_CORPUS_CHARS = 2_000_000_000  # absolute ceiling on the uploaded corpus blob
# Sandbox lifetime derivation. Prime and Modal v1 runtimes both cap a remote
# sandbox at 24 hours.
_MAX_SANDBOX_TIMEOUT_MINUTES = 1440
_MIN_SANDBOX_TIMEOUT_MINUTES = 30  # historical default; floor keeps small budgets unchanged
_SANDBOX_SETUP_MINUTES = 15  # image pull + pip(tiktoken) + uploads + val download
_SANDBOX_TOKENS_PER_MINUTE = 500_000  # conservative floor for benchmark-sized runs

# Default container image for the DOCKER trainer backend (the prime backend keeps
# its historical ``docker_image`` default, below, for byte-identical behavior). It
# MUST ship torch + CUDA; this is the runtime image — switch to the matching
# ``-devel`` tag if a run needs build tooling (e.g. nvcc for torch.compile/custom
# kernels). torch 2.7 / CUDA 12.6 matches recent H100/H200 driver stacks.
_DEFAULT_DOCKER_TRAINER_IMAGE = "pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime"

# The installed ``prime_sandboxes`` SDK validates ``gpu_type`` only as a non-empty
# string (``Optional[str]``; no enum/Literal — see prime_sandboxes/models.py), so
# there is no discoverable allow-list. The verifiers+Prime ecosystem documents
# bare type names like "A100"/"H100" (verifiers/v1/task.py, runtimes/prime.py),
# passed straight through to the SDK. We default to H100 and keep the field a free
# ``str`` so other/future types (e.g. "H200") stay reachable via config.
GPU_TYPE_H100 = "H100"
GPU_TYPE_H200 = "H200"


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
    text_field: str | None = None
    weight: float = Field(default=1.0, ge=0.0)
    filters: list[FilterSpec] = Field(default_factory=list)
    sampling: Sampling = Field(default_factory=Sampling)

class Manifest(BaseModel):
    """The agent's deliverable: a weighted, filtered mixture of sources."""

    token_budget: int = Field(default=1_000_000, gt=0)
    sources: list[Source] = Field(default_factory=list)

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
    train_token_budget: int | None = Field(default=None, ge=1, le=_MAX_TRAIN_TOKEN_BUDGET)
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
    # Which backend runs the real proxy-student training. ``'prime'`` (default)
    # provisions a Prime GPU sandbox via ``prime_sandboxes`` exactly as before —
    # the byte-identical historical path. ``'docker'`` places the rollout harness
    # and proxy-student training in one declarative v1 ``DockerRuntime`` on the
    # local/co-located Docker daemon (see ``docker_backend.py``).
    # ``'modal'`` places both phases in one declarative v1 ``ModalRuntime``.
    trainer_backend: Literal["prime", "docker", "modal"] = "prime"
    # Retained for config compatibility, but remote Docker is not supported by
    # the shared harness-runtime path. ``load_environment`` rejects a non-None
    # value for the docker backend. Ignored by prime and modal.
    docker_host: str | None = None
    # Modal GPU type for the modal backend. Maps to a Modal GPU specifier string:
    # ``"H100"`` → ``"H100"``, ``"H200"`` → ``"H200"``, ``"A100"`` → ``"A100-80GB"``,
    # anything else → ``"L4"`` (default, cheapest; adequate for the 55M-param default
    # student). L4 billing is ~$0.80/hr per-second; the default 200-step run takes
    # ~50s ≈ $0.011. Ignored by the prime and docker backends.
    modal_gpu: str = Field(default="L4", min_length=1)
    # Sandbox/container backend settings (used by both real-trainer backends).
    #
    # ``docker_image`` is the container image. For the PRIME backend it defaults to
    # the historical pytorch 2.3.1 / cuda 12.1 image (unchanged). For the DOCKER
    # and MODAL backends, if left unset, it defaults to ``_DEFAULT_DOCKER_TRAINER_IMAGE``
    # (pytorch 2.7 / cuda 12.6; see ``_default_docker_image_for_docker_backend``).
    # Either way the image MUST ship torch + CUDA (use a ``-devel`` tag for build
    # tooling / torch.compile nvcc).
    docker_image: str = Field(
        default="pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime", min_length=1
    )
    gpu_count: int = Field(default=1, ge=0, le=8)
    # Concrete H100 default (configurable; "H200" reachable). See the GPU_TYPE_*
    # note above on why this is a free str and not an enum.
    gpu_type: str | None = GPU_TYPE_H100
    # VM mode: the Prime SDK requires vm=True whenever gpu_count > 0.
    vm: bool = True
    cpu_cores: int = Field(default=4, ge=1, le=256)
    memory_gb: int = Field(default=16, ge=1, le=2048)
    disk_size_gb: int = Field(default=20, ge=1, le=10_000)
    # Upper char cap on the corpus blob uploaded to the sandbox. ``None`` (default)
    # derives it from the training budget (so a large run is not silently capped at
    # the historical ~1.25M-unique-token corpus); an explicit value overrides.
    max_corpus_chars: int | None = Field(default=None, ge=1, le=_MAX_CORPUS_CHARS)
    # Sandbox/container lifetime / command timeout (minutes). ``None`` (default)
    # derives a budget-sized timeout (floored at the historical 30). The Prime 24h
    # ceiling is enforced backend-awarely in ``_check_prime_timeout_ceiling`` (the
    # prime backend rejects > 24h; the self-hosted docker backend has no such cap),
    # so the static field bound is just the lower bound.
    timeout_minutes: int | None = Field(default=None, ge=1)
    # Hard wall-clock bounds (seconds) on individual sandbox lifecycle steps.
    create_timeout_seconds: float = Field(default=300.0, gt=0.0)
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
        (setup overhead + tokens / throughput), floored at the historical 30. The
        prime backend additionally clamps the derived value to the Prime
        ``_MAX_SANDBOX_TIMEOUT_MINUTES`` (24h) platform max; the self-hosted docker
        backend has no such ceiling, so its derived timeout is left uncapped.
        """
        if self.timeout_minutes is not None:
            return self.timeout_minutes
        derived = max(
            _MIN_SANDBOX_TIMEOUT_MINUTES,
            _SANDBOX_SETUP_MINUTES
            + math.ceil(self.effective_train_tokens / _SANDBOX_TOKENS_PER_MINUTE),
        )
        if self.trainer_backend == "prime":
            return min(_MAX_SANDBOX_TIMEOUT_MINUTES, derived)
        if self.trainer_backend == "modal":
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
    def _check_gpu_request(self) -> "ProxyStudentConfig":
        # These preconditions are Prime-sandbox-specific (the SDK's
        # validate_gpu_fields). The docker backend maps gpu_count -> ``--gpus N``
        # and ignores gpu_type/vm entirely, and a self-hosted host has none of
        # Prime's vm/gpu_type requirements — so only constrain the prime backend
        # here (a valid docker config with vm=False must NOT be rejected).
        if self.trainer_backend != "prime":
            return self
        # Mirror the prime_sandboxes CreateSandboxRequest GPU precondition AT
        # CONFIG TIME, so a misconfigured GPU request fails LOUDLY here instead of
        # raising a ValidationError deep in the SDK that the rubric silently
        # degrades to the infinite-loss sentinel (every rollout scoring perf=0).
        # SDK rule (prime_sandboxes/models.py:106-114, validate_gpu_fields):
        #   gpu_count > 0  -> gpu_type required (non-empty) AND vm must be True
        #   gpu_count == 0 -> gpu_type must be None
        if self.gpu_count > 0:
            if not self.gpu_type or not self.gpu_type.strip():
                raise ValueError(
                    f"gpu_type is required when gpu_count ({self.gpu_count}) > 0 "
                    f"(e.g. {GPU_TYPE_H100!r} or {GPU_TYPE_H200!r})"
                )
            if not self.vm:
                raise ValueError(
                    f"vm must be True when gpu_count ({self.gpu_count}) > 0 "
                    "(Prime GPU sandboxes require VM mode)"
                )
        elif self.gpu_type is not None:
            raise ValueError(
                f"gpu_type ({self.gpu_type!r}) requires gpu_count > 0; raise "
                "gpu_count or set gpu_type=None"
            )
        return self

    @model_validator(mode="after")
    def _default_docker_image_for_docker_backend(self) -> "ProxyStudentConfig":
        # The shared ``docker_image`` default is the historical prime image, kept
        # byte-identical for the prime path. When the docker backend is selected
        # and the user did NOT pin an image, default to the torch 2.7 / cuda 12.6
        # image instead. ``model_fields_set`` distinguishes an explicit value from
        # the field default; assignment here does not re-run validators
        # (validate_assignment is off), so there is no recursion.
        if (
            self.trainer_backend == "docker"
            and "docker_image" not in self.model_fields_set
        ):
            self.docker_image = _DEFAULT_DOCKER_TRAINER_IMAGE
        return self

    @model_validator(mode="after")
    def _default_image_for_modal_backend(self) -> "ProxyStudentConfig":
        if (
            self.trainer_backend == "modal"
            and "docker_image" not in self.model_fields_set
        ):
            self.docker_image = _DEFAULT_DOCKER_TRAINER_IMAGE
        return self

    @model_validator(mode="after")
    def _check_prime_timeout_ceiling(self) -> "ProxyStudentConfig":
        # The Prime platform pins any sandbox to a 24h lifetime, so an explicit
        # timeout above that is unschedulable on the prime backend — fail loud here
        # (as the GPU-request guard does) rather than deep in the SDK. The docker
        # backend runs on a self-hosted host with no such cap, so it is unbounded.
        if (
            self.trainer_backend == "prime"
            and self.timeout_minutes is not None
            and self.timeout_minutes > _MAX_SANDBOX_TIMEOUT_MINUTES
        ):
            raise ValueError(
                f"timeout_minutes ({self.timeout_minutes}) exceeds the Prime 24h "
                f"sandbox maximum ({_MAX_SANDBOX_TIMEOUT_MINUTES}); lower it or set "
                "trainer_backend='docker'"
            )
        return self

    @model_validator(mode="after")
    def _check_modal_timeout_ceiling(self) -> "ProxyStudentConfig":
        if (
            self.trainer_backend == "modal"
            and self.timeout_minutes is not None
            and self.timeout_minutes > _MAX_SANDBOX_TIMEOUT_MINUTES
        ):
            raise ValueError(
                f"timeout_minutes ({self.timeout_minutes}) exceeds the Modal 24h "
                f"sandbox maximum ({_MAX_SANDBOX_TIMEOUT_MINUTES}); lower it"
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

    # Reward coefficients: R = a1*CEPerf - l1*Cost - l2*Leakage
    alpha_perf: float = Field(default=1.0, ge=0.0)
    lambda_cost: float = Field(default=0.1, ge=0.0)
    lambda_leakage: float = Field(default=1.0, ge=0.0)

    # --- baseline-relative Perf signal (additive; default-OFF) ----------------
    # ``perf_baseline_loss`` is the reference val cross-entropy of a neutral
    # (untrained) student. It is a CHEAP CONSTANT — no second training run is ever
    # performed, so default runs do not pay for it. Default: the CE of a uniform
    # student over the padded GPT-2 vocab (ln(50304) ~= 10.83 nats/token), the
    # no-information reference for the real trainer's nats/token CE. (The default
    # HeuristicProxyTrainer uses a smaller synthetic ``reference_loss`` of 5.0.)
    perf_baseline_loss: float = Field(default=math.log(50304), gt=0.0)
    # When True (the default), the Perf REWARD term is the bounded relative loss
    # reduction over ``perf_baseline_loss`` instead of ``exp(-loss)``.  Set to
    # False only when the absolute loss is meaningful (e.g. a tiny toy model with
    # loss < 1): for real LMs (loss ~ 9 nats/token, 50K vocab) exp(-loss) ≈ 0
    # and collapses the reward to zero.  The relative-improvement diagnostic
    # (perf_vs_baseline) is always surfaced regardless of this flag.
    baseline_relative_perf: bool = True

    # Bounded concurrency and robustness knobs for external (HF/sandbox) access.
    max_concurrent_fetches: int = Field(default=8, ge=1, le=1024)
    max_concurrent_training: int = Field(default=1, ge=1, le=256)
    fetch_timeout_seconds: float = Field(default=30.0, gt=0.0)
    fetch_max_attempts: int = Field(default=3, ge=1, le=20)

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
