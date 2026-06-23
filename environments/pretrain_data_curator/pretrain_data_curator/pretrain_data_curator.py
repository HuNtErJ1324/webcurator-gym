"""Entrypoint for the pretraining-data curation environment."""

from __future__ import annotations

import os
from typing import Any

import verifiers as vf

from .corpus import CorpusBuilder
from .environment import PretrainDataCuratorEnv
from .eval_corpus import DEFAULT_EVAL_CORPUS
from .hf_access import DatasetSearchClient, HuggingFaceDatasetClient, RetryPolicy
from .leakage import LeakageDetector
from .models import CuratorConfig, ProxyStudentConfig
from .rewards import CuratorRubric
from .tasks import build_dataset
from .trainer import HeuristicProxyTrainer, ProxyStudentTrainer, SandboxProxyTrainer
from .val_set import ValidationSetConfig, ValTokenLoader

SYSTEM_PROMPT = """You are a pretraining-data curation agent. Your job is to assemble \
a dataset mixture that, when used to train a fixed small GPT-2-scale student (everything \
fixed but the data), maximizes the student's performance.

Use only the provided tools and only Hugging Face datasets at or before the cutoff date.
Workflow:
1. search_datasets to discover candidates.
2. inspect_dataset to sample documents and judge quality.
3. set_source to add weighted sources, applying filters to remove low-quality text.
4. compute_manifest_stats to preview quality, diversity, leakage, and cost.
5. finalize_manifest when satisfied.

You are rewarded for proxy-student performance, corpus quality, and domain diversity, and \
penalized for cost (queries, calls, tokens, training FLOPs) and for leakage/contamination \
against the held-out evaluation set. Always finalize a non-empty manifest before finishing."""
 

def load_environment(
    cutoff_date: str = "2024-12-31",
    token_budget: int = 1_000_000,
    hf_token_env: str = "HF_TOKEN",
    candidate_limit: int = 8,
    scan_limit: int = 50,
    sample_docs_per_source: int = 64,
    max_turns: int = 12,
    alpha_perf: float = 1.0,
    alpha_quality: float = 0.3,
    alpha_diversity: float = 0.2,
    lambda_cost: float = 0.1,
    lambda_leakage: float = 1.0,
    leakage_severe_threshold: float = 0.5,
    max_concurrent_fetches: int = 8,
    max_concurrent_training: int = 1,
    fetch_timeout_seconds: float = 30.0,
    fetch_max_attempts: int = 3,
    enable_run_code: bool = False,
    use_real_trainer: bool = False,
    proxy_student: dict[str, Any] | None = None,
    validation_set: dict[str, Any] | None = None,
    eval_corpus: list[str] | None = None,
    client: DatasetSearchClient | None = None,
    **kwargs: Any,
) -> vf.Environment:
    vf.ensure_keys([hf_token_env])

    config = CuratorConfig(
        cutoff_date=cutoff_date,
        token_budget=token_budget,
        candidate_limit=candidate_limit,
        scan_limit=scan_limit,
        sample_docs_per_source=sample_docs_per_source,
        max_turns=max_turns,
        alpha_perf=alpha_perf,
        alpha_quality=alpha_quality,
        alpha_diversity=alpha_diversity,
        lambda_cost=lambda_cost,
        lambda_leakage=lambda_leakage,
        leakage_severe_threshold=leakage_severe_threshold,
        max_concurrent_fetches=max_concurrent_fetches,
        max_concurrent_training=max_concurrent_training,
        fetch_timeout_seconds=fetch_timeout_seconds,
        fetch_max_attempts=fetch_max_attempts,
        enable_run_code=enable_run_code,
        use_real_trainer=use_real_trainer,
        proxy_student=ProxyStudentConfig(**(proxy_student or {})),
        validation_set=ValidationSetConfig(**(validation_set or {})),
    )

    resolved_client = client or HuggingFaceDatasetClient(token=os.environ[hf_token_env])
    fetch_policy = RetryPolicy(
        attempts=config.fetch_max_attempts, timeout=config.fetch_timeout_seconds
    )
    corpus_builder = CorpusBuilder(
        client=resolved_client,
        sample_docs_per_source=config.sample_docs_per_source,
        retry_policy=fetch_policy,
        fetch_limit=config.max_concurrent_fetches,
    )
    leakage_detector = LeakageDetector(eval_corpus or DEFAULT_EVAL_CORPUS)
    # The held-out validation token stream (NanoGPT speedrun set by default).
    # Loaded through the same robustness path as Hub fetches; consumed by the real
    # (sandbox) trainer as the cross-entropy eval set.
    val_loader = ValTokenLoader(
        config.validation_set,
        retry_policy=fetch_policy,
        fetch_limit=config.max_concurrent_fetches,
    )
    trainer: ProxyStudentTrainer = (
        SandboxProxyTrainer(
            concurrency_limit=config.max_concurrent_training,
            val_loader=val_loader,
        )
        if config.use_real_trainer
        else HeuristicProxyTrainer()
    )
    rubric = CuratorRubric(
        config=config,
        corpus_builder=corpus_builder,
        trainer=trainer,
        leakage_detector=leakage_detector,
    )
    dataset = build_dataset(config)

    return PretrainDataCuratorEnv(
        client=resolved_client,
        config=config,
        corpus_builder=corpus_builder,
        leakage_detector=leakage_detector,
        dataset=dataset,
        eval_dataset=dataset,
        system_prompt=SYSTEM_PROMPT,
        rubric=rubric,
        env_id="pretrain-data-curator",
        env_args={
            "cutoff_date": config.cutoff_date,
            "token_budget": config.token_budget,
            "hf_token_env": hf_token_env,
            "candidate_limit": config.candidate_limit,
            "scan_limit": config.scan_limit,
            "use_real_trainer": config.use_real_trainer,
            "enable_run_code": config.enable_run_code,
            "val_dataset_id": config.validation_set.dataset_id,
            "val_tokens": config.validation_set.val_tokens,
        },
        **kwargs,
    )
