"""Shared test helpers — keep the default suite fast for local iteration."""

from __future__ import annotations

import pytest

from pretrain_data_curator.leakage import LeakageScores
from pretrain_data_curator.rewards import CuratorScorer
from pretrain_data_curator.taskset import CuratorTaskset


@pytest.fixture(autouse=True)
def _skip_host_memory_preflight_in_unit_tests(monkeypatch):
    """Unit tests must not depend on pod-sized host RAM; dedicated tests cover preflight."""
    monkeypatch.setenv("PDC_SKIP_MEMORY_PREFLIGHT", "1")


class NoOpLeakageDetector:
    """Skip the vendored decon subprocess in unit tests (see DeconLeakageDetector._check_binary fallback)."""

    def score(self, docs, val_set=None) -> LeakageScores:
        return LeakageScores(0.0, 0, [])


def bind_fast_scorer(
    taskset: CuratorTaskset,
    *,
    corpus_builder,
    trainer,
    leakage_detector,
) -> CuratorScorer:
    """Wire a scorer that never downloads the held-out val shard in unit tests."""
    taskset._corpus_builder = corpus_builder
    taskset._trainer = trainer
    taskset._decon_detector = leakage_detector
    taskset._val_loader = None
    scorer = CuratorScorer(
        taskset.curator,
        corpus_builder,
        trainer,
        leakage_detector,
        val_loader=None,
        screen_val_set=False,
    )
    taskset._scorer = scorer
    return scorer
