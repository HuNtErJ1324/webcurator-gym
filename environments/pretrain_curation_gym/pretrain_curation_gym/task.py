"""The v1 task lifecycle for one pretraining-corpus curation episode."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
from importlib import resources
from typing import Literal, cast

import verifiers.v1 as vf
from verifiers.v1.errors import RolloutError

from .config import CuratorTaskConfig
from .utils.async_utils import run_blocking_drained
from .utils.corpus import EST_TOKENS_PER_DOC, CorpusBuilder
from .gpu.self_score import (
    SELF_SCORE_FILENAME,
    SELF_SCORE_HISTORY_FILENAME,
    SELF_SCORE_TRAIN_FILENAME,
    render_self_score_script,
    render_self_score_train_script,
)
from .gpu.turns import (
    TURNS_FILENAME,
    TURN_STATE_FILENAME,
    render_turn_state,
    render_turns_script,
)
from .utils.leakage import DEFAULT_EVAL_SETS_DIR, DeconLeakageDetector
from .utils.manifest import ManifestParser, TraceManifestCandidates
from .utils.models import (
    CuratorConfig,
    MANIFEST_CANDIDATE_ASSISTANT_MESSAGE,
    MANIFEST_CANDIDATE_TRACE_FALLBACK,
    MANIFEST_PROVENANCE_INVALID_WORKSPACE_FILE,
    MANIFEST_PROVENANCE_MISSING,
    MANIFEST_PROVENANCE_WORKSPACE_FILE,
    Manifest,
    ManifestCandidate,
    ManifestProvenance,
    ScoringResult,
)
from .rewards import CuratorScorer
from .state import CuratorState
from .taskdata import CuratorTaskData
from .utils.trainer import HeuristicProxyTrainer, RuntimeProxyTrainer, TrainerError
from .utils.hf_access import HuggingFaceDatasetClient, RetryPolicy
from .utils.val_set import ValTokenLoader

logger = logging.getLogger(__name__)


class EmptyRolloutError(RolloutError):
    """A rollout produced no usable artifact — no valid workspace manifest and
    zero self-scores — while ``error_on_empty_rollout`` was enabled.

    Under an opaque agent harness (e.g. codex) the trace carries no turns,
    messages, or token usage, so this workspace-derived signature is the only
    reliable evidence that the agent never engaged the task. Because setup and
    the harness came up first, it points to an infrastructure or model-endpoint
    failure rather than a legitimate empty submission. Subclassing
    ``RolloutError`` makes the framework capture it as
    ``error.type='EmptyRolloutError'`` so ``[retries.rollout]
    include=['EmptyRolloutError']`` retries the whole rollout — a transient blip
    self-heals instead of being scored as a silent zero-reward success.
    """


class CuratorTask(vf.Task[CuratorTaskData, CuratorState, CuratorTaskConfig]):
    """Own setup, finalization, and one keyed scoring pass.

    The stock v1 task machinery records metric and reward mappings. The heavy
    scoring result is retained only on rollout state so those primitives share
    one materialization/training pass.
    """

    FINALIZE_ATTEMPTS = 6
    FINALIZE_INTERVAL_SECONDS = 0.5
    HF_SKILL_PATH = ".agents/skills/hf-cli/SKILL.md"
    TRAINER_ERROR_LIMIT = 20_000

    # The v1 base class stores its generic arguments at runtime but its current
    # type hints expose the unspecialized attributes to subclasses.
    data: CuratorTaskData
    config: CuratorTaskConfig

    @staticmethod
    def hf_skill_package_file():
        """Locate the vendored skill with the Python 3.11 Traversable API."""
        node = resources.files("pretrain_curation_gym")
        for part in ("skills", "hf-cli", "SKILL.md"):
            node = node.joinpath(part)
        return node

    def __init__(
        self, data: CuratorTaskData, config: CuratorTaskConfig | None = None
    ) -> None:
        super().__init__(data, config)
        self.parser = ManifestParser()
        self.candidates = TraceManifestCandidates(self.parser)
        self._scorer: CuratorScorer | None = None

    @property
    def curator(self) -> CuratorConfig:
        return self.config.curator

    def fetch_policy(self) -> RetryPolicy:
        return RetryPolicy(
            attempts=self.curator.fetch_max_attempts,
            timeout=self.curator.fetch_timeout_seconds,
            per_doc_seconds=self.curator.fetch_timeout_per_doc_seconds,
        )

    def scorer(self) -> CuratorScorer:
        """Build immutable scoring collaborators once per task group."""
        if self._scorer is not None:
            return self._scorer
        policy = self.fetch_policy()
        val_loader = ValTokenLoader(
            self.curator.validation_set,
            token=os.environ.get(self.config.hf_token_env),
            retry_policy=policy,
            fetch_limit=self.curator.max_concurrent_fetches,
        )
        corpus_builder = CorpusBuilder(
            client=HuggingFaceDatasetClient(token_env=self.config.hf_token_env),
            retry_policy=policy,
            fetch_limit=self.curator.max_concurrent_fetches,
            allow_local_sources=self.curator.allow_local_sources,
            max_local_source_bytes=self.curator.max_local_source_bytes,
        )
        trainer = (
            RuntimeProxyTrainer(
                concurrency_limit=self.curator.max_concurrent_training,
                val_loader=val_loader,
            )
            if self.curator.use_real_trainer
            else HeuristicProxyTrainer()
        )
        detector = DeconLeakageDetector(
            decon_binary=self.config.decon_binary,
            evals_dir=self.config.decon_evals_dir or DEFAULT_EVAL_SETS_DIR,
            threshold=self.config.decon_threshold,
            screen_val_set=self.config.screen_val_set,
        )
        self._scorer = CuratorScorer(
            self.curator,
            corpus_builder,
            trainer,
            detector,
            val_loader=val_loader,
            screen_val_set=self.config.screen_val_set,
        )
        return self._scorer

    async def setup(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        token_env = self.config.hf_token_env
        if not os.environ.get(token_env):
            raise RuntimeError(
                f"Hugging Face token environment variable {token_env!r} is required "
                "before starting a rollout"
            )
        if self.curator.use_real_trainer and runtime.type not in {"docker", "modal"}:
            raise TrainerError(
                "use_real_trainer=True requires a Docker or Modal harness runtime "
                f"(got {runtime.type!r})"
            )
        await runtime.write(
            SELF_SCORE_FILENAME,
            render_self_score_script(
                self.curator,
                hf_token_env=token_env,
                decon_binary=self.config.decon_binary,
                decon_evals_dir=self.config.decon_evals_dir,
                decon_threshold=self.config.decon_threshold,
            ),
        )
        skill = self.hf_skill_package_file()
        await runtime.write(self.HF_SKILL_PATH, skill.read_bytes())
        state = cast(CuratorState, trace.state)
        state._turn_runtime = runtime
        await runtime.write(TURNS_FILENAME, render_turns_script())
        await runtime.write(
            TURN_STATE_FILENAME,
            render_turn_state(trace.num_turns, self.data.max_turns),
        )
        if self.curator.use_real_trainer:
            await runtime.write(
                SELF_SCORE_TRAIN_FILENAME, render_self_score_train_script()
            )

    @vf.stop
    async def refresh_turn_state(self, trace: vf.Trace) -> bool:
        """Refresh agent-visible telemetry without enforcing a second limit."""
        state = cast(CuratorState, trace.state)
        if state._turn_runtime is not None:
            try:
                await state._turn_runtime.write(
                    TURN_STATE_FILENAME,
                    render_turn_state(trace.num_turns, self.data.max_turns),
                )
            except Exception:  # runtime backends expose different transport errors
                logger.debug("turn-state refresh failed", exc_info=True)
        return False

    async def read_workspace_manifest(
        self, runtime: vf.Runtime | None, *, warn_invalid: bool = True
    ) -> tuple[Literal["absent", "invalid", "valid"], Manifest | None]:
        if runtime is None:
            return "absent", None
        try:
            text = (await runtime.read(self.config.manifest_filename)).decode("utf-8")
            data = json.loads(text)
        except UnicodeDecodeError:
            (logger.warning if warn_invalid else logger.debug)(
                "Manifest file %r exists but is not valid UTF-8",
                self.config.manifest_filename,
            )
            return "invalid", None
        except json.JSONDecodeError:
            (logger.warning if warn_invalid else logger.debug)(
                "Manifest file %r exists but does not contain a valid non-empty "
                "manifest",
                self.config.manifest_filename,
            )
            return "invalid", None
        except Exception:
            return "absent", None
        manifest = (
            self.parser.parse_object(
                data,
                default_token_budget=self.data.token_budget,
                reserved_local_filename=self.config.manifest_filename,
            )
            if isinstance(data, dict)
            else None
        )
        if manifest and manifest.sources:
            return "valid", manifest
        (logger.warning if warn_invalid else logger.debug)(
            "Manifest file %r exists but does not contain a valid non-empty manifest",
            self.config.manifest_filename,
        )
        return "invalid", None

    async def await_workspace_manifest(
        self, runtime: vf.Runtime | None
    ) -> tuple[Literal["absent", "invalid", "valid"], Manifest | None]:
        status: Literal["absent", "invalid", "valid"] = "absent"
        manifest: Manifest | None = None
        for _ in range(self.FINALIZE_ATTEMPTS):
            await asyncio.sleep(self.FINALIZE_INTERVAL_SECONDS)
            status, manifest = await self.read_workspace_manifest(
                runtime, warn_invalid=False
            )
            if status == "valid":
                break
        return status, manifest

    async def ingest_self_score_history(
        self, state: CuratorState, runtime: vf.Runtime | None
    ) -> None:
        if runtime is None:
            return
        try:
            lines = (
                (await runtime.read(SELF_SCORE_HISTORY_FILENAME)).decode().splitlines()
            )
        except Exception:
            return
        runs = 0
        rewards: list[float] = []
        for line in lines:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            runs += 1
            reward = record.get("reward")
            if (
                record.get("ok") is True
                and isinstance(reward, (int, float))
                and not isinstance(reward, bool)
                and math.isfinite(reward)
            ):
                rewards.append(float(reward))
        state.set_self_score_summary(runs=runs, rewards=rewards)

    async def finalize(self, trace: vf.Trace, runtime: vf.Runtime) -> None:
        state = cast(CuratorState, trace.state)
        await self.ingest_self_score_history(state, runtime)

        status, manifest = await self.read_workspace_manifest(runtime)
        if status != "valid":
            status, manifest = await self.await_workspace_manifest(runtime)

        candidate: ManifestCandidate | None = None
        telemetry: Manifest | None = None
        if status == "valid":
            provenance: ManifestProvenance = MANIFEST_PROVENANCE_WORKSPACE_FILE
        elif status == "invalid":
            provenance = MANIFEST_PROVENANCE_INVALID_WORKSPACE_FILE
        else:
            provenance = MANIFEST_PROVENANCE_MISSING
            telemetry = self.candidates.from_messages(
                trace,
                token_budget=self.data.token_budget,
                reserved_filename=self.config.manifest_filename,
            )
            if telemetry is not None:
                candidate = MANIFEST_CANDIDATE_ASSISTANT_MESSAGE
            elif self.curator.allow_trace_id_manifest_fallback:
                telemetry = self.candidates.from_dataset_ids(
                    trace,
                    token_budget=self.data.token_budget,
                    limit=self.curator.candidate_limit,
                )
                if telemetry is not None:
                    candidate = MANIFEST_CANDIDATE_TRACE_FALLBACK

        state.manifest_provenance = provenance
        trace.info["manifest_provenance"] = provenance
        if candidate is None:
            trace.info.pop("manifest_candidate", None)
        else:
            trace.info["manifest_candidate"] = candidate

        if provenance == MANIFEST_PROVENANCE_WORKSPACE_FILE and manifest is not None:
            self.warn_unreachable_budget(manifest)
            state.set_manifest(manifest, finalized=True)
        elif telemetry is not None:
            state.set_manifest(telemetry, finalized=False)
        else:
            state.manifest_finalized = False
        state.val_set_access = self.accessed_validation_set(trace)

    def warn_unreachable_budget(self, manifest: Manifest) -> None:
        if manifest.sample_docs_per_source is None:
            return
        reachable = (
            len(manifest.sources) * manifest.sample_docs_per_source * EST_TOKENS_PER_DOC
        )
        if manifest.token_budget > reachable:
            logger.warning(
                "TOKEN BUDGET IS NOT REACHABLE with the configured fetch cap: "
                "token_budget=%d sources=%d fetch_cap=%d "
                "estimated_tokens_per_doc=%d estimated_max_tokens=%d",
                manifest.token_budget,
                len(manifest.sources),
                manifest.sample_docs_per_source,
                EST_TOKENS_PER_DOC,
                reachable,
            )

    def accessed_validation_set(self, trace: vf.Trace) -> bool:
        dataset_id = self.curator.validation_set.dataset_id
        for message in trace.assistant_messages:
            if dataset_id in self.candidates.message_text(message.content):
                return True
            if any(
                dataset_id in self.candidates.shell_command(call.arguments)
                for call in message.tool_calls or []
            ):
                return True
        return False

    def _empty_rollout(self, state: CuratorState) -> bool:
        """Whether the rollout produced nothing usable: no valid workspace
        manifest (provenance is ``missing``) and the agent never ran the
        self-score script. See ``EmptyRolloutError``."""
        return (
            state.manifest_provenance == MANIFEST_PROVENANCE_MISSING
            and state.self_score_runs == 0
        )

    async def _compute_scoring(
        self, trace: vf.Trace, runtime: vf.Runtime | None = None
    ) -> ScoringResult:
        """Run and retain the one expensive scoring pass for this rollout."""
        state = cast(CuratorState, trace.state)
        if state._scoring_result is None:
            state._scoring_result = await self.scorer().compute_scoring(state, runtime)
        return state._scoring_result

    @vf.metric
    async def hf_cli_calls(self, trace: vf.Trace) -> float:
        """Number of observed ``hf`` CLI invocations in agent tool calls."""
        return float(
            sum(
                len(
                    self.candidates.hf_commands(
                        self.candidates.shell_command(call.arguments)
                    )
                )
                for message in trace.assistant_messages
                for call in message.tool_calls or []
            )
        )

    @vf.metric
    async def num_turns(self, trace: vf.Trace) -> float:
        """Framework-recorded model turns for the rollout."""
        return float(trace.num_turns)

    @vf.metric
    async def scoring_diagnostics(
        self, trace: vf.Trace, runtime: vf.Runtime | None = None
    ) -> dict[str, float]:
        """Materialize once and expose scoring/lifecycle diagnostics as metrics."""
        state = cast(CuratorState, trace.state)
        scoring = await self._compute_scoring(trace, runtime)
        first = state.self_score_first_reward
        last = state.self_score_last_reward
        trainer_error = (state.trainer_error or "")[: self.TRAINER_ERROR_LIMIT]
        if trainer_error:
            logger.warning("trainer error: %s", trainer_error)

        return {
            "perf_loss": scoring.loss,
            "perf_accuracy": scoring.accuracy,
            "perf_vs_baseline": scoring.perf_vs_baseline,
            "train_flops": scoring.flops,
            "corpus_tokens": scoring.tokens,
            "budget_fill_ratio": scoring.budget_fill_ratio,
            "num_sources": scoring.num_sources,
            "local_source_count": state.local_source_count,
            "local_source_bytes": state.local_source_bytes,
            "local_source_truncated": float(state.local_source_truncated),
            "val_set_access": float(state.val_set_access),
            "leakage_score": scoring.leakage_score,
            "num_contaminated_matches": scoring.num_contaminated_matches,
            "self_score_runs": state.self_score_runs,
            "self_score_ok_runs": state.self_score_ok_runs,
            "self_score_best_reward": state.self_score_best_reward or 0.0,
            "self_score_last_reward": last or 0.0,
            "self_score_improvement": (last - first)
            if first is not None and last is not None
            else 0.0,
            "finalized": float(state.manifest_finalized),
            "manifest_missing": float(
                state.manifest_provenance == MANIFEST_PROVENANCE_MISSING
            ),
            "manifest_invalid": float(
                state.manifest_provenance == MANIFEST_PROVENANCE_INVALID_WORKSPACE_FILE
            ),
            "tool_error_count": state.tool_error_count,
            "external_failure": float(state.external_failure),
            "empty_rollout": float(self._empty_rollout(state)),
            "decon_error": float(scoring.decon_error),
            "val_screen_skipped": float(scoring.val_screen_skipped),
            "trainer_error_msg": float(bool(trainer_error)),
        }

    @vf.reward
    async def score_manifest(
        self, trace: vf.Trace, runtime: vf.Runtime | None = None
    ) -> dict[str, float]:
        """Emit reward components from the scoring pass already used by metrics."""
        scoring = await self._compute_scoring(trace, runtime)
        return {
            "perf_reward": self.curator.alpha_perf * scoring.perf,
            "leakage_penalty": -self.curator.lambda_leakage * scoring.leakage_score,
        }

    async def score(self, trace: vf.Trace, runtime: vf.Runtime | None = None) -> None:
        try:
            await super().score(trace, runtime)
        finally:
            # Scratch corpora can be large; recursive deletion must not stall
            # unrelated rollouts sharing this event loop.
            await run_blocking_drained(cast(CuratorState, trace.state).cleanup)
        if self.config.error_on_empty_rollout and self._empty_rollout(
            cast(CuratorState, trace.state)
        ):
            raise EmptyRolloutError(
                "rollout produced no usable artifact (no valid workspace "
                "manifest, self_score_runs=0); treating as an infrastructure / "
                "model-endpoint failure. Set error_on_empty_rollout=false to "
                "record it as a zero-reward sample instead."
            )


__all__ = ["CuratorTask", "EmptyRolloutError"]
