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

from .config import CuratorTaskConfig
from .corpus import EST_TOKENS_PER_DOC, CorpusBuilder
from .gpu.self_score import (
    SELF_SCORE_FILENAME,
    SELF_SCORE_HISTORY_FILENAME,
    SELF_SCORE_TRAIN_FILENAME,
    render_self_score_script,
    render_self_score_train_script,
)
from .gpu.turns import (
    TURN_COUNT_FILENAME,
    TURN_STATE_FILENAME,
    render_turn_count_script,
    render_turn_state,
)
from .leakage import DEFAULT_EVAL_SETS_DIR, DeconLeakageDetector
from .manifest import ManifestParser, TraceManifestCandidates
from .models import (
    CuratorConfig,
    MANIFEST_CANDIDATE_ASSISTANT_MESSAGE,
    MANIFEST_CANDIDATE_TRACE_FALLBACK,
    MANIFEST_PROVENANCE_INVALID_WORKSPACE_FILE,
    MANIFEST_PROVENANCE_MISSING,
    MANIFEST_PROVENANCE_WORKSPACE_FILE,
    Manifest,
    ManifestCandidate,
    ManifestProvenance,
)
from .rewards import CuratorScorer
from .state import CuratorState
from .tasks import CuratorTaskData
from .trainer import HeuristicProxyTrainer, RuntimeProxyTrainer, TrainerError
from .util.hf_access import HuggingFaceDatasetClient, RetryPolicy
from .util.utils import content_text
from .val_set import ValTokenLoader

logger = logging.getLogger(__name__)


class CuratorTask(vf.Task[CuratorTaskData, CuratorState, CuratorTaskConfig]):
    """Own setup, finalization, and one keyed scoring pass.

    The stock v1 task machinery records the keyed reward mapping.  The same pass
    records all diagnostics on ``Trace`` directly, removing the old collection
    of metric wrappers and its per-trace scoring cache.
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
        # ``@stop`` receives no runtime.  This narrow process handle exists only
        # to refresh the agent-visible turn counter and is removed at finalize.
        self._turn_runtimes: dict[str, vf.Runtime] = {}

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
        if (
            self.curator.use_real_trainer
            and runtime.type == "docker"
            and self.curator.proxy_student.runtime_backend == "docker"
        ):
            from .util.container_memory import (
                resolve_container_memory_gb,
                verify_runtime_memory_limit,
            )

            await asyncio.to_thread(
                verify_runtime_memory_limit,
                runtime,
                configured_gb=resolve_container_memory_gb(
                    self.curator.proxy_student.memory_gb,
                    backend="docker",
                ),
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
        await runtime.write(TURN_COUNT_FILENAME, render_turn_count_script())
        await runtime.write(
            TURN_STATE_FILENAME,
            render_turn_state(trace.num_turns, self.curator.max_turns),
        )
        self._turn_runtimes[trace.id] = runtime
        if self.curator.use_real_trainer:
            await runtime.write(
                SELF_SCORE_TRAIN_FILENAME, render_self_score_train_script()
            )

    @vf.stop
    async def max_turns_reached(self, trace: vf.Trace) -> bool:
        stopping = trace.num_turns >= self.curator.max_turns
        runtime = self._turn_runtimes.get(trace.id)
        if runtime is not None and not stopping:
            try:
                await runtime.write(
                    TURN_STATE_FILENAME,
                    render_turn_state(trace.num_turns, self.curator.max_turns),
                )
            except Exception:  # runtime backends expose different transport errors
                logger.debug("turn-state refresh failed", exc_info=True)
        return stopping

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
        manifest = self.parser.parse(
            json.dumps(data) if isinstance(data, dict) else "",
            default_token_budget=self.data.token_budget,
            reserved_local_filename=self.config.manifest_filename,
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
        self._turn_runtimes.pop(trace.id, None)
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
            if dataset_id in content_text(message.content):
                return True
            if any(
                dataset_id in self.candidates.shell_command(call.arguments)
                for call in message.tool_calls or []
            ):
                return True
        return False

    @vf.reward
    async def score_manifest(
        self, trace: vf.Trace, runtime: vf.Runtime | None = None
    ) -> dict[str, float]:
        """Run materialization/training/decon once and emit keyed rewards."""
        state = cast(CuratorState, trace.state)
        scoring = await self.scorer().compute_scoring(state, runtime)
        first = state.self_score_first_reward
        last = state.self_score_last_reward
        trainer_error = (state.trainer_error or "")[: self.TRAINER_ERROR_LIMIT]
        if trainer_error:
            logger.warning("trainer error: %s", trainer_error)

        trace.record_metrics(
            {
                "perf_loss": scoring["loss"],
                "perf_accuracy": scoring["accuracy"],
                "perf_vs_baseline": scoring["perf_vs_baseline"],
                "train_flops": scoring["flops"],
                "corpus_tokens": scoring["tokens"],
                "budget_fill_ratio": scoring["budget_fill_ratio"],
                "num_sources": scoring["num_sources"],
                "local_source_count": state.local_source_count,
                "local_source_bytes": state.local_source_bytes,
                "local_source_truncated": float(state.local_source_truncated),
                "val_set_access": float(state.val_set_access),
                "leakage_score": scoring["leakage"]["leakage_score"],
                "num_contaminated_matches": scoring["leakage"][
                    "num_contaminated_matches"
                ],
                "self_score_runs": state.self_score_runs,
                "self_score_ok_runs": state.self_score_ok_runs,
                "self_score_best_reward": state.self_score_best_reward or 0.0,
                "self_score_last_reward": last or 0.0,
                "self_score_improvement": (last - first)
                if first is not None and last is not None
                else 0.0,
                "num_turns": trace.num_turns,
                "finalized": float(state.manifest_finalized),
                "manifest_missing": float(
                    state.manifest_provenance == MANIFEST_PROVENANCE_MISSING
                ),
                "manifest_invalid": float(
                    state.manifest_provenance
                    == MANIFEST_PROVENANCE_INVALID_WORKSPACE_FILE
                ),
                "tool_error_count": state.tool_error_count,
                "external_failure": float(state.external_failure),
                "decon_error": scoring.get("decon_error", 0.0),
                "val_screen_skipped": scoring.get("val_screen_skipped", 0.0),
                "trainer_error_msg": float(bool(trainer_error)),
            }
        )
        return {
            "perf_reward": self.curator.alpha_perf * scoring["perf"],
            "leakage_penalty": -self.curator.lambda_leakage
            * scoring["leakage"]["leakage_score"],
        }

    async def score(self, trace: vf.Trace, runtime: vf.Runtime | None = None) -> None:
        try:
            await super().score(trace, runtime)
        finally:
            self._turn_runtimes.pop(trace.id, None)
            cast(CuratorState, trace.state).cleanup()


__all__ = ["CuratorTask"]
