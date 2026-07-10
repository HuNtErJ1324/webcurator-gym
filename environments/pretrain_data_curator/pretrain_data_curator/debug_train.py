"""Manifest-backed, locally-cacheable training-debug workflow.

This module turns the project's *real* curation and training paths into a
reproducible local debug loop:

1. **Materialize once.** Given an explicit local ``Manifest`` (``kind: "local"``
   sources only — no Hugging Face network), ``materialize_bundle`` runs the
   environment's *supported* curation path (``CorpusBuilder.materialize``) and
   writes a stable bundle dir: ``corpus.txt`` (the joined curated documents),
   ``manifest.json`` (a copy), and ``provenance.json`` (a manifest digest +
   token budget + source fingerprint so a later run can prove identity).
2. **Reuse the bundle.** ``resolve_corpus`` checks the existing bundle's
   ``provenance.json`` against the manifest *before* any training. A matching
   bundle is reused and curation is skipped by default; re-curation is an
   explicit ``--refresh`` only. A mismatched digest/budget fails loudly so a
   stale bundle is never silently reused.
3. **Hand the corpus to the real trainer.** ``train_debug`` GPT-2-BPE tokenizes
   the bundle's ``corpus.txt`` and runs the project's *actual* proxy-student
   recipe (``student_train.averaged_train_and_eval`` — byte-identical to the
   sandbox script) on CPU with a small bounded budget, writing a clear result
   dir. The handoff is exact: the trainer sees the tokens of ``corpus.txt``.

Nothing here touches production 400M configs, provider/launcher code, pods,
Hub benchmarks, or external state. Local sources are read directly from disk
through a tiny ``LocalProcessRuntime`` that emulates the ``wc -c`` / ``head -c``
shell contract the curation path expects.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import math
import shlex
import sys
import types
from pathlib import Path
from typing import Any, Callable

import tiktoken
import torch

from .corpus import CorpusBuilder, CuratedCorpus
from .models import Manifest, ProxyStudentConfig
from .rollout_state import CuratorState
from .student_model import GPT
from .student_train import averaged_train_and_eval, encode_document_tokens

logger = logging.getLogger(__name__)

CORPUS_NAME = "corpus.txt"
MANIFEST_NAME = "manifest.json"
PROVENANCE_NAME = "provenance.json"
RESULT_NAME = "result.json"

DEFAULT_BUNDLE_DIR = Path("pdc-debug-bundle")
DEFAULT_OUTPUT_DIR = Path("pdc-debug-out")

# Small, bounded CPU debug budget — enough to exercise the real recipe end to
# end without needing a GPU.
DEFAULT_STEPS = 40
DEFAULT_BLOCK_SIZE = 128
DEFAULT_BATCH_SIZE = 8
DEFAULT_N_TRAIN_RUNS = 1
DEFAULT_SEED = 0
DEFAULT_VAL_FRACTION = 0.1
GPT2_VOCAB_SIZE = 50304


class DebugError(RuntimeError):
    """Base class for manifest-debug workflow failures."""


class ManifestValidationError(DebugError):
    """The manifest file is missing or fails schema validation."""


class ManifestMismatchError(DebugError):
    """The provided manifest does not match the materialized bundle's provenance."""


class BundleError(DebugError):
    """The bundle directory is missing expected artifacts or is unusable."""


# --------------------------------------------------------------------------- #
# Manifest identity / provenance
# --------------------------------------------------------------------------- #
def manifest_digest(manifest: Manifest) -> str:
    """Stable SHA-256 of the canonical manifest serialization (identity + budget)."""
    return hashlib.sha256(manifest.model_dump_json().encode("utf-8")).hexdigest()


def source_fingerprint(manifest: Manifest) -> list[dict[str, Any]]:
    """Order-independent identity summary of the manifest's sources."""
    return [
        {
            "dataset_id": s.dataset_id,
            "kind": s.kind,
            "local_path": s.local_path,
            "weight": s.weight,
        }
        for s in manifest.sources
    ]


def load_manifest(path: Path) -> Manifest:
    """Read and validate an explicit manifest file."""
    path = Path(path)
    if not path.is_file():
        raise ManifestValidationError(f"manifest not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return Manifest.model_validate(data)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ManifestValidationError(f"invalid manifest {path}: {exc}") from exc


def bundle_paths(bundle_dir: Path) -> dict[str, Path]:
    bundle_dir = Path(bundle_dir)
    return {
        "corpus": bundle_dir / CORPUS_NAME,
        "manifest": bundle_dir / MANIFEST_NAME,
        "provenance": bundle_dir / PROVENANCE_NAME,
    }


def read_provenance(provenance_path: Path) -> dict[str, Any] | None:
    if not provenance_path.is_file():
        return None
    try:
        return json.loads(provenance_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise BundleError(f"unreadable provenance {provenance_path}: {exc}") from exc


def _build_provenance(
    manifest: Manifest,
    corpus: CuratedCorpus,
    corpus_path: Path,
) -> dict[str, Any]:
    text = corpus_path.read_text(encoding="utf-8")
    return {
        "tool": "pdc-debug-train",
        "manifest_digest": manifest_digest(manifest),
        "token_budget": manifest.token_budget,
        "source_fingerprint": source_fingerprint(manifest),
        "source_doc_counts": [s.doc_count for s in corpus.sources],
        "source_token_counts": [s.tokens for s in corpus.sources],
        "corpus_docs": sum(s.doc_count for s in corpus.sources),
        "corpus_chars": len(text),
        "corpus_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


# --------------------------------------------------------------------------- #
# Local curation path (reuses CorpusBuilder.materialize)
# --------------------------------------------------------------------------- #
class _NullSearchClient:
    """Rejects any HF (network) source so the debug workflow stays local-only."""

    def sample_documents(self, *args: Any, **kwargs: Any) -> list[str]:
        raise DebugError(
            "hf sources require network access; the debug workflow is local-only. "
            "Use 'kind': 'local' sources with paths under --base-dir."
        )


class LocalProcessRuntime:
    """Emulates the ``wc -c`` / ``head -c`` shell contract against local files.

    ``CorpusBuilder.fetch_local_docs`` drives local sources through
    ``runtime.run(["sh", "-c", "<cmd>"], ...)``. This runtime executes the two
    commands it actually issues (``wc -c < path`` and ``head -c N -- path``)
    against real files under ``base_dir``, with path-escape protection, so the
    *supported* curation path runs unchanged without a Docker/Modal sandbox.
    """

    def __init__(self, base_dir: Path = Path.cwd()) -> None:
        self.base_dir = Path(base_dir)
        self.commands: list[tuple[list[str], dict[str, str]]] = []

    def _resolve(self, raw: str) -> Path:
        path = (self.base_dir / raw).resolve()
        if self.base_dir.resolve() not in path.parents and path != self.base_dir.resolve():
            raise FileNotFoundError(f"path escapes base_dir: {raw}")
        return path

    async def run(self, argv: list[str], env: dict[str, str]) -> Any:
        self.commands.append((list(argv), dict(env)))
        command = shlex.split(argv[2]) if len(argv) >= 3 else []
        try:
            if command[:2] == ["wc", "-c"] and "<" in command:
                path = self._resolve(command[command.index("<") + 1])
                size = path.stat().st_size
                return types.SimpleNamespace(
                    exit_code=0, stdout=f"{size}\n", stderr=""
                )
            if command[:2] == ["head", "-c"]:
                cap = int(command[2])
                path = self._resolve(command[-1])
                data = path.read_bytes()[:cap]
                return types.SimpleNamespace(
                    exit_code=0,
                    stdout=data.decode("utf-8", errors="replace"),
                    stderr="",
                )
        except (FileNotFoundError, ValueError, OSError) as exc:
            return types.SimpleNamespace(exit_code=1, stdout="", stderr=str(exc))
        return types.SimpleNamespace(
            exit_code=1, stdout="", stderr=f"unexpected command: {command}"
        )


def materialize_bundle(
    manifest: Manifest,
    bundle_dir: Path,
    *,
    base_dir: Path = Path.cwd(),
    client: Any | None = None,
    runtime: Any | None = None,
    allow_local_sources: bool = True,
    max_local_source_bytes: int = 33_554_432,
) -> dict[str, Any]:
    """Curate ``manifest`` into ``bundle_dir`` via the supported curation path.

    Returns the written provenance dict. Writes ``corpus.txt``, ``manifest.json``,
    and ``provenance.json`` into ``bundle_dir``.
    """
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    if runtime is None:
        runtime = LocalProcessRuntime(base_dir)
    if client is None:
        client = _NullSearchClient()

    builder = CorpusBuilder(
        client,
        allow_local_sources=allow_local_sources,
        max_local_source_bytes=max_local_source_bytes,
    )
    state = CuratorState()
    corpus = asyncio.run(builder.materialize(manifest, state, runtime=runtime))

    corpus_path = bundle_dir / CORPUS_NAME
    _write_corpus(corpus, corpus_path)
    (bundle_dir / MANIFEST_NAME).write_text(
        manifest.model_dump_json(indent=2), encoding="utf-8"
    )
    provenance = _build_provenance(manifest, corpus, corpus_path)
    (bundle_dir / PROVENANCE_NAME).write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    logger.info(
        "materialized bundle: docs=%d chars=%d budget=%d at %s",
        provenance["corpus_docs"],
        provenance["corpus_chars"],
        provenance["token_budget"],
        bundle_dir,
    )
    return provenance


def _write_corpus(corpus: CuratedCorpus, path: Path) -> None:
    """Stream the tagged document-list payload consumed by the trainer."""
    with path.open("w", encoding="utf-8") as fh:
        fh.write('{"format":"document-list-v1","documents":[')
        first = True
        for doc in corpus.iter_documents():
            if not first:
                fh.write(",")
            json.dump(doc, fh, ensure_ascii=False)
            first = False
        fh.write("]}")


# --------------------------------------------------------------------------- #
# Cache resolution (skip curation unless explicitly refreshing or mismatched)
# --------------------------------------------------------------------------- #
def resolve_corpus(
    manifest: Manifest,
    bundle_dir: Path,
    *,
    refresh: bool = False,
    expected_token_budget: int | None = None,
    materialize_fn: Callable[..., dict[str, Any]] = materialize_bundle,
    **materialize_kwargs: Any,
) -> tuple[Path, dict[str, Any], bool]:
    """Return ``(corpus_path, provenance, curated)``.

    - A valid, matching bundle is reused and curation is skipped (``curated=False``).
    - ``--refresh`` (or a missing bundle) triggers materialization (``curated=True``).
    - A manifest whose digest or token budget disagrees with the bundle fails with
      :class:`ManifestMismatchError` *before* any training/curation.
    """
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    paths = bundle_paths(bundle_dir)
    provenance_path = paths["provenance"]

    if expected_token_budget is not None and manifest.token_budget != expected_token_budget:
        raise ManifestMismatchError(
            f"manifest token_budget {manifest.token_budget} != "
            f"expected {expected_token_budget}"
        )

    if not refresh and provenance_path.is_file():
        provenance = read_provenance(provenance_path)
        if provenance is None:
            raise BundleError(f"unreadable provenance in {bundle_dir}")
        if provenance.get("manifest_digest") != manifest_digest(manifest):
            raise ManifestMismatchError(
                "manifest digest does not match the existing bundle; "
                "use --refresh to re-curate, or fix the manifest."
            )
        if provenance.get("token_budget") != manifest.token_budget:
            raise ManifestMismatchError(
                f"manifest token_budget {manifest.token_budget} != bundle "
                f"{provenance.get('token_budget')}; use --refresh to re-curate."
            )
        if not paths["corpus"].is_file():
            raise BundleError(f"bundle corpus missing: {paths['corpus']}")
        logger.info("reusing existing bundle at %s (no re-curation)", bundle_dir)
        return paths["corpus"], provenance, False

    provenance = materialize_fn(manifest, bundle_dir, **materialize_kwargs)
    return bundle_dir / CORPUS_NAME, provenance, True


def load_manifest_from_bundle(bundle_dir: Path) -> Manifest:
    """Recover the manifest stored in a bundle (when only the bundle is given)."""
    manifest_path = bundle_paths(bundle_dir)["manifest"]
    if not manifest_path.is_file():
        raise BundleError(f"no manifest in bundle {bundle_dir}")
    return Manifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Corpus handoff to the real proxy-student training recipe
# --------------------------------------------------------------------------- #
def build_debug_config(
    *,
    steps: int = DEFAULT_STEPS,
    block_size: int = DEFAULT_BLOCK_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    n_train_runs: int = DEFAULT_N_TRAIN_RUNS,
    seed: int = DEFAULT_SEED,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    training_recipe: str = "speedrun_muon",
    n_layer: int = 12,
    n_embd: int = 768,
    n_head: int = 6,
) -> ProxyStudentConfig:
    """A small, bounded CPU training budget for the debug run."""
    # Validate eagerly (e.g. n_embd divisible by n_head) so failures surface here.
    return ProxyStudentConfig(
        steps=steps,
        block_size=block_size,
        batch_size=batch_size,
        n_train_runs=n_train_runs,
        seed=seed,
        val_fraction=val_fraction,
        training_recipe=training_recipe,
        n_layer=n_layer,
        n_embd=n_embd,
        n_head=n_head,
    )


def prepare_training_data(
    corpus_path: Path,
    *,
    tokenizer: str = "gpt2",
    val_fraction: float = DEFAULT_VAL_FRACTION,
    eos_aligned_batches: bool = True,
    max_document_tokens: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]] | None]:
    """Tokenize the preserved source-document list and split train/validation."""
    raw = Path(corpus_path).read_text(encoding="utf-8")
    enc = tiktoken.get_encoding(tokenizer)
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = None
    documents = (
        payload.get("documents")
        if isinstance(payload, dict) and payload.get("format") == "document-list-v1"
        else None
    )
    if eos_aligned_batches:
        if not isinstance(documents, list) or not all(isinstance(doc, str) for doc in documents):
            raise ValueError("EOS-aligned training requires document-list-v1 corpus data")
        ids, document_ranges = encode_document_tokens(
            documents, enc, max_document_tokens
        )
    else:
        text = raw if documents is None else "\n\n".join(documents)
        ids = enc.encode_ordinary(text)
        document_ranges = None
        if len(ids) < 64:
            ids = (ids * math.ceil(64 / max(len(ids), 1)))[:64] or [0] * 64
    corpus = torch.tensor(ids, dtype=torch.long)
    n_val = max(1, int(len(corpus) * val_fraction))
    train_data, val_data = corpus[:-n_val], corpus[-n_val:]
    if document_ranges is not None:
        document_ranges = [bounds for bounds in document_ranges if bounds[1] <= len(train_data)]
    return train_data, val_data, document_ranges


def _training_kwargs(config: ProxyStudentConfig, vocab_size: int) -> dict[str, Any]:
    """Build the kwargs for the real training recipe, mirroring the sandbox call.

    The recipe's ``train_and_eval_student`` names (``base_lr``/``beta1``/``beta2``/
    ``eps``) map from the config's ``learning_rate``/``adam_beta1``/``adam_beta2``/
    ``adam_eps`` — the same remap the embedded sandbox script performs, so this
    debug path drives byte-identical training code with identical hyperparameters.
    """
    return {
        "block_size": config.block_size,
        "batch_size": config.batch_size,
        "steps": config.effective_steps,
        "vocab_size": vocab_size,
        "training_recipe": config.training_recipe,
        "base_lr": config.learning_rate,
        "warmup_steps": config.effective_warmup_steps,
        "weight_decay": config.weight_decay,
        "grad_clip": config.grad_clip,
        "beta1": config.adam_beta1,
        "beta2": config.adam_beta2,
        "eps": config.record_adam_eps,
        "lr_min_ratio": config.lr_min_ratio,
        "muon_lr": config.muon_lr,
        "muon_weight_decay": config.muon_weight_decay,
        "muon_momentum_min": config.muon_momentum_min,
        "muon_momentum_max": config.muon_momentum_max,
        "muon_warmup_steps": config.muon_warmup_steps,
        "muon_cooldown_steps": config.muon_cooldown_steps,
        "adam_lr": config.adam_lr,
        "adam_eps": config.adam_eps,
        "adam_weight_decay": config.adam_weight_decay,
        "embed_lr_mul": config.embed_lr_mul,
        "lm_head_lr_mul": config.lm_head_lr_mul,
        "value_embed_lr_mul": config.value_embed_lr_mul,
        "scalar_lr_mul": config.scalar_lr_mul,
        "embed_wd_mul": config.embed_wd_mul,
        "lm_head_wd_mul": config.lm_head_wd_mul,
        "value_embed_wd_mul": config.value_embed_wd_mul,
        "scalar_wd_mul": config.scalar_wd_mul,
        "adam_on_odd_steps": config.adam_on_odd_steps,
        "batch_schedule_enabled": config.batch_schedule_enabled,
        "batch_stage_fracs": config.batch_stage_fracs,
        "batch_stage_muls": config.batch_stage_muls,
        "lr_stage_muls": config.lr_stage_muls,
        "lr_cooldown_frac": config.lr_cooldown_frac,
        "lr_cooldown_floor": config.lr_cooldown_floor,
        # portable feature flags (all default-off)
        "grad_accum_embed_head_steps": config.grad_accum_embed_head_steps,
        "seq_len_schedule": config.seq_len_schedule,
        "multi_token_pred": config.multi_token_pred,
        "untie_at_frac": config.untie_at_frac,
        "cautious_wd": config.cautious_wd,
        "nor_muon": config.nor_muon,
        "polar_express": config.polar_express,
    }


def train_debug(
    corpus_path: Path,
    output_dir: Path,
    config: ProxyStudentConfig,
    *,
    train_fn: Callable[..., Any] = averaged_train_and_eval,
    tokenizer: str = "gpt2",
    device: str = "cpu",
    clear_output: bool = True,
) -> dict[str, Any]:
    """Train on the bundle's ``corpus.txt`` with a bounded CPU budget.

    Returns the result dict and writes ``result.json`` into ``output_dir``. The
    ``train_fn`` is injectable (defaults to the real ``averaged_train_and_eval``)
    so tests can record the exact corpus handed off.
    """
    output_dir = Path(output_dir)
    if output_dir.exists() and clear_output:
        for child in output_dir.iterdir():
            if child.is_file():
                child.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_data, val_data, document_ranges = prepare_training_data(
        corpus_path,
        tokenizer=tokenizer,
        val_fraction=config.val_fraction,
        eos_aligned_batches=config.eos_aligned_batches,
        max_document_tokens=config.max_document_tokens,
    )
    vocab_size = tiktoken.get_encoding(tokenizer).n_vocab

    def build_model() -> GPT:
        return GPT(
            vocab_size=vocab_size,
            num_layers=config.n_layer,
            model_dim=config.n_embd,
            num_heads=config.n_head,
            mlp_ratio=config.mlp_ratio,
            softcap=config.lm_head_softcap,
            num_value_embeds=config.num_value_embeds,
            attn_scale=config.attn_scale,
            sliding_window_size=config.sliding_window_size,
            bigram_hash_embed=config.bigram_hash_embed,
            smear_embed=config.smear_embed,
            partial_key_offset=config.partial_key_offset,
            paired_head=config.paired_head,
            mudd_pairs=config.mudd_pairs,
            xsa_enabled=config.xsa_enabled,
            xsa_pairs=config.xsa_pairs,
            single_act_last_k=config.single_act_last_k,
            exp_residual_decay=config.exp_residual_decay,
            multi_token_pred=config.multi_token_pred,
        ).to(device)

    kwargs = _training_kwargs(config, vocab_size)
    kwargs["document_ranges"] = document_ranges
    result = train_fn(
        build_model,
        train_data,
        val_data,
        n_runs=config.n_train_runs,
        base_seed=config.seed,
        device=device,
        **kwargs,
    )
    if isinstance(result, tuple):
        loss, acc, flops, tokens_trained, n_params = result
    else:
        loss = result.get("loss")
        acc = result.get("accuracy")
        flops = result.get("flops")
        tokens_trained = result.get("tokens_trained")
        n_params = result.get("n_params")

    payload = {
        "loss": loss,
        "accuracy": acc,
        "flops": flops,
        "tokens_trained": tokens_trained,
        "n_params": n_params,
        "vocab_size": vocab_size,
        "val_tokens": int(len(val_data)),
        "train_tokens": int(len(train_data)),
        "training_recipe": config.training_recipe,
        "tokenizer": tokenizer,
    }
    (output_dir / RESULT_NAME).write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    logger.info(
        "debug training done: loss=%.4f acc=%.4f tokens=%d -> %s",
        loss if loss is not None else float("nan"),
        acc if acc is not None else 0.0,
        tokens_trained,
        output_dir,
    )
    return payload


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pdc-debug-train",
        description=(
            "Manifest-backed, locally-cacheable NanoGPT/proxy-student training "
            "debug workflow. Materialize a curated corpus from an explicit local "
            "manifest once, then repeatedly debug training against the same "
            "bundle without re-curating."
        ),
    )
    parser.add_argument("--manifest", type=Path, help="Explicit manifest JSON file.")
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        default=DEFAULT_BUNDLE_DIR,
        help=f"Stable bundle directory (default: {DEFAULT_BUNDLE_DIR}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Clear output directory for training results (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path.cwd(),
        help="Base dir for resolving local source paths (default: cwd).",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force re-curation of the bundle even if a matching one exists.",
    )
    parser.add_argument(
        "--expected-token-budget",
        type=int,
        default=None,
        help="Fail if the manifest token_budget differs from this value.",
    )
    parser.add_argument("--no-train", action="store_true", help="Only materialize; skip training.")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--n-train-runs", type=int, default=DEFAULT_N_TRAIN_RUNS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # Resolve the manifest: explicit, else recovered from the bundle.
    manifest: Manifest | None = None
    if args.manifest is not None:
        manifest = load_manifest(args.manifest)
    elif not args.refresh and bundle_paths(args.bundle_dir)["provenance"].is_file():
        manifest = load_manifest_from_bundle(args.bundle_dir)
    if manifest is None:
        print(
            "error: provide --manifest, or a bundle with a stored manifest. "
            "--refresh requires --manifest.",
            file=sys.stderr,
        )
        return 2
    if args.expected_token_budget is not None and manifest.token_budget != args.expected_token_budget:
        print(
            f"error: manifest token_budget {manifest.token_budget} != "
            f"--expected-token-budget {args.expected_token_budget}",
            file=sys.stderr,
        )
        return 2

    try:
        corpus_path, provenance, curated = resolve_corpus(
            manifest,
            args.bundle_dir,
            refresh=args.refresh,
            expected_token_budget=args.expected_token_budget,
            base_dir=args.base_dir,
        )
    except (ManifestMismatchError, BundleError, ManifestValidationError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"{'curated' if curated else 'reused'} bundle at {args.bundle_dir} "
        f"(docs={provenance['corpus_docs']}, chars={provenance['corpus_chars']}, "
        f"budget={provenance['token_budget']})"
    )

    if args.no_train:
        return 0

    config = build_debug_config(
        steps=args.steps,
        block_size=args.block_size,
        batch_size=args.batch_size,
        n_train_runs=args.n_train_runs,
        seed=args.seed,
        val_fraction=args.val_fraction,
    )
    try:
        result = train_debug(corpus_path, args.output_dir, config)
    except DebugError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"debug training: loss={result['loss']:.4f} acc={result['accuracy']:.4f} "
        f"tokens={result['tokens_trained']} -> {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
