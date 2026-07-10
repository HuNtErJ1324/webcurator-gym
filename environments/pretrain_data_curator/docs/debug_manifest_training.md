# Manifest-backed training debug workflow

A small, local-only workflow to **materialize a curated corpus from an explicit
manifest once**, then **repeatedly debug the NanoGPT/proxy-student training path
against that same bundle** without re-curating every run.

It reuses two pieces of the project's *real* code paths unchanged:

- **Curation** — `CorpusBuilder.materialize` (the environment's supported
  curation path) turns the manifest into documents, applying the same
  filters/sampling. Local sources are read from disk through a tiny
  `LocalProcessRuntime` that emulates the `wc -c` / `head -c` shell contract the
  curation path expects, so no Docker/Modal sandbox (and no GPU) is needed.
- **Training** — `student_train.averaged_train_and_eval` (the modded-nanogpt
  speedrun recipe, byte-identical to the GPU sandbox script) trains the fixed
  GPT-2-scale proxy-student on the bundle's `corpus.txt` on CPU with a small
  bounded budget.

Nothing here touches the production 400M configs, the provider/launcher code,
pods, Hub benchmarks, or any external state.

## Local preflights

| Dependency | Why | Check |
| --- | --- | --- |
| Python ≥ 3.11, `torch` | CPU training recipe (`student_train`) | `python -c "import torch"` |
| `tiktoken` | GPT-2-BPE tokenization of `corpus.txt` | `python -c "import tiktoken"` |
| `huggingface_hub` / `hf` | **Not required** — the debug workflow is local-only and rejects any `kind: "hf"` source. | n/a |
| `decon` | **Not used** — leakage screening is out of scope for this debug path. | n/a |

The manifest must use `kind: "local"` sources with **workspace-relative** paths
(no leading `/` or `..`) under `--base-dir`. See the environment README's
`Source` schema for `local_format`, `text_field`, `weight`, `filters`, and
`sampling` options.

## Usage

```bash
# 1) First run: curate the bundle from the manifest, then train (CPU).
python -m pretrain_data_curator.debug_train \
  --manifest ws/manifest.json \
  --base-dir ws \
  --bundle-dir pdc-debug-bundle \
  --output-dir pdc-debug-out \
  --steps 10 --block-size 64 --batch-size 4

# 2) Later runs: SAME command reuses the existing bundle, no re-curation.
#    (manifest identity is checked against the bundle's provenance first.)

# Force re-curation after you change the manifest:
python -m pretrain_data_curator.debug_train --manifest ws/manifest.json \
  --base-dir ws --bundle-dir pdc-debug-bundle --output-dir pdc-debug-out --refresh

# Only materialize (skip training):
python -m pretrain_data_curator.debug_train --manifest ws/manifest.json \
  --base-dir ws --bundle-dir pdc-debug-bundle --no-train

# Recover the manifest from a bundle that already has one:
python -m pretrain_data_curator.debug_train --bundle-dir pdc-debug-bundle --output-dir pdc-debug-out
```

A console script alias is also installed: `pdc-debug-train` (same arguments).

### Stable directories

- `--bundle-dir` (default `./pdc-debug-bundle`) holds the stable bundle:
  `corpus.txt` (the joined curated documents), `manifest.json` (a copy), and
  `provenance.json` (manifest SHA-256 digest, `token_budget`, source
  fingerprint, and corpus stats). These are git-ignored by default.
- `--output-dir` (default `./pdc-debug-out`) is cleared of files and receives
  `result.json` (loss / accuracy / flops / tokens trained / vocab / val tokens).

## Cache / safety guarantees

- **Cache hit, no re-curation.** A bundle whose `provenance.json` digest matches
  the manifest is reused; curation is skipped. (`--refresh` is the only way to
  re-curate.)
- **Mismatch fails before training.** If the manifest's digest or
  `token_budget` disagrees with the bundle, `resolve_corpus` raises
  `ManifestMismatchError` — a stale/wrong bundle is never silently reused. Use
  `--expected-token-budget` to assert the budget explicitly.
- **Exact corpus handoff.** The trainer trains on the tokens of the bundle's
  `corpus.txt` (GPT-2-BPE, then a tail `val_fraction` split). No other data is
  ever fed to the trainer.

## Example manifest (`ws/manifest.json`)

```json
{
  "token_budget": 20000,
  "sources": [
    {
      "dataset_id": "local:ws/data/a.jsonl",
      "kind": "local",
      "local_path": "data/a.jsonl",
      "local_format": "jsonl",
      "text_field": "text",
      "weight": 1
    },
    {
      "dataset_id": "local:ws/data/b.txt",
      "kind": "local",
      "local_path": "data/b.txt",
      "local_format": "txt",
      "weight": 1
    }
  ]
}
```

`data/a.jsonl` is newline-delimited JSON (`{"text": "..."}`); `data/b.txt` is
documents separated by blank lines.

## Notes

- CPU training with the default 278M-param student (12 layers / 768 dim) is slow;
  keep `--steps` small for iteration, or pass a tiny architecture via
  `build_debug_config(n_layer=2, n_embd=64, n_head=4)` if you script it. The
  recipe and hyperparameters are identical to the GPU path regardless of size.
- The bundle stores a manifest digest, not file contents. If you edit the
  underlying source files without changing the manifest, pass `--refresh` to
  re-curate (the manifest itself is the identity boundary).
