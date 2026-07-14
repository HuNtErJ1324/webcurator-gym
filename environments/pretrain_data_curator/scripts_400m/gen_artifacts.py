"""Generate the training artifacts consumed by the GPU sandbox script.

Produces, into the output dir:
  - ``config.json``   the proxy-student training payload (NanoGPT-speedrun aligned)
  - ``train.py``      the byte-identical self-contained sandbox training script
  - ``val.bin``       the held-out FineWeb val tokens (header-free LE uint16)

The config is built from the canonical 400M eval TOML so the recipe matches the
project's NanoGPT-speedrun baseline exactly. Memory-neutral knobs
(``train_microbatch_size``, ``val_batch_size``, ``val_logit_chunk_tokens``) are
applied: they preserve the scheduled effective batch and the full held-out
validation pass while capping peak activation memory so the run fits the
A100-80GB ceiling without changing the trained token budget or loss semantics.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pretrain_data_curator.models import ProxyStudentConfig
from pretrain_data_curator.trainer import _nanogpt_train_script
from pretrain_data_curator.val_set import ValTokenLoader, ValidationSetConfig

HERE = Path(__file__).resolve().parent
TOML = HERE.parent / "configs" / "eval" / "400M-300turn-codex.toml"
OUT_DIR = Path(os.environ.get("PDC_OUT_DIR", "/home/hunterj/pdc-out"))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    toml = tomllib.loads(TOML.read_text(encoding="utf-8"))
    ps = toml["args"]["proxy_student"]
    cfg = ProxyStudentConfig(**ps)

    # Memory-neutral knobs (preserve effective batch + held-out validation).
    cfg.train_microbatch_size = 16
    cfg.val_batch_size = 8
    cfg.val_logit_chunk_tokens = 131072

    payload = cfg.training_payload()
    (OUT_DIR / "config.json").write_text(json.dumps(payload, indent=2))

    script = _nanogpt_train_script()
    (OUT_DIR / "train.py").write_text(script)

    loader = ValTokenLoader(ValidationSetConfig())
    vset = asyncio.run(loader.load())
    (OUT_DIR / "val.bin").write_bytes(vset.to_uint16_bytes())

    print(
        "config: steps=%d batch=%d block=%d train_microbatch=%s val_batch=%s "
        "val_logit_chunk=%s"
        % (
            payload["steps"],
            payload["batch_size"],
            payload["block_size"],
            payload["train_microbatch_size"],
            payload["val_batch_size"],
            payload["val_logit_chunk_tokens"],
        ),
        file=sys.stderr,
    )
    print(
        "val tokens=%d train.py bytes=%d config.json bytes=%d"
        % (vset.n_tokens, len(script), (OUT_DIR / "config.json").stat().st_size),
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
