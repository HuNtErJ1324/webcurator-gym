"""Decode FineWeb train shards (kjj0/fineweb10B-gpt2) into a local jsonl corpus.

The fineweb_train_*.bin shards are GPT-2-BPE uint16 token streams where
documents are separated by the GPT-2 EOT token (50256), mirroring the format
the proxy-student trainer consumes. We split on EOT to recover source documents,
decode each to text, and write one JSON object per line
(``{"text": "<doc>"}``) so the project's local-source materialization path can
re-tokenize them with the same GPT-2 BPE the held-out val set uses.

This is the canonical local data source for the merged debug-manifest 400M run.
The resulting file is consumed by ``materialize.py`` (which respects the
``max_local_source_bytes`` cap and never re-curates on retry).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import tiktoken
from huggingface_hub import hf_hub_download, list_repo_files

REPO = "kjj0/fineweb10B-gpt2"
EOT = 50256
SHARD_HEADER_BYTES = 256 * 4
ENCODER = tiktoken.get_encoding("gpt2")


def main() -> None:
    data_dir = Path(os.environ.get("PDC_DATA_DIR", "/home/hunterj/pdc-data"))
    out_jsonl = data_dir / "corpus.jsonl"
    target_tokens = int(os.environ.get("PDC_TARGET_TOKENS", "460000000"))
    data_dir.mkdir(parents=True, exist_ok=True)

    shards = sorted(
        f for f in list_repo_files(REPO, repo_type="dataset") if f.startswith("fineweb_train_")
    )
    if not shards:
        raise SystemExit(f"no fineweb_train_* shards found in {REPO}")

    collected = 0
    written = 0
    with out_jsonl.open("w") as fh:
        for shard in shards:
            path = hf_hub_download(REPO, filename=shard, repo_type="dataset")
            raw = open(path, "rb").read()
            header = np.frombuffer(raw[:SHARD_HEADER_BYTES], dtype="<i4")
            n_tokens = int(header[2])
            toks = np.frombuffer(
                raw[SHARD_HEADER_BYTES : SHARD_HEADER_BYTES + n_tokens * 2], dtype="<u2"
            ).astype(np.int64)
            positions = np.where(toks == EOT)[0]
            start = 0
            for pos in positions:
                seg = toks[start:pos]
                start = pos + 1
                if len(seg) < 32:
                    continue
                text = ENCODER.decode(seg.tolist())
                if not text or not text.strip():
                    continue
                fh.write(json.dumps({"text": text}) + "\n")
                collected += int(seg.shape[0]) + 1
                written += 1
                if collected >= target_tokens:
                    break
            del toks, raw
            if collected >= target_tokens:
                break

    print(
        f"collected {collected} tokens across {written} docs -> {out_jsonl}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
