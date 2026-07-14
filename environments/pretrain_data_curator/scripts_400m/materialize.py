"""Materialize the canonical 400M local manifest into a stable, cached bundle.

Runs the project's supported curation path (``CorpusBuilder.materialize``) via
``debug_train.materialize_bundle`` against the single local FineWeb source. The
result is a bundle dir with:

  - ``corpus.txt``      joined curated documents (document-list-v1 tag)
  - ``manifest.json``   a copy of the manifest
  - ``provenance.json`` manifest digest + token budget + source fingerprint

The bundle is reused on retry (no re-curation) unless --refresh is passed, so
the 400M tokens are materialized/cached exactly once.
"""

from __future__ import annotations

import faulthandler
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

faulthandler.dump_traceback_later(1500, exit=True)  # 25 min hang guard

ERR_LOG = Path(os.environ.get("PDC_OUT_DIR", "/home/hunterj/pdc-out")) / "materialize_error.log"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import codecs

from pretrain_data_curator import debug_train
from pretrain_data_curator.debug_train import (
    load_manifest,
    manifest_digest,
    source_fingerprint,
)

HERE = Path(__file__).resolve().parent
MANIFEST = HERE.parent / "manifests" / "canonical-400m-fineweb-local.json"
DATA_DIR = Path(os.environ.get("PDC_DATA_DIR", "/home/hunterj/pdc-data"))
BUNDLE_DIR = Path(os.environ.get("PDC_BUNDLE_DIR", "/home/hunterj/pdc-bundle"))
OUT_DIR = Path(os.environ.get("PDC_OUT_DIR", "/home/hunterj/pdc-out"))
MAX_LOCAL_BYTES = int(os.environ.get("PDC_MAX_LOCAL_BYTES", "3000000000"))


def _streaming_provenance(manifest, corpus, corpus_path):
    """Same fields/semantics as debug_train._build_provenance, computed by
    streaming the written corpus.txt instead of holding it in memory — this
    box has ~15GB RAM and the corpus.txt is ~1.6GB of UTF-8 text, so a full
    read_text() + re-encode (as the default helper does) risks OOM."""
    h = hashlib.sha256()
    decoder = codecs.getincrementaldecoder("utf-8")()
    n_chars = 0
    with open(corpus_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
            n_chars += len(decoder.decode(chunk))
    n_chars += len(decoder.decode(b"", final=True))
    return {
        "tool": "pdc-debug-train",
        "manifest_digest": manifest_digest(manifest),
        "token_budget": manifest.token_budget,
        "source_fingerprint": source_fingerprint(manifest),
        "source_doc_counts": [s.doc_count for s in corpus.sources],
        "source_token_counts": [s.tokens for s in corpus.sources],
        "corpus_docs": sum(s.doc_count for s in corpus.sources),
        "corpus_chars": n_chars,
        "corpus_sha256": h.hexdigest(),
    }


debug_train._build_provenance = _streaming_provenance

from pretrain_data_curator.debug_train import materialize_bundle  # noqa: E402


def main() -> None:
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(MANIFEST)
    provenance = materialize_bundle(
        manifest,
        BUNDLE_DIR,
        base_dir=DATA_DIR,
        allow_local_sources=True,
        max_local_source_bytes=MAX_LOCAL_BYTES,
    )

    corpus_path = BUNDLE_DIR / "corpus.txt"
    report = {
        "manifest_path": str(MANIFEST),
        "manifest_digest": manifest_digest(manifest),
        "token_budget": manifest.token_budget,
        "bundle_dir": str(BUNDLE_DIR),
        "corpus_path": str(corpus_path),
        "corpus_sha256": provenance["corpus_sha256"],
        "manifest_copy_path": str(BUNDLE_DIR / "manifest.json"),
        "provenance": provenance,
        "command": "uv run python scripts_400m/materialize.py",
    }
    (OUT_DIR / "materialize_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        ERR_LOG.write_text(traceback.format_exc())
        raise
