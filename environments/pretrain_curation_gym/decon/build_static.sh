#!/usr/bin/env bash
# Rebuild the portable static musl decon binary vendored at decon/bin/decon.
# Requires: rustup, musl-tools (apt install musl-tools), x86_64-unknown-linux-musl target.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$ROOT/bin/decon"
WORKDIR="${DECON_BUILD_DIR:-/tmp/decon-static-build}"

log() { printf '[decon-static] %s\n' "$*"; }

export PATH="${HOME}/.cargo/bin:${PATH}"
command -v cargo >/dev/null || { log "cargo not found"; exit 1; }
command -v musl-gcc >/dev/null || { log "musl-tools not found (apt install musl-tools)"; exit 1; }

rustup target add x86_64-unknown-linux-musl

if [[ ! -d "$WORKDIR/.git" ]]; then
  rm -rf "$WORKDIR"
  git clone --depth 1 https://github.com/allenai/decon.git "$WORKDIR"
fi

log "Building static musl release"
cd "$WORKDIR"
RUSTFLAGS='-C target-feature=+crt-static' \
  cargo build --release --target x86_64-unknown-linux-musl

BIN="$WORKDIR/target/x86_64-unknown-linux-musl/release/decon"
strip "$BIN"
mkdir -p "$(dirname "$DEST")"
install -m 0755 "$BIN" "$DEST"

log "Smoke test"
"$DEST" --version
file "$DEST"
log "Installed static decon at $DEST"
