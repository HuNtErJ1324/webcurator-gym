#!/usr/bin/env bash
# Build the decon CLI for the *current* machine's glibc (Ubuntu 22 pods, etc.).
# Use when the vendored static binary is missing or you prefer a native build.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$ROOT/bin/decon"
SRC_DIR="${DECON_SRC_DIR:-/tmp/decon-src}"

log() { printf '[decon-build] %s\n' "$*"; }

if command -v apt-get >/dev/null 2>&1; then
  log "Installing native build dependencies"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y -qq --no-install-recommends \
    build-essential pkg-config ca-certificates git curl
fi

export PATH="${HOME}/.cargo/bin:${PATH}"
if ! command -v cargo >/dev/null 2>&1; then
  log "Installing Rust toolchain"
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain stable
  # shellcheck disable=SC1091
  . "${HOME}/.cargo/env"
fi

if [[ ! -d "$SRC_DIR/.git" ]]; then
  log "Cloning allenai/decon into $SRC_DIR"
  rm -rf "$SRC_DIR"
  git clone --depth 1 https://github.com/allenai/decon.git "$SRC_DIR"
fi

log "Building decon (release)"
cd "$SRC_DIR"
cargo build --release

mkdir -p "$(dirname "$DEST")"
install -m 0755 target/release/decon "$DEST"

log "Smoke test"
"$DEST" --version
log "Installed native decon at $DEST"
