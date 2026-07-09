#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/webcurator-gym"
ENV_DIR="$ROOT/environments/pretrain_data_curator"
LOG_DIR="$ROOT/eval-logs"
LOG_FILE="$LOG_DIR/400M-300turn-$(date -u +%Y%m%dT%H%M%SZ).log"
RUNTIME_IMAGE="webcurator-runtime:latest"
# Same absolute path self_score.py probes inside the agent harness container.
AGENT_DECON_BIN="/workspace/decon/bin/decon"

mkdir -p "$LOG_DIR"
cd "$ENV_DIR"

set -a
# shellcheck disable=SC1091
source "$ROOT/secrets.env"
set +a
: "${HF_TOKEN:?HF_TOKEN must be set}"
: "${PRIME_API_KEY:?PRIME_API_KEY must be set}"

export PATH="$ENV_DIR/decon/bin:$HOME/.local/bin:$PATH"

echo "[setup] syncing python deps with uv..."
uv sync

echo "[setup] verifying decon binary (compile if needed)..."
DECON_BIN="$ENV_DIR/decon/bin/decon"
if ! "$DECON_BIN" --version >/dev/null 2>&1; then
  echo "[setup] Vendored decon missing or incompatible; building natively"
  bash "$ENV_DIR/decon/build_from_source.sh"
fi
"$DECON_BIN" --version

echo "[setup] building $RUNTIME_IMAGE from Dockerfile.runtime (after decon)..."
# Agent harness runs inside this image; bare pytorch lacks hf + decon.
docker build -f Dockerfile.runtime -t "$RUNTIME_IMAGE" .

echo "[setup] preflight inside $RUNTIME_IMAGE (hf/huggingface_hub + $AGENT_DECON_BIN)..."
docker run --rm -w /workspace "$RUNTIME_IMAGE" bash -lc \
  "command -v hf && python -c 'import huggingface_hub; print(\"huggingface_hub\", huggingface_hub.__version__)' && test -x ${AGENT_DECON_BIN} && ${AGENT_DECON_BIN} --version"

echo "[setup] verifying GPU docker access with $RUNTIME_IMAGE..."
docker run --rm --gpus 1 "$RUNTIME_IMAGE" nvidia-smi

echo "[run] starting 400M / 300-turn eval at $(date -u) (log: $LOG_FILE)"
set +e
uv run eval @ configs/eval/deepseek-v4-pro-400M-300turn-codex.toml 2>&1 | tee "$LOG_FILE"
status=${PIPESTATUS[0]}
set -e
exit "$status"
