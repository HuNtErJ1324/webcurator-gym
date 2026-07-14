#!/usr/bin/env bash
# Run the merged debug-manifest 400M proxy-student training on the A100 pod.
#
# Preconditions (set up by deploy_400m.sh on the host):
#   /root/webcurator-gym/environments/pretrain_data_curator  (has Dockerfile.runtime + decon/)
#   /root/pdc-bundle/{corpus.txt, manifest.json, provenance.json}
#   /root/pdc-out/{config.json, val.bin, train.py, materialize_report.json}
#
# This script:
#   1. builds webcurator-runtime:latest from Dockerfile.runtime (idempotent)
#   2. assembles /root/pdc-workspace from the bundle + generated artifacts
#   3. trains inside the container (--gpus all), logging VRAM + stdout
#   4. on CUDA OOM, retries once with tighter memory-neutral knobs
#   5. writes /root/pdc-out/run_report.json
set -uo pipefail

ENV_DIR="/root/webcurator-gym/environments/pretrain_data_curator"
BUNDLE="/root/pdc-bundle"
OUT="/root/pdc-out"
WORKSPACE="/root/pdc-workspace"
IMAGE="webcurator-runtime:latest"
RUNTIME_IMAGE_BUILD_CTX="$ENV_DIR"

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

mkdir -p "$OUT" "$WORKSPACE"

# ---- 1. build runtime image (idempotent) ----
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  log "building $IMAGE from Dockerfile.runtime"
  ( cd "$RUNTIME_IMAGE_BUILD_CTX" && docker build -f Dockerfile.runtime -t "$IMAGE" . ) \
    || { log "FATAL: image build failed"; exit 1; }
else
  log "reusing existing $IMAGE"
fi

# ---- 2. assemble workspace (corpus + config + val + train script) ----
cp -f "$BUNDLE/corpus.txt" "$WORKSPACE/corpus.txt"
cp -f "$OUT/config.json"   "$WORKSPACE/config.json"
cp -f "$OUT/val.bin"       "$WORKSPACE/val.bin"
cp -f "$OUT/train.py"      "$WORKSPACE/train.py"
log "workspace ready: $(du -h "$WORKSPACE/corpus.txt" | cut -f1) corpus"

# ---- 3. train (with VRAM logging) ----
run_training() {
  log "starting training container (attempt $1)"
  # Background VRAM sampler on the host.
  nvidia-smi --query-gpu=memory.used,memory.total --format=csv,nounits -l 1 > "$OUT/vram.log" 2>&1 &
  local NSMI=$!
  local CODE=0
  docker run --rm --gpus all \
    -v "$WORKSPACE:/workspace" \
    -v "$OUT:/out" \
    -w /workspace \
    "$IMAGE" python /workspace/train.py > "$OUT/train_stdout.log" 2>&1
  CODE=$?
  kill "$NSMI" 2>/dev/null || true
  wait "$NSMI" 2>/dev/null || true
  return "$CODE"
}

ATTEMPT=1
run_training "$ATTEMPT"
RC=$?

# ---- 4. memory-neutral retry on CUDA OOM ----
if [ "$RC" -ne 0 ] && grep -qi "out of memory\|CUDA error\|RuntimeError: CUDA" "$OUT/train_stdout.log" 2>/dev/null; then
  log "detected OOM; retrying with tighter memory-neutral knobs"
  python3 - "$OUT/config.json" <<'PY'
import json, sys
p = sys.argv[1]
cfg = json.load(open(p))
cfg["train_microbatch_size"] = 8
cfg["val_batch_size"] = 4
cfg["val_logit_chunk_tokens"] = 65536
json.dump(cfg, open(p, "w"), indent=2)
print("patched config:", {k: cfg[k] for k in ("train_microbatch_size","val_batch_size","val_logit_chunk_tokens")})
PY
  cp -f "$OUT/config.json" "$WORKSPACE/config.json"
  ATTEMPT=2
  run_training "$ATTEMPT"
  RC=$?
fi

# ---- 5. parse + report ----
RESULT_LINE=$(grep -m1 '^RESULT_JSON ' "$OUT/train_stdout.log" 2>/dev/null || true)
RESULT_JSON=""
if [ -n "$RESULT_LINE" ]; then
  RESULT_JSON="${RESULT_LINE#RESULT_JSON }"
fi

REPORT="$OUT/run_report.json"
python3 - "$RC" "$ATTEMPT" "$RESULT_JSON" "$OUT/materialize_report.json" <<'PY'
import json, sys
rc = int(sys.argv[1]); attempt = int(sys.argv[2]); result = sys.argv[3]
mat = sys.argv[4]
report = {"train_exit_code": rc, "attempts": attempt}
if result:
    try: report["result"] = json.loads(result)
    except Exception as e: report["result_parse_error"] = str(e)
try:
    report["materialize"] = json.load(open(mat))
except Exception:
    pass
# surface tail of stdout if no RESULT_JSON
if not result:
    try:
        lines = open("/root/pdc-out/train_stdout.log").read().splitlines()[-40:]
        report["train_stdout_tail"] = "\n".join(lines)
    except Exception:
        pass
json.dump(report, open("/root/pdc-out/run_report.json", "w"), indent=2)
print(json.dumps(report, indent=2)[:2000])
PY

if [ "$RC" -eq 0 ] && [ -n "$RESULT_LINE" ]; then
  log "TRAINING SUCCEEDED"
else
  log "TRAINING FAILED (exit=$RC) — pod left running for inspection"
fi
exit "$RC"
