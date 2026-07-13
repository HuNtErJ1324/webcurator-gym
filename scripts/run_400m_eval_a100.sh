#!/usr/bin/env bash
# Provision a Prime A100 pod, run a full 400M pretrain-data-curator eval end-to-end,
# pull artifacts back, rebuild the docs bench site, and terminate the pod.
#
# Usage:
#   ./scripts/run_400m_eval_a100.sh --model z-ai/glm-5.2
#   ./scripts/run_400m_eval_a100.sh --model deepseek/deepseek-v4-pro --curation-only
#
# Curation-only mode uses use_real_trainer=false (heuristic scoring, full trace,
# no GPU training) on a CPU pod. Artifacts land in outputs/debug/<model>-400M-curation-<date>/.
#   - prime CLI logged in (prime whoami)
#   - SSH key at ~/.prime/config.json ssh_key_path (default ~/.ssh/id_rsa)
#   - secrets.env at repo root with HF_TOKEN (PRIME_API_KEY optional if prime login)
#
# Eval uses the canonical swappable config with only the inference model changed:
#   configs/eval/400M-300turn-codex.toml  +  -m <provider/model>
#
# Options:
#   --model SLUG        Prime inference model id, e.g. z-ai/glm-5.2 (required)
#   --gpu-type TYPE     A100_80GB (default) or A100_40GB; ignored with --curation-only
#   --pod-name NAME     Pod name (default: wcg-400m-<model> or wcg-curation-<model>)
#   --disk-gb N         Pod disk size in GB (default: 120)
#   --curation-only     Heuristic trainer only (outputs/debug/, full trace)
#   --keep-pod          Do not terminate the pod on exit (debugging)
#   --skip-site         Skip docs/site rebuild after download
#   --dry-run           Print planned actions without creating a pod
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="$ROOT/environments/pretrain_data_curator"
BASE_EVAL_CONFIG="configs/eval/400M-300turn-codex.toml"
BASE_EVAL_CONFIG_PATH="$ENV_DIR/$BASE_EVAL_CONFIG"
RUN_SUFFIX="400M-300turn-codex"
SECRETS_FILE="$ROOT/secrets.env"
GPU_TYPE="A100_80GB"
POD_IMAGE="ubuntu_22_cuda_12"
DISK_GB=120
KEEP_POD=0
SKIP_SITE=0
DRY_RUN=0
CURATION_ONLY=0
MODEL=""
MODEL_SLUG=""
POD_NAME=""
EVAL_CONFIG=""
EVAL_CONFIG_PATH=""
RUN_NAME=""

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
die() { log "FATAL: $*"; exit 1; }

usage() {
  sed -n '2,22p' "$0"
  echo ""
  echo "Examples:"
  echo "  --model z-ai/glm-5.2"
  echo "  --model deepseek/deepseek-v4-pro"
  echo "  --model openai/gpt-4.1"
  exit 1
}

slugify_model() {
  python3 - "$1" <<'PY'
import re, sys
slug = sys.argv[1].strip().lower()
slug = slug.replace("/", "-").replace("_", "-")
slug = re.sub(r"[^a-z0-9.-]+", "-", slug)
slug = re.sub(r"-{2,}", "-", slug).strip("-")
print(slug or "model")
PY
}

resolve_eval() {
  local model="$1"
  [[ "$model" == */* ]] || die "Model must be a Prime inference slug like z-ai/glm-5.2 (got: $model)"
  if [[ "$CURATION_ONLY" -eq 1 ]]; then
    BASE_EVAL_CONFIG="configs/eval/400M-300turn-codex-curation.toml"
    RUN_SUFFIX="400M-300turn-codex-curation"
  fi
  BASE_EVAL_CONFIG_PATH="$ENV_DIR/$BASE_EVAL_CONFIG"
  [[ -f "$BASE_EVAL_CONFIG_PATH" ]] || die "Missing base config: $BASE_EVAL_CONFIG_PATH"
  MODEL="$model"
  MODEL_SLUG="$(slugify_model "$model")"
  EVAL_CONFIG="$BASE_EVAL_CONFIG"
  EVAL_CONFIG_PATH="$BASE_EVAL_CONFIG_PATH"
  RUN_NAME="${MODEL_SLUG}-${RUN_SUFFIX}"
  if [[ "$CURATION_ONLY" -eq 1 ]]; then
    LOCAL_OUT_DIR="$ENV_DIR/outputs/debug/${MODEL_SLUG}-400M-curation-$(date -u +%Y%m%d)"
    POD_NAME="${POD_NAME:-wcg-curation-${MODEL_SLUG}}"
  else
    LOCAL_OUT_DIR="$ENV_DIR/outputs/evals-400m/$RUN_NAME"
    POD_NAME="${POD_NAME:-wcg-400m-${MODEL_SLUG}}"
  fi
}

write_resolved_config() {
  local dest="$1"
  python3 - "$MODEL" "$BASE_EVAL_CONFIG_PATH" "$dest" <<'PY'
import pathlib, re, sys
model, src, dest = sys.argv[1:4]
text = pathlib.Path(src).read_text()
if re.search(r"^model\s*=", text, flags=re.M):
    text = re.sub(r'^model\s*=.*$', f'model = "{model}"', text, count=1, flags=re.M)
else:
    raise SystemExit("base config missing model = ...")
pathlib.Path(dest).write_text(text)
PY
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="${2:-}"; shift 2 ;;
    --gpu-type) GPU_TYPE="${2:-}"; shift 2 ;;
    --pod-name) POD_NAME="${2:-}"; shift 2 ;;
    --disk-gb) DISK_GB="${2:-}"; shift 2 ;;
    --keep-pod) KEEP_POD=1; shift ;;
    --curation-only) CURATION_ONLY=1; shift ;;
    --skip-site) SKIP_SITE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage ;;
    *) die "Unknown argument: $1 (try --help)" ;;
  esac
done

[[ -n "$MODEL" ]] || usage
resolve_eval "$MODEL"

if [[ "$CURATION_ONLY" -eq 1 ]]; then
  GPU_TYPE="CPU_NODE"
  POD_IMAGE="ubuntu_22"
fi

command -v prime >/dev/null || die "prime CLI not found"
command -v rsync >/dev/null || die "rsync not found"
command -v ssh >/dev/null || die "ssh not found"
command -v scp >/dev/null || die "scp not found"
[[ -f "$SECRETS_FILE" ]] || die "Missing $SECRETS_FILE (need at least HF_TOKEN)"

prime_api_key_from_cli() {
  python3 - <<'PY'
import json, pathlib
cfg_path = pathlib.Path.home() / ".prime/config.json"
if not cfg_path.exists():
    raise SystemExit("no ~/.prime/config.json")
key = json.loads(cfg_path.read_text()).get("api_key") or ""
if not key:
    raise SystemExit("no api_key in ~/.prime/config.json")
print(key)
PY
}

upload_secrets() {
  local tmp
  tmp="$(mktemp)"
  cp "$SECRETS_FILE" "$tmp"
  if ! grep -q '^PRIME_API_KEY=' "$tmp"; then
    log "Injecting PRIME_API_KEY from local prime CLI config into remote secrets.env"
    printf 'PRIME_API_KEY=%s\n' "$(prime_api_key_from_cli)" >> "$tmp"
  fi
  scp -i "$SSH_KEY" -o StrictHostKeyChecking=no -P "$SSH_PORT" \
    "$tmp" "${SSH_USER}@${SSH_HOST}:/root/webcurator-gym/secrets.env" >/dev/null
  rm -f "$tmp"
  # Enforce restrictive perms on the remote secrets file: a world-readable
  # secrets.env is a leak vector. Fail the run (never leaving secrets exposed)
  # if chmod cannot be applied. Secret contents are never logged.
  if ! remote chmod 0600 /root/webcurator-gym/secrets.env; then
    die "Failed to chmod 0600 the remote secrets file at /root/webcurator-gym/secrets.env"
  fi
}

SSH_KEY="$(python3 - <<'PY'
import json, pathlib
cfg_path = pathlib.Path.home() / ".prime/config.json"
cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
candidates = [
    cfg.get("ssh_key_path"),
    pathlib.Path.home() / ".ssh/id_ed25519",
    pathlib.Path.home() / ".ssh/id_rsa",
]
for path in candidates:
    if path and pathlib.Path(path).exists():
        print(path)
        break
else:
    raise SystemExit("no ssh key found")
PY
)"
[[ -f "$SSH_KEY" ]] || die "SSH key not found at $SSH_KEY"

prime whoami --plain >/dev/null 2>&1 || die "prime CLI not authenticated (run: prime login)"

POD_ID=""
SSH_USER="root"
SSH_HOST=""
SSH_PORT="22"

cleanup() {
  local code=$?
  if [[ -n "$POD_ID" && "$KEEP_POD" -eq 0 ]]; then
    log "Terminating pod $POD_ID"
    prime pods terminate "$POD_ID" -y --plain >/dev/null 2>&1 || true
  elif [[ -n "$POD_ID" && "$KEEP_POD" -eq 1 ]]; then
    log "Keeping pod $POD_ID (--keep-pod)"
  fi
  exit "$code"
}
trap cleanup EXIT

pick_cloud_id() {
  local try_type="$1"
  local json
  json="$(prime availability list --gpu-type "$try_type" --gpu-count 1 --output json 2>/dev/null)"
  python3 - "$json" "${EXCLUDED_CLOUD_IDS:-}" <<'PY'
import json, re, sys
data = json.loads(sys.argv[1])
excluded = set(sys.argv[2].split())
resources = [
    r for r in data.get("gpu_resources", [])
    if (r.get("stock_status") or "").lower() == "available"
    and r.get("cloud_id") not in excluded
    # crusoecloud and massedcompute pods default to a non-root "ubuntu" login
    # user (observed 2026-07-08 on crusoecloud's a100-80gb.1x; MassedCompute
    # ships ubuntu images). This launcher is root-home/rootful-Docker only:
    # every remote path (repo sync destination, secrets.env, decon binary,
    # Docker) assumes a root home directory, and rootful Docker is required.
    # MassedCompute ubuntu is unsupported -- skip both providers rather than
    # special-case paths or add rootless-Docker / single-shell support.
    and (r.get("provider") or "").lower() != "crusoecloud"
    and (r.get("provider") or "").lower() != "massedcompute"
]
if not resources:
    raise SystemExit(1)

def price(r):
    raw = r.get("price_per_hour", r.get("price_value", 1e9))
    if isinstance(raw, str):
        m = re.search(r"[\d.]+", raw)
        return float(m.group(0)) if m else 1e9
    return float(raw)

# Prefer on-demand over spot: spot capacity can be reclaimed mid-run (observed
# 2026-07-08 on a 1A100.22V_SPOT offer, killing an in-progress smoke eval),
# which is worse for a multi-turn agent + GPU training run than paying more.
resources.sort(key=lambda r: (bool(r.get("is_spot")), price(r)))
print(resources[0]["cloud_id"])
PY
}

pick_cpu_cloud_id() {
  local json
  json="$(prime availability list --gpu-type CPU_NODE --gpu-count 1 --output json 2>/dev/null)"
  python3 - "$json" <<'PY'
import json, re, sys

data = json.loads(sys.argv[1])
resources = [
    r for r in data.get("gpu_resources", [])
    if (r.get("stock_status") or "").lower() == "available"
    # crusoecloud and massedcompute default to a non-root "ubuntu" login user
    # and this launcher is root-home/rootful-Docker only (MassedCompute ubuntu
    # is unsupported). Skip both rather than special-case paths.
    and (r.get("provider") or "").lower() != "crusoecloud"
    and (r.get("provider") or "").lower() != "massedcompute"
]

def memory_gb(r):
    raw = str(r.get("memory_gb") or "0")
    m = re.search(r"\d+", raw)
    return int(m.group()) if m else 0

def vcpus(r):
    raw = str(r.get("vcpus") or "0")
    m = re.search(r"\d+", raw)
    return int(m.group()) if m else 0

def price(r):
    raw = r.get("price_per_hour", r.get("price_value", 1e9))
    if isinstance(raw, str):
        m = re.search(r"[\d.]+", raw)
        return float(m.group(0)) if m else 1e9
    return float(raw)

resources = [r for r in resources if memory_gb(r) >= 32 and vcpus(r) >= 8]
if not resources:
    raise SystemExit(1)
resources.sort(key=lambda r: (price(r), -memory_gb(r), -vcpus(r)))
print(resources[0]["cloud_id"])
PY
}

pick_compute() {
  # Emits "<gpu_type> <cloud_id>". Callers MUST parse both fields and assign
  # GPU_TYPE themselves: this runs inside a command substitution (subshell), so
  # any GPU_TYPE assignment made here is discarded when the subshell exits.
  # (Observed 2026-07-12: the previous version assigned GPU_TYPE in-subshell and
  # returned only the cloud_id, so after an A100_80GB->A100_40GB fallback the
  # parent still believed GPU_TYPE=A100_80GB and logged it that way while
  # provisioning a 40GB pod -- a silent, invisible GPU downgrade.)
  if [[ "$CURATION_ONLY" -eq 1 ]]; then
    local cpu_id
    if cpu_id="$(pick_cpu_cloud_id)"; then
      echo "CPU_NODE $cpu_id"
      return 0
    fi
    die "No available CPU_NODE slots with >=8 vCPU and >=32 GB RAM"
  fi
  # No GPU fallback: the requested --gpu-type is honored exactly, or we fail.
  # Silently substituting a smaller/different GPU changes the economics and the
  # memory envelope of a multi-hour paid training run, so an unavailable
  # requested type is a hard error, not something to paper over.
  local gpu_id
  if gpu_id="$(pick_cloud_id "$GPU_TYPE")"; then
    echo "$GPU_TYPE $gpu_id"
    return 0
  fi
  return 1
}

pick_compute_with_retries() {
  # Emits "<gpu_type> <cloud_id>" (see pick_compute).
  # Shared inventory has been observed to flicker within single-digit seconds
  # (2026-07-08: an offer reported Available by `availability list` vanished
  # 15-30s later; 2026-07-12: A100_80GB capacity confirmed Available vanished
  # within ~7s, before the pick). A single miss shouldn't be fatal -- retry a
  # few times with backoff before giving up.
  local attempt picked
  for attempt in 1 2 3 4; do
    if picked="$(pick_compute)"; then
      echo "$picked"
      return 0
    fi
    log "No $GPU_TYPE capacity on pick attempt $attempt/4; retrying in 10s"
    sleep 10
  done
  return 1
}

parse_ssh_target() {
  local status_json="$1"
  python3 - <<'PY' "$status_json"
import json, re, sys

d = json.loads(sys.argv[1])
ip = (d.get("ip") or "").strip()
ssh = (d.get("ssh") or "").strip()

# Default login user is root, but some providers (observed 2026-07-08:
# crusoecloud's a100-80gb.1x images use "ubuntu") inject a different default
# user. Prefer whatever user the API's `ssh` connection string states, even
# when the host/port come from the separate `ip`/`port_mappings` fields.
user = "root"
ssh_port = None
ssh_host = None
if ssh:
    stripped = re.sub(r"^ssh\s+", "", ssh)
    match = re.search(r"\s+-p\s+(\d+)\s*$", stripped)
    if match:
        ssh_port = match.group(1)
        stripped = stripped[: match.start()].strip()
    if "@" in stripped:
        maybe_user, _, maybe_host = stripped.partition("@")
        if maybe_user.strip():
            user = maybe_user.strip()
        stripped = maybe_host
    if stripped.strip():
        ssh_host = stripped.strip().split()[0]

port = "22"
if ip:
    for mapping in d.get("port_mappings") or []:
        internal = str(mapping.get("internal", ""))
        external = mapping.get("external")
        if internal in {"22", "2222"} and external:
            port = str(external)
            break
    if ssh_port:
        port = ssh_port
    print(user)
    print(ip)
    print(port)
    raise SystemExit(0)

if not ssh_host:
    raise SystemExit("ssh connection not ready")

if ssh_port:
    port = ssh_port

print(user)
print(ssh_host)
print(port)
PY
}

wait_for_pod() {
  local pod_id="$1"
  log "Waiting for pod $pod_id to become SSH-ready"
  while true; do
    local status_json
    status_json="$(prime pods status "$pod_id" --output json --plain 2>/dev/null || true)"
    [[ -n "$status_json" ]] || { sleep 10; continue; }
    local state install
    state="$(python3 - <<'PY' "$status_json"
import json, sys
print(json.loads(sys.argv[1]).get("status", ""))
PY
)"
    install="$(python3 - <<'PY' "$status_json"
import json, sys
d = json.loads(sys.argv[1])
print(d.get("installation_status") or "")
PY
)"
    if [[ "$state" == "ACTIVE" && -n "$(python3 - <<'PY' "$status_json"
import json, sys
print(json.loads(sys.argv[1]).get("ssh") or "")
PY
)" ]]; then
      if [[ -z "$install" || "$install" == "FINISHED" ]]; then
        mapfile -t _ssh_parts < <(parse_ssh_target "$status_json")
        SSH_USER="${_ssh_parts[0]:-root}"
        SSH_HOST="${_ssh_parts[1]:-}"
        SSH_PORT="${_ssh_parts[2]:-22}"
        [[ -n "$SSH_HOST" ]] || die "Could not parse SSH target from pod status"
        # Root-home/rootful-Docker only: remote paths assume /root. Non-root
        # provider images are rejected after SSH auth (see main flow).
        REMOTE_ROOT="/root"
        # Cloud IPs get recycled across pods; a stale known_hosts entry from a
        # prior pod at this same IP makes even StrictHostKeyChecking=no refuse
        # to connect (observed 2026-07-08 after a spot-instance reclaim).
        ssh-keygen -f "$HOME/.ssh/known_hosts" -R "$SSH_HOST" >/dev/null 2>&1 || true
        log "Pod ready: ssh -i $SSH_KEY -p $SSH_PORT ${SSH_USER}@${SSH_HOST}"
        return 0
      fi
    fi
    sleep 15
  done
}

remote() {
  ssh -T -i "$SSH_KEY" -o StrictHostKeyChecking=no -p "$SSH_PORT" "${SSH_USER}@${SSH_HOST}" "$@"
}

wait_for_ssh_auth() {
  # `prime pods status` reporting ACTIVE + ssh-ready is not sufficient on some
  # providers (observed 2026-07-08 on crusoecloud's a100-80gb.1x: the API said
  # ready, but publickey auth returned "Permission denied" for ~90s straight
  # before the script gave up) -- authorized_keys can propagate after the
  # instance is otherwise reachable. Probe real SSH auth directly, separately
  # from the rsync retry loop below, before trusting the pod is usable.
  #
  # Also a pragmatic safety net for a related failure: observed 2026-07-08 on
  # crusoecloud's a100-80gb.1x, the parsed SSH_USER (root, the previous
  # hardcoded default whenever the API returns a bare `ip` field) was flat-out
  # rejected -- the pod's actual default login user was "ubuntu".
  # parse_ssh_target now prefers the API's stated user, but also try
  # known-good alternates here so a future provider quirk degrades to a
  # slower connection instead of a dead run.
  local attempt candidate
  local candidates=("$SSH_USER")
  for candidate in ubuntu root; do
    if [[ ! " ${candidates[*]} " == *" $candidate "* ]]; then
      candidates+=("$candidate")
    fi
  done
  for attempt in $(seq 1 20); do
    for candidate in "${candidates[@]}"; do
      if ssh -T -i "$SSH_KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=10 \
        -o BatchMode=yes -p "$SSH_PORT" "${candidate}@${SSH_HOST}" true 2>/dev/null; then
        if [[ "$candidate" != "$SSH_USER" ]]; then
          log "SSH auth confirmed as fallback user '$candidate' (parsed user '$SSH_USER' was rejected)"
          SSH_USER="$candidate"
        else
          log "SSH auth confirmed (attempt $attempt)"
        fi
        return 0
      fi
    done
    log "SSH auth not ready yet (attempt $attempt/20, tried: ${candidates[*]}); retrying in 15s"
    sleep 15
  done
  die "SSH auth never became usable at ${SSH_HOST}:${SSH_PORT} (tried users: ${candidates[*]}) after 20 attempts"
}

remote_rsync() {
  local attempt stderr ssh_cmd
  ssh_cmd="ssh -T -i $SSH_KEY -o StrictHostKeyChecking=no -p $SSH_PORT"
  # Permanent, non-transient rsync failures. A retried rsync would only
  # re-trigger the same fatal error and burn the whole retry budget on a
  # known-dead run, so these fail fast (one attempt, no sleep).
  # Covers permission errors and path errors (e.g. a missing destination
  # directory: `rsync: mkdir "/root/webcurator-gym" failed: No such file or
  # directory (2)`) even when no "permission denied" text is present. Must
  # NOT match transient connection errors (Connection refused/timed out,
  # "connection unexpectedly closed", timeouts, I/O errors).
  local -r PERMANENT_RE='permission denied|no such file or directory|cannot stat|link_stat|mkdir .*failed|read-only file system|read-only filesystem'
  for attempt in 1 2 3 4 5; do
    stderr="$(mktemp)"
    if rsync -az --delete \
      --exclude '.git/' \
      --exclude '.venv/' \
      --exclude '**/.venv/' \
      --exclude 'outputs/' \
      --exclude '**/__pycache__/' \
      --exclude '**/.pytest_cache/' \
      --exclude '**/.mypy_cache/' \
      --exclude 'docs/site/data/' \
      -e "$ssh_cmd" \
      "$ROOT/" "${SSH_USER}@${SSH_HOST}:/root/webcurator-gym/" 2>"$stderr"; then
      rm -f "$stderr"
      return 0
    fi
    if grep -Eqi "$PERMANENT_RE" "$stderr"; then
      local reason
      reason="$(grep -Ei "$PERMANENT_RE" "$stderr" | head -n1)"
      rm -f "$stderr"
      die "rsync permanent failure (no retry): $reason"
    fi
    rm -f "$stderr"
    log "rsync attempt $attempt failed; retrying in 20s"
    sleep 20
  done
  die "rsync failed after 5 attempts"
}

remote_provision_cpu() {
  log "Provisioning CPU pod software stack (uv, py3.12, decon, webcurator-runtime)"
  remote bash -s <<'REMOTE'
set -euo pipefail
[ "$(id -u)" = "0" ] && REMOTE_ROOT=/root || REMOTE_ROOT=/home/$(whoami)
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
exec > $REMOTE_ROOT/wcg-provision.log 2>&1

echo "=== Docker (agent harness runtime; no GPU) ==="
if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sh
fi
systemctl start docker || true

echo "=== uv + python env ==="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
cd $REMOTE_ROOT/webcurator-gym
rm -rf .venv
uv venv -p 3.12
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install prime
cd environments/pretrain_data_curator
uv pip install -e .
# torch lives only in this project's own uv "dev" dependency-group (uv.lock),
# which plain `uv pip install -e .` does not pull in. student_model.py now
# imports torch at module level (real, non-AST import, via trainer.py's
# `estimate_instantiated_param_count`), so verify through `uv run` -- the
# same resolution path `uv run eval` uses below -- not the bare venv above.
uv run python -c "import pretrain_data_curator; print('import OK', __import__('sys').version.split()[0])"

echo "=== decon (static vendored binary or native fallback) ==="
DECON_BIN=$REMOTE_ROOT/webcurator-gym/environments/pretrain_data_curator/decon/bin/decon
if ! "$DECON_BIN" --version >/dev/null 2>&1; then
  echo "Vendored decon missing or incompatible (often glibc); building natively"
  bash $REMOTE_ROOT/webcurator-gym/environments/pretrain_data_curator/decon/build_from_source.sh
fi
"$DECON_BIN" --version
file "$DECON_BIN"

echo "=== build custom harness-runtime image (hf + decon for the agent container) ==="
# Build only after decon is compiled so COPY decon/ picks up a runnable binary.
cd $REMOTE_ROOT/webcurator-gym/environments/pretrain_data_curator
docker build -f Dockerfile.runtime -t webcurator-runtime:latest .
echo "--- preflight: hf/huggingface_hub + zstd codec + /workspace/decon/bin/decon ---"
# Validate the same absolute path self_score.py probes in the agent container.
# The zstd round-trip guards against the production "Compression type zstd not
# supported" failure when materializing zstd-compressed Hub datasets
# (e.g. mlfoundations/dclm-baseline-1.0, monology/pile-uncopyrighted): the
# datasets/fsspec read path needs the zstandard codec installed in the image.
docker run --rm -w /workspace webcurator-runtime:latest bash -lc \
  'command -v hf && python -c "import huggingface_hub; print(\"huggingface_hub\", huggingface_hub.__version__)" && python -c "import zstandard; c=zstandard.ZstdCompressor(); d=zstandard.ZstdDecompressor(); raw=b\"webcurator-zstd-preflight\"; assert d.decompress(c.compress(raw))==raw; print(\"zstandard\", zstandard.__version__)" && test -x /workspace/decon/bin/decon && /workspace/decon/bin/decon --version'

echo "=== PROVISION DONE ==="
REMOTE
}

remote_provision_gpu() {
  log "Provisioning pod software stack (docker, uv, py3.12, decon)"
  remote bash -s <<'REMOTE'
set -euo pipefail
[ "$(id -u)" = "0" ] && REMOTE_ROOT=/root || REMOTE_ROOT=/home/$(whoami)
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
exec > $REMOTE_ROOT/wcg-provision.log 2>&1

echo "=== GPU ==="
nvidia-smi -L

echo "=== Docker ==="
if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sh
fi
systemctl start docker || true

echo "=== NVIDIA container toolkit ==="
if ! command -v nvidia-ctk >/dev/null 2>&1; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -sL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  apt-get update -qq && apt-get install -y -qq nvidia-container-toolkit
fi
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

echo "=== GPU docker smoke + base image pull ==="
docker pull pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime
docker run --rm --gpus all pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime nvidia-smi -L

echo "=== uv + python env ==="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
cd $REMOTE_ROOT/webcurator-gym
rm -rf .venv
uv venv -p 3.12
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install prime
cd environments/pretrain_data_curator
uv pip install -e .
# torch lives only in this project's own uv "dev" dependency-group (uv.lock),
# which plain `uv pip install -e .` does not pull in. student_model.py now
# imports torch at module level (real, non-AST import, via trainer.py's
# `estimate_instantiated_param_count`), so verify through `uv run` -- the
# same resolution path `uv run eval` uses below -- not the bare venv above.
uv run python -c "import pretrain_data_curator; print('import OK', __import__('sys').version.split()[0])"

echo "=== decon (static vendored binary or native fallback) ==="
DECON_BIN=$REMOTE_ROOT/webcurator-gym/environments/pretrain_data_curator/decon/bin/decon
if ! "$DECON_BIN" --version >/dev/null 2>&1; then
  echo "Vendored decon missing or incompatible (often glibc); building natively"
  bash $REMOTE_ROOT/webcurator-gym/environments/pretrain_data_curator/decon/build_from_source.sh
fi
"$DECON_BIN" --version
file "$DECON_BIN"

echo "=== build custom harness-runtime image (hf + decon for the agent container) ==="
# The codex AGENT runs inside the harness-runtime container (DockerConfig,
# workdir=/workspace), not on this host. The bare pytorch image ships neither
# the decon binary nor huggingface-hub, so the agent's self_score.py silently
# skipped leakage and had no hf. Dockerfile.runtime bakes both in; build it only
# after decon is compiled above so COPY decon/ picks up a runnable binary.
# 400M configs point docker_image at "webcurator-runtime:latest".
cd $REMOTE_ROOT/webcurator-gym/environments/pretrain_data_curator
docker build -f Dockerfile.runtime -t webcurator-runtime:latest .
echo "--- preflight: hf/huggingface_hub + zstd codec + /workspace/decon/bin/decon ---"
# Validate the same absolute path self_score.py probes in the agent container.
# The zstd round-trip guards against the production "Compression type zstd not
# supported" failure when materializing zstd-compressed Hub datasets
# (e.g. mlfoundations/dclm-baseline-1.0, monology/pile-uncopyrighted): the
# datasets/fsspec read path needs the zstandard codec installed in the image.
docker run --rm -w /workspace webcurator-runtime:latest bash -lc \
  'command -v hf && python -c "import huggingface_hub; print(\"huggingface_hub\", huggingface_hub.__version__)" && python -c "import zstandard; c=zstandard.ZstdCompressor(); d=zstandard.ZstdDecompressor(); raw=b\"webcurator-zstd-preflight\"; assert d.decompress(c.compress(raw))==raw; print(\"zstandard\", zstandard.__version__)" && test -x /workspace/decon/bin/decon && /workspace/decon/bin/decon --version'
docker run --rm --gpus all webcurator-runtime:latest nvidia-smi -L

echo "=== PROVISION DONE ==="
REMOTE
}

remote_patch_codex_for_prime_inference() {
  # Codex 0.137.0 enables stable `type=namespace` features by default
  # (`apps` and `multi_agent`). Prime Inference rejects every Responses-API
  # `type=namespace` tool (HTTP 400 invalid_request) on the Codex/Responses
  # models used by the 400M eval, so disable BOTH namespace features at the
  # Codex harness boundary (idempotently, via a marker) before launching eval.
  # Function tools, web_search, the shell tool, and HF CLI skill access are
  # left intact -- only the namespace-producing features are disabled.
  log "Patching remote Codex harness to --disable apps --disable multi_agent (Prime Inference compat)"
  remote bash -s <<'REMOTE' || die "remote Codex harness patch failed (Prime Inference namespace-tool compat); refusing to launch eval against an unpatched harness"
set -euo pipefail
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
cd /root/webcurator-gym/environments/pretrain_data_curator
uv run python <<'PY'
from pathlib import Path
import verifiers

path = Path(verifiers.__file__).parent / "v1/harnesses/codex/harness.py"
text = path.read_text()
marker = "PRIME_INFERENCE_DISABLE_NAMESPACE_TOOLS"
if marker in text:
    print(f"already patched: {path}")
else:
    needle = (
        "        tool_config = [\n"
        "            arg\n"
        "            for tool in self.config.disabled_tools or []\n"
        "            for arg in (\"--disable\", tool)\n"
        "        ]"
    )
    if needle not in text:
        raise SystemExit(f"codex harness patch needle missing in {path}")
    repl = (
        "        # "
        + marker
        + "\n"
        "        _disabled = list(self.config.disabled_tools or [])\n"
        "        for _feature in (\"apps\", \"multi_agent\"):\n"
        "            if _feature not in _disabled:\n"
        "                _disabled.append(_feature)\n"
        "        tool_config = [\n"
        "            arg\n"
        "            for tool in _disabled\n"
        "            for arg in (\"--disable\", tool)\n"
        "        ]"
    )
    path.write_text(text.replace(needle, repl, 1))
    print(f"patched: {path}")
PY
REMOTE
}

run_remote_eval() {
  # Durable, detached remote eval. The eval is launched under setsid/nohup on the
  # pod with a PID marker, a RUNNING status marker, and a durable remote log, so a
  # transient SSH disconnect during the multi-hour run cannot kill it. The local
  # launcher only polls status over fresh SSH connections (tolerating bounded
  # transient SSH failures with backoff) and proceeds to result validation only
  # after confirmed remote completion. The EXIT trap still terminates the pod.
  local run_dir="$REMOTE_ROOT/wcg-eval-${RUN_NAME}.d"
  local eval_log="$run_dir/eval.log"
  local eval_pid="$run_dir/eval.pid"
  local eval_status="$run_dir/status"
  if [[ "$CURATION_ONLY" -eq 1 ]]; then
    log "Launching curation-only eval on CPU pod (detached, ~30-60 min)"
  else
    log "Launching eval on pod (detached; this can take ~2h for 400M)"
  fi
  # Prime Inference compat: disable Codex's default namespace tools before eval.
  remote_patch_codex_for_prime_inference

  # --- durable detached launch ------------------------------------------------
  # Writes eval.sh + PID marker + RUNNING status. Idempotent: if a live eval is
  # already tracked (PID marker present and process alive), do NOT start a
  # duplicate -- this protects reconnect/re-entry while a prior eval runs.
  # All dynamic values (paths, MODEL, EVAL_CONFIG, REMOTE_ROOT) are passed as
  # positional args to a QUOTED heredoc, so user-controlled values (MODEL /
  # EVAL_CONFIG) are never embedded unescaped in remote shell text -- this
  # prevents local command-substitution / quote breakout.
  remote bash -s "$run_dir" "$eval_log" "$eval_pid" "$eval_status" "$MODEL" "$EVAL_CONFIG" "$REMOTE_ROOT" <<'REMOTE'
set -euo pipefail
export PATH="$7/webcurator-gym/environments/pretrain_data_curator/decon/bin:$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
RD="$1"; LOG="$2"; PID="$3"; ST="$4"
MODEL="$5"; EVAL_CONFIG="$6"
mkdir -p "$RD"
cat > "$RD/eval.sh" <<'EVAL'
#!/usr/bin/env bash
# Detached Codex/Responses eval wrapper. Secrets are sourced (never echoed).
set -uo pipefail
REPO_ROOT="${WCG_REPO_ROOT:-/root/webcurator-gym}"
export PATH="$REPO_ROOT/environments/pretrain_data_curator/decon/bin:$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
MODEL="$1"
EVAL_CONFIG="$2"
ST="$3"
LOG="$4"

# `uv run eval` exits 0 even when the rollout itself failed (harness error,
# truncation, no metrics), so rc=0 alone must never be reported as success.
# Validate the exact results dir this run logged -- never a repo-wide scan.
validate_run_results() {
  local log="$1" line rel
  line="$(grep -Eo 'results: outputs/[^[:space:]]+' "$log" 2>/dev/null | tail -1 || true)"
  rel="${line#results: }"
  if [[ -z "$rel" || "$rel" == /* || "$rel" == *".."* ]]; then
    echo "[validate] FAIL: no usable results path logged by this run" >&2
    return 1
  fi
  python3 - "$PWD/$rel/results.jsonl" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])


def bad(message):
    print("[validate] FAIL: %s (%s)" % (message, path), file=sys.stderr)
    raise SystemExit(1)


if not path.is_file() or path.stat().st_size == 0:
    bad("results.jsonl is missing or empty")
rows = [line for line in path.read_text().splitlines() if line.strip()]
if not rows:
    bad("results.jsonl has no rows")
for index, line in enumerate(rows):
    try:
        row = json.loads(line)
    except json.JSONDecodeError as exc:
        bad("row %d is not valid JSON: %s" % (index, exc))
    if not row.get("is_completed"):
        bad("row %d is not finalized (is_completed is false)" % index)
    stop = row.get("stop_condition")
    if stop in ("error", "truncation"):
        bad("row %d stop_condition=%s" % (index, stop))
    errors = row.get("errors") or []
    if errors:
        bad("row %d has %d rollout error(s): %s" % (index, len(errors), str(errors[0])[:300]))
    if not (row.get("metrics") or {}):
        bad("row %d has empty metrics" % index)
    if "reward" not in (row.get("rewards") or {}) and row.get("reward") is None:
        bad("row %d has no reward" % index)
print("[validate] OK: %d finalized row(s) with metrics in %s" % (len(rows), path))
PY
}

cd "$REPO_ROOT"
set -a
source secrets.env
set +a
: "${HF_TOKEN:?HF_TOKEN missing in secrets.env}"
: "${PRIME_API_KEY:?PRIME_API_KEY missing in secrets.env}"
cd environments/pretrain_data_curator
uv run eval -m "$MODEL" @ "$EVAL_CONFIG"
rc=$?
if [[ $rc -eq 0 ]]; then
  validate_run_results "$LOG" || rc=65
fi
# Atomic status completion: write to a temp file, then rename into place.
echo "EXIT=$rc" > "$ST.tmp" && mv "$ST.tmp" "$ST"
exit $rc
EVAL
chmod +x "$RD/eval.sh"
if [[ -f "$PID" ]]; then
  OPID=$(cat "$PID" 2>/dev/null || true)
  if [[ -n "$OPID" ]] && kill -0 "$OPID" 2>/dev/null; then
    echo "eval already running (pid=$OPID); not starting duplicate"
    exit 0
  fi
fi
echo "RUNNING" > "$ST.tmp" && mv "$ST.tmp" "$ST"
setsid nohup bash "$RD/eval.sh" "$MODEL" "$EVAL_CONFIG" "$ST" "$LOG" > "$LOG" 2>&1 &
echo $! > "$PID"
echo "launched eval (pid=$(cat "$PID"), log=$LOG)"
REMOTE

  # --- monitor (tolerates transient SSH disconnects) --------------------------
  monitor_remote_eval "$run_dir" "$eval_log" "$eval_pid" "$eval_status"
}

# Read the durable status marker over a fresh SSH connection.
# Args: run_dir eval_log eval_pid eval_status
_remote_eval_probe() {
  remote bash -s "$1" "$2" "$3" "$4" <<'RM'
# WCG_PROBE
RD="$1"; LOG="$2"; PID="$3"; ST="$4"
if [[ -f "$ST" ]]; then
  s=$(cat "$ST")
  if [[ "$s" == EXIT=* ]]; then echo "STATUS=done EXIT=${s#EXIT=}"; exit 0; fi
  echo "STATUS=running"; exit 0
fi
if [[ -f "$PID" ]]; then
  OPID=$(cat "$PID" 2>/dev/null || true)
  if [[ -n "$OPID" ]] && kill -0 "$OPID" 2>/dev/null; then echo "STATUS=running"; exit 0; fi
  echo "STATUS=nostatus_deadpid"; exit 0
fi
echo "STATUS=nostatus_nopid"; exit 0
RM
}

# Best-effort progress tail; never owns the remote process lifetime.
_remote_tail_progress() {
  remote bash -s "$1" <<'RT' 2>/dev/null || true
# WCG_TAIL
LOG="$1"
if [[ -f "$LOG" ]]; then tail -c 2000 "$LOG"; fi
RT
}

# Poll the remote status over fresh SSH connections. Tolerates bounded transient
# SSH failures (retries + backoff) without killing the detached remote eval; only
# proceeds after confirmed completion, or fails safe / times out.
monitor_remote_eval() {
  local run_dir="$1" eval_log="$2" eval_pid="$3" eval_status="$4"
  local timeout="${WCG_EVAL_TIMEOUT_SECONDS:-$(( CURATION_ONLY == 1 ? 7200 : 14400 ))}"
  local poll="${WCG_EVAL_POLL_INTERVAL:-60}"
  local retries="${WCG_EVAL_MON_RETRIES:-5}"
  local backoff="${WCG_EVAL_MON_BACKOFF:-10}"
  local deadline=$(( $(date +%s) + timeout ))
  log "Monitoring remote eval (log=$eval_log, timeout=${timeout}s, poll=${poll}s)"
  while true; do
    local now; now=$(date +%s)
    if (( now > deadline )); then
      die "eval monitor timed out after ${timeout}s (transient SSH limit / pod unresponsive); remote log+status preserved at ${run_dir}"
    fi
    local probe="" ok=0
    local attempt
    for attempt in $(seq 1 "$retries"); do
      if probe_out=$(_remote_eval_probe "$run_dir" "$eval_log" "$eval_pid" "$eval_status"); then
        probe="$probe_out"; ok=1; break
      fi
      sleep "$backoff"
    done
    if (( ok )); then
      local st; st=$(printf '%s\n' "$probe" | grep -E '^STATUS=' | tail -1)
      case "$st" in
        STATUS=done*)
          local code="${st#STATUS=done EXIT=}"
          if [[ "$code" == "0" ]]; then
            log "remote eval completed successfully"
            return 0
          fi
          die "remote eval exited non-zero (code=$code); see remote log $eval_log"
          ;;
        STATUS=running)
          : ;;
        STATUS=nostatus_deadpid|STATUS=nostatus_nopid)
          die "remote eval has no completion marker and no live PID; failing safe (log $eval_log)"
          ;;
        *)
          log "monitor probe inconclusive; will retry"
          ;;
      esac
    else
      log "monitor SSH probe failed (transient); will retry while pod active"
    fi
    _remote_tail_progress "$eval_log" || true
    sleep "$poll"
  done
}

find_remote_results_dir() {
  local remote_log="$REMOTE_ROOT/wcg-eval-${RUN_NAME}.d/eval.log"
  remote bash -s <<REMOTE
# WCG_FIND_RESULTS
set -euo pipefail
LOG="${remote_log}"
_normalize_rel() {
  # Accept only a non-empty, relative, traversal-free outputs subpath.
  local p="\$1"
  [[ -n "\$p" ]] || return 1
  [[ "\$p" == /* ]] && return 1
  [[ "\$p" == *".."* ]] && return 1
  [[ "\$p" == *"/./"* || "\$p" == */. || "\$p" == ./* ]] && return 1
  return 0
}
if [[ -f "\$LOG" ]]; then
  RESULTS_LINE="\$(grep -Eo 'results: outputs/[^[:space:]]+' "\$LOG" | tail -1 || true)"
  if [[ -n "\$RESULTS_LINE" ]]; then
    REL="\${RESULTS_LINE#results: }"
    REL="\${REL#outputs/}"            # strip exactly one leading outputs/
    _normalize_rel "\$REL" || exit 1
    echo "\$REL"
    exit 0
  fi
fi
# Fallback: newest non-empty results.jsonl under outputs/.
python3 - <<'PY'
import json
from pathlib import Path
root = Path("$REMOTE_ROOT/webcurator-gym/environments/pretrain_data_curator/outputs")
candidates = []
for path in root.rglob("results.jsonl"):
    try:
        if path.stat().st_size <= 0:
            continue
        row = json.loads(path.read_text().splitlines()[0])
        if row.get("is_completed"):
            candidates.append((path.stat().st_mtime, path.parent))
    except Exception:
        continue
if not candidates:
    raise SystemExit(1)
candidates.sort()
print(candidates[-1][1].relative_to(root))
PY
REMOTE
}

download_results() {
  local rel_dir="$1"
  log "Downloading results from outputs/$rel_dir"
  mkdir -p "$LOCAL_OUT_DIR"
  rsync -az \
    -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no -p $SSH_PORT" \
    "${SSH_USER}@${SSH_HOST}:$REMOTE_ROOT/webcurator-gym/environments/pretrain_data_curator/outputs/${rel_dir}/" \
    "$LOCAL_OUT_DIR/"
  [[ -s "$LOCAL_OUT_DIR/results.jsonl" ]] || die "Downloaded results.jsonl is empty"
  write_resolved_config "$LOCAL_OUT_DIR/config.toml"
  local remote_log="$REMOTE_ROOT/wcg-eval-${RUN_NAME}.d/eval.log"
  scp -i "$SSH_KEY" -o StrictHostKeyChecking=no -P "$SSH_PORT" \
    "${SSH_USER}@${SSH_HOST}:${remote_log}" "$LOCAL_OUT_DIR/eval-stream.log" >/dev/null 2>&1 || true
  if [[ "$CURATION_ONLY" -eq 1 ]]; then
    cat > "$LOCAL_OUT_DIR/README.txt" <<EOF
${MODEL} 400M curation-only eval ($(date -u +%Y-%m-%d))

use_real_trainer=false — heuristic proxy trainer for fast curation iteration.
Full conversation trace is in results.jsonl. Perf metrics are not meaningful.

Config: ${EVAL_CONFIG}
EOF
  fi
}

rebuild_site() {
  log "Rebuilding docs bench site"
  python3 "$ENV_DIR/docs/build_site.py"
  python3 - <<'PY' "$ENV_DIR/docs/site/data/manifest.json"
import json, sys
from pathlib import Path
m = json.loads(Path(sys.argv[1]).read_text())
print(f"site runs={m['run_count']}")
for r in m["runs"]:
    print(f"  {r['model']:28} reward={r['reward']:.4f}")
PY
}

summarize_results() {
  python3 - <<'PY' "$LOCAL_OUT_DIR/results.jsonl"
import json, sys
row = json.loads(open(sys.argv[1]).read().splitlines()[0])
reward = (row.get("rewards") or {}).get("reward", row.get("reward"))
metrics = row.get("metrics") or {}
print("reward", reward)
for k in ("perf_loss", "perf_vs_baseline", "corpus_tokens", "num_sources", "leakage_score", "decon_error"):
    if k in metrics:
        print(f"{k} {metrics[k]}")
PY
}

if [[ "$DRY_RUN" -eq 1 ]]; then
  log "DRY RUN"
  log "model=$MODEL"
  log "base_config=$EVAL_CONFIG"
  log "run_name=$RUN_NAME"
  log "compute_type=$GPU_TYPE pod_image=$POD_IMAGE disk_gb=$DISK_GB pod_name=$POD_NAME curation_only=$CURATION_ONLY"
  log "local_out=$LOCAL_OUT_DIR"
  exit 0
fi

PICKED="$(pick_compute_with_retries)" || die "No available $GPU_TYPE capacity (no GPU fallback; not substituting another GPU type)"
GPU_TYPE="${PICKED%% *}"
CLOUD_ID="${PICKED##* }"
log "Selected cloud_id=$CLOUD_ID ($GPU_TYPE)"

# Shared A100 inventory (e.g. the 22V pool) can flicker between
# `availability list` and `pods create` under contention; retry a few times
# with a re-pick before giving up (observed 2026-07-08: "No valid GPU
# configuration found" on a cloud_id that `availability list` just reported
# as Available, succeeded on immediate retry).
CREATE_LOG="$(mktemp)"
POD_ID=""
EXCLUDED_CLOUD_IDS=""
for attempt in 1 2 3 4 5; do
  if prime pods create \
    --cloud-id "$CLOUD_ID" \
    --name "$POD_NAME" \
    --disk-size "$DISK_GB" \
    --image "$POD_IMAGE" \
    --yes --plain 2>&1 | tee "$CREATE_LOG"; then
    POD_ID="$(sed -n 's/.*Successfully created pod //p' "$CREATE_LOG" | awk '{print $1}')"
    if [[ -n "$POD_ID" ]]; then
      # Last line of defense against a GPU downgrade: assert the pod Prime
      # actually created matches the GPU type we asked for. POD_ID is already
      # set, so the EXIT trap terminates the pod if this fails.
      CREATED_GPU="$(sed -n 's/^gpuType:[[:space:]]*//p' "$CREATE_LOG" | head -n1 | tr -d '[:space:]')"
      if [[ -n "$CREATED_GPU" && "$CREATED_GPU" != "$GPU_TYPE" ]]; then
        die "pod $POD_ID was created as $CREATED_GPU but $GPU_TYPE was requested (no GPU fallback allowed) -- terminating"
      fi
      log "Verified created pod GPU type: ${CREATED_GPU:-unreported} (requested $GPU_TYPE)"
      break
    fi
  fi
  # Exclude the cloud_id that just failed so a single consistently-broken
  # offer (observed 2026-07-08: "1A100.40S.22V" repeatedly returned "No valid
  # GPU configuration found" from `pods create` despite `availability list`
  # reporting it Available, and kept getting re-picked as the cheapest
  # remaining option) can't eat the whole retry budget.
  log "prime pods create attempt $attempt failed on cloud_id=$CLOUD_ID; excluding it and retrying in 15s"
  EXCLUDED_CLOUD_IDS="$EXCLUDED_CLOUD_IDS $CLOUD_ID"
  sleep 15
  PICKED="$(pick_compute_with_retries)" || die "No available $GPU_TYPE capacity (no GPU fallback; not substituting another GPU type)"
  GPU_TYPE="${PICKED%% *}"
  CLOUD_ID="${PICKED##* }"
  log "Re-selected cloud_id=$CLOUD_ID ($GPU_TYPE)"
done
rm -f "$CREATE_LOG"
[[ -n "$POD_ID" ]] || die "prime pods create failed after 5 attempts"

log "Created pod $POD_ID ($POD_NAME)"
wait_for_pod "$POD_ID"
wait_for_ssh_auth

# This launcher is root-home/rootful-Docker only: every remote path
# (/root/webcurator-gym, secrets.env, decon binary, Docker) assumes a root
# login. A pod that authenticated as a non-root user (e.g. MassedCompute /
# crusoecloud ubuntu images) is unsupported and would silently corrupt the
# run, so fail fast. The EXIT trap terminates the pod before we bail out.
if [[ "$SSH_USER" != "root" ]]; then
  die "FATAL: pod authenticated as non-root user '$SSH_USER'; this launcher requires root SSH access (rootful Docker / root home). Unsupported provider image -- terminating pod."
fi
REMOTE_ROOT="/root"

log "Syncing repository to pod"
remote_rsync
upload_secrets

if [[ "$CURATION_ONLY" -eq 1 ]]; then
  remote_provision_cpu
else
  remote_provision_gpu
fi
run_remote_eval

RESULTS_REL="$(find_remote_results_dir)" || die "Could not locate remote results directory"
download_results "$RESULTS_REL"

log "Eval summary:"
summarize_results

if [[ "$SKIP_SITE" -eq 0 ]]; then
  rebuild_site
  log "Site: file://$ENV_DIR/docs/site/index.html"
fi

log "Done. Artifacts: $LOCAL_OUT_DIR"
