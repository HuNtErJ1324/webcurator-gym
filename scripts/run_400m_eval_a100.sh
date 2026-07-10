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

# Progress/status always on stderr so command substitutions (e.g.
# stage="$(stage_massedcompute_workspace)") capture only intended path/data.
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
command -v tar >/dev/null || die "tar not found"
command -v ssh >/dev/null || die "ssh not found"
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

remote_repo_dir() {
  # root@datacrunch → /root/webcurator-gym; ubuntu@massedcompute → /home/ubuntu/webcurator-gym
  echo "${REMOTE_ROOT}/webcurator-gym"
}

set_remote_root_for_user() {
  local user="$1"
  if [[ "$user" == "root" ]]; then
    REMOTE_ROOT="/root"
  else
    REMOTE_ROOT="/home/${user}"
  fi
}

# Build the remote shell snippet that creates + verifies a repo directory.
# Kept as a pure string builder so unit tests can assert absolute paths survive
# OpenSSH's argv-flattening (see ensure_remote_repo_dir).
ensure_remote_repo_dir_cmd() {
  local remote_dir="$1"
  # %q shell-escapes so spaces/metacharacters in the path stay one word remotely.
  printf 'mkdir -p %q && test -d %q && test -w %q\n' \
    "$remote_dir" "$remote_dir" "$remote_dir"
}

probe_remote_repo_dir_cmd() {
  local remote_dir="$1"
  printf 'test -d %q\n' "$remote_dir"
}

# Write a remote file from SSH stdin (exec channel). Avoids scp/SFTP, which on
# MassedCompute can fail to open paths that SSH-exec mkdir/test just verified
# (pod 60ee57c4). mkdir of the parent is folded into THIS same remote shell so
# a later cross-exec write is not required. Payload stays on stdin — never in
# argv or launcher logs.
write_remote_file_from_stdin_cmd() {
  local remote_path="$1"
  printf 'umask 077 && mkdir -p "$(dirname %q)" && cat > %q && chmod 600 %q\n' \
    "$remote_path" "$remote_path" "$remote_path"
}

cat_remote_file_cmd() {
  local remote_path="$1"
  printf 'cat %q\n' "$remote_path"
}

test_remote_file_cmd() {
  local remote_path="$1"
  printf 'test -f %q\n' "$remote_path"
}

# Fail-fast cross-exec affinity probe (MassedCompute pod 8b009f19): mkdir in
# one remote() exec, then create a NEW file inside that dir in a second exec.
# ControlMaster keeps the TCP session; this proves durable path visibility
# across multiplexed writes before expensive provision/eval.
affinity_preflight_mkdir_cmd() {
  local marker_dir="$1"
  local host_note="$2"
  printf 'umask 077 && mkdir -p %q && test -d %q && test -w %q && hostname && echo %q\n' \
    "$marker_dir" "$marker_dir" "$marker_dir" "$host_note"
}

affinity_preflight_create_file_cmd() {
  local marker_dir="$1"
  local marker_path="$2"
  local token="$3"
  # Second exec: do NOT mkdir here — require the prior exec's directory to exist,
  # then create a brand-new file inside it (the cross-exec write under test).
  printf 'test -d %q || { echo "affinity-preflight: missing dir at %q" >&2; exit 1; }; umask 077 && echo %q > %q && chmod 600 %q && hostname\n' \
    "$marker_dir" "$marker_dir" "$token" "$marker_path" "$marker_path"
}

affinity_preflight_read_cmd() {
  local marker_path="$1"
  # Fail loudly if the marker is missing (affinity break); then cat + hostname.
  printf 'test -f %q || { echo "affinity-preflight: missing marker at %q" >&2; exit 1; }; cat %q && hostname\n' \
    "$marker_path" "$marker_path" "$marker_path"
}

verify_ssh_connection_affinity() {
  # Exec1: mkdir. Exec2: create NEW file in that dir. Exec3: read token back.
  local marker_dir marker_path token host_note mkdir_out create_out read_out
  local mkdir_cmd create_cmd read_cmd
  [[ -n "${SSH_CONTROL_PATH:-}" ]] || die "SSH ControlMaster not configured; cannot verify connection affinity"
  [[ -n "${REMOTE_ROOT:-}" ]] || die "REMOTE_ROOT unset; cannot verify connection affinity"
  marker_dir="${REMOTE_ROOT}/.wcg-ssh-affinity"
  marker_path="${marker_dir}/marker.$$.$RANDOM"
  token="wcg-affinity-$(date -u +%Y%m%dT%H%M%SZ)-$$"
  host_note="control_path=${SSH_CONTROL_DIR:-unset}"
  mkdir_cmd="$(affinity_preflight_mkdir_cmd "$marker_dir" "$host_note")"
  create_cmd="$(affinity_preflight_create_file_cmd "$marker_dir" "$marker_path" "$token")"
  read_cmd="$(affinity_preflight_read_cmd "$marker_path")"
  log "SSH affinity preflight: mkdir then create-file across ControlMaster execs (${SSH_USER}@${SSH_HOST})"
  if ! mkdir_out="$(remote "$mkdir_cmd")"; then
    die "SSH affinity preflight MKDIR failed on ${SSH_USER}@${SSH_HOST} (${host_note}). Refusing provision/eval."
  fi
  if ! create_out="$(remote "$create_cmd")"; then
    die "SSH affinity preflight CREATE-FILE failed on ${SSH_USER}@${SSH_HOST} after mkdir (${host_note}). Cross-exec write into ${marker_dir} failed; mkdir evidence: ${mkdir_out}"
  fi
  if ! read_out="$(remote "$read_cmd")"; then
    die "SSH affinity preflight READ failed on ${SSH_USER}@${SSH_HOST} after create (${host_note}). Marker ${marker_path} missing; mkdir=[${mkdir_out}] create=[${create_out}]"
  fi
  if [[ "$read_out" != *"$token"* ]]; then
    die "SSH affinity preflight token mismatch on ${SSH_USER}@${SSH_HOST}. mkdir=[${mkdir_out}] create=[${create_out}] read=[${read_out}] (${host_note})"
  fi
  log "SSH affinity preflight OK (hostname/session evidence): $(printf '%s' "$mkdir_out" | tr '\n' ' ' | head -c 200)"
  # Best-effort cleanup of the probe marker (ignore failures).
  remote "$(printf 'rm -f %q\n' "$marker_path")" >/dev/null 2>&1 || true
}

ensure_remote_repo_dir() {
  # Do not rely on rsync's implicit mkdir: MassedCompute (ubuntu@) has been
  # observed to report a successful sync while $REMOTE_ROOT/webcurator-gym is
  # still missing, so scp of secrets.env fails with "No such file or directory".
  # Create + verify the configured remote repo dir before any secrets upload.
  #
  # Critical: pass ONE remote command string to `remote` / ssh. Do NOT use
  # `remote bash -lc "mkdir -p '$dir' && ..."`. OpenSSH joins remote argv with
  # spaces before the login shell runs them, so bash -c's script becomes only
  # "mkdir" and the path is lost → "mkdir: missing operand" (pod b89b046e).
  local remote_dir remote_cmd
  [[ -n "${REMOTE_ROOT:-}" ]] || die "REMOTE_ROOT is unset; cannot ensure remote repo directory"
  [[ -n "${SSH_USER:-}" && -n "${SSH_HOST:-}" ]] || die "SSH target unset; cannot ensure remote repo directory"
  remote_dir="$(remote_repo_dir)"
  log "Ensuring remote repo directory exists and is writable: ${SSH_USER}@${SSH_HOST}:${remote_dir}"
  remote_cmd="$(ensure_remote_repo_dir_cmd "$remote_dir")"
  if ! remote "$remote_cmd"; then
    die "Remote repo directory unavailable or not writable: ${SSH_USER}@${SSH_HOST}:${remote_dir}. Refusing to upload secrets."
  fi
  # Fail loudly if a subsequent probe still cannot see the directory.
  remote_cmd="$(probe_remote_repo_dir_cmd "$remote_dir")"
  if ! remote "$remote_cmd"; then
    die "Remote repo directory missing after mkdir: ${SSH_USER}@${SSH_HOST}:${remote_dir}"
  fi
}

upload_secrets() {
  # MassedCompute: scp/SFTP can fail with "No such file or directory" even after
  # SSH-exec mkdir+test succeeds (pod 60ee57c4). Upload via the same SSH exec
  # channel as `remote()`: stdin → mkdir parent + `cat > secrets.env` + chmod 600
  # in ONE remote shell (mkdir folded into the write — no cross-exec parent create).
  # Secret contents never appear in argv or log lines.
  local tmp remote_dir secrets_path remote_cmd
  remote_dir="$(remote_repo_dir)"
  secrets_path="${remote_dir}/secrets.env"
  tmp="$(mktemp)"
  cp "$SECRETS_FILE" "$tmp"
  if ! grep -q '^PRIME_API_KEY=' "$tmp"; then
    log "Injecting PRIME_API_KEY from local prime CLI config into remote secrets.env"
    printf 'PRIME_API_KEY=%s\n' "$(prime_api_key_from_cli)" >> "$tmp"
  fi
  remote_cmd="$(write_remote_file_from_stdin_cmd "$secrets_path")"
  log "Uploading secrets.env via SSH exec (not scp/sftp) to ${SSH_USER}@${SSH_HOST}:${secrets_path}"
  if ! remote "$remote_cmd" < "$tmp"; then
    rm -f "$tmp"
    die "Failed to upload secrets.env via SSH exec to ${SSH_USER}@${SSH_HOST}:${secrets_path}"
  fi
  rm -f "$tmp"
  if ! remote "$(test_remote_file_cmd "$secrets_path")"; then
    die "Remote secrets.env missing after SSH-exec upload: ${SSH_USER}@${SSH_HOST}:${secrets_path}"
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
POD_PROVIDER=""
# Per-launch SSH ControlMaster socket dir (set after auth). Collision-safe via mktemp.
SSH_CONTROL_DIR=""
SSH_CONTROL_PATH=""

ssh_common_opts() {
  # Echo space-separated shared SSH options for remote(). When ControlPath is
  # set, ControlMaster=auto reuses one TCP session so MassedCompute mkdir/write
  # affinity holds across remote() calls (8b009f19).
  local opts="-o StrictHostKeyChecking=no -o BatchMode=yes"
  if [[ -n "${SSH_CONTROL_PATH:-}" ]]; then
    opts+=" -o ControlMaster=auto -o ControlPersist=600 -o ControlPath=${SSH_CONTROL_PATH}"
  fi
  printf '%s' "$opts"
}

setup_ssh_control_master() {
  # Collision-safe local socket directory for this launch only.
  [[ -n "${SSH_USER:-}" && -n "${SSH_HOST:-}" ]] || die "SSH target unset; cannot start ControlMaster"
  if [[ -z "${SSH_CONTROL_DIR:-}" ]]; then
    SSH_CONTROL_DIR="$(mktemp -d "${TMPDIR:-/tmp}/wcg-ssh-cm.XXXXXX")"
    # %C hashes local/remote identity — unique per target, safe for paths.
    SSH_CONTROL_PATH="${SSH_CONTROL_DIR}/cm-%C"
  fi
  log "Starting SSH ControlMaster (${SSH_USER}@${SSH_HOST}:${SSH_PORT}, socket_dir=${SSH_CONTROL_DIR})"
  # First connection with ControlMaster=auto creates the master socket.
  # shellcheck disable=SC2086
  ssh -T -i "$SSH_KEY" -p "$SSH_PORT" $(ssh_common_opts) \
    -o ConnectTimeout=15 \
    "${SSH_USER}@${SSH_HOST}" true \
    || die "Failed to establish SSH ControlMaster to ${SSH_USER}@${SSH_HOST}:${SSH_PORT}"
}

cleanup_ssh_control_master() {
  if [[ -n "${SSH_CONTROL_PATH:-}" && -n "${SSH_USER:-}" && -n "${SSH_HOST:-}" ]]; then
    # shellcheck disable=SC2086
    ssh -T -i "$SSH_KEY" -p "$SSH_PORT" $(ssh_common_opts) \
      -O exit \
      "${SSH_USER}@${SSH_HOST}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${SSH_CONTROL_DIR:-}" && -d "${SSH_CONTROL_DIR}" ]]; then
    rm -rf "${SSH_CONTROL_DIR}"
  fi
  SSH_CONTROL_DIR=""
  SSH_CONTROL_PATH=""
}

cleanup() {
  local code=$?
  cleanup_ssh_control_master
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
    # crusoecloud pods (observed 2026-07-08 on a100-80gb.1x) default to a
    # non-root "ubuntu" login user, but this script's remote paths (repo
    # sync destination, secrets.env, decon binary, etc.) all assume a root
    # home directory -- skip that provider rather than special-case paths.
    and (r.get("provider") or "").lower() != "crusoecloud"
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
    and (r.get("provider") or "").lower() != "crusoecloud"
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

pick_cloud_id_with_fallback() {
  if [[ "$CURATION_ONLY" -eq 1 ]]; then
    if CLOUD_ID="$(pick_cpu_cloud_id)"; then
      GPU_TYPE="CPU_NODE"
      echo "$CLOUD_ID"
      return 0
    fi
    die "No available CPU_NODE slots with >=8 vCPU and >=32 GB RAM"
  fi
  local types=("$GPU_TYPE")
  if [[ "$GPU_TYPE" == "A100_80GB" ]]; then
    types+=("A100_40GB")
  fi
  local t
  for t in "${types[@]}"; do
    if CLOUD_ID="$(pick_cloud_id "$t")"; then
      GPU_TYPE="$t"
      echo "$CLOUD_ID"
      return 0
    fi
    log "No $t capacity; trying next option"
  done
  return 1
}

pick_cloud_id_with_retries() {
  # Shared inventory has been observed to flicker within single-digit seconds
  # (2026-07-08: an offer reported Available by `availability list` vanished
  # 15-30s later). A single miss on re-pick shouldn't be fatal -- retry a few
  # times with backoff before giving up.
  local attempt picked
  for attempt in 1 2 3 4; do
    if picked="$(pick_cloud_id_with_fallback)"; then
      echo "$picked"
      return 0
    fi
    log "No compute slots on pick attempt $attempt/4; retrying in 10s"
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
        # Some provider images (massedcompute DGX_A100, crusoecloud) log in as a
        # non-root user (ubuntu) whose home is /home/$SSH_USER, not /root. Derive
        # the remote repo root so all remote paths land somewhere the user can write.
        set_remote_root_for_user "$SSH_USER"
        POD_PROVIDER="$(python3 - <<'PY' "$status_json"
import json, sys
print((json.loads(sys.argv[1]).get("provider") or "").lower())
PY
)"
        # Cloud IPs get recycled across pods; a stale known_hosts entry from a
        # prior pod at this same IP makes even StrictHostKeyChecking=no refuse
        # to connect (observed 2026-07-08 after a spot-instance reclaim).
        ssh-keygen -f "$HOME/.ssh/known_hosts" -R "$SSH_HOST" >/dev/null 2>&1 || true
        log "Pod ready: ssh -i $SSH_KEY -p $SSH_PORT ${SSH_USER}@${SSH_HOST} (REMOTE_ROOT=$REMOTE_ROOT provider=${POD_PROVIDER:-unknown})"
        return 0
      fi
    fi
    sleep 15
  done
}

remote() {
  # shellcheck disable=SC2086
  ssh -T -i "$SSH_KEY" -p "$SSH_PORT" $(ssh_common_opts) "${SSH_USER}@${SSH_HOST}" "$@"
}

wait_for_ssh_auth() {
  # `prime pods status` reporting ACTIVE + ssh-ready is not sufficient on some
  # providers (observed 2026-07-08 on crusoecloud's a100-80gb.1x: the API said
  # ready, but publickey auth returned "Permission denied" for ~90s straight
  # before the script gave up) -- authorized_keys can propagate after the
  # instance is otherwise reachable. Probe real SSH auth directly, separately
  # from the tar-sync retry loop below, before trusting the pod is usable.
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
          # Keep REMOTE_ROOT in lockstep with the login user so root@datacrunch
          # (/root/...) and ubuntu@massedcompute (/home/ubuntu/...) both work.
          set_remote_root_for_user "$SSH_USER"
          log "Updated REMOTE_ROOT=$REMOTE_ROOT for SSH_USER=$SSH_USER"
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

# Remote shell: wipe + mkdir + extract stdin tar in ONE exec (MassedCompute
# cross-exec write fix). Paths %q-escaped; no bash -lc.
extract_repo_tar_stdin_cmd() {
  local remote_dir="$1"
  printf 'rm -rf %q && mkdir -p %q && tar -xzf - -C %q\n' \
    "$remote_dir" "$remote_dir" "$remote_dir"
}

# Inverse: stream a remote directory as a tar archive on stdout (artifact pull).
create_tar_stdout_cmd() {
  local remote_dir="$1"
  printf 'tar -czf - -C %q .\n' "$remote_dir"
}

remote_sync_repo() {
  # Stream a local tar of $ROOT through one `remote` exec that mkdir+extracts
  # in the SAME shell. Replaces rsync/scp so MassedCompute cannot split
  # directory creation from file writes across sessions (8b009f19 / 60ee57c4).
  # Excludes mirror the former rsync filters (.git/.venv/outputs/caches/site data).
  local remote_dir remote_cmd attempt
  [[ -n "${REMOTE_ROOT:-}" ]] || die "REMOTE_ROOT is unset; cannot sync repository"
  [[ -n "${SSH_USER:-}" && -n "${SSH_HOST:-}" ]] || die "SSH target unset; cannot sync repository"
  remote_dir="$(remote_repo_dir)"
  remote_cmd="$(extract_repo_tar_stdin_cmd "$remote_dir")"
  log "Syncing repository via tar-over-SSH-exec to ${SSH_USER}@${SSH_HOST}:${remote_dir}"
  for attempt in 1 2 3 4 5; do
    if tar -czf - \
      --exclude='.git' \
      --exclude='.venv' \
      --exclude='*/.venv' \
      --exclude='outputs' \
      --exclude='*/outputs' \
      --exclude='__pycache__' \
      --exclude='*/__pycache__' \
      --exclude='.pytest_cache' \
      --exclude='*/.pytest_cache' \
      --exclude='.mypy_cache' \
      --exclude='*/.mypy_cache' \
      --exclude='docs/site/data' \
      -C "$ROOT" . \
      | remote "$remote_cmd"; then
      return 0
    fi
    log "tar-over-SSH sync attempt $attempt failed; retrying in 20s"
    sleep 20
  done
  die "tar-over-SSH repository sync failed after 5 attempts"
}

# --- MassedCompute single-remote-shell path (2b9a785 evidence) ----------------
# Live MassedCompute: ControlMaster affinity can pass, tar-sync can "succeed",
# then a subsequent exec cannot see /home/ubuntu/webcurator-gym (secrets upload
# → No such file). Fix: one remote bash process extracts a staged archive
# (repo excludes + secrets.env mode 600 + driver), provisions, runs eval, and
# emits the result tar on stdout only; logs go to stderr.

repo_tar_exclude_args() {
  # Shared excludes for staged archives / sync (equivalent to former rsync filters).
  printf '%s\n' \
    --exclude='.git' \
    --exclude='.venv' \
    --exclude='*/.venv' \
    --exclude='outputs' \
    --exclude='*/outputs' \
    --exclude='__pycache__' \
    --exclude='*/__pycache__' \
    --exclude='.pytest_cache' \
    --exclude='*/.pytest_cache' \
    --exclude='.mypy_cache' \
    --exclude='*/.mypy_cache' \
    --exclude='docs/site/data'
}

prepare_secrets_env_file() {
  # Write secrets to $1 with mode 0600. Values never printed/logged by caller.
  local dest="$1"
  local tmp
  tmp="$(mktemp)"
  cp "$SECRETS_FILE" "$tmp"
  if ! grep -q '^PRIME_API_KEY=' "$tmp"; then
    log "Injecting PRIME_API_KEY from local prime CLI config into staged secrets.env"
    printf 'PRIME_API_KEY=%s\n' "$(prime_api_key_from_cli)" >> "$tmp"
  fi
  install -m 600 "$tmp" "$dest"
  rm -f "$tmp"
}

write_massedcompute_single_shell_driver() {
  # Driver lives INSIDE the staged archive. Logs → stderr; success → result tar on stdout.
  local dest="$1"
  local curation_flag=0
  [[ "$CURATION_ONLY" -eq 1 ]] && curation_flag=1
  cat > "$dest" <<EOF
#!/usr/bin/env bash
set -euo pipefail
# MassedCompute single-shell driver (no second SSH exec for artifacts).
log() { printf '[remote %s] %s\n' "\$(date -u +%H:%M:%S)" "\$*" >&2; }
die() { log "FATAL: \$*"; exit 1; }

SELF_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
# Wipe the staging tree (secrets.env + checkout) on every exit — success,
# failure, or local --keep-pod (pod may remain; secrets must not). Result
# artifacts are copied to an external mktemp BUNDLE before stdout export, so
# this never deletes the result tar payload mid-stream.
cleanup_stage() {
  # Preserve the driver's exit status: do not `exit`/`return` nonzero from here.
  if [[ -n "\${SELF_DIR:-}" && -d "\$SELF_DIR" ]]; then
    rm -rf -- "\$SELF_DIR" >/dev/null 2>&1 || true
  fi
}
trap cleanup_stage EXIT

export PATH="\$SELF_DIR/environments/pretrain_data_curator/decon/bin:\$HOME/.local/bin:\$HOME/.cargo/bin:\$PATH"
cd "\$SELF_DIR"

[[ -f secrets.env ]] || die "staged secrets.env missing"
chmod 600 secrets.env

set -a
# shellcheck disable=SC1091
source secrets.env
set +a
: "\${HF_TOKEN:?HF_TOKEN missing in secrets.env}"
: "\${PRIME_API_KEY:?PRIME_API_KEY missing in secrets.env}"
# Never echo secret values.
log "secrets.env sourced (keys present; values not logged)"

CURATION_ONLY=${curation_flag}
MODEL=$(printf '%q' "$MODEL")
EVAL_CONFIG=$(printf '%q' "$EVAL_CONFIG")
RUN_NAME=$(printf '%q' "$RUN_NAME")

if [[ "\$CURATION_ONLY" -eq 1 ]]; then
  log "Provisioning CPU stack in-session"
else
  log "Provisioning GPU stack in-session"
fi

export PATH="\$HOME/.local/bin:\$HOME/.cargo/bin:\$PATH"

if [[ "\$CURATION_ONLY" -eq 0 ]]; then
  log "=== GPU ==="
  nvidia-smi -L >&2
fi

log "=== Docker ==="
if ! command -v docker >/dev/null 2>&1; then
  curl -fsSL https://get.docker.com | sh >&2
fi
systemctl start docker || true

if [[ "\$CURATION_ONLY" -eq 0 ]]; then
  log "=== NVIDIA container toolkit ==="
  if ! command -v nvidia-ctk >/dev/null 2>&1; then
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -sL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \\
      sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \\
      > /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update -qq && apt-get install -y -qq nvidia-container-toolkit
  fi
  nvidia-ctk runtime configure --runtime=docker >&2
  systemctl restart docker
  log "=== GPU docker smoke ==="
  docker pull pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime >&2
  docker run --rm --gpus all pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime nvidia-smi -L >&2
fi

log "=== uv + python env ==="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh >&2
fi
rm -rf .venv
uv venv -p 3.12 >&2
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install prime >&2
cd environments/pretrain_data_curator
uv pip install -e . >&2
uv run python -c "import pretrain_data_curator; print('import OK', __import__('sys').version.split()[0])" >&2

log "=== decon ==="
DECON_BIN="\$SELF_DIR/environments/pretrain_data_curator/decon/bin/decon"
if ! "\$DECON_BIN" --version >/dev/null 2>&1; then
  log "Vendored decon missing or incompatible; building natively"
  bash "\$SELF_DIR/environments/pretrain_data_curator/decon/build_from_source.sh" >&2
fi
"\$DECON_BIN" --version >&2
file "\$DECON_BIN" >&2

log "=== build webcurator-runtime ==="
docker build -f Dockerfile.runtime -t webcurator-runtime:latest . >&2
log "=== runtime preflight ==="
docker run --rm -w /workspace webcurator-runtime:latest bash -lc \\
  'command -v hf && python -c "import huggingface_hub; print(\\"huggingface_hub\\", huggingface_hub.__version__)" && test -x /workspace/decon/bin/decon && /workspace/decon/bin/decon --version' >&2
if [[ "\$CURATION_ONLY" -eq 0 ]]; then
  docker run --rm --gpus all webcurator-runtime:latest nvidia-smi -L >&2
fi
log "=== PROVISION DONE ==="

LOG_FILE="\$SELF_DIR/wcg-eval-\${RUN_NAME}.log"
log "Launching eval (stream → stderr + \$LOG_FILE)"
# Eval progress to stderr AND log file; never to stdout (stdout reserved for result tar).
set +e
uv run eval -m "\${MODEL}" @ \${EVAL_CONFIG} 2>&1 | tee "\$LOG_FILE" >&2
EVAL_RC=\${PIPESTATUS[0]}
set -e
if [[ "\$EVAL_RC" -ne 0 ]]; then
  die "eval failed with exit code \$EVAL_RC (see stderr log; no result tar emitted)"
fi

RESULTS_REL=""
if RESULTS_LINE="\$(grep -Eo 'results: outputs/[^[:space:]]+' "\$LOG_FILE" | tail -1 || true)"; then
  if [[ -n "\$RESULTS_LINE" ]]; then
    RESULTS_REL="\${RESULTS_LINE#results: }"
  fi
fi
if [[ -z "\$RESULTS_REL" ]]; then
  RESULTS_REL="\$(python3 - <<'PY'
import json
from pathlib import Path
root = Path("outputs")
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
)" || die "could not locate results directory after eval"
fi

RESULTS_DIR="\$SELF_DIR/environments/pretrain_data_curator/\${RESULTS_REL}"
[[ -d "\$RESULTS_DIR" ]] || die "results dir missing: \$RESULTS_DIR"
[[ -s "\$RESULTS_DIR/results.jsonl" ]] || die "results.jsonl empty in \$RESULTS_DIR"
# Bundle results + eval log OUTSIDE SELF_DIR so EXIT trap can wipe staging
# (including secrets.env) after the result tar is fully written to stdout.
BUNDLE="\$(mktemp -d "\${TMPDIR:-/tmp}/wcg-mc-results.XXXXXX")"
cp -a "\$RESULTS_DIR/." "\$BUNDLE/"
cp -f "\$LOG_FILE" "\$BUNDLE/eval-stream.log"
log "Emitting result archive on stdout (logs were on stderr)"
# stdout ONLY: the result tar. No log lines after this.
tar -czf - -C "\$BUNDLE" .
RC=\$?
rm -rf -- "\$BUNDLE"
# EXIT trap removes SELF_DIR (secrets) after artifacts are already on stdout.
exit "\$RC"
EOF
  chmod 700 "$dest"
}

stage_massedcompute_workspace() {
  # Build a temp dir: excluded repo tree + secrets.env (0600) + single-shell driver.
  # stdout: ONLY the stage directory path (no log lines — a863bcc: prepare_secrets
  # log-to-stdout polluted stage="$(...)"). Progress goes through log() → stderr.
  local stage
  stage="$(mktemp -d "${TMPDIR:-/tmp}/wcg-mc-stage.XXXXXX")"
  # shellcheck disable=SC2046
  tar -cf - $(repo_tar_exclude_args) -C "$ROOT" . | tar -xf - -C "$stage"
  prepare_secrets_env_file "$stage/secrets.env"
  write_massedcompute_single_shell_driver "$stage/wcg_single_shell_driver.sh"
  # Ensure secrets mode survived copy/install.
  chmod 600 "$stage/secrets.env"
  printf '%s\n' "$stage"
}

massedcompute_single_shell_bootstrap_cmd() {
  # Secret-free remote bootstrap as ONE shell string (%q discipline for paths
  # is N/A here — no user paths; fixed mktemp + relative driver name).
  # stdin = staged tar.gz; stdout = result tar (from driver); stderr = logs.
  printf '%s\n' \
    'set -euo pipefail' \
    'STAGE="$(mktemp -d "${TMPDIR:-/tmp}/wcg-mc.XXXXXX")"' \
    'tar -xzf - -C "$STAGE"' \
    'exec bash "$STAGE/wcg_single_shell_driver.sh"'
}

run_massedcompute_single_shell_eval() {
  # One remote shell: stdin=staged archive, stderr=logs, stdout=result tar on success.
  # On failure: nonzero rc, preserve local stderr log; no follow-up remote exec.
  local stage archive stderr_log bootstrap_cmd rc bundle_tar
  [[ -n "${SSH_USER:-}" && -n "${SSH_HOST:-}" ]] || die "SSH target unset; cannot run MassedCompute single-shell eval"
  mkdir -p "$LOCAL_OUT_DIR"
  stderr_log="$LOCAL_OUT_DIR/remote-single-shell.stderr.log"
  bundle_tar="$LOCAL_OUT_DIR/results-bundle.tar.gz"
  stage="$(stage_massedcompute_workspace)"
  archive="$(mktemp "${TMPDIR:-/tmp}/wcg-mc-archive.XXXXXX.tar.gz")"
  tar -czf "$archive" -C "$stage" .
  rm -rf "$stage"
  stage=""
  bootstrap_cmd="$(massedcompute_single_shell_bootstrap_cmd)"
  # Hygiene: bootstrap must never contain secret material (only in the archive).
  if grep -E 'HF_TOKEN=|PRIME_API_KEY=|sk-' <<<"$bootstrap_cmd" >/dev/null 2>&1; then
    rm -f "$archive"
    die "internal error: bootstrap cmd leaked secret-like tokens"
  fi
  log "MassedCompute single-shell path: one remote bash extracts staged archive, provisions, evals, emits result tar on stdout"
  log "Remote logs → $stderr_log ; result archive ← stdout (success only)"
  set +e
  # One remote bash session: bootstrap is a single %q-safe command string; stdin
  # is the staged archive (secrets only inside the tar — never argv/logs).
  # Prefer bash -c over bash -s so stdin remains the archive (bash -s would
  # consume stdin as the script). OpenSSH argv-flattening: one string only.
  remote "bash -c $(printf '%q' "$bootstrap_cmd")" <"$archive" >"$bundle_tar" 2>"$stderr_log"
  rc=$?
  set -e
  rm -f "$archive"
  archive=""

  if [[ "$rc" -ne 0 ]]; then
    log "MassedCompute single-shell eval FAILED (rc=$rc). Preserving stderr log: $stderr_log"
    rm -f "$bundle_tar"
    # Never fetch diagnostics via a second remote exec (MassedCompute affinity).
    die "MassedCompute single-shell eval failed (rc=$rc); see $stderr_log"
  fi
  [[ -s "$bundle_tar" ]] || die "success rc but empty result archive at $bundle_tar"
  tar -xzf "$bundle_tar" -C "$LOCAL_OUT_DIR"
  rm -f "$bundle_tar"
  [[ -s "$LOCAL_OUT_DIR/results.jsonl" ]] || die "Downloaded results.jsonl is empty"
  write_resolved_config "$LOCAL_OUT_DIR/config.toml"
  if [[ "$CURATION_ONLY" -eq 1 ]]; then
    cat > "$LOCAL_OUT_DIR/README.txt" <<EOF
${MODEL} 400M curation-only eval ($(date -u +%Y-%m-%d))

use_real_trainer=false — heuristic proxy trainer for fast curation iteration.
Full conversation trace is in results.jsonl. Perf metrics are not meaningful.

Config: ${EVAL_CONFIG}
EOF
  fi
  log "MassedCompute single-shell eval OK; artifacts in $LOCAL_OUT_DIR"
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
echo "--- preflight: hf/huggingface_hub + /workspace/decon/bin/decon ---"
# Validate the same absolute path self_score.py probes in the agent container.
docker run --rm -w /workspace webcurator-runtime:latest bash -lc \
  'command -v hf && python -c "import huggingface_hub; print(\"huggingface_hub\", huggingface_hub.__version__)" && test -x /workspace/decon/bin/decon && /workspace/decon/bin/decon --version'

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
echo "--- preflight: hf/huggingface_hub + /workspace/decon/bin/decon ---"
# Validate the same absolute path self_score.py probes in the agent container.
docker run --rm -w /workspace webcurator-runtime:latest bash -lc \
  'command -v hf && python -c "import huggingface_hub; print(\"huggingface_hub\", huggingface_hub.__version__)" && test -x /workspace/decon/bin/decon && /workspace/decon/bin/decon --version'
docker run --rm --gpus all webcurator-runtime:latest nvidia-smi -L

echo "=== PROVISION DONE ==="
REMOTE
}

run_remote_eval() {
  local remote_log="$REMOTE_ROOT/wcg-eval-${RUN_NAME}.log"
  if [[ "$CURATION_ONLY" -eq 1 ]]; then
    log "Launching curation-only eval on CPU pod (heuristic trainer, ~30-60 min)"
  else
    log "Launching eval on pod (this can take ~2h for 400M)"
  fi
  remote bash -s <<REMOTE
set -euo pipefail
export PATH="$REMOTE_ROOT/webcurator-gym/environments/pretrain_data_curator/decon/bin:\$HOME/.local/bin:\$HOME/.cargo/bin:\$PATH"
cd $REMOTE_ROOT/webcurator-gym
set -a
source secrets.env
set +a
: "\${HF_TOKEN:?HF_TOKEN missing in secrets.env}"
: "\${PRIME_API_KEY:?PRIME_API_KEY missing in secrets.env}"
cd environments/pretrain_data_curator
uv run eval -m "${MODEL}" @ ${EVAL_CONFIG} 2>&1 | tee ${remote_log}
REMOTE
}

find_remote_results_dir() {
  local remote_log="$REMOTE_ROOT/wcg-eval-${RUN_NAME}.log"
  remote bash -s <<REMOTE
set -euo pipefail
LOG="${remote_log}"
if [[ -f "\$LOG" ]]; then
  RESULTS_LINE="\$(grep -Eo 'results: outputs/[^[:space:]]+' "\$LOG" | tail -1 || true)"
  if [[ -n "\$RESULTS_LINE" ]]; then
    echo "\${RESULTS_LINE#results: }"
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
  local remote_results remote_cmd
  remote_results="$REMOTE_ROOT/webcurator-gym/environments/pretrain_data_curator/outputs/${rel_dir}"
  log "Downloading results from outputs/$rel_dir via tar-over-SSH-exec"
  mkdir -p "$LOCAL_OUT_DIR"
  remote_cmd="$(create_tar_stdout_cmd "$remote_results")"
  if ! remote "$remote_cmd" | tar -xzf - -C "$LOCAL_OUT_DIR"; then
    die "Failed to download results via tar-over-SSH-exec from ${SSH_USER}@${SSH_HOST}:${remote_results}"
  fi
  [[ -s "$LOCAL_OUT_DIR/results.jsonl" ]] || die "Downloaded results.jsonl is empty"
  write_resolved_config "$LOCAL_OUT_DIR/config.toml"
  local remote_log="$REMOTE_ROOT/wcg-eval-${RUN_NAME}.log"
  # MassedCompute: do not use scp/SFTP for the eval stream log — same provider
  # path-namespace mismatch that broke secrets scp. Pull via SSH exec `cat`.
  download_remote_log_via_exec "$remote_log" "$LOCAL_OUT_DIR/eval-stream.log"
  if [[ "$CURATION_ONLY" -eq 1 ]]; then
    cat > "$LOCAL_OUT_DIR/README.txt" <<EOF
${MODEL} 400M curation-only eval ($(date -u +%Y-%m-%d))

use_real_trainer=false — heuristic proxy trainer for fast curation iteration.
Full conversation trace is in results.jsonl. Perf metrics are not meaningful.

Config: ${EVAL_CONFIG}
EOF
  fi
}

download_remote_log_via_exec() {
  local remote_path="$1"
  local local_path="$2"
  local remote_cmd
  if remote "$(test_remote_file_cmd "$remote_path")"; then
    remote_cmd="$(cat_remote_file_cmd "$remote_path")"
    log "Downloading eval log via SSH exec (not scp/sftp): ${SSH_USER}@${SSH_HOST}:${remote_path}"
    if ! remote "$remote_cmd" > "$local_path"; then
      die "Failed to download eval log via SSH exec from ${SSH_USER}@${SSH_HOST}:${remote_path}"
    fi
    [[ -s "$local_path" ]] || die "Downloaded eval log is empty: $local_path"
  else
    log "WARNING: remote eval log missing at ${SSH_USER}@${SSH_HOST}:${remote_path}; results.jsonl was still downloaded"
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

CLOUD_ID="$(pick_cloud_id_with_retries)" || die "No available compute slots ($GPU_TYPE fallback exhausted)"
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
    [[ -n "$POD_ID" ]] && break
  fi
  # Exclude the cloud_id that just failed so a single consistently-broken
  # offer (observed 2026-07-08: "1A100.40S.22V" repeatedly returned "No valid
  # GPU configuration found" from `pods create` despite `availability list`
  # reporting it Available, and kept getting re-picked as the cheapest
  # remaining option) can't eat the whole retry budget.
  log "prime pods create attempt $attempt failed on cloud_id=$CLOUD_ID; excluding it and retrying in 15s"
  EXCLUDED_CLOUD_IDS="$EXCLUDED_CLOUD_IDS $CLOUD_ID"
  sleep 15
  CLOUD_ID="$(pick_cloud_id_with_retries)" || die "No available compute slots ($GPU_TYPE fallback exhausted)"
  log "Re-selected cloud_id=$CLOUD_ID ($GPU_TYPE)"
done
rm -f "$CREATE_LOG"
[[ -n "$POD_ID" ]] || die "prime pods create failed after 5 attempts"

log "Created pod $POD_ID ($POD_NAME)"
wait_for_pod "$POD_ID"
wait_for_ssh_auth

# MassedCompute (2b9a785): post-sync separate exec cannot see checkout even with
# ControlMaster. Scope the true single-remote-shell path to that provider;
# other providers keep the multi-step sync/provision/eval path.
if [[ "${POD_PROVIDER:-}" == "massedcompute" ]]; then
  log "Provider=massedcompute → single-remote-shell path (staged archive + one bash session)"
  setup_ssh_control_master
  run_massedcompute_single_shell_eval
else
  setup_ssh_control_master
  verify_ssh_connection_affinity
  log "Syncing repository to pod"
  remote_sync_repo
  upload_secrets
  if [[ "$CURATION_ONLY" -eq 1 ]]; then
    remote_provision_cpu
  else
    remote_provision_gpu
  fi
  run_remote_eval
  RESULTS_REL="$(find_remote_results_dir)" || die "Could not locate remote results directory"
  download_results "$RESULTS_REL"
fi

log "Eval summary:"
summarize_results

if [[ "$SKIP_SITE" -eq 0 ]]; then
  rebuild_site
  log "Site: file://$ENV_DIR/docs/site/index.html"
fi

log "Done. Artifacts: $LOCAL_OUT_DIR"
