#!/usr/bin/env bash
# Provision a Prime A100 pod, run a small pretrain-data-curator smoke eval
# end-to-end (real GPU trainer, current speedrun_muon proxy-student recipe),
# pull artifacts back, validate results, optionally rebuild the docs site, and
# terminate the pod.
#
# Fast-turnaround sibling of run_400m_eval_a100.sh: same pod / provisioning /
# rsync / decon-build machinery. Default profile is 25M tokens; pass
# --profile 10M (or --config configs/eval/10M-60turn-codex-smoke.toml) for the
# shipped 10M smoke. Website rebuild runs only after the valid-result gate.
#
# Usage:
#   ./scripts/run_25m_smoke_a100.sh --model deepseek/deepseek-v4-pro
#   ./scripts/run_25m_smoke_a100.sh --model deepseek/deepseek-v4-pro --profile 10M --gpu-type A100_80GB
#   ./scripts/run_25m_smoke_a100.sh --model z-ai/glm-5.2 --curation-only
#
# Curation-only mode uses use_real_trainer=false (heuristic scoring, full
# trace, no GPU training) on a CPU pod. Artifacts land in
# outputs/debug/<model>-<budget>-smoke-curation-<date>/.
#   - prime CLI logged in (prime whoami)
#   - SSH key at ~/.prime/config.json ssh_key_path (default ~/.ssh/id_rsa)
#   - secrets.env at repo root with HF_TOKEN (PRIME_API_KEY optional if prime login)
#
# Eval uses a swappable smoke config with only the inference model changed:
#   configs/eval/25M-60turn-codex-smoke.toml  (default)
#   configs/eval/10M-60turn-codex-smoke.toml  (--profile 10M)
#
# Options:
#   --model SLUG        Prime inference model id, e.g. deepseek/deepseek-v4-pro (required)
#   --profile NAME      25M (default) or 10M; selects shipped smoke config + run suffix
#   --config PATH       Explicit eval config under the env dir (or absolute); overrides --profile
#   --gpu-type TYPE     A100_80GB (default) or A100_40GB; ignored with --curation-only.
#                       10M profile never falls back to a non-80GB GPU.
#   --pod-name NAME     Pod name (default: wcg-<budget>-smoke-<model> or wcg-<budget>-curation-<model>)
#   --disk-gb N         Pod disk size in GB (default: 80)
#   --curation-only     Heuristic trainer only (outputs/debug/, full trace)
#   --skip-site         Skip docs/site rebuild after a valid download
#   --validate-only DIR Validate an existing local results dir and exit (no pod)
#   --keep-pod          Do not terminate the pod on exit (debugging)
#   --dry-run           Print planned actions without creating a pod
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="$ROOT/environments/pretrain_data_curator"
SMOKE_PROFILE="25M"
BASE_EVAL_CONFIG="configs/eval/25M-60turn-codex-smoke.toml"
BASE_EVAL_CONFIG_PATH="$ENV_DIR/$BASE_EVAL_CONFIG"
RUN_SUFFIX="25M-60turn-codex-smoke"
EXPECTED_TOKEN_BUDGET=25000000
ALLOW_GPU_FALLBACK=1
CONFIG_OVERRIDE=""
SECRETS_FILE="$ROOT/secrets.env"
GPU_TYPE="A100_80GB"
POD_IMAGE="ubuntu_22_cuda_12"
DISK_GB=80
KEEP_POD=0
DRY_RUN=0
CURATION_ONLY=0
SKIP_SITE=0
VALIDATE_ONLY=""
MODEL=""
MODEL_SLUG=""
POD_NAME=""
EVAL_CONFIG=""
EVAL_CONFIG_PATH=""
RUN_NAME=""
BUDGET_LABEL="25M"

log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
die() { log "FATAL: $*"; exit 1; }

usage() {
  sed -n '2,40p' "$0"
  echo ""
  echo "Examples:"
  echo "  --model deepseek/deepseek-v4-pro"
  echo "  --model deepseek/deepseek-v4-pro --profile 10M --gpu-type A100_80GB"
  echo "  --model z-ai/glm-5.2 --curation-only"
  exit 1
}

apply_smoke_profile() {
  local profile="$1"
  case "$profile" in
    25M|25m)
      SMOKE_PROFILE="25M"
      BASE_EVAL_CONFIG="configs/eval/25M-60turn-codex-smoke.toml"
      RUN_SUFFIX="25M-60turn-codex-smoke"
      EXPECTED_TOKEN_BUDGET=25000000
      ALLOW_GPU_FALLBACK=1
      BUDGET_LABEL="25M"
      ;;
    10M|10m)
      SMOKE_PROFILE="10M"
      BASE_EVAL_CONFIG="configs/eval/10M-60turn-codex-smoke.toml"
      RUN_SUFFIX="10M-60turn-codex-smoke"
      EXPECTED_TOKEN_BUDGET=10000000
      ALLOW_GPU_FALLBACK=0
      BUDGET_LABEL="10M"
      ;;
    *)
      die "Unknown --profile '$profile' (expected 10M or 25M)"
      ;;
  esac
}

apply_smoke_config() {
  local raw="$1"
  local path
  if [[ "$raw" = /* ]]; then
    path="$raw"
  else
    path="$ENV_DIR/$raw"
  fi
  [[ -f "$path" ]] || die "Missing eval config: $path"
  BASE_EVAL_CONFIG_PATH="$path"
  if [[ "$path" == "$ENV_DIR/"* ]]; then
    BASE_EVAL_CONFIG="${path#"$ENV_DIR"/}"
  else
    BASE_EVAL_CONFIG="$path"
  fi
  RUN_SUFFIX="$(basename "$path" .toml)"
  EXPECTED_TOKEN_BUDGET="$(python3 - "$path" <<'PY'
import pathlib, re, sys
text = pathlib.Path(sys.argv[1]).read_text()
m = re.search(r"(?m)^token_budget\s*=\s*([0-9_]+)\s*$", text)
if not m:
    raise SystemExit("config missing top-level token_budget")
print(int(m.group(1).replace("_", "")))
PY
)"
  if [[ "$EXPECTED_TOKEN_BUDGET" -eq 10000000 ]]; then
    SMOKE_PROFILE="10M"
    BUDGET_LABEL="10M"
    ALLOW_GPU_FALLBACK=0
  elif [[ "$EXPECTED_TOKEN_BUDGET" -eq 25000000 ]]; then
    SMOKE_PROFILE="25M"
    BUDGET_LABEL="25M"
    ALLOW_GPU_FALLBACK=1
  else
    SMOKE_PROFILE="custom"
    BUDGET_LABEL="${EXPECTED_TOKEN_BUDGET}"
    ALLOW_GPU_FALLBACK=0
  fi
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
  [[ "$model" == */* ]] || die "Model must be a Prime inference slug like deepseek/deepseek-v4-pro (got: $model)"
  if [[ -n "$CONFIG_OVERRIDE" ]]; then
    apply_smoke_config "$CONFIG_OVERRIDE"
  else
    apply_smoke_profile "$SMOKE_PROFILE"
    BASE_EVAL_CONFIG_PATH="$ENV_DIR/$BASE_EVAL_CONFIG"
  fi
  # Curation-only keeps the selected budget/profile config; only the trainer
  # flag is flipped remotely via the same TOML (heuristic path).
  [[ -f "$BASE_EVAL_CONFIG_PATH" ]] || die "Missing base config: $BASE_EVAL_CONFIG_PATH"
  MODEL="$model"
  MODEL_SLUG="$(slugify_model "$model")"
  EVAL_CONFIG="$BASE_EVAL_CONFIG"
  EVAL_CONFIG_PATH="$BASE_EVAL_CONFIG_PATH"
  RUN_NAME="${MODEL_SLUG}-${RUN_SUFFIX}"
  local pod_budget
  pod_budget="$(printf '%s' "$BUDGET_LABEL" | tr '[:upper:]' '[:lower:]')"
  if [[ "$CURATION_ONLY" -eq 1 ]]; then
    LOCAL_OUT_DIR="$ENV_DIR/outputs/debug/${MODEL_SLUG}-${BUDGET_LABEL}-smoke-curation-$(date -u +%Y%m%d)"
    POD_NAME="${POD_NAME:-wcg-${pod_budget}-curation-${MODEL_SLUG}}"
  else
    LOCAL_OUT_DIR="$ENV_DIR/outputs/evals/$RUN_NAME"
    POD_NAME="${POD_NAME:-wcg-${pod_budget}-smoke-${MODEL_SLUG}}"
  fi
}

write_resolved_config() {
  local dest="$1"
  python3 - "$MODEL" "$BASE_EVAL_CONFIG_PATH" "$dest" <<'PY'
import json, pathlib, re, sys
model, src, dest = sys.argv[1:4]
text = pathlib.Path(src).read_text()
# json.dumps produces a TOML-safe basic string literal for arbitrary model ids
# (quotes, backslashes, newlines, control chars). Use a lambda replacement so
# re.sub does not reinterpret escape sequences like \n inside the literal.
model_lit = json.dumps(model, ensure_ascii=False)
if not re.search(r"^model\s*=", text, flags=re.M):
    raise SystemExit("base config missing model = ...")
text = re.sub(
    r"^model\s*=.*$",
    lambda _m: f"model = {model_lit}",
    text,
    count=1,
    flags=re.M,
)
pathlib.Path(dest).write_text(text)
PY
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="${2:-}"; shift 2 ;;
    --profile) SMOKE_PROFILE="${2:-}"; shift 2 ;;
    --config) CONFIG_OVERRIDE="${2:-}"; shift 2 ;;
    --gpu-type) GPU_TYPE="${2:-}"; shift 2 ;;
    --pod-name) POD_NAME="${2:-}"; shift 2 ;;
    --disk-gb) DISK_GB="${2:-}"; shift 2 ;;
    --keep-pod) KEEP_POD=1; shift ;;
    --curation-only) CURATION_ONLY=1; shift ;;
    --skip-site) SKIP_SITE=1; shift ;;
    --validate-only) VALIDATE_ONLY="${2:-}"; shift 2 ;;
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

if [[ -n "$VALIDATE_ONLY" ]]; then
  LOCAL_OUT_DIR="$VALIDATE_ONLY"
  [[ -d "$LOCAL_OUT_DIR" ]] || die "validate-only dir missing: $LOCAL_OUT_DIR"
  log "Validating existing results in $LOCAL_OUT_DIR (profile=$SMOKE_PROFILE suffix=$RUN_SUFFIX budget=$EXPECTED_TOKEN_BUDGET)"
  (
    cd "$ENV_DIR"
    python3 -m pretrain_data_curator.smoke_result_gate \
      "$LOCAL_OUT_DIR" "$EXPECTED_TOKEN_BUDGET" "$RUN_SUFFIX"
  ) || die "Downloaded results failed validation"
  log "Validation OK"
  exit 0
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  log "DRY RUN"
  log "model=$MODEL"
  log "profile=$SMOKE_PROFILE"
  log "base_config=$EVAL_CONFIG"
  log "run_suffix=$RUN_SUFFIX"
  log "expected_token_budget=$EXPECTED_TOKEN_BUDGET"
  log "allow_gpu_fallback=$ALLOW_GPU_FALLBACK"
  log "run_name=$RUN_NAME"
  log "compute_type=$GPU_TYPE pod_image=$POD_IMAGE disk_gb=$DISK_GB pod_name=$POD_NAME curation_only=$CURATION_ONLY skip_site=$SKIP_SITE"
  log "local_out=$LOCAL_OUT_DIR"
  exit 0
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
    # Do not `die` here: this function is often called inside $(...), where
    # die only kills the subshell and the outer retry loop would sleep/retry.
    return 1
  fi
  local types=("$GPU_TYPE")
  # 10M (and other non-25M) smokes stay on the requested GPU class — no
  # silent downgrade to A100_40GB when the caller asked for A100_80GB.
  if [[ "$ALLOW_GPU_FALLBACK" -eq 1 && "$GPU_TYPE" == "A100_80GB" ]]; then
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
  if [[ "$CURATION_ONLY" -eq 1 ]]; then
    # CPU_NODE miss is definitive for curation-only; fail immediately.
    if picked="$(pick_cloud_id_with_fallback)"; then
      echo "$picked"
      return 0
    fi
    return 1
  fi
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

select_cloud_id_or_die() {
  # Sets CLOUD_ID in the caller's scope. Must be invoked directly (not via $())
  # so `die` terminates the launcher instead of a command-substitution subshell.
  local picked
  if picked="$(pick_cloud_id_with_retries)"; then
    CLOUD_ID="$picked"
    return 0
  fi
  if [[ "$CURATION_ONLY" -eq 1 ]]; then
    die "No available CPU_NODE slots with >=8 vCPU and >=32 GB RAM"
  fi
  die "No available compute slots ($GPU_TYPE fallback exhausted)"
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
  # providers: observed 2026-07-08 on crusoecloud's a100-80gb.1x, the parsed
  # SSH_USER (root, the previous hardcoded default whenever the API returns a
  # bare `ip` field) was flat-out rejected -- the pod's actual default login
  # user was "ubuntu". parse_ssh_target now prefers the API's stated user, but
  # as a pragmatic safety net also try known-good alternates here so a future
  # provider quirk degrades to a slower connection instead of a dead run.
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
  local attempt
  local ssh_cmd="ssh -T -i $SSH_KEY -o StrictHostKeyChecking=no -p $SSH_PORT"
  for attempt in 1 2 3 4 5; do
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
      "$ROOT/" "${SSH_USER}@${SSH_HOST}:/root/webcurator-gym/"; then
      return 0
    fi
    log "rsync attempt $attempt failed; retrying in 20s"
    sleep 20
  done
  die "rsync failed after 5 attempts"
}

remote_provision_cpu() {
  log "Provisioning CPU pod software stack (uv, py3.12, decon)"
  remote bash -s <<'REMOTE'
set -euo pipefail
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
exec > /root/wcg-provision.log 2>&1

echo "=== uv + python env ==="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
cd /root/webcurator-gym
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
DECON_BIN=/root/webcurator-gym/environments/pretrain_data_curator/decon/bin/decon
if ! "$DECON_BIN" --version >/dev/null 2>&1; then
  echo "Vendored decon missing or incompatible (often glibc); building natively"
  bash /root/webcurator-gym/environments/pretrain_data_curator/decon/build_from_source.sh
fi
"$DECON_BIN" --version
file "$DECON_BIN"

echo "=== PROVISION DONE ==="
REMOTE
}

remote_provision_gpu() {
  log "Provisioning pod software stack (docker, uv, py3.12, decon)"
  remote bash -s <<'REMOTE'
set -euo pipefail
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
exec > /root/wcg-provision.log 2>&1

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

echo "=== GPU docker smoke + trainer image pull ==="
docker run --rm --gpus all pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime nvidia-smi -L
docker pull pytorch/pytorch:2.7.0-cuda12.6-cudnn9-runtime

echo "=== uv + python env ==="
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
cd /root/webcurator-gym
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
DECON_BIN=/root/webcurator-gym/environments/pretrain_data_curator/decon/bin/decon
if ! "$DECON_BIN" --version >/dev/null 2>&1; then
  echo "Vendored decon missing or incompatible (often glibc); building natively"
  bash /root/webcurator-gym/environments/pretrain_data_curator/decon/build_from_source.sh
fi
"$DECON_BIN" --version
file "$DECON_BIN"

echo "=== build custom harness-runtime image (hf + decon for the agent container) ==="
# The codex AGENT runs inside the harness-runtime container (DockerConfig,
# workdir=/workspace), not on this host. The bare pytorch image ships neither
# the decon binary nor huggingface-hub, so the agent's self_score.py silently
# skipped leakage and had no hf. Dockerfile.runtime bakes both in; build it now
# (decon is already compiled above, so COPY decon/ picks up a runnable binary)
# and point configs' docker_image at "webcurator-runtime:latest".
cd /root/webcurator-gym/environments/pretrain_data_curator
docker build -f Dockerfile.runtime -t webcurator-runtime:latest .
echo "--- verify decon + hf inside the image ---"
docker run --rm -w /workspace webcurator-runtime:latest bash -lc \
  'decon/bin/decon --version && python -c "import huggingface_hub, tiktoken; print(\"huggingface_hub\", huggingface_hub.__version__)" && command -v hf'

echo "=== PROVISION DONE ==="
REMOTE
}

run_remote_eval() {
  local remote_log="/root/wcg-eval-${RUN_NAME}.log"
  if [[ "$CURATION_ONLY" -eq 1 ]]; then
    log "Launching curation-only smoke eval on CPU pod (heuristic trainer, ~10-20 min)"
  else
    log "Launching ${BUDGET_LABEL} smoke eval on pod (real GPU trainer, expect well under 1h)"
  fi
  remote bash -s <<REMOTE
set -euo pipefail
export PATH="/root/webcurator-gym/environments/pretrain_data_curator/decon/bin:\$HOME/.local/bin:\$HOME/.cargo/bin:\$PATH"
cd /root/webcurator-gym
set -a
source secrets.env
set +a
: "\${HF_TOKEN:?HF_TOKEN missing in secrets.env}"
: "\${PRIME_API_KEY:?PRIME_API_KEY missing in secrets.env}"
cd environments/pretrain_data_curator
uv run eval -m "${MODEL}" @ "${EVAL_CONFIG}" 2>&1 | tee "${remote_log}"
REMOTE
}

find_remote_results_dir() {
  local remote_log="/root/wcg-eval-${RUN_NAME}.log"
  remote bash -s <<REMOTE
set -euo pipefail
LOG="${remote_log}"
if [[ -f "\$LOG" ]]; then
  RESULTS_LINE="\$(grep -Eo 'results: outputs/[^[:space:]]+' "\$LOG" | tail -1 || true)"
  if [[ -n "\$RESULTS_LINE" ]]; then
    REL="\${RESULTS_LINE#results: }"
    # download_results prepends '.../outputs/', so emit a path relative to
    # outputs/ (matching the python fallback below). Leaving the leading
    # 'outputs/' here produced '.../outputs/outputs/...' and a failed rsync
    # that lost results when the pod auto-terminated.
    echo "\${REL#outputs/}"
    exit 0
  fi
fi
# Fallback: newest non-empty results.jsonl under outputs/.
python3 - <<'PY'
import json
from pathlib import Path
root = Path("/root/webcurator-gym/environments/pretrain_data_curator/outputs")
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
    "${SSH_USER}@${SSH_HOST}:/root/webcurator-gym/environments/pretrain_data_curator/outputs/${rel_dir}/" \
    "$LOCAL_OUT_DIR/"
  [[ -s "$LOCAL_OUT_DIR/results.jsonl" ]] || die "Downloaded results.jsonl is empty"
  write_resolved_config "$LOCAL_OUT_DIR/config.toml"
  local remote_log="/root/wcg-eval-${RUN_NAME}.log"
  scp -i "$SSH_KEY" -o StrictHostKeyChecking=no -P "$SSH_PORT" \
    "${SSH_USER}@${SSH_HOST}:${remote_log}" "$LOCAL_OUT_DIR/eval-stream.log" >/dev/null 2>&1 || true
  cat > "$LOCAL_OUT_DIR/README.txt" <<EOF
${MODEL} ${BUDGET_LABEL}-token A100 smoke eval ($(date -u +%Y-%m-%d))

Fast end-to-end pipeline check on real A100 hardware: agent curation
(codex harness, ${RUN_SUFFIX}) + $( [[ "$CURATION_ONLY" -eq 1 ]] && echo "heuristic proxy trainer (use_real_trainer=false)" || echo "real GPU trainer (use_real_trainer=true, speedrun_muon recipe)" )
+ decon leakage scoring, at a ${BUDGET_LABEL}-token budget instead of the full 400M.

Config: ${EVAL_CONFIG}
EOF
}

validate_downloaded_results() {
  # Gate success + site rebuild via shared Python helper (importable in tests).
  (
    cd "$ENV_DIR"
    python3 -m pretrain_data_curator.smoke_result_gate \
      "$LOCAL_OUT_DIR" "$EXPECTED_TOKEN_BUDGET" "$RUN_SUFFIX"
  )
}

rebuild_site() {
  log "Rebuilding docs bench site"
  python3 "$ENV_DIR/docs/build_site.py"
  python3 - <<'PY' "$ENV_DIR/docs/site/data/manifest.json"
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.exists():
    print("site manifest missing after rebuild")
    raise SystemExit(1)
m = json.loads(path.read_text())
print(f"site runs={m.get('run_count', len(m.get('runs', [])))}")
for r in m.get("runs", [])[:8]:
    print(f"  {r.get('model', '?'):28} reward={r.get('reward')}")
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

select_cloud_id_or_die
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
  select_cloud_id_or_die
  log "Re-selected cloud_id=$CLOUD_ID ($GPU_TYPE)"
done
rm -f "$CREATE_LOG"
[[ -n "$POD_ID" ]] || die "prime pods create failed after 5 attempts"

log "Created pod $POD_ID ($POD_NAME)"
wait_for_pod "$POD_ID"
wait_for_ssh_auth

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

# Always retain downloaded artifacts + logs. Only treat the run as successful
# (and rebuild the site) when the valid-result gate passes.
if ! validate_downloaded_results; then
  log "Valid-result gate FAILED; retaining artifacts at $LOCAL_OUT_DIR (site rebuild skipped)"
  die "Downloaded results failed validation"
fi

log "Eval summary:"
summarize_results

if [[ "$SKIP_SITE" -eq 0 ]]; then
  rebuild_site
  log "Site: file://$ENV_DIR/docs/site/index.html"
else
  log "Skipping docs/site rebuild (--skip-site)"
fi

log "Done. Artifacts: $LOCAL_OUT_DIR"
