#!/usr/bin/env bash
# GenET stage1 DDP training entry (Hyperbolic 4×8).
#
# Default sequence inside launch_cluster_ssh.sh:
#   host-check → image pull → release verify → Gloo/NCCL preflight →
#   32-GPU dry-run → real training
#
# Usage (from repo root, preferably inside tmux):
#   bash scripts/entry_s1_train.sh
#   bash scripts/entry_s1_train.sh --dry-run-only
#   bash scripts/entry_s1_train.sh --preflight-only
#   bash scripts/entry_s1_train.sh --max-steps 20   # smoke: gates + dry-run + N steps
#   MODE=dry-run-only bash scripts/entry_s1_train.sh
#   MAX_STEPS=20 bash scripts/entry_s1_train.sh
#
# Monitor:
#   tmux attach -t genet-train
#   tail -F /mnt/nvme/genet/logs/<run-id>/train-rank-0.log
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "${ROOT}"

HOSTS="${HOSTS:-/secure/path/genet-hosts.txt}"
ENV_FILE="${ENV_FILE:-/secure/path/s1.env}"
LOCK_ENV="${LOCK_ENV:-/secure/path/lock.host.env}"
SSH_IDENTITY="${SSH_IDENTITY:-/root/.ssh/id_cluster}"
SNAPSHOT="${SNAPSHOT:-/mnt/nvme/genet/checkpoints/Cosmos3-Edge}"
CONFIG="${CONFIG:-configs/experiments/stage1_control_32gpu_ddp.yaml}"
COMMITTED_ROOT="${GENET_COMMITTED_ROOT:-/mnt/nvme/genet/committed}"
COMMIT_FINAL_DCP="${COMMIT_FINAL_DCP:-1}"

# shellcheck disable=SC1090
source "${ENV_FILE}"
REV="${GENET_CODE_REVISION:?GENET_CODE_REVISION missing in ${ENV_FILE}}"
RELEASE_DIR="${GENET_RELEASE_DIR:-/mnt/nvme/genet/release/s1-${REV}}"
if [[ -f /secure/path/image-ref.txt ]]; then
  # shellcheck disable=SC1091
  source /secure/path/image-ref.txt
fi
IMAGE_REF="${GENET_IMAGE_REF:?Set GENET_IMAGE_REF or /secure/path/image-ref.txt}"

RUN_ID_FROM_ENV=0
if [[ -n "${RUN_ID:-}" ]]; then
  RUN_ID_FROM_ENV=1
fi
RUN_ID="${RUN_ID:-s1-ddp-$(date +%Y%m%d-%H%M%S)}"
SSH=(ssh -i "${SSH_IDENTITY}" -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=15)

LOG() { echo "[$(date '+%F %T')] $*"; }

MODE="${MODE:-full}"
MAX_STEPS="${MAX_STEPS:-}"
RUNS_TRAINING=1
LAUNCH_EXTRA=()
TRAIN_EXTRA=()
while (( $# > 0 )); do
  case "$1" in
    --dry-run-only|--preflight-only|--verify-release-only)
      LAUNCH_EXTRA+=("$1")
      RUNS_TRAINING=0
      shift
      ;;
    --skip-dry-run|--skip-nccl-preflight|--skip-preflight|--skip-release-verify|--skip-host-check)
      LAUNCH_EXTRA+=("$1")
      shift
      ;;
    --max-steps)
      MAX_STEPS="$2"
      shift 2
      ;;
    --mode)
      MODE="$2"
      shift 2
      ;;
    full|dry-run-only|preflight-only|verify-release-only)
      MODE="$1"
      shift
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -n "${MAX_STEPS}" ]]; then
  [[ "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]] || {
    echo "MAX_STEPS/--max-steps must be a positive integer, got: ${MAX_STEPS}" >&2
    exit 2
  }
  TRAIN_EXTRA+=(--max-steps "${MAX_STEPS}")
  if (( RUN_ID_FROM_ENV == 0 )); then
    RUN_ID="s1-smoke-${MAX_STEPS}st-$(date +%Y%m%d-%H%M%S)"
  fi
fi

LOG_DIR="${LOG_DIR:-/mnt/nvme/genet/logs/${RUN_ID}}"

case "${MODE}" in
  full) ;;
  dry-run-only) LAUNCH_EXTRA+=(--dry-run-only); RUNS_TRAINING=0 ;;
  preflight-only) LAUNCH_EXTRA+=(--preflight-only); RUNS_TRAINING=0 ;;
  verify-release-only) LAUNCH_EXTRA+=(--verify-release-only); RUNS_TRAINING=0 ;;
  *) echo "Unknown MODE=${MODE}" >&2; exit 2 ;;
esac
case "${COMMIT_FINAL_DCP}" in
  1|true|TRUE|yes|YES|on|ON) COMMIT_FINAL_DCP=1 ;;
  0|false|FALSE|no|NO|off|OFF) COMMIT_FINAL_DCP=0 ;;
  *) echo "COMMIT_FINAL_DCP must be boolean, got: ${COMMIT_FINAL_DCP}" >&2; exit 2 ;;
esac
OUTPUT_ROOT="/mnt/nvme/genet/outputs/${RUN_ID}"

# A. Ensure release lock exists (hashes ~370G the first time).
if [[ ! -f "${RELEASE_DIR}/cluster-lock.json" ]]; then
  LOG "creating release lock at ${RELEASE_DIR}"
  mkdir -p "${RELEASE_DIR}"
  bash scripts/run_hyperbolic_container.sh "${LOCK_ENV}" "${IMAGE_REF}" \
    bash -lc "
      set -euo pipefail
      genet-cluster-lock create \
        --output '${RELEASE_DIR}/cluster-lock.json' \
        --artifact processed_data=/mnt/nvme/mds-cache/robotwin_v1/genet/processed/train \
        --artifact wan_vae=/mnt/nvme/genet/artifacts/wan22_vae/Wan2.2_VAE.pth \
        --artifact artifact_receipt=/mnt/nvme/genet/artifacts/ARTIFACTS.json \
        --artifact training_checkpoint='${SNAPSHOT}' \
        --artifact hf_cache=/mnt/nvme/genet/hf-cache
    "
  LOG "lock created"
else
  LOG "reusing lock ${RELEASE_DIR}/cluster-lock.json"
fi

# B. Confirm processed data on workers.
for ip in 10.0.2.4 10.0.2.3 10.0.2.1; do
  n="$("${SSH[@]}" "root@${ip}" 'wc -l < /mnt/nvme/mds-cache/robotwin_v1/genet/processed/train/manifest.jsonl' 2>/dev/null || echo 0)"
  [[ "${n}" == "34822" ]] || { LOG "FATAL: ${ip} manifest has ${n} lines (want 34822)"; exit 1; }
done
LOG "processed data OK on all workers"

# C. Distribute lock + pin image ref.
for ip in 10.0.2.4 10.0.2.3 10.0.2.1; do
  "${SSH[@]}" "root@${ip}" "mkdir -p '${RELEASE_DIR}'"
  rsync -a -e "${SSH[*]}" \
    "${RELEASE_DIR}/cluster-lock.json" "root@${ip}:${RELEASE_DIR}/cluster-lock.json"
done
echo "GENET_IMAGE_REF=${IMAGE_REF}" > /secure/path/image-ref.txt
LOG "lock distributed; image=${IMAGE_REF}"

# D. Launch.
LOG "launching MODE=${MODE} run-id=${RUN_ID}${MAX_STEPS:+ max_steps=${MAX_STEPS}}"
LOG "logs: ${LOG_DIR}"
bash scripts/launch_cluster_ssh.sh \
  --hosts "${HOSTS}" \
  --env "${ENV_FILE}" \
  --image "${IMAGE_REF}" \
  --config "${CONFIG}" \
  --run-id "${RUN_ID}" \
  --log-dir "${LOG_DIR}" \
  --ssh-option "IdentityFile=${SSH_IDENTITY}" \
  --ssh-option "IdentitiesOnly=yes" \
  ${LAUNCH_EXTRA[@]+"${LAUNCH_EXTRA[@]}"} \
  -- \
  --manifest /mnt/nvme/mds-cache/robotwin_v1/genet/processed/train/manifest.jsonl \
  --warm-start "${SNAPSHOT}" \
  --output-dir "${OUTPUT_ROOT}" \
  ${TRAIN_EXTRA[@]+"${TRAIN_EXTRA[@]}"}

if (( RUNS_TRAINING == 1 && COMMIT_FINAL_DCP == 1 )); then
  LOG "consolidating final checkpoint under ${COMMITTED_ROOT}/${RUN_ID}"
  SSH_IDENTITY="${SSH_IDENTITY}" \
    bash scripts/commit_cluster_dcp_ssh.sh \
      "${HOSTS}" \
      "${IMAGE_REF}" \
      "${OUTPUT_ROOT}" \
      "${COMMITTED_ROOT}"
fi
