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
#   MODE=dry-run-only bash scripts/entry_s1_train.sh
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

# shellcheck disable=SC1090
source "${ENV_FILE}"
REV="${GENET_CODE_REVISION:?GENET_CODE_REVISION missing in ${ENV_FILE}}"
RELEASE_DIR="${GENET_RELEASE_DIR:-/mnt/nvme/genet/release/s1-${REV}}"
if [[ -f /secure/path/image-ref.txt ]]; then
  # shellcheck disable=SC1091
  source /secure/path/image-ref.txt
fi
IMAGE_REF="${GENET_IMAGE_REF:?Set GENET_IMAGE_REF or /secure/path/image-ref.txt}"

RUN_ID="${RUN_ID:-s1-ddp-$(date +%Y%m%d-%H%M%S)}"
LOG_DIR="${LOG_DIR:-/mnt/nvme/genet/logs/${RUN_ID}}"
SSH=(ssh -i "${SSH_IDENTITY}" -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=15)

LOG() { echo "[$(date '+%F %T')] $*"; }

MODE="${MODE:-full}"
LAUNCH_EXTRA=()
while (( $# > 0 )); do
  case "$1" in
    --dry-run-only|--preflight-only|--verify-release-only|--skip-dry-run|--skip-nccl-preflight|--skip-preflight|--skip-release-verify|--skip-host-check)
      LAUNCH_EXTRA+=("$1")
      shift
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

case "${MODE}" in
  full) ;;
  dry-run-only) LAUNCH_EXTRA+=(--dry-run-only) ;;
  preflight-only) LAUNCH_EXTRA+=(--preflight-only) ;;
  verify-release-only) LAUNCH_EXTRA+=(--verify-release-only) ;;
  *) echo "Unknown MODE=${MODE}" >&2; exit 2 ;;
esac

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
LOG "launching MODE=${MODE} run-id=${RUN_ID}"
LOG "logs: ${LOG_DIR}"
exec bash scripts/launch_cluster_ssh.sh \
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
  --output-dir "/mnt/nvme/genet/outputs/${RUN_ID}"
