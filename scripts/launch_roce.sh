#!/usr/bin/env bash
set -euo pipefail

: "${NNODES:?Set NNODES (4 for the full cluster)}"
: "${NPROC_PER_NODE:?Set NPROC_PER_NODE (8 for the full cluster)}"
: "${NODE_RANK:?Set the unique NODE_RANK on each node (0..NNODES-1)}"
: "${MASTER_ADDR:?Set MASTER_ADDR to the reachable bootstrap address of node 0}"
: "${MASTER_PORT:?Set MASTER_PORT}"

for integer_name in NNODES NPROC_PER_NODE NODE_RANK MASTER_PORT; do
  integer_value="${!integer_name}"
  if [[ ! "${integer_value}" =~ ^[0-9]+$ ]]; then
    echo "${integer_name} must be a non-negative integer, got: ${integer_value}" >&2
    exit 2
  fi
done
if (( NNODES < 1 || NPROC_PER_NODE < 1 )); then
  echo "NNODES and NPROC_PER_NODE must be positive" >&2
  exit 2
fi
if (( NODE_RANK >= NNODES )); then
  echo "NODE_RANK must be in [0, NNODES), got ${NODE_RANK} for NNODES=${NNODES}" >&2
  exit 2
fi
if (( MASTER_PORT < 1 || MASTER_PORT > 65535 )); then
  echo "MASTER_PORT must be in [1, 65535], got ${MASTER_PORT}" >&2
  exit 2
fi

case "${GENET_STRICT_ENV:-0}" in
  1|true|TRUE|yes|YES|on|ON) strict_environment=1 ;;
  0|false|FALSE|no|NO|off|OFF) strict_environment=0 ;;
  *)
    echo "GENET_STRICT_ENV must be a boolean value" >&2
    exit 2
    ;;
esac

if (( strict_environment == 1 )); then
  : "${GENET_IMAGE_DIGEST:?Strict mode requires an immutable image/SIF SHA256}"
  : "${GENET_CODE_REVISION:?Strict mode requires the GenET git revision}"
  : "${GENET_BUILD_REVISION_FILE:?Strict mode requires the embedded build revision file}"
  : "${GENET_COSMOS_REVISION:?Strict mode requires the Cosmos git revision}"
  : "${GENET_CLUSTER_LOCK:?Strict mode requires the verified cluster lock path}"
  : "${GENET_CLUSTER_RECEIPT:?Strict mode requires the node-local verification receipt}"
  : "${GENET_HF_SNAPSHOT_REVISION:?Strict mode requires the pinned HF snapshot revision}"
  : "${WAN_VAE_PATH:?Strict Cosmos mode requires WAN_VAE_PATH}"
  if [[ ! -f "${GENET_CLUSTER_LOCK}" ]]; then
    echo "GENET_CLUSTER_LOCK does not exist: ${GENET_CLUSTER_LOCK}" >&2
    exit 2
  fi
  if [[ ! -f "${GENET_BUILD_REVISION_FILE}" ]]; then
    echo "GENET_BUILD_REVISION_FILE does not exist: ${GENET_BUILD_REVISION_FILE}" >&2
    exit 2
  fi
  if [[ ! -f "${GENET_CLUSTER_RECEIPT}" ]]; then
    echo "GENET_CLUSTER_RECEIPT does not exist: ${GENET_CLUSTER_RECEIPT}" >&2
    exit 2
  fi
  if [[ ! -f "${WAN_VAE_PATH}" ]]; then
    echo "WAN_VAE_PATH does not exist: ${WAN_VAE_PATH}" >&2
    exit 2
  fi
  hf_cache_path="${HF_HUB_CACHE:-${HF_HOME:-}}"
  if [[ -z "${hf_cache_path}" || ! -d "${hf_cache_path}" ]]; then
    echo "Strict Cosmos mode requires an existing HF_HOME or HF_HUB_CACHE" >&2
    exit 2
  fi
  if [[ ! "${GENET_IMAGE_DIGEST}" =~ ^(sha256:)?[0-9a-fA-F]{64}$ ]]; then
    echo "GENET_IMAGE_DIGEST must be an immutable SHA256 digest" >&2
    exit 2
  fi
  if [[ ! "${GENET_CODE_REVISION}" =~ ^[0-9a-fA-F]{40}$ ]]; then
    echo "GENET_CODE_REVISION must be a full 40-character Git commit" >&2
    exit 2
  fi
  if [[ "$(<"${GENET_BUILD_REVISION_FILE}")" != "${GENET_CODE_REVISION}" ]]; then
    echo "Embedded build revision does not match GENET_CODE_REVISION" >&2
    exit 2
  fi
  if [[ "${GENET_COSMOS_REVISION}" != "a904d2d36b774a51dd06ff9ff906816b1a04f579" ]]; then
    echo "GENET_COSMOS_REVISION does not match the project pin" >&2
    exit 2
  fi
  if [[ ! "${GENET_HF_SNAPSHOT_REVISION}" =~ ^[0-9a-fA-F]{40}$ ]]; then
    echo "GENET_HF_SNAPSHOT_REVISION must be a full 40-character repository commit" >&2
    exit 2
  fi
fi

CONFIG_PATH="${1:-configs/base.yaml}"
shift || true

export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

torchrun \
  --nnodes="${NNODES}" \
  --nproc-per-node="${NPROC_PER_NODE}" \
  --node-rank="${NODE_RANK}" \
  --master-addr="${MASTER_ADDR}" \
  --master-port="${MASTER_PORT}" \
  --max-restarts=0 \
  -m genet.cli.train --config "${CONFIG_PATH}" "$@"
