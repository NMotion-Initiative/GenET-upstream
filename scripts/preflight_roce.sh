#!/usr/bin/env bash
set -euo pipefail

: "${NNODES:?Set NNODES (4 for the full cluster)}"
: "${NPROC_PER_NODE:?Set NPROC_PER_NODE (8 for the full cluster)}"
: "${NODE_RANK:?Set the unique NODE_RANK on each node (0..NNODES-1)}"
: "${MASTER_ADDR:?Set MASTER_ADDR to the reachable bootstrap address of node 0}"
: "${PREFLIGHT_PORT:?Set a dedicated PREFLIGHT_PORT shared by this job}"

GENET_PREFLIGHT_TIMEOUT_SECONDS="${GENET_PREFLIGHT_TIMEOUT_SECONDS:-120}"
export GENET_PREFLIGHT_TIMEOUT_SECONDS

for integer_name in NNODES NPROC_PER_NODE NODE_RANK PREFLIGHT_PORT GENET_PREFLIGHT_TIMEOUT_SECONDS; do
  integer_value="${!integer_name}"
  if [[ ! "${integer_value}" =~ ^[0-9]+$ ]]; then
    echo "${integer_name} must be a non-negative integer, got: ${integer_value}" >&2
    exit 2
  fi
done
if (( NNODES < 1 || NPROC_PER_NODE < 1 || NODE_RANK >= NNODES )); then
  echo "Invalid node topology: NNODES=${NNODES}, NPROC_PER_NODE=${NPROC_PER_NODE}, NODE_RANK=${NODE_RANK}" >&2
  exit 2
fi
if (( PREFLIGHT_PORT < 1 || PREFLIGHT_PORT > 65535 )); then
  echo "PREFLIGHT_PORT must be in [1, 65535], got ${PREFLIGHT_PORT}" >&2
  exit 2
fi

torchrun \
  --nnodes="${NNODES}" \
  --nproc-per-node=1 \
  --node-rank="${NODE_RANK}" \
  --master-addr="${MASTER_ADDR}" \
  --master-port="${PREFLIGHT_PORT}" \
  --rdzv-conf="timeout=${GENET_PREFLIGHT_TIMEOUT_SECONDS}" \
  --max-restarts=0 \
  -m genet.cli.cluster_preflight --expected-gpus "${NPROC_PER_NODE}"
