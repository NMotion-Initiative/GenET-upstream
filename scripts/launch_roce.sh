#!/usr/bin/env bash
set -euo pipefail

: "${NNODES:?Set NNODES (4 for the full cluster)}"
: "${NPROC_PER_NODE:?Set NPROC_PER_NODE (8 for the full cluster)}"
: "${NODE_RANK:?Set the unique NODE_RANK on each node (0..NNODES-1)}"
: "${MASTER_ADDR:?Set MASTER_ADDR to the reachable bootstrap address of node 0}"
: "${MASTER_PORT:?Set MASTER_PORT}"
: "${RDZV_ID:?Set one unique RDZV_ID shared by all nodes in this job}"

CONFIG_PATH="${1:-configs/base.yaml}"
shift || true

export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

torchrun \
  --nnodes="${NNODES}" \
  --nproc-per-node="${NPROC_PER_NODE}" \
  --node-rank="${NODE_RANK}" \
  --rdzv-backend=c10d \
  --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  --rdzv-id="${RDZV_ID}" \
  -m genet.cli.train --config "${CONFIG_PATH}" "$@"
