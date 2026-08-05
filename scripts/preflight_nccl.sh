#!/usr/bin/env bash
set -euo pipefail

: "${NNODES:?Set NNODES (4 for the full cluster)}"
: "${NPROC_PER_NODE:?Set NPROC_PER_NODE (8 for the full cluster)}"
: "${NODE_RANK:?Set the unique NODE_RANK on each node (0..NNODES-1)}"
: "${MASTER_ADDR:?Set MASTER_ADDR to the reachable bootstrap address of node 0}"

NCCL_PREFLIGHT_PORT="${NCCL_PREFLIGHT_PORT:-${PREFLIGHT_PORT:-}}"
if [[ -z "${NCCL_PREFLIGHT_PORT}" ]]; then
  echo "Set NCCL_PREFLIGHT_PORT (or PREFLIGHT_PORT) to a dedicated shared port" >&2
  exit 2
fi

GENET_NCCL_PREFLIGHT_BUFFER_MIB="${GENET_NCCL_PREFLIGHT_BUFFER_MIB:-64}"
GENET_NCCL_PREFLIGHT_WARMUP_ITERATIONS="${GENET_NCCL_PREFLIGHT_WARMUP_ITERATIONS:-3}"
GENET_NCCL_PREFLIGHT_ITERATIONS="${GENET_NCCL_PREFLIGHT_ITERATIONS:-10}"
GENET_NCCL_PREFLIGHT_TIMEOUT_SECONDS="${GENET_NCCL_PREFLIGHT_TIMEOUT_SECONDS:-180}"

for integer_name in \
  NNODES \
  NPROC_PER_NODE \
  NODE_RANK \
  NCCL_PREFLIGHT_PORT \
  GENET_NCCL_PREFLIGHT_BUFFER_MIB \
  GENET_NCCL_PREFLIGHT_WARMUP_ITERATIONS \
  GENET_NCCL_PREFLIGHT_ITERATIONS \
  GENET_NCCL_PREFLIGHT_TIMEOUT_SECONDS; do
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
if (( NCCL_PREFLIGHT_PORT < 1 || NCCL_PREFLIGHT_PORT > 65535 )); then
  echo "NCCL_PREFLIGHT_PORT must be in [1, 65535], got ${NCCL_PREFLIGHT_PORT}" >&2
  exit 2
fi
if (( GENET_NCCL_PREFLIGHT_BUFFER_MIB < 1 )); then
  echo "GENET_NCCL_PREFLIGHT_BUFFER_MIB must be positive" >&2
  exit 2
fi
if (( GENET_NCCL_PREFLIGHT_ITERATIONS < 1 )); then
  echo "GENET_NCCL_PREFLIGHT_ITERATIONS must be positive" >&2
  exit 2
fi
if (( GENET_NCCL_PREFLIGHT_TIMEOUT_SECONDS < 10 )); then
  echo "GENET_NCCL_PREFLIGHT_TIMEOUT_SECONDS must be at least 10" >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"

exec torchrun \
  --nnodes="${NNODES}" \
  --nproc-per-node="${NPROC_PER_NODE}" \
  --node-rank="${NODE_RANK}" \
  --master-addr="${MASTER_ADDR}" \
  --master-port="${NCCL_PREFLIGHT_PORT}" \
  --rdzv-conf="timeout=${GENET_NCCL_PREFLIGHT_TIMEOUT_SECONDS}" \
  --max-restarts=0 \
  -m genet.cli.nccl_preflight \
  --expected-nnodes "${NNODES}" \
  --expected-local-world-size "${NPROC_PER_NODE}" \
  --buffer-mib "${GENET_NCCL_PREFLIGHT_BUFFER_MIB}" \
  --warmup-iterations "${GENET_NCCL_PREFLIGHT_WARMUP_ITERATIONS}" \
  --iterations "${GENET_NCCL_PREFLIGHT_ITERATIONS}" \
  --timeout-seconds "${GENET_NCCL_PREFLIGHT_TIMEOUT_SECONDS}"
