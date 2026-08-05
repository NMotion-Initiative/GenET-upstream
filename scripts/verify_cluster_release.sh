#!/usr/bin/env bash
set -euo pipefail

# Re-hash the node-local immutable inputs against the shared release lock and
# refresh this node's path-binding receipt. This runs inside the same immutable
# training image immediately before distributed preflight/training.

: "${GENET_CLUSTER_LOCK:?Set GENET_CLUSTER_LOCK to the shared release lock}"
: "${GENET_CLUSTER_RECEIPT:?Set GENET_CLUSTER_RECEIPT to this node-local receipt path}"
: "${GENET_NODE_RUN_ROOT:?Set GENET_NODE_RUN_ROOT to the staged artifact root}"
: "${GENET_PROCESSED_DATA:?Set GENET_PROCESSED_DATA to the processed dataset root}"
: "${WAN_VAE_PATH:?Set WAN_VAE_PATH to the staged Wan VAE}"
: "${BASE_CHECKPOINT_PATH:?Set BASE_CHECKPOINT_PATH to the exact load DCP}"
: "${HF_HOME:?Set HF_HOME to the complete offline Hugging Face cache}"
: "${GENET_ARTIFACT_RECEIPT_PATH:?Set GENET_ARTIFACT_RECEIPT_PATH to ARTIFACTS.json}"

readonly DATA_NAME="${GENET_DATA_ARTIFACT:-processed_data}"
readonly WAN_NAME="${GENET_WAN_VAE_ARTIFACT:-wan_vae}"
readonly CHECKPOINT_NAME="${GENET_CHECKPOINT_ARTIFACT:-training_checkpoint}"
readonly HF_NAME="${GENET_HF_ARTIFACT:-hf_cache}"
readonly ARTIFACT_RECEIPT_NAME="${GENET_ARTIFACT_RECEIPT_ARTIFACT:-artifact_receipt}"

verify_args=(
  verify
  --lock "${GENET_CLUSTER_LOCK}"
  --receipt "${GENET_CLUSTER_RECEIPT}"
  --artifact "${DATA_NAME}=${GENET_PROCESSED_DATA}"
  --artifact "${WAN_NAME}=${WAN_VAE_PATH}"
  --artifact "${ARTIFACT_RECEIPT_NAME}=${GENET_ARTIFACT_RECEIPT_PATH}"
  --artifact "${CHECKPOINT_NAME}=${BASE_CHECKPOINT_PATH}"
  --artifact "${HF_NAME}=${HF_HOME}"
)

if [[ -n "${GENET_NORMALIZATION_ARTIFACT:-}" || -n "${GENET_NORMALIZATION_PATH:-}" ]]; then
  : "${GENET_NORMALIZATION_ARTIFACT:?Set both GENET_NORMALIZATION_ARTIFACT and GENET_NORMALIZATION_PATH}"
  : "${GENET_NORMALIZATION_PATH:?Set both GENET_NORMALIZATION_ARTIFACT and GENET_NORMALIZATION_PATH}"
  verify_args+=(
    --artifact "${GENET_NORMALIZATION_ARTIFACT}=${GENET_NORMALIZATION_PATH}"
  )
fi

# Validate the internal Cosmos snapshot, Wan VAE, DCP manifest, and staging
# receipt before comparing the complete release bytes with the cluster lock.
genet-stage-artifacts \
  --run-root "${GENET_NODE_RUN_ROOT}" \
  --verify-only

exec genet-cluster-lock "${verify_args[@]}"
