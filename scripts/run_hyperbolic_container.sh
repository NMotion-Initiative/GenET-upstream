#!/usr/bin/env bash
set -euo pipefail

if (( $# < 2 )); then
  echo "Usage: $0 HOST_ENV_FILE|- IMAGE_REF [COMMAND ...]" >&2
  exit 2
fi

readonly HOST_ENV_FILE="$1"
readonly IMAGE_REF="$2"
shift 2

if [[ "${HOST_ENV_FILE}" != "-" ]]; then
  if [[ ! -f "${HOST_ENV_FILE}" ]]; then
    echo "Host environment file does not exist: ${HOST_ENV_FILE}" >&2
    exit 2
  fi
  set -a
  # shellcheck disable=SC1090
  source "${HOST_ENV_FILE}"
  set +a
fi

: "${NODE_RANK:?Set the unique NODE_RANK in HOST_ENV_FILE or the calling environment}"
readonly CACHE_ROOT="${GENET_NODE_CACHE:-/mnt/nvme/mds-cache/robotwin_v1}"
readonly RUN_ROOT="${GENET_NODE_RUN_ROOT:-/mnt/nvme/genet}"
readonly CONTAINER_NAME="${GENET_CONTAINER_NAME:-genet-node-${NODE_RANK}}"
readonly IMAGE_PULL_POLICY="${GENET_IMAGE_PULL_POLICY:-missing}"
readonly LAUNCH_ID="${GENET_LAUNCH_ID:-manual}"

if [[ ! -d "${CACHE_ROOT}" ]]; then
  echo "Node-local cache does not exist: ${CACHE_ROOT}" >&2
  exit 2
fi
mkdir -p -- "${RUN_ROOT}"

case "${IMAGE_REF}" in
  *@sha256:[0-9a-fA-F][0-9a-fA-F]*) image_digest="${IMAGE_REF##*@}" ;;
  sha256:[0-9a-fA-F][0-9a-fA-F]*) image_digest="${IMAGE_REF}" ;;
  *)
    echo "IMAGE_REF must be immutable: repository@sha256:<digest> or sha256:<image-id>" >&2
    exit 2
    ;;
esac
if [[ ! "${image_digest}" =~ ^sha256:[0-9a-fA-F]{64}$ ]]; then
  echo "IMAGE_REF contains an invalid SHA256 digest: ${IMAGE_REF}" >&2
  exit 2
fi
export GENET_IMAGE_DIGEST="${image_digest}"
export GENET_NODE_ID="${GENET_NODE_ID:-$(hostname -f 2>/dev/null || hostname)}"

if docker container inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  echo "Container name already exists; inspect or remove it explicitly: ${CONTAINER_NAME}" >&2
  exit 2
fi

rdma_args=()
while IFS= read -r -d '' device_path; do
  rdma_args+=(--device "${device_path}:${device_path}")
done < <(find /dev/infiniband -maxdepth 1 -type c -print0 2>/dev/null)
if (( ${#rdma_args[@]} == 0 )); then
  echo "No RDMA character devices found under /dev/infiniband" >&2
  exit 2
fi

env_args=()
while IFS= read -r variable_name; do
  case "${variable_name}" in
    NNODES|NPROC_PER_NODE|NODE_RANK|MASTER_ADDR|MASTER_PORT|PREFLIGHT_PORT|BASE_CHECKPOINT_PATH|DATASET_PATH|OMP_NUM_THREADS|HF_HOME|HF_HUB_CACHE|HF_HUB_OFFLINE|TRANSFORMERS_OFFLINE|WAN_*|GENET_*|NCCL_*|TORCH_NCCL_*|GLOO_SOCKET_IFNAME|WANDB_*|HTTP_PROXY|HTTPS_PROXY|http_proxy|https_proxy|NO_PROXY|no_proxy|ALL_PROXY|all_proxy)
      env_args+=(--env "${variable_name}")
      ;;
    HF_TOKEN|HUGGING_FACE_HUB_TOKEN)
      case "${GENET_ALLOW_HF_TOKEN:-0}" in
        1|true|TRUE|yes|YES|on|ON) env_args+=(--env "${variable_name}") ;;
      esac
      ;;
  esac
done < <(compgen -e | LC_ALL=C sort)

detach_args=()
case "${GENET_CONTAINER_DETACH:-0}" in
  1|true|TRUE|yes|YES|on|ON) detach_args+=(--detach) ;;
  0|false|FALSE|no|NO|off|OFF) ;;
  *)
    echo "GENET_CONTAINER_DETACH must be boolean" >&2
    exit 2
    ;;
esac

cache_mount="type=bind,src=${CACHE_ROOT},dst=${CACHE_ROOT}"
case "${GENET_CACHE_READONLY:-1}" in
  1|true|TRUE|yes|YES|on|ON) cache_mount+=",readonly" ;;
  0|false|FALSE|no|NO|off|OFF) ;;
  *)
    echo "GENET_CACHE_READONLY must be boolean" >&2
    exit 2
    ;;
esac

protected_input_args=()
case "${GENET_PROTECT_INPUTS:-0}" in
  1|true|TRUE|yes|YES|on|ON)
    protected_variables=(
      GENET_PROCESSED_DATA
      HF_HOME
      WAN_VAE_PATH
      BASE_CHECKPOINT_PATH
      GENET_ARTIFACT_RECEIPT_PATH
      GENET_CLUSTER_LOCK
      GENET_CLUSTER_RECEIPT
    )
    if [[ -n "${GENET_NORMALIZATION_PATH:-}" ]]; then
      protected_variables+=(GENET_NORMALIZATION_PATH)
    fi
    for variable_name in "${protected_variables[@]}"; do
      protected_path="${!variable_name:-}"
      if [[ -z "${protected_path}" || "${protected_path}" != /* ]]; then
        echo "${variable_name} must be an absolute path when GENET_PROTECT_INPUTS=1" >&2
        exit 2
      fi
      if [[ ! -e "${protected_path}" ]]; then
        echo "Protected input does not exist (${variable_name}): ${protected_path}" >&2
        exit 2
      fi
      protected_input_args+=(
        --mount "type=bind,src=${protected_path},dst=${protected_path},readonly"
      )
    done
    ;;
  0|false|FALSE|no|NO|off|OFF) ;;
  *)
    echo "GENET_PROTECT_INPUTS must be boolean" >&2
    exit 2
    ;;
esac

case "${IMAGE_PULL_POLICY}" in
  always|missing|never) ;;
  *)
    echo "GENET_IMAGE_PULL_POLICY must be always, missing, or never" >&2
    exit 2
    ;;
esac

command_args=("$@")
if (( ${#command_args[@]} == 0 )); then
  command_args=(bash)
fi

exec docker run \
  --name "${CONTAINER_NAME}" \
  --label "ai.genet.launch-id=${LAUNCH_ID}" \
  --rm \
  --pull "${IMAGE_PULL_POLICY}" \
  --init \
  --gpus all \
  --network host \
  --ipc host \
  --ulimit memlock=-1:-1 \
  --ulimit stack=67108864:67108864 \
  --cap-add IPC_LOCK \
  --workdir /opt/genet \
  --mount "${cache_mount}" \
  --mount "type=bind,src=${RUN_ROOT},dst=${RUN_ROOT}" \
  ${protected_input_args[@]+"${protected_input_args[@]}"} \
  "${rdma_args[@]}" \
  "${env_args[@]}" \
  ${detach_args[@]+"${detach_args[@]}"} \
  "${IMAGE_REF}" \
  "${command_args[@]}"
