#!/usr/bin/env bash
set -euo pipefail

readonly PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
readonly RUNNER_SCRIPT="${PROJECT_ROOT}/scripts/run_hyperbolic_container.sh"
readonly HOST_CHECK_SCRIPT="${PROJECT_ROOT}/scripts/check_hyperbolic_host.sh"
readonly IMAGE_PULL_SCRIPT="${PROJECT_ROOT}/scripts/pull_hyperbolic_image.sh"

usage() {
  cat <<'EOF'
Launch one Dockerized GenET job on a rank-ordered set of SSH hosts.

Usage:
  launch_cluster_ssh.sh \
    --hosts HOSTFILE \
    --env ENV_FILE \
    --image REPOSITORY@sha256:DIGEST \
    --config CONFIG_PATH \
    [options] -- [genet-train arguments]

Required inputs:
  --hosts FILE          One SSH target per non-comment line, in NODE_RANK order.
                        The literal "local" runs rank 0 on the coordinator.
  --env FILE            Shared shell environment (NNODES, ports, lock paths,
                        RoCE variables, revisions, and node-local mount paths).
  --image REF           Immutable application image reference by OCI digest.
  --config PATH         Training config path inside the application image.

Default sequence:
  host checks -> concurrent image pull -> CPU/Gloo preflight -> 32-rank dry run -> training

Options:
  --run-id ID           Stable container/log prefix (default: UTC timestamp + PID).
  --log-dir DIR         New coordinator-side log directory (default: ./logs/<run-id>).
  --ssh-option VALUE    Additional ssh -o option; may be repeated.
  --skip-host-check     Skip the read-only host prerequisite inventory.
  --skip-preflight      Skip the CPU/Gloo cluster preflight.
  --skip-dry-run        Start training without the automatic dry run.
  --preflight-only      Stop after host checks and CPU/Gloo preflight.
  --dry-run-only        Stop after the distributed GenET dry run.
  -h, --help            Show this help.

The environment file is sourced as shell code on the coordinator. It is not
copied to the workers. Only an allowlisted set of distributed/runtime variables
is forwarded, and NODE_RANK plus the container name are assigned by this script.
Run the coordinator in tmux/systemd for a long job; closing it stops the attached
SSH sessions and this script removes only containers carrying its launch label.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

quote_command() {
  local quoted=""
  local item
  for item in "$@"; do
    printf -v quoted '%s%q ' "${quoted}" "${item}"
  done
  printf '%s' "${quoted}"
}

trim_line() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}

HOSTFILE=""
ENV_FILE=""
IMAGE_REF=""
CONFIG_PATH=""
RUN_ID=""
LOG_DIR=""
SKIP_HOST_CHECK=0
SKIP_PREFLIGHT=0
SKIP_DRY_RUN=0
STOP_AFTER_PREFLIGHT=0
STOP_AFTER_DRY_RUN=0
SSH_OPTIONS=(
  -o BatchMode=yes
  -o ConnectTimeout=15
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=3
)

while (( $# > 0 )); do
  case "$1" in
    --hosts)
      (( $# >= 2 )) || die "--hosts requires a value"
      HOSTFILE="$2"
      shift 2
      ;;
    --env)
      (( $# >= 2 )) || die "--env requires a value"
      ENV_FILE="$2"
      shift 2
      ;;
    --image)
      (( $# >= 2 )) || die "--image requires a value"
      IMAGE_REF="$2"
      shift 2
      ;;
    --config)
      (( $# >= 2 )) || die "--config requires a value"
      CONFIG_PATH="$2"
      shift 2
      ;;
    --run-id)
      (( $# >= 2 )) || die "--run-id requires a value"
      RUN_ID="$2"
      shift 2
      ;;
    --log-dir)
      (( $# >= 2 )) || die "--log-dir requires a value"
      LOG_DIR="$2"
      shift 2
      ;;
    --ssh-option)
      (( $# >= 2 )) || die "--ssh-option requires a value"
      SSH_OPTIONS+=(-o "$2")
      shift 2
      ;;
    --skip-host-check)
      SKIP_HOST_CHECK=1
      shift
      ;;
    --skip-preflight)
      SKIP_PREFLIGHT=1
      shift
      ;;
    --skip-dry-run)
      SKIP_DRY_RUN=1
      shift
      ;;
    --preflight-only)
      STOP_AFTER_PREFLIGHT=1
      shift
      ;;
    --dry-run-only)
      STOP_AFTER_DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

TRAIN_ARGS=("$@")

[[ -n "${HOSTFILE}" ]] || die "--hosts is required"
[[ -f "${HOSTFILE}" ]] || die "hostfile does not exist: ${HOSTFILE}"
[[ -n "${ENV_FILE}" ]] || die "--env is required"
[[ -f "${ENV_FILE}" ]] || die "environment file does not exist: ${ENV_FILE}"
[[ -n "${IMAGE_REF}" ]] || die "--image is required"
[[ "${IMAGE_REF}" =~ ^[^[:space:]]+@sha256:[0-9a-fA-F]{64}$ ]] \
  || die "--image must use repository@sha256:<64 hex>"
[[ -n "${CONFIG_PATH}" ]] || die "--config is required"
(( STOP_AFTER_PREFLIGHT == 0 || STOP_AFTER_DRY_RUN == 0 )) \
  || die "--preflight-only and --dry-run-only are mutually exclusive"
(( STOP_AFTER_PREFLIGHT == 0 || SKIP_PREFLIGHT == 0 )) \
  || die "--preflight-only cannot be combined with --skip-preflight"
(( STOP_AFTER_DRY_RUN == 0 || SKIP_DRY_RUN == 0 )) \
  || die "--dry-run-only cannot be combined with --skip-dry-run"

for argument in ${TRAIN_ARGS[@]+"${TRAIN_ARGS[@]}"}; do
  [[ "${argument}" != "--dry-run" ]] \
    || die "do not forward --dry-run; use --dry-run-only or the default dry-run-first sequence"
done

set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

for secret_name in HF_TOKEN HUGGING_FACE_HUB_TOKEN; do
  if [[ -n "${!secret_name+x}" ]]; then
    die "unset ${secret_name} before training; credentials are permitted only in a private preparation-container environment"
  fi
done

: "${NNODES:?Set NNODES in the shared environment file}"
: "${NPROC_PER_NODE:?Set NPROC_PER_NODE in the shared environment file}"
: "${MASTER_ADDR:?Set MASTER_ADDR in the shared environment file}"
: "${MASTER_PORT:?Set MASTER_PORT in the shared environment file}"
: "${PREFLIGHT_PORT:?Set PREFLIGHT_PORT in the shared environment file}"
: "${GENET_NODE_CACHE:?Set GENET_NODE_CACHE in the shared environment file}"
: "${GENET_NODE_RUN_ROOT:?Set GENET_NODE_RUN_ROOT in the shared environment file}"
GENET_IMAGE_PULL_POLICY="${GENET_IMAGE_PULL_POLICY:-missing}"
case "${GENET_IMAGE_PULL_POLICY}" in
  always|missing|never) ;;
  *) die "GENET_IMAGE_PULL_POLICY must be always, missing, or never" ;;
esac

for integer_name in NNODES NPROC_PER_NODE MASTER_PORT PREFLIGHT_PORT; do
  integer_value="${!integer_name}"
  [[ "${integer_value}" =~ ^[0-9]+$ ]] \
    || die "${integer_name} must be a non-negative integer: ${integer_value}"
done
(( NNODES > 0 && NPROC_PER_NODE > 0 )) || die "NNODES and NPROC_PER_NODE must be positive"
(( MASTER_PORT > 0 && MASTER_PORT <= 65535 )) || die "invalid MASTER_PORT: ${MASTER_PORT}"
(( PREFLIGHT_PORT > 0 && PREFLIGHT_PORT <= 65535 )) || die "invalid PREFLIGHT_PORT: ${PREFLIGHT_PORT}"
(( MASTER_PORT != PREFLIGHT_PORT )) || die "MASTER_PORT and PREFLIGHT_PORT must differ"
if (( NNODES > 1 )); then
  case "${MASTER_ADDR}" in
    localhost|127.*|::1) die "MASTER_ADDR must be reachable from peer nodes" ;;
  esac
fi

HOSTS=()
while IFS= read -r raw_line || [[ -n "${raw_line}" ]]; do
  host_line="$(trim_line "${raw_line}")"
  case "${host_line}" in
    ''|'#'*) continue ;;
  esac
  [[ "${host_line}" != *[[:space:]]* ]] \
    || die "hostfile entries must be one SSH target without spaces: ${host_line}"
  HOSTS+=("${host_line}")
done < "${HOSTFILE}"

(( ${#HOSTS[@]} == NNODES )) \
  || die "hostfile has ${#HOSTS[@]} nodes but NNODES=${NNODES}"
for (( left = 0; left < ${#HOSTS[@]}; left += 1 )); do
  for (( right = left + 1; right < ${#HOSTS[@]}; right += 1 )); do
    [[ "${HOSTS[left]}" != "${HOSTS[right]}" ]] \
      || die "duplicate hostfile target: ${HOSTS[left]}"
  done
done

if [[ -z "${RUN_ID}" ]]; then
  RUN_ID="genet-$(date -u +%Y%m%dT%H%M%SZ)-$$"
fi
[[ "${RUN_ID}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,62}$ ]] \
  || die "--run-id must be 1-63 characters from [A-Za-z0-9_.-] and start alphanumeric"
if [[ -z "${LOG_DIR}" ]]; then
  LOG_DIR="${PROJECT_ROOT}/logs/${RUN_ID}"
fi
mkdir -p -- "$(dirname -- "${LOG_DIR}")"
if ! mkdir -- "${LOG_DIR}"; then
  die "log directory must be new; choose a unique --run-id/--log-dir: ${LOG_DIR}"
fi
readonly LAUNCH_ID="$(od -An -N16 -tx1 /dev/urandom | tr -d '[:space:]')"
[[ "${LAUNCH_ID}" =~ ^[0-9a-f]{32}$ ]] || die "failed to create a unique launch identity"

# The helper performs the same allowlist again before passing variables to
# Docker. Exclude controller/rank-specific values and assign them below.
FORWARDED_ENV=()
while IFS= read -r variable_name; do
  case "${variable_name}" in
    NODE_RANK|GENET_NODE_ID|GENET_CONTAINER_NAME|GENET_CONTAINER_DETACH|GENET_CONTAINER_RM)
      ;;
    NNODES|NPROC_PER_NODE|MASTER_ADDR|MASTER_PORT|PREFLIGHT_PORT|WAN_VAE_PATH|BASE_CHECKPOINT_PATH|OMP_NUM_THREADS|HF_HOME|HF_HUB_CACHE|HF_HUB_OFFLINE|TRANSFORMERS_OFFLINE|GENET_*|NCCL_*|TORCH_NCCL_*|GLOO_SOCKET_IFNAME)
      FORWARDED_ENV+=("${variable_name}=${!variable_name}")
      ;;
  esac
done < <(compgen -e | LC_ALL=C sort)

ACTIVE_PIDS=()
ACTIVE_CONTAINERS=()
ACTIVE_TARGETS=()
CLEANUP_RUNNING=0

run_transport() {
  local target="$1"
  local script_path="$2"
  shift 2
  local environment=(env "${FORWARDED_ENV[@]}")
  if [[ "${target}" == "local" ]]; then
    "${environment[@]}" bash "${script_path}" "$@"
    return
  fi
  local remote_command
  remote_command="$(quote_command "${environment[@]}" bash -s -- "$@")"
  ssh "${SSH_OPTIONS[@]}" "${target}" "${remote_command}" < "${script_path}"
}

remove_remote_container() {
  local target="$1"
  local container_name="$2"
  local cleanup_command
  cleanup_command='name="$1"; launch_id="$2"; for attempt in 1 2 3 4 5; do actual="$(docker container inspect --format '\''{{ index .Config.Labels "ai.genet.launch-id" }}'\'' "${name}" 2>/dev/null || true)"; if [[ -z "${actual}" ]]; then sleep 1; continue; fi; if [[ "${actual}" != "${launch_id}" ]]; then echo "Refusing to remove container with foreign launch label: ${name}" >&2; exit 3; fi; docker container rm --force "${name}" >/dev/null; exit 0; done; exit 0'
  if [[ "${target}" == "local" ]]; then
    bash -c "${cleanup_command}" bash "${container_name}" "${LAUNCH_ID}" || true
  else
    local remote_command
    remote_command="$(quote_command bash -c "${cleanup_command}" bash "${container_name}" "${LAUNCH_ID}")"
    ssh "${SSH_OPTIONS[@]}" "${target}" "${remote_command}" || true
  fi
}

cleanup_active() {
  if (( CLEANUP_RUNNING != 0 )); then
    return 0
  fi
  CLEANUP_RUNNING=1
  local index cleanup_pid pid
  local -a cleanup_pids=()
  for pid in ${ACTIVE_PIDS[@]+"${ACTIVE_PIDS[@]}"}; do
    kill "${pid}" >/dev/null 2>&1 || true
  done
  for pid in ${ACTIVE_PIDS[@]+"${ACTIVE_PIDS[@]}"}; do
    wait "${pid}" || true
  done
  for (( index = 0; index < ${#ACTIVE_CONTAINERS[@]}; index += 1 )); do
    if [[ -n "${ACTIVE_CONTAINERS[index]}" ]]; then
      remove_remote_container "${ACTIVE_TARGETS[index]}" "${ACTIVE_CONTAINERS[index]}" &
      cleanup_pids+=("$!")
    fi
  done
  for cleanup_pid in ${cleanup_pids[@]+"${cleanup_pids[@]}"}; do
    wait "${cleanup_pid}" || true
  done
  ACTIVE_PIDS=()
  ACTIVE_CONTAINERS=()
  ACTIVE_TARGETS=()
  CLEANUP_RUNNING=0
}

on_signal() {
  echo "Stopping ${RUN_ID} containers after coordinator signal" >&2
  cleanup_active
  exit 130
}
on_exit() {
  local status=$?
  if (( ${#ACTIVE_PIDS[@]} > 0 )); then
    cleanup_active
  fi
  return "${status}"
}
trap on_signal INT TERM HUP
trap on_exit EXIT

wait_for_phase() {
  local phase="$1"
  local remaining=${#ACTIVE_PIDS[@]}
  local -a completed=()
  local index pid status
  for (( index = 0; index < remaining; index += 1 )); do
    completed+=(0)
  done
  while (( remaining > 0 )); do
    for (( index = 0; index < ${#ACTIVE_PIDS[@]}; index += 1 )); do
      (( completed[index] == 0 )) || continue
      pid="${ACTIVE_PIDS[index]}"
      if kill -0 "${pid}" >/dev/null 2>&1; then
        continue
      fi
      status=0
      wait "${pid}" || status=$?
      completed[index]=1
      remaining=$((remaining - 1))
      if (( status != 0 )); then
        echo "${phase} failed on rank ${index} (${HOSTS[index]}), exit=${status}" >&2
        cleanup_active
        return "${status}"
      fi
    done
    (( remaining == 0 )) || sleep 1
  done
  ACTIVE_PIDS=()
  ACTIVE_CONTAINERS=()
  ACTIVE_TARGETS=()
  echo "Completed phase: ${phase}"
}

launch_host_checks() {
  local rank target log_path
  echo "Starting read-only host checks on ${NNODES} nodes"
  ACTIVE_PIDS=()
  ACTIVE_CONTAINERS=()
  ACTIVE_TARGETS=()
  for (( rank = 0; rank < NNODES; rank += 1 )); do
    target="${HOSTS[rank]}"
    log_path="${LOG_DIR}/host-check-rank-${rank}.log"
    (
      set -o pipefail
      run_transport "${target}" "${HOST_CHECK_SCRIPT}" 2>&1 | tee "${log_path}"
    ) &
    ACTIVE_PIDS+=("$!")
    ACTIVE_CONTAINERS+=("")
    ACTIVE_TARGETS+=("${target}")
  done
  wait_for_phase host-check
}

launch_image_pulls() {
  local rank target log_path
  echo "Resolving immutable image on ${NNODES} nodes"
  ACTIVE_PIDS=()
  ACTIVE_CONTAINERS=()
  ACTIVE_TARGETS=()
  for (( rank = 0; rank < NNODES; rank += 1 )); do
    target="${HOSTS[rank]}"
    log_path="${LOG_DIR}/image-pull-rank-${rank}.log"
    (
      set -o pipefail
      run_transport \
        "${target}" \
        "${IMAGE_PULL_SCRIPT}" \
        "${IMAGE_REF}" \
        "${GENET_IMAGE_PULL_POLICY}" 2>&1 | tee "${log_path}"
    ) &
    ACTIVE_PIDS+=("$!")
    ACTIVE_CONTAINERS+=("")
    ACTIVE_TARGETS+=("${target}")
  done
  wait_for_phase image-pull
}

launch_container_phase() {
  local phase="$1"
  shift
  local -a payload=("$@")
  local rank target container_name log_path remote_command
  local -a environment
  echo "Starting phase ${phase} on ${NNODES} nodes; logs: ${LOG_DIR}"
  ACTIVE_PIDS=()
  ACTIVE_CONTAINERS=()
  ACTIVE_TARGETS=()
  for (( rank = 0; rank < NNODES; rank += 1 )); do
    target="${HOSTS[rank]}"
    container_name="${RUN_ID}-${LAUNCH_ID:0:12}-${phase}-r${rank}"
    log_path="${LOG_DIR}/${phase}-rank-${rank}.log"
    environment=(
      env
      "${FORWARDED_ENV[@]}"
      "NODE_RANK=${rank}"
      "GENET_CONTAINER_NAME=${container_name}"
      "GENET_LAUNCH_ID=${LAUNCH_ID}"
      GENET_IMAGE_PULL_POLICY=never
      GENET_CONTAINER_DETACH=0
      GENET_CONTAINER_RM=1
    )
    if [[ "${target}" == "local" ]]; then
      (
        set -o pipefail
        "${environment[@]}" bash "${RUNNER_SCRIPT}" - "${IMAGE_REF}" "${payload[@]}" \
          2>&1 | tee "${log_path}"
      ) &
    else
      remote_command="$(quote_command "${environment[@]}" bash -s -- - "${IMAGE_REF}" "${payload[@]}")"
      (
        set -o pipefail
        ssh "${SSH_OPTIONS[@]}" "${target}" "${remote_command}" < "${RUNNER_SCRIPT}" \
          2>&1 | tee "${log_path}"
      ) &
    fi
    ACTIVE_PIDS+=("$!")
    ACTIVE_CONTAINERS+=("${container_name}")
    ACTIVE_TARGETS+=("${target}")
  done
  wait_for_phase "${phase}"
}

echo "GenET cluster launch"
echo "  run id: ${RUN_ID}"
echo "  launch id: ${LAUNCH_ID}"
echo "  topology: ${NNODES} nodes x ${NPROC_PER_NODE} ranks"
echo "  master: ${MASTER_ADDR}:${MASTER_PORT}"
echo "  image: ${IMAGE_REF}"
echo "  data: ${GENET_NODE_CACHE}"
echo "  logs: ${LOG_DIR}"

if (( SKIP_HOST_CHECK == 0 )); then
  launch_host_checks
fi
launch_image_pulls
if (( SKIP_PREFLIGHT == 0 )); then
  launch_container_phase preflight bash scripts/preflight_roce.sh
fi
if (( STOP_AFTER_PREFLIGHT == 1 )); then
  echo "Preflight-only launch completed"
  exit 0
fi
if (( SKIP_DRY_RUN == 0 )); then
  launch_container_phase dry-run \
    bash scripts/launch_roce.sh "${CONFIG_PATH}" ${TRAIN_ARGS[@]+"${TRAIN_ARGS[@]}"} --dry-run
fi
if (( STOP_AFTER_DRY_RUN == 1 )); then
  echo "Dry-run-only launch completed"
  exit 0
fi

launch_container_phase train \
  bash scripts/launch_roce.sh "${CONFIG_PATH}" ${TRAIN_ARGS[@]+"${TRAIN_ARGS[@]}"}
echo "Training completed successfully on all nodes"
