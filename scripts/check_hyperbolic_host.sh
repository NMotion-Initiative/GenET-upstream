#!/usr/bin/env bash
set -euo pipefail

readonly EXPECTED_GPUS="${GENET_EXPECTED_GPUS:-8}"
readonly NVME_ROOT="${GENET_NVME_ROOT:-/mnt/nvme}"
readonly CACHE_ROOT="${GENET_NODE_CACHE:-/mnt/nvme/mds-cache/robotwin_v1}"
readonly BUILD_ROOT="${GENET_BUILD_CONTEXT_ROOT:-/dev/shm}"

failures=()
warnings=()

fail() {
  failures+=("$1")
}

warn() {
  warnings+=("$1")
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    fail "missing host command: $1"
  fi
}

for command_name in docker nvidia-smi ip findmnt df du find awk grep wc tr ssh rsync; do
  require_command "${command_name}"
done

if [[ ! "${EXPECTED_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
  fail "GENET_EXPECTED_GPUS must be a positive integer: ${EXPECTED_GPUS}"
fi

if [[ ! -d "${NVME_ROOT}" ]]; then
  fail "node-local NVMe root does not exist: ${NVME_ROOT}"
elif command -v findmnt >/dev/null 2>&1; then
  nvme_mount="$(findmnt -n -o TARGET --target "${NVME_ROOT}" 2>/dev/null || true)"
  if [[ "${nvme_mount}" != "${NVME_ROOT}" ]]; then
    fail "${NVME_ROOT} is not its own mount point (resolved mount: ${nvme_mount:-none}); refusing to treat the OS filesystem as the 42 TB RAID0"
  fi
fi

if [[ ! -d "${CACHE_ROOT}" ]]; then
  fail "prewarmed node-local cache does not exist: ${CACHE_ROOT}"
elif [[ ! -r "${CACHE_ROOT}" || ! -x "${CACHE_ROOT}" ]]; then
  fail "prewarmed node-local cache is not readable/searchable: ${CACHE_ROOT}"
fi

if [[ ! -d "${BUILD_ROOT}" || ! -w "${BUILD_ROOT}" ]]; then
  fail "build-context root must be an existing writable directory: ${BUILD_ROOT}"
elif command -v findmnt >/dev/null 2>&1; then
  build_fstype="$(findmnt -n -o FSTYPE --target "${BUILD_ROOT}" 2>/dev/null || true)"
  if [[ "${BUILD_ROOT}" == "/dev/shm" && "${build_fstype}" != "tmpfs" ]]; then
    fail "/dev/shm is expected to be tmpfs, got: ${build_fstype:-unknown}"
  fi
fi

if [[ ! -d /dev/infiniband ]]; then
  fail "RoCE device directory is missing: /dev/infiniband"
else
  rdma_devices=0
  while IFS= read -r device_path; do
    ((rdma_devices += 1))
  done < <(find /dev/infiniband -maxdepth 1 -type c -print 2>/dev/null)
  if (( rdma_devices == 0 )); then
    fail "no RDMA character devices found under /dev/infiniband"
  fi
  if [[ ! -c /dev/infiniband/rdma_cm ]]; then
    fail "RDMA connection-manager device is missing: /dev/infiniband/rdma_cm"
  fi
  if ! find /dev/infiniband -maxdepth 1 -type c -name 'uverbs*' -print -quit | grep -q .; then
    fail "no userspace verbs device found under /dev/infiniband (expected uverbs*)"
  fi
fi

if ! command -v ibdev2netdev >/dev/null 2>&1 && ! command -v rdma >/dev/null 2>&1; then
  fail "install RDMA discovery tools: neither ibdev2netdev nor rdma is available"
fi

if command -v nvidia-smi >/dev/null 2>&1 && [[ "${EXPECTED_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
  gpu_inventory=""
  if ! gpu_inventory="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null)"; then
    fail "nvidia-smi could not query the local GPU inventory"
  else
    gpu_count="$(printf '%s\n' "${gpu_inventory}" | awk 'NF { count += 1 } END { print count + 0 }')"
    if [[ "${gpu_count}" != "${EXPECTED_GPUS}" ]]; then
      fail "expected ${EXPECTED_GPUS} GPUs, nvidia-smi reported ${gpu_count}"
    fi
  fi
fi

docker_root=""
docker_driver=""
if command -v docker >/dev/null 2>&1; then
  if ! docker version >/dev/null 2>&1; then
    fail "Docker daemon is unavailable to the current user"
  else
    docker_root="$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || true)"
    docker_driver="$(docker info --format '{{.Driver}} {{json .DriverStatus}}' 2>/dev/null || true)"
    docker_runtimes="$(docker info --format '{{json .Runtimes}}' 2>/dev/null || true)"
    if [[ "${docker_runtimes}" != *'nvidia'* ]]; then
      fail "Docker has no registered NVIDIA runtime; configure NVIDIA Container Toolkit"
    fi
    if [[ -z "${docker_root}" ]]; then
      fail "could not determine DockerRootDir"
    elif [[ "${docker_root}" != "${NVME_ROOT}"/* ]]; then
      warn "DockerRootDir is ${docker_root}, not beneath ${NVME_ROOT}; changing the build cwd to /dev/shm will not move image layers"
    fi
    if [[ "${docker_driver}" == *containerd* ]]; then
      warn "Docker is using the containerd image store; also inspect containerd root (commonly /var/lib/containerd), because Docker data-root alone may not relocate snapshots"
    fi
  fi
fi

if [[ -n "${NCCL_SOCKET_IFNAME:-}" && "${NCCL_SOCKET_IFNAME}" != *','* && "${NCCL_SOCKET_IFNAME}" != '^'* ]]; then
  socket_interface="${NCCL_SOCKET_IFNAME#=}"
  if [[ ! -e "/sys/class/net/${socket_interface}" ]]; then
    fail "NCCL_SOCKET_IFNAME does not name a local interface: ${NCCL_SOCKET_IFNAME}"
  fi
fi

if [[ -n "${NCCL_IB_HCA:-}" && "${NCCL_IB_HCA}" != '^'* ]]; then
  IFS=',' read -r -a requested_hcas <<< "${NCCL_IB_HCA#=}"
  for requested_hca in "${requested_hcas[@]}"; do
    requested_hca="${requested_hca%%:*}"
    if [[ -n "${requested_hca}" && ! -e "/sys/class/infiniband/${requested_hca}" ]]; then
      fail "NCCL_IB_HCA contains an HCA not present on this host: ${requested_hca}"
    fi
  done
fi

echo "Hyperbolic host inventory"
echo "  NVMe root: ${NVME_ROOT}"
if [[ -d "${NVME_ROOT}" ]]; then
  findmnt --target "${NVME_ROOT}" || true
  df -h "${NVME_ROOT}" || true
fi
echo "  Prewarmed cache: ${CACHE_ROOT}"
if [[ -d "${CACHE_ROOT}" ]]; then
  case "${GENET_MEASURE_CACHE_BYTES:-0}" in
    1|true|TRUE|yes|YES|on|ON)
      cache_bytes="$(du -sb "${CACHE_ROOT}" 2>/dev/null | awk '{print $1}' || true)"
      echo "  Prewarmed cache bytes: ${cache_bytes:-unavailable}"
      ;;
    0|false|FALSE|no|NO|off|OFF)
      echo "  Prewarmed cache bytes: skipped (set GENET_MEASURE_CACHE_BYTES=1 for a recursive scan)"
      ;;
    *) fail "GENET_MEASURE_CACHE_BYTES must be boolean" ;;
  esac
fi
echo "  Build-context root: ${BUILD_ROOT}"
if [[ -d "${BUILD_ROOT}" ]]; then
  df -h "${BUILD_ROOT}" || true
fi
echo "  DockerRootDir: ${docker_root:-unavailable}"
echo "  Docker storage driver: ${docker_driver:-unavailable}"
echo "  Host Python/Miniconda: not required; Python is supplied by the immutable training image"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader || true
fi
if command -v ibdev2netdev >/dev/null 2>&1; then
  ibdev2netdev || true
elif command -v rdma >/dev/null 2>&1; then
  rdma link || true
fi

for warning in ${warnings[@]+"${warnings[@]}"}; do
  echo "WARNING: ${warning}" >&2
done
if (( ${#failures[@]} > 0 )); then
  for failure in ${failures[@]+"${failures[@]}"}; do
    echo "ERROR: ${failure}" >&2
  done
  exit 1
fi

echo "Hyperbolic host checks passed"
