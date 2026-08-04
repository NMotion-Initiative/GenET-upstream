#!/usr/bin/env bash
set -euo pipefail

readonly PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
readonly BUILD_CONTEXT_ROOT="${GENET_BUILD_CONTEXT_ROOT:-/dev/shm}"
: "${BASE_IMAGE:?Set BASE_IMAGE to the immutable NVIDIA Cosmos/PyTorch base image digest}"
: "${GENET_IMAGE_TAG:?Set GENET_IMAGE_TAG to the image name/tag to build}"
readonly COSMOS_DEPENDENCY_GROUP="${COSMOS_DEPENDENCY_GROUP:-cu130-train}"

if [[ ! "${BASE_IMAGE}" =~ @sha256:[0-9a-fA-F]{64}$ ]]; then
  echo "BASE_IMAGE must be pinned by registry digest: repository@sha256:<64 hex>" >&2
  exit 2
fi
case "${COSMOS_DEPENDENCY_GROUP}" in
  cu130-train|cu128-train) ;;
  *)
    echo "COSMOS_DEPENDENCY_GROUP must be cu130-train or cu128-train" >&2
    exit 2
    ;;
esac
if [[ ! -d "${BUILD_CONTEXT_ROOT}" || ! -w "${BUILD_CONTEXT_ROOT}" ]]; then
  echo "GENET_BUILD_CONTEXT_ROOT must be an existing writable directory: ${BUILD_CONTEXT_ROOT}" >&2
  exit 2
fi

cd "${PROJECT_ROOT}"
readonly GENET_REVISION="$(git rev-parse HEAD)"
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "Refusing to label a dirty tracked worktree as ${GENET_REVISION}" >&2
  exit 2
fi

readonly BUILD_WORKSPACE="$(mktemp -d "${BUILD_CONTEXT_ROOT%/}/genet-build.XXXXXX")"
trap 'rm -rf -- "${BUILD_WORKSPACE}"' EXIT
readonly CONTEXT_TAR="${BUILD_WORKSPACE}/context.tar"
readonly IMAGE_ID_FILE="${BUILD_WORKSPACE}/image.id"
export DOCKER_BUILDKIT=1

git archive --format=tar \
  --add-virtual-file="GENET_BUILD_REVISION:${GENET_REVISION}" \
  HEAD > "${CONTEXT_TAR}"

docker_root="$(docker info --format '{{.DockerRootDir}}')"
echo "Build context is staged at ${CONTEXT_TAR}"
echo "Docker image layers are stored by the daemon under ${docker_root}, not in ${BUILD_CONTEXT_ROOT}"
echo "Cosmos locked dependency group: ${COSMOS_DEPENDENCY_GROUP}"
if [[ "${docker_root}" != /mnt/nvme/* ]]; then
  echo "WARNING: DockerRootDir is not on /mnt/nvme; verify the daemon/containerd store has enough space" >&2
fi

docker build \
  --pull \
  --file containers/Dockerfile \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg "GENET_CODE_REVISION=${GENET_REVISION}" \
  --build-arg "COSMOS_DEPENDENCY_GROUP=${COSMOS_DEPENDENCY_GROUP}" \
  --iidfile "${IMAGE_ID_FILE}" \
  --tag "${GENET_IMAGE_TAG}" \
  - < "${CONTEXT_TAR}"

echo "Built ${GENET_IMAGE_TAG} at revision ${GENET_REVISION}"
echo "Local image ID: $(<"${IMAGE_ID_FILE}")"
echo "Push once, resolve the registry digest, and run that digest on every node"
