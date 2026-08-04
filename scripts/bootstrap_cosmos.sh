#!/usr/bin/env bash
set -euo pipefail

readonly COSMOS_REPOSITORY="https://github.com/NVIDIA/cosmos-framework.git"
readonly COSMOS_REVISION="a904d2d36b774a51dd06ff9ff906816b1a04f579"
readonly PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly COSMOS_DIR="${COSMOS_DIR:-${PROJECT_ROOT}/third_party/cosmos-framework}"

if [[ ! -d "${COSMOS_DIR}/.git" ]]; then
  mkdir -p "$(dirname "${COSMOS_DIR}")"
  git clone "${COSMOS_REPOSITORY}" "${COSMOS_DIR}"
fi

git -C "${COSMOS_DIR}" fetch --depth 1 origin "${COSMOS_REVISION}"
git -C "${COSMOS_DIR}" checkout --detach "${COSMOS_REVISION}"

actual_revision="$(git -C "${COSMOS_DIR}" rev-parse HEAD)"
if [[ "${actual_revision}" != "${COSMOS_REVISION}" ]]; then
  echo "Cosmos revision mismatch: ${actual_revision}" >&2
  exit 1
fi

echo "Pinned Cosmos Framework ready at ${COSMOS_DIR}"
echo "Install it in the NVIDIA Cosmos container with:"
echo "  python -m pip install -e '${COSMOS_DIR}[train]'"
echo "Then install GenET: python -m pip install -e '${PROJECT_ROOT}'"

