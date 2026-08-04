#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 ARCHIVE_CHECKPOINT LOCAL_DESTINATION" >&2
  exit 2
fi

readonly ARCHIVE_CHECKPOINT="$1"
readonly LOCAL_DESTINATION="$2"
readonly CANONICAL_DESTINATION="$(realpath -m -- "${LOCAL_DESTINATION}")"
readonly CANONICAL_HOME="$(realpath -m -- "${HOME}")"
readonly PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
readonly CURRENT_DIRECTORY="$(pwd -P)"
readonly TARGET_MARKER=".genet-prestage-target"

if [[ -z "${LOCAL_DESTINATION}" \
  || "${CANONICAL_DESTINATION}" == "/" \
  || "${CANONICAL_DESTINATION}" == "${CANONICAL_HOME}" \
  || "${CANONICAL_DESTINATION}" == "${PROJECT_ROOT}" \
  || "${CANONICAL_DESTINATION}" == "${CURRENT_DIRECTORY}" ]]; then
  echo "Refusing unsafe LOCAL_DESTINATION: ${CANONICAL_DESTINATION}" >&2
  exit 2
fi

if [[ "${ARCHIVE_CHECKPOINT}" != *:* ]]; then
  python -m genet.cli.checkpoint verify --checkpoint-dir "${ARCHIVE_CHECKPOINT}"
fi
mkdir -p -- "${CANONICAL_DESTINATION}"
shopt -s nullglob dotglob
existing_entries=("${CANONICAL_DESTINATION}"/*)
shopt -u nullglob dotglob
if (( ${#existing_entries[@]} > 0 )) \
  && [[ ! -f "${CANONICAL_DESTINATION}/${TARGET_MARKER}" ]] \
  && ! ([[ -f "${CANONICAL_DESTINATION}/MANIFEST.json" ]] \
    && [[ -f "${CANONICAL_DESTINATION}/COMMITTED" ]]); then
  echo "Refusing to synchronize into a non-checkpoint directory: ${CANONICAL_DESTINATION}" >&2
  exit 2
fi
touch "${CANONICAL_DESTINATION}/${TARGET_MARKER}"
rsync -a --delete-delay --exclude="/${TARGET_MARKER}" \
  "${ARCHIVE_CHECKPOINT%/}/" "${CANONICAL_DESTINATION%/}/"
python -m genet.cli.checkpoint verify --checkpoint-dir "${CANONICAL_DESTINATION}"
