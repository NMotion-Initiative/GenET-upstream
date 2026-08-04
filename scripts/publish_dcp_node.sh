#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 4 ]]; then
  echo "Usage: $0 LOCAL_CHECKPOINT ARCHIVE_DEST ITERATION NODE_RANK" >&2
  echo "ARCHIVE_DEST may be /mounted/path or user@host:/path." >&2
  exit 2
fi

readonly LOCAL_CHECKPOINT="$1"
readonly ARCHIVE_DEST="$2"
readonly ITERATION="$3"
readonly NODE_RANK="$4"
readonly NODE_NAME="node_$(printf '%02d' "${NODE_RANK}")"
readonly TEMP_DIR="$(mktemp -d)"
readonly ITERATION_DIR="iter_${ITERATION}.incomplete"
trap 'rm -rf -- "${TEMP_DIR}"' EXIT

if [[ "${ARCHIVE_DEST}" == *:* ]]; then
  readonly ARCHIVE_HOST="${ARCHIVE_DEST%%:*}"
  readonly ARCHIVE_PATH="${ARCHIVE_DEST#*:}"
  if [[ -z "${ARCHIVE_HOST}" || "${ARCHIVE_PATH}" != /* || "${ARCHIVE_PATH}" =~ [[:space:]] ]]; then
    echo "Remote ARCHIVE_DEST must be host:/absolute/path without whitespace" >&2
    exit 2
  fi
  ssh "${ARCHIVE_HOST}" mkdir -p -- \
    "${ARCHIVE_PATH%/}/${ITERATION_DIR}/${NODE_NAME}/files"
else
  mkdir -p -- "${ARCHIVE_DEST%/}/${ITERATION_DIR}/${NODE_NAME}/files"
fi

python -m genet.cli.checkpoint manifest \
  --checkpoint-dir "${LOCAL_CHECKPOINT}" \
  --node-rank "${NODE_RANK}" \
  --output "${TEMP_DIR}/NODE_MANIFEST.json"

# A trailing slash copies the checkpoint contents beneath files/ without
# nesting the checkpoint directory name. rsync writes partial files atomically.
rsync -a --partial "${LOCAL_CHECKPOINT}/" \
  "${ARCHIVE_DEST%/}/${ITERATION_DIR}/${NODE_NAME}/files/"
rsync -a "${TEMP_DIR}/NODE_MANIFEST.json" \
  "${ARCHIVE_DEST%/}/${ITERATION_DIR}/${NODE_NAME}/NODE_MANIFEST.json"

printf 'node_rank=%s\n' "${NODE_RANK}" > "${TEMP_DIR}/NODE_DONE"
rsync -a "${TEMP_DIR}/NODE_DONE" \
  "${ARCHIVE_DEST%/}/${ITERATION_DIR}/${NODE_NAME}/NODE_DONE"

echo "Published node ${NODE_RANK} shards for iteration ${ITERATION}"
