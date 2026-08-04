#!/usr/bin/env bash
set -euo pipefail

if (( $# != 2 )); then
  echo "Usage: $0 REPOSITORY@sha256:DIGEST always|missing|never" >&2
  exit 2
fi

readonly IMAGE_REF="$1"
readonly PULL_POLICY="$2"

[[ "${IMAGE_REF}" =~ ^[^[:space:]]+@sha256:[0-9a-fA-F]{64}$ ]] || {
  echo "IMAGE_REF must use repository@sha256:<64 hex>" >&2
  exit 2
}

case "${PULL_POLICY}" in
  always)
    docker pull "${IMAGE_REF}"
    ;;
  missing)
    if ! docker image inspect "${IMAGE_REF}" >/dev/null 2>&1; then
      docker pull "${IMAGE_REF}"
    fi
    ;;
  never)
    docker image inspect "${IMAGE_REF}" >/dev/null
    ;;
  *)
    echo "pull policy must be always, missing, or never" >&2
    exit 2
    ;;
esac

readonly RESOLVED_ID="$(docker image inspect --format '{{.Id}}' "${IMAGE_REF}")"
[[ "${RESOLVED_ID}" =~ ^sha256:[0-9a-fA-F]{64}$ ]] || {
  echo "Docker returned an invalid local image ID for ${IMAGE_REF}: ${RESOLVED_ID}" >&2
  exit 1
}
echo "Image ready: ${IMAGE_REF} (${RESOLVED_ID})"
