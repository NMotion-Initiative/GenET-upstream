#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: commit_cluster_dcp_ssh.sh HOSTS IMAGE_REF OUTPUT_ROOT COMMITTED_ROOT

Discover the final DCP on every host, publish node manifests/shards onto the
local coordinator, consolidate atomically, and verify COMMITTED.

Environment:
  SSH_IDENTITY              default: /root/.ssh/id_cluster
  GENET_DCP_TMP_ROOT         default: /mnt/nvme/genet/.dcp-publish
EOF
  exit 2
}

[[ "$#" -eq 4 ]] || usage

readonly HOSTS_FILE="$1"
readonly IMAGE_REF="$2"
readonly OUTPUT_ROOT="$(realpath -m -- "$3")"
readonly COMMITTED_ROOT="$(realpath -m -- "$4")"
readonly SSH_IDENTITY="${SSH_IDENTITY:-/root/.ssh/id_cluster}"
readonly TMP_ROOT="${GENET_DCP_TMP_ROOT:-/mnt/nvme/genet/.dcp-publish}"
readonly RUN_ID="$(basename "${OUTPUT_ROOT}")"
readonly RSYNC_SHELL="ssh -i ${SSH_IDENTITY} -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=15"
readonly -a SSH=(ssh -i "${SSH_IDENTITY}" -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=15)

[[ -f "${HOSTS_FILE}" ]] || { echo "Host file does not exist: ${HOSTS_FILE}" >&2; exit 2; }
[[ "${IMAGE_REF}" =~ ^[^[:space:]]+@sha256:[0-9a-fA-F]{64}$ ]] \
  || { echo "IMAGE_REF must be immutable: repository@sha256:<digest>" >&2; exit 2; }
for unsafe in "/" "${HOME}" "$(pwd -P)" "$(realpath -m -- "$(dirname "${BASH_SOURCE[0]}")/..")"; do
  [[ "${COMMITTED_ROOT}" != "${unsafe}" ]] \
    || { echo "Refusing unsafe COMMITTED_ROOT: ${COMMITTED_ROOT}" >&2; exit 2; }
done
[[ "${COMMITTED_ROOT}" != "${OUTPUT_ROOT}" ]] \
  || { echo "COMMITTED_ROOT must differ from OUTPUT_ROOT" >&2; exit 2; }

mapfile -t HOSTS < <(
  awk '
    {
      sub(/[[:space:]]*#.*/, "")
      gsub(/^[[:space:]]+|[[:space:]]+$/, "")
      if (length($0)) print
    }
  ' "${HOSTS_FILE}"
)
(( ${#HOSTS[@]} > 0 )) || { echo "Host file contains no hosts" >&2; exit 2; }

discover_script='
set -euo pipefail
root="$1"
shopt -s globstar nullglob
latest=( "${root}"/**/checkpoints/latest_checkpoint.txt )
(( ${#latest[@]} == 1 )) || {
  echo "Expected exactly one latest_checkpoint.txt under ${root}, found ${#latest[@]}" >&2
  exit 2
}
name="$(<"${latest[0]}")"
[[ "${name}" =~ ^iter_[0-9]{9}$ ]] || {
  echo "Invalid latest checkpoint name: ${name}" >&2
  exit 2
}
checkpoint="$(dirname "${latest[0]}")/${name}"
[[ -d "${checkpoint}" ]] || { echo "Checkpoint missing: ${checkpoint}" >&2; exit 2; }
printf "%s\n%s\n" "${checkpoint#"${root}"/}" "${checkpoint}"
'

discover_checkpoint() {
  local target="$1"
  if [[ "${target}" == "local" ]]; then
    bash -c "${discover_script}" -- "${OUTPUT_ROOT}"
  else
    "${SSH[@]}" "${target}" bash -c "$(printf '%q' "${discover_script}")" -- "${OUTPUT_ROOT}"
  fi
}

reference_relative=""
for rank in "${!HOSTS[@]}"; do
  target="${HOSTS[rank]}"
  if [[ -z "${reference_relative}" ]]; then
    mapfile -t discovered < <(discover_checkpoint "${target}")
    (( ${#discovered[@]} == 2 )) || {
      echo "Could not discover checkpoint on coordinator ${target}" >&2
      exit 1
    }
    relative="${discovered[0]}"
    reference_relative="${relative}"
    continue
  fi
  checkpoint="${OUTPUT_ROOT}/${reference_relative}"
  verify_script='
set -euo pipefail
checkpoint="$1"
[[ -d "${checkpoint}" ]] || { echo "Checkpoint missing: ${checkpoint}" >&2; exit 1; }
'
  if [[ "${target}" == "local" ]]; then
    bash -c "${verify_script}" -- "${checkpoint}"
  else
    "${SSH[@]}" "${target}" bash -c "$(printf '%q' "${verify_script}")" -- "${checkpoint}"
  fi
done

readonly ITERATION="$(basename "${reference_relative}")"
readonly RUN_ARCHIVE="${COMMITTED_ROOT}/${RUN_ID}"
readonly INCOMPLETE="${RUN_ARCHIVE}/${ITERATION}.incomplete"
readonly FINAL="${RUN_ARCHIVE}/${ITERATION}"

mkdir -p -- "${RUN_ARCHIVE}"
if [[ -d "${FINAL}" ]]; then
  docker run --rm --entrypoint python \
    --mount "type=bind,src=${RUN_ARCHIVE},dst=${RUN_ARCHIVE},readonly" \
    "${IMAGE_REF}" -m genet.cli.checkpoint verify --checkpoint-dir "${FINAL}"
  echo "Committed checkpoint already verified: ${FINAL}"
  exit 0
fi
[[ ! -e "${FINAL}.consolidating" ]] || {
  echo "Forensic consolidation directory already exists: ${FINAL}.consolidating" >&2
  exit 1
}

for rank in "${!HOSTS[@]}"; do
  target="${HOSTS[rank]}"
  node_name="node_$(printf '%02d' "${rank}")"
  node_dir="${INCOMPLETE}/${node_name}"
  remote_tmp="${TMP_ROOT}/${RUN_ID}/${ITERATION}/${node_name}"
  checkpoint="${OUTPUT_ROOT}/${reference_relative}"
  mkdir -p -- "${node_dir}/files"

  container_script='
set -euo pipefail
image="$1"; checkpoint="$2"; temporary="$3"; node_rank="$4"
mkdir -p -- "${temporary}"
docker run --rm --entrypoint python \
  --mount "type=bind,src=${checkpoint},dst=${checkpoint},readonly" \
  --mount "type=bind,src=${temporary},dst=${temporary}" \
  "${image}" -m genet.cli.checkpoint manifest \
  --checkpoint-dir "${checkpoint}" \
  --node-rank "${node_rank}" \
  --output "${temporary}/NODE_MANIFEST.json"
'
  if [[ "${target}" == "local" ]]; then
    bash -c "${container_script}" -- "${IMAGE_REF}" "${checkpoint}" "${remote_tmp}" "${rank}"
    rsync -a --partial "${checkpoint}/" "${node_dir}/files/"
    rsync -a "${remote_tmp}/NODE_MANIFEST.json" "${node_dir}/NODE_MANIFEST.json"
    rm -rf -- "${remote_tmp}"
  else
    "${SSH[@]}" "${target}" bash -c "$(printf '%q' "${container_script}")" -- \
      "${IMAGE_REF}" "${checkpoint}" "${remote_tmp}" "${rank}"
    rsync -a --partial -e "${RSYNC_SHELL}" \
      "${target}:${checkpoint}/" "${node_dir}/files/"
    rsync -a -e "${RSYNC_SHELL}" \
      "${target}:${remote_tmp}/NODE_MANIFEST.json" "${node_dir}/NODE_MANIFEST.json"
    "${SSH[@]}" "${target}" rm -rf -- "${remote_tmp}"
  fi
  printf 'node_rank=%s\n' "${rank}" > "${node_dir}/NODE_DONE"
  echo "Published ${node_name} from ${target}"
done

docker run --rm --entrypoint python \
  --mount "type=bind,src=${RUN_ARCHIVE},dst=${RUN_ARCHIVE}" \
  "${IMAGE_REF}" -m genet.cli.checkpoint consolidate \
  --archive-dir "${INCOMPLETE}" \
  --output-dir "${FINAL}" \
  --expected-nodes "${#HOSTS[@]}"
docker run --rm --entrypoint python \
  --mount "type=bind,src=${RUN_ARCHIVE},dst=${RUN_ARCHIVE},readonly" \
  "${IMAGE_REF}" -m genet.cli.checkpoint verify --checkpoint-dir "${FINAL}"

echo "Committed checkpoint: ${FINAL}"
