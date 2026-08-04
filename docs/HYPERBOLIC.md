# Hyperbolic 4×8-GPU Runbook

This is the end-to-end operating guide for running GenET on four Hyperbolic nodes with eight GPUs per node and no
shared training filesystem. It covers host preparation, an immutable Docker image, node-local RoboTwin v1 data,
environment and artifact verification, distributed launch, checkpoint transport, and long-horizon inference.

The known prewarmed RoboTwin cache is:

```text
/mnt/nvme/mds-cache/robotwin_v1
```

The currently observed roughly 41 GB per node is an operational inventory value, not a completeness criterion. Do not
hard-code it as a gate: the approved manifest plus content lock is authoritative. The host check skips an expensive
recursive size walk by default; set `GENET_MEASURE_CACHE_BYTES=1` only when that scan is intentional.

The host helpers keep writable run state under `/mnt/nvme/genet`. Do not infer from the common pathnames that the four
nodes see the same filesystem. Each node must have its own complete copies, and every training artifact copy must pass
the cluster-lock verification described below. Local NVMe may also disappear when a cloud instance is terminated;
publish checkpoints and inference journals to durable storage before releasing any node.

> **Current data boundary:** GenET consumes a `genet.raw-pair/v1` JSONL manifest. The final RoboTwin-v1-to-GenET manifest
> adapter is intentionally still a TODO until the project-specific dataschema defines action fields, units, coordinate
> frames, normalization, task IDs, and timestamps. The preprocessing commands below are executable after that manifest
> exists; they do not guess those semantics from the RoboTwin directory tree.

## 1. Deployment contract

Use one release identity for a job:

- one committed GenET revision;
- one image reference pinned by OCI digest, never only a mutable tag;
- the pinned Cosmos Framework revision in this repository;
- one fixed Hugging Face revision for `nvidia/Cosmos3-Edge`;
- one semantic training configuration;
- one cluster lock covering the processed data, Wan VAE, exact load checkpoint, normalization artifact when present,
  and the complete offline Hugging Face cache;
- one node-local verification receipt per node;
- one hostfile whose order defines `NODE_RANK=0,1,2,3`.

The cluster lock proves content equality. The receipt binds those verified bytes to the paths used at runtime. The
distributed preflight compares software, GPU, network-environment, code, image, lock, and receipt identities across
nodes. All three layers are required.

## 2. Recommended node-local layout

Keep immutable training inputs under the actual cache mount and mutable run state under a separate node-local root:

```text
/mnt/nvme/mds-cache/robotwin_v1/
  ...                         existing node-local RoboTwin v1 media
  genet/processed/train/      manifest.jsonl, index.json, stats.json, samples/*.npz

/mnt/nvme/genet/
  data/raw/
    train.jsonl               genet.raw-pair/v1; paths point into the prewarmed cache
  normalization/             optional train-split-only normalization artifacts
  artifacts/
    wan22_vae/Wan2.2_VAE.pth
  hf-cache/                   complete offline Hugging Face cache
  release/<release-id>/
    cluster-lock.json
    node-receipt.json         generated independently on each node
  checkpoints/
    Cosmos3-Edge/             converted base DCP
    <stage>/iter_N/           prestaged committed training DCPs
  outputs/                   node-local Cosmos training outputs and DCP shards
  inference/                 long-generation RUN.json and immutable chunk NPZs
```

Keep the existing RoboTwin layout in place and make `data/raw/train.jsonl` point at the real media with absolute paths.
Do not move or rewrite the original data merely to match this example.

For shell examples, set a task-specific variable rather than repurposing a system variable:

```bash
export GENET_NODE_CACHE=/mnt/nvme/mds-cache/robotwin_v1
export GENET_NODE_RUN_ROOT=/mnt/nvme/genet
export GENET_PROCESSED_DATA=/mnt/nvme/mds-cache/robotwin_v1/genet/processed/train
export GENET_MANIFEST="${GENET_PROCESSED_DATA}/manifest.jsonl"
```

## 3. Prepare the four hosts

Hyperbolic exposes SSH connection details for each allocated node. Put the corresponding SSH targets in a local
hostfile in deterministic rank order. Use SSH aliases when ports, users, or identity files differ; keep credentials out
of Git. The launcher accepts a parameterized hostfile, so this guide does not invent public IPs or hostnames.

On the release/coordinator host, verify non-interactive access to all four entries before doing anything expensive:

```bash
while IFS= read -r GENET_SSH_TARGET; do
  case "${GENET_SSH_TARGET}" in ''|'#'*) continue ;; esac
  if [[ "${GENET_SSH_TARGET}" == local ]]; then
    true
  else
    ssh -o BatchMode=yes "${GENET_SSH_TARGET}" true
  fi
done < /secure/path/genet-hosts.txt
```

On every GPU node, run the versioned host check first:

```bash
GENET_NODE_CACHE=/mnt/nvme/mds-cache/robotwin_v1 \
GENET_NVME_ROOT=/mnt/nvme \
GENET_EXPECTED_GPUS=8 \
bash scripts/check_hyperbolic_host.sh
```

Its inventory verifies the NVMe mount, the prewarmed cache, eight GPUs, Docker access, the NVIDIA runtime, RDMA character
devices, and required host transport tools. Use these read-only commands to discover the allocation-specific network:

```bash
nvidia-smi -L
docker version
docker info
df -h /mnt/nvme/mds-cache
test -d /mnt/nvme/mds-cache/robotwin_v1
ip -br address
ibdev2netdev
show_gids
rdma link
```

Expected invariants are eight visible GPUs, enough free local NVMe, a working NVIDIA container runtime, and mutually
reachable bootstrap and RDMA networks. Hyperbolic allocations can expose different network fabrics; use the actual
reservation details and command output. Do not copy `NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`, `NCCL_IB_HCA`, a GID
index, IP address, MTU, traffic class, or service level from this guide.

The four nodes must be able to reach rank 0's chosen `MASTER_ADDR` on both the training and preflight ports. Public SSH
addresses and the private cluster bootstrap address are separate concepts. Validate the actual RDMA path with a
multi-node `nccl-tests` run; the CPU/Gloo preflight does not replace it.

If Docker or the NVIDIA container runtime is missing, install it once from NVIDIA's and Docker's current official
instructions, then verify the intended base image with `docker run --rm --gpus all ... nvidia-smi`. Host Miniconda,
Python, a host CUDA toolkit, and a host source checkout are not part of the training environment: Python, the uv-managed
environment, CUDA user-space libraries, Cosmos, GenET, configs, and launch scripts live in the immutable image. The host
needs only the NVIDIA driver, NVIDIA container runtime, Docker, SSH/rsync, RDMA devices, and enough local storage.

Verify Docker's persistent storage before building a multi-gigabyte image:

```bash
docker info --format 'DockerRootDir={{.DockerRootDir}} Driver={{.Driver}}'
docker system df
df -h /mnt/nvme /var/lib/docker /var/lib/containerd 2>/dev/null || true
```

`GENET_BUILD_CONTEXT_ROOT=/dev/shm` moves only the temporary Git build context. It does **not** move pulled base layers,
BuildKit cache, image layers, or container writable layers. If the reported daemon/containerd storage is on a small root
disk, ask the node administrator to move it to a dedicated directory under `/mnt/nvme` before the first build. For the
classic Docker store this normally means merging a `"data-root": "/mnt/nvme/docker"` key into the existing
`/etc/docker/daemon.json` and restarting Docker; do not overwrite an existing daemon configuration. Newer Docker installs
using the containerd image store can keep image data under `/var/lib/containerd`, which must be relocated/configured
separately by the administrator. Re-run the commands above after any change. No Docker storage reconfiguration is needed
when the existing store already has adequate capacity.

## 4. Build and publish one immutable image

Build on one trusted release host from a committed tree with the versioned helper. It stages only the small Git archive
under `/dev/shm`; Docker/BuildKit layers still use the daemon's configured storage root, which the helper prints:

```bash
export GENET_REVISION="$(git rev-parse HEAD)"
export BASE_IMAGE_TAG='nvcr.io/nvidia/pytorch:26.06-py3'
docker pull "${BASE_IMAGE_TAG}"
export BASE_IMAGE="$(docker image inspect --format '{{index .RepoDigests 0}}' "${BASE_IMAGE_TAG}")"
[[ "${BASE_IMAGE}" =~ @sha256:[0-9a-f]{64}$ ]]
docker run --rm --gpus all "${BASE_IMAGE}" nvidia-smi

export GENET_IMAGE_TAG='registry.example/genet:'"${GENET_REVISION}"
export GENET_BUILD_CONTEXT_ROOT=/dev/shm
export COSMOS_DEPENDENCY_GROUP=cu130-train

bash scripts/build_hyperbolic_image.sh

docker push "${GENET_IMAGE_TAG}"
export GENET_IMAGE_REF="$(docker image inspect --format '{{index .RepoDigests 0}}' "${GENET_IMAGE_TAG}")"
[[ "${GENET_IMAGE_REF}" =~ @sha256:[0-9a-f]{64}$ ]]
```

The helper refuses dirty tracked source, builds from `git archive HEAD`, embeds the full Git revision, and requires the
base image to be pinned by digest. Untracked files are never included. Use `cu130-train` with the CUDA 13 / NGC PyTorch
26.06 base, or `cu128-train` with the CUDA 12.8 / NGC PyTorch 25.06 base; resolve the selected base tag to a digest before
building and confirm the node driver supports it.

The example selects the pinned digest corresponding to the official Cosmos-recommended CUDA 13 base. Use
`nvcr.io/nvidia/pytorch:25.06-py3` plus `cu128-train` when the installed driver supports CUDA 12.8 but not CUDA 13.
Record the digest printed by the registry and use the resolved `GENET_IMAGE_REF` as the only production reference:

```bash
export GENET_IMAGE_REF='registry.example/genet@sha256:<pushed-image-digest>'
```

Authenticate the four nodes to the registry using the provider's credential mechanism, pull that exact digest, and
confirm it is available. The SSH launcher can fan out the pull, but it must never convert `GENET_IMAGE_REF` back to a
tag. The image contains versioned `configs/` and `scripts/` under `/opt/genet`, the installed GenET wheel, the pinned
Cosmos checkout, and a sorted package inventory.

## 5. Prepare RoboTwin v1 once, then replicate the bytes

### 5.1 Create the raw pair manifest

After the project-specific dataschema adapter is added, scan the node-local cache and generate:

```text
/mnt/nvme/genet/data/raw/train.jsonl
```

Copy `configs/cluster/hyperbolic_4x8.env.example` to a private preparation-host file, replace its placeholders, add
`NODE_RANK=0`, set `GENET_CONTAINER_NAME=genet-preprocess`, and set `GENET_CACHE_READONLY=0`. The cache is writable only
while the designated preparation node creates the canonical processed tree; restore `GENET_CACHE_READONLY=1` before
validation locks or training. Inspect the temporary schema through that same digest-pinned container environment:

```bash
export GENET_PREP_HOST_ENV=/secure/path/preprocess.host.env
export GENET_IMAGE_REF='registry.example/genet@sha256:<pushed-image-digest>'

bash scripts/run_hyperbolic_container.sh \
  "${GENET_PREP_HOST_ENV}" \
  "${GENET_IMAGE_REF}" \
  genet-preprocess --print-raw-schema
```

Each JSONL record must contain paired `source` and `target_gt` streams. `reference_target` may be supplied explicitly or
selected deterministically from the same Target embodiment while excluding the `target_gt` episode. Task-level
reference exclusion is not implemented by v1 and must be enforced by the final manifest adapter.

### 5.2 Run canonical preprocessing

Use one designated preparation node to create the canonical processed copy on its local NVMe. With the private host
environment created in Section 5.1, run the command through the container helper:

```bash
export GENET_PREP_HOST_ENV=/secure/path/preprocess.host.env
export GENET_IMAGE_REF='registry.example/genet@sha256:<pushed-image-digest>'

bash scripts/run_hyperbolic_container.sh \
  "${GENET_PREP_HOST_ENV}" \
  "${GENET_IMAGE_REF}" \
  bash -lc '
    set -euo pipefail
    python -c "import av; from PIL import Image"
    genet-preprocess \
      --manifest /mnt/nvme/genet/data/raw/train.jsonl \
      --output /mnt/nvme/mds-cache/robotwin_v1/genet/processed/train \
      --config /opt/genet/configs/schema.example.json \
      --num-frames 81 \
      --sample-fps 16 \
      --height 192 \
      --width 320 \
      --action-dim 64 \
      --short-policy drop \
      --action-resample linear
    genet-validate-data \
      --manifest /mnt/nvme/mds-cache/robotwin_v1/genet/processed/train/manifest.jsonl \
      --num-frames 81 \
      --height 192 \
      --width 320 \
      --action-dim 64 \
      --cosmos
  '
```

The immutable image build validates the preprocessing imports (`av` and `Pillow`) before it can succeed. The explicit
import above fails early if a different image is selected accidentally; do not resolve dependencies separately on each
node.

The safest no-shared-storage workflow is to preprocess once and copy the resulting
`/mnt/nvme/mds-cache/robotwin_v1/genet/processed/train` directory byte-for-byte to the other three nodes. Independently
preprocessing all four raw copies is permitted only if every result later matches the same content lock. Do not train
from a copy that merely has the same sample count.

### 5.3 Stage model artifacts

On the same canonical preparation node:

1. pin and download the accepted `nvidia/Cosmos3-Edge` revision into
   `${GENET_NODE_RUN_ROOT}/hf-cache`;
2. pin and materialize `Wan2.2_VAE.pth` at
   `${GENET_NODE_RUN_ROOT}/artifacts/wan22_vae/Wan2.2_VAE.pth`;
3. convert the official Cosmos3-Edge checkpoint to a DCP at
   `${GENET_NODE_RUN_ROOT}/checkpoints/Cosmos3-Edge` using the pinned Cosmos conversion entry point;
4. write train-split-only normalization artifacts under `${GENET_NODE_RUN_ROOT}/normalization` when the final dataschema
   provides them.

Resolve and approve the full Cosmos3-Edge repository commit while the preparation node is online; do not use `main` as
the production identity. The following private staging environment must set `NODE_RANK=0`, a unique container name,
`HF_HOME=/mnt/nvme/genet/hf-cache`, and `GENET_ALLOW_HF_TOKEN=1`. Add `HF_TOKEN` only if the account requires one; Docker
can inspect that environment variable while the short-lived preparation container exists, so never reuse this private
environment for training:

```bash
export GENET_HF_SNAPSHOT_REVISION="$(
  bash scripts/run_hyperbolic_container.sh \
    /secure/path/download.host.env \
    "${GENET_IMAGE_REF}" \
    python -c 'from huggingface_hub import HfApi; print(HfApi().model_info("nvidia/Cosmos3-Edge", revision="main").sha)'
)"

[[ "${GENET_HF_SNAPSHOT_REVISION}" =~ ^[0-9a-f]{40}$ ]]

bash scripts/run_hyperbolic_container.sh \
  /secure/path/download.host.env \
  "${GENET_IMAGE_REF}" \
  bash -c '
    set -euo pipefail
    revision="$1"
    hf download nvidia/Cosmos3-Edge --revision "${revision}"
    ref_dir="${HF_HOME}/hub/models--nvidia--Cosmos3-Edge/refs"
    mkdir -p "${ref_dir}"
    printf "%s\n" "${revision}" > "${ref_dir}/main"
    test "$(cat "${ref_dir}/main")" = "${revision}"
  ' bash "${GENET_HF_SNAPSHOT_REVISION}"

bash scripts/run_hyperbolic_container.sh \
  /secure/path/download.host.env \
  "${GENET_IMAGE_REF}" \
  hf download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth \
    --revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e \
    --local-dir /mnt/nvme/genet/artifacts/wan22_vae
```

The explicit `refs/main` file makes every later offline Cosmos lookup of the registered `Cosmos3-Edge` name resolve to
the approved snapshot rather than a mutable network ref. Record `GENET_HF_SNAPSHOT_REVISION` in every private job
environment. Inject download credentials only into the one preparation container, never into the shared production job
environment. After staging is complete, production uses:

```bash
export HF_HOME="${GENET_NODE_RUN_ROOT}/hf-cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WAN_VAE_PATH="${GENET_NODE_RUN_ROOT}/artifacts/wan22_vae/Wan2.2_VAE.pth"
```

Convert the exact local snapshot with the pinned Cosmos entry point inside the immutable container. Use another private
host environment copy with `GENET_CONTAINER_NAME=genet-convert`; its `HF_HOME` and
`GENET_HF_SNAPSHOT_REVISION` must point at the already populated cache and approved commit:

```bash
bash scripts/run_hyperbolic_container.sh \
  /secure/path/convert.host.env \
  "${GENET_IMAGE_REF}" \
  bash -lc '
    set -euo pipefail
    GENET_EDGE_SNAPSHOT="${HF_HOME}/hub/models--nvidia--Cosmos3-Edge/snapshots/${GENET_HF_SNAPSHOT_REVISION}"
    test -f "${GENET_EDGE_SNAPSHOT}/config.json"
    python -m cosmos_framework.scripts.convert_model_to_dcp \
      -o /mnt/nvme/genet/checkpoints/Cosmos3-Edge \
      --checkpoint-path "${GENET_EDGE_SNAPSHOT}"
  '
```

The conversion command is the interface for the repository's pinned Cosmos revision. Do not substitute a standalone
PyTorch checkpoint or a partially downloaded model directory for the resulting DCP.

For S1, the `training_checkpoint` artifact is the converted Cosmos3-Edge base DCP. For S2/S3 it is the exact committed
checkpoint selected from the previous stage. For an exact resume it is the complete committed DCP being resumed. Never
reuse an old lock after changing the load checkpoint.

### 5.4 Replicate to every node

Use `rsync`, a durable object store, or Hyperbolic's stage-in facilities to copy these exact trees to the same node-local
root on all four hosts:

- `/mnt/nvme/mds-cache/robotwin_v1/genet/processed/train`;
- `normalization`, if present;
- `artifacts/wan22_vae`;
- the exact base/warm-start/resume DCP;
- `hf-cache`.

Copy into dedicated destinations without deleting unrelated files. A successful copy is not accepted until the
cluster-lock verification succeeds on that node.

When the preparation node is rank 0 and the coordinator, this non-destructive pattern copies the canonical processed
tree to ranks 1–3. Run analogous `rsync -aH --partial` commands for the model/cache trees listed above; deliberately omit
`--delete`:

```bash
while IFS= read -r GENET_SSH_TARGET; do
  case "${GENET_SSH_TARGET}" in ''|'#'*|local) continue ;; esac
  ssh "${GENET_SSH_TARGET}" \
    'mkdir -p /mnt/nvme/mds-cache/robotwin_v1/genet/processed/train'
  rsync -aH --partial --info=progress2 \
    /mnt/nvme/mds-cache/robotwin_v1/genet/processed/train/ \
    "${GENET_SSH_TARGET}:/mnt/nvme/mds-cache/robotwin_v1/genet/processed/train/"
done < /secure/path/genet-hosts.txt
```

## 6. Create the release lock and node receipts

Create a distinct release directory and lock for each job input set. Run the lock tool through the same immutable image;
the private host environment used here should set `GENET_CONTAINER_NAME=genet-lock`. The following is an S1 example:

```bash
export GENET_NODE_RUN_ROOT=/mnt/nvme/genet
export GENET_PROCESSED_DATA=/mnt/nvme/mds-cache/robotwin_v1/genet/processed/train
export GENET_IMAGE_REF='registry.example/genet@sha256:<pushed-image-digest>'
export GENET_RELEASE_ID="s1-${GENET_REVISION}"
export GENET_RELEASE_DIR="${GENET_NODE_RUN_ROOT}/release/${GENET_RELEASE_ID}"

bash scripts/run_hyperbolic_container.sh \
  /secure/path/lock.host.env \
  "${GENET_IMAGE_REF}" \
  bash -lc '
    set -euo pipefail
    genet-cluster-lock create \
      --output "${GENET_RELEASE_DIR}/cluster-lock.json" \
      --artifact processed_data="${GENET_PROCESSED_DATA}" \
      --artifact wan_vae="${GENET_NODE_RUN_ROOT}/artifacts/wan22_vae/Wan2.2_VAE.pth" \
      --artifact training_checkpoint="${GENET_NODE_RUN_ROOT}/checkpoints/Cosmos3-Edge" \
      --artifact hf_cache="${GENET_NODE_RUN_ROOT}/hf-cache"
  '
```

Add `--artifact normalization="${GENET_NODE_RUN_ROOT}/normalization"` to both lock commands once that artifact exists.
Copy `cluster-lock.json` to the identical release directory on every node. Then verify locally on every node and create
that node's receipt:

```bash
bash scripts/run_hyperbolic_container.sh \
  /secure/path/lock.host.env \
  "${GENET_IMAGE_REF}" \
  bash -lc '
    set -euo pipefail
    genet-cluster-lock verify \
      --lock "${GENET_RELEASE_DIR}/cluster-lock.json" \
      --receipt "${GENET_RELEASE_DIR}/node-receipt.json" \
      --artifact processed_data="${GENET_PROCESSED_DATA}" \
      --artifact wan_vae="${GENET_NODE_RUN_ROOT}/artifacts/wan22_vae/Wan2.2_VAE.pth" \
      --artifact training_checkpoint="${GENET_NODE_RUN_ROOT}/checkpoints/Cosmos3-Edge" \
      --artifact hf_cache="${GENET_NODE_RUN_ROOT}/hf-cache"
  '
```

Do not copy a receipt from another node. It is deliberately regenerated from that node's verified local paths. Keep the
verified artifact directories read-only for the lifetime of the job.

## 7. Create the hostfile and environment file

The hostfile has exactly one SSH target per non-comment line. Its order assigns `NODE_RANK=0,1,2,3`:

```text
# rank 0
<ssh-target-0>
<ssh-target-1>
<ssh-target-2>
<ssh-target-3>
```

Targets must be unique. Each target is interpreted by the coordinator's SSH configuration and may therefore be an
alias; use the literal `local` only when rank 0 is the coordinator itself. Set `MASTER_ADDR` separately to rank 0's
private address visible from every training node. It is usually not the public SSH address or SSH alias.

Copy the versioned shared environment and hostfile templates to a private job directory on the coordinator host, then
edit the copies:

```bash
cp configs/cluster/hyperbolic_4x8.env.example /secure/path/s1.env
cp configs/cluster/hyperbolic_4x8.hosts.example /secure/path/genet-hosts.txt
```

The image reference is a separate required launcher argument so it cannot be hidden or accidentally overridden by the
environment file. Keep `NODE_RANK` and container names out of `s1.env`; the launcher assigns both. A private copy of the
same shell template can be used with `scripts/run_hyperbolic_container.sh` for one-off preparation tasks after adding
`NODE_RANK=0` and a unique `GENET_CONTAINER_NAME`.

Required job-specific values include:

```bash
export NNODES=4
export NPROC_PER_NODE=8
export MASTER_ADDR='<rank-0-private-address>'
export MASTER_PORT='<unique-training-port>'
export PREFLIGHT_PORT='<unique-preflight-port>'
export GENET_PREFLIGHT_TIMEOUT_SECONDS=120

export GENET_STRICT_ENV=1
export GENET_CODE_REVISION='<full-40-character-GenET-commit>'
export GENET_COSMOS_REVISION='a904d2d36b774a51dd06ff9ff906816b1a04f579'
export GENET_CLUSTER_LOCK=/mnt/nvme/genet/release/<release-id>/cluster-lock.json
export GENET_CLUSTER_RECEIPT=/mnt/nvme/genet/release/<release-id>/node-receipt.json
export GENET_HF_SNAPSHOT_REVISION='<full-40-character-HF-repository-commit>'

export GENET_DATA_ARTIFACT=processed_data
export GENET_WAN_VAE_ARTIFACT=wan_vae
export GENET_CHECKPOINT_ARTIFACT=training_checkpoint
export GENET_HF_ARTIFACT=hf_cache
export GENET_PROCESSED_DATA=/mnt/nvme/mds-cache/robotwin_v1/genet/processed/train
export GENET_MANIFEST=/mnt/nvme/mds-cache/robotwin_v1/genet/processed/train/manifest.jsonl
export BASE_CHECKPOINT_PATH=/mnt/nvme/genet/checkpoints/Cosmos3-Edge
export WAN_VAE_PATH=/mnt/nvme/genet/artifacts/wan22_vae/Wan2.2_VAE.pth
export HF_HOME=/mnt/nvme/genet/hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export GENET_NVME_ROOT=/mnt/nvme
export GENET_NODE_CACHE=/mnt/nvme/mds-cache/robotwin_v1
export GENET_NODE_RUN_ROOT=/mnt/nvme/genet
export GENET_EXPECTED_GPUS=8
export GENET_CACHE_READONLY=1
export GENET_IMAGE_PULL_POLICY=missing
```

Also set the network variables after discovery on the real allocation. All ranks compare these values, so choose an
interface/HCA expression valid on every node. Keep `NODE_RANK` out of the shared environment file; the SSH launcher
assigns it from hostfile order.

The hostfile and environment file are separate inputs:

- the hostfile contains SSH transport identities in rank order;
- the environment file contains public distributed/runtime values and node-local artifact paths;
- neither file contains registry passwords, Hugging Face tokens, or private SSH keys.

## 8. One-command SSH launch

The versioned coordinator entry point is `scripts/launch_cluster_ssh.sh`. Run it from a trusted machine that can SSH to
all hostfile entries. It validates the ordered targets and node count, streams the versioned host/container helpers over
SSH, assigns deterministic node ranks, starts the same digest-pinned image on every node, saves one coordinator-side log
per rank, waits for the complete group, and returns failure if any node fails. It also attempts to clean up only its own
uniquely labelled containers when a grouped phase fails or the coordinator receives a termination signal. The launcher
requires a new log directory for every invocation and refuses `HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN` in the training
environment.

The exact launcher syntax is documented by:

```bash
bash scripts/launch_cluster_ssh.sh --help
```

Use the same hostfile, environment, image digest, and config for all phases. Run the read-only host checks and Gloo
preflight alone first:

```bash
bash scripts/launch_cluster_ssh.sh \
  --hosts /secure/path/genet-hosts.txt \
  --env /secure/path/s1.env \
  --image "${GENET_IMAGE_REF}" \
  --config configs/experiments/stage1_control_32gpu.yaml \
  --run-id s1-control-preflight \
  --preflight-only
```

After the provider-appropriate `nccl-tests` job passes, start the default guarded sequence. It performs host checks,
pulls or verifies the image concurrently on all nodes before any rendezvous, runs the Gloo preflight, runs a 32-rank
GenET dry run, and only then starts the real training command:

```bash
bash scripts/launch_cluster_ssh.sh \
  --hosts /secure/path/genet-hosts.txt \
  --env /secure/path/s1.env \
  --image "${GENET_IMAGE_REF}" \
  --config configs/experiments/stage1_control_32gpu.yaml \
  --run-id s1-control-seed42 \
  -- \
  --manifest /mnt/nvme/mds-cache/robotwin_v1/genet/processed/train/manifest.jsonl \
  --warm-start /mnt/nvme/genet/checkpoints/Cosmos3-Edge \
  --output-dir /mnt/nvme/genet/outputs/s1-control-seed42
```

Do not add `--dry-run` after `--`; the coordinator owns that guard. Use `--dry-run-only` when deliberately stopping after
the distributed dry run. The `--skip-host-check`, `--skip-preflight`, and `--skip-dry-run` switches exist for controlled
diagnostics, not for bypassing a failed production gate. Keep the coordinator alive in `tmux` or `systemd`: closing its
attached SSH sessions stops the job.

The required sequence is:

1. pull/verify the immutable image on all nodes;
2. verify each node's cluster lock and receipt;
3. run the four-node CPU/Gloo preflight;
4. run the provider-appropriate multi-node `nccl-tests` job;
5. let the coordinator run a 32-rank GenET dry run;
6. let the same guarded invocation proceed to training.

Do not launch training if the lock, receipt, Gloo preflight, or NCCL transport test fails. Do not background four
independent SSH commands manually and then accept only rank 0's exit status.

## 9. Training stages

The submitted configurations keep effective batch size comparable across the main stages:

| Stage | Configuration | Topology | Load mode | Purpose |
| --- | --- | --- | --- | --- |
| S1 | `configs/experiments/stage1_control_32gpu.yaml` | 4×8 GPUs | warm-start base DCP | train Source Control |
| S2 no-ref | `configs/experiments/stage2_no_ref_8gpu.yaml` | 1×8 GPUs | warm-start S1 | reference-disabled control |
| S2 shared | `configs/experiments/stage2_shared_8gpu.yaml` | 1×8 GPUs | warm-start S1 | shared K/V projection |
| S2 dual | `configs/experiments/stage2_dual_8gpu.yaml` | 1×8 GPUs | warm-start S1 | independent K/V projections |
| S3 shared | `configs/experiments/stage3_shared_32gpu.yaml` | 4×8 GPUs | warm-start S2 shared | joint confirmation |
| S3 dual | `configs/experiments/stage3_dual_32gpu.yaml` | 4×8 GPUs | warm-start S2 dual | joint confirmation |

For each stage:

1. prestage the complete load DCP to every participating node;
2. create a new cluster lock whose `training_checkpoint` is that exact DCP;
3. regenerate each node receipt;
4. create a new, empty output root and unique ports;
5. invoke the default coordinator sequence, which runs the distributed dry run first;
6. let the coordinator start the real job only after that dry run passes.

S1/S3 use 8-way FSDP sharding × 4-way replication. S2 uses one eight-GPU node with 8-way sharding and gradient
accumulation 4. The default effective global batch is therefore 32 in both cases. Run S2 variants with the same committed
S1 checkpoint, processed data, reference selection, seed, and sample budget. `stage2_dual_8gpu.yaml` initializes route B
from route A for a controlled shared-to-dual branch point.

An example S1 training payload passed through the SSH launcher is equivalent to this per-node command:

```bash
bash scripts/launch_roce.sh \
  configs/experiments/stage1_control_32gpu.yaml \
  --manifest /mnt/nvme/mds-cache/robotwin_v1/genet/processed/train/manifest.jsonl \
  --warm-start /mnt/nvme/genet/checkpoints/Cosmos3-Edge \
  --output-dir /mnt/nvme/genet/outputs/s1-control-seed42 \
  --dry-run
```

Run each S2 branch on a one-line hostfile with `NNODES=1` in its environment file. All three branches load the same
committed S1 DCP; only the experiment config and output root change. For example:

```bash
bash scripts/launch_cluster_ssh.sh \
  --hosts /secure/path/genet-s2-node0.txt \
  --env /secure/path/s2-shared.env \
  --image "${GENET_IMAGE_REF}" \
  --config configs/experiments/stage2_shared_8gpu.yaml \
  --run-id s2-shared-seed42 \
  -- \
  --manifest /mnt/nvme/mds-cache/robotwin_v1/genet/processed/train/manifest.jsonl \
  --warm-start /mnt/nvme/genet/checkpoints/s1/iter_N \
  --output-dir /mnt/nvme/genet/outputs/s2-shared-seed42
```

Use `stage2_no_ref_8gpu.yaml` and `stage2_dual_8gpu.yaml` with distinct output roots for the other branches. After
publishing and consolidating S2, a shared S3 confirmation is:

```bash
bash scripts/launch_cluster_ssh.sh \
  --hosts /secure/path/genet-hosts.txt \
  --env /secure/path/s3-shared.env \
  --image "${GENET_IMAGE_REF}" \
  --config configs/experiments/stage3_shared_32gpu.yaml \
  --run-id s3-shared-seed42 \
  -- \
  --manifest /mnt/nvme/mds-cache/robotwin_v1/genet/processed/train/manifest.jsonl \
  --warm-start /mnt/nvme/genet/checkpoints/s2-shared/iter_N \
  --output-dir /mnt/nvme/genet/outputs/s3-shared-seed42
```

The dual confirmation uses `stage3_dual_32gpu.yaml` and the committed S2 dual DCP. Each default launcher invocation
performs its dry run and real run as separate container phases, with separate per-rank logs.

The real Cosmos run directory is nested below the supplied output root at `genet/cross_embodiment/run`. Remove
the coordinator's dry-run gate only for controlled diagnostics. That gate initializes the full distributed group,
checks environment and artifacts, validates every node-local dataset copy, builds the upstream experiment, and loads one
rank-local sample contract.

Use warm-start when changing stage, topology, or optimizer parameter groups. Use `--resume` only for the same stage,
world size, parallel topology, data order, and optimizer contract. A resume attempt must use a new empty output root but
the same semantic configuration and a fully prestaged committed DCP.

## 10. Checkpoints without shared storage

During a 32-rank save, each node initially owns only its local DCP shards; node 0 also owns coordinator metadata. A
same-named directory on all four nodes is not four complete checkpoints.

After the upstream save barrier, run one publisher per physical node:

```bash
bash scripts/publish_dcp_node.sh \
  <node-local-DCP-directory> \
  <archive-user@archive-host:/absolute/experiment-path> \
  <zero-padded-iteration> \
  "${NODE_RANK}"
```

The publisher creates a per-node manifest, uploads files into `iter_<N>.incomplete/node_XX/files/`, and writes
`NODE_DONE` last. The provided publisher supports a mounted path or an SSH/rsync archive target; use a separate, audited
stage-out adapter for an object-store URI.

On the archive coordinator, consolidate only after all expected node uploads exist:

```bash
python -m genet.cli.checkpoint consolidate \
  --archive-dir <archive-root>/iter_<N>.incomplete \
  --output-dir <archive-root>/iter_<N> \
  --expected-nodes 4

genet-checkpoint verify \
  --checkpoint-dir <archive-root>/iter_<N>
```

For an S2 single-node job, use logical node rank 0 and `--expected-nodes 1`. Never point `latest` at `.incomplete`; only
the consolidated directory with valid `MANIFEST.json` and `COMMITTED` may be used.

Before warm-start or resume, copy the complete committed DCP to every participating node:

```bash
bash scripts/prestage_dcp.sh \
  <archive-user@archive-host:/absolute/experiment-path/iter_N> \
  /mnt/nvme/genet/checkpoints/<stage>/iter_N
```

Then rebuild the job's cluster lock against that destination and regenerate receipts. Keep the prior local output until
archive consolidation, verification, and at least one resume dry run have all succeeded.

## 11. Inference and long video–action generation

Prefer one eight-GPU node for one long-generation run when the model fits; this avoids cross-node solver collectives and
makes the journal owner unambiguous. Prestage the selected committed checkpoint and freeze the Source, Reference,
normalization, model, code, solver, and seed identities in the factory's `RunIdentity`.

The current CLI requires a project-specific factory that supplies the real model, Source/Reference readers, action
semantics, and quality evaluator. Build that adapter into the immutable image. Copy the per-node host template to
`/secure/path/inference.host.env`, set rank 0, a unique container name, and the same node-local roots, then validate the
configuration without importing it:

```bash
bash scripts/run_hyperbolic_container.sh \
  /secure/path/inference.host.env \
  "${GENET_IMAGE_REF}" \
  genet-generate-long \
  --config /opt/genet/configs/inference/long_video_action.yaml \
  --factory my_robot.genet_factory:create_runner \
  --output /mnt/nvme/genet/inference/pick_place_00042 \
  --dry-run
```

Before a long job, run a real two-window smoke test and inspect video/action prefix equality, timestamps, seam metrics,
and recovery. Start a new run with `--no-resume`:

```bash
bash scripts/run_hyperbolic_container.sh \
  /secure/path/inference.host.env \
  "${GENET_IMAGE_REF}" \
  genet-generate-long \
  --config /opt/genet/configs/inference/long_video_action.yaml \
  --factory my_robot.genet_factory:create_runner \
  --output /mnt/nvme/genet/inference/pick_place_00042 \
  --no-resume
```

Recover an interrupted compatible run with the identical config, factory identity, Source length, and output root:

```bash
bash scripts/run_hyperbolic_container.sh \
  /secure/path/inference.host.env \
  "${GENET_IMAGE_REF}" \
  genet-generate-long \
  --config /opt/genet/configs/inference/long_video_action.yaml \
  --factory my_robot.genet_factory:create_runner \
  --output /mnt/nvme/genet/inference/pick_place_00042 \
  --resume
```

Default long generation uses `T=81`, overlap `O=17`, and stride `S=64`. Each later window receives the last five clean
Wan video-latent tokens and 17 clean action steps from the checksum-verified accepted Target context. Video and action
are one transaction: a candidate is accepted, retried, committed, or rolled back jointly.

The default recovery budget is three attempts per window, one rollback-eligible accepted contribution, one contribution
rolled back at a time, and eight total rollbacks. Context is reconstructed only from immutable NPZ chunks referenced by
`RUN.json`; it is never reconstructed from a lossy preview video. The current CLI does not automatically encode a final
MP4, upload the journal, or garbage-collect superseded chunks.

Because the inference directory is node-local, continuously stage `RUN.json`, `RUN.lock` policy metadata, and immutable
`chunks/*.npz` to durable storage without modifying the live files. Only the coordinator writes `RUN.json`. If rank 0
moves to another physical node for resume, copy the complete output directory to that node first. Never let multiple
nodes independently write the same logical run journal.

Generated actions remain offline/simulator outputs until the final dataschema and embodiment-specific safety gates are
implemented. Do not send them directly to a real robot.

## 12. Stop/go checklist

Do not start paid production training until every item is true:

- [ ] hostfile contains exactly the intended physical nodes in deterministic rank order;
- [ ] every node exposes eight GPUs, contains the prewarmed RoboTwin cache, and has enough free space under `/mnt/nvme`;
- [ ] every node pulled the same OCI digest and the embedded GenET revision matches;
- [ ] the raw RoboTwin manifest was generated by the approved dataschema adapter;
- [ ] canonical preprocessing and `genet-validate-data --cosmos` passed;
- [ ] every node verified the same cluster lock and created its own receipt;
- [ ] the receipt binds the actual manifest, Wan VAE, HF cache, and exact load DCP paths;
- [ ] the four-node CPU/Gloo preflight passed with unique node identities;
- [ ] multi-node `nccl-tests` confirmed the intended RDMA transport rather than socket fallback;
- [ ] the 32-rank GenET dry run passed;
- [ ] checkpoint archive publication, consolidation, verification, and prestage were rehearsed;
- [ ] training and preflight ports are unique to this active job;
- [ ] the output root is new and node-local;
- [ ] long inference passed a real two-window clean-prefix and rollback smoke test;
- [ ] durable stage-out is active before any Hyperbolic node can be terminated.

For model/data internals, see [Architecture](ARCHITECTURE.md) and [Preprocessing](PREPROCESSING.md). For the detailed
training and checkpoint contracts, see [Training](TRAINING.md). For rolling context, retry, rollback, and journal
semantics, see [Long-horizon inference](INFERENCE.md).

## 13. Operational references

- [Pinned Cosmos Framework setup guide](https://github.com/NVIDIA/cosmos-framework/blob/a904d2d36b774a51dd06ff9ff906816b1a04f579/docs/setup.md)
- [Pinned Cosmos Framework training guide](https://github.com/NVIDIA/cosmos-framework/blob/a904d2d36b774a51dd06ff9ff906816b1a04f579/docs/training.md)
- [NVIDIA Container Toolkit installation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- [NVIDIA Container Toolkit GPU smoke test](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/sample-workload.html)
- [Docker build contexts](https://docs.docker.com/build/concepts/context/)
- [Docker daemon data directory](https://docs.docker.com/engine/daemon/)
- [NCCL user guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/)
- [Hyperbolic on-demand quick start](https://www.hyperbolic.ai/docs/on-demand/quickstart)
