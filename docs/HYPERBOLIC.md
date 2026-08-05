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

> **Current data boundary:** the fixed RoboTwin-v1 MDS adapter is implemented and directly validates the supplied
> manifest, columns, action widths, aggregate counts, episodes, pairing keys, and reference policy. The MDS `state` is
> intentionally labeled measured `joint_position_state`; physical timestamps, units/frames, train-only normalization,
> and executable command semantics are not present in the source schema and remain explicit downstream decisions.

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

Keep the existing RoboTwin MDS layout in place. The direct adapter reads it in situ and writes only the processed GenET
tree shown above; it does not require an intermediate media export or raw-pair JSONL.

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

docker run --rm --gpus all "${GENET_IMAGE_TAG}" \
  python -c 'import cosmos_framework, genet, numpy, torch; assert numpy.__version__ == "2.2.6"; print(torch.__version__, torch.version.cuda)'
docker run --rm --gpus all "${GENET_IMAGE_TAG}" \
  /opt/genet-preprocess-venv/bin/python -c 'import genet, numpy, streaming; assert tuple(map(int, numpy.__version__.split(".")[:2])) < (2, 2)'
docker image inspect --format 'bytes={{.Size}}' "${GENET_IMAGE_TAG}"
docker history "${GENET_IMAGE_TAG}"

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
Cosmos checkout, and package inventories for both Python environments. Its default PATH is the frozen Cosmos training
venv. RoboTwin preprocessing runs from `/opt/genet-preprocess-venv`; this separation prevents Streaming's NumPy
constraint from changing the Cosmos lock.

## 5. Prepare RoboTwin v1 once, then replicate the bytes

### 5.1 Confirm the fixed MDS contract and semantic choices

The repository now consumes the probed RoboTwin-v1 MDS schema directly through
`configs/data/robotwin_v1.json`; no intermediate raw-pair JSONL is needed. The adapter opens each aggregate at
`{root}/{split}/{embodiment}`, validates every row and episode, and pairs Source/Target only when `split`, `task`, and
`episode_idx` match.

Copy `configs/cluster/hyperbolic_4x8.env.example` to a private preparation-host file, replace its placeholders, add
`NODE_RANK=0`, set `GENET_CONTAINER_NAME=genet-preprocess`, and set `GENET_CACHE_READONLY=0`. The cache is writable only
while the designated preparation node creates the canonical processed tree; restore `GENET_CACHE_READONLY=1` before
validation locks or training. This write access is also required because MosaicML Streaming can lazily materialize
compressed `.mds.zstd` shards beside the source shards on first access. Inspect the resolved adapter configuration
through that same digest-pinned container environment:

```bash
export GENET_PREP_HOST_ENV=/secure/path/preprocess.host.env
export GENET_IMAGE_REF='registry.example/genet@sha256:<pushed-image-digest>'

bash scripts/run_hyperbolic_container.sh \
  "${GENET_PREP_HOST_ENV}" \
  "${GENET_IMAGE_REF}" \
  /opt/genet-preprocess-venv/bin/python -m genet.cli.preprocess_robotwin \
    --root /mnt/nvme/mds-cache/robotwin_v1 \
    --schema /opt/genet/configs/data/robotwin_v1.json \
    --split train \
    --mds-index-fps 16 \
    --output /mnt/nvme/mds-cache/robotwin_v1/genet/processed/train \
    --print-config
```

`--mds-index-fps 16` is a declared canonical index cadence: the MDS has integer `t` but no timestamps, so it must not be
described as a recovered physical camera clock. The adapter labels the per-row `state` as frame-aligned
`joint_position_state`; it does not claim this measured state is an executable actuator command. Change this value only
as a reviewed data-semantics decision, then keep it identical for every release.

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
    /opt/genet-preprocess-venv/bin/python -c "import streaming; from PIL import Image"
    /opt/genet-preprocess-venv/bin/python -m genet.cli.preprocess_robotwin \
      --root /mnt/nvme/mds-cache/robotwin_v1 \
      --schema /opt/genet/configs/data/robotwin_v1.json \
      --split train \
      --mds-index-fps 16 \
      --output /mnt/nvme/mds-cache/robotwin_v1/genet/processed/train \
      --num-frames 81 \
      --sample-fps 16 \
      --height 192 \
      --width 320 \
      --action-dim 64 \
      --action-resample linear
    /opt/genet-preprocess-venv/bin/python -m genet.cli.validate_data \
      --manifest /mnt/nvme/mds-cache/robotwin_v1/genet/processed/train/manifest.jsonl \
      --num-frames 81 \
      --height 192 \
      --width 320 \
      --action-dim 64 \
      --cosmos
  '
```

The default `episode_start` policy emits one clip for every matched episode and every ordered pair of distinct
embodiments when both sides cover the full `T=81` window; shorter pairs are counted in `dropped_short`. Repeat
`--source-embodiment` and `--target-embodiment` to restrict directions. The default stored Reference comes from the
Target embodiment and split but a different task and episode. Production must keep
`reference_mode=stored`; the generic runtime pool has weaker task-exclusion semantics.

The immutable image build validates `streaming==0.13.0` and Pillow in the isolated preprocessing venv. Do not resolve
dependencies separately on each node. The raw MDS cache is a staging input and is not part of the training release lock;
the finished processed tree is the locked training artifact. After a successful full export, restore
`GENET_CACHE_READONLY=1` before copying or training. Restricted-direction exports and failed runs can leave only some
compressed shards materialized; that is acceptable for the staging cache but never evidence that preprocessing
completed. Completion is established by the export's exact aggregate-count checks, Cosmos validation, and content lock.

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
4. write train-split-only normalization artifacts under `${GENET_NODE_RUN_ROOT}/normalization` once that policy and its
   per-embodiment fields are approved.

The reviewed artifact identities are versioned in `configs/checkpoints/cosmos3_edge.json`. As of this release they are
Cosmos3-Edge revision `2a00e87e9976dc3ed5533dd18caf4cdbc3a1bcb2`, Wan VAE revision
`921dbaf3f1674a56f47e83fb80a34bac8a8f203e`, and Wan file SHA-256
`20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36`. The following single command downloads the exact
snapshot and VAE, validates all mandatory/indexed model shards, updates the offline `refs/main`, converts the snapshot to
DCP, records a recursive DCP SHA-256 manifest, and writes
`/mnt/nvme/genet/artifacts/ARTIFACTS.json`:

```bash
bash scripts/run_hyperbolic_container.sh \
  /secure/path/download.host.env \
  "${GENET_IMAGE_REF}" \
  bash scripts/stage_model_artifacts.sh --run-root /mnt/nvme/genet
```

Run the same command later with `--verify-only` for an offline, non-mutating check. Use
`--cosmos-revision <reviewed-40-hex>` only for an intentional upgrade. `--resolve-main` explicitly queries the mutable
upstream branch and records the result, but it must go through review before becoming a release identity. A conflicting
existing `refs/main`, VAE, or DCP is rejected; `--force` is required to replace it, and an old DCP is preserved under a
timestamped backup name.

The private staging environment must set `NODE_RANK=0`, a unique container name,
`HF_HOME=/mnt/nvme/genet/hf-cache`, and `GENET_ALLOW_HF_TOKEN=1`. Add `HF_TOKEN` only if the account requires one; Docker
can inspect that environment while the short-lived preparation container exists, so never reuse it for training.
The script writes `refs/main` so the pinned upstream recipe's offline catalog lookup resolves to the reviewed snapshot.
After staging, record the spec revision in every job environment and use:

```bash
export HF_HOME="${GENET_NODE_RUN_ROOT}/hf-cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export GENET_HF_SNAPSHOT_REVISION=2a00e87e9976dc3ed5533dd18caf4cdbc3a1bcb2
export WAN_VAE_PATH="${GENET_NODE_RUN_ROOT}/artifacts/wan22_vae/Wan2.2_VAE.pth"
```

The script passes the exact local snapshot to the pinned Cosmos conversion interface. Do not substitute a standalone
PyTorch checkpoint, the mutable catalog name, or a partially downloaded directory for the resulting DCP. The training
load path is `/mnt/nvme/genet/checkpoints/Cosmos3-Edge`; `model/.metadata` lives beneath it.

For S1, the `training_checkpoint` artifact is the converted Cosmos3-Edge base DCP. For S2/S3 it is the exact committed
checkpoint selected from the previous stage. For an exact resume it is the complete committed DCP being resumed. Never
reuse an old lock after changing the load checkpoint.

### 5.4 Replicate to every node

Use `rsync`, a durable object store, or Hyperbolic's stage-in facilities to copy these exact trees to the same node-local
root on all four hosts:

- `/mnt/nvme/mds-cache/robotwin_v1/genet/processed/train`;
- `normalization`, if present;
- `artifacts/wan22_vae`;
- `artifacts/ARTIFACTS.json`;
- the exact base/warm-start/resume DCP;
- `hf-cache`.

Copy into dedicated destinations without deleting unrelated files. A successful copy is not accepted until the
offline artifact verifier and cluster-lock verification both succeed on that node. Because all Hyperbolic nodes use the
same `/mnt/nvme/genet` layout, the copied artifact receipt remains valid:

```bash
bash scripts/run_hyperbolic_container.sh \
  /secure/path/lock.host.env \
  "${GENET_IMAGE_REF}" \
  bash scripts/stage_model_artifacts.sh --run-root /mnt/nvme/genet --verify-only
```

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
      --artifact artifact_receipt="${GENET_NODE_RUN_ROOT}/artifacts/ARTIFACTS.json" \
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
      --artifact artifact_receipt="${GENET_NODE_RUN_ROOT}/artifacts/ARTIFACTS.json" \
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
export GENET_HF_SNAPSHOT_REVISION='2a00e87e9976dc3ed5533dd18caf4cdbc3a1bcb2'

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

Generated actions remain offline/simulator outputs until the remaining command semantics and embodiment-specific safety gates
are
implemented. Do not send them directly to a real robot.

## 12. Stop/go checklist

Do not start paid production training until every item is true:

- [ ] hostfile contains exactly the intended physical nodes in deterministic rank order;
- [ ] every node exposes eight GPUs, contains the prewarmed RoboTwin cache, and has enough free space under `/mnt/nvme`;
- [ ] every node pulled the same OCI digest and the embedded GenET revision matches;
- [ ] the direct RoboTwin MDS adapter validated the approved schema and recorded the declared index FPS;
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
