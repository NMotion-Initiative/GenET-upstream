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

### 3.1 Control-node checkout and SSH

The machine that builds the image and runs the SSH coordinator needs one clean GenET checkout. The other three workers
do not: the image contains `/opt/genet`, and the coordinator streams the small host helpers over SSH. On the selected
control node:

```bash
git clone https://github.com/ACondaway/GenET.git
cd GenET
git pull --ff-only origin main

export GENET_REVISION="$(git rev-parse HEAD)"
test "$(printf '%s' "${GENET_REVISION}" | wc -c)" -eq 40
test -z "$(git status --porcelain)"

install -d -m 0700 /secure/path
cp configs/cluster/hyperbolic_4x8.hosts.example \
  /secure/path/genet-hosts.txt
chmod 0600 /secure/path/genet-hosts.txt
# Edit this file now with local/rank1/rank2/rank3 SSH aliases in rank order.
```

Build and launch only committed bytes. For a reviewed release, replace `main` with its exact 40-character commit and
use `git checkout --detach <commit>`. The worker environment is defined by the OCI digest, not by four independent Git
checkouts or `pip install` runs.

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

### 3.2 One-time Ubuntu host packages

Hyperbolic images often already include the NVIDIA driver, Docker, and the NVIDIA container runtime. Inspect first; do
not reinstall a working cloud driver. If `nvidia-smi` is missing or the provisioned driver does not expose all eight
GPUs, stop and ask Hyperbolic support rather than installing an arbitrary driver into the running allocation.

For a missing Docker Engine on an Ubuntu node, install it from Docker's official apt repository. Run directly as root,
or add `sudo` as a normal administrator:

```bash
apt-get update
apt-get install -y ca-certificates curl gnupg git openssh-client rsync \
  iproute2 util-linux rdma-core infiniband-diags ibverbs-utils perftest

install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc

. /etc/os-release
cat > /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: ${UBUNTU_CODENAME:-$VERSION_CODENAME}
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF

apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io \
  docker-buildx-plugin docker-compose-plugin
systemctl enable --now docker
```

Then install/configure the NVIDIA Container Toolkit only when `nvidia-ctk` or the Docker NVIDIA runtime is missing:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update
apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

docker run --rm --runtime=nvidia --gpus all ubuntu:24.04 nvidia-smi
```

These commands follow the current [Docker Ubuntu installation guide](https://docs.docker.com/engine/install/ubuntu/)
and [NVIDIA Container Toolkit guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html),
including NVIDIA's [sample GPU workload](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/sample-workload.html).
The host does not need Miniconda, a Python environment, or a CUDA toolkit. A non-root coordinator account needs Docker
daemon access; use the administrator-approved Docker group or rootless policy and log in again before continuing.

One OCI digest makes the container user space identical; it does not freeze the host kernel, NVIDIA driver, OFED/RDMA
stack, NIC firmware, GID table, MTU, or switch QoS. The Gloo hardware signature compares GPU/driver identity, while the
32-rank NCCL preflight and provider benchmark validate the live fabric. Keep all three gates.

### 3.3 Validate every allocated node

From the control-node checkout, stream the versioned host check to every worker; no worker checkout is required:

```bash
while IFS= read -r GENET_SSH_TARGET; do
  case "${GENET_SSH_TARGET}" in ''|'#'*) continue ;; esac
  if [[ "${GENET_SSH_TARGET}" == local ]]; then
    env \
      GENET_NODE_CACHE=/mnt/nvme/mds-cache/robotwin_v1 \
      GENET_NVME_ROOT=/mnt/nvme \
      GENET_EXPECTED_GPUS=8 \
      bash scripts/check_hyperbolic_host.sh
  else
    ssh "${GENET_SSH_TARGET}" \
      'env GENET_NODE_CACHE=/mnt/nvme/mds-cache/robotwin_v1 \
        GENET_NVME_ROOT=/mnt/nvme GENET_EXPECTED_GPUS=8 bash -s' \
      < scripts/check_hyperbolic_host.sh
  fi
done < /secure/path/genet-hosts.txt
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

The four nodes must be able to reach rank 0's chosen `MASTER_ADDR` on the training, Gloo-preflight, and NCCL-preflight
ports. Public SSH addresses and the private cluster bootstrap address are separate concepts. Validate the actual RDMA
path with both the built-in 32-rank collective and a provider-baselined multi-node `nccl-tests` run; the CPU/Gloo
preflight does not replace them.

Host Miniconda, Python, and a host CUDA toolkit are not part of the training environment: Python, the uv-managed
environment, CUDA user-space libraries, Cosmos, GenET, configs, and launch scripts live in the immutable image. Only the
control node needs the source checkout described in Section 3.1. Each GPU host needs the NVIDIA driver, NVIDIA container
runtime, Docker, SSH/rsync, RDMA devices, and enough local storage.

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
export REGISTRY_HOST='<registry-host>'
export REGISTRY_USER='<registry-user>'
export GENET_IMAGE_REPO="${REGISTRY_HOST}/<namespace>/genet"

# Example for this repository's owner, if GHCR package permissions are enabled:
# export REGISTRY_HOST=ghcr.io
# export REGISTRY_USER=ACondaway
# export GENET_IMAGE_REPO=ghcr.io/acondaway/genet

# Use a short-lived registry credential. Repeat docker login on every worker.
# The build identity needs push permission; worker identities need pull permission.
read -r -s -p 'Registry token: ' GENET_REGISTRY_TOKEN
printf '\n'
printf '%s' "${GENET_REGISTRY_TOKEN}" \
  | docker login "${REGISTRY_HOST}" --username "${REGISTRY_USER}" --password-stdin
unset GENET_REGISTRY_TOKEN

export BASE_IMAGE_REPO='nvcr.io/nvidia/pytorch'
export BASE_IMAGE_TAG="${BASE_IMAGE_REPO}:26.06-py3"
export BASE_IMAGE_DIGEST="$(
  docker buildx imagetools inspect "${BASE_IMAGE_TAG}" \
    --format '{{.Manifest.Digest}}'
)"
export BASE_IMAGE="${BASE_IMAGE_REPO}@${BASE_IMAGE_DIGEST}"
[[ "${BASE_IMAGE}" =~ @sha256:[0-9a-f]{64}$ ]]
docker pull "${BASE_IMAGE}"
docker run --rm --gpus all "${BASE_IMAGE}" nvidia-smi

export GENET_REVISION="$(git rev-parse HEAD)"
export GENET_IMAGE_TAG="${GENET_IMAGE_REPO}:${GENET_REVISION}"
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
export GENET_IMAGE_DIGEST="$(
  docker buildx imagetools inspect "${GENET_IMAGE_TAG}" \
    --format '{{.Manifest.Digest}}'
)"
export GENET_IMAGE_REF="${GENET_IMAGE_REPO}@${GENET_IMAGE_DIGEST}"
[[ "${GENET_IMAGE_REF}" =~ @sha256:[0-9a-f]{64}$ ]]
docker pull "${GENET_IMAGE_REF}"
```

Authenticate every worker before asking the launcher to pull the private image. This sends the short-lived token only
over SSH standard input; it is never placed in `s1.env`, a remote argv, or a file:

```bash
read -r -s -p 'Registry pull token: ' GENET_REGISTRY_PULL_TOKEN
printf '\n'
while IFS= read -r GENET_SSH_TARGET; do
  case "${GENET_SSH_TARGET}" in ''|'#'*) continue ;; esac
  if [[ "${GENET_SSH_TARGET}" == local ]]; then
    printf '%s' "${GENET_REGISTRY_PULL_TOKEN}" \
      | docker login "${REGISTRY_HOST}" \
          --username "${REGISTRY_USER}" --password-stdin
  else
    printf '%s' "${GENET_REGISTRY_PULL_TOKEN}" \
      | ssh "${GENET_SSH_TARGET}" \
          docker login "${REGISTRY_HOST}" \
            --username "${REGISTRY_USER}" --password-stdin
  fi
done < /secure/path/genet-hosts.txt
unset GENET_REGISTRY_PULL_TOKEN
```

Use a pull-only credential on workers when the registry supports separate permissions. The coordinated launcher later
pulls `GENET_IMAGE_REF` concurrently and checks that the local manifest digest is the requested digest before starting
any rendezvous.

The helper refuses dirty tracked source, builds from `git archive HEAD`, embeds the full Git revision, and requires the
base image to be pinned by digest. Untracked files are never included. Use `cu130-train` with the CUDA 13 / NGC PyTorch
26.06 base, or `cu128-train` with the CUDA 12.8 / NGC PyTorch 25.06 base; resolve the selected base tag to a digest before
building and confirm the node driver supports it.

The example selects the pinned digest corresponding to the official Cosmos-recommended CUDA 13 base. Use
`nvcr.io/nvidia/pytorch:25.06-py3` plus `cu128-train` when the installed driver supports CUDA 12.8 but not CUDA 13.
Record the digest printed by the registry and use the resolved `GENET_IMAGE_REF` as the only production reference:

```bash
: "${GENET_IMAGE_REF:?Keep the digest-pinned image reference produced above}"
[[ "${GENET_IMAGE_REF}" =~ @sha256:[0-9a-f]{64}$ ]]
```

Authenticate all four nodes to the registry using its credential mechanism before the coordinated launch. Do not put a
registry password in the shared job environment. The SSH launcher fans out the digest pull, but it must never convert
`GENET_IMAGE_REF` back to a tag. The image contains versioned `configs/` and `scripts/` under `/opt/genet`, the installed
GenET wheel, the pinned Cosmos checkout, and package inventories for both Python environments. Its default PATH is the
frozen Cosmos training venv. RoboTwin preprocessing runs from `/opt/genet-preprocess-venv`; this separation prevents
Streaming's NumPy constraint from changing the Cosmos lock.

## 5. Prepare RoboTwin v1 once, then replicate the bytes

### 5.1 Confirm the fixed MDS contract and semantic choices

The repository now consumes the probed RoboTwin-v1 MDS schema directly through
`configs/data/robotwin_v1.json`; no intermediate raw-pair JSONL is needed. The adapter opens each aggregate at
`{root}/{split}/{embodiment}`, validates every row and episode, and pairs Source/Target only when `split`, `task`, and
`episode_idx` match.

Copy the dedicated one-node template rather than the strict training environment:

```bash
cp configs/cluster/hyperbolic_prepare.env.example /secure/path/preprocess.host.env
```

It sets `NODE_RANK=0`, a unique container name, `GENET_CACHE_READONLY=0`, and `GENET_PROTECT_INPUTS=0`. The cache is
writable only while the designated preparation node creates the canonical processed tree; training later uses a
read-only mount. This write access is also required because MosaicML Streaming can lazily materialize compressed
`.mds.zstd` shards beside the source shards on first access. Inspect the resolved adapter configuration through that
same digest-pinned container environment:

```bash
export GENET_PREP_HOST_ENV=/secure/path/preprocess.host.env
: "${GENET_IMAGE_REF:?Export the digest-pinned image reference produced in Section 4}"

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
: "${GENET_IMAGE_REF:?Export the digest-pinned image reference produced in Section 4}"

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

The trainer consumes the locked train manifest. Export `val` separately before releasing the preparation node so the
fixed validation split is also checked and available for evaluation:

```bash
bash scripts/run_hyperbolic_container.sh \
  "${GENET_PREP_HOST_ENV}" \
  "${GENET_IMAGE_REF}" \
  /opt/genet-preprocess-venv/bin/python -m genet.cli.preprocess_robotwin \
    --root /mnt/nvme/mds-cache/robotwin_v1 \
    --schema /opt/genet/configs/data/robotwin_v1.json \
    --split val \
    --mds-index-fps 16 \
    --output /mnt/nvme/mds-cache/robotwin_v1/genet/processed/val \
    --num-frames 81 \
    --sample-fps 16 \
    --height 192 \
    --width 320 \
    --action-dim 64 \
    --action-resample linear
```

Add the validation tree to a separate evaluation lock if an evaluation job consumes it; do not silently add it to a
training release whose lock was already published.

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
cp configs/cluster/hyperbolic_artifacts.env.example \
  /secure/path/download.host.env

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

The supplied staging template sets `NODE_RANK=0`, a unique container name,
`HF_HOME=/mnt/nvme/genet/hf-cache`, and `GENET_ALLOW_HF_TOKEN=1`. Add `HF_TOKEN` only if the account requires one; Docker
can inspect that environment while the short-lived preparation container exists, so never reuse it for training and
unset the token immediately after staging.
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

Before copying, re-run the offline verifier on the canonical node from a fresh token-free template. A successful copy is
not accepted until the all-node semantic and cluster-lock verification in Section 7 also succeeds:

```bash
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN
cp configs/cluster/hyperbolic_artifacts.env.example \
  /secure/path/artifact-verify.host.env

bash scripts/run_hyperbolic_container.sh \
  /secure/path/artifact-verify.host.env \
  "${GENET_IMAGE_REF}" \
  bash scripts/stage_model_artifacts.sh --run-root /mnt/nvme/genet --verify-only
```

When the preparation node is rank 0 and the coordinator, the following non-destructive loop copies every S1 input to
ranks 1–3. It deliberately omits `--delete`; use a new versioned release destination for upgrades instead of mutating a
previously verified input set:

```bash
while IFS= read -r GENET_SSH_TARGET; do
  case "${GENET_SSH_TARGET}" in ''|'#'*|local) continue ;; esac
  ssh "${GENET_SSH_TARGET}" \
    'mkdir -p \
      /mnt/nvme/mds-cache/robotwin_v1/genet/processed/train \
      /mnt/nvme/genet/hf-cache \
      /mnt/nvme/genet/artifacts/wan22_vae \
      /mnt/nvme/genet/checkpoints/Cosmos3-Edge'

  rsync -aH --partial --info=progress2 \
    /mnt/nvme/mds-cache/robotwin_v1/genet/processed/train/ \
    "${GENET_SSH_TARGET}:/mnt/nvme/mds-cache/robotwin_v1/genet/processed/train/"

  rsync -aH --partial --info=progress2 \
    /mnt/nvme/genet/hf-cache/ \
    "${GENET_SSH_TARGET}:/mnt/nvme/genet/hf-cache/"

  rsync -aH --partial --info=progress2 \
    /mnt/nvme/genet/artifacts/wan22_vae/ \
    "${GENET_SSH_TARGET}:/mnt/nvme/genet/artifacts/wan22_vae/"

  rsync -aH --partial \
    /mnt/nvme/genet/artifacts/ARTIFACTS.json \
    "${GENET_SSH_TARGET}:/mnt/nvme/genet/artifacts/ARTIFACTS.json"

  rsync -aH --partial --info=progress2 \
    /mnt/nvme/genet/checkpoints/Cosmos3-Edge/ \
    "${GENET_SSH_TARGET}:/mnt/nvme/genet/checkpoints/Cosmos3-Edge/"
done < /secure/path/genet-hosts.txt
```

Copy `normalization/` in the same way only after that artifact exists and has been added to the lock. Raw RoboTwin MDS
already exists independently on every node and is not copied by this loop. A later full release verification hashes all
bytes and rejects missing, stale, or changed files.

## 6. Create the release lock and node receipts

Create a distinct release directory and lock for each job input set. Run the lock tool through the same immutable image;
use the dedicated one-node lock template. The following is an S1 example:

```bash
export GENET_NODE_RUN_ROOT=/mnt/nvme/genet
export GENET_PROCESSED_DATA=/mnt/nvme/mds-cache/robotwin_v1/genet/processed/train
export GENET_RELEASE_ID="s1-${GENET_REVISION}"
export GENET_RELEASE_DIR="${GENET_NODE_RUN_ROOT}/release/${GENET_RELEASE_ID}"

cp configs/cluster/hyperbolic_lock.env.example /secure/path/lock.host.env
sed -i "s/REPLACE_WITH_RELEASE_ID/${GENET_RELEASE_ID}/g" \
  /secure/path/lock.host.env

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

Add `--artifact normalization="${GENET_NODE_RUN_ROOT}/normalization"` once that artifact exists. The shared environment
must then set both `GENET_NORMALIZATION_ARTIFACT=normalization` and its exact `GENET_NORMALIZATION_PATH`.

Do not create or copy a node receipt yet. It must be generated independently from each node's local paths after the
shared job environment is complete. That environment includes:

```bash
export GENET_ARTIFACT_RECEIPT_ARTIFACT=artifact_receipt
export GENET_ARTIFACT_RECEIPT_PATH=/mnt/nvme/genet/artifacts/ARTIFACTS.json
export GENET_PROTECT_INPUTS=1
```

After creating the lock on rank 0, copy only that lock file to the identical release directory on ranks 1–3:

```bash
while IFS= read -r GENET_SSH_TARGET; do
  case "${GENET_SSH_TARGET}" in ''|'#'*|local) continue ;; esac
  ssh "${GENET_SSH_TARGET}" "mkdir -p '${GENET_RELEASE_DIR}'"
  rsync -a --partial \
    "${GENET_RELEASE_DIR}/cluster-lock.json" \
    "${GENET_SSH_TARGET}:${GENET_RELEASE_DIR}/cluster-lock.json"
done < /secure/path/genet-hosts.txt
```

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

The hostfile was created and filled in Section 3.1. Now copy the shared training environment template to the same private
job directory and edit every placeholder:

```bash
cp configs/cluster/hyperbolic_4x8.env.example /secure/path/s1.env
chmod 0600 /secure/path/s1.env
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
export NCCL_PREFLIGHT_PORT='<third-unique-nccl-preflight-port>'
export GENET_PREFLIGHT_TIMEOUT_SECONDS=120
export GENET_NCCL_PREFLIGHT_TIMEOUT_SECONDS=180
export GENET_NCCL_PREFLIGHT_BUFFER_MIB=64
export GENET_NCCL_PREFLIGHT_WARMUP_ITERATIONS=3
export GENET_NCCL_PREFLIGHT_ITERATIONS=10

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
export GENET_ARTIFACT_RECEIPT_ARTIFACT=artifact_receipt
export GENET_PROCESSED_DATA=/mnt/nvme/mds-cache/robotwin_v1/genet/processed/train
export GENET_MANIFEST=/mnt/nvme/mds-cache/robotwin_v1/genet/processed/train/manifest.jsonl
export BASE_CHECKPOINT_PATH=/mnt/nvme/genet/checkpoints/Cosmos3-Edge
export WAN_VAE_PATH=/mnt/nvme/genet/artifacts/wan22_vae/Wan2.2_VAE.pth
export GENET_ARTIFACT_RECEIPT_PATH=/mnt/nvme/genet/artifacts/ARTIFACTS.json
export HF_HOME=/mnt/nvme/genet/hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export GENET_NVME_ROOT=/mnt/nvme
export GENET_NODE_CACHE=/mnt/nvme/mds-cache/robotwin_v1
export GENET_NODE_RUN_ROOT=/mnt/nvme/genet
export GENET_EXPECTED_GPUS=8
export GENET_CACHE_READONLY=1
export GENET_PROTECT_INPUTS=1
export GENET_IMAGE_PULL_POLICY=missing
```

Also set the network variables after discovery on the real allocation. All ranks compare these values, so choose an
interface/HCA expression valid on every node. Keep `NODE_RANK` out of the shared environment file; the SSH launcher
assigns it from hostfile order.

The hostfile and environment file are separate inputs:

- the hostfile contains SSH transport identities in rank order;
- the environment file contains public distributed/runtime values and node-local artifact paths;
- neither file contains registry passwords, Hugging Face tokens, or private SSH keys.

With both files now present, perform the first all-node release verification. This runs the offline Cosmos/Wan/DCP
semantic verifier, hashes every locked byte, and creates a separate node receipt on each host:

```bash
bash scripts/launch_cluster_ssh.sh \
  --hosts /secure/path/genet-hosts.txt \
  --env /secure/path/s1.env \
  --image "${GENET_IMAGE_REF}" \
  --config configs/experiments/stage1_control_32gpu.yaml \
  --run-id s1-release-verify \
  --log-dir /mnt/nvme/genet/logs/s1-release-verify \
  --verify-release-only
```

Do not copy a receipt from another node. The normal production launch repeats this semantic verification and full hash
immediately before distributed preflight. It then mounts processed data, the HF cache, Wan VAE, exact load DCP, staging
receipt, lock, and node receipt read-only inside dry-run/training containers; only outputs remain writable. Use
`--skip-release-verify` only for controlled diagnostics.

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

Use the same hostfile, environment, image digest, and config for all phases. First stop after the guarded release,
CPU/Gloo, and 32-rank NCCL collective preflights:

```bash
bash scripts/launch_cluster_ssh.sh \
  --hosts /secure/path/genet-hosts.txt \
  --env /secure/path/s1.env \
  --image "${GENET_IMAGE_REF}" \
  --config configs/experiments/stage1_control_32gpu.yaml \
  --run-id s1-control-preflight \
  --log-dir /mnt/nvme/genet/logs/s1-control-preflight \
  --preflight-only
```

The native NCCL phase runs one process per GPU, validates all-reduce contents, benchmarks a configurable buffer, and
writes an `nccl_preflight_passed` JSON report. Inspect the coordinator logs before training:

```bash
grep -E 'nccl_preflight_passed|NET/(IB|Socket)|Using network' \
  /mnt/nvme/genet/logs/s1-control-preflight/nccl-preflight-rank-*.log
```

A passing collective proves that all 32 ranks can communicate, but it does not by itself prove the provider's expected
RoCE throughput. Confirm that NCCL selected the IB/RDMA transport instead of an unintended Socket fallback, and compare
the reported bandwidth with the allocation baseline. For initial cluster qualification, also run the official
[`NVIDIA/nccl-tests`](https://github.com/NVIDIA/nccl-tests) all-reduce benchmark in the provider-supported MPI/container
environment; NVIDIA documents that multi-node builds require `MPI=1`. Do not invent MPI/SSH arguments that conflict
with Hyperbolic's current image or fabric setup.

After those gates pass, start the default guarded sequence. It performs host checks, pulls or verifies the image on all
nodes, re-hashes the complete release and refreshes receipts, runs Gloo, runs the 32-rank NCCL preflight, runs a 32-rank
GenET dry run, and only then starts the real training command:

```bash
bash scripts/launch_cluster_ssh.sh \
  --hosts /secure/path/genet-hosts.txt \
  --env /secure/path/s1.env \
  --image "${GENET_IMAGE_REF}" \
  --config configs/experiments/stage1_control_32gpu.yaml \
  --run-id s1-control-seed42 \
  --log-dir /mnt/nvme/genet/logs/s1-control-seed42 \
  -- \
  --manifest /mnt/nvme/mds-cache/robotwin_v1/genet/processed/train/manifest.jsonl \
  --warm-start /mnt/nvme/genet/checkpoints/Cosmos3-Edge \
  --output-dir /mnt/nvme/genet/outputs/s1-control-seed42
```

Do not add `--dry-run` after `--`; the coordinator owns that guard. Use `--dry-run-only` when deliberately stopping after
the distributed dry run. The `--skip-host-check`, `--skip-release-verify`, `--skip-preflight`,
`--skip-nccl-preflight`, and `--skip-dry-run` switches exist for controlled diagnostics, not for bypassing a failed
production gate. Keep the coordinator alive in `tmux` or `systemd`: closing its attached SSH sessions stops the job.

The required sequence is:

1. pull/verify the immutable image on all nodes;
2. re-hash each node's artifacts against the cluster lock and refresh its receipt;
3. run the four-node CPU/Gloo preflight;
4. run the built-in 32-rank NCCL correctness/bandwidth preflight and confirm `NET/IB`;
5. qualify a new allocation/fabric with provider-appropriate `nccl-tests` and its bandwidth baseline;
6. let the coordinator run a 32-rank GenET dry run;
7. let the same guarded invocation proceed to training.

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

Generated actions remain offline/simulator outputs until the remaining command semantics and embodiment-specific safety
gates are implemented. Do not send them directly to a real robot.

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
- [ ] the built-in 32-rank NCCL preflight passed and its logs confirmed the intended RDMA transport;
- [ ] provider-baselined multi-node `nccl-tests` passed on a new allocation/fabric;
- [ ] the 32-rank GenET dry run passed;
- [ ] checkpoint archive publication, consolidation, verification, and prestage were rehearsed;
- [ ] training, Gloo-preflight, and NCCL-preflight ports are distinct and unique to this active job;
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
