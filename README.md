# GenET: Cross-Embodiment Synchronized Video–Action Generation

GenET is a Cosmos3-Edge training project for cross-robot-embodiment generation. It takes:

1. a task video–action pair from a Source Embodiment; and
2. an independently sampled, fixed-length reference video–action segment from the Target Embodiment.

The model jointly generates a Target Embodiment video and action sequence that follows the Source task semantics and
timing. Supervised training still requires a paired `target_gt`; the reference only supplies Target appearance, motion,
and action-space priors. The generic raw-pair path excludes the `target_gt` episode. The fixed RoboTwin-v1 adapter is
stricter by default: it selects a stored reference from the same Target embodiment and split, but from a different task
and episode.

> **Implementation status:** preprocessing, validation, lightweight joint rectified-flow training, the shared/dual
> reference ablation, a conditional Cosmos sampling API, checkpoints, and `81/17` rolling-window long-horizon
> orchestration are wired and covered by lightweight tests. Real Cosmos weights, clean-prefix two-window sampling,
> long-video quality, and 4×8-GPU RoCE jobs must still be validated in the NVIDIA Cosmos training container on the target
> cluster. This repository does not claim that those expensive production runs were completed on the development host.

## Design Highlights

- Video and action use the same rectified-flow `sigma`, preserving joint generation time.
- Source video is encoded by the Wan2.2 causal VAE, then a zero-initialized ControlNet-style residual is added to the
  noisy Target latent. Source action is injected through a zero-initialized mapping with Source/Target domain embeddings.
- Reference video and action form one aligned token set. Fixed temporal/spatial positional encodings preserve frame order
  and vision–action alignment before gated cross-attention in the MoT layers.
- The `shared` versus `dual` ablation changes only whether the two consuming MoT routes share the reference K/V projector.
  Queries, outputs, gates, reference encoders, injection layers, data, and initialization are otherwise controlled.
- Cosmos3-Edge's Cosmos MoT is the generator backbone. Wan2.2 supplies the causal VAE/tokenizer on this path; GenET does
  not replace Cosmos MoT with the complete Wan DiT.
- Production parallelism defaults to 8-way FSDP sharding × 4-way replication (HSDP), keeping frequent shard traffic
  inside each eight-GPU node.
- Long generation uses `T=81`, overlap `O=17`, and stride `S=64`. Video and action are accepted, retried, and rolled back
  as one transaction. The next window keeps five clean video-latent tokens and 17 clean action steps.

Detailed documentation:

- [Architecture](docs/ARCHITECTURE.md)
- [Preprocessing and validation](docs/PREPROCESSING.md)
- [Training, RoCE deployment, and checkpoints](docs/TRAINING.md)
- [Hyperbolic 4×8-GPU end-to-end runbook](docs/HYPERBOLIC.md)
- [Long-horizon inference](docs/INFERENCE.md)

## Quick Start

Do not install into Ubuntu's externally managed system Python. No Miniconda is required; create a project virtual
environment (install `python3-full` first if `venv` is missing):

```bash
# Run apt-get directly as root, or prefix it with sudo as a normal user.
apt-get update
apt-get install -y python3-full
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[preprocess,robotwin,dev]'
```

Production preprocessing and training should instead use the same immutable Docker image. The default environment is
the frozen Cosmos training runtime; RoboTwin preprocessing is deliberately isolated in
`/opt/genet-preprocess-venv` because `mosaicml-streaming==0.13.0` and the Cosmos lock require incompatible NumPy
versions. Running one OCI digest gives all four nodes the same container Python environments without mutating Cosmos
dependencies.

For the fixed node-local RoboTwin schema, export directed cross-embodiment pairs directly from MDS. The cache has no
physical timestamps, so `--mds-index-fps 16` explicitly declares one MDS row per canonical 16 Hz GenET index:

```bash
# Inside the production image. In the local venv, use the unqualified command.
/opt/genet-preprocess-venv/bin/python -m genet.cli.preprocess_robotwin \
  --root /mnt/nvme/mds-cache/robotwin_v1 \
  --schema configs/data/robotwin_v1.json \
  --split train \
  --mds-index-fps 16 \
  --camera head \
  --reference-policy different_task \
  --require-bidirectional \
  --output /mnt/nvme/mds-cache/robotwin_v1/genet/processed/train
```

The default creates one `T=81` clip at each matched episode start for every ordered pair of distinct embodiments when
both Source and Target contain at least 81 frames. With five embodiments this covers 20 directions, including both
`A->B` and `B->A`; each direction stores a separate reference from its current Target embodiment. The release flag also
proves exact reverse coverage and rejects the direction-biased global `--max-samples` cap. Do not add a second random
Source/Target swap in the Dataset. Validate before training:

```bash
genet-validate-data \
  --manifest /mnt/nvme/mds-cache/robotwin_v1/genet/processed/train/manifest.jsonl \
  --num-frames 81 --height 192 --width 320 --action-dim 64 \
  --cosmos \
  --require-bidirectional \
  --expected-embodiment ARX-X5 \
  --expected-embodiment aloha-agilex \
  --expected-embodiment franka-panda \
  --expected-embodiment piper \
  --expected-embodiment ur5-wsg
```

This validates the canonical five-embodiment inventory and exact reciprocal records in the static manifest. It does
not promise exact `A->B`/`B->A` exposure in every training epoch or arbitrary `max_steps` prefix: distributed loading
uses a without-replacement shuffle of `floor(N/WORLD_SIZE) * WORLD_SIZE` records and rotates the at-most
`WORLD_SIZE-1` omitted tail records across epochs. The startup `pair_direction_summary` describes manifest inventory,
not samples consumed so far.

The exporter records a versioned `content_sha256` for every Source, Target, and Reference stream. Strict validation
recomputes those identities from the decompressed NPZ arrays and requires the Source/Target hashes to swap in every
reverse record, so reciprocal metadata cannot hide unrelated payloads.

A strict release combines four independent requirements: preprocessing explicitly uses
`--reference-policy different_task`, training uses `data.reference_mode=stored`, canonical validation pins the five
expected embodiments and reciprocal coverage, and the published content lock hashes the processed manifest, NPZs,
`index.json`, and `stats.json`. Changing any of those inputs creates a new release rather than an in-place update.

The original `genet-preprocess` command remains available for file-based `genet.raw-pair/v1` JSONL datasets; see
[Preprocessing](docs/PREPROCESSING.md).

Run the repository tests and a lightweight dry run:

```bash
pytest -q

python -m genet.cli.train \
  --config configs/base.yaml \
  --manifest data/processed/train/manifest.jsonl \
  --dry-run
```

`configs/base.yaml` uses the `toy` backend so the data, gradient, mask, joint-flow, and checkpoint contracts can be tested
on CPU or one GPU. It is an executable preflight baseline, not a replacement for Cosmos.

## Hyperbolic Cluster Fast Path

For four eight-GPU Hyperbolic nodes without shared storage, use the complete
[Hyperbolic runbook](docs/HYPERBOLIC.md). It treats `/mnt/nvme/mds-cache/robotwin_v1` as the node-local data cache and
`/mnt/nvme/genet` as node-local run storage. It covers the full path from immutable image build and canonical
preprocessing through artifact locks, hostfile-driven remote launch, S1/S2/S3 checkpoint transport, and rollback-safe
long video–action inference. Hostnames, SSH ports, bootstrap addresses, network interfaces, HCAs, registry references,
and model revisions remain explicit parameters; the guide does not guess them from provider defaults.

Host Miniconda and a host CUDA toolkit are not required. Install/verify the NVIDIA driver, Docker, NVIDIA Container
Toolkit, SSH/rsync, and RDMA devices; Python, CUDA user-space libraries, Cosmos, and GenET are supplied by one immutable
image. After copying and filling the environment/host templates and creating the release lock/receipts, launch all four
nodes from rank 0 with one attached command:

```bash
export GENET_IMAGE_REF='registry.example/genet@sha256:<image-digest>'

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

The launcher runs host checks, resolves the immutable image concurrently, re-hashes every node-local release and
refreshes its receipt, runs a four-node Gloo consistency check, runs a 32-rank NCCL correctness/bandwidth preflight, and
runs a 32-rank Cosmos dry run before training. Verified model inputs are read-only binds while the output root remains
writable. Keep this attached coordinator command inside `tmux` or a supervised service. See the runbook for the
control-node checkout, host installation, image construction under `/dev/shm`, registry login, model/data staging,
Docker storage placement, RoCE discovery, inference, and failure recovery.

## Cosmos3-Edge Production Environment

The project pins Cosmos Framework to:

```text
a904d2d36b774a51dd06ff9ff906816b1a04f579
```

The exact reviewed artifacts come from the official
[`nvidia/Cosmos3-Edge`](https://huggingface.co/nvidia/Cosmos3-Edge) and
[`Wan-AI/Wan2.2-TI2V-5B`](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) repositories and are pinned in
`configs/checkpoints/cosmos3_edge.json`. Inside the immutable image on
the staging node, one command downloads the complete Cosmos3-Edge snapshot and Wan VAE, verifies every indexed model
shard plus the VAE byte count/SHA-256, converts the snapshot to DCP, writes a recursive DCP SHA-256 manifest, updates the
offline HF ref, and publishes an artifact receipt under one run-root lock:

```bash
bash scripts/stage_model_artifacts.sh --run-root /mnt/nvme/genet

# Later, with networking disabled:
bash scripts/stage_model_artifacts.sh --run-root /mnt/nvme/genet --verify-only
```

The resulting paths are `/mnt/nvme/genet/hf-cache`,
`/mnt/nvme/genet/artifacts/wan22_vae/Wan2.2_VAE.pth`, and
`/mnt/nvme/genet/checkpoints/Cosmos3-Edge`. No separate Wan DiT, DROID policy, Reasoner checkpoint, ControlNet, or
reference encoder checkpoint is required for S1; the new GenET branches initialize during training.
`HF_HOME` must be exactly `/mnt/nvme/genet/hf-cache`; use only an ephemeral `HF_TOKEN` environment variable if access
policy requires authentication, because the script rejects persisted token files in the replicated cache.
Budget at least 100 GB free for the current 29.5 GB Cosmos snapshot, 2.82 GB Wan VAE, converted DCP, temporary files,
and any explicitly preserved `--force` backup.

For interactive bring-up inside the NVIDIA Cosmos training container:

```bash
bash scripts/bootstrap_cosmos.sh
python -m pip install -e 'third_party/cosmos-framework[train]'
python -m pip install -e .
```

Production nodes should not resolve and install dependencies independently. Build one immutable image from
[`containers/Dockerfile`](containers/Dockerfile), push it once, and run that exact OCI digest on all nodes. If the cluster
uses Apptainer, convert the image once, distribute the same SIF, and use its SHA-256 as the image identity.

```bash
GENET_REVISION="$(git rev-parse HEAD)"

git archive --format=tar \
  --add-virtual-file="GENET_BUILD_REVISION:${GENET_REVISION}" \
  HEAD | docker build -f containers/Dockerfile \
  --build-arg BASE_IMAGE='registry.example/cosmos-train@sha256:<base-image-digest>' \
  --build-arg GENET_CODE_REVISION="${GENET_REVISION}" \
  --build-arg COSMOS_DEPENDENCY_GROUP=cu130-train \
  -t registry.example/genet:${GENET_REVISION} -

docker push registry.example/genet:${GENET_REVISION}
```

Building from `git archive HEAD` deliberately excludes dirty and untracked working-tree files. The virtual revision file
is checked against the image build argument, remains available to the runtime preflight, and prevents arbitrary source
from being labeled as that commit. The image retains the versioned `configs/` and `scripts/`, installs GenET in the
frozen Cosmos training venv without resolving new dependencies, and builds a separate `/opt/genet-preprocess-venv` for
RoboTwin. Sorted package inventories and checksums for both environments are written under `/opt`. Record the pushed
image digest; never rely on a mutable tag for a multi-node run.

## Multi-Node Training Without Shared Storage

Shared storage is not required. All nodes must run the same immutable image and semantic training configuration, while
consuming byte-identical, checksum-verified replicas of the dataset, model artifacts, tokenizer cache, and committed
checkpoint from node-local storage. Absolute local paths may differ.

The reliable deployment flow is:

```text
build one immutable image
        ↓
prepare data, weights, tokenizer cache, and one cluster lock on a staging host
        ↓
copy the release to each node's local NVMe and verify every copy
        ↓
re-hash/refresh receipts, run CPU/Gloo, 32-rank NCCL, then the GenET dry run
        ↓
start training with the runtime all-rank preflight enabled
        ↓
publish each node's DCP shards, consolidate them, and prestage the committed DCP before resume
```

### 1. Create one content lock

Create the lock from the canonical, materialized staging copy. Artifact names are logical; absolute paths may differ on
each node. Directory hashes include every regular file's relative path, size, and SHA-256. Internal symlinks, such as the
ones used by the Hugging Face cache, are recorded by their root-relative target; broken links and links escaping the
artifact root are rejected.

```bash
genet-cluster-lock create \
  --output /staging/genet-release/cluster-lock.json \
  --artifact processed_data=/staging/genet/data/processed/train \
  --artifact wan_vae=/staging/genet/artifacts/wan22_vae/Wan2.2_VAE.pth \
  --artifact artifact_receipt=/staging/genet/artifacts/ARTIFACTS.json \
  --artifact training_checkpoint=/staging/genet/checkpoints/stage2_shared_committed \
  --artifact normalization=/staging/genet/data/normalization \
  --artifact hf_cache=/staging/genet/hf-cache
```

If this release has no approved normalization artifact yet, omit that logical entry from both the create and verify
commands. `training_checkpoint` must point to the exact DCP passed through `--warm-start`, `--resume`, or
`BASE_CHECKPOINT_PATH` for this job: use the Cosmos3-Edge base DCP for S1, the selected prior-stage DCP for S2/S3, and the
prestaged committed DCP for resume. Hashing the full processed dataset is intentionally a staging/deployment operation
rather than a per-step training operation.

### 2. Copy and verify every local replica

Use `rsync`, an object store, or the scheduler's stage-in mechanism. Then run this command once in every node prologue:

```bash
genet-cluster-lock verify \
  --lock /local_nvme/genet/release/cluster-lock.json \
  --receipt /local_nvme/genet/release/node-receipt.json \
  --artifact processed_data=/local_nvme/genet/data/processed/train \
  --artifact wan_vae=/local_nvme/genet/artifacts/wan22_vae/Wan2.2_VAE.pth \
  --artifact artifact_receipt=/local_nvme/genet/artifacts/ARTIFACTS.json \
  --artifact training_checkpoint=/local_nvme/genet/checkpoints/stage2_shared_committed \
  --artifact normalization=/local_nvme/genet/data/normalization \
  --artifact hf_cache=/local_nvme/genet/hf-cache
```

The node-local receipt binds logical artifacts to the paths that were actually verified. Strict Cosmos startup rejects a
manifest, Wan VAE, HF cache, or load checkpoint that is not bound to that receipt. The Hyperbolic coordinator performs
this full re-hash concurrently on all nodes before every production launch, refreshes each node's receipt, and mounts the
verified inputs read-only. Any mismatch stops the complete job.

### 3. Enable the strict runtime contract

Start from [`configs/cluster/roce_4x8.env.example`](configs/cluster/roce_4x8.env.example), replace every placeholder, and
export the same public identity values on all nodes:

```bash
export GENET_STRICT_ENV=1
export GENET_IMAGE_DIGEST='sha256:<64-hex-image-or-SIF-digest>'
export GENET_CODE_REVISION='<40-hex-GenET-commit>'
export GENET_COSMOS_REVISION='a904d2d36b774a51dd06ff9ff906816b1a04f579'
export GENET_CLUSTER_LOCK=/local_nvme/genet/release/cluster-lock.json
export GENET_CLUSTER_RECEIPT=/local_nvme/genet/release/node-receipt.json
export GENET_HF_SNAPSHOT_REVISION='2a00e87e9976dc3ed5533dd18caf4cdbc3a1bcb2'
export GENET_PROTECT_INPUTS=1
export GENET_ARTIFACT_RECEIPT_ARTIFACT=artifact_receipt
export GENET_ARTIFACT_RECEIPT_PATH=/local_nvme/genet/artifacts/ARTIFACTS.json
export WAN_VAE_PATH=/local_nvme/genet/artifacts/wan22_vae/Wan2.2_VAE.pth
export HF_HOME=/local_nvme/genet/hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Run the one-process-per-node CPU/Gloo preflight first, followed by the one-process-per-GPU NCCL preflight:

```bash
bash scripts/preflight_roce.sh
bash scripts/preflight_nccl.sh
```

It verifies the strict runtime identity, one unique physical node identity per rank, eight visible GPUs per node, GPU
model/capability/memory, and the NVIDIA driver, then compares all four node reports over a bounded, dedicated
`PREFLIGHT_PORT`. The NCCL phase then runs a correct 32-rank all-reduce and reports timing/bandwidth. Confirm its
`NCCL_DEBUG=INFO` logs selected `NET/IB`; run official multi-node `nccl-tests` separately when qualifying a new
allocation or fabric against the provider bandwidth baseline.

At startup, GenET all-gathers and compares:

- the semantic training configuration, processed manifest SHA-256, and sorted embodiment map;
- Python, PyTorch, CUDA runtime, cuDNN, NCCL, GPU model/capability, and the installed package-set hash;
- the installed GenET Python-source hash, embedded build revision, and discoverable GenET/Cosmos Git revisions;
- the declared immutable image digest, pinned Hugging Face revision, cluster-lock hash, and path-independent receipt
  contract.

The lock verification protects the bytes of the processed NPZ files, normalization data, Wan VAE, actual load DCP, and
tokenizer cache. The receipt binds those verified bytes to runtime paths; the collective protects against rank-level
software drift. Strict Cosmos startup also verifies that the cached `nvidia/Cosmos3-Edge` `refs/main` value equals
`GENET_HF_SNAPSHOT_REVISION`. All three checks solve different failure modes and are required for a production run.

### 4. Launch with deterministic node ranks

`scripts/launch_roce.sh` uses static `torchrun` rendezvous so `NODE_RANK=0..3` is authoritative. This also prevents a
checkpoint publisher from mislabeling a node because elastic c10d join order differed from the scheduler's node rank.
Every node uses the same command; only `NODE_RANK` changes.

```bash
source configs/cluster/roce_4x8.env.example
export NODE_RANK=0  # 0, 1, 2, or 3 on the corresponding scheduler node

bash scripts/launch_roce.sh \
  configs/experiments/stage3_shared_32gpu.yaml \
  --manifest /local_nvme/genet/data/processed/train/manifest.jsonl \
  --warm-start /local_nvme/genet/checkpoints/stage2_shared_committed \
  --output-dir /local_nvme/genet/outputs/stage3_shared_seed42 \
  --dry-run
```

After the dry run and `nccl-tests` pass, remove `--dry-run`. Use a unique `MASTER_PORT` per concurrent job. Node-local
paths may differ because they are excluded from the semantic config fingerprint, but their logical artifacts must match
the same cluster lock.

For checkpoint persistence, each node publishes only its local DCP shards to an archive host or object store. The archive
job consolidates all four node manifests, verifies every shard, and writes `COMMITTED`. Before resume, copy that complete
committed DCP back to every node and verify it. See [Training](docs/TRAINING.md) for the exact
`publish → consolidate → verify → prestage` commands. A shared training filesystem is unnecessary, but a durable archive
or object store is strongly recommended because no single node initially contains a complete 32-rank checkpoint.

## Long-Horizon Video–Action Generation

The long-horizon entry point injects the real model, Source/Reference readers, and robot action semantics through a
factory. Validate configuration without importing the factory:

```bash
genet-generate-long \
  --config configs/inference/long_video_action.yaml \
  --factory my_robot.genet_factory:create_runner \
  --output /local_nvme/genet/inference/pick_place_00042 \
  --dry-run
```

Start a new run or recover from the last checksum-complete accepted window boundary:

```bash
genet-generate-long \
  --config configs/inference/long_video_action.yaml \
  --factory my_robot.genet_factory:create_runner \
  --output /local_nvme/genet/inference/pick_place_00042 \
  --no-resume

genet-generate-long \
  --config configs/inference/long_video_action.yaml \
  --factory my_robot.genet_factory:create_runner \
  --output /local_nvme/genet/inference/pick_place_00042 \
  --resume
```

`--dry-run` does not import or call the factory. Passing it does not validate Cosmos weights, clean-prefix wiring, or GPU
memory. Before release, run at least one real two-window smoke test and verify the video/action prefix, timestamps, and
journal recovery. The quality gates, retry policy, rollback protocol, and multi-rank restrictions are documented in
[Long-horizon inference](docs/INFERENCE.md).

## Repository Layout

```text
configs/                    base, ablation, inference, and RoCE configuration
containers/                 immutable production image template
docs/                       architecture, preprocessing, training, and inference guides
scripts/                    upstream pinning, torchrun, and DCP transport scripts
src/genet/data/             schema, synchronized sampling, preprocessing, and Dataset
src/genet/models/           Source Control, reference attention, toy model, and Cosmos adapter
src/genet/inference/        81/17 rolling windows, quality gates, rollback, and journal/resume
src/genet/integrations/     Cosmos data, loader, and clean-prefix long-horizon bridges
src/genet/training/         distributed runtime, environment lock, RF, stages, and checkpoints
src/genet/cli/              preprocessing, validation, training, lock, checkpoint, and generation CLIs
tests/                      data, model, training, checkpoint, environment, and generation tests
```

## Current Data Contract and Remaining Semantic Decisions

`genet.processed-pair/v1` requires fixed, equal Source, Target GT, and Reference lengths. Video is `[T,H,W,3]`; action
and dimension/time masks are `[T,64]`. The repository provides generic linear and nearest action resampling, but the final
dataschema still needs to define:

- action-field semantics, units, normalization, and valid dimensions for every embodiment;
- SO(3)/SE(3)-aware interpolation and frame conventions;
- the exact observation/action timestamp offset;
- train-split-only normalization statistics;
- dataset versioning and lineage for non-RoboTwin adapters (RoboTwin references already exclude the target task);
- rational timebases and terminal stop/padding semantics for long Source episodes;
- embodiment-specific joint/SE(3)/gripper quality thresholds, kinematics, and safety gates.

The fixed RoboTwin adapter now validates the structural fields, dimensions, episode ordering, overlapping action
windows, pairing key, and reference leakage policy. Its `state` signal is explicitly labeled `joint_position_state`, not
an executable actuator command. Physical timestamps, train-only normalization, coordinate frames, and robot-specific
safety semantics remain explicit schema extension points; the pipeline does not silently guess them.

## Upstream Projects and Licensing

- [NVIDIA Cosmos](https://github.com/NVIDIA/cosmos)
- [Cosmos Framework training documentation](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/training.md)
- [Cosmos3-Edge model card](https://huggingface.co/nvidia/Cosmos3-Edge)
- [Wan2.2](https://github.com/Wan-Video/Wan2.2)
- [VACE](https://github.com/ali-vilab/VACE) (zero-initialized control design reference)

GenET-owned code is Apache-2.0. Cosmos source and weights are governed by OpenMDW-1.1 and the relevant model cards. See
[THIRD_PARTY.md](THIRD_PARTY.md) for the remaining dependencies and design references.
