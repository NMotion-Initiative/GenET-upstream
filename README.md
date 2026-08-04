# GenET: Cross-Embodiment Synchronized Video–Action Generation

GenET is a Cosmos3-Edge training project for cross-robot-embodiment generation. It takes:

1. a task video–action pair from a Source Embodiment; and
2. an independently sampled, fixed-length reference video–action segment from the Target Embodiment.

The model jointly generates a Target Embodiment video and action sequence that follows the Source task semantics and
timing. Supervised training still requires a paired `target_gt`; the reference only supplies Target appearance, motion,
and action-space priors. The current `genet.processed-pair/v1` contract excludes the `target_gt` episode when selecting a
reference, but it does not yet exclude every sample from the same task. Task-level leakage prevention remains a TODO for
the final dataschema.

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
- [Long-horizon inference](docs/INFERENCE.md)

## Quick Start

Install the lightweight training, preprocessing, and test dependencies:

```bash
python -m pip install -e '.[preprocess,dev]'
```

Inspect the raw JSONL schema:

```bash
genet-preprocess --print-raw-schema
```

Synchronize and crop all three vision–action streams to `T=81`:

```bash
genet-preprocess \
  --manifest data/raw/train.jsonl \
  --output data/processed/train \
  --config configs/schema.example.json

genet-validate-data \
  --manifest data/processed/train/manifest.jsonl \
  --num-frames 81 --height 192 --width 320 --action-dim 64 \
  --cosmos
```

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

## Cosmos3-Edge Production Environment

The project pins Cosmos Framework to:

```text
a904d2d36b774a51dd06ff9ff906816b1a04f579
```

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
  -t registry.example/genet:${GENET_REVISION} -

docker push registry.example/genet:${GENET_REVISION}
```

Building from `git archive HEAD` deliberately excludes dirty and untracked working-tree files. The virtual revision file
is checked against the image build argument, remains available to the runtime preflight, and prevents arbitrary source
from being labeled as that commit. The image retains the versioned `configs/` and `scripts/`, installs GenET as a wheel,
and writes a sorted `pip freeze` plus its checksum into `/opt`. Record the pushed image digest; never rely on a mutable tag
for a multi-node run.

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
run the CPU/Gloo node preflight, nccl-tests, then a 4-node GenET dry run
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
  --artifact wan_vae=/staging/genet/checkpoints/wan22_vae/Wan2.2_VAE.pth \
  --artifact training_checkpoint=/staging/genet/checkpoints/stage2_shared_committed \
  --artifact normalization=/staging/genet/data/normalization \
  --artifact hf_cache=/staging/genet/hf-cache
```

If the temporary dataschema has no normalization artifact yet, omit that logical entry from both the create and verify
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
  --artifact wan_vae=/local_nvme/genet/checkpoints/wan22_vae/Wan2.2_VAE.pth \
  --artifact training_checkpoint=/local_nvme/genet/checkpoints/stage2_shared_committed \
  --artifact normalization=/local_nvme/genet/data/normalization \
  --artifact hf_cache=/local_nvme/genet/hf-cache
```

The node-local receipt binds logical artifacts to the paths that were actually verified. Strict Cosmos startup rejects a
manifest, Wan VAE, HF cache, or load checkpoint that is not bound to that receipt. The prologue must stop the complete job
when any node fails verification; it must also keep the verified release read-only until the job finishes.

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
export GENET_HF_SNAPSHOT_REVISION='<Cosmos3-Edge-HF-commit>'
export WAN_VAE_PATH=/local_nvme/genet/checkpoints/wan22_vae/Wan2.2_VAE.pth
export HF_HOME=/local_nvme/genet/hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

Before starting the 32-rank NCCL group, run the one-process-per-node CPU/Gloo preflight on all four nodes:

```bash
bash scripts/preflight_roce.sh
```

It verifies the strict runtime identity, one unique physical node identity per rank, eight visible GPUs per node, GPU
model/capability/memory, and the NVIDIA driver, then compares all four node reports over a bounded, dedicated
`PREFLIGHT_PORT`. This catches environment drift before the expensive NCCL model process group is created. Run multi-node
`nccl-tests` separately to validate the actual RoCE data path.

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

## Current Data Contract and Dataschema TODOs

`genet.processed-pair/v1` requires fixed, equal Source, Target GT, and Reference lengths. Video is `[T,H,W,3]`; action
and dimension/time masks are `[T,64]`. The repository provides generic linear and nearest action resampling, but the final
dataschema still needs to define:

- action-field semantics, units, normalization, and valid dimensions for every embodiment;
- SO(3)/SE(3)-aware interpolation and frame conventions;
- the exact observation/action timestamp offset;
- train-split-only normalization statistics;
- task-level reference exclusion, dataset versioning, and lineage;
- rational timebases and terminal stop/padding semantics for long Source episodes;
- embodiment-specific joint/SE(3)/gripper quality thresholds, kinematics, and safety gates.

These are explicit schema extension points. The pipeline must not silently guess robot action semantics.

## Upstream Projects and Licensing

- [NVIDIA Cosmos](https://github.com/NVIDIA/cosmos)
- [Cosmos Framework training documentation](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/training.md)
- [Cosmos3-Edge model card](https://huggingface.co/nvidia/Cosmos3-Edge)
- [Wan2.2](https://github.com/Wan-Video/Wan2.2)
- [VACE](https://github.com/ali-vilab/VACE) (zero-initialized control design reference)

GenET-owned code is Apache-2.0. Cosmos source and weights are governed by OpenMDW-1.1 and the relevant model cards. See
[THIRD_PARTY.md](THIRD_PARTY.md) for the remaining dependencies and design references.
