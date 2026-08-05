# GenET Data Preprocessing and Validation

GenET has two preprocessing inputs that produce the same fixed-shape
`genet.processed-pair/v1` training contract:

1. the fixed RoboTwin-v1 MosaicML Streaming schema used on Hyperbolic; and
2. the generic `genet.raw-pair/v1` JSONL interface for other datasets.

The RoboTwin path is the production path for the cache at
`/mnt/nvme/mds-cache/robotwin_v1`. It reads MDS directly and does not expand JPEG frames into intermediate episode
videos. The generic path remains useful when a dataset already provides video and action files with a real shared
timestamp domain.

Implementation entry points:

- RoboTwin schema and exporter: [`src/genet/data/robotwin.py`](../src/genet/data/robotwin.py)
- RoboTwin CLI: [`src/genet/cli/preprocess_robotwin.py`](../src/genet/cli/preprocess_robotwin.py)
- Versioned RoboTwin contract: [`configs/data/robotwin_v1.json`](../configs/data/robotwin_v1.json)
- Generic raw schema: [`src/genet/data/schema.py`](../src/genet/data/schema.py)
- Generic synchronized preprocessor: [`src/genet/data/preprocess.py`](../src/genet/data/preprocess.py)
- Processed Dataset: [`src/genet/data/dataset.py`](../src/genet/data/dataset.py)
- Production validator: [`src/genet/cli/validate_data.py`](../src/genet/cli/validate_data.py)

## 1. Environment

### 1.1 Production Docker image

Use the immutable GenET image for canonical preprocessing. The image contains two intentionally separate Python
environments: the default frozen Cosmos training environment and `/opt/genet-preprocess-venv` with Pillow, PyAV, and
`mosaicml-streaming==0.13.0`. Streaming requires NumPy below 2.2 while the Cosmos lock currently uses NumPy 2.2.6, so
never install the RoboTwin extra into the Cosmos venv. All canonical commands in this guide use the preprocessing venv.

The image contains the committed repository under `/opt/genet`, including all configs and scripts. A host checkout is
needed to build a new image, but it is not needed merely to run an already-built image. The host-side launch helper does
need to be available somewhere on the coordinator.

### 1.2 Local development on Ubuntu

Ubuntu 24.04 marks the system interpreter as externally managed. Do not use `pip --break-system-packages`. Create a
virtual environment instead:

```bash
# Run as root, or prefix apt-get with sudo.
apt-get update
apt-get install -y python3-full

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[preprocess,robotwin,dev]'
```

Miniconda is optional and is not required by this project.

## 2. Fixed RoboTwin-v1 schema

The supplied probe is represented by the versioned adapter contract. Its observed source-schema SHA-256 is:

```text
e97af74a65748aff2b16ee0d9bd5fd7f2299af14f04a083ea2fef0f1cf435446
```

The on-node layout is:

```text
/mnt/nvme/mds-cache/robotwin_v1/
  manifest.json
  train/<embodiment>/index.json
  train/<embodiment>/<task>/shard.*.mds.zstd
  val/<embodiment>/index.json
  val/<embodiment>/<task>/shard.*.mds.zstd
```

Each MDS row is one episode timestep with these fields:

| Field | Contract | GenET use |
| --- | --- | --- |
| `task` | string | pair key and Cosmos caption |
| `embodiment` | string | domain identity |
| `episode_idx` | integer | task-local episode identity |
| `episode_len` | integer | completeness and clip coverage |
| `t` | integer | discrete episode index |
| `head_rgb` | JPEG bytes | default video stream |
| `left_rgb`, `right_rgb` | JPEG bytes | selectable alternatives |
| `state` | float32 `[D]` | frame-aligned joint-position state |
| `action_window` | float32 `[min(100,L-t),D]` | state/future-window consistency check |
| `endpose` | float32 `[16]` | validated for schema integrity, then discarded by this adapter |

Action widths are:

| Embodiment | D |
| --- | ---: |
| `ARX-X5` | 14 |
| `aloha-agilex` | 14 |
| `franka-panda` | 16 |
| `piper` | 14 |
| `ur5-wsg` | 14 |

The adapter pads these values to the Cosmos boundary `D=64` and emits a boolean mask whose first 14 or 16 channels are
true. It calls the signal `joint_position_state`; it does not claim that a measured joint state is a low-level actuator
command.

The contract also pins the probed aggregate counts and rejects a truncated local cache:

| Embodiment | Train tasks / episodes / rows | Val tasks / episodes / rows |
| --- | ---: | ---: |
| `ARX-X5` | 22 / 2,086 / 264,725 | 22 / 110 / 13,991 |
| `aloha-agilex` | 22 / 2,090 / 382,736 | 22 / 110 / 20,333 |
| `franka-panda` | 22 / 2,090 / 248,442 | 22 / 110 / 13,132 |
| `piper` | 22 / 2,090 / 270,124 | 22 / 110 / 14,151 |
| `ur5-wsg` | 22 / 2,090 / 250,615 | 22 / 110 / 13,247 |
| **Total** | **10,446 episodes / 1,416,642 rows** | **550 episodes / 74,854 rows** |

## 3. Validation performed before export

For every aggregate stream, the adapter requires:

- every configured column is present;
- exact string, integer, bytes, float32 dtype, and shape contracts;
- finite `state`, `endpose`, and `action_window` values;
- valid JPEG start markers for all three cameras and successful decode for selected frames;
- a single contiguous row range for each `(split, task, embodiment, episode_idx)`;
- `t == 0, 1, ..., episode_len - 1` with a stable `episode_len`;
- `action_window.shape == [min(100, episode_len - t), D]`;
- `action_window[0]` agrees with the current `state`; and
- adjacent overlapping action windows agree on their common future suffix.

These checks catch offset, truncation, shard-order, and action-window corruption before expensive model training. Do not
use `--skip-action-window-validation` for a release export.

## 4. Pair and reference policy

Source and Target GT are paired only when all of the following match:

```text
split + task + episode_idx
```

Their embodiments must differ. A missing episode is reported for the affected direction and is never replaced by a
different episode. By default, every ordered pair of distinct embodiments is exported. Repeat
`--source-embodiment NAME` and `--target-embodiment NAME` to restrict storage and experiments to selected directions.

The default Reference Target is selected with a stable BLAKE2-based index from candidates that have:

- the same split as the pair;
- the same embodiment as Target GT;
- a different episode; and
- a different task.

The reference episode and its random fixed-length start are written into the processed manifest. Production training
must use `reference_mode=stored` so the release keeps this task-exclusion policy. The generic Dataset's dynamic pool only
guarantees same embodiment and different episode.

## 5. Time semantics and fixed length

The MDS schema contains integer `t` but no timestamps or authoritative physical FPS. The adapter therefore requires an
explicit `--mds-index-fps`. The recommended current release choice is:

```text
--mds-index-fps 16 --sample-fps 16
```

This means one MDS row is treated as one canonical 16 Hz GenET index. It is a declared modeling convention, not a claim
that physical capture had an exact 16 Hz clock. The choice is recorded in every stream's metadata and in `stats.json`.

The default output is `T=81`, satisfying Wan's `1 + 4N` temporal constraint. `short_policy=drop` is fixed for RoboTwin
production because the Cosmos bridge rejects temporal padding. The approved release path keeps
`--mds-index-fps 16 --sample-fps 16`, so video and joint-state rows are selected one-to-one and no action interpolation
occurs. A non-1:1 cadence is not production-safe until the unnamed action channels receive reviewed per-field semantics.

`window_policy=episode_start` writes one source/target clip from `t=0` per matched episode only when both sides cover
the full window; pairs shorter than `num_frames` are counted in `dropped_short`. This is the production policy. The
adapter also exposes the following experimental option:

```text
--window-policy sliding --clip-stride-native-frames 64
```

It applies the same native start index to embodiments whose episode lengths can differ, so use it only after approving a
task-phase retiming rule. Reference starts remain deterministic random fixed-length windows in either mode.

## 6. Canonical preprocessing command

MosaicML Streaming can lazily decompress `.mds.zstd` files beside the source shards. The canonical preparation cache
must therefore be writable during this one-time operation. Do it before computing any raw-cache content lock. Training
reads only the finished processed tree and can restore the cache mount to read-only.

For a bounded-output integration check of one direction (the adapter still scans and validates both complete aggregate
streams before `--max-samples` limits writes):

```bash
/opt/genet-preprocess-venv/bin/python -m genet.cli.preprocess_robotwin \
  --root /mnt/nvme/mds-cache/robotwin_v1 \
  --schema configs/data/robotwin_v1.json \
  --split train \
  --mds-index-fps 16 \
  --camera head \
  --source-embodiment franka-panda \
  --target-embodiment ur5-wsg \
  --max-samples 8 \
  --output /mnt/nvme/mds-cache/robotwin_v1/genet/smoke/train
```

For the canonical train export:

```bash
/opt/genet-preprocess-venv/bin/python -m genet.cli.preprocess_robotwin \
  --root /mnt/nvme/mds-cache/robotwin_v1 \
  --schema configs/data/robotwin_v1.json \
  --split train \
  --mds-index-fps 16 \
  --camera head \
  --num-frames 81 \
  --sample-fps 16 \
  --height 192 \
  --width 320 \
  --action-dim 64 \
  --action-resample linear \
  --reference-policy different_task \
  --output /mnt/nvme/mds-cache/robotwin_v1/genet/processed/train
```

Run `val` separately with a different output root. Never combine train and val candidates into one reference pool.

## 7. Processed output

The exporter writes:

```text
processed/train/
  samples/*.npz
  manifest.jsonl
  index.json
  stats.json
```

Each NPZ contains Source, Target, and Reference arrays:

```text
<role>_video        [81, 192, 320, 3] uint8
<role>_actions      [81, 64] float32
<role>_action_mask  [81, 64] bool
<role>_frame_mask   [81] bool
```

All frame masks and every real action channel are true in production output. The manifest stores split, task,
embodiment, episode index, camera, declared MDS index FPS, raw action width, action alignment, and action signal. The
stats receipt stores the committed adapter-contract checksum, the supplied observed-schema checksum, and the node's
top-level MDS `manifest.json` checksum.

## 8. Validate before training

```bash
/opt/genet-preprocess-venv/bin/python -m genet.cli.validate_data \
  --manifest /mnt/nvme/mds-cache/robotwin_v1/genet/processed/train/manifest.jsonl \
  --num-frames 81 \
  --height 192 \
  --width 320 \
  --action-dim 64 \
  --cosmos
```

`--cosmos` additionally rejects non-uint8 video, temporal padding, action-mask holes, and non-prefix action channels.
Do not train merely because sample counts look plausible; require a clean validation result and a content lock.

## 9. Four nodes without shared storage

The recommended workflow is:

1. preprocess once on a canonical node;
2. validate the canonical tree;
3. copy that processed tree byte-for-byte to the other three nodes;
4. create one logical artifact lock and verify every local copy; and
5. keep `reference_mode=stored` during training.

Independent preprocessing on all nodes is acceptable only when every result verifies against the same content lock.
The local path may be identical while the underlying filesystems remain completely separate.

## 10. Generic raw-pair path

For a non-RoboTwin dataset, print the generic JSON Schema with:

```bash
genet-preprocess --print-raw-schema
```

Each JSONL record supplies `source`, `target_gt`, and optionally `reference_target`. Every episode points to a video file
and an action file and may provide timestamps/FPS and logical start/end times. Relative paths resolve against the JSONL
directory. The generic preprocessor samples all three streams onto one fixed time grid, center-crops video, resamples
actions, pads action dimensions, and writes the same processed format:

```bash
genet-preprocess \
  --manifest /data/raw/pairs.train.jsonl \
  --output /data/processed/train \
  --config configs/schema.example.json
```

Use this path only when the dataset really has the timestamp and action semantics represented by the raw manifest. Do
not fabricate video/action files solely to route RoboTwin MDS through it.

## 11. Remaining semantic work

The structural RoboTwin schema is implemented. The following are still explicit modeling or downstream-control
decisions:

- physical timestamps and observation/action delay, which are absent from MDS;
- train-split-only normalization statistics;
- embodiment-specific units, joint ordering, coordinate frames, and limits;
- whether a future model should generate measured state, desired state, delta, torque, or another command signal;
- SE(3)-aware use of `endpose`; and
- kinematic, simulator, and real-robot safety gates.

Generated actions remain offline/simulator artifacts until those semantics and safety checks are approved.
