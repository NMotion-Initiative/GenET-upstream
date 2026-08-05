# GenET 训练指南

本文描述如何在 4 节点、每节点 8 张 GPU 的 RoCE 集群上训练 GenET，以及如何在单节点 8 卡上完成 reference projection 消融。目标任务是：给定 Source Embodiment 的视频与动作，以及从 Target Embodiment 随机抽取的固定长度 reference 视频与动作，同步生成与 Source 任务一致的 Target 视频与动作。

本文中的相对路径都以仓库根目录为起点。生产训练使用 NVIDIA Cosmos3-Edge；仓库内的轻量 `toy` backend 只用于数据契约、梯度和单批过拟合检查。

## 1. 当前实现边界

先区分仓库已经实现的能力与后续数据 schema 仍需补齐的部分：

- 项目配置由 [`configs/base.yaml`](../configs/base.yaml) 提供默认值，并由 [`src/genet/config.py`](../src/genet/config.py) 严格校验。实验配置通过 `_base_` 继承；未知字段会直接报错。
- Cosmos3-Edge 适配器位于 [`src/genet/models/cosmos_adapter.py`](../src/genet/models/cosmos_adapter.py)，生产 experiment builder 位于 [`src/genet/training/cosmos.py`](../src/genet/training/cosmos.py)。builder 从固定上游版本的 `vision_sft_edge` recipe 构建配置，复用 Cosmos 的 VAE 编码、rectified-flow noising/loss、FSDP、EMA 和 DCP，只替换 GenET 数据与 Source/Reference 条件路径。
- 当前 Cosmos 适配器要求每个 rank 一个 packed sample、two-way attention、`context_parallel_shard_degree=1`，且 `video_temporal_causal=false`。不要把 CP 改为 2 后期待当前代码可以运行。
- Source Control 和 reference projection 的轻量可测试实现分别位于 [`src/genet/models/control.py`](../src/genet/models/control.py) 与 [`src/genet/models/reference_attention.py`](../src/genet/models/reference_attention.py)。
- RoboTwin-v1 的固定 MDS schema、计数、episode、配对与 reference 规则已经实现并硬校验；源数据未提供的 action 单位/坐标系、旋转语义、物理时间戳与不同本体阶段重定时仍需作为显式模型契约补充。当前 release 固定使用 16→16 Hz，不做 action 插值。
- [`src/genet/training/checkpoint.py`](../src/genet/training/checkpoint.py) 同时包含 standalone checkpoint 和无共享文件系统下的 DCP 汇总工具。两者格式不同，不可混用。

## 2. 依赖、上游代码与权重

### 2.1 软件与硬件前提

建议使用 Linux、NVIDIA 驱动、CUDA、PyTorch 与 NCCL 版本相互匹配的 NVIDIA PyTorch 容器。Cosmos3-Edge 模型卡只测试了 BF16；本项目的生产配置也使用 `bfloat16`。

基础要求：

- Python 3.10 或更高版本；
- 轻量 toy/预处理路径要求 PyTorch 2.4 或更高版本；production 必须服从固定 Cosmos commit 的容器依赖（该版本官方 setup 固定 PyTorch 2.10），不要在自行拼装的 2.4 环境中运行 Cosmos；
- 每节点 8 张 GPU，节点内建议 NVLink/NVSwitch；
- 节点间 RoCEv2，且交换机 PFC/ECN、MTU、traffic class 已由集群管理员正确配置；
- 每节点本地 NVMe 上有一份完整、内容一致的数据副本；
- archive 主机、对象存储，或可通过 SSH/rsync 到达的 checkpoint 汇总位置。训练节点之间不需要共享数据文件系统。

安装项目的轻量依赖、预处理依赖和开发工具：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'huggingface_hub[cli]'
python -m pip install -e '.[preprocess,robotwin,dev]'
```

`pyproject.toml` 中的核心依赖只有 `torch`、`numpy` 和 `pyyaml`；视频预处理额外使用 PyAV 和 Pillow。

### 2.2 固定 Cosmos Framework 版本

仓库不 vendor 上游源码。[`scripts/bootstrap_cosmos.sh`](../scripts/bootstrap_cosmos.sh) 会把 Cosmos Framework 固定到：

```text
a904d2d36b774a51dd06ff9ff906816b1a04f579
```

执行：

```bash
bash scripts/bootstrap_cosmos.sh
python -m pip install -e 'third_party/cosmos-framework[train]'
python -m pip install -e .
```

如果通过 `COSMOS_DIR` 把上游 checkout 放到其他本地目录，第二条命令也应使用同一路径。四个节点必须使用同一容器镜像、同一 GenET commit 和同一 Cosmos commit。

第三方代码、权重许可证及固定版本说明见 [`THIRD_PARTY.md`](../THIRD_PARTY.md)。GenET 自身为 Apache-2.0；Cosmos Framework/Cosmos3 权重受 OpenMDW-1.1 约束，Wan2.2 VAE 以其模型卡条款为准。

### 2.3 Stage Cosmos3-Edge and the Wan2.2 VAE

Both reviewed repositories are public at the pinned revisions. If access policy later requires authentication, inject
an ephemeral `HF_TOKEN` only into the short-lived staging container; never run `hf auth login` after `HF_HOME` has been
set to the cache that will be copied to the training nodes, and never inject the token into training:

```bash
export HF_TOKEN='<short-lived-read-token>'
bash scripts/stage_model_artifacts.sh --run-root /mnt/nvme/genet
unset HF_TOKEN
```

The reviewed identities and official locations are committed in
[`configs/checkpoints/cosmos3_edge.json`](../configs/checkpoints/cosmos3_edge.json):

- `nvidia/Cosmos3-Edge` at `2a00e87e9976dc3ed5533dd18caf4cdbc3a1bcb2`;
- `Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth` at
  `921dbaf3f1674a56f47e83fb80a34bac8a8f203e`;
- Wan VAE SHA-256 `20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36` and byte count
  `2818839170`.

Run the staging script inside the immutable GenET Cosmos image on one canonical node:

```bash
bash scripts/stage_model_artifacts.sh --run-root /mnt/nvme/genet
```

It downloads the complete exact HF snapshot, verifies all mandatory model files and every weight-index shard, writes the
offline `refs/main`, downloads and verifies the VAE, passes the exact local snapshot into the pinned Cosmos converter,
validates `model/.metadata`, writes a recursive SHA-256 manifest for the complete DCP, and writes
`/mnt/nvme/genet/artifacts/ARTIFACTS.json`. The resulting training load path is
`/mnt/nvme/genet/checkpoints/Cosmos3-Edge`, not its `model/` child. Verify later without networking:

```bash
bash scripts/stage_model_artifacts.sh --run-root /mnt/nvme/genet --verify-only
```

Copy the completed HF cache, Wan artifact, `ARTIFACTS.json`, and converted DCP byte-for-byte to the other nodes. Run
`--verify-only` and then verify the release through the cluster lock on each node. The production environment uses:

```bash
export HF_HOME=/mnt/nvme/genet/hf-cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export GENET_HF_SNAPSHOT_REVISION=2a00e87e9976dc3ed5533dd18caf4cdbc3a1bcb2
export WAN_VAE_PATH=/mnt/nvme/genet/artifacts/wan22_vae/Wan2.2_VAE.pth
```

No separate full Wan DiT, Reasoner, DROID policy, ControlNet, reference encoder, or action head checkpoint is required.
The new GenET branches initialize during S1; S2/S3 load the selected previous-stage GenET DCP.

### 2.4 无共享存储的环境与内容锁

四个节点不能各自执行不带 lock 的 `pip install` 并假定环境相同。生产作业应从
[`containers/Dockerfile`](../containers/Dockerfile) 构建一次镜像，固定 base image digest、GenET commit 与
Cosmos commit，再让四个节点拉取同一个 OCI digest；Apptainer 集群则只转换一次并复制同一个 SIF。必须按
README 使用 `git archive HEAD` 加入 `GENET_BUILD_REVISION`，不能用 dirty worktree 直接构建并贴上 HEAD 标签。

在 staging 副本上为数据、VAE、DCP、normalization 和固定 HF snapshot 建立内容锁：

```bash
genet-cluster-lock create \
  --output /staging/genet-release/cluster-lock.json \
  --artifact processed_data=/staging/genet/data/processed/train \
  --artifact wan_vae=/staging/genet/artifacts/wan22_vae/Wan2.2_VAE.pth \
  --artifact training_checkpoint=/staging/genet/checkpoints/stage2_shared_committed \
  --artifact normalization=/staging/genet/data/normalization \
  --artifact hf_cache=/staging/genet/hf-cache
```

复制到每个节点的本地 NVMe 后，在 scheduler prologue 中用相同逻辑名称验证本地路径：

```bash
genet-cluster-lock verify \
  --lock /local_nvme/genet/release/cluster-lock.json \
  --receipt /local_nvme/genet/release/node-receipt.json \
  --artifact processed_data=/local_nvme/genet/data/processed/train \
  --artifact wan_vae=/local_nvme/genet/artifacts/wan22_vae/Wan2.2_VAE.pth \
  --artifact training_checkpoint=/local_nvme/genet/checkpoints/stage2_shared_committed \
  --artifact normalization=/local_nvme/genet/data/normalization \
  --artifact hf_cache=/local_nvme/genet/hf-cache
```

`training_checkpoint` 必须是本作业实际通过 `--warm-start`、`--resume` 或 `BASE_CHECKPOINT_PATH` 加载的 DCP；
S1、S2/S3 和 resume 作业应分别替换成各自真实路径。当前 release 尚未生成批准的 normalization 文件时，
create/verify 两边同时删掉该逻辑项。lock 会扫描目录中的每个
regular file；HF cache 使用的内部 symlink 会按 root-relative target 记录，broken link 或逃逸 artifact root 的
link 会被拒绝。这一步可能较慢，但只在 staging/deployment 阶段执行，不在每个训练 step 执行。
receipt 会把逻辑 artifact 绑定到本节点实际验证过的路径；训练启动会拒绝使用 receipt 之外的 manifest、VAE、
HF cache 或 load DCP。verified release 在作业期间必须只读。任一节点失败时 scheduler 必须终止整个作业，
不能让该节点继续加入 `torchrun`。

提交作业时启用 `GENET_STRICT_ENV=1`，并设置镜像 digest、GenET/Cosmos revision、cluster lock 和固定 HF
revision，示例见 [`configs/cluster/roce_4x8.env.example`](../configs/cluster/roce_4x8.env.example)。训练入口还会
跨 rank 自动比较 Python/Torch/CUDA/cuDNN/NCCL、GPU 型号、package-set hash、installed GenET Python-source
hash、embedded build revision、可发现的 Git revision、cluster-lock hash 与 path-independent receipt contract。
内容锁负责“大文件实际字节”，receipt 负责运行路径绑定，collective 负责“rank 环境漂移”；三者不能互相替代。

正式训练前先运行 one-process-per-node CPU/Gloo identity preflight，再运行 one-process-per-GPU 的 32-rank
NCCL correctness/bandwidth preflight：

```bash
bash scripts/preflight_roce.sh
bash scripts/preflight_nccl.sh
```

它使用带 timeout 的独立 `PREFLIGHT_PORT` 比较四个节点的严格运行时 identity，拒绝重复物理 node identity，
并检查每节点可见 GPU 数量、型号、compute capability、显存与 NVIDIA driver。随后 NCCL preflight 通过独立
`NCCL_PREFLIGHT_PORT` 在全部 GPU 上验证 all-reduce 数值并报告带宽。必须从 `NCCL_DEBUG=INFO` 日志确认选择
`NET/IB` 而非非预期 Socket fallback；新 allocation/fabric 仍需按第 9 节用 provider baseline 跑官方
nccl-tests，内置 preflight 不是完整硬件带宽验收的替代品。

## 3. 数据训练前检查

原始数据格式示例与预处理默认项见 [`configs/schema.example.json`](../configs/schema.example.json)。正式训练默认形状来自 [`configs/base.yaml`](../configs/base.yaml)：

| 项目 | 默认值 |
| --- | ---: |
| FPS | 16 |
| Source/Target/Reference 帧数 | 81 |
| 分辨率 | 320×192 |
| 最大 action 维度 | 64 |
| temporal compression factor | 4 |

Wan causal VAE 要求帧数满足：

```text
T = 1 + N × temporal_compression_factor
```

因此默认值 81 合法。`genet.processed-pair/v1` 明确定义 Source、Target 和 Reference 使用同一个固定 T；[`configs/base.yaml`](../configs/base.yaml) 中 `num_frames` 与 `reference_num_frames` 均为 81，配置校验也要求二者相等。`genet-validate-data` 会进一步检查三路视频和 action shape 一致。

在生产镜像中，一个与训练默认形状一致的预处理命令如下（本地开发 venv 可去掉绝对路径前缀）：

```bash
/opt/genet-preprocess-venv/bin/python -m genet.cli.preprocess_robotwin \
  --root /mnt/nvme/mds-cache/robotwin_v1 \
  --schema configs/data/robotwin_v1.json \
  --split train \
  --mds-index-fps 16 \
  --output /mnt/nvme/mds-cache/robotwin_v1/genet/processed/train \
  --num-frames 81 \
  --sample-fps 16 \
  --height 192 \
  --width 320 \
  --action-dim 64 \
  --action-resample linear \
  --reference-policy different_task \
  --require-bidirectional
```

预处理输出的 manifest 为 `/mnt/nvme/mds-cache/robotwin_v1/genet/processed/train/manifest.jsonl`。用
`--manifest` 显式传入该节点本地路径。

在所有节点上验证本地副本：

```bash
genet-validate-data \
  --manifest /mnt/nvme/mds-cache/robotwin_v1/genet/processed/train/manifest.jsonl \
  --num-frames 81 \
  --height 192 \
  --width 320 \
  --action-dim 64 \
  --cosmos \
  --require-bidirectional \
  --expected-embodiment ARX-X5 \
  --expected-embodiment aloha-agilex \
  --expected-embodiment franka-panda \
  --expected-embodiment piper \
  --expected-embodiment ur5-wsg
```

然后计算并比较 manifest 与统计文件的 SHA256。至少校验：

- processed manifest；
- 预处理 `index.json`/`stats.json`；
- action normalization 文件；
- resolved 训练配置；
- GenET 与 Cosmos git revision；
- 基础 DCP 和 Wan VAE fingerprint。

训练时禁止“遇到坏样本就跳过”。不同 rank 跳过不同数量的样本会让 collective 次数不一致并最终 hang；坏样本应在预处理或 `genet-validate-data` 阶段隔离。

RoboTwin 正式数据按有向本体组合物化：5 个本体对应 20 个方向，每个 `A→B` 都有使用同一
task/episode/window 的 `B→A`，但两个方向各自从“当前 Target 本体”选择 stored Reference。不能在 Dataset
里只交换 Source/Target tensor；否则 reference、domain、raw action dim 和 mask 都会属于错误角色。六个
Cosmos production 配置均设置 `data.require_bidirectional_pairs: true`，启动前会检查 exact reverse、方向计数、
每本体 Source/Target 边际，并在日志输出 `pair_direction_summary`。

`pair_direction_summary` 只描述静态 manifest 的 reciprocal inventory，不是训练至当前 step 已消费样本的统计。
训练 sampler 对 directed manifest 做无放回、可恢复的 rank-local epoch shuffle，不使用 replacement 权重采样；
每个 epoch 全局消费 `floor(N/WORLD_SIZE)×WORLD_SIZE` 条记录，最多省略 `WORLD_SIZE-1` 条，并通过 epoch rotation
让尾部记录不会永久丢失。因此完整 manifest 保持 A↔B 对称，但单个 epoch（尤其被 tail 截断时）和任意
`max_steps` prefix 都不保证两个方向曝光数严格相等。需要审计实际利用率时，应从已消费 batch metadata 另行累计
per-direction exposure，而不能把启动时的 manifest summary 当作消费计数。

比较 shared/dual/no-ref 时必须固定同一 directed manifest、reference、seed、max steps 和 global batch，并同时
报告 manifest 的 `directed_samples`、`unique_pair_windows`、per-direction inventory，以及训练实际消费的
per-direction exposure。`--max-samples` 按方向顺序提前停止，只允许单向 smoke test，不能进入 production lock。
未来 action normalization 也必须按 train split 的 unique 本体/episode/timestep 统计，不能直接使用因 20 个方向
重复曝光而加权的 role aggregate。

严格 release 同时固定四层条件：export 使用 `--reference-policy different_task`，production config 使用
`data.reference_mode=stored`，canonical validator 要求上述五个 `--expected-embodiment` 与 exact reverse，cluster
content lock 则哈希 manifest、全部 NPZ、`index.json`、`stats.json` 和批准的 normalization artifact。任何一层变化
都必须生成新的 release ID 与 lock。
validator 还会从解压后的四类数组重新计算每一路 `content_sha256`，并要求反向记录交换 Source/Target hash，防止
仅交换 metadata、却继续使用无关 payload 的伪反向样本通过 release gate。

Cosmos3-Edge production bridge 还有三项硬约束：

- `data.action_dim` 必须是 64，原始本体动作可以少于 64 维，但有效通道必须是从 0 开始的连续前缀；bridge 会补零到 64 并传递 `raw_action_dim`；
- `model.num_embodiments` 必须是 32，manifest 中实际本体名按排序稳定映射到 32 个 domain slot，实际种类不能超过 32；
- 当前 Cosmos loss bridge 尚不消费 temporal padding mask，因而 production 数据必须用 `short_policy=drop`。只要任一路 `frame_mask` 含无效帧，或 action 在时间轴上含 padded step，bridge 会 fail-fast，而不是把 padding 送入 loss。
- `data.require_bidirectional_pairs` 在所有 S1/S2/S3 production 配置中必须保持开启；关闭只适用于明确标记的 directed smoke/ablation。

训练配置的 `data.reference_mode` 默认为 `stored`，直接使用预处理时写入 NPZ 的 reference。切到 `deterministic` 后，Dataset 会依据 `data.reference_seed` 从 processed manifest 的同本体、不同 target episode 中稳定重选；两种模式会改变实验数据，必须写入 run metadata，shared/dual 间不得混用。

## 4. 实验配置

已有配置如下：

| 阶段 | 配置 | GPU | reference |
| --- | --- | ---: | --- |
| S1 | [`configs/experiments/stage1_control_32gpu.yaml`](../configs/experiments/stage1_control_32gpu.yaml) | 32 | 关闭 |
| S2 | [`configs/experiments/stage2_no_ref_8gpu.yaml`](../configs/experiments/stage2_no_ref_8gpu.yaml) | 8 | 关闭 |
| S2 | [`configs/experiments/stage2_shared_8gpu.yaml`](../configs/experiments/stage2_shared_8gpu.yaml) | 8 | shared，AR+DM |
| S2 | [`configs/experiments/stage2_dual_8gpu.yaml`](../configs/experiments/stage2_dual_8gpu.yaml) | 8 | dual，AR+DM |
| S3 | [`configs/experiments/stage3_shared_32gpu.yaml`](../configs/experiments/stage3_shared_32gpu.yaml) | 32 | shared，AR+DM |
| S3 | [`configs/experiments/stage3_dual_32gpu.yaml`](../configs/experiments/stage3_dual_32gpu.yaml) | 32 | dual，AR+DM |

`shared` 和 `dual` 的唯一主变量是 reference K/V projector 是否共享：

- `shared`：两个消费路由复用 `route_a`；
- `dual`：`route_a`、`route_b` 独立，但 `route_b` 从 `route_a` 复制初始化；
- `dual_tied`：配置类型支持的数值等价/回归检查模式，两个路由仍复用同一 projector。

Query projection、output projection、gate、reference token、注入层数和 dropout 在 shared/dual 之间保持一致。迁移逻辑位于 [`migrate_reference_projectors`](../src/genet/models/reference_attention.py)；shared 到 dual 可以复制，dual 到 shared 不做隐式平均。

[`configs/experiments/stage2_dual_8gpu.yaml`](../configs/experiments/stage2_dual_8gpu.yaml) 显式设置 `checkpoint.copy_shared_reference_to_dual: true`。这个开关只用于从 shared/common checkpoint 分支到 dual 的第一次 warm-start；已经是 dual 的 S2 → S3 不应再次做结构迁移。

Cosmos adapter 在上游 DCP 完成加载后复制 route-A 到 route-B，并在 EMA 开启时同步处理 EMA；builder 在显式 `--resume` 时会先禁用迁移，adapter 也会再次检查上游 resumable 状态，双重保证 exact resume 不会覆盖已经分化的 dual 权重。

## 5. 有效 batch 与公平消融

一般公式为：

```text
effective_global_batch
  = micro_batch_size
  × grad_accum_steps
  × data_parallel_shard_degree
  × data_parallel_replicate_degree
```

等价地，在 CFG/CP 不复制数据样本的实现中：

```text
effective_global_batch
  = micro_batch_size × grad_accum_steps
  × WORLD_SIZE / context_parallel_degree / cfg_parallel_degree
```

当前所有已提供配置都是 `CP=1`、`CFG parallel=1`、每 rank 一个样本，因此：

- 32 卡：`1 × 1 × 32 = 32`；
- 8 卡：`1 × 4 × 8 = 32`。

这正是 stage2 配置将 `grad_accum_steps` 设为 4、stage1/stage3 设为 1 的原因。不要因为单节点吞吐较低就减少 accumulation；否则消融同时改变了 batch、每步样本数和 scheduler 语义。

公平比较 shared 与 dual 至少固定以下项目：

- 同一个 S1 warm-start 权重；
- 同一份 manifest、reference 选择、样本顺序与固定 validation reference；
- 相同有效 batch、优化器步数、warmup、LR、condition dropout 和 loss 权重；
- 相同随机种子集合，建议至少 3 个 seed；
- 相同注入层、reference video/action 开关、gate 初始化与冻结策略；
- 同时报告 trainable parameter 数和 FLOPs。shared/dual 的参数量差异是参数共享问题本身的一部分，不能通过增加无效参数伪装“参数匹配”。

production adapter 会把 `train.condition_dropout` 分别应用到 Source 和 Reference，并以固定的 Source→Reference 顺序每 step 恰好抽取两个随机数。因此 shared 与 dual 在相同 seed、相同数据顺序和相同 resume 状态下会获得相同的 conditioning dropout 日程。

主实验至少包括 `no_ref`、`shared`、`dual`。建议增加 `dual_tied` 数值回归、wrong-embodiment/shuffled-reference 诊断，以及 video-only、action-only 条件诊断，但这些附加配置当前未作为独立 YAML 提交。

## 6. S0–S4 训练流程

### S0：数据与单批过拟合

目的不是获得可用模型，而是验证：

- 三路视频/action shape 与 mask；
- `T=1+4N`；
- target、source、reference embodiment/domain id；
- Source Control 的 zero-init 初始为严格 no-op；
- shared 与 dual 在复制初始化后 step 0 数值一致；
- video/action flow loss 都有有限梯度。

先运行轻量测试：

```bash
pytest -q
```

使用 [`configs/base.yaml`](../configs/base.yaml) 的 `model.backend: toy` 在一张 GPU 上做很短的训练。先在 processed 数据目录中生成一个只含一条记录、仍能按相对路径访问原 NPZ 的 manifest：

```bash
sed -n '1p' data/processed/train/manifest.jsonl \
  > data/processed/train/s0_one_sample.jsonl
```

训练入口支持命令行覆盖 manifest、步数和输出目录：

```bash
CUDA_VISIBLE_DEVICES=0 torchrun --standalone --nproc-per-node=1 \
  -m genet.cli.train \
  --config configs/base.yaml \
  --manifest data/processed/train/s0_one_sample.jsonl \
  --max-steps 20 \
  --output-dir outputs/s0_toy_overfit
```

先用 `--dry-run` 只验证 config、数据与模型构造，再移除它进行 optimizer steps。单批应能明显过拟合；若不能，先不要提交 Cosmos 作业。

### S1：Source Control 公共起点

使用 [`configs/experiments/stage1_control_32gpu.yaml`](../configs/experiments/stage1_control_32gpu.yaml)：

- reference 关闭；
- 冻结 VAE 与基础 MoT；
- 训练 Source Control 的视觉与动作分支新参数；
- zero-init 保证刚开始不破坏基础生成器；
- 产出 shared、dual、no-ref 三组消融共同使用的 committed checkpoint。

### S2：单机 8 卡筛选

将同一个 S1 checkpoint 以 `checkpoint.warm_start` 分别交给：

- `stage2_no_ref_8gpu.yaml`；
- `stage2_shared_8gpu.yaml`；
- `stage2_dual_8gpu.yaml`。

四个节点各有完整数据副本时，可同时让三个节点各跑一个实验，第四个节点跑 `dual_tied`、shuffled reference 或第二个 seed。每个实验都是独立的 `WORLD_SIZE=8`，不能把四个单机作业误组成一个 32-rank process group。

### S3：32 卡确认实验

对 S2 中的 shared 和 dual 主方案分别运行：

- [`configs/experiments/stage3_shared_32gpu.yaml`](../configs/experiments/stage3_shared_32gpu.yaml)；
- [`configs/experiments/stage3_dual_32gpu.yaml`](../configs/experiments/stage3_dual_32gpu.yaml)。

`train.stage: joint` 会在 Control/Reference 参数之外启用 Cosmos generator/action boundary 的选择性训练参数，同时继续冻结 video tokenizer。stage3 把新模块 LR 从基础默认的 `1e-4` 降到 `5e-5`，base LR 为 `1e-5`。

### S4：高分辨率或长时序低 LR 微调

仓库当前没有提交 S4 YAML。应从获胜的 S3 配置继承，逐项改变，而不是同时改变分辨率、时长、batch 和冻结策略：

1. 先延长 T，保持分辨率；
2. 再提高到 480p；
3. 降低 LR；
4. 重新计算 accumulation，保持目标有效 batch；
5. 重新确认 `T=1+4N`、本地 NVMe 带宽与显存峰值。

当前 adapter 明确拒绝 `CP>1`，因此 S4 OOM 时优先使用 full activation checkpointing、减小 T/分辨率或等待 CP 支持，不要直接配置 CP=2。

## 7. 单节点 8 卡命令

单节点消融可以直接使用 `torchrun --standalone`。以下三条命令应在不同节点或依次执行；每个作业使用各自配置中不同的 output directory：

```bash
torchrun --standalone --nproc-per-node=8 \
  -m genet.cli.train \
  --config configs/experiments/stage2_no_ref_8gpu.yaml \
  --manifest /local_nvme/genet/data/train/manifest.jsonl \
  --warm-start /local_nvme/genet/checkpoints/stage1_committed \
  --output-dir /local_nvme/genet/outputs/stage2_no_ref_seed42
```

```bash
torchrun --standalone --nproc-per-node=8 \
  -m genet.cli.train \
  --config configs/experiments/stage2_shared_8gpu.yaml \
  --manifest /local_nvme/genet/data/train/manifest.jsonl \
  --warm-start /local_nvme/genet/checkpoints/stage1_committed \
  --output-dir /local_nvme/genet/outputs/stage2_shared_seed42
```

```bash
torchrun --standalone --nproc-per-node=8 \
  -m genet.cli.train \
  --config configs/experiments/stage2_dual_8gpu.yaml \
  --manifest /local_nvme/genet/data/train/manifest.jsonl \
  --warm-start /local_nvme/genet/checkpoints/stage1_committed \
  --output-dir /local_nvme/genet/outputs/stage2_dual_seed42
```

三个作业必须指向各自节点的本地完整数据副本，并从同一个已预分发 S1 checkpoint warm-start。若通过 scheduler 并发执行，给每个作业分配不同的 job name、日志目录和随机种子；不要让它们写同一 `checkpoint.output_dir`。

## 8. 4 节点 32 卡 torchrun

集群环境示例位于 [`configs/cluster/roce_4x8.env.example`](../configs/cluster/roce_4x8.env.example)。不要原样复制接口、HCA 或 IP；先按第 9 节发现实际网络。

四个节点都执行以下准备，其中 `NODE_RANK` 分别为 0、1、2、3：

```bash
source configs/cluster/roce_4x8.env.example
export NODE_RANK=0                 # 其他节点依次为 1/2/3
```

然后四个节点近似同时运行相同配置：

```bash
bash scripts/launch_roce.sh \
  configs/experiments/stage3_shared_32gpu.yaml \
  --manifest /local_nvme/genet/data/train/manifest.jsonl \
  --warm-start /local_nvme/genet/checkpoints/stage2_shared_committed \
  --output-dir /local_nvme/genet/outputs/stage3_shared_seed42
```

dual 确认实验只替换配置：

```bash
bash scripts/launch_roce.sh \
  configs/experiments/stage3_dual_32gpu.yaml \
  --manifest /local_nvme/genet/data/train/manifest.jsonl \
  --warm-start /local_nvme/genet/checkpoints/stage2_dual_committed \
  --output-dir /local_nvme/genet/outputs/stage3_dual_seed42
```

S1 使用：

```bash
bash scripts/launch_roce.sh \
  configs/experiments/stage1_control_32gpu.yaml \
  --manifest /local_nvme/genet/data/train/manifest.jsonl \
  --warm-start /local_nvme/genet/checkpoints/Cosmos3-Edge \
  --output-dir /local_nvme/genet/outputs/stage1_control_seed42
```

对 Cosmos backend，`--output-dir` 设置的是 `IMAGINAIRE_OUTPUT_ROOT`，不是最终 run directory。production builder 固定 `job.project=genet`、`job.group=cross_embodiment`、`job.name=run`，因此上例 S3 shared 的实际 run directory 为：

```text
/local_nvme/genet/outputs/stage3_shared_seed42/genet/cross_embodiment/run
```

DCP、resolved config 和训练日志都应从这个目录下查找。standalone/toy backend 则直接把 `--output-dir` 当作 checkpoint 根目录。

[`scripts/launch_roce.sh`](../scripts/launch_roce.sh) 会调用：

```text
torchrun --nnodes=4 --nproc-per-node=8 --node-rank=<0..3>
         --master-addr=<MASTER_ADDR>
         --master-port=<MASTER_PORT>
         --max-restarts=0
         -m genet.cli.train --config <CONFIG>
```

这里故意使用 static rendezvous，使 scheduler 给出的 `NODE_RANK=0..3` 真正决定 global-rank block，并与无共享
checkpoint 的 node publish ID 保持一致。c10d elastic rendezvous 会按 join order 分配 group rank，不能把它与
手工 `NODE_RANK` 混用。所有节点必须使用完全相同的 `NNODES`、`NPROC_PER_NODE`、`MASTER_ADDR`、
`MASTER_PORT` 和配置内容；只有 `NODE_RANK` 不同。`MASTER_ADDR` 是 rank-0 节点所有节点均可访问的 bootstrap
地址，不一定是 RDMA HCA 地址；并发作业必须使用不同 `MASTER_PORT`。

第一次提交某个配置时，先在四节点命令末尾增加 `--dry-run`。production dry-run 会初始化 process group、
比较运行时环境 contract，解析固定 upstream recipe、确认 DCP 目录含 `.metadata`、确认 `WAN_VAE_PATH` 存在、
比较 manifest/语义配置/embodiment map，并由每个节点的 8 个 local rank 并行分片完成一次等价于
`genet-validate-data --cosmos --require-bidirectional` 的全量 shape/reciprocity 校验；随后每个 rank 再读取、转换
自己的首个 global-rank shard 样本。这个 dry-run 不替代预处理阶段带五个 `--expected-embodiment` 的 canonical
inventory 验证，也不替代 content-lock hash verification。
它不会构造昂贵模型或进入 optimizer step。通过后再去掉该参数。

## 9. RoCE/NCCL 环境与诊断

### 9.1 发现接口

在每个节点执行：

```bash
nvidia-smi topo -m
ibdev2netdev
show_gids
ip -br address
```

需要确认：

- `NCCL_SOCKET_IFNAME` 对应可用于 rendezvous/bootstrap 的以太网接口；
- `NCCL_IB_HCA` 只包含本作业应使用的 RoCE HCA/端口；
- 四节点 MTU、VLAN、RoCEv2 GID address family 一致；
- GPU 与 HCA 的 NUMA/PCIe 亲和关系合理。

对两个节点先做 verbs 连通性检查。服务端：

```bash
rping -s -a <server-roce-ip> -V -C 10
```

客户端：

```bash
rping -c -a <server-roce-ip> -S <client-roce-ip> -V -C 10
```

### 9.2 推荐环境变量

示例文件已经包含：

```bash
export NCCL_SOCKET_IFNAME=eno1
export NCCL_IB_HCA=mlx5_0,mlx5_1
export NCCL_IB_DISABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,NET,ENV
```

NCCL 2.21 及以上版本优先让 NCCL 动态选择 RoCE GID，不要设置 `NCCL_IB_GID_INDEX`。老版本只有在 `show_gids` 确认 RoCEv2 index 后才设置该变量。

`NCCL_IB_TC`、`NCCL_IB_SL`、`NCCL_NET_GDR_LEVEL` 与交换机 PFC/ECN、路由及 GPU-HCA 拓扑有关，必须由集群管理员给定或通过 nccl-tests 验证；禁止从其他集群照抄。

### 9.3 nccl-tests

正式训练前至少运行多节点 all-reduce：

```bash
mpirun -np 32 -N 8 \
  -x NCCL_SOCKET_IFNAME \
  -x NCCL_IB_HCA \
  -x NCCL_IB_DISABLE=0 \
  -x NCCL_DEBUG=INFO \
  /path/to/nccl-tests/build/all_reduce_perf \
  -b 8M -e 8G -f 2 -g 1
```

实际 `mpirun` hostfile、SSH 与 scheduler 参数按集群调整。检查日志：

- 网络插件应选择 IB/RDMA 路径，不应无故退化为 `NET/Socket`；
- 所有 32 个 rank 都连接成功；
- 大消息带宽与集群基线一致；
- 没有 `ibv_modify_qp`、GID、retry exceeded 或 async error。

若出现 `ibv_modify_qp failed with Invalid argument`，优先检查 GID 选择、地址族和 HCA 端口；若长时间后超时，检查 PFC/ECN、MTU、拥塞、`NCCL_IB_TIMEOUT` 和交换机计数器，而不是直接无限增大 timeout。

## 10. Global-rank 数据采样

四节点各有完整数据副本，并不意味着每个节点都应遍历整份数据。一个 global step 内，32 个 rank 应得到 32 个不同样本。

[`ProcessedPairDataset`](../src/genet/data/dataset.py) 已实现 dataset-level stride sharding：

```python
ProcessedPairDataset(
    manifest,
    shard_by_rank=True,
    global_rank=int(os.environ['RANK']),
    world_size=int(os.environ['WORLD_SIZE']),
)
```

其索引为：

```text
rank, rank + WORLD_SIZE, rank + 2 × WORLD_SIZE, ...
```

使用该模式时不能再套 `DistributedSampler`，否则数据会被二次切分。另一种合法方式是 `shard_by_rank=False`，并由上游 Cosmos/标准 `DistributedSampler(num_replicas=WORLD_SIZE, rank=global_rank)` 切分；两种方式只能选一种。

当前轻量训练器 [`src/genet/training/standalone.py`](../src/genet/training/standalone.py) 明确使用后一种方式：`shard_by_rank=False` 加一个 global-rank `DistributedSampler`。

生产 Cosmos 路径使用 [`CosmosProcessedPairDataset`](../src/genet/integrations/cosmos_data.py) 与 [`CosmosInfiniteRankPartitionedDataLoader`](../src/genet/integrations/cosmos_loader.py)：它先复用上游 `RankPartitionedDataLoader` 写入 global shard 信息，再对 rank-local 索引做确定性 epoch shuffle，并把有限 map dataset 循环为无限流供上游 `PackingDataLoader` 使用。不要再叠加 `DistributedSampler`。

关键规则：

- sampler 使用 `RANK`，不是 `LOCAL_RANK`；每个节点都有 `LOCAL_RANK=0..7`，用 local rank 会造成四节点重复数据；
- dataset/manifest 的长度与顺序必须在所有 rank 上一致；production 每个 epoch 使用 `floor(N/WORLD_SIZE)×WORLD_SIZE` 个样本以保证各 rank 等长，并按 epoch 循环平移全局索引，使最多 `WORLD_SIZE-1` 个暂未使用的样本不会永久停留在尾部；同时要求 `N >= WORLD_SIZE`；
- `DeterministicEpochSampler` 使用 `train.seed + shard_rank + epoch` 生成每个本地 epoch 的 permutation；
- `loader.drop_last=true` 且 production micro-batch 固定为 1；
- reference 由稳定 hash 选择，不依赖 worker-local RNG。实现位于 [`src/genet/data/reference.py`](../src/genet/data/reference.py)；当前训练入口固定 reference 不随 epoch 改变（底层 Dataset API 可显式启用 epoch salt）；
- 精确 resume 时，上游传入 `optimizer_iteration × grad_accum`；resume-aware packing loader 将其 `divmod(local_epoch_length)` 还原为 epoch 与 offset，并丢弃 prewarm buffer，因此不会从本地 index 0 静默重放。

standalone 与 production 训练入口都会在第一次前向前自动比对排除节点本地路径后的语义 config
fingerprint、manifest SHA256、排序后的 embodiment map，以及 Python/Torch/CUDA/cuDNN/NCCL、GPU、
package-set、installed GenET Python-source、embedded build revision、Git revision 声明、镜像 digest 声明、
cluster-lock hash 与 path-independent receipt contract。因此各节点 processed/checkpoint 根路径可以不同，
实验参数和内容 identity 不能不同。大文件本身不会在 32 个 rank 中重复扫描；它们必须先由每节点一次的
`genet-cluster-lock verify --receipt ...` 验证，训练再把实际 runtime paths 与 receipt 绑定。

## 11. HSDP/FSDP 拓扑

配置校验要求：

```text
data_parallel_shard_degree
× data_parallel_replicate_degree
× context_parallel_shard_degree
× cfg_parallel_shard_degree
= WORLD_SIZE
```

### 推荐：节点内 FSDP shard，节点间 replicate

当前 32 卡配置使用：

```yaml
data_parallel_shard_degree: 8
data_parallel_replicate_degree: 4
context_parallel_shard_degree: 1
cfg_parallel_shard_degree: 1
```

这相当于 HSDP：8-way FSDP shard 尽量留在节点内，四个节点形成 replicate 维度。相比 32-way FSDP，它减少了每层参数 all-gather/reduce-scatter 穿过 RoCE 的频率，更适合 4 节点×8 卡拓扑。

必须检查上游 device mesh 的维度顺序，确认连续 8 个 global rank 确实落在同一 sharding group。`torchrun` 默认 global rank 0–7、8–15、16–23、24–31 分别位于四节点，仍应从启动日志中验证实际 mesh。

### 备选：32-way FSDP

理论配置为：

```yaml
data_parallel_shard_degree: 32
data_parallel_replicate_degree: 1
context_parallel_shard_degree: 1
cfg_parallel_shard_degree: 1
```

它能进一步分散参数/optimizer state，但把 FSDP 高频通信放到 RoCE，通常吞吐更差。仅在 8-way shard 显存不足、且 nccl-tests/端到端 profiling 证明可接受时使用。仓库没有为此提交独立实验 YAML。

### Context Parallel

未来若 adapter 支持 CP，可考虑 `shard=4, replicate=4, CP=2` 并确保 `shard×CP=8` 留在单节点。但当前 [`CosmosCrossEmbodimentModel`](../src/genet/models/cosmos_adapter.py) 会拒绝 CP，因此这不是现在可运行的配置。

## 12. 无共享文件系统下的 DCP

### 12.1 为什么不能直接使用同名本地目录

PyTorch DCP 通常为每个 rank 生成一个或多个 `.distcp` 文件，并由 coordinator 写全局 `.metadata`。如果 4 个节点分别写 `/local/checkpoints/iter_N`，每个节点实际上只拥有本节点 8 个 rank 的 shard；只有 coordinator 节点通常拥有 `.metadata`。

因此，“四节点都有一个同名目录”不等于“每个目录都是完整 checkpoint”。在汇总、校验并写 `COMMITTED` 之前，任何一份都不能作为可恢复 checkpoint 发布。

production builder 已显式设置 `checkpoint.broadcast_via_filesystem=false`、关闭 async DCP，并关闭上游 object-store save/load，避免错误假设节点间存在共享目录。跨节点持久化完全由下面的 node publish → consolidate → prestage 流程负责。

### 12.2 发布每个节点的 shard

上游 Cosmos 完成一次 DCP save 并经过全局 barrier 后，每个节点只运行一次 node-leader 发布命令：

```bash
bash scripts/publish_dcp_node.sh \
  /local_nvme/genet/outputs/stage3_shared_seed42/genet/cross_embodiment/run/checkpoints/iter_000001000 \
  ckpt-user@archive-host:/srv/genet/stage3_shared \
  000001000 \
  "${NODE_RANK}"
```

[`scripts/publish_dcp_node.sh`](../scripts/publish_dcp_node.sh) 会：

1. 对本节点文件生成 `NODE_MANIFEST.json`；
2. 自动创建并上传到 `iter_000001000.incomplete/node_XX/files/`；
3. 最后上传 `NODE_DONE`。

四个节点必须上传到同一个 iteration 目录，但各自只写自己的 `node_XX` 子目录。

单节点 8 卡的 S2 DCP 已在一台机器上包含全部 8 个 rank shard。要把它发布为 S3 warm-start，可复用同一流程，但该作业的逻辑 `NODE_RANK` 是 0，并在 consolidate 时使用 `--expected-nodes 1`；物理机器在四节点集群中的编号不应写成 1/2/3。

### 12.3 汇总并提交

在能看到完整 archive 的主机上执行：

```bash
python -m genet.cli.checkpoint consolidate \
  --archive-dir /srv/genet/stage3_shared/iter_000001000.incomplete \
  --output-dir /srv/genet/stage3_shared/iter_000001000 \
  --expected-nodes 4
```

汇总器会检查：

- `node_00` 到 `node_03` 的 manifest 与 DONE；
- 每个文件的 size 和 SHA256；
- 重名文件只能在内容完全一致时接受；
- consolidated 目录中必须存在 DCP `.metadata`。

汇总器使用独立的 `iter_000001000.consolidating` 事务目录，因此不会与上传用的 `.incomplete` 目录冲突。全部通过后才原子生成最终目录、`MANIFEST.json` 和 `COMMITTED`。验证：

```bash
genet-checkpoint verify \
  --checkpoint-dir /srv/genet/stage3_shared/iter_000001000
```

只有这一步成功后，外部作业管理器才可以更新 `latest` 指针。不要把 `.incomplete` 目录或仅有 `NODE_DONE` 的目录标记为 latest。

### 12.4 恢复前预分发

重新提交 32 卡作业前，将完整 committed DCP 复制到每个节点本地 NVMe：

```bash
bash scripts/prestage_dcp.sh \
  ckpt-user@archive-host:/srv/genet/stage3_shared/iter_000001000 \
  /local_nvme/genet/resume/iter_000001000
```

该脚本使用 `rsync --delete-delay` 同步并自动执行 `genet.cli.checkpoint verify`。目标必须是本次作业专用 checkpoint 目录，不能指向数据根目录或包含其他实验的宽泛目录。

四节点预分发结束后，再比较本地 `MANIFEST.json` 与 `COMMITTED`；所有 rank 从同一相对布局加载。完整 DCP 被复制四份会占用较多磁盘。当前 standalone manager 与生产 Cosmos bridge 都不会自动轮转 checkpoint。只能在 archive 已 committed、checksum 验证通过且恢复演练成功后，由外部运维脚本人工清理旧的节点本地副本；不得以未完成的 `.incomplete` 上传替代旧 checkpoint。

## 13. Resume 与 warm-start

两种语义必须严格区分。

### Resume

`checkpoint.resume` 用于同一个 stage 的故障恢复，应加载：

- model；
- optimizer；
- scheduler；
- trainer/optimizer iteration；
- Python/NumPy/PyTorch/CUDA RNG；
- dataloader 消费位置（由 optimizer iteration、grad accumulation 和确定性 sampler 重新构造）；
- 若启用，EMA 与 scaler 状态。

精确 resume 应保持：

- 4×8 拓扑、WORLD_SIZE 和 parallel degrees 不变；
- reference mode、manifest 与数据顺序不变；
- optimizer 参数组不变；
- committed 完整 DCP 已预分发到每个节点。

32 卡恢复命令示例；四个节点使用相同参数，仅 `NODE_RANK` 不同：

```bash
bash scripts/launch_roce.sh \
  configs/experiments/stage3_shared_32gpu.yaml \
  --manifest /local_nvme/genet/data/train/manifest.jsonl \
  --resume /local_nvme/genet/resume/iter_000001000 \
  --output-dir /local_nvme/genet/outputs/stage3_shared_seed42_attempt2
```

恢复作业建议写入新的、空的 attempt output root，避免与上一次作业遗留的局部 shard、日志或同 iteration 名目录冲突；通过 run metadata 把 attempt 关联到同一逻辑实验。旧目录应先保留到 archive commit 和恢复验证完成。

standalone/toy backend 的 `CheckpointManager` 使用 `model.pt`、`optimizer.pt`、`scheduler.pt`、`rng.pt`、`trainer.json` 和 `COMMITTED`；保存时所有 rank 会把 Python、NumPy、torch 和当前 CUDA RNG 汇总到 rank 0，恢复时按 global rank 取回各自状态。Cosmos backend 使用上游 DCP。不要把 standalone 目录传给 Cosmos DCP loader。

### Warm-start

`checkpoint.warm_start` 只加载主 model weights，并重新创建 optimizer、scheduler、sampler 与 trainer state。production builder 在 DCP load 时跳过旧 `net_ema.*`，随后由已加载的 regular net 初始化一份新的 EMA；exact resume 则恢复原 EMA，绝不执行这次重置。以下情况必须使用 warm-start：

- S1 → S2、S2 → S3、S3 → S4；
- 8 GPU → 32 GPU；
- HSDP/FSDP/CP 拓扑改变；
- no-ref/shared → dual 并新增 route-B 参数；
- 冻结策略或 optimizer 参数组改变。

shared → dual 时，route-B 必须由 route-A 复制初始化，不能随机初始化；这样 shared 与 dual 在分支起点尽可能函数一致。dual → shared 没有函数等价的自动合并，不应静默平均权重。

如果确实需要利用 DCP 的 load-time resharding，也必须先把完整 DCP 预分发；但 topology 改变后 optimizer、dataloader 与样本语义很难保持 bit-exact，因此仍应把 stage 转换记录为 warm-start，而不是 resume。

## 14. 训练与评估指标

训练日志至少记录：

- video flow loss、action flow loss及其加权总和；
- 每个 action embodiment/domain 的有效维度与 masked loss；
- adapter/base 两个 optimizer 参数组的 LR；
- gradient norm、clip 比例、NaN/Inf；
- source/reference condition dropout 比例；
- trainable/total parameter 数；
- samples/s、frames/s、step time、data wait time；
- 每 rank GPU memory 和跨节点通信时间；
- 当前 manifest/config/base-checkpoint fingerprint。

评估不能只看 FVD。建议固定 validation reference，并对每个 pair 再评估多个 reference 以测量敏感度：

当前 production builder 将 `dataloader_val=None`、`run_validation=false`、W&B 设为 disabled，并移除在线采样 callback，以免 32 卡训练被解码和可视化阻塞。模型的 `generate_samples_from_batch()` 已支持三路条件；下表中的生成式评估仍应由独立、固定版本的离线评估作业执行，并显式固定 sample/reference IDs、seed、solver 和 checkpoint。

| 类别 | 指标示例 |
| --- | --- |
| 任务一致性 | source-output task classifier、video retrieval、人工任务判断 |
| Target 本体一致性 | embodiment classifier、形态/reference similarity |
| 视频质量 | FVD/KVD、时序一致性、光流稳定性、感知质量 |
| 动作准确性 | 反归一化 MAE/RMSE、末端位姿误差、gripper 准确率 |
| 视频-动作同步 | cross-correlation、DTW、inverse-dynamics consistency |
| 任务效果 | simulator success、真实机器人小规模验证 |
| 稳健性 | seen/unseen task、本体组合、不同 reference 的均值与方差 |

对 shared/dual/no-ref 必须使用相同的 validation sample IDs、reference IDs 和推理 seed。

## 15. 长时推理与训练的边界

长视频不通过把训练 `T` 直接增大到数百帧实现。默认推理保持训练窗口 `T=81`，以 17-frame overlap 和 64-frame stride 滚动：上一 Target 窗口的 5 个尾部 video latent token 与 17 个 action step 同时成为下一窗的 Cosmos clean condition，视频和动作再作为一个候选事务接受、重试或回滚。完整运行方式见 [`docs/INFERENCE.md`](INFERENCE.md)。

以下能力纯属独立推理编排，不改变 optimizer、loss 或 S1–S4 checkpoint：

- 窗口切分、候选 seed、retry ladder 和 parent rollback；
- seam/action/sync 质量门与长程 drift 检测；
- active/finalized chain、journal、crash resume 和部署侧流式 exporter/mux；
- 不同 run 或候选的部署级调度。

Target clean prefix 与 Source/Reference 条件不同。固定上游 Cosmos 的 `SequencePlan` 已提供 `condition_frame_indexes_vision/action` 和 sampler clean-token preservation；但当前 GenET 训练样本仍把全部 Target token 作为 noisy supervision。第一版应先做推理侧验证，至少记录 prefix error、seam rejection、attempt/window、rollback rate，以及 2、10、100 窗口上的 identity/task/sync drift。

只有当这些指标表明明显的分布偏移时，才增加 continuation 训练。建议保持 `T=81`，仅在 10%–20% batch 上随机取 `O∈{9,17,33}`，让 video/action 共用 clean-prefix mask，并只在未知 suffix 上计算 loss。shared、dual 和 no-ref 的对比必须获得相同 mask 日程、样本预算和 seed；不能只给获胜分支追加 continuation 数据后继续把结果归因于 reference projector。

长时作业不得复用训练进程中的在线 callback。正式入口是：

```bash
genet-generate-long \
  --config configs/inference/long_video_action.yaml \
  --factory my_robot.genet_factory:create_runner \
  --output /local_nvme/genet/inference/pick_place_00042 \
  --dry-run
```

`--dry-run` 不导入或调用 factory，也不读取 resume journal；它只解析严格配置并打印窗口/action plan。之后必须用真实 factory 做至少两窗口 smoke test，再以 `--no-resume` 启动新 run；故障恢复使用相同参数和 output root 改为 `--resume`。model、Source、Reference、normalization、窗口或 seed fingerprint 改变时必须新建 output root。

多 rank 推理也不等于 32-rank 训练数据并行。同一候选若依赖 FSDP，所有参与 rank 必须执行相同 solver collective，只由 coordinator 写 journal；连续窗口存在父子依赖，不能按 global rank 独立分片。入口会比较各 rank 的 config/identity/Source 长度以及逐窗 shape/dtype，并协调 Python 异常。当前 CLI 不自动拆分 replica group；不同 run 的并行应由调度器分配独立 output root。无共享文件系统下不要让四个节点各写一份同名 `RUN.json`，resume 还必须让新 global rank 0 落到原 coordinator 节点，或预先搬运完整 output root。若 Edge 模型可在单节点推理，优先避免跨 RoCE 的逐 solver-step all-gather；确需跨节点时复用第 9 节的 NCCL/RoCE 预检并做故障注入。

正式机器人 dataschema 还需给出 action 单位/坐标系、SE(3) 与 gripper 处理、observation-action offset、长 Source rational timebase、末尾停机/padding、kinematics/URDF hash 和每本体安全阈值。在这些契约到位前，生成 action 只用于离线和 simulator 验证，不能直接下发真机。

## 16. 常见故障

| 现象 | 首要检查 | 处理 |
| --- | --- | --- |
| 启动时 parallel degree 报错 | 四个 degree 的乘积 | 必须等于 WORLD_SIZE；8 卡消融为 `8×1×1×1`，32 卡为 `8×4×1×1` |
| 四节点看到重复样本 | sampler 是否用了 `LOCAL_RANK` | 改用 global `RANK`；dataset sharding 与 DistributedSampler 二选一 |
| 某些 rank 提前结束或 hang | `drop_last`、坏样本跳过、manifest 长度 | 开启 `drop_last`；训练时 fail-fast；比较四节点 manifest SHA |
| reference 泄漏 target | episode ID、embodiment pool | reference 必须与 target 同 embodiment、不同 episode；验证工具会检查 |
| `num_frames` 校验失败 | `T=1+4N` | 使用 49、81 等合法长度；reference 同样要满足 |
| `context parallel degree` 报错 | CP 是否大于 1 | 当前 adapter 只支持 CP=1 |
| Cosmos import 失败 | 上游 checkout/install | 运行 `scripts/bootstrap_cosmos.sh` 并安装固定 commit 的 train extra |
| action reference 报 `action_gen=False` | Cosmos net 是否启用 action 路径 | 使用含 action generation 的 Cosmos 配置和正确 domain/action_dim |
| NCCL 日志出现 `NET/Socket` | HCA/GID/RDMA plugin | 检查 `NCCL_IB_HCA`、RoCEv2 GID、verbs 权限和 nccl-tests |
| `ibv_modify_qp Invalid argument` | GID index/address family | NCCL≥2.21 取消手设 GID；老版按 `show_gids` 选择 RoCEv2 |
| DCP verify 缺 `.metadata` | coordinator shard 未上传 | 重新发布 node 0；不要手工写 COMMITTED |
| DCP 缺 rank shard | 某节点 NODE_DONE/manifest 缺失 | 重新上传该 node；consolidate 必须等齐四节点 |
| resume 后 loss/数据跳变 | 实际做了 warm-start、sampler offset 未恢复 | 核对加载日志、trainer state、epoch/batches-in-epoch 和 topology |
| shared/dual step 0 不一致 | route-B 未从 route-A 复制 | 使用迁移函数并在训练前运行数值回归测试 |
| OOM | T、分辨率、activation、optimizer state | 保持 CP=1；启用 full checkpointing，先降 T/分辨率，必要时评估 32-way FSDP |
| 长视频每 64 帧跳变 | 只做 RGB 拼接、clean prefix 未进入 SequencePlan | 同时设置 5 个 vision latent 与 17 个 action clean indexes，检查 prefix MAE |
| resume 后分支/seed 改变 | 用循环计数或全局 RNG、journal 不完整 | seed 纳入 parent/window/attempt；只从 checksum 完整的 state 恢复 |
| 多节点推理互相覆盖 RUN.json | 每个节点各自写本地同名 journal | 只允许 coordinator 持有 run journal，其他 rank 通过 collective 参与候选 |

## 17. 作业提交清单

提交前：

- [ ] 四节点 GenET/Cosmos commit、容器、依赖版本一致；
- [ ] immutable image digest、HF snapshot revision 与 `GENET_STRICT_ENV=1` 已设置；
- [ ] `genet-cluster-lock verify --receipt ...` 已在四节点通过，manifest、NPZ、stats、normalization、VAE、
      实际 load DCP 与 HF cache 一致且绑定到运行路径；
- [ ] 单进程/节点 `scripts/preflight_roce.sh` 已通过；
- [ ] 数据已通过 `genet-validate-data --cosmos --require-bidirectional`，且 expected embodiment 集合严格为
      `ARX-X5, aloha-agilex, franka-panda, piper, ur5-wsg`，无训练期 skip；
- [ ] `T=1+4N`，Source/Target/Reference shape 与配置一致；
- [ ] effective global batch 已按 DP degree 计算；
- [ ] shared/dual 使用同一 warm-start、seed、reference 与样本预算；
- [ ] `ibdev2netdev`、`show_gids`、`rping`、多节点 nccl-tests 正常；
- [ ] NCCL 日志确认 RDMA 而非意外 Socket fallback；
- [ ] checkpoint archive 可达，node leader 发布与 consolidate 流程已演练；
- [ ] resume 使用完整 committed DCP；stage/topology 改变使用 warm-start；
- [ ] output、日志和 static rendezvous `MASTER_PORT` 不与其他作业冲突。
- [ ] 长时推理另起作业；factory、normalization 和 Source/Reference fingerprint 已冻结；
- [ ] 已通过真实两窗口 clean-prefix smoke test，视频/action overlap、timestamp、质量门和 journal 恢复一致。

## 18. 官方参考

- [NVIDIA Cosmos Framework：Post-Training](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/training.md)
- [NVIDIA Cosmos3 Technical Report](https://research.nvidia.com/labs/cosmos-lab/cosmos3/technical-report.pdf)
- [NVIDIA Cosmos3-Edge 模型卡](https://huggingface.co/nvidia/Cosmos3-Edge)
- [Wan2.2 TI2V-5B / VAE](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B)
- [PyTorch Distributed Checkpoint](https://docs.pytorch.org/docs/main/distributed.checkpoint.html)
- [PyTorch DistributedSampler](https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler)
- [NCCL 环境变量](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- [NCCL RoCE 网络排错](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/networking_troubleshooting.html)
