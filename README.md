# GenET：跨本体同步视频与动作生成

GenET 是一个面向 Cosmos3-Edge 的跨机器人本体生成训练工程。给定：

1. Source Embodiment 的任务视频与动作；
2. Target Embodiment 中独立随机抽取的固定长度 reference 视频与动作；

模型联合生成与 Source 任务语义、时序一致的 Target Embodiment 视频和动作。监督训练仍需要配对的 `target_gt`；reference 只提供目标本体的外观、运动与动作空间先验。当前 v1 契约强制排除 `target_gt` 所在 episode，但尚未强制排除相同 task；task-level 去泄漏会随正式 dataschema 接入。

> 状态说明：数据预处理、校验、轻量联合 RF 训练、shared/dual 消融、条件式 Cosmos sampling API、训练 checkpoint，以及 `81/17` rolling-window 长时推理编排均已接线。仓库测试会执行轻量闭环；真正的 Cosmos 权重、clean-prefix 两窗口采样、长视频质量和 4×8 GPU RoCE 作业仍需在 NVIDIA Cosmos 训练容器与目标集群验收，本仓库不声称在当前开发机上完成过这些昂贵运行。

## 设计摘要

- 视频与动作使用同一个 rectified-flow `sigma`，保证联合生成时间一致。
- Source 视频经 Wan2.2 causal VAE 得到 latent，再由 zero-init control residual 加到 Target noisy latent；Source action 经带 source/target domain embedding 的 zero-init 映射加到 noisy Target action。
- Reference 视频/action 编码为同一个 reference token 集合；固定 `(t,h,w)`/时间位置编码保留帧序、空间和 vision-action 对齐，再在 MoT 层做 gated cross-attention。
- `shared` 与 `dual` 消融只改变 reference K/V projector 是否在两条消费路由间共享；query、output、gate、reference encoder、注入层和训练数据保持一致。
- Cosmos3-Edge 的生成骨干是 Cosmos MoT；Wan2.2 在该路径中提供 causal VAE/tokenizer，并不是把整个 Wan DiT 当作基座。
- 生产并行默认使用 8-way FSDP shard × 4-way replicate 的 HSDP 拓扑，使主要 shard 通信留在节点内。
- 长视频按 `T=81`、overlap `O=17`、stride `S=64` 滚动；视频和动作以一个事务接受、重试和回滚，下一窗通过 Cosmos clean condition index 同时保留 5 个视频 latent token 与 17 个 action step。

详细设计见 [架构文档](docs/ARCHITECTURE.md)，数据、训练集群和长时生成分别见 [预处理指南](docs/PREPROCESSING.md)、[训练指南](docs/TRAINING.md) 与 [推理指南](docs/INFERENCE.md)。

## 快速开始

安装轻量训练、预处理和测试依赖：

```bash
python -m pip install -e '.[preprocess,dev]'
```

查看 raw JSONL schema：

```bash
genet-preprocess --print-raw-schema
```

将三路 vision-action 数据同步裁成 `T=81`：

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

运行全仓测试和轻量 dry-run：

```bash
pytest -q

python -m genet.cli.train \
  --config configs/base.yaml \
  --manifest data/processed/train/manifest.jsonl \
  --dry-run
```

`configs/base.yaml` 使用可在 CPU/单卡验证契约的 `toy` backend。它不是 Cosmos 的替代品，而是提交昂贵作业前用于确认数据、梯度、mask、联合 flow 和 checkpoint 的可执行基线。

## Cosmos3-Edge 生产环境

项目固定 Cosmos Framework commit：

```text
a904d2d36b774a51dd06ff9ff906816b1a04f579
```

在 NVIDIA Cosmos 训练容器中执行：

```bash
bash scripts/bootstrap_cosmos.sh
python -m pip install -e 'third_party/cosmos-framework[train]'
python -m pip install -e .
```

然后准备转换后的 Cosmos3-Edge DCP 与 `Wan2.2_VAE.pth`，按 [训练指南](docs/TRAINING.md) 完成 S0–S4、单节点消融、4 节点 RoCE 启动，以及无共享文件系统下的 checkpoint 汇总与预分发。

## 长视频与动作推理

长时入口通过 factory 注入真实模型、Source/Reference 读取和机器人 action 语义。先做不加载 factory 的配置检查：

```bash
genet-generate-long \
  --config configs/inference/long_video_action.yaml \
  --factory my_robot.genet_factory:create_runner \
  --output /local_nvme/genet/inference/pick_place_00042 \
  --dry-run
```

然后用同一 output root 执行新 run，或从 checksum 完整的已接受窗口边界恢复：

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

`--dry-run` 不导入或调用 factory，因此通过 dry-run 不等于 Cosmos 权重、clean-prefix 接线或显存已经验收。正式发布前至少完成两窗口真实 smoke test，并验证 video/action prefix、timestamp 与 journal 恢复。完整算法、质量门、rollback 和多 rank 限制见 [推理指南](docs/INFERENCE.md)。

## 目录

```text
configs/                    基础、消融与 RoCE 配置
docs/                       架构、预处理、训练和长时推理指南
scripts/                    上游固定、torchrun、DCP 搬运脚本
src/genet/data/             schema、同步采样、预处理、Dataset
src/genet/models/           Control、reference attention、toy 与 Cosmos adapter
src/genet/inference/        81/17 rolling、质量门、rollback、journal/resume
src/genet/integrations/     Cosmos 数据、loader 与长时 clean-prefix bridge
src/genet/training/         distributed、stage、RF、checkpoint、训练 runtime
src/genet/cli/              preprocess、validate、train、checkpoint、长时生成 CLI
tests/                      数据、模型、训练、checkpoint 与长时生成测试
```

## 当前数据契约与后续接线

`genet.processed-pair/v1` 目前要求 Source、Target GT、Reference 三路都为相同固定长度，视频为 `[T,H,W,3]`，动作与维度/时间 mask 为 `[T,64]`。本版已提供通用 linear/nearest action 重采样，但正式 dataschema 接入时仍应补齐：

- 每个本体的 action 字段语义、单位、归一化与有效维；
- SO(3)/SE(3) 专用插值与 frame convention；
- observation/action 的精确时间偏移；
- train split-only normalization stats；
- 按任务排除 reference、数据集版本和 lineage；
- 长 Source 的 rational timebase、末尾停机/padding 语义；
- 每本体 joint/SE(3)/gripper 质量阈值、kinematics 与安全门。

这些项目被明确保留为 schema 层扩展点，不应通过猜测机器人动作含义来静默处理。

## 上游与许可证

- [NVIDIA Cosmos](https://github.com/NVIDIA/cosmos)
- [Cosmos Framework 训练文档](https://github.com/NVIDIA/cosmos-framework/blob/main/docs/training.md)
- [Cosmos3-Edge 模型卡](https://huggingface.co/nvidia/Cosmos3-Edge)
- [Wan2.2](https://github.com/Wan-Video/Wan2.2)
- [VACE](https://github.com/ali-vilab/VACE)（zero-init control 设计参考）

GenET 自有代码采用 Apache-2.0。Cosmos 源码和权重受 OpenMDW-1.1 及模型卡约束；其他依赖和设计参考见 [THIRD_PARTY.md](THIRD_PARTY.md)。
