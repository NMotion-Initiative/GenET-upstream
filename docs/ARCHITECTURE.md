# GenET 架构与消融设计

## 1. 问题定义

每个训练样本包含三个同步的 vision-action stream：

- `source = (V_s, A_s, e_s)`：Source Embodiment 执行任务的视频、动作和 domain；
- `target_gt = (V_t, A_t, e_t)`：同一任务在 Target Embodiment 上的监督真值；
- `reference = (V_r, A_r, e_t)`：从 Target Embodiment 的其他 episode 稳定随机采样的片段。

目标分布为：

```text
p(V_t, A_t | V_s, A_s, V_r, A_r, e_s, e_t)
```

Reference 必须与 Target GT 属于同一 target embodiment，但必须排除 Target GT episode。当前 v1 允许它来自相同或不同 task 的其他 episode，因此模型不能把 reference 当成目标轨迹复制；它只提供本体外观、运动学风格和 action boundary 信息。正式 schema 可再启用 task-level 排除。

## 2. 同步联合 rectified flow

Target 视频经冻结的 causal VAE 编码为 `z_video`，Target action padding 到 64 维得到 `z_action`。对两种模态采同一个 `sigma`：

```text
x_video(sigma)  = (1-sigma) z_video  + sigma epsilon_video
x_action(sigma) = (1-sigma) z_action + sigma epsilon_action

v*_video  = epsilon_video  - z_video
v*_action = epsilon_action - z_action
```

模型同时预测 `v_video` 和 `v_action`。轻量 standalone backend 的动作损失只在 `action_mask[T,D]` 为真的元素上计算，pad 视频损失也会按 causal VAE 的 `1 + 4N` latent 时间块聚合 `frame_mask`。当前 production Cosmos bridge 尚未把 temporal mask 接入上游 loss，因此 production 数据必须使用 `short_policy=drop`，任何时间 padding 都会 fail-fast。共享噪声时间并不意味着两种模态尺度相同，二者仍通过独立 loss weight 调节。

## 3. Source Control

### 3.1 视频分支

Source 与 Target 使用同一冻结 Wan2.2-TI2V-5B causal VAE，预处理保证它们的 `T/H/W` 对齐。Source latent 经过 GroupNorm、SiLU 和 `1×1×1 Conv3d`：

```text
x'_video = x_video + alpha_v * ZeroConv(source_latent)
```

末层权重和 bias 初始化为零，因此新分支刚挂到预训练生成器时是严格 no-op。此处借鉴 ControlNet/VACE 的安全初始化不变量，但没有复制其源码。

### 3.2 动作分支

不同机器人原始 action 维度不同，预处理统一 pad 到 64 维并保留 mask。动作 control 同时接收：

- Source action；
- Source domain embedding；
- Target domain embedding。

MLP 的输出层 zero-init：

```text
x'_action = x_action + alpha_a * MLP(A_s, Emb(e_s), Emb(e_t))
```

该残差会按 Target 的 `raw_action_dim`/`action_mask` 截断，不能把 Source 的有效维掩码误当成 Target 输出掩码。Cosmos 原生的 action boundary 仍负责 Target 域的 `action2llm/llm2action`；GenET 的 control 只负责把 Source 轨迹变成 Target noisy action 空间中的条件残差。

## 4. Reference Cross-Attention

Reference 视频经同一个冻结 VAE，Reference action 经同一个 domain-aware action input boundary。VAE patch token 按 `(t,h,w)` 加固定三维 sinusoidal encoding；action token 在相同的片段相对时间轴上加 `(t,0,0)` encoding，并保留 Cosmos 原生 action modality embedding。这样 cross-attention 能区分帧序与空间位置，而不是把 reference 当成可任意置换的 token bag。位置编码无训练参数，且在 shared/dual K/V projector 之前完全共用。两者形成一个 token 集合：

```text
R = concat(ProjectVideo(V_r) + PE_3D, ProjectAction(A_r, e_t) + PE_time)
```

在选择的 MoT 层中，两个固定消费路由对 `R` 做 gated cross-attention：

```text
H_route <- H_route + tanh(g_route) * Out_route(Attention(Q_route, K_route(R), V_route(R)))
```

gate 默认初始化为 0，确保 reference 分支也是初始 no-op。

支持两种路由解释：

- `vision_action`：生成 token 的视觉部分走 route A，动作部分走 route B；
- `ar_dm`：Cosmos two-way attention 中 understanding/AR stream 走 route A，diffusion-generation stream 走 route B。

当前生产配置使用 `ar_dm`。它更直接检验“同一 reference 表征是否应为 MoT 的不同功能部分提供独立 K/V 空间”。

## 5. Shared / Dual 主消融

主变量只位于 `RoutedReferenceProjector`：

| 模式 | Route A K/V | Route B K/V | 用途 |
| --- | --- | --- | --- |
| `shared` | projector A | projector A | 单一 reference feature projection |
| `dual` | projector A | projector B | 两个独立 projected feature |
| `dual_tied` | projector A | projector A | 配置/数值回归对照 |

以下项目必须相同：

- reference encoder 与 token 数；
- reference 三维/时间位置编码；
- query/output projection；
- gate 初始化；
- 注入层和 attention heads；
- Source Control、数据顺序、reference 选择、RF sigma；
- 有效 global batch、优化器步数、scheduler 和 seed。

Shared checkpoint 迁移到 dual 时，必须先把加载后的 route A 复制到 route B。不能让 route B 保留随机构造值，否则 step 0 已不是函数等价起点。Dual 到 shared 不做隐式平均，因为不存在唯一的函数保持映射。

## 6. Cosmos3-Edge 接线

生产 backend 不复制或 fork 上游实现，而是在固定的 Cosmos Framework API 边界做扩展：

1. 从官方 `vision_sft_edge` recipe 构造配置；
2. 重新启用 Edge 原生 action generation；
3. 用 processed pair Dataset 替换官方 SFT Dataset，并复用 Edge 官方 processor 把 caption 生成 `text_token_ids`；
4. 在 Cosmos MoT 进入 FSDP/compile 前注册 Source/Reference 模块；
5. 让上游继续负责 Wan VAE 编码、RF noising/loss、MoT、EMA、FSDP/HSDP 和 DCP；
6. 训练作业默认关闭昂贵的在线可视化 callback，把固定 reference 的生成评估留给独立作业。

Production loader 每个 epoch 取 `floor(N/WORLD_SIZE)×WORLD_SIZE` 个样本，并循环平移全局索引，使当轮最多 `WORLD_SIZE-1` 个未使用样本随 epoch 轮换；随后对每个不重叠的 rank shard 做确定性 shuffle 并无限循环。上游 exact-resume 给出的已消费 micro-batch 数会还原成 sampler 的 epoch/offset；这同时避免 finite map dataset 在一个 epoch 后耗尽。

Wan2.2 在 Cosmos3-Edge generator 中提供 16× spatial、4× temporal 的 causal VAE。实际 denoising backbone 是 Cosmos3 MoT，而不是 Wan 官方 DiT。因此“使用 Wan Control 训练方式”在本工程中对应的是：沿用 causal latent 对齐和 zero-init control 原则，同时把残差接入 Cosmos 的联合 vision-action flow，而不是直接调用不存在的 Wan2.2 官方 ControlNet trainer。

### 6.1 条件采样契约

`CosmosCrossEmbodimentModel.generate_samples_from_batch()` 已覆盖上游 sampling API：在 solver 循环前编码一次 Source/Reference 条件，并在每次 denoise/文本 CFG forward 中复用。普通单窗口 processed sample 的 Target `video`/`action` 只提供输出形状、Target domain 与 `raw_action_dim`；`SequencePlan` 的 target condition index 为空，因此 Target GT 数值不会成为生成条件。返回值中的 `vision` 是 Wan latent，可交给上游 `model.decode()`；`action` 已通过 `ActionProcessingRecord` 去掉 64 维 padding。`use_source_condition`、`use_reference_condition` 可用于离线诊断。长时推理的 Target prefix 例外由下一节单独定义。

### 6.2 长时 rolling 与 clean prefix

模型仍以训练时的固定 `T=81` 为唯一采样窗口。长时 orchestrator 使用 `O=17` overlap 和 `S=64` stride：第一窗贡献 81 个 step，之后每窗把上一 Target 尾部作为 clean prefix，并只贡献 64 个新 step。Wan temporal compression 为 4，因此 17 个像素帧对应 5 个 latent time token，64 帧位移对应 16 个 latent token；窗口起点和 causal latent 网格保持对齐。

第二窗起的 `SequencePlan` 使用：

```text
condition_frame_indexes_vision = [0, 1, 2, 3, 4]
condition_frame_indexes_action = [0, 1, ..., 16]
```

上游 Cosmos 的 `condition_mask`/`condition_reference` 在 sampler 中保留这些 clean token，suffix 仍为联合 noisy target。它与 Source Control、Reference cross-attention 是三类不同条件：Source 表达任务轨迹，Reference 表达 Target 本体先验，Target clean prefix 表达已接受的生成历史。只在最终 RGB 上复制 overlap 不具备这个语义，不能替代 clean-prefix conditioning。

每个窗口的 decoded canonical frames 与 action 构成一个不可拆分的候选事务；正式 dataschema/exporter 还必须绑定 timestamp。候选通过视觉 seam、动作动力学和 vision-action sync 质量门后才进入 active chain；最近若干 contribution 暂不 finalized，从而允许在后继窗口全部失败时回到父节点换 seed 重试。rollback 不得越过 finalized boundary。

`RunIdentity` 固定 model/Source/Reference/normalization，窗口 seed 再由 base seed、全局起点、parent chunk、attempt 和 rollback count 稳定派生。每个 accepted video-action suffix 先写临时 NPZ、fsync/rename，再用带 checksum 的原子 `RUN.json` 推进 active chain；中断恢复从最近完整窗口边界继续，而不是从 MP4 反解上下文。当前 v1 的 durability/bit-exact 边界和后续加固项见 [`docs/INFERENCE.md`](INFERENCE.md)。

## 7. 训练阶段

- S0：toy backend 数据契约、一步训练、单批过拟合；
- S1 `control`：冻结 base、VAE 和 reference，训练 Source Control；
- S2 `reference`：保留 Source Control 可训练，加入 no-ref/shared/dual；
- S3 `joint`：训练 adapter，并选择性打开 Cosmos generator 与 action boundary；
- S4：获胜方案的长时序/高分辨率低 LR 微调。

Stage2 的 8 卡配置用 `grad_accum=4`；Stage1/3 的 32 卡配置用 `grad_accum=1`，三者在 micro batch 为 1 时保持 global batch 32。

## 8. HSDP 与无共享文件系统

4×8 GPU 默认 mesh：

```text
dp_replicate = 4
dp_shard     = 8
cp           = 1
cfg          = 1
```

这样每个 8-way shard group 可以落在单个节点内，参数 all-gather/reduce-scatter 主要走 NVLink/NVSwitch；跨节点只承担 replicate 维同步。所有 rank 仍按 global rank 对同一 manifest 做确定性 partition，不能使用 local rank 采样。

DCP 的一个完整 checkpoint 包含所有 32 rank 的 shard 和 coordinator `.metadata`。每个节点本地目录只有该节点产生的部分文件，不能单独恢复。项目脚本执行：node manifest/hash → archive 汇总 → 4 节点完整性检查 → 原子 `COMMITTED` → 完整 checkpoint 预分发回每个节点。

## 9. 明确限制

- processed-pair/v1 当前三路同 T；不同 reference 时长是后续格式版本，不应绕过 validator。
- Context Parallel 当前固定为 1；adapter 对 CP>1 明确报错。
- Cosmos packing 当前限制每 rank 一个样本。
- production builder 默认关闭训练中的在线 sampling callback；长时生成由独立 `genet-generate-long` 作业运行，不能阻塞训练 collective。
- `genet-generate-long --dry-run` 不加载部署 factory；它只验证 orchestration 配置，不能替代真实 Cosmos 两窗口 smoke test。
- clean-prefix 条件虽然复用 Cosmos 原生 condition index，但现有 GenET 微调仍以全 Target noisy 为主；若长时 seam/rollback 指标不合格，再公平地加入少量 prefix-mask continuation augmentation。
- 通用 action linear interpolation 不懂 SO(3)/SE(3)；正式 schema 必须提供本体语义和专用处理器。
- 32 卡性能、收敛和 RoCE 参数必须在目标集群实测；CPU CI 只验证工程契约，不代表模型质量。
