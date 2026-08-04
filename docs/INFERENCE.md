# GenET 长视频与动作联合推理

本文定义基于固定 `T=81` 模型窗口生成长时 Target vision-action pair 的生产契约。长推理由滚动窗口、Cosmos clean-prefix conditioning、候选重试、有限回滚和事务式 journal 组成；它不把多个独立短视频简单拼接，也不允许视频和动作分别决定是否接受。

本文中的相对路径都以仓库根目录为起点。当前正式入口是 `genet-generate-long`；模型加载和单窗口采样由部署方通过 `--factory module:function` 注入，长时状态机不猜测 checkpoint、机器人 dataschema 或 action 物理语义。

## 1. 实现边界

长时推理层负责：

- 把长 Source vision-action pair 按同一全局时间网格切成 81-step 窗口；
- 把上一 Target 窗口的尾部作为下一窗口的视频和动作 clean prefix；
- 以 vision-action pair 为原子生成、评分、接受、回滚和恢复；
- 从 master seed 派生与执行顺序无关的窗口/分支 seed；
- 将候选、active chain、finalized boundary、质量指标和 retry counter 事务式写盘；
- 在最后一个窗口去掉 padding，并幂等组装最终视频和动作。

部署 factory 负责：

- 加载 GenET/Cosmos 权重、Wan VAE、normalization 和 embodiment registry；
- 构造与训练时相同的 Source/Reference 条件；
- 把 Target clean-prefix mask 接入 Cosmos `SequencePlan`；
- 联合采样视频 latent 与 action，解码 canonical 视频帧；
- 提供本体相关的 action 反归一化、物理约束和可选质量评估器。

`--dry-run` 只解析严格 YAML、计算窗口/action plan，并打印将使用的 factory、output 和 resume 选择；它不导入或调用 factory，也不创建/检查 output journal。因此它不能证明输入可读、checkpoint 可加载、Cosmos clean prefix 已接线、GPU 显存足够或模型输出质量合格。正式作业必须另外完成一个真实的两窗口 smoke test。

## 2. 默认滚动窗口

训练窗口和生产默认值为：

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `T` | 81 | 每次联合采样的视频帧数和 action step 数 |
| `fps` | 16 | canonical 视频/action 时间网格 |
| `q` | 4 | Wan causal VAE temporal compression factor |
| `O` | 17 | 相邻窗口 overlap，约 1 秒 |
| `S=T-O` | 64 | 每个后续窗口新增的 step 数，约 4 秒 |
| `rollback_depth` | 1 | 保留多少个最新 contribution 可替换 |
| `max_attempts_per_chunk` | 3 | 一个父上下文下的最大候选数 |

`fps=16` 来自训练/数据时间契约，不是 `LongHorizonConfig` 中可随意覆盖的字段。factory 必须从正式 Source schema 校验并固定 canonical timebase；orchestrator 只按整数 frame/action index 推进。

它们满足：

```text
T_latent = (81 - 1) / 4 + 1 = 21
O_latent = (17 - 1) / 4 + 1 = 5
latent stride = 64 / 4 = 16
```

窗口起点是 `0, 64, 128, ...`。窗口 0 贡献全部 `[0,81)`；后续起点为 `s` 的窗口以 `[s,s+17)` 为 clean prefix，只向全局结果贡献 `[s+17,s+81)` 的 64 个新 step。以三个窗口为例：

```text
window 0: [  0,  81)  contribution [  0,  81)
window 1: [ 64, 145)  contribution [ 81, 145)
window 2: [128, 209)  contribution [145, 209)
```

因此输出没有重复或缺口。末窗口可以在 Source 输入侧按部署策略 padding，但 `valid_length` 必须同时作用于视频、action、质量指标和最终组装；padding 不能被发布。

不要在一次 run 的 retry 中临时把 `O=17` 改成 33。改变 overlap 会改变窗口起点、Source 对齐、seed 和 journal 语义，必须创建一个新的 run。若修改默认值，至少保证 `T=1+4N`、`O=1+4M` 且 `S` 能整除 4。

## 3. 视频与动作的联合事务

同一窗口始终使用同一个全局区间：

```text
source_video/action = source[start:start+81]
target_clean_prefix = accepted_target[start:start+17]
new_target          = candidate[17:81]
reference           = 同一 run 固定的 Target embodiment reference
```

以下操作必须对 pair 原子执行：

- 视频帧与 action step 使用完全相同的 timestamp、窗口起点和有效长度；
- video/action clean-prefix mask 从同一个 overlap 定义生成；
- solver 中的两种模态共享训练时约定的 RF 时间，但使用独立、可复现的 noise 子流；
- 一个候选的 accept/reject 同时决定视频和动作；
- rollback 同时撤销该分支 journal 中的 canonical frames、actions 和指标；
- 部署 exporter 最终以同一个 manifest 引用视频、动作、timestamp、normalization 与 embodiment metadata。

不要在长时层盲目 cross-fade action。关节角、末端 SE(3)、速度、力矩和 gripper 的插值规则不同；视频单独平滑而 action 不变也可能破坏视觉-动作同步。默认方案是对两种模态 hard prefix，并对第 16→17 step 的真实新旧边界做质量门控。若正式 dataschema 允许 blend，必须由 embodiment adapter 同时处理视频和动作，并记录所用语义。

配置同时支持 `action_alignment: frame` 和 `transition`。默认 `frame` 表示 81 帧对应 81 个 action step，overlap action 为 17；`transition` 表示 81 帧对应 80 个转移动作，overlap action 为 16。transition 模式强制 `overlap_frames>=1`，否则每个窗口边界会漏掉一个 transition；当 overlap 恰好为 1 帧时，action context 合法地是 `[0,D]` 空 tensor，而不是整段 action。factory 的 `WindowSource`、Target template 和最终 exporter 必须使用同一模式，resume 时也不能改变。

## 4. Cosmos clean-prefix conditioning

当前单窗口条件采样将 Target `condition_frame_indexes_vision/action` 留空，表示全部 Target token 都是生成目标。滚动窗口从第二窗起必须构造混合 `SequencePlan`：

```text
condition_frame_indexes_vision = [0, 1, 2, 3, 4]
condition_frame_indexes_action = [0, 1, ..., 16]
```

vision index 是 VAE latent 时间索引，因此 17 个像素帧对应 5 个 causal latent token；action 不做 temporal compression，所以保留 17 个 step。除 prefix 外的 Target token 仍是 noisy generation target。Source Control 和固定 Reference 条件保持不变。

应复用固定上游 Cosmos 的 `SequencePlan`、`condition_mask`、`condition_reference` 和 sampler clean-token preservation 路径。不要只在最终 RGB 上贴回 17 帧：那样 denoiser 没有看到 Target 历史，第一帧新内容仍可能产生动作、姿态和光流跳变。

当前 [`CosmosLongHorizonSampler`](../src/genet/integrations/cosmos_long_horizon.py) 会把 journal 恢复出的 17 个 canonical context frame 放进 Target batch，再由 Wan VAE 编成 5 个 clean latent token；它同时把 17 个 action step 写入 Target action prefix。实现必须满足：

- 上下文只能来自 journal 中带 checksum 的 canonical chunk，不从有损 MP4 或预览重建；
- video/action condition mask 的全局时间范围完全相同；
- 每个 solver step 都保持 clean condition，最终输出再逐元素验证 prefix；
- candidate 的 prefix 与 journal context 按阈值校验，commit 时只保存去除 overlap 后的新 suffix；
- 用同一个 `raw_action_dim` 和 Target domain 处理 prefix 与 suffix；
- 记录 pre-clamp/preservation error，发现 prefix 不相等时 hard fail。

Cosmos clean conditioning 是首选路径。若部署 factory 使用自定义 solver，则必须实现等价的 scheduler-aware context preservation；不能把 clean `x0` 无条件写进所有噪声层级。具体 RF/noise 公式应由固定上游 scheduler 提供，不能在编排层复制一份可能随版本漂移的公式。

## 5. Reference 与长 Source

Reference 是 Target embodiment 中独立任务片段，不随 rolling window 改变。整个 run 固定 reference ID、原始数据 hash、预处理版本和 encoder fingerprint。retry 或 rollback 不得重新随机抽 Reference，否则候选差异不再只来自采样分支。当前 `CosmosLongHorizonSampler` 每个窗口仍通过公开采样 API 构造完整条件；若 factory 缓存 Reference encoding，必须证明缓存与模型/EMA/device/dtype 绑定且数值等价，不能绕过 adapter 的 condition-state 生命周期。

长 Source 必须先转换为一条 canonical vision-action 时间线。每个窗口在相同 `[start,start+81)` 上读取 Source 视频和动作。若视频帧率、action 频率或 observation/action offset 不同，必须先由正式 dataschema adapter 对齐，不能让 rolling loop 分别四舍五入两个流的时间戳。

## 6. 候选、质量门和选择

每个窗口在同一父上下文下最多生成 `recovery.max_attempts_per_chunk` 个候选。当前 orchestrator 按 attempt 顺序运行，并接受第一个 `QualityReport.accepted=true` 的候选；它不会先生成 K 个候选再按主观分数挑选。该规则计算成本低、恢复语义明确，而且候选 seed 不依赖调度顺序。

内置 `DefaultContinuityEvaluator` 当前执行的通用门限是：

- 输出视频 `[C,T,H,W]` 与 action `[T,D]`/`[T-1,D]` shape 合法；
- 配置要求时，所有帧和动作有限，无 NaN/Inf；
- video prefix normalized MAE 不超过 `max_video_prefix_mae`；
- action prefix relative MAE 不超过 `max_action_prefix_relative_mae`；
- 有可比较上下文时记录 video/action boundary jump，并在配置非 `null` 时启用对应门限；transition + 1-frame overlap 没有 clean action prefix，正式 evaluator 需从全局 action history 另取前一 transition 做边界检查。

默认配置特意把本体相关的 boundary hard gate 留为 `null`，因为通用代码不知道机器人单位和动力学。生产 factory 应通过 `LongGenerationJob.evaluator` 注入额外门限，至少检查：

- timestamp、有效长度、action mask 和 padded 维；
- joint/EE/gripper 范围、速度、加速度与 jerk；
- 无损坏解码、黑屏/纯色退化或明显重复帧；
- Source/Target 全局时间索引与任务相位；
- vision-action 同步和 simulator safety（若可用）。

软指标至少覆盖：

| 类别 | 推荐指标 |
| --- | --- |
| 视频边界 | seam LPIPS/DINO、VAE latent 距离、光流 warping error、光流加速度、曝光/颜色跳变 |
| 视频长程 | flicker、Target identity/embodiment drift、Source task/phase 相似度 |
| 动作边界 | 反归一化 `Δa`、速度、加速度、jerk、FK 末端位姿连续性、gripper 抖动 |
| 视觉-动作同步 | inverse-dynamics consistency、motion/action cross-correlation、DTW lag |
| 任务效果 | simulator success、任务 classifier/retrieval、人工或真机小样本评估 |

因为 prefix 是 clean condition，overlap 内部误差主要用于检查接线；真正 seam 是 candidate index `16→17`，应在其两侧 3–5 step 上评估。指标阈值按 Target embodiment 和 action 类型用 validation 分位数标定，不应给所有机器人共用一套未经归一化的绝对阈值。

高级 evaluator 可以维护 identity、任务相位、sync lag 和 action energy 的 EMA/CUSUM；单窗过门但长期漂移持续恶化时也可拒绝候选。当前 v1 journal 只保存每次返回的标量 `metrics`/`failures`，不会自动序列化 evaluator 的隐藏状态；需要精确恢复长程指标时，应把状态显式编码进可恢复 metadata，而不是只留在 Python 对象中。

## 7. Retry 与 rollback

内置 `CosmosLongHorizonSampler` 的 retry 保持 `T/O/S` 和 sampling config 不变，只更换确定性 seed。部署 factory 也可以依据 `ChunkRequest.attempt_id` 实现预先声明的 ladder，例如增加 solver steps、微调 Source/context scale，或降低 CFG/Reference scale；这些变化必须确定性生成并写入 `ChunkOutput.metadata`。不要在 retry 中改变窗口算术或 Reference。

预算由 `max_attempts_per_chunk`、`rollback_chunks` 和 `max_total_rollbacks` 限制。当前 v1 没有单独的 `max_total_candidates`；需要严格成本上限时由作业调度器根据 journal attempts 数量终止。

状态分为：

- `durable`：候选的新 suffix NPZ 已完整写盘，路径与 checksum 已进入 `RUN.json`；
- `accepted`：候选位于当前 active chain；
- `finalized`：候选 contribution 已越过 rollback horizon，之后不可改写。

默认 `rollback_depth=1` 时，最近一个 accepted contribution 暂不 finalized。只有其后继窗口被接受，才 finalize 它。若下一窗口的所有 retry 都失败：

1. 按 `rollback_chunks` 回到最近的未 finalized contribution；
2. 原子更新 journal，把旧 record 从 active chain 转入 `superseded_chunks`；immutable NPZ 保持原路径；
3. 为父位置使用新的 attempt/branch seed 生成替代候选；
4. 从新父候选重新向前扩展。

rollback 绝不能越过 `finalized_frames`。耗尽 `max_total_rollbacks` 后会 fail-closed，保留最后一个可恢复状态并抛出错误；不能发布只有视频、没有 action，或长度未经确认的结果。`rollback_depth=1` 会保留最近一个 contribution 可替换；深度 2 可提高纠错能力，但增加延迟、推理成本和本地磁盘占用，且 `rollback_chunks` 不得大于该深度。

## 8. 确定性 seed

orchestrator 不使用进程全局 RNG。当前 v1 用 BLAKE2b 从以下字段派生一个 31-bit `ChunkRequest.seed`：

```text
seed = H(
  genet.long-generation/v1,
  base_seed,
  global_window_start,
  RunIdentity_digest,
  parent_chunk_sha256,
  attempt_id,
  total_rollbacks
)
```

attempt 在调用 sampler 前写入 `RUN.json`。因此执行顺序和 rank 不改变已经分配的 seed；identity、parent chunk 或 rollback 分支改变时也不会复用旧子分支 seed。若进程在 sampler 中途退出，resume 会保留该次 `started` 记录、把它计入该 parent/rollback generation 的预算，并在尚有预算时分配下一个 attempt，而不是重置计数或假装旧候选已经完成。

factory 只收到一个窗口 seed。它应从该值稳定派生 video noise、action noise 和任何 stochastic evaluator 子流，不能再使用未记录的全局 RNG。model/Source/Reference/normalization 标识通过 `RunIdentity_digest` 进入 seed，resume 也会逐项拒绝 identity 不一致。当前 journal 不保存初始 noise 或 CUDA RNG；恢复 SLA 是“已接受 prefix 和下一窗口位置精确”，不是跨驱动重新生成同一 in-flight suffix 的 bit-exact 保证。若部署需要后者，应由 factory 把 noise/solver fingerprint 写入外部 artifact，并限制非确定性 kernel。

## 9. Journal 与精确恢复

当前 v1 目录：

```text
<output>/
  RUN.json
  RUN.lock
  chunks/chunk-<sequence>-<start>-<seed>.npz
```

只有 coordinator/rank 0 持有 `RUN.lock` 并写 journal。`RUN.json` 记录：

- format version、完整 long-horizon config 与其 fingerprint；
- `RunIdentity` 中的 Source、model、Reference、normalization 和 code ID；
- 总视频帧数、status/failure、`committed_frames` 与 `finalized_frames`；
- attempt counter、每次 seed/status/metrics/failures；
- active chunks、superseded chunks、每块 video/action shape、时间范围和 SHA-256；
- 首个候选确定的 video geometry/action width `output_contract`；
- 累计 rollback 数。

一次 accepted commit 的顺序为：

1. 从 candidate 同时切出 video/action 的新 suffix；
2. 将 video 按 `store_video_dtype`、action 按 float32 写入临时 NPZ；
3. flush/fsync 后原子 rename 到 `chunks/`，并 fsync 父目录；
4. 计算 SHA-256，把路径、范围、seed、metrics 和 metadata 加入 active chain；
5. 通过临时 JSON + fsync + `os.replace` 原子更新 `RUN.json`。

`--resume` 先验证 config fingerprint、完整 `RunIdentity` 和总长度，再逐个验证 active 路径仍位于 `chunks/`、NPZ checksum、shape/output contract、accepted attempt 对应、连续 video/action 时间范围，以及 committed/finalized/status 关系。上下文从 active chunks 的尾部重建，下一窗口从最近完整边界继续。

rollback 不搬动或覆盖 NPZ。它在内存 state 中同时把末尾 video-action records 从 `active_chunks` 转入 `superseded_chunks`，校验 finalized boundary 后只做一次原子 `RUN.json` 切换；chunk 文件继续以原路径保持 immutable，便于 crash-safe 回滚和审计。可选 GC 只能在 run 完成并按 active/superseded 引用核对后运行。

如果 crash 发生在 sampler 内，已接受 prefix 不变，未完成 attempt 在 journal 中保持 `started`，恢复时以剩余预算和新 attempt/seed 重试。如果 crash 恰好发生在 chunk rename 与 `RUN.json` 更新之间，可能留下一个未引用 NPZ；它不属于 active chain，不会被自动发布，可在 run 完成后按 journal 引用集合清理。预算彻底耗尽且不能 rollback 时，journal 会持久标记 `failed`；该 output root 不能继续 resume。当前 v1 不保留历史 state generations，因此 `RUN.json` 本身损坏时不会自动回退旧版本；生产归档应备份该文件。

NPZ chunk 是当前恢复真值，默认视频存为 float16、action 存为 float32；它不包含原始 Wan latent、initial noise 或独立 timestamp 文件。CLI 完成后保留 chunks 和 `RUN.json`，目前不自动编码最终 MP4。长序列 exporter 应使用 `GenerationJournal.iter_chunks()` 流式编码视频和增量写 action；`materialize()` 会把全部 chunk 载入内存，只适合测试或短序列。部署 exporter 应从 active chain 幂等生成视频、action/timestamp 和总 manifest，不要增量 append 有损 MP4。

当前实现已经对 NPZ/JSON 文件及其 rename 所在目录执行 fsync。若部署要求跨存储设备验证的掉电级 durability、跨软件栈 bit-exact replay 或法规审计，还应扩展为版本化 state fallback、latent/noise/timestamp artifact、远端持久化确认和最终发布 manifest；这些不是当前 v1 已实现能力。训练 DCP 的 `COMMITTED` 也不能拿来替代推理 `RUN.json` 或反向使用。

## 10. CLI 与 factory

正式入口支持模块和 console-script 两种写法：

```bash
genet-generate-long \
  --config configs/inference/long_video_action.yaml \
  --factory my_robot.genet_factory:create_runner \
  --output /local_nvme/genet/inference/pick_place_00042 \
  --no-resume
```

等价地：

```bash
python -m genet.cli.generate_long \
  --config configs/inference/long_video_action.yaml \
  --factory my_robot.genet_factory:create_runner \
  --output /local_nvme/genet/inference/pick_place_00042 \
  --no-resume
```

首次提交先执行：

```bash
genet-generate-long \
  --config configs/inference/long_video_action.yaml \
  --factory does.not.need.to.exist:create_runner \
  --output /local_nvme/genet/inference/pick_place_00042 \
  --dry-run
```

`--dry-run` 不加载 factory。它通过后，再用真实 factory 做两窗口 smoke test。中断后使用同一 config、factory 和 output root：

```bash
genet-generate-long \
  --config configs/inference/long_video_action.yaml \
  --factory my_robot.genet_factory:create_runner \
  --output /local_nvme/genet/inference/pick_place_00042 \
  --resume
```

`--resume` 会比较完整 long-horizon config、总 Source 帧数和 factory 提供的 `RunIdentity`。因此 factory 必须把内容 fingerprint 分别填入 `source_id`、`model_id`、`reference_id`、`normalization_id` 和 `code_revision`；需要改变这些内容时创建新的 output root，不要复用模糊 ID 把它伪装成 resume。

factory 使用 `module:function`，用于隔离机器人私有 schema、权重布局和评估器。解析结果可以直接是 `LongGenerationJob`，也可以是接受 `LongHorizonConfig`（关键字、位置参数或无参数）的 callable，并返回：

```python
LongGenerationJob(
    source=window_source,       # WindowSource
    sampler=window_sampler,     # ChunkRequest -> ChunkOutput
    identity=run_identity,      # RunIdentity，必须使用内容 ID
    evaluator=quality_gate,     # 可选；ChunkRequest, ChunkOutput -> QualityReport
    target_action_dim=action_d, # 推荐显式提供；首窗即检查 action width
)
```

`WindowSource` 提供 `num_video_frames` 和 `read_window(start_frame, config)`；内存或 memmap 数据可使用 `TensorWindowSource`。`WindowSampler` 必须联合返回 `ChunkOutput.video[C,T,H,W]` 和 `action[T,D]`（transition 模式为 `[T-1,D]`），可在 `metadata` 中记录 solver/normalizer 信息。`target_action_dim` 会让默认 evaluator 在首窗就校验 Target action width；若省略，sampler 也可暴露同名属性。官方 bridge 是 [`CosmosLongHorizonSampler`](../src/genet/integrations/cosmos_long_horizon.py)，它会从 Target template 的 `raw_action_dim` 自动提供该属性，再构造 clean-prefix batch、调用 `generate_samples_from_batch()` 和 `decode()`。

factory 不应自行推进全局窗口、选择候选或改写 `RUN.json`，这些职责属于 orchestrator。当前 `ChunkOutput` 没有独立 timestamp/latent 字段；正式 exporter 应依据 Source canonical timebase 和 active chunk manifest 输出 timestamp，并把该契约纳入 `RunIdentity`/dataschema。

## 11. RoCE 与多 rank 限制

长推理是一个有序状态机，不等同于训练时的 32-rank 数据并行：

- 同一 run 的窗口存在父子依赖，不能把连续窗口分发给不同 rank 独立生成；
- 如果 checkpoint 需要 FSDP/HSDP，参与一次候选采样的所有 rank 必须执行相同 collective；只有 coordinator 写 journal 和最终 artifact；
- 当前 GenET Cosmos adapter 仍要求 `CP=1`、CFG parallel=1、每 rank 一个 packed sample；不能照搬训练的 `8 shard × 4 replicate` 就假定单个推理 run 获得 4 倍加速；
- 当前 CLI 不会自动拆分 replica group 或并行候选；如需吞吐并行，优先由调度器启动多个独立 output run，每组必须拥有完整模型通信拓扑；
- 多节点无共享文件系统时，一个 run 的 journal/output root 只需由 rank-0 coordinator 节点持有；其他 rank 仍需能读取相同 Source/factory 条件并参与 sampler collective，但不能各写一份同名 `RUN.json`；
- 若一次推理必须跨 RoCE 做 FSDP all-gather，使用训练指南中的 HCA/GID/NCCL 预检，并实测每 solver step 的通信；Cosmos3-Edge 若能在单节点完成，优先避免跨节点时延与恢复复杂度。

启动时所有 rank 会比较 config fingerprint、`RunIdentity` 和 Source 总帧数；每窗还比较 Source/candidate shape 与 dtype。所有 rank 都调用 sampler，只有 rank 0 执行 evaluator 和 journal mutation；rank-0 mutation 和 rank-local Python 异常会协调成跨 rank 失败，避免其他 rank 静默推进。底层 CUDA/NCCL 进程直接崩溃仍依赖 torchrun/NCCL timeout 拉起新作业。

无共享文件系统上的 resume 要固定 rank-0 物理节点，或在重启前把完整 output root 预分发到新的 rank-0 本地同一路径；仅其他 rank 拥有副本没有作用。任一 rank OOM/NaN/退出时，该候选整体失败，其他 rank 不得跳过 collective 后继续写 accepted state。跨 replica 候选并行属于后续调度扩展，不能假定当前 CLI 已实现。

## 12. 与训练的关系

rolling 编排、retry/rollback、质量门、seed DAG、journal 和 mux 都是推理侧能力，不需要改变现有 S1–S4 训练流程。

但 clean Target prefix 是现有 GenET 微调分布中没有显式训练的条件形态。固定上游 Cosmos 已提供 clean vision/action condition index，第一版应先复用它做推理验证。必须记录以下指标：

- prefix preservation error；
- seam hard-gate rejection rate；
- 每窗口平均 attempts 与 rollback rate；
- identity/task/action-sync 随生成长度的漂移曲线。

如果真实模型在 2、10、100 窗口测试中 seam/rollback 明显恶化，再加入轻量 continuation augmentation，而不是先改变主训练：

- 仍用 `T=81`；
- 仅 10%–20% batch 随机取 `O∈{9,17,33}`；
- vision/action 共用 prefix mask；
- prefix 是 clean condition，loss 只计算未知 suffix；
- 其余 batch 保持全窗口生成，shared/dual 主消融仍使用完全相同的数据和 mask 日程。

这属于后续可选实验；不能在 shared/dual 对比中只给一个分支加入 continuation 数据。

## 13. 正式 dataschema TODO

当前通用 schema 不足以安全驱动真实机器人长 rollout。接入正式 dataschema 时必须补齐：

- action 每一维的名称、类型、单位、坐标系、有效范围和 normalization hash；
- position/velocity/torque/SE(3)/gripper 的专用插值、边界和 blend policy；
- `action[t]` 与 frame `t` 的因果含义，以及 observation/action 精确 offset；
- 原始视频/动作 timestamp 和 canonical rational timebase；
- Source 末尾 padding、停机动作与 gripper hold 的明确语义；
- embodiment registry/version、kinematics/URDF hash、FK/IK evaluator；
- reference task/episode exclusion、数据 lineage 和 license；
- 每本体质量阈值、simulator/real-robot safety gate；
- 长 Source、Reference、model、VAE、normalizer 的内容 fingerprint；
- schema migration 与旧 journal 的兼容/拒绝规则。

在这些字段到位前，生成 action 只能用于离线研究和 simulator 验证，不能直接下发真实机器人。

## 14. 验收测试

至少覆盖：

- 长度 1、17、81、82、145 等边界无漏帧、重复或 video/action 数量分歧；
- transition alignment 拒绝 overlap 0，并正确处理 overlap 1 的零长度 action context；
- `81/17/64` 与 VAE 5-token overlap 的索引对齐；
- 每个 solver step 的 vision/action clean mask 正确，最终 prefix 相等；
- seed 不受候选执行顺序、rank 和恢复影响，换 parent 后子分支 seed 改变；
- synthetic seam 能触发视频、action 和 sync gate；
- 单候选失败触发 retry，全候选失败触发 parent rollback；
- rollback 永不跨 finalized boundary；
- 在 attempt 写入、chunk fsync/rename、`RUN.json` 更新位置故障注入，恢复 active chain 一致；
- active NPZ 损坏时 fail-fast，孤立/未引用 chunk 不被接受；
- toy factory 中断恢复后已接受 canonical video/action prefix 一致；
- Cosmos checkpoint 完成 2、10、100 窗口 smoke/soak，并记录显存、实时因子、attempt/rollback 与漂移；
- 部署 exporter/mux 可重复执行，输出长度、timestamp 和总 manifest 幂等。

两进程 Gloo 的 rank 分叉与 rank-0 journal/commit 失败测试默认不在受限桌面 sandbox 中打开；在允许本地 socket 的 Linux/集群节点执行：

```bash
GENET_RUN_DISTRIBUTED_TESTS=1 pytest -q tests/test_long_horizon_distributed.py
```
