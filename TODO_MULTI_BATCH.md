# TODO: 跨本体适配器多样本打包支持(max_samples_per_batch > 1)

状态:方案(未实施)。目标:在不改模型尺寸、不动 DDP 整模型部署的前提下,
把 packing loader 的 `max_samples_per_batch` 从 1 提到 N(目标 4),吃满当前
仅 ~24/80 GiB 的显存,吞吐预期 ×3–4。

背景:实测 `s1-smoke-5000st-20260806-071831` iter 1000 峰值显存 24.4 GiB。
限制不在 Cosmos 框架(其 packing、σ 调度、采样均原生支持多样本),而在
GenET 适配层的三处单样本假设,见下。

## 现状:单样本假设清单

| # | 位置 | 假设 |
|---|---|---|
| 1 | `src/genet/training/cosmos.py:549`(train loader)与 eval_video 回调的 `packing_kwargs` | `max_samples_per_batch=1` 硬编码 |
| 2 | `src/genet/models/cosmos_adapter.py:312-323`(非 `ar_dm` 路由) | 把 gen 序列当作单一 `[vision | action | 其余]` 连续区段,只用两个标量 `state.vision_token_count` / `action_token_count` 切片;报错信息即写明 "keep batch_size=1" |
| 3 | `src/genet/models/cosmos_adapter.py:414-416` | token 计数取整个 pack 的总长(`len(packed_seq.vision.sequence_indexes)`),多样本时区段边界信息丢失 |
| 4 | `cosmos_adapter.py::_build_reference_tokens` | `len(state.reference_vision) != 1 → RuntimeError`,参考条件只支持一个样本 |
| 5 | `src/genet/config.py:285` | `cosmos3_edge` 强制 `loader.micro_batch_size == 1`(外层 batch,可保留;打包内样本数是另一维度) |

有利条件:`CosmosConditionState` 的字段本来就是 per-sample list
(`source_vision: list[Tensor]` 等),`source_control` 路径已按
`zip(original_vision, state.source_vision, strict=True)` 逐样本处理——
基础数据结构无需重构。

## 开发方案

### M1 配置与装载管道(~2h)

- `genet/config.py`:`LoaderConfig` 新增 `max_samples_per_pack: int = 1`,
  校验 `>= 1`;`cosmos3_edge` 的 `micro_batch_size==1` 门保持不变。
- `training/cosmos.py`:train loader 与 eval_video 回调的
  `max_samples_per_batch` 改为取 `project.loader.max_samples_per_pack`
  (eval 先固定 1,见 M5)。
- 验证 `CosmosProcessedPairDataset` 的 source/reference 辅助流在
  `CosmosResumeAwarePackingDataLoader` 打包后仍按样本顺序对齐
  (`_encode_auxiliary` 的输出 list 顺序 == pack 内样本顺序)。

### M2 逐样本 token 区段(~3h)

- `CosmosConditionState` 增加 `vision_spans: list[tuple[int, int]]` 与
  `action_spans: list[tuple[int, int]]`,替代两个标量计数
  (标量字段保留一版做兼容断言后删除)。
- `cosmos_adapter.py:414` 处:从 `packed_seq.vision.sequence_indexes`
  按样本边界(与 `packed_seq.vision.tokens` 的 per-item 长度一致)累计出
  每样本 `(start, end)`;action 同理。
- `cosmos_adapter.py:312` 的非 `ar_dm` 路由改为:对每个样本 i,
  `gen[vision_spans[i]]` / `gen[action_spans[i]]` 分别过 adapter;
  越界校验按 span 末端逐样本判断。

### M3 参考条件多样本化(~3h,风险最高)

- `_build_reference_tokens`:去掉 `!= 1` 拦截,为每个样本构建独立的
  reference token 张量(带各自的坐标编码)。
- `apply_reference` 的 cross-attention 必须**样本隔离**:样本 i 的 query
  只许 attend 样本 i 的 reference。两种实现二选一:
  a) 逐样本循环调用 adapter(实现简单,N≤4 时开销可接受,首选);
  b) 拼接 reference + block-diagonal attention mask(高效但改动大)。
- `use_source` / `use_reference` 的 condition dropout 维持 pack 级全局抽样
  (期望等价,保持与现有消融可比)。

### M4 单元与等价性测试(~3h)

- 合成 2–3 个不等长样本的 pack:断言 span 划分正确。
- **等价性金标准**:同一组样本,"逐个单样本 pack 前向"的输出与
  "合并成一个多样本 pack 前向"的输出逐样本一致(容差内),
  两种 routing 模式都测。
- **隔离性测试**:改变样本 B 的 reference,断言样本 A 的输出不变
  (防 cross-attention 泄漏)。
- eval seed 契约:`generate_samples_from_batch` 的 `seed` 列表长度改为
  按 pack 内样本数生成(`[base + i for i in range(n)]`)。

### M5 集群验证与放量(~2h)

1. 单 GPU load check → 8 GPU → 32 GPU(复用 `/mnt/nvme/genet/load_check_8gpu.py` 模式)。
2. `--max-steps 200` smoke,`max_samples_per_pack: 2`:对比 mbs=1 基线的
   loss 曲线形态、峰值显存、步时;再升 4(预估峰值 ~40 GiB)。
3. eval_video 回调维持每 pack 单样本(`batch_size=1`),放量与训练解耦。
4. 全部通过后在 `stage1_control_32gpu_ddp.yaml` 设
   `loader.max_samples_per_pack: 4`。

### 注意事项

- **有效全局 batch 32 → ~128**:LR/warmup 是否随之调整属训练决策,
  放量前与负责人确认(线性或 sqrt 缩放二选一,或保持不变并记录)。
- token 预算 `max_num_tokens_after_packing=45056` 来自 Cosmos 模型注册表:
  当前单样本约 1.3k vision token,4 样本远未触顶,无需调整。
- 打包样本数是"尽力装箱",步间可能波动(3 或 4);loss 归一化按
  Cosmos 的 per-sample plan 处理,无需额外改动,但 W&B 建议记录
  `samples_per_pack` 便于诊断。
- 改动全部在 GenET 侧;Cosmos 框架零修改,镜像只需常规重建。

预计总工作量:1.5–2 天(含 32 卡验证)。
