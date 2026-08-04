# GenET 数据预处理与校验

本文档描述仓库当前实现的 `genet.raw-pair/v1` → `genet.processed-pair/v1` 数据流程。实现入口分别是：

- 原始 schema 与路径解析：[src/genet/data/schema.py](../src/genet/data/schema.py)
- Reference 选择：[src/genet/data/reference.py](../src/genet/data/reference.py)
- 视频处理：[src/genet/data/video.py](../src/genet/data/video.py)
- 动作处理：[src/genet/data/actions.py](../src/genet/data/actions.py)
- 预处理主流程：[src/genet/data/preprocess.py](../src/genet/data/preprocess.py)
- 训练 Dataset：[src/genet/data/dataset.py](../src/genet/data/dataset.py)
- 示例配置：[configs/schema.example.json](../configs/schema.example.json)

当前 v1 schema 是一个可运行的通用中间格式，不是最终的机器人动作语义标准。动作语义、旋转表示、归一化和更严格的任务去泄漏规则仍属于后续工作，见[后续 schema TODO](#后续-schema-todo)。

## 1. 环境准备

在仓库根目录安装项目。MP4 等编码视频需要可选的 PyAV 依赖；视频已经保存为 `.npy`/`.npz` 时不需要 PyAV 解码。

```bash
python -m pip install -e '.[preprocess]'
```

如需运行测试和静态检查：

```bash
python -m pip install -e '.[preprocess,dev]'
```

所有命令均同时支持已安装的 console script 和 Python module 两种调用方式，例如：

```bash
genet-preprocess --help
python -m genet.cli.preprocess --help
```

## 2. 数据流概览

一次预处理以一个 raw JSONL 为输入。每条记录包含同步的 Source、监督 Target，以及一个可选的 Reference Target：

```text
raw pairs JSONL
  ├── source                  Source 本体的任务视频 + 动作
  ├── target_gt               Target 本体的同任务 GT 视频 + 动作
  └── reference_target?       Target 本体的参考片段；缺省时从池中确定性选择
          │
          ▼
固定时间网格 + 视频采样/裁剪 + 动作重采样/补维
          │
          ▼
processed/
  ├── samples/*.npz           每个 pair 一个压缩样本
  ├── manifest.jsonl          样本与本体/episode 元数据
  ├── index.json              按 id、本体建立的索引
  └── stats.json              计数、错误、动作统计和有效帧比例
```

Source、Target、Reference 三路最终使用同一个 `T`、采样 FPS、空间分辨率和 `action_dim`。这保证训练时三路视频与动作可以逐时间步对齐。

## 3. Raw JSONL schema

JSONL 每个非空行必须是一个完整 JSON object，`id` 在文件内必须唯一。下面为了可读性进行了换行；实际 JSONL 中应写为一行：

```json
{
  "id": "task-000001",
  "source": {
    "episode_id": "franka-0042",
    "embodiment": "franka_panda",
    "video": "media/franka-0042.mp4",
    "actions": "media/franka-0042.actions.npz",
    "action_key": "actions",
    "timestamp_key": "timestamps",
    "start_time": 12.5,
    "end_time": 20.0,
    "metadata": {"task": "pick_red_cube"}
  },
  "target_gt": {
    "episode_id": "ur5-0117",
    "embodiment": "ur5e",
    "video": "media/ur5-0117.mp4",
    "actions": "media/ur5-0117.actions.npy",
    "action_timestamps": "media/ur5-0117.timestamps.npy",
    "start_time": 5.0,
    "end_time": 12.5
  },
  "reference_target": {
    "episode_id": "ur5-0302",
    "embodiment": "ur5e",
    "video": "media/ur5-0302.mp4",
    "actions": "media/ur5-0302.actions.pt",
    "action_key": "actions",
    "timestamp_key": "timestamps",
    "start_time": 0.0,
    "end_time": 18.0
  },
  "metadata": {
    "task_id": "pick_red_cube",
    "split": "train"
  }
}
```

可以直接打印代码中的 v1 JSON Schema：

```bash
genet-preprocess --print-raw-schema
```

### 3.1 Pair 顶层字段

| 字段 | 必需 | 类型 | 当前含义 |
| --- | --- | --- | --- |
| `id` | 是 | 非空字符串 | pair 的唯一稳定 ID；也参与 reference 的无状态选择和输出文件命名 |
| `source` | 是 | object | Source 本体的 vision-action episode/片段 |
| `target_gt` | 是 | object | 与 Source 任务对应的 Target 本体 GT episode/片段 |
| `reference_target` | 否 | object | Target 本体参考 episode/时间范围；缺省时从 reference pool 选择 |
| `metadata` | 否 | object | 当前按原样保留的 pair 级扩展元数据 |

### 3.2 Episode 字段

`source`、`target_gt` 和 `reference_target` 使用相同的 episode 结构。

| 字段 | 必需 | 类型/默认值 | 当前含义 |
| --- | --- | --- | --- |
| `episode_id` | 是 | 非空字符串 | episode 身份；reference 排除规则以它为准 |
| `embodiment` | 是 | 非空字符串 | 机器人本体 ID |
| `video` | 是 | 路径字符串 | 编码视频或 `.npy`/`.npz` 帧数组 |
| `actions` | 是 | 路径字符串 | `.npy`、`.npz`、`.json`、`.pt` 或 `.pth` 动作文件 |
| `start_time` | 否 | 秒，默认 `0.0` | 该片段在媒体共同时间轴上的起点 |
| `end_time` | 否 | 秒 | 片段逻辑终点，必须大于 `start_time` |
| `duration` | 否 | 秒 | `end_time = start_time + duration`；不能与 `end_time` 同时设置 |
| `video_fps` | 否 | 正数 | 无内嵌时间戳的数组视频所用 FPS；缺省使用预处理配置 |
| `action_fps` | 否 | 正数 | 无动作时间戳时所用 FPS；缺省使用预处理配置 |
| `action_start_time` | 否 | 秒，默认 `0.0` | 仅在根据 FPS 合成动作时间戳时作为第一个动作的时间 |
| `action_timestamps` | 否 | 路径字符串 | 独立动作时间戳文件；优先于动作容器内的时间戳 |
| `video_key` | 否 | 字符串 | `.npz` 视频数组的显式 key |
| `action_key` | 否 | 字符串 | `.npz`/JSON/PT mapping 内动作数组的显式 key |
| `timestamp_key` | 否 | 字符串 | mapping 内时间戳数组的显式 key |
| `metadata` | 否 | object | 当前按原样保留的 episode 级扩展元数据 |

不要同时填写 `end_time` 和 `duration`。当二者都不提供时，可用范围由媒体的最后时间戳决定。

### 3.3 相对路径规则

`video`、`actions` 和 `action_timestamps` 的相对路径都相对于 **raw JSONL 所在目录** 解析，而不是相对于启动命令时的 working directory。例如：

```text
/datasets/genet/raw/
  ├── pairs.train.jsonl
  └── media/
      ├── episode-1.mp4
      └── episode-1.actions.npy
```

在 `pairs.train.jsonl` 中应写 `"video": "media/episode-1.mp4"`。四个节点上的 raw 根目录可以不同，只要 JSONL 与其相对媒体目录的关系一致。

### 3.4 时间轴前提

视频时间戳、动作时间戳、`start_time`/`end_time` 必须采用同一秒级时间原点。当前实现不会估计视频与机器人日志之间的 clock offset，也不会自动做事件对齐。若原始采集系统有不同 clock domain，必须先在数据导出阶段校正。

## 4. Reference Target 规则

### 4.1 显式 reference

当 raw pair 含 `reference_target` 时，schema 会立即检查：

1. `reference_target.embodiment == target_gt.embodiment`；
2. `reference_target.episode_id != target_gt.episode_id`。

违反任一条件都会使该 raw record 无效。显式 reference 不需要等到训练时再抽样，它会作为已选择候选写入该样本 NPZ。

需要注意：当 `randomize_reference_start=true` 时，**显式 reference 也会确定性取窗**。给出的 `start_time`/`end_time` 表示允许取窗的范围，而不一定是最终窗口的精确起点。若必须从 `start_time` 精确开始，请设置：

```json
{"randomize_reference_start": false}
```

### 4.2 缺省 reference 的确定性候选池

当 pair 没有 `reference_target` 时，预处理器会：

1. 收集整个 raw manifest 内唯一的 `target_gt`，以及所有显式 `reference_target`；
2. 按 `embodiment` 分池并稳定排序；
3. 删除所有 `episode_id == 当前 target_gt.episode_id` 的候选；
4. 由 `reference_seed + sample id + target embodiment + target episode` 的稳定 BLAKE2 哈希选择一个候选。

这个过程不使用进程内随机数，也不依赖 DataLoader worker、global rank 或遍历顺序。同一 manifest、同一 seed 和同一 pair ID 会得到同一个 reference。如果目标本体池中没有另一个 episode，预处理会报错；它不会回退到当前 target episode。

### 4.3 Reference 时间窗

固定片段覆盖的时间跨度为：

```text
clip_span_seconds = (num_frames - 1) / sample_fps
```

开启 `randomize_reference_start` 后，如果 reference 可用范围长于该跨度，起点会在：

```text
[reference.start_time, reference_end - clip_span_seconds]
```

内由稳定哈希确定。实际上下界会再收紧到 reference 视频和动作的共同时间覆盖；范围不足以移动时使用共同覆盖的最早时间。

### 4.4 训练时重新选择 reference

预处理 NPZ 始终保存当时选定的 `reference_*`。`ProcessedPairDataset` 默认 `reference_mode="stored"`，直接使用这些数组。

如配置 `reference_mode="deterministic"`，Dataset 会从 processed manifest 的全部 `target_gt` 中重新建立目标本体池，用同一本体且不同 target episode 的另一条 `target_*` 替换 `reference_*`。该模式仍是无状态的。底层 `ProcessedPairDataset` 构造器另外提供 `reference_per_epoch=True` API，使 epoch 成为额外 salt；当前项目级训练配置没有暴露该开关并固定为 `False`。它与 raw 预处理候选池不是完全同一个池，因此实验配置必须记录所用 mode 和 seed。

## 5. 同步时间网格

三路分别确定 clip 起点后，都采样到同样的固定网格：

```text
t[i] = clip_start + i / sample_fps,  i = 0 ... num_frames - 1
```

- Source 和 Target 从各自 `start_time` 开始，但若边界没有与媒体时间戳对齐，会向前吸附到视频和动作都存在的第一个时间戳；
- Reference 的 `clip_start` 按上一节规则在视频/动作共同覆盖内确定；
- 三路最终均为同一个 `T = num_frames`；
- 视频和动作查询同一组秒级时间点；
- `end_time` 是逻辑边界，pad 时边界外不会读取下一任务片段，而是重复边界值并关闭 mask。

预处理不会进行 DTW、视觉事件匹配或任务阶段重定时。这里的“同步”是指每一路内部的 vision-action 按共同时间网格同步，以及三路输出具有相同时间索引长度；Source 与 Target 的语义阶段同步仍依赖 raw pair 本身正确配对。

## 6. 视频输入与处理

### 6.1 支持格式

| 输入 | 解码方式 | 时间戳/FPS |
| --- | --- | --- |
| MP4、WebM 等编码视频 | 可选 PyAV；实际格式取决于本机 FFmpeg/PyAV | 优先使用 frame PTS/time base；缺失时尝试视频流 average rate |
| `.npy` | 直接加载单个帧数组 | 没有内嵌时间戳，必须由 `video_fps` 或 `default_video_fps` 合成 |
| `.npz` | `video_key`；否则依次尝试 `video`、`frames`、`rgb`；仍不唯一则报错 | 若含固定 key `timestamps` 则使用它，否则按 FPS 合成 |

数组视频支持以下 layout：

- `THWC`
- `TCHW`
- `CTHW`
- `auto`：根据 channel 维为 1、3 或 4 自动推断

灰度帧会复制成 RGB，RGBA 会丢弃 alpha。浮点数组若最大值不超过 1，会按 `[0, 1] → [0, 255]` 转换；processed NPZ 中的视频统一存为 `uint8`。

### 6.2 时间采样与空间变换

视频在固定时间网格上采用最近时间戳采样。空间变换先保持宽高比缩放到覆盖目标画布，再做 center crop，最终得到：

```text
[T, height, width, 3] uint8
```

此处不是随机 crop。相同输入与配置会得到相同帧。

## 7. 动作输入与处理

### 7.1 支持格式

| 后缀 | 支持内容 |
| --- | --- |
| `.npy` | 单个动作数组 |
| `.npz` | mapping；优先 key 为 `actions`、`action`、`data`、`values`，也可显式设置 `action_key` |
| `.json` | 直接数组，或包含动作/时间戳的 mapping |
| `.pt` / `.pth` | PyTorch Tensor，或包含动作/时间戳的 mapping |

Mapping 内时间戳默认依次查找 `timestamps`、`time`、`t`，也可以设置 `timestamp_key`。独立 `action_timestamps` 文件的优先级更高。

动作数组按第一维解释为时间：

- `[T]` 转为 `[T, 1]`；
- `[T, ...]` 且 rank 大于 2 时，将尾部维度 flatten 为 `[T, D_source]`；
- 动作和时间戳必须全部 finite；
- 时间戳数量必须等于动作步数，并且严格递增；
- 没有时间戳时，使用 `action_fps` 或 `default_action_fps`，从 `action_start_time` 开始合成。

### 7.2 重采样

`action_resample` 支持：

- `linear`：每一个标量动作维独立线性插值；
- `nearest`：选择时间上最近的动作步。

当前线性插值不理解关节类型或旋转流形。特别是 quaternion、Euler angle 周期边界和 SE(3) 位姿不能直接假设逐标量线性插值正确，必须在后续正式动作 schema 中处理。

### 7.3 `action_dim` 与 mask

若原始 `D_source < action_dim`，动作尾部补零；若 `D_source > action_dim`：

- 默认 `truncate_actions=false`，直接报错；
- 只有显式设置 `truncate_actions=true` 才截断尾部维。

最终输出：

```text
actions      [T, action_dim] float32
action_mask  [T, action_dim] bool
```

`action_mask[t, d]` 同时表达两个条件：

1. 第 `d` 维来自真实动作而不是补零维；
2. 第 `t` 步位于该动作序列和逻辑片段的有效时间范围内。

因此 loss 必须使用 `action_mask`；不能仅根据数值是否为零判断有效性。

## 8. Wan 帧长约束与短片策略

### 8.1 `1 + 4N`

Wan 时序 VAE 默认要求：

```text
num_frames = wan_frame_offset + wan_temporal_stride * N
           = 1 + 4 * N
```

[configs/schema.example.json](../configs/schema.example.json) 当前使用 `num_frames=81`，即 `81 = 1 + 4×20`。`PreprocessConfig` 初始化时就会检查该关系。只有明确知道下游基座允许其他长度时，才应使用：

```bash
genet-preprocess \
  --manifest /data/raw/pairs.train.jsonl \
  --output /data/processed/train \
  --config configs/schema.example.json \
  --disable-wan-frame-validation
```

关闭校验不会改变采样逻辑，也不会保证 Wan 能接受该长度。

### 8.2 `drop`

`short_policy="drop"` 时，只要 Source、Target、Reference 任一路的视频/动作共同覆盖没有容纳完整网格，整个 pair 就会丢弃。非对齐的逻辑起点会先向前吸附到首个共同观测，不会仅因时间戳未对齐而误丢。短片错误会记录到 `stats.json` 的 `errors.dropped`，并计入 `dropped_short_samples`。

短片 drop 与 `--on-error` 无关：短片会继续处理下一条；`--on-error` 控制的是其他异常。

### 8.3 `pad`

`short_policy="pad"` 时：

- 视频查询越界会重复最近的首/尾边界帧；
- 动作查询越界会使用边界动作值；
- 超出真实媒体覆盖或逻辑 `end_time` 的 `frame_mask` 为 false；
- 对应动作时间步的 `action_mask` 为 false；
- `end_time` 之后不会继续读取同一底层文件里的后续任务内容。

视频另有：

```text
frame_mask [T] bool
```

它表示固定时间网格上的真实视频覆盖。训练时应将它用于视频/latent loss 或 batch 有效性判断。

## 9. 预处理配置

示例配置的 `preprocess` 部分当前为：

| 字段 | 示例值 | 作用 |
| --- | ---: | --- |
| `num_frames` | `81` | 三路固定 T |
| `sample_fps` | `16.0` | 固定时间网格 FPS |
| `height` / `width` | `192` / `320` | resize + center crop 后分辨率 |
| `action_dim` | `64` | 动作统一维度 |
| `action_resample` | `linear` | `linear` 或 `nearest` |
| `short_policy` | `drop` | `drop` 或 `pad` |
| `truncate_actions` | `false` | 是否允许截断超过 `action_dim` 的动作 |
| `default_video_fps` | `30.0` | 数组视频无时间戳/FPS时的默认值 |
| `default_action_fps` | `30.0` | 动作无时间戳/FPS时的默认值 |
| `video_layout` | `auto` | `auto`、`THWC`、`TCHW`、`CTHW` |
| `reference_seed` | `2026` | Reference 候选和窗口的稳定 seed |
| `randomize_reference_start` | `true` | 是否在 reference 时间范围内确定性取窗 |
| `validate_wan_frames` | `true` | 是否检查 Wan 帧长公式 |
| `wan_frame_offset` | `1` | Wan 公式 offset |
| `wan_temporal_stride` | `4` | Wan 公式 stride |

注意：代码内 `PreprocessConfig` 的 `reference_seed` 默认值是 `0`，而示例配置明确覆盖为 `2026`。正式实验应始终传入版本化配置，不应依赖隐式默认值。

配置 JSON 可以直接是 preprocess object，也可以像示例一样在顶层包含 `"preprocess": {...}`。未知配置 key 会报错，避免拼写错误被静默忽略。

## 10. 运行预处理

推荐先将 raw manifest 与配置纳入数据版本记录，再输出到一个新的空目录：

```bash
genet-preprocess \
  --manifest /data/genet/raw/pairs.train.jsonl \
  --output /data/genet/processed/train-v1 \
  --config configs/schema.example.json
```

成功后 stdout 输出 JSON 报告，包含 `manifest`、`index`、`stats`、`written`、`dropped` 和 `failed`。

CLI 可覆盖以下常用配置：

```text
--num-frames
--sample-fps
--height
--width
--action-dim
--reference-seed
--short-policy {drop,pad}
--action-resample {linear,nearest}
```

其他选项：

- `--on-error raise`：默认；非短片错误立即中止；
- `--on-error skip`：记录非短片错误到 stats 并继续；
- `--overwrite`：允许覆盖现有 metadata 文件和同名样本；
- `--disable-wan-frame-validation`：关闭帧长公式检查。

`--overwrite` 不会清理 `samples/` 中已经不再被新 manifest 引用的旧 NPZ。为获得可核验的规范数据版本，建议使用新的空输出目录；若复用目录，复制到各节点前应确认没有 stale files。

## 11. 输出格式

### 11.1 目录

```text
train-v1/
  ├── manifest.jsonl
  ├── index.json
  ├── stats.json
  └── samples/
      ├── task-000001-<stable-id-hash>.npz
      └── ...
```

样本文件名由清洗后的 `id` 和稳定摘要组成。manifest 中的 `npz` 是相对于 processed 根目录的路径，因此整个 processed 目录可移动到不同节点路径。

### 11.2 每样本 NPZ

每个压缩 NPZ 包含 12 个不使用 pickle 的数组：

| Key | Shape | Dtype |
| --- | --- | --- |
| `source_video` | `[T,H,W,3]` | `uint8` |
| `source_actions` | `[T,D]` | `float32` |
| `source_action_mask` | `[T,D]` | `bool` |
| `source_frame_mask` | `[T]` | `bool` |
| `target_video` | `[T,H,W,3]` | `uint8` |
| `target_actions` | `[T,D]` | `float32` |
| `target_action_mask` | `[T,D]` | `bool` |
| `target_frame_mask` | `[T]` | `bool` |
| `reference_video` | `[T,H,W,3]` | `uint8` |
| `reference_actions` | `[T,D]` | `float32` |
| `reference_action_mask` | `[T,D]` | `bool` |
| `reference_frame_mask` | `[T]` | `bool` |

其中 `T=num_frames`、`D=action_dim`。Dataset 读取后会把视频转为 `[C,T,H,W]`；默认 `normalize_video="minus_one_one"` 将 `uint8` 映射到 `[-1,1]`。这个加载期视频变换不要与磁盘 NPZ layout 混淆。

### 11.3 `manifest.jsonl`

每行包含：

- `format_version: "genet.processed-pair/v1"`；
- `id`；
- 相对 `npz` 路径；
- `source`、`target_gt`、`reference_target` 的 `episode_id`、`embodiment`、最终 `clip_start` 和保留的 metadata；
- `shape.video` 与 `shape.actions`；
- 原 pair 的 `metadata`。

### 11.4 `index.json`

索引包含：

- `num_samples`；
- `by_id`：样本 ID 到 manifest 的零基 line index 和 NPZ 路径；
- `by_target_embodiment`：目标本体到样本 ID 列表；
- manifest 文件名和 format version。

### 11.5 `stats.json`

统计文件包含：

- 完整 preprocess config；
- raw、written、dropped-short、failed 样本数；
- dropped/failed 的样本 ID 与错误文本；
- Source、Target、Reference 各自逐动作维的 `count`、`mean`、`std`；
- 三路 `frame_valid_fraction`。

动作统计只对 `action_mask=true` 的元素计数。当前 stats 是该次输入 manifest 的描述性统计，不应直接假定它是可用于模型归一化的 train-only statistics，见[后续 schema TODO](#后续-schema-todo)。

## 12. 校验 processed 数据

预处理完成后、复制到节点前必须运行：

```bash
genet-validate-data \
  --manifest /data/genet/processed/train-v1/manifest.jsonl \
  --num-frames 81 \
  --height 192 \
  --width 320 \
  --action-dim 64 \
  --cosmos \
  --output /data/genet/processed/train-v1.validation.json
```

`--manifest` 也可以传 processed 目录。若不指定 `--num-frames`，校验器从第一个可用样本推断 T，并要求后续样本一致。

校验器逐样本检查：

- manifest JSON、ID 唯一性、format version；
- NPZ 路径必须是 processed 根目录内的相对路径，且存在、可读取；绝对路径、`..` 和 symlink escape 均拒绝；
- 三路 12 个数组都存在；
- video 为 `[T,H,W,3]`，actions 为 `[T,D]`；
- action mask 与 actions 同 shape，frame mask 为 `[T]`；
- 三路视频 shape 相同、三路动作 shape 相同；
- 所有数组 numeric 且 finite；
- mask 仅含 bool/0/1；
- Source/Target/Reference metadata 均存在且含 embodiment/episode；Target/Reference 本体一致且 episode 不同；
- manifest 非空。

`--cosmos` 另外要求视频为 `uint8`、三路 frame mask 全有效、每个 action mask 是从第 0 维开始的连续有效前缀且真实通道没有 temporal padding。这是 production 必选预检；只用于 standalone pad-mask 实验时可以省略。

摘要始终以 JSON 输出到 stdout；`--output` 会额外保存相同 JSON。返回码：

- `0`：全部有效；
- `1`：存在任一数据或参数错误。

训练阶段不要按 rank 各自跳过坏样本。不同 rank 的 collective 次数可能因此不一致并 hang；坏样本应在本步骤完成隔离或重新预处理。

## 13. 四节点无共享存储部署

集群为 4 节点、32 GPU，节点之间没有共享文件系统。推荐流程是：**只在一个 staging 节点预处理一次，然后逐字节复制同一 processed 版本到四个节点**。不要在四个节点独立生成 NPZ 后期待压缩文件字节级 hash 必然相同；ZIP 容器元数据可能导致内容相同但文件 SHA-256 不同。

### 13.1 Staging 节点生成校验清单

```bash
cd /data/genet/processed/train-v1

(
  printf '%s\n' manifest.jsonl index.json stats.json
  find samples -type f -name '*.npz' -print
) | LC_ALL=C sort | xargs sha256sum > SHA256SUMS

sha256sum -c SHA256SUMS
sha256sum SHA256SUMS
```

第一条摘要覆盖 manifest、index、stats 和所有 NPZ，但故意不包含节点路径相关的 `*.validation.json`。最后一条输出 `SHA256SUMS` 自身的聚合 hash，应记录在数据版本日志中。

### 13.2 复制到每个节点

以下命令只是示意；请替换节点名和专用目标目录：

```bash
rsync -a /data/genet/processed/train-v1/ node0:/local-data/genet/train-v1/
rsync -a /data/genet/processed/train-v1/ node1:/local-data/genet/train-v1/
rsync -a /data/genet/processed/train-v1/ node2:/local-data/genet/train-v1/
rsync -a /data/genet/processed/train-v1/ node3:/local-data/genet/train-v1/
```

请使用新的空目标目录，或由运维流程显式清理旧版本，避免 stale NPZ。不要对含其他数据的宽泛目录使用破坏性同步选项。

### 13.3 每节点核验

在四个节点分别执行：

```bash
cd /local-data/genet/train-v1
sha256sum -c SHA256SUMS
sha256sum SHA256SUMS

genet-validate-data \
  --manifest manifest.jsonl \
  --num-frames 81 \
  --height 192 \
  --width 320 \
  --action-dim 64 \
  --cosmos \
  --output validation.local.json
```

四个节点必须同时满足：

1. `sha256sum -c` 全部为 `OK`；
2. `SHA256SUMS` 自身 hash 完全一致；
3. `genet-validate-data` 返回 0，且 `samples`、`valid_samples`、`expected_num_frames` 一致。

processed manifest 的 NPZ 路径是相对路径，因此每个节点的 processed 根目录无需相同。训练进程只需指向本节点的 `manifest.jsonl`。

### 13.4 32 个 global rank 的读取

当每个节点都有完整副本时，`ProcessedPairDataset(shard_by_rank=True)` 会按：

```text
global_rank, global_rank + world_size, global_rank + 2 * world_size, ...
```

对全局 manifest 做 stride sharding；`RANK`/`WORLD_SIZE` 可由 torchrun 环境读取。启用 dataset-level sharding 后不要再叠加 `DistributedSampler`，否则会重复分片。也可以关闭 `shard_by_rank` 并只使用标准 `DistributedSampler`，但两种策略必须二选一。

## 14. 常见错误

### `no reference for embodiment ...`

该目标本体没有另一个 episode 可作为 reference。增加同本体候选，或提供 episode 不同的显式 `reference_target`；不要复用 target episode。

### `video/action trajectory does not cover ... time grid`

输入没有覆盖 `(T-1)/FPS` 的完整跨度，或视频/动作 clock 不一致。确认时间戳与 `start_time`；根据训练策略选择 `drop` 或 `pad`。

### `source action dim ... exceeds configured action_dim`

当前动作维超过统一 `action_dim`。优先修正正式动作 schema 或增大 `action_dim`。仅在确认尾部维可安全舍弃后开启 `truncate_actions`。

### `timestamps must be strictly increasing`

原始日志存在乱序或重复时间戳。应在导出阶段排序并定义重复采样的聚合规则；预处理器不会静默重排或去重。

### PyAV import/解码失败

安装 `.[preprocess]` 并检查系统 FFmpeg codec。也可以先离线导出 `.npy`/`.npz` 视频帧和时间戳。

### 校验通过但训练读取 shape 不同

磁盘视频是 `[T,H,W,C]`，Dataset 输出是 `[C,T,H,W]`；batch 后再增加 batch 维。检查问题发生在磁盘 schema、Dataset 转换还是训练 adapter。

## 15. 后续 schema TODO

以下内容当前**未实现或未标准化**，在接入真实 dataschema 前必须明确：

### 15.1 动作语义

- 为每个 embodiment 定义稳定的 action field 名称和顺序，而不是只依赖匿名列索引；
- 明确 joint position/velocity/torque、Cartesian pose/twist、gripper 等字段；
- 明确单位、符号、joint convention、控制频率、控制模式；
- 明确 action 是 absolute command、delta command 还是 observation-to-next-step target；
- 定义从本体原生动作到共享 `action_dim` 的映射，并把映射版本写入 manifest。

当前补零 mask 只能表达“哪些列存在”，不能说明这些列在不同本体之间是否具有相同物理语义。

### 15.2 Rotation 与位姿插值

- 统一 Euler、quaternion、axis-angle、6D rotation 等表示；
- 规定 quaternion 分量顺序、单位化和符号连续性；
- 为角度 wrap-around 与 quaternion 使用几何正确的插值，例如 unwrap/SLERP；
- 明确 pose 所在坐标系、handedness、base/tool/camera frame 以及 frame transform 版本。

当前 `linear` 是逐标量 `np.interp`，不应直接用于未处理的旋转表示。

### 15.3 归一化

- 当前预处理仅把动作转为 `float32` 并补维，不执行动作归一化；
- 需要决定 per-dimension、per-embodiment 或共享归一化策略；
- 需要定义 continuous、binary、categorical、angle、gripper 等不同字段的变换；
- 所有统计和变换必须 mask-aware，并保存 epsilon、clip 范围和版本。

Dataset 的视频 `[-1,1]` 映射属于加载期像素变换，不等同于动作归一化。

### 15.4 Train-only statistics

当前 `stats.json` 对传入预处理 manifest 的所有 written 样本统计。正式流程必须先固定 train/val/test split，再只用 train split 计算 normalization statistics：

- val/test 不得参与 mean/std、分位数或裁剪阈值估计；
- 统计应按 `action_mask` 排除补零维和 pad 时间步；
- 统计 artifact 应有独立版本、SHA-256，并复制到全部节点；
- val/test/inference 只能读取冻结后的 train statistics。

### 15.5 Task exclusion 与数据泄漏

当前 reference 只强制 `episode_id != target_gt.episode_id`。它**不会**自动排除：

- 相同 `task_id`；
- 同一原始 rollout 被切出的另一个 clip；
- 同一场景、物体实例或相机轨迹；
- train/val/test 之间的近重复任务；
- 与 target GT 具有强内容重叠的参考片段。

正式 schema 应增加稳定的 `task_id`、`rollout_id`、`scene_id`、`object_instance_id`、`split` 和 lineage/group ID，并让 reference selector 支持按实验要求排除同 task、同 rollout、同 scene 或跨 split 候选。split 和 task exclusion 必须在建池阶段执行，而不是在训练 worker 内临时跳过。

## 16. 发布数据版本前检查表

- [ ] raw JSONL 中 ID 唯一，相对路径可在 staging 节点解析；
- [ ] 三路视频/动作采用同一时间原点，Source/Target 任务配对正确；
- [ ] Target 本体池中每条样本至少有一个不同 episode 的 reference 候选；
- [ ] `num_frames` 满足 `1+4N`，且三路使用同一个 T；
- [ ] `action_dim`、动作字段映射和 `short_policy` 已记录；
- [ ] 预处理的 written/dropped/failed 数量符合预期；
- [ ] `genet-validate-data` 返回 0；
- [ ] staging 节点生成并验证 `SHA256SUMS`；
- [ ] 四个节点逐文件 SHA-256 和聚合 hash 一致；
- [ ] 四个节点本地 validator 摘要一致；
- [ ] 训练期使用 dataset sharding 或 `DistributedSampler`，没有双重分片；
- [ ] 动作归一化只使用冻结的 train-only statistics；
- [ ] split、task exclusion 和 reference mode/seed 已写入实验记录。

多阶段训练、RoCE 启动和 checkpoint 规则见 [docs/TRAINING.md](TRAINING.md)。
