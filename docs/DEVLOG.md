# GenET 开发日志（Hyperbolic / h32）

入口节点：`h32` = `10.0.2.2`。代码：`/root/GenET`。实时心跳：`/root/GENET_HEARTBEAT.md`。

## 训练入口（直接启动）

正式入口脚本（含 release lock、数据检查、单轨 RoCE env、唯一 run-id）：

```bash
cd /root/GenET

# 建议 tmux
tmux new -s genet-train
bash scripts/entry_s1_train.sh

# 或只做闸门
bash scripts/entry_s1_train.sh --dry-run-only
bash scripts/entry_s1_train.sh --preflight-only

# 真实 forward/backward smoke，结束后自动汇总 final DCP
bash scripts/entry_s1_train.sh --max-steps 20
```

监控：

```bash
tmux attach -t genet-train
# 另一个窗
ls /mnt/nvme/genet/logs/ | tail
tail -F /mnt/nvme/genet/logs/<run-id>/train-rank-0.log
nvidia-smi
```

关键环境：

| 文件 | 用途 |
|------|------|
| `/secure/path/s1.env` | 训练 job env（**单轨** `NCCL_IB_HCA==mlx5_2:1`） |
| `/secure/path/genet-hosts.txt` | 四节点 rank 序 |
| `/secure/path/image-ref.txt` | 镜像 digest |
| `/mnt/nvme/genet/checkpoints/Cosmos3-Edge` | HF 实体权重（非 symlink snapshot） |
| `configs/experiments/stage1_control_32gpu_ddp.yaml` | 32-GPU stage1 + online W&B + eval video |
| `/mnt/nvme/genet/committed/<run-id>/iter_*` | rank0 上已校验、带 `COMMITTED` 的完整 DCP |

## DONE（截至 2026-08-06）

- 双向 embodiment / MDS 预处理 / 镜像 `d66bdf0`→`4369598` 修复链
- Docker 存储迁 `/dev/shm`；预处理 train=34822 / val=1808
- 节点互联 + 单轨 RoCE 结论（见 `nm-genet`）；GenET `s1.env` 已改回单轨
- HF 软链接指纹：改为锁实体 checkpoint 目录
- `ARTIFACTS.json` `converted` 推断：`4369598`
- 日志目录重名：入口脚本用时间戳 `run-id`
- **增量**：`logging.wandb` + `logging.eval_video`（本地 MP4，可选 W&B）
- **增量**：`scripts/entry_s1_train.sh` 训练入口
- 32-GPU `s1-smoke-20st-20260806-054430` 已完成 20 steps，W&B online 同步成功
- Cosmos config/launch 兼容问题已修：OmegaConf struct、`DATASET_PATH`、reasoner JSON、`args.opts`
- dry-run 成功语义校验会写 launch-scoped attestation；同一 launch 的 train 命中后不再重复扫描 304G NPZ
- full run 成功后入口默认把四节点 final DCP 汇总到 rank0 并执行 checksum/`COMMITTED` 验证

## ONGOING / 遗留

- 20-step smoke 已跑通；50k 正式训练尚未启动
- W&B 默认 online；若后续 allocation 无直连出口，在 `/secure/path/wandb.env` 配 `HTTPS_PROXY` / `WANDB_BASE_URL`
- multi-rail 仍不可用；保持单轨 `mlx5_2`
- 缺 `nvidia-peermem` 时 GDR 可能打折（容器会提示）
- committed DCP 当前只保存在 rank0 NVMe；外部 archive/S3 仍需单独配置

## W&B / eval video 说明

配置在 YAML `logging:` 下（见 `configs/base.yaml`）。

- `logging.wandb.mode`: `disabled` | `offline` | `online`
- online 凭证从 `/secure/path/wandb.env` 注入；必要时同时设置 `HTTPS_PROXY` 或 `WANDB_BASE_URL`
- `logging.eval_video.enabled: true` 时每 `every_n_steps` 在条件推理后写  
  `{output}/eval_videos/step_XXXXXXXX/{sampleN}_pred.mp4`（及 GT）
- stage1 DDP 默认读 val manifest：  
  `/mnt/nvme/mds-cache/robotwin_v1/genet/processed/val/manifest.jsonl`

## GitHub 备份

- Mirror: https://github.com/NMotion-Initiative/genet （private，`main` = 当前 h32 HEAD）
- 本地 remote：`nm-backup`
- 推送：`git push nm-backup main`

## 相关材料

- 接手上下文：`CLAUDE.md`
- 集群基建验证：`/root/nm-genet`
- Helin Claude 会话 timeline：`/root/.claude/jobs/18748897/timeline.jsonl`
