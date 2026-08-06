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
| `configs/experiments/stage1_control_32gpu_ddp.yaml` | DDP stage1 + offline W&B + eval video |

## DONE（截至 2026-08-06）

- 双向 embodiment / MDS 预处理 / 镜像 `d66bdf0`→`4369598` 修复链
- Docker 存储迁 `/dev/shm`；预处理 train=34822 / val=1808
- 节点互联 + 单轨 RoCE 结论（见 `nm-genet`）；GenET `s1.env` 已改回单轨
- HF 软链接指纹：改为锁实体 checkpoint 目录
- `ARTIFACTS.json` `converted` 推断：`4369598`
- 日志目录重名：入口脚本用时间戳 `run-id`
- **增量**：`logging.wandb` + `logging.eval_video`（本地 MP4，可选 W&B）
- **增量**：`scripts/entry_s1_train.sh` 训练入口

## ONGOING / 遗留

- 完整 32 卡正式训练尚未跑通验证（闸门多次卡在 release/物料，后已修；需用入口脚本重跑）
- W&B **online** 需要本机/跳板端口转发或 HTTPS 代理；当前 stage1 默认 `offline`（曲线写在节点本地 wandb 目录）
- multi-rail 仍不可用；保持单轨 `mlx5_2`
- 缺 `nvidia-peermem` 时 GDR 可能打折（容器会提示）
- checkpoint 外传 S3 / registry 重建 runbook 仍待补

## W&B / eval video 说明

配置在 YAML `logging:` 下（见 `configs/base.yaml`）。

- `logging.wandb.mode`: `disabled` | `offline` | `online`
- online 时在 launch 环境导出：`WANDB_API_KEY`，以及 `HTTPS_PROXY` 或 `WANDB_BASE_URL`
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
