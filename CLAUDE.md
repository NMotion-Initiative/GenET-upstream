# GenET @ Hyperbolic 集群 — 会话上下文（2026-08-05 由 Helin 的助手写入）

**先看实时状态：`/root/GENET_STATUS.md`、`/root/GENET_HEARTBEAT.md`，以及开发日志 `docs/DEVLOG.md`。**

## 训练入口（直接启动）

```bash
cd /root/GenET
tmux new -s genet-train   # 或 attach
bash scripts/entry_s1_train.sh              # 全闸门 → 正式训
bash scripts/entry_s1_train.sh --dry-run-only
```

详情与 W&B / eval video 说明见 [`docs/DEVLOG.md`](docs/DEVLOG.md)。

## 你在哪
- 本机 = h32 = **rank0** = 10.0.2.2（antelope-0）；docker registry `10.0.2.2:5000` 也在本机（数据持久在 /mnt/nvme/docker-registry）。
- 三台 worker：10.0.2.4（antelope-1）、10.0.2.3（antelope-2）、10.0.2.1（antelope-3），`ssh -i /root/.ssh/id_cluster root@10.0.2.x`。每节点 8×H100。

## 项目一句话
跨本体机器人视频+动作联合生成（Cosmos3-Edge MoT 3.37B + 冻结 Wan2.2 VAE，rectified flow），数据 = RoboTwin v1 MDS（5 本体×22 任务），位于 `/mnt/nvme/mds-cache/robotwin_v1`（`genet/processed/` 为预处理产物，约 300G+）。负责人：Helin（2026-08-05 起从聪晟/ACondaway 接手）。

## 训练方案（已定，勿走回头路）
- **DDP whole-model-per-GPU + HF snapshot warm start**（commit `d66bdf0`）；**不做 DCP 转换**。
- config：`configs/experiments/stage1_control_32gpu_ddp.yaml`；启动链：`scripts/launch_cluster_ssh.sh`（支持 `--verify-release-only` / `--preflight-only` / `--dry-run-only` 分段旗标）。
- `s1.env` 的 `BASE_CHECKPOINT_PATH` 已改指 HF snapshot（原值指向空 DCP 目录会崩；备份在 `/secure/path/s1.env.bak-20260805`）。
- ⚠️ `docs/HYPERBOLIC.md` 的 DCP 章节与启动示例已过时（用的旧 HSDP config），勿照抄。

## 硬约束（除非 Helin 明确要求，绝不）
1. **不启动、不终止真实训练**；不 kill 正在跑的 preprocess / 编排器进程。
2. **不重启本机、不重启 docker/containerd**——四节点 docker 存储都在 /dev/shm（tmpfs），重启即丢；registry 容器的重建命令没有留档。
3. 不动 `/mnt/nvme` 下的数据与 processed 产物；不改已有代码；`main` 上有 7 个未 push 的 commit（等 GitHub push 权限）。
4. GPU 在按小时烧钱，别跑非必要的大作业；短的验证/预检类命令可以。

## 已知待办（接手清单）
开训确认（Helin 本人）→ 用 `scripts/entry_s1_train.sh` 跑通闸门/训练 → push commits（要 ACondaway 授权）→ checkpoint 外传 S3 方案（无自动外传，RAID0 随租约丢）→ registry 重建 runbook 补档 → W&B online 需代理/端口转发（stage1 默认 offline）→ NCCL 保持单轨 `mlx5_2`（勿开 multi-rail）。
