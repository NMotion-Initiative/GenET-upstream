"""Small DDP trainer for data-contract bring-up and adapter smoke training.

The production Cosmos3 path delegates RF/FSDP/DCP to cosmos-framework.  This
trainer intentionally stays dependency-light so preprocessing, synchronized
video/action flow targets, condition routing, resume, and loss masks can be
verified before allocating the 32-GPU cluster.
"""

from __future__ import annotations

import contextlib
import json
import math
from pathlib import Path
import time
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from genet.config import ProjectConfig
from genet.data import ProcessedPairDataset, collate_pairs
from genet.models.flow import (
    interpolate_rectified_flow,
    masked_token_mean,
    sample_logit_normal_sigma,
)
from genet.models.generator import CrossEmbodimentGenerator
from genet.training.checkpoint import (
    CheckpointManager,
    TrainerState,
    capture_rng_state,
    verify_committed_checkpoint,
)
from genet.training.distributed import (
    DistributedContext,
    assert_same_across_ranks,
    barrier,
    seed_everything,
    sha256_file,
)
from genet.training.stages import (
    build_optimizer,
    build_warmup_cosine_scheduler,
    configure_trainable_stage,
)


def _dtype(name: str, device: torch.device) -> torch.dtype:
    if device.type == "cpu":
        return torch.float32
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def _domain_map(dataset: ProcessedPairDataset, maximum: int) -> dict[str, int]:
    names: set[str] = set()
    for entry in dataset.entries:
        for key in ("source", "target_gt", "reference_target"):
            value = entry.get(key, {}).get("embodiment")
            if isinstance(value, str) and value:
                names.add(value)
    ordered = sorted(names)
    if len(ordered) > maximum:
        raise ValueError(
            f"processed manifest contains {len(ordered)} embodiments, but the model "
            f"supports only {maximum}: {ordered}"
        )
    return {name: index for index, name in enumerate(ordered)}


def _validate_sample_contract(sample: dict[str, Any], config: ProjectConfig) -> None:
    """Read one local NPZ and fail early on config/data shape drift."""

    expected_video = (3, config.data.num_frames, config.data.height, config.data.width)
    expected_action = (config.data.num_frames, config.data.action_dim)
    for role, key in (
        ("source", "source"),
        ("target", "target"),
        ("reference", "reference_target"),
    ):
        stream = sample[key]
        video_shape = tuple(stream["video"].shape)
        action_shape = tuple(stream["actions"].shape)
        if video_shape != expected_video:
            raise ValueError(
                f"{role} video shape {video_shape} does not match configured {expected_video}; "
                "run genet-validate-data before training"
            )
        if action_shape != expected_action:
            raise ValueError(
                f"{role} action shape {action_shape} does not match configured {expected_action}; "
                "run genet-validate-data before training"
            )


def _domain_ids(
    metadata: list[dict[str, Any]],
    domain_map: dict[str, int],
    key: str,
    device: torch.device,
) -> torch.Tensor:
    ids: list[int] = []
    for item in metadata:
        name = item[key].get("embodiment")
        if name not in domain_map:
            raise KeyError(f"unknown embodiment in batch metadata: {name!r}")
        ids.append(domain_map[name])
    return torch.tensor(ids, device=device, dtype=torch.long)


def _move_stream(stream: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        key: value.to(device=device, non_blocking=True)
        for key, value in stream.items()
    }


def _latent_frame_weights(frame_mask: torch.Tensor, temporal_factor: int) -> torch.Tensor:
    """Map raw-frame validity to the causal VAE's ``1 + T/factor`` grid."""

    if frame_mask.ndim != 2:
        raise ValueError(f"frame_mask must be [B,T], got {tuple(frame_mask.shape)}")
    if (frame_mask.shape[1] - 1) % temporal_factor:
        raise ValueError("frame mask violates the configured temporal compression factor")
    first = frame_mask[:, :1].float()
    tail = frame_mask[:, 1:].float()
    if tail.numel():
        tail = tail.reshape(tail.shape[0], -1, temporal_factor).mean(dim=-1)
        weights = torch.cat([first, tail], dim=1)
    else:
        weights = first
    return weights[:, None, :, None, None]


def _reference_mask(
    model: CrossEmbodimentGenerator,
    reference_latent: torch.Tensor,
    frame_mask: torch.Tensor,
    action_mask: torch.Tensor,
) -> torch.Tensor | None:
    parts: list[torch.Tensor] = []
    if model.config.reference.use_video:
        latent_valid = _latent_frame_weights(frame_mask, model.tokenizer.temporal_factor)
        latent_valid = latent_valid[:, 0, :, 0, 0] > 0
        patches_h = math.ceil(reference_latent.shape[-2] / model.patch_size)
        patches_w = math.ceil(reference_latent.shape[-1] / model.patch_size)
        parts.append(latent_valid.repeat_interleave(patches_h * patches_w, dim=1))
    if model.config.reference.use_action:
        parts.append(action_mask.bool().any(dim=-1))
    return torch.cat(parts, dim=1) if parts else None


def _all_reduce_mean(value: torch.Tensor, context: DistributedContext) -> torch.Tensor:
    value = value.detach().float()
    if context.world_size > 1:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value.div_(context.world_size)
    return value


def _checkpoint_path(path: str | None) -> Path | None:
    if not path:
        return None
    candidate = Path(path).expanduser().resolve()
    if candidate.is_dir() and (candidate / "latest.json").is_file():
        latest = json.loads((candidate / "latest.json").read_text(encoding="utf-8"))
        candidate = candidate / str(latest["path"])
    return candidate


def _load_warm_start(
    manager: CheckpointManager,
    path: Path,
    model: torch.nn.Module,
    *,
    copy_shared_reference_to_dual: bool,
) -> None:
    if path.is_dir():
        verify_committed_checkpoint(path)
        path = path / "model.pt"
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    if copy_shared_reference_to_dual:
        state = dict(state)
        for key, value in list(state.items()):
            marker = ".projector.route_a."
            if marker in key:
                state.setdefault(key.replace(marker, ".projector.route_b."), value.clone())
    target = model.module if hasattr(model, "module") else model
    missing, unexpected = target.load_state_dict(state, strict=False)
    print(
        json.dumps(
            {
                "event": "warm_start",
                "path": str(path),
                "missing_keys": len(missing),
                "unexpected_keys": len(unexpected),
            },
            sort_keys=True,
        ),
        flush=True,
    )


class _BatchCursor:
    def __init__(
        self,
        loader: DataLoader[dict[str, Any]],
        sampler: DistributedSampler,
        state: TrainerState,
    ) -> None:
        self.loader = loader
        self.sampler = sampler
        self.state = state
        self.iterator: Any = None
        self._reset(skip=state.batches_in_epoch)

    def _reset(self, *, skip: int = 0) -> None:
        self.sampler.set_epoch(self.state.epoch)
        self.iterator = iter(self.loader)
        for _ in range(skip):
            try:
                next(self.iterator)
            except StopIteration as exc:
                raise ValueError(
                    "resume batches_in_epoch exceeds the current dataloader length; "
                    "the manifest or loader configuration changed"
                ) from exc

    def next(self) -> dict[str, Any]:
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.state.epoch += 1
            self.state.batches_in_epoch = 0
            self._reset()
            batch = next(self.iterator)
        self.state.batches_in_epoch += 1
        return batch


def run_standalone_training(
    config: ProjectConfig,
    context: DistributedContext,
    *,
    dry_run: bool = False,
) -> None:
    """Run the dependency-light synchronized RF trainer."""

    seed_everything(config.train.seed, context.rank)
    manifest = Path(config.data.manifest).expanduser().resolve()
    if not manifest.is_file():
        raise FileNotFoundError(f"processed manifest does not exist: {manifest}")
    assert_same_across_ranks("manifest_sha256", sha256_file(manifest))
    assert_same_across_ranks(
        "distributed_config_fingerprint", config.distributed_fingerprint()
    )

    dataset = ProcessedPairDataset(
        manifest,
        sample_format="generic",
        reference_mode=config.data.reference_mode,
        reference_seed=config.data.reference_seed,
        shard_by_rank=False,
    )
    if len(dataset) == 0:
        raise ValueError("processed manifest contains no samples")
    _validate_sample_contract(dataset[context.rank % len(dataset)], config)
    domains = _domain_map(dataset, config.model.num_embodiments)
    assert_same_across_ranks("embodiment_map", domains)
    sampler = DistributedSampler(
        dataset,
        num_replicas=context.world_size,
        rank=context.rank,
        shuffle=True,
        seed=config.train.seed,
        drop_last=config.loader.drop_last,
    )
    loader_generator = torch.Generator()
    loader_generator.manual_seed(config.train.seed + 100_000 + context.rank)
    loader_kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": config.loader.micro_batch_size,
        "sampler": sampler,
        "num_workers": config.loader.num_workers,
        "pin_memory": config.loader.pin_memory and context.device.type == "cuda",
        "drop_last": config.loader.drop_last,
        "collate_fn": collate_pairs,
        # Keep worker base-seed draws off the model/noise RNG stream so an
        # iterator reconstructed during resume cannot perturb RF noise.
        "generator": loader_generator,
    }
    if config.loader.num_workers > 0:
        loader_kwargs.update(
            persistent_workers=config.loader.persistent_workers,
            prefetch_factor=config.loader.prefetch_factor,
        )
    loader: DataLoader[dict[str, Any]] = DataLoader(**loader_kwargs)
    if len(loader) == 0:
        raise ValueError(
            "dataloader has zero batches; add samples or disable loader.drop_last for smoke tests"
        )

    precision = _dtype(config.model.dtype, context.device)
    core_model = CrossEmbodimentGenerator(
        config.model,
        action_dim=config.data.action_dim,
        temporal_factor=config.data.temporal_compression_factor,
    ).to(device=context.device, dtype=precision)
    counts = configure_trainable_stage(core_model, config.train.stage)
    if config.model.compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError("model.compile requires torch.compile support")
        forward_model: torch.nn.Module = torch.compile(core_model)
    else:
        forward_model = core_model
    model: torch.nn.Module = forward_model
    if context.world_size > 1:
        model = DistributedDataParallel(
            forward_model,
            device_ids=[context.local_rank] if context.device.type == "cuda" else None,
            output_device=context.local_rank if context.device.type == "cuda" else None,
            broadcast_buffers=False,
            # Some reference layers are intentionally disabled by the injection cadence.
            find_unused_parameters=True,
        )
    optimizer = build_optimizer(
        model,
        new_module_lr=config.train.new_module_lr,
        base_lr=config.train.base_lr,
        weight_decay=config.train.weight_decay,
    )
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        warmup_steps=config.train.warmup_steps,
        max_steps=config.train.max_steps,
    )
    checkpoint_manager = CheckpointManager(config.checkpoint.output_dir)
    state = TrainerState(step=0, epoch=0, batches_in_epoch=0)
    resume = _checkpoint_path(config.checkpoint.resume)
    warm_start = _checkpoint_path(config.checkpoint.warm_start)
    if resume is not None:
        state = checkpoint_manager.load(
            resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            rank=context.rank,
        )
    elif warm_start is not None:
        _load_warm_start(
            checkpoint_manager,
            warm_start,
            model,
            copy_shared_reference_to_dual=(
                config.checkpoint.copy_shared_reference_to_dual
                and config.model.reference.projection_mode == "dual"
            ),
        )

    if context.is_main:
        print(
            json.dumps(
                {
                    "event": "initialized",
                    "backend": "toy",
                    "samples": len(dataset),
                    "world_size": context.world_size,
                    "micro_batch_size": config.loader.micro_batch_size,
                    "grad_accum_steps": config.train.grad_accum_steps,
                    "global_batch_size": (
                        config.loader.micro_batch_size
                        * config.train.grad_accum_steps
                        * context.world_size
                    ),
                    "embodiment_map": domains,
                    "parameters": counts,
                    "precision": str(precision).removeprefix("torch."),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if dry_run:
        barrier()
        return

    cursor = _BatchCursor(loader, sampler, state)
    train_start = time.monotonic()
    while state.step < config.train.max_steps:
        optimizer.zero_grad(set_to_none=True)
        aggregate_video = torch.zeros((), device=context.device)
        aggregate_action = torch.zeros((), device=context.device)
        aggregate_total = torch.zeros((), device=context.device)
        for accumulation_index in range(config.train.grad_accum_steps):
            batch = cursor.next()
            source = _move_stream(batch["source"], context.device)
            target = _move_stream(batch["target"], context.device)
            reference = _move_stream(batch["reference_target"], context.device)
            metadata = batch["metadata"]
            source_domain = _domain_ids(metadata, domains, "source", context.device)
            target_domain = _domain_ids(metadata, domains, "target_gt", context.device)
            reference_domain = _domain_ids(
                metadata, domains, "reference_target", context.device
            )
            with torch.no_grad():
                target_latent = core_model.encode_video(target["video"])
                source_latent = core_model.encode_video(source["video"])
                reference_latent = (
                    core_model.encode_video(reference["video"])
                    if config.model.reference.enabled
                    else target_latent
                )
            target_action = target["actions"].to(dtype=precision)
            source_action = source["actions"].to(dtype=precision)
            reference_action = reference["actions"].to(dtype=precision)
            sigma = sample_logit_normal_sigma(
                target_latent.shape[0], device=context.device
            )
            video_flow = interpolate_rectified_flow(target_latent, sigma)
            action_flow = interpolate_rectified_flow(target_action, sigma)
            use_source = bool(
                torch.rand((), device=context.device) >= config.train.condition_dropout
            )
            use_reference = bool(
                torch.rand((), device=context.device) >= config.train.condition_dropout
            )
            ref_mask = _reference_mask(
                core_model,
                reference_latent,
                reference["frame_mask"],
                reference["action_mask"],
            )
            sync_context = (
                model.no_sync()  # type: ignore[attr-defined]
                if isinstance(model, DistributedDataParallel)
                and accumulation_index + 1 < config.train.grad_accum_steps
                else contextlib.nullcontext()
            )
            with sync_context:
                with torch.autocast(
                    device_type=context.device.type,
                    dtype=precision,
                    enabled=context.device.type == "cuda" and precision != torch.float32,
                ):
                    output = model(
                        noisy_target_video=video_flow.noisy,
                        noisy_target_action=action_flow.noisy,
                        sigma=sigma,
                        source_video=source_latent,
                        source_action=source_action,
                        source_domain_id=source_domain,
                        target_domain_id=target_domain,
                        reference_video=reference_latent,
                        reference_action=reference_action,
                        reference_domain_id=reference_domain,
                        target_action_mask=target["action_mask"],
                        reference_mask=ref_mask,
                        use_source=use_source,
                        use_reference=use_reference,
                    )
                    per_video = F.mse_loss(
                        output.video_velocity,
                        video_flow.target_velocity,
                        reduction="none",
                    )
                    video_loss = masked_token_mean(
                        per_video,
                        _latent_frame_weights(
                            target["frame_mask"],
                            config.data.temporal_compression_factor,
                        ),
                    )
                    action_loss = masked_token_mean(
                        F.mse_loss(
                            output.action_velocity,
                            action_flow.target_velocity,
                            reduction="none",
                        ),
                        target["action_mask"],
                    )
                    total_loss = (
                        config.train.loss.video * video_loss
                        + config.train.loss.action * action_loss
                    )
                    scaled_loss = total_loss / config.train.grad_accum_steps
                scaled_loss.backward()
            aggregate_video += video_loss.detach() / config.train.grad_accum_steps
            aggregate_action += action_loss.detach() / config.train.grad_accum_steps
            aggregate_total += total_loss.detach() / config.train.grad_accum_steps

        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            config.train.grad_clip,
        )
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"non-finite gradient norm at step {state.step}")
        optimizer.step()
        scheduler.step()
        state.step += 1

        should_log = state.step == 1 or state.step % config.train.log_every == 0
        if should_log:
            metrics = {
                "loss": _all_reduce_mean(aggregate_total, context).item(),
                "video_loss": _all_reduce_mean(aggregate_video, context).item(),
                "action_loss": _all_reduce_mean(aggregate_action, context).item(),
                "grad_norm": _all_reduce_mean(gradient_norm, context).item(),
            }
            if context.is_main:
                print(
                    json.dumps(
                        {
                            "event": "train",
                            "step": state.step,
                            "epoch": state.epoch,
                            "lr": [group["lr"] for group in optimizer.param_groups],
                            "elapsed_seconds": round(time.monotonic() - train_start, 3),
                            **metrics,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

        should_save = (
            state.step % config.checkpoint.save_every == 0
            or state.step == config.train.max_steps
        )
        if should_save:
            local_rng = capture_rng_state()
            if context.world_size > 1:
                gathered_rng: list[dict[str, Any] | None] = [None] * context.world_size
                dist.all_gather_object(gathered_rng, local_rng)
                rng_states = {
                    rank: value
                    for rank, value in enumerate(gathered_rng)
                    if value is not None
                }
            else:
                rng_states = {0: local_rng}
            barrier()
            if context.is_main:
                saved = checkpoint_manager.save(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    trainer_state=state,
                    config=config.as_dict(),
                    rng_states=rng_states,
                )
                print(
                    json.dumps(
                        {"event": "checkpoint", "path": str(saved), "step": state.step},
                        sort_keys=True,
                    ),
                    flush=True,
                )
            barrier()
