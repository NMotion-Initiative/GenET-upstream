"""Production Cosmos3-Edge experiment builder.

All heavyweight imports are local so the base package and data tooling remain
usable without the NVIDIA training container.  The builder starts from the
pinned upstream ``vision_sft_edge`` recipe, restores action generation, swaps
in GenET's model/dataset adapters, and leaves RF/FSDP/EMA/DCP to upstream.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
from typing import Any

import torch.distributed as dist

from genet.config import ProjectConfig
from genet.training.checkpoint import verify_committed_checkpoint
from genet.training.distributed import assert_same_across_ranks, raise_if_any_rank_failed, sha256_file
from genet.training.environment import assert_artifact_bound_to_receipt, assert_runtime_environment_consistent


def _manifest_domains(path: Path, maximum: int) -> dict[str, int]:
    names: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            for key in ("source", "target_gt", "reference_target"):
                name = entry.get(key, {}).get("embodiment")
                if isinstance(name, str) and name:
                    names.add(name)
    ordered = sorted(names)
    if not ordered:
        raise ValueError(f"processed manifest has no embodiment metadata: {path}")
    if len(ordered) > maximum:
        raise ValueError(
            f"manifest has {len(ordered)} embodiments but Cosmos is configured for "
            f"{maximum}: {ordered}"
        )
    return {name: index for index, name in enumerate(ordered)}


def is_hf_snapshot(path: Path) -> bool:
    """Whole-model-per-GPU deployments warm-start straight from the official
    safetensors/diffusers snapshot instead of a converted DCP."""

    return path.is_dir() and (
        (path / "model_index.json").is_file()
        or (path / "model.safetensors.index.json").is_file()
    )


def _checkpoint_load_path(config: ProjectConfig) -> tuple[str, bool]:
    """Return ``(path, exact_resume)`` for the upstream DCP checkpointer."""

    def require_dcp(path: Path) -> None:
        if not path.is_dir():
            raise FileNotFoundError(f"Cosmos DCP directory does not exist: {path}")
        if is_hf_snapshot(path):
            return
        if not any(candidate.name == ".metadata" for candidate in path.rglob(".metadata")):
            raise ValueError(f"Cosmos DCP directory has no .metadata: {path}")

    if config.checkpoint.resume:
        path = Path(config.checkpoint.resume).expanduser().resolve()
        if is_hf_snapshot(path):
            raise ValueError(
                "an exact resume needs a committed training DCP with optimizer "
                f"state, not the pretrained HF snapshot: {path}"
            )
        if config.checkpoint.require_committed:
            verify_committed_checkpoint(path)
        require_dcp(path)
        return str(path), True
    if config.checkpoint.warm_start:
        path = Path(config.checkpoint.warm_start).expanduser().resolve()
        # Converted official checkpoints do not use GenET's transport marker;
        # consolidated GenET checkpoints do, and are verified when detectable.
        if (path / "MANIFEST.json").is_file():
            verify_committed_checkpoint(path)
        require_dcp(path)
        return str(path), False
    external = os.environ.get("BASE_CHECKPOINT_PATH")
    if not external:
        raise ValueError(
            "Cosmos training requires checkpoint.warm_start/--warm-start or "
            "BASE_CHECKPOINT_PATH pointing at the converted Cosmos3-Edge DCP"
        )
    path = Path(external).expanduser().resolve()
    require_dcp(path)
    return str(path), False


def _wan_vae_path() -> Path:
    value = os.environ.get("WAN_VAE_PATH")
    if not value:
        raise ValueError("Cosmos training requires WAN_VAE_PATH")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"WAN_VAE_PATH does not exist: {path}")
    return path


def _validate_hf_cache_revision(cache_root: str | Path, expected_revision: str) -> None:
    root = Path(cache_root).expanduser().resolve()
    candidates = (
        root / "hub" / "models--nvidia--Cosmos3-Edge" / "refs" / "main",
        root / "models--nvidia--Cosmos3-Edge" / "refs" / "main",
        root / "refs" / "main",
    )
    reference = next((candidate for candidate in candidates if candidate.is_file()), None)
    if reference is None:
        raise FileNotFoundError(
            "offline Cosmos3-Edge cache has no refs/main; prefetch the pinned revision "
            f"under HF_HOME or HF_HUB_CACHE: {root}"
        )
    actual_revision = reference.read_text(encoding="utf-8").strip()
    if actual_revision != expected_revision:
        raise ValueError(
            "Cosmos3-Edge cache revision differs from GENET_HF_SNAPSHOT_REVISION: "
            f"{actual_revision} != {expected_revision}"
        )


def _disable_online_sampling(callbacks: Any) -> None:
    """Remove callbacks that call inference without source/reference inputs."""

    if callbacks is None:
        return
    for key in ("every_n_sample_reg", "every_n_sample_ema"):
        if key in callbacks:
            del callbacks[key]


def _apply_wandb_logging(resolved: Any, project: ProjectConfig) -> None:
    wandb = project.logging.wandb
    resolved.job.project = wandb.project
    resolved.job.group = wandb.group
    resolved.job.name = wandb.name or Path(project.checkpoint.output_dir).name or "run"
    resolved.job.wandb_mode = wandb.mode
    if wandb.entity:
        # Cosmos JobConfig has no entity field; W&B reads WANDB_ENTITY.
        os.environ.setdefault("WANDB_ENTITY", wandb.entity)


def _install_eval_video_callback(
    resolved: Any,
    project: ProjectConfig,
    *,
    embodiment_map: dict[str, int],
    model_config: Any,
) -> None:
    eval_video = project.logging.eval_video
    if not eval_video.enabled:
        return
    from cosmos_framework.utils.lazy_config import LazyCall as L

    from genet.training.eval_video import GenETEvalVideoCallback

    output_root = Path(project.checkpoint.output_dir).expanduser().resolve()
    fps = float(eval_video.fps if eval_video.fps is not None else project.data.fps)
    resolved.trainer.callbacks["genet_eval_video"] = L(GenETEvalVideoCallback)(
        every_n=eval_video.every_n_steps,
        num_samples=eval_video.num_samples,
        manifest=eval_video.manifest,
        output_dir=str(output_root / eval_video.output_subdir),
        fps=fps,
        log_to_wandb=eval_video.log_to_wandb,
        run_at_start=eval_video.run_at_start,
        num_sampling_steps=eval_video.num_sampling_steps,
        embodiment_map=embodiment_map,
        dataset_kwargs={
            "fps": project.data.fps,
            "reference_mode": project.data.reference_mode,
            "reference_seed": project.data.reference_seed,
            "require_bidirectional_pairs": project.data.require_bidirectional_pairs,
            "expected_embodiments": project.data.expected_embodiments,
            "action_alignment": "frame",
            "tokenizer_config": model_config.vlm_config.tokenizer,
            "max_caption_tokens": 2048,
            "use_system_prompt": model_config.vlm_config.use_system_prompt,
            "shard_seed": project.train.seed,
            "max_action_dim": project.data.action_dim,
        },
        packing_kwargs={
            "audio_sample_rate": 48_000,
            "dataset_name": "genet_pairs",
            "max_samples_per_batch": 1,
            "max_sequence_length": None,
            "patch_spatial": model_config.diffusion_expert_config.patch_spatial,
            "sound_latent_fps": 0,
            "tokenizer_spatial_compression_factor": (
                model_config.tokenizer.spatial_compression_factor
            ),
            "tokenizer_temporal_compression_factor": (
                model_config.tokenizer.temporal_compression_factor
            ),
        },
        loader_kwargs={"num_workers": 0},
        barrier_after_run=True,
    )


def _probe_cosmos_data_contract(
    project: ProjectConfig,
    *,
    manifest: Path,
    embodiment_map: dict[str, int],
    rank: int,
    world_size: int,
) -> tuple[str, dict[str, Any]]:
    """Convert one rank-local sample before constructing the expensive VFM."""

    from genet.integrations.cosmos_data import CosmosProcessedPairDataset

    dataset = CosmosProcessedPairDataset(
        manifest,
        embodiment_map=embodiment_map,
        fps=project.data.fps,
        reference_mode=project.data.reference_mode,
        reference_seed=project.data.reference_seed,
        require_bidirectional_pairs=project.data.require_bidirectional_pairs,
        expected_embodiments=project.data.expected_embodiments,
        action_alignment="frame",
        max_action_dim=project.data.action_dim,
    )
    dataset.shard_world_size = world_size
    dataset.shard_rank = rank
    if len(dataset) == 0:
        raise ValueError(
            "processed manifest must contain at least WORLD_SIZE samples so every "
            "rank receives a non-empty, equal-length shard"
        )
    sample = dataset[0]
    expected_video = (3, project.data.num_frames, project.data.height, project.data.width)
    expected_action = (project.data.num_frames, project.data.action_dim)
    for role, video_key, action_key in (
        ("source", "source_video", "source_action"),
        ("target", "video", "action"),
        ("reference", "reference_video", "reference_action"),
    ):
        video_shape = tuple(sample[video_key].shape)
        action_shape = tuple(sample[action_key].shape)
        if video_shape != expected_video:
            raise ValueError(
                f"{role} video shape {video_shape} does not match configured {expected_video}; "
                "run genet-validate-data on every node"
            )
        if action_shape != expected_action:
            raise ValueError(
                f"{role} action shape {action_shape} does not match configured {expected_action}; "
                "run genet-validate-data on every node"
            )
    return (
        str(sample.get("sample_id", rank % len(dataset))),
        dataset.direction_summary,
    )


def _validate_local_cosmos_copy(
    project: ProjectConfig,
    *,
    manifest: Path,
) -> None:
    """Validate every node-local copy in parallel, then share failures globally."""

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
    report: dict[str, Any] | None = None
    try:
        from genet.cli.validate_data import validate_manifest

        summary = validate_manifest(
            manifest,
            num_frames=project.data.num_frames,
            height=project.data.height,
            width=project.data.width,
            action_dim=project.data.action_dim,
            cosmos=True,
            require_bidirectional_pairs=project.data.require_bidirectional_pairs,
            expected_embodiments=project.data.expected_embodiments,
            shard_rank=local_rank,
            shard_world_size=local_world_size,
        )
    except Exception as exc:  # keep every rank in the following collective
        report = {
            "rank": dist.get_rank() if dist.is_initialized() else 0,
            "manifest": str(manifest),
            "error_count": 1,
            "errors": [f"{type(exc).__name__}: {exc}"],
        }
    else:
        if not summary["valid"]:
            report = {
                "rank": dist.get_rank() if dist.is_initialized() else 0,
                "manifest": str(manifest),
                "error_count": summary["error_count"],
                "errors": summary["errors"][:20],
            }

    reports: list[dict[str, Any] | None]
    if dist.is_initialized():
        reports = [None] * dist.get_world_size()
        dist.all_gather_object(reports, report)
    else:
        reports = [report]
    failures = [item for item in reports if item is not None]
    if failures:
        raise ValueError(
            "Cosmos data validation failed on one or more node-local copies: "
            + json.dumps(failures, ensure_ascii=False, sort_keys=True)
        )


def _build_cosmos_config(
    project: ProjectConfig,
    *,
    manifest: Path,
    embodiment_map: dict[str, int],
    checkpoint_load: tuple[str, bool] | None = None,
) -> Any:
    try:
        from cosmos_framework.configs.base.config import make_config
        from cosmos_framework.utils.config_helper import override
        from cosmos_framework.utils.lazy_config import LazyCall as L
    except ImportError as exc:  # pragma: no cover - requires NVIDIA container
        raise RuntimeError(
            "Cosmos backend is unavailable. Run scripts/bootstrap_cosmos.sh, install "
            "the pinned cosmos-framework environment, then install GenET with the "
            "'cosmos' extra."
        ) from exc

    from genet.integrations.cosmos_data import CosmosProcessedPairDataset
    from genet.integrations.cosmos_loader import (
        CosmosInfiniteRankPartitionedDataLoader,
        CosmosResumeAwarePackingDataLoader,
    )
    from genet.models.cosmos_adapter import CosmosCrossEmbodimentModel

    # Resolve the official Edge recipe first, preserving upstream model/checkpoint
    # defaults not owned by this project.
    resolved = override(make_config(), ["--", "experiment=vision_sft_edge"])
    model_config = resolved.model.config
    model_config.action_gen = True
    model_config.sound_gen = False
    model_config.max_action_dim = project.data.action_dim
    model_config.num_embodiment_domains = project.model.num_embodiments
    model_config.precision = project.model.dtype
    model_config.compile.enabled = project.model.compile
    model_config.activation_checkpointing.mode = project.model.activation_checkpointing
    model_config.parallelism.data_parallel_shard_degree = (
        project.model.parallelism.data_parallel_shard_degree
    )
    model_config.parallelism.data_parallel_replicate_degree = (
        project.model.parallelism.data_parallel_replicate_degree
    )
    model_config.parallelism.context_parallel_shard_degree = (
        project.model.parallelism.context_parallel_shard_degree
    )
    model_config.parallelism.cfg_parallel_shard_degree = (
        project.model.parallelism.cfg_parallel_shard_degree
    )
    model_config.rectified_flow_training_config.loss_scale = project.train.loss.video
    model_config.rectified_flow_training_config.action_loss_weight = (
        project.train.loss.action
    )
    wan_vae_path = _wan_vae_path()
    model_config.tokenizer.vae_path = str(wan_vae_path)

    cross_config = dataclasses.asdict(project.model)
    resolved.model["_target_"] = CosmosCrossEmbodimentModel
    resolved.model["_recursive_"] = False
    resolved.model["cross_embodiment_config"] = cross_config
    resolved.model["copy_shared_reference_to_dual_on_warm_start"] = (
        project.checkpoint.copy_shared_reference_to_dual
        and project.checkpoint.resume is None
    )
    resolved.model["initialize_ema_from_regular_on_warm_start"] = (
        project.checkpoint.resume is None
    )
    resolved.model["condition_dropout"] = project.train.condition_dropout

    worker_kwargs: dict[str, Any] = {
        "batch_size": project.loader.micro_batch_size,
        "in_order": True,
        "num_workers": project.loader.num_workers,
        "persistent_workers": (
            project.loader.persistent_workers and project.loader.num_workers > 0
        ),
        "pin_memory": project.loader.pin_memory,
        "drop_last": project.loader.drop_last,
        "sampler": None,
    }
    if project.loader.num_workers > 0:
        worker_kwargs["prefetch_factor"] = project.loader.prefetch_factor
    resolved.dataloader_train = L(CosmosResumeAwarePackingDataLoader)(
        audio_sample_rate=48_000,
        dataset_name="genet_pairs",
        max_samples_per_batch=1,
        max_sequence_length=None,
        patch_spatial=model_config.diffusion_expert_config.patch_spatial,
        sound_latent_fps=0,
        tokenizer_spatial_compression_factor=(
            model_config.tokenizer.spatial_compression_factor
        ),
        tokenizer_temporal_compression_factor=(
            model_config.tokenizer.temporal_compression_factor
        ),
        dataloader=L(CosmosInfiniteRankPartitionedDataLoader)(
            datasets={
                "genet_pairs": {
                    "ratio": 1,
                    "dataset": L(CosmosProcessedPairDataset)(
                        manifest=str(manifest),
                        embodiment_map=embodiment_map,
                        fps=project.data.fps,
                        reference_mode=project.data.reference_mode,
                        reference_seed=project.data.reference_seed,
                        require_bidirectional_pairs=(
                            project.data.require_bidirectional_pairs
                        ),
                        expected_embodiments=project.data.expected_embodiments,
                        action_alignment="frame",
                        tokenizer_config=model_config.vlm_config.tokenizer,
                        max_caption_tokens=2048,
                        use_system_prompt=model_config.vlm_config.use_system_prompt,
                        shard_seed=project.train.seed,
                        max_action_dim=project.data.action_dim,
                    ),
                }
            },
            seed=project.train.seed,
            **worker_kwargs,
        ),
    )
    resolved.dataloader_val = None

    adapter_keys = ["cross_embodiment_adapter.source_control"]
    if project.model.reference.enabled and project.train.stage in {"reference", "joint"}:
        adapter_keys.append("cross_embodiment_adapter.reference_layers")
    base_keys: list[str] = []
    if project.train.stage == "joint":
        base_keys = [
            "moe_gen",
            "time_embedder",
            "vae2llm",
            "llm2vae",
            "action2llm",
            "llm2action",
            "k_norm_und_for_gen",
        ]
    resolved.optimizer.keys_to_select = adapter_keys + base_keys
    resolved.optimizer.lr = project.train.new_module_lr
    resolved.optimizer.weight_decay = project.train.weight_decay
    if base_keys:
        ratio = project.train.base_lr / project.train.new_module_lr
        resolved.optimizer.lr_multipliers = {key: ratio for key in base_keys}
    else:
        resolved.optimizer.lr_multipliers = {}
    resolved.scheduler.warm_up_steps = [project.train.warmup_steps]
    resolved.scheduler.cycle_lengths = [project.train.max_steps]
    resolved.scheduler.f_start = [0.0]
    resolved.scheduler.f_max = [1.0]
    resolved.scheduler.f_min = [0.0]

    resolved.trainer.seed = project.train.seed
    resolved.trainer.max_iter = project.train.max_steps
    resolved.trainer.grad_accum_iter = project.train.grad_accum_steps
    resolved.trainer.logging_iter = project.train.log_every
    resolved.trainer.run_validation = False
    resolved.trainer.run_validation_on_start = False
    if "grad_clip" in resolved.trainer.callbacks:
        resolved.trainer.callbacks.grad_clip.clip_norm = project.train.grad_clip
    _disable_online_sampling(resolved.trainer.callbacks)
    _install_eval_video_callback(
        resolved,
        project,
        embodiment_map=embodiment_map,
        model_config=model_config,
    )

    checkpoint_path, exact_resume = checkpoint_load or _checkpoint_load_path(project)
    if is_hf_snapshot(Path(checkpoint_path)):
        # DDP whole-model warm start: the upstream DCP checkpointer never sees
        # the safetensors snapshot. Every rank loads the complete model inside
        # CosmosCrossEmbodimentModel.load_pretrained_model_if_needed instead.
        assert not exact_resume
        resolved.checkpoint.load_path = ""
        resolved.model["hf_warm_start_path"] = checkpoint_path
    else:
        resolved.checkpoint.load_path = checkpoint_path
    resolved.checkpoint.load_training_state = exact_resume
    resolved.checkpoint.strict_resume = exact_resume
    resolved.checkpoint.save_iter = project.checkpoint.save_every
    resolved.checkpoint.broadcast_via_filesystem = False
    resolved.checkpoint.dcp_async_mode_enabled = False
    resolved.checkpoint.save_to_object_store.enabled = False
    resolved.checkpoint.load_from_object_store.enabled = False
    resolved.checkpoint.keys_to_skip_loading = [] if exact_resume else ["net_ema."]

    output_root = Path(project.checkpoint.output_dir).expanduser().resolve()
    os.environ["IMAGINAIRE_OUTPUT_ROOT"] = str(output_root)
    _apply_wandb_logging(resolved, project)
    resolved.upload_reproducible_setup = False
    return resolved


def run_cosmos_training(project: ProjectConfig, *, dry_run: bool = False) -> None:
    """Launch the pinned upstream trainer with the GenET Edge experiment."""

    try:
        from cosmos_framework.scripts.train import launch
        from cosmos_framework.utils import distributed as cosmos_distributed
    except ImportError as exc:  # pragma: no cover - requires NVIDIA container
        raise RuntimeError(
            "Cosmos backend is not installed; see scripts/bootstrap_cosmos.sh"
        ) from exc

    # Use upstream initialization so GPU affinity and NCCL timeout handling are
    # identical to official recipes. Its launch() sees the initialized group and
    # safely treats its own init call as a no-op.
    cosmos_distributed.init()
    environment_signature = assert_runtime_environment_consistent()
    manifest = Path(project.data.manifest).expanduser().resolve()
    assert_same_across_ranks(
        "distributed_config_fingerprint", project.distributed_fingerprint()
    )
    manifest_hash: str | None = None
    domains: dict[str, int] | None = None
    manifest_error: str | None = None
    try:
        if not manifest.is_file():
            raise FileNotFoundError(f"processed manifest does not exist: {manifest}")
        manifest_hash = sha256_file(manifest)
        domains = _manifest_domains(manifest, project.model.num_embodiments)
    except Exception as exc:
        manifest_error = f"{type(exc).__name__}: {exc}"
    raise_if_any_rank_failed("local manifest preflight", manifest_error)
    assert manifest_hash is not None and domains is not None
    assert_same_across_ranks("manifest_sha256", manifest_hash)
    assert_same_across_ranks("embodiment_map", domains)
    assert_artifact_bound_to_receipt(
        os.environ.get("GENET_DATA_ARTIFACT", "processed_data"),
        manifest,
        allow_descendant=True,
    )

    wan_vae_path: Path | None = None
    checkpoint_load: tuple[str, bool] | None = None
    artifact_error: str | None = None
    try:
        wan_vae_path = _wan_vae_path()
        checkpoint_load = _checkpoint_load_path(project)
    except Exception as exc:
        artifact_error = f"{type(exc).__name__}: {exc}"
    raise_if_any_rank_failed("Cosmos model artifact preflight", artifact_error)
    assert wan_vae_path is not None and checkpoint_load is not None
    assert_artifact_bound_to_receipt(
        os.environ.get("GENET_WAN_VAE_ARTIFACT", "wan_vae"),
        wan_vae_path,
    )
    assert_artifact_bound_to_receipt(
        os.environ.get("GENET_CHECKPOINT_ARTIFACT", "training_checkpoint"),
        checkpoint_load[0],
    )
    hf_cache = os.environ.get("HF_HUB_CACHE") or os.environ.get("HF_HOME")
    hf_error = None
    strict_environment = os.environ.get("GENET_STRICT_ENV", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if strict_environment:
        if not hf_cache:
            hf_error = "strict Cosmos training requires HF_HOME or HF_HUB_CACHE"
        else:
            try:
                _validate_hf_cache_revision(
                    hf_cache,
                    os.environ["GENET_HF_SNAPSHOT_REVISION"],
                )
            except Exception as exc:
                hf_error = f"{type(exc).__name__}: {exc}"
    raise_if_any_rank_failed("Hugging Face cache preflight", hf_error)
    if hf_cache:
        assert_artifact_bound_to_receipt(
            os.environ.get("GENET_HF_ARTIFACT", "hf_cache"),
            hf_cache,
        )
    _validate_local_cosmos_copy(project, manifest=manifest)
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    probe_id: str | None = None
    direction_summary: dict[str, Any] | None = None
    probe_error: str | None = None
    try:
        probe_id, direction_summary = _probe_cosmos_data_contract(
            project,
            manifest=manifest,
            embodiment_map=domains,
            rank=rank,
            world_size=world_size,
        )
    except Exception as exc:
        probe_error = f"{type(exc).__name__}: {exc}"
    raise_if_any_rank_failed("rank-local Cosmos sample probe", probe_error)
    assert probe_id is not None and direction_summary is not None
    assert_same_across_ranks("pair_direction_summary", direction_summary)
    if rank == 0:
        print(
            json.dumps(
                {
                    "event": "pair_direction_summary",
                    **direction_summary,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    config = None
    config_error: str | None = None
    try:
        config = _build_cosmos_config(
            project,
            manifest=manifest,
            embodiment_map=domains,
            checkpoint_load=checkpoint_load,
        )
    except Exception as exc:
        config_error = f"{type(exc).__name__}: {exc}"
    raise_if_any_rank_failed("Cosmos experiment construction", config_error)
    assert config is not None
    if dry_run:
        config.validate()
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                json.dumps(
                    {
                        "event": "cosmos_dry_run",
                        "manifest": str(manifest),
                        "probed_sample": probe_id,
                        "embodiment_map": domains,
                        "pair_directions": direction_summary,
                        "output": str(Path(project.checkpoint.output_dir).resolve()),
                        "runtime_environment": environment_signature,
                        "world_size": world_size,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                flush=True,
            )
        return

    args = argparse.Namespace(
        attach_vscode_debugger=False,
        config="genet:cosmos3_edge",
        deterministic=False,
        dryrun=False,
    )
    launch(config, args)
