"""Typed project configuration with strict YAML overrides.

The config intentionally stays independent from Hydra.  The Cosmos adapter
translates the relevant fields into the upstream framework at the boundary.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeVar, get_args, get_origin, get_type_hints

import yaml


@dataclass
class DataConfig:
    manifest: str = "data/processed/train/manifest.jsonl"
    fps: float = 16.0
    num_frames: int = 81
    reference_num_frames: int = 81
    height: int = 192
    width: int = 320
    temporal_compression_factor: int = 4
    action_dim: int = 64
    reference_mode: Literal["stored", "deterministic"] = "stored"
    reference_seed: int = 1234
    expected_embodiments: list[str] = field(default_factory=list)
    require_bidirectional_pairs: bool = False


@dataclass
class LoaderConfig:
    micro_batch_size: int = 1
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    drop_last: bool = True


@dataclass
class SourceControlConfig:
    enabled: bool = True
    zero_init: bool = True
    vision_scale: float = 1.0
    action_scale: float = 1.0
    domain_embedding_dim: int = 32


@dataclass
class ReferenceConfig:
    enabled: bool = True
    projection_mode: Literal["shared", "dual", "dual_tied"] = "shared"
    routing: Literal["vision_action", "ar_dm"] = "vision_action"
    num_heads: int = 8
    dropout: float = 0.0
    gate_init: float = 0.0
    inject_every_n_layers: int = 1
    use_video: bool = True
    use_action: bool = True


@dataclass
class ParallelismConfig:
    data_parallel_shard_degree: int = 8
    data_parallel_replicate_degree: int = 4
    context_parallel_shard_degree: int = 1
    cfg_parallel_shard_degree: int = 1


@dataclass
class ModelConfig:
    backend: Literal["toy", "cosmos3_edge"] = "toy"
    dtype: Literal["float32", "float16", "bfloat16"] = "bfloat16"
    hidden_size: int = 512
    num_layers: int = 8
    num_heads: int = 8
    latent_channels: int = 16
    patch_size: int = 2
    num_embodiments: int = 32
    freeze_vae: bool = True
    activation_checkpointing: Literal["none", "selective", "full"] = "full"
    compile: bool = False
    source_control: SourceControlConfig = field(default_factory=SourceControlConfig)
    reference: ReferenceConfig = field(default_factory=ReferenceConfig)
    parallelism: ParallelismConfig = field(default_factory=ParallelismConfig)


@dataclass
class LossConfig:
    video: float = 1.0
    action: float = 1.0


@dataclass
class TrainConfig:
    stage: Literal["control", "reference", "joint"] = "control"
    seed: int = 42
    max_steps: int = 50_000
    grad_accum_steps: int = 1
    base_lr: float = 1.0e-5
    new_module_lr: float = 1.0e-4
    weight_decay: float = 0.0
    warmup_steps: int = 2_000
    grad_clip: float = 1.0
    log_every: int = 10
    condition_dropout: float = 0.1
    loss: LossConfig = field(default_factory=LossConfig)


@dataclass
class CheckpointConfig:
    output_dir: str = "outputs/run"
    resume: str | None = None
    warm_start: str | None = None
    save_every: int = 1_000
    require_committed: bool = True
    copy_shared_reference_to_dual: bool = False


@dataclass
class ProjectConfig:
    data: DataConfig = field(default_factory=DataConfig)
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    def validate(self, world_size: int | None = None) -> None:
        if self.data.reference_mode not in {"stored", "deterministic"}:
            raise ValueError("data.reference_mode must be 'stored' or 'deterministic'")
        expected_embodiments = self.data.expected_embodiments
        if not isinstance(expected_embodiments, list):
            raise ValueError("data.expected_embodiments must be a list")
        if any(
            not isinstance(embodiment, str) or not embodiment.strip()
            for embodiment in expected_embodiments
        ):
            raise ValueError(
                "data.expected_embodiments must contain only non-empty strings"
            )
        if len(set(expected_embodiments)) != len(expected_embodiments):
            raise ValueError("data.expected_embodiments must not contain duplicates")
        if not isinstance(self.data.require_bidirectional_pairs, bool):
            raise ValueError("data.require_bidirectional_pairs must be boolean")
        if self.data.require_bidirectional_pairs:
            if not expected_embodiments:
                raise ValueError(
                    "data.require_bidirectional_pairs requires non-empty "
                    "data.expected_embodiments"
                )
            if self.data.reference_mode != "stored":
                raise ValueError(
                    "data.require_bidirectional_pairs requires "
                    "data.reference_mode='stored'"
                )
        if self.data.num_frames < 1:
            raise ValueError("data.num_frames must be positive")
        if self.data.height < 1 or self.data.width < 1:
            raise ValueError("data.height and data.width must be positive")
        if self.data.fps <= 0:
            raise ValueError("data.fps must be positive")
        if self.data.action_dim < 1:
            raise ValueError("data.action_dim must be positive")
        factor = self.data.temporal_compression_factor
        if factor < 1 or (self.data.num_frames - 1) % factor:
            raise ValueError(
                "Wan causal VAE requires num_frames == 1 + N * temporal_compression_factor; "
                f"got num_frames={self.data.num_frames}, factor={factor}"
            )
        if self.data.reference_num_frames < 1 or (self.data.reference_num_frames - 1) % factor:
            raise ValueError(
                "reference_num_frames must also equal 1 + N * temporal_compression_factor; "
                f"got {self.data.reference_num_frames}"
            )
        if self.data.reference_num_frames != self.data.num_frames:
            raise ValueError(
                "processed-pair/v1 uses one fixed length for source, target, and reference; "
                "data.reference_num_frames must equal data.num_frames"
            )
        if self.model.backend not in {"toy", "cosmos3_edge"}:
            raise ValueError("model.backend must be 'toy' or 'cosmos3_edge'")
        if self.model.dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError("model.dtype must be float32, float16, or bfloat16")
        if self.model.activation_checkpointing not in {"none", "selective", "full"}:
            raise ValueError(
                "model.activation_checkpointing must be none, selective, or full"
            )
        if self.model.hidden_size < 1 or self.model.num_heads < 1:
            raise ValueError("model.hidden_size and model.num_heads must be positive")
        if self.model.num_layers < 1 or self.model.latent_channels < 1:
            raise ValueError("model.num_layers and model.latent_channels must be positive")
        if self.model.patch_size < 1 or self.model.num_embodiments < 1:
            raise ValueError("model.patch_size and model.num_embodiments must be positive")
        if self.model.source_control.domain_embedding_dim < 1:
            raise ValueError("model.source_control.domain_embedding_dim must be positive")
        reference = self.model.reference
        if reference.projection_mode not in {"shared", "dual", "dual_tied"}:
            raise ValueError(
                "model.reference.projection_mode must be shared, dual, or dual_tied"
            )
        if reference.routing not in {"vision_action", "ar_dm"}:
            raise ValueError("model.reference.routing must be vision_action or ar_dm")
        if reference.num_heads < 1:
            raise ValueError("model.reference.num_heads must be positive")
        if reference.inject_every_n_layers < 1:
            raise ValueError("model.reference.inject_every_n_layers must be positive")
        if not 0.0 <= reference.dropout < 1.0:
            raise ValueError("model.reference.dropout must be in [0, 1)")
        if reference.enabled and not (reference.use_video or reference.use_action):
            raise ValueError(
                "an enabled reference branch must use video, action, or both"
            )
        if self.model.hidden_size % self.model.num_heads:
            raise ValueError("model.hidden_size must be divisible by model.num_heads")
        if self.model.hidden_size % reference.num_heads:
            raise ValueError("model.hidden_size must be divisible by model.reference.num_heads")
        degrees = self.model.parallelism
        if min(
            degrees.data_parallel_shard_degree,
            degrees.data_parallel_replicate_degree,
            degrees.context_parallel_shard_degree,
            degrees.cfg_parallel_shard_degree,
        ) < 1:
            raise ValueError("all model.parallelism degrees must be positive")
        parallel_world = (
            degrees.data_parallel_shard_degree
            * degrees.data_parallel_replicate_degree
            * degrees.context_parallel_shard_degree
            * degrees.cfg_parallel_shard_degree
        )
        if world_size is not None and self.model.backend == "cosmos3_edge" and parallel_world != world_size:
            raise ValueError(
                "Cosmos parallel degrees must multiply to WORLD_SIZE: "
                f"{parallel_world} != {world_size}"
            )
        if self.model.backend == "cosmos3_edge" and self.data.action_dim != 64:
            raise ValueError(
                "Cosmos3-Edge checkpoints use a 64-D padded action boundary; "
                "set data.action_dim=64 and carry real dimensions in action_mask"
            )
        if self.model.backend == "cosmos3_edge" and self.model.num_embodiments != 32:
            raise ValueError(
                "Cosmos3-Edge checkpoints use 32 domain slots; set model.num_embodiments=32"
            )
        if self.model.backend == "cosmos3_edge" and self.loader.micro_batch_size != 1:
            raise ValueError(
                "the current Cosmos adapter supports loader.micro_batch_size=1 only"
            )
        if (
            self.model.backend == "cosmos3_edge"
            and degrees.context_parallel_shard_degree != 1
        ):
            raise ValueError(
                "the current Cosmos adapter supports context_parallel_shard_degree=1 only"
            )
        if not self.model.freeze_vae:
            raise ValueError(
                "GenET currently requires model.freeze_vae=true; trainable Wan VAE "
                "parameters are not part of the optimizer/checkpoint contract"
            )
        if not 0.0 <= self.train.condition_dropout < 1.0:
            raise ValueError("train.condition_dropout must be in [0, 1)")
        if self.train.max_steps < 1:
            raise ValueError("train.max_steps must be positive")
        if self.train.stage not in {"control", "reference", "joint"}:
            raise ValueError("train.stage must be control, reference, or joint")
        if self.train.grad_accum_steps < 1:
            raise ValueError("train.grad_accum_steps must be positive")
        if self.train.base_lr <= 0 or self.train.new_module_lr <= 0:
            raise ValueError("train.base_lr and train.new_module_lr must be positive")
        if self.train.weight_decay < 0:
            raise ValueError("train.weight_decay must be non-negative")
        if self.train.warmup_steps < 0:
            raise ValueError("train.warmup_steps must be non-negative")
        if self.train.grad_clip <= 0:
            raise ValueError("train.grad_clip must be positive")
        if self.train.log_every < 1:
            raise ValueError("train.log_every must be positive")
        if self.train.loss.video < 0 or self.train.loss.action < 0:
            raise ValueError("train.loss weights must be non-negative")
        if self.train.loss.video == 0 and self.train.loss.action == 0:
            raise ValueError("at least one train.loss weight must be positive")
        if self.loader.micro_batch_size < 1:
            raise ValueError("loader.micro_batch_size must be positive")
        if self.loader.num_workers < 0:
            raise ValueError("loader.num_workers must be non-negative")
        if self.loader.prefetch_factor < 1:
            raise ValueError("loader.prefetch_factor must be positive")
        if self.checkpoint.save_every < 1:
            raise ValueError("checkpoint.save_every must be positive")
        if self.checkpoint.resume and self.checkpoint.warm_start:
            raise ValueError("checkpoint.resume and checkpoint.warm_start are mutually exclusive")
        if (
            self.checkpoint.copy_shared_reference_to_dual
            and self.model.reference.projection_mode != "dual"
        ):
            raise ValueError(
                "checkpoint.copy_shared_reference_to_dual requires "
                "model.reference.projection_mode=dual"
            )

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def fingerprint(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def distributed_fingerprint(self) -> str:
        """Hash semantic settings while excluding node-local filesystem paths."""

        values = self.as_dict()
        values["data"].pop("manifest", None)
        values["checkpoint"].pop("output_dir", None)
        # Paths may differ per node, but resume vs warm-start is semantic and
        # must still agree across every rank.
        values["checkpoint"]["resume"] = self.checkpoint.resume is not None
        values["checkpoint"]["warm_start"] = self.checkpoint.warm_start is not None
        payload = json.dumps(values, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


T = TypeVar("T")


def _construct(cls: type[T], values: dict[str, Any], path: str = "") -> T:
    if not dataclasses.is_dataclass(cls):
        return values  # type: ignore[return-value]
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(values) - set(fields)
    if unknown:
        prefix = f"{path}." if path else ""
        raise ValueError(f"Unknown config key(s): {', '.join(prefix + key for key in sorted(unknown))}")
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for name, value in values.items():
        annotation = hints[name]
        nested_type = annotation
        origin = get_origin(annotation)
        if origin is not None and origin is not Literal:
            candidates = [arg for arg in get_args(annotation) if arg is not type(None)]
            if len(candidates) == 1:
                nested_type = candidates[0]
        if isinstance(value, dict) and isinstance(nested_type, type) and dataclasses.is_dataclass(nested_type):
            kwargs[name] = _construct(nested_type, value, f"{path}.{name}" if path else name)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml_mapping(path: Path, seen: set[Path]) -> dict[str, Any]:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"Config inheritance cycle at {path}")
    seen.add(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Top-level YAML value must be a mapping: {path}")
    base_path = raw.pop("_base_", None)
    if base_path is not None:
        if not isinstance(base_path, str):
            raise ValueError("_base_ must be one relative YAML path")
        base = _load_yaml_mapping(path.parent / base_path, seen)
        raw = _deep_merge(base, raw)
    seen.remove(path)
    return raw


def load_config(path: str | Path) -> ProjectConfig:
    """Load and strictly validate a YAML project config."""

    raw = _load_yaml_mapping(Path(path), set())
    config = _construct(ProjectConfig, raw)
    config.validate()
    return config
