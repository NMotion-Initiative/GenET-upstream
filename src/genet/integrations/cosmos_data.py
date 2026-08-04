"""Pure-data bridge from GenET processed pairs to the pinned Cosmos batch API.

The production dataset intentionally does not import the Cosmos model stack.
When ``cosmos-framework`` is importable it uses the upstream ``SequencePlan``
class; otherwise a field-compatible dataclass keeps preprocessing, sharding,
and unit tests usable on lightweight machines.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import Dataset

from genet.data.dataset import ProcessedPairDataset

COSMOS_EDGE_MAX_ACTION_DIM = 64
COSMOS_EDGE_NUM_EMBODIMENT_DOMAINS = 32
COSMOS_OFFICIAL_BATCH_KEYS = frozenset(
    {
        "video",
        "text_token_ids",
        "action",
        "domain_id",
        "sequence_plan",
        "ai_caption",
        "image_size",
        "conditioning_fps",
        "raw_action_dim",
        "action_processing_record",
    }
)

_SEQUENCE_PLAN_IMPORT_ERROR: ImportError | None = None
try:  # Keep this integration importable without the full Cosmos environment.
    from cosmos_framework.data.generator.sequence_packing import (
        SequencePlan as _UpstreamSequencePlan,
    )
except ImportError as exc:  # pragma: no cover - depends on the developer environment
    _SEQUENCE_PLAN_IMPORT_ERROR = exc
    _UpstreamSequencePlan = None

try:  # Required only when generated actions are converted back to raw width.
    from cosmos_framework.data.generator.action.action_processing import (
        ActionProcessingRecord as _UpstreamActionProcessingRecord,
    )
except ImportError:  # pragma: no cover - depends on the developer environment
    _UpstreamActionProcessingRecord = None

try:  # Production text tokenization follows the official Edge SFT dataset.
    from cosmos_framework.data.generator.sequence_packing.modalities import (
        add_special_tokens as _add_special_tokens,
    )
    from cosmos_framework.model.generator.reasoner.qwen3_vl.utils import (
        tokenize_caption as _tokenize_caption,
    )
    from cosmos_framework.utils.lazy_config import instantiate as _lazy_instantiate
except ImportError:  # pragma: no cover - depends on the developer environment
    _add_special_tokens = None
    _tokenize_caption = None
    _lazy_instantiate = None


@dataclass
class CosmosSequencePlan:
    """Field-compatible fallback for the pinned upstream ``SequencePlan``."""

    has_text: bool
    has_vision: bool = False
    condition_frame_indexes_vision: list[int] = field(default_factory=list)
    share_vision_temporal_positions: bool = False
    has_action: bool = False
    condition_frame_indexes_action: list[int] = field(default_factory=list)
    action_start_frame_offset: int = 1
    has_sound: bool = False
    condition_frame_indexes_sound: list[int] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "has_text": self.has_text,
            "has_vision": self.has_vision,
            "has_action": self.has_action,
            "has_sound": self.has_sound,
            "condition_frame_indexes_vision": self.condition_frame_indexes_vision,
            "condition_frame_indexes_action": self.condition_frame_indexes_action,
            "condition_frame_indexes_sound": self.condition_frame_indexes_sound,
            "share_vision_temporal_positions": self.share_vision_temporal_positions,
        }


@dataclass(frozen=True)
class CosmosActionProcessingRecord:
    """Fallback matching the upstream no-normalization action record."""

    raw_action_dim: int
    action_normalizer: None = None


def sequence_plan_backend() -> str:
    """Return the active plan implementation without requiring Cosmos weights."""

    return "cosmos-framework" if _UpstreamSequencePlan is not None else "compatible-fallback"


def sequence_plan_import_error() -> ImportError | None:
    """Expose why the optional upstream plan class was unavailable, if applicable."""

    return _SEQUENCE_PLAN_IMPORT_ERROR


def make_cosmos_sequence_plan(
    *,
    action_start_frame_offset: int,
    condition_frame_indexes_vision: list[int] | tuple[int, ...] = (),
    condition_frame_indexes_action: list[int] | tuple[int, ...] = (),
) -> Any:
    """Build the pinned Cosmos plan for training or rolling-prefix inference."""

    plan_type = _UpstreamSequencePlan or CosmosSequencePlan
    return plan_type(
        has_text=True,
        has_vision=True,
        condition_frame_indexes_vision=list(condition_frame_indexes_vision),
        share_vision_temporal_positions=False,
        has_action=True,
        condition_frame_indexes_action=list(condition_frame_indexes_action),
        action_start_frame_offset=action_start_frame_offset,
        has_sound=False,
        condition_frame_indexes_sound=[],
    )


def _make_action_processing_record(*, raw_action_dim: int) -> Any:
    record_type = _UpstreamActionProcessingRecord or CosmosActionProcessingRecord
    return record_type(raw_action_dim=raw_action_dim, action_normalizer=None)


def _load_embodiment_map(value: Mapping[str, int] | str | Path) -> dict[str, int]:
    if isinstance(value, (str, Path)):
        path = Path(value).expanduser().resolve()
        try:
            with path.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot load embodiment_map JSON from {path}: {exc}") from exc
    if not isinstance(value, Mapping) or not value:
        raise ValueError("embodiment_map must be a non-empty mapping or JSON path")
    result: dict[str, int] = {}
    for embodiment, domain_id in value.items():
        if not isinstance(embodiment, str) or not embodiment:
            raise ValueError("embodiment_map keys must be non-empty strings")
        if isinstance(domain_id, bool) or not isinstance(domain_id, int):
            raise ValueError(f"domain id for {embodiment!r} must be an integer")
        if not 0 <= domain_id < COSMOS_EDGE_NUM_EMBODIMENT_DOMAINS:
            raise ValueError(
                f"domain id for {embodiment!r} must be in [0, "
                f"{COSMOS_EDGE_NUM_EMBODIMENT_DOMAINS}), got {domain_id}"
            )
        result[embodiment] = domain_id
    return result


_FRAME_ALIGNMENT_NAMES = {
    "auto",
    "frame",
    "frame_aligned",
    "per_frame",
    "same_length",
    "state",
    "state_aligned",
}
_TRANSITION_ALIGNMENT_NAMES = {
    "between_frames",
    "delta",
    "t_minus_1",
    "transition",
    "transition_aligned",
}


def _normalize_action_alignment(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    if normalized in _FRAME_ALIGNMENT_NAMES:
        return "frame"
    if normalized in _TRANSITION_ALIGNMENT_NAMES:
        return "transition"
    raise ValueError(
        "action_alignment must select per-frame/state actions ('frame') or "
        "between-frame actions ('transition')"
    )


def _require_tensor(stream: Mapping[str, Any], key: str, role: str) -> torch.Tensor:
    value = stream.get(key)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{role}.{key} must be a torch.Tensor")
    return value


def _extract_streams(sample: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    if all(key in sample for key in ("source", "target", "reference_target")):
        source = sample["source"]
        target = sample["target"]
        reference = sample["reference_target"]
        if not all(isinstance(item, Mapping) for item in (source, target, reference)):
            raise TypeError("source, target, and reference_target must be stream mappings")
        return source, target, reference

    def flat_stream(role: str, video_keys: tuple[str, ...], action_keys: tuple[str, ...]) -> dict[str, Any]:
        video = next((sample[key] for key in video_keys if key in sample), None)
        action = next((sample[key] for key in action_keys if key in sample), None)
        return {
            "video": video,
            "actions": action,
            "action_mask": sample.get(f"{role}_action_mask", sample.get("action_mask") if role == "target" else None),
            "frame_mask": sample.get(f"{role}_frame_mask", sample.get("frame_mask") if role == "target" else None),
        }

    return (
        flat_stream(
            "source",
            ("source_video", "control_video"),
            ("source_action", "source_actions", "control_action", "control_actions"),
        ),
        flat_stream("target", ("video",), ("action", "actions")),
        flat_stream("reference", ("reference_video",), ("reference_action", "reference_actions")),
    )


def _validate_video(video: torch.Tensor, *, role: str) -> torch.Tensor:
    if video.ndim != 4 or video.shape[0] not in (1, 3, 4):
        raise ValueError(f"{role}.video must have shape [C,T,H,W], got {tuple(video.shape)}")
    if video.dtype != torch.uint8:
        raise TypeError(
            f"{role}.video must be uint8 for Cosmos worker-side normalization; "
            f"got {video.dtype}. Load ProcessedPairDataset with normalize_video='none'."
        )
    return video.contiguous()


def _raw_action_dim(action: torch.Tensor, mask: torch.Tensor | None, *, role: str) -> int:
    if mask is None:
        return int(action.shape[-1])
    if mask.shape != action.shape or mask.dtype != torch.bool:
        raise ValueError(
            f"{role}.action_mask must be bool with shape {tuple(action.shape)}, "
            f"got {tuple(mask.shape)} / {mask.dtype}"
        )
    active = mask.any(dim=0)
    raw_dim = int(active.sum().item())
    if raw_dim <= 0:
        raise ValueError(f"{role} has no valid action channels")
    expected = torch.arange(action.shape[-1], device=active.device) < raw_dim
    if not torch.equal(active, expected):
        raise ValueError(
            f"{role}.action_mask must describe a contiguous prefix of real channels; "
            "Cosmos raw_action_dim cannot represent holes"
        )
    if not bool(mask[:, :raw_dim].all()):
        raise ValueError(
            f"{role} contains temporally padded action steps; preprocess training data "
            "with short_policy='drop' or add an explicit loss-mask integration"
        )
    return raw_dim


def _align_and_pad_action(
    stream: Mapping[str, Any],
    *,
    role: str,
    video_frames: int,
    alignment: str,
    max_action_dim: int,
) -> tuple[torch.Tensor, int]:
    action = _require_tensor(stream, "actions", role)
    if action.ndim != 2:
        raise ValueError(f"{role}.actions must have shape [T,D], got {tuple(action.shape)}")
    action = action.to(dtype=torch.float32).contiguous()
    mask_value = stream.get("action_mask")
    mask = mask_value if isinstance(mask_value, torch.Tensor) else None

    if alignment == "frame":
        if action.shape[0] != video_frames:
            raise ValueError(
                f"{role} frame-aligned actions require T_action == T_video, got "
                f"{action.shape[0]} != {video_frames}"
            )
    else:
        if action.shape[0] == video_frames:
            # Convention: action[t] describes the transition frame[t-1] -> frame[t];
            # action[0] has no preceding frame and is omitted in transition mode.
            action = action[1:]
            if mask is not None:
                mask = mask[1:]
        elif action.shape[0] != video_frames - 1:
            raise ValueError(
                f"{role} transition actions require T_action in {{T_video, T_video-1}}, "
                f"got {action.shape[0]} for {video_frames} frames"
            )

    raw_dim = _raw_action_dim(action, mask, role=role)
    if action.shape[-1] > max_action_dim:
        raise ValueError(
            f"{role} action dim {action.shape[-1]} exceeds Cosmos max_action_dim={max_action_dim}"
        )
    if action.shape[-1] < max_action_dim:
        padded = action.new_zeros((action.shape[0], max_action_dim))
        padded[:, : action.shape[-1]] = action
        action = padded
    return action, raw_dim


def _validate_frame_mask(stream: Mapping[str, Any], *, role: str, frames: int) -> None:
    value = stream.get("frame_mask")
    if value is None:
        return
    if not isinstance(value, torch.Tensor) or value.dtype != torch.bool or value.shape != (frames,):
        shape = tuple(value.shape) if isinstance(value, torch.Tensor) else None
        raise ValueError(f"{role}.frame_mask must be bool with shape {(frames,)}, got {shape}")
    if not bool(value.all()):
        raise ValueError(
            f"{role} contains padded video frames; preprocess training data with "
            "short_policy='drop' or add an explicit loss-mask integration"
        )


def _domain_id(metadata: Mapping[str, Any], role: str, embodiment_map: Mapping[str, int]) -> torch.Tensor:
    metadata_key = "target_gt" if role == "target" else ("reference_target" if role == "reference" else role)
    role_metadata = metadata.get(metadata_key)
    if not isinstance(role_metadata, Mapping):
        raise KeyError(f"sample metadata is missing {metadata_key!r}")
    embodiment = role_metadata.get("embodiment")
    if not isinstance(embodiment, str) or not embodiment:
        raise KeyError(f"sample metadata {metadata_key!r} is missing embodiment")
    if embodiment not in embodiment_map:
        raise KeyError(
            f"embodiment_map has no domain id for {role} embodiment {embodiment!r}; "
            f"known embodiments: {sorted(embodiment_map)}"
        )
    return torch.tensor(embodiment_map[embodiment], dtype=torch.long)


def _caption(sample: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    keys = ("ai_caption", "caption", "task", "task_description", "instruction", "language_instruction", "task_id")
    containers: list[Mapping[str, Any]] = [sample]
    for name in ("manifest", "source", "target_gt"):
        value = metadata.get(name)
        if isinstance(value, Mapping):
            containers.append(value)
            nested = value.get("metadata")
            if isinstance(nested, Mapping):
                containers.append(nested)
    for container in containers:
        for key in keys:
            value = container.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def convert_processed_sample_to_cosmos(
    sample: Mapping[str, Any],
    *,
    embodiment_map: Mapping[str, int] | str | Path,
    fps: float,
    action_alignment: str = "frame",
    max_action_dim: int = COSMOS_EDGE_MAX_ACTION_DIM,
) -> dict[str, Any]:
    """Convert one GenET processed pair to the official Cosmos sample keys.

    The returned object is a *single sample*.  The pinned
    ``RankPartitionedDataLoader``/``PackingDataLoader`` owns collation and turns
    these values into the list-based batch layout consumed by ``OmniMoTModel``.
    """

    if not isinstance(fps, (int, float)) or isinstance(fps, bool) or fps <= 0:
        raise ValueError("fps must be positive")
    if not isinstance(max_action_dim, int) or isinstance(max_action_dim, bool) or max_action_dim <= 0:
        raise ValueError("max_action_dim must be a positive integer")
    alignment = _normalize_action_alignment(action_alignment)
    domains = _load_embodiment_map(embodiment_map)
    source_stream, target_stream, reference_stream = _extract_streams(sample)

    source_video = _validate_video(_require_tensor(source_stream, "video", "source"), role="source")
    target_video = _validate_video(_require_tensor(target_stream, "video", "target"), role="target")
    reference_video = _validate_video(
        _require_tensor(reference_stream, "video", "reference"), role="reference"
    )
    if source_video.shape != target_video.shape:
        raise ValueError(
            "source and target videos must be frame/spatial aligned before ControlNet encoding: "
            f"{tuple(source_video.shape)} != {tuple(target_video.shape)}"
        )
    for role, stream, video in (
        ("source", source_stream, source_video),
        ("target", target_stream, target_video),
        ("reference", reference_stream, reference_video),
    ):
        _validate_frame_mask(stream, role=role, frames=int(video.shape[1]))

    source_action, source_raw_dim = _align_and_pad_action(
        source_stream,
        role="source",
        video_frames=int(source_video.shape[1]),
        alignment=alignment,
        max_action_dim=max_action_dim,
    )
    target_action, target_raw_dim = _align_and_pad_action(
        target_stream,
        role="target",
        video_frames=int(target_video.shape[1]),
        alignment=alignment,
        max_action_dim=max_action_dim,
    )
    reference_action, reference_raw_dim = _align_and_pad_action(
        reference_stream,
        role="reference",
        video_frames=int(reference_video.shape[1]),
        alignment=alignment,
        max_action_dim=max_action_dim,
    )
    if source_action.shape != target_action.shape:
        raise ValueError(
            "source and target actions must be aligned for residual injection: "
            f"{tuple(source_action.shape)} != {tuple(target_action.shape)}"
        )

    metadata = sample.get("metadata")
    if not isinstance(metadata, Mapping):
        raise KeyError("processed sample is missing metadata required for embodiment domains")
    target_domain = _domain_id(metadata, "target", domains)
    source_domain = _domain_id(metadata, "source", domains)
    reference_domain = _domain_id(metadata, "reference", domains)

    def image_size(video: torch.Tensor) -> torch.Tensor:
        height, width = video.shape[-2:]
        return torch.tensor([height, width, height, width], dtype=torch.float32)

    fps_tensor = torch.tensor(float(fps), dtype=torch.float32)
    output: dict[str, Any] = {
        "video": target_video,
        "action": target_action,
        "domain_id": target_domain,
        "sequence_plan": make_cosmos_sequence_plan(
            action_start_frame_offset=0 if alignment == "frame" else 1
        ),
        "ai_caption": _caption(sample, metadata),
        "image_size": image_size(target_video),
        "conditioning_fps": fps_tensor,
        "raw_action_dim": torch.tensor(target_raw_dim, dtype=torch.long),
        # Lets upstream inference unpad generated actions. v1 performs no
        # semantic normalization, so postprocessing only slices raw_action_dim.
        "action_processing_record": _make_action_processing_record(
            raw_action_dim=target_raw_dim
        ),
        # Canonical auxiliary names consumed by CosmosCrossEmbodimentModel.
        "source_video": source_video,
        "source_action": source_action,
        "source_domain_id": source_domain,
        "source_image_size": image_size(source_video),
        "source_conditioning_fps": fps_tensor.clone(),
        "source_raw_action_dim": torch.tensor(source_raw_dim, dtype=torch.long),
        "reference_video": reference_video,
        "reference_action": reference_action,
        "reference_domain_id": reference_domain,
        "reference_image_size": image_size(reference_video),
        "reference_conditioning_fps": fps_tensor.clone(),
        "reference_raw_action_dim": torch.tensor(reference_raw_dim, dtype=torch.long),
    }
    if "sample_id" in sample:
        output["sample_id"] = str(sample["sample_id"])
    return output


class CosmosProcessedPairDataset(Dataset[dict[str, Any]]):
    """Cosmos-ready map dataset with loader-assigned dynamic rank sharding.

    Args mirror the production ``LazyCall`` surface.  ``shard_world_size`` and
    ``shard_rank`` deliberately remain mutable: the pinned
    ``RankPartitionedDataLoader`` assigns them *after* dataset construction.
    ``__len__`` and ``__getitem__`` consult their current values rather than
    caching a rank slice at initialization. Each epoch selects a rotating
    multiple-of-world-size window, so every rank has the same finite length
    without permanently excluding the manifest tail.
    """

    def __init__(
        self,
        manifest: str | Path,
        embodiment_map: Mapping[str, int] | str | Path,
        fps: float,
        reference_mode: str = "stored",
        reference_seed: int = 0,
        action_alignment: str = "frame",
        tokenizer_config: Any | None = None,
        max_caption_tokens: int = 2048,
        use_system_prompt: bool = False,
        shard_seed: int = 0,
        *,
        max_action_dim: int = COSMOS_EDGE_MAX_ACTION_DIM,
    ) -> None:
        self.embodiment_map = _load_embodiment_map(embodiment_map)
        self.fps = float(fps)
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        self.action_alignment = _normalize_action_alignment(action_alignment)
        self.max_action_dim = int(max_action_dim)
        if self.max_action_dim <= 0:
            raise ValueError("max_action_dim must be positive")
        self.max_caption_tokens = int(max_caption_tokens)
        if self.max_caption_tokens < 1:
            raise ValueError("max_caption_tokens must be positive")
        self.use_system_prompt = bool(use_system_prompt)
        self.shard_seed = int(shard_seed)
        self._vlm_tokenizer = None
        if tokenizer_config is not None:
            if (
                _lazy_instantiate is None
                or _add_special_tokens is None
                or _tokenize_caption is None
            ):
                raise RuntimeError(
                    "tokenizer_config requires the pinned Cosmos Framework environment"
                )
            processor = _lazy_instantiate(tokenizer_config)
            tokenizer = getattr(processor, "tokenizer", None)
            if tokenizer is None:
                raise TypeError("Cosmos tokenizer processor has no tokenizer attribute")
            self._vlm_tokenizer, _ = _add_special_tokens(tokenizer)
        self._dataset = ProcessedPairDataset(
            manifest,
            sample_format="generic",
            reference_mode=reference_mode,
            reference_seed=reference_seed,
            normalize_video="none",
            shard_by_rank=False,
        )
        self.global_length = len(self._dataset)
        self.shard_world_size = 1
        self.shard_rank = 0
        self.shard_id = 0

    def _shard(self) -> tuple[int, int]:
        try:
            world_size = int(self.shard_world_size)
            rank = int(self.shard_rank)
        except (TypeError, ValueError) as exc:
            raise ValueError("shard_world_size and shard_rank must be integers") from exc
        if world_size <= 0:
            raise ValueError("shard_world_size must be positive")
        if not 0 <= rank < world_size:
            raise ValueError("shard_rank must satisfy 0 <= shard_rank < shard_world_size")
        return world_size, rank

    def __len__(self) -> int:
        world_size, _rank = self._shard()
        return self.global_length // world_size

    def __getitem__(self, index: int | tuple[int, int]) -> dict[str, Any]:
        epoch = 0
        epoch_tagged = isinstance(index, tuple)
        if epoch_tagged:
            if len(index) != 2:
                raise IndexError(index)
            epoch, index = index
            if not isinstance(epoch, int) or epoch < 0 or not isinstance(index, int):
                raise IndexError((epoch, index))
        local_length = len(self)
        if index < 0:
            index += local_length
        if not 0 <= index < local_length:
            raise IndexError(index)
        world_size, rank = self._shard()
        global_position = rank + index * world_size
        # The sampler passes an epoch-tagged local index. A cyclic global
        # permutation preserves disjoint equal-length shards while rotating the
        # at-most WORLD_SIZE-1 omitted samples across epochs. Plain integer
        # indexing remains stable for preflight probes and interactive use.
        rotation = 0
        if epoch_tagged:
            rotation = (self.shard_seed + epoch) % self.global_length
        global_index = (global_position + rotation) % self.global_length
        sample = self._dataset[global_index]
        output = convert_processed_sample_to_cosmos(
            sample,
            embodiment_map=self.embodiment_map,
            fps=self.fps,
            action_alignment=self.action_alignment,
            max_action_dim=self.max_action_dim,
        )
        if self._vlm_tokenizer is not None:
            assert _tokenize_caption is not None
            token_ids = _tokenize_caption(
                output["ai_caption"],
                self._vlm_tokenizer,
                is_video=True,
                use_system_prompt=self.use_system_prompt,
            )
            output["text_token_ids"] = torch.tensor(
                token_ids[: self.max_caption_tokens],
                dtype=torch.long,
            )
        return output

    def set_epoch(self, epoch: int) -> None:
        self._dataset.set_epoch(epoch)


__all__ = [
    "COSMOS_EDGE_MAX_ACTION_DIM",
    "COSMOS_EDGE_NUM_EMBODIMENT_DOMAINS",
    "COSMOS_OFFICIAL_BATCH_KEYS",
    "CosmosProcessedPairDataset",
    "CosmosActionProcessingRecord",
    "CosmosSequencePlan",
    "convert_processed_sample_to_cosmos",
    "make_cosmos_sequence_plan",
    "sequence_plan_backend",
    "sequence_plan_import_error",
]
