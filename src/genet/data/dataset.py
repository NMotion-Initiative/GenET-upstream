"""PyTorch dataset for processed GenET vision/action pairs."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .preprocess import PROCESSED_FORMAT_VERSION
from .reference import StatelessReferencePool


@dataclass(frozen=True)
class _TargetCandidate:
    entry_index: int
    episode_id: str
    embodiment: str

    @property
    def pool_key(self) -> tuple[str, str, int]:
        return (self.embodiment, self.episode_id, self.entry_index)


def _distributed_value(name: str, fallback: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return fallback
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"environment variable {name} must be an integer") from exc


def _read_processed_manifest(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(entry, dict):
                raise ValueError(f"{path}:{line_number}: entry must be an object")
            sample_id = entry.get("id")
            if not isinstance(sample_id, str) or not sample_id:
                raise ValueError(f"{path}:{line_number}: missing id")
            if sample_id in seen:
                raise ValueError(f"{path}:{line_number}: duplicate id {sample_id!r}")
            seen.add(sample_id)
            if entry.get("format_version") != PROCESSED_FORMAT_VERSION:
                raise ValueError(
                    f"{path}:{line_number}: unsupported format_version "
                    f"{entry.get('format_version')!r}"
                )
            if not isinstance(entry.get("npz"), str):
                raise ValueError(f"{path}:{line_number}: missing npz path")
            entries.append(entry)
    return entries


class ProcessedPairDataset(Dataset[dict[str, Any]]):
    """Load fixed-shape compressed pair samples.

    Args:
        manifest: Processed ``manifest.jsonl`` path or its containing directory.
        sample_format: ``generic`` returns three nested streams. ``cosmos`` returns
            flat target/source-control/reference keys suitable for a Cosmos/Wan
            training adapter.
        reference_mode: ``stored`` uses preprocessing's reference. ``deterministic``
            reselects a target clip of the same embodiment from the full manifest,
            excluding the current target episode.
        shard_by_rank: Slice the global manifest as ``rank, rank+world_size, ...``.
            This is useful when every RoCE node has a complete local data copy. Do
            not additionally use a DistributedSampler when this is enabled.
    """

    def __init__(
        self,
        manifest: str | Path,
        *,
        sample_format: str = "generic",
        reference_mode: str = "stored",
        reference_seed: int = 0,
        reference_per_epoch: bool = False,
        normalize_video: str = "minus_one_one",
        shard_by_rank: bool = False,
        global_rank: Optional[int] = None,
        world_size: Optional[int] = None,
    ) -> None:
        path = Path(manifest).expanduser().resolve()
        if path.is_dir():
            path = path / "manifest.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"processed manifest does not exist: {path}")
        if sample_format not in {"generic", "cosmos"}:
            raise ValueError("sample_format must be 'generic' or 'cosmos'")
        if reference_mode == "deterministic_pool":
            reference_mode = "deterministic"
        if reference_mode not in {"stored", "deterministic"}:
            raise ValueError("reference_mode must be 'stored' or 'deterministic'")
        if normalize_video not in {"minus_one_one", "zero_one", "none"}:
            raise ValueError(
                "normalize_video must be 'minus_one_one', 'zero_one', or 'none'"
            )

        self.manifest_path = path
        self.root = path.parent
        self.entries = _read_processed_manifest(path)
        self.sample_format = sample_format
        self.reference_mode = reference_mode
        self.reference_seed = int(reference_seed)
        self.reference_per_epoch = bool(reference_per_epoch)
        self.normalize_video = normalize_video
        self.shard_by_rank = bool(shard_by_rank)
        self.global_rank = (
            _distributed_value("RANK", 0) if global_rank is None else int(global_rank)
        )
        self.world_size = (
            _distributed_value("WORLD_SIZE", 1)
            if world_size is None
            else int(world_size)
        )
        if self.world_size <= 0:
            raise ValueError("world_size must be positive")
        if self.global_rank < 0 or self.global_rank >= self.world_size:
            raise ValueError("global_rank must satisfy 0 <= rank < world_size")
        self.global_indices = tuple(
            range(
                self.global_rank if self.shard_by_rank else 0,
                len(self.entries),
                self.world_size if self.shard_by_rank else 1,
            )
        )
        self.global_length = len(self.entries)
        self.epoch = 0

        candidates: list[_TargetCandidate] = []
        for index, entry in enumerate(self.entries):
            streams: dict[str, Mapping[str, Any]] = {}
            for role in ("source", "target_gt", "reference_target"):
                metadata = entry.get(role)
                if not isinstance(metadata, Mapping):
                    raise ValueError(
                        f"processed entry {entry['id']!r} has no valid {role} metadata"
                    )
                episode = metadata.get("episode_id")
                embodiment = metadata.get("embodiment")
                if not isinstance(episode, str) or not episode:
                    raise ValueError(
                        f"processed entry {entry['id']!r} has invalid {role}.episode_id"
                    )
                if not isinstance(embodiment, str) or not embodiment:
                    raise ValueError(
                        f"processed entry {entry['id']!r} has invalid {role}.embodiment"
                    )
                streams[role] = metadata
            target = streams["target_gt"]
            reference = streams["reference_target"]
            if reference["embodiment"] != target["embodiment"]:
                raise ValueError(
                    f"processed entry {entry['id']!r} reference embodiment differs from target"
                )
            if reference["episode_id"] == target["episode_id"]:
                raise ValueError(
                    f"processed entry {entry['id']!r} reference reuses the target episode"
                )
            episode_id = target.get("episode_id")
            embodiment = target.get("embodiment")
            assert isinstance(episode_id, str) and isinstance(embodiment, str)
            candidates.append(_TargetCandidate(index, episode_id, embodiment))
        self._reference_pool = StatelessReferencePool(
            candidates, seed=self.reference_seed
        )

    def set_epoch(self, epoch: int) -> None:
        """Set optional epoch salt without introducing worker-local RNG state."""

        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.global_indices)

    def _entry_for_local_index(self, index: int) -> tuple[int, dict[str, Any]]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        global_index = self.global_indices[index]
        return global_index, self.entries[global_index]

    def _load_npz(self, entry: Mapping[str, Any]) -> dict[str, np.ndarray]:
        raw = Path(str(entry["npz"]))
        if raw.is_absolute():
            raise ValueError("processed npz paths must be relative to the manifest directory")
        path = (self.root / raw).resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"processed npz path escapes manifest root: {raw}") from exc
        if not path.is_file():
            raise FileNotFoundError(f"processed sample does not exist: {path}")
        with np.load(path, allow_pickle=False) as archive:
            required = {
                f"{role}_{suffix}"
                for role in ("source", "target", "reference")
                for suffix in ("video", "actions", "action_mask", "frame_mask")
            }
            missing = sorted(required - set(archive.files))
            if missing:
                raise ValueError(f"{path} is missing arrays: {missing}")
            return {key: archive[key] for key in required}

    def _dynamic_reference(
        self,
        *,
        entry: Mapping[str, Any],
        arrays: dict[str, np.ndarray],
    ) -> Mapping[str, Any]:
        target = entry["target_gt"]
        salt = str(self.epoch) if self.reference_per_epoch else ""
        candidate = self._reference_pool.choose(
            embodiment=str(target["embodiment"]),
            exclude_episode_id=str(target["episode_id"]),
            sample_key=str(entry["id"]),
            salt=salt,
        )
        candidate_entry = self.entries[candidate.entry_index]
        candidate_arrays = self._load_npz(candidate_entry)
        for suffix in ("video", "actions", "action_mask", "frame_mask"):
            arrays[f"reference_{suffix}"] = candidate_arrays[f"target_{suffix}"]
        return candidate_entry["target_gt"]

    def _video_tensor(self, value: np.ndarray) -> torch.Tensor:
        if value.ndim != 4 or value.shape[-1] != 3:
            raise ValueError(f"processed video must be THWC RGB, got {value.shape}")
        tensor = torch.from_numpy(np.ascontiguousarray(value)).permute(3, 0, 1, 2)
        if self.normalize_video == "none":
            return tensor.contiguous()
        tensor = tensor.float().div_(255.0)
        if self.normalize_video == "minus_one_one":
            tensor.mul_(2.0).sub_(1.0)
        return tensor.contiguous()

    @staticmethod
    def _stream(
        arrays: Mapping[str, np.ndarray], prefix: str, video_fn: Any
    ) -> dict[str, torch.Tensor]:
        actions = np.asarray(arrays[f"{prefix}_actions"], dtype=np.float32)
        action_mask = np.asarray(arrays[f"{prefix}_action_mask"], dtype=np.bool_)
        frame_mask = np.asarray(arrays[f"{prefix}_frame_mask"], dtype=np.bool_)
        if actions.ndim != 2 or action_mask.shape != actions.shape:
            raise ValueError(f"invalid processed {prefix} action/mask shapes")
        if frame_mask.shape != (actions.shape[0],):
            raise ValueError(f"invalid processed {prefix} frame-mask shape")
        return {
            "video": video_fn(arrays[f"{prefix}_video"]),
            "actions": torch.from_numpy(np.ascontiguousarray(actions)),
            "action_mask": torch.from_numpy(np.ascontiguousarray(action_mask)),
            "frame_mask": torch.from_numpy(np.ascontiguousarray(frame_mask)),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        global_index, entry = self._entry_for_local_index(index)
        arrays = self._load_npz(entry)
        reference_metadata: Mapping[str, Any] = entry["reference_target"]
        if self.reference_mode == "deterministic":
            reference_metadata = self._dynamic_reference(entry=entry, arrays=arrays)

        source = self._stream(arrays, "source", self._video_tensor)
        target = self._stream(arrays, "target", self._video_tensor)
        reference = self._stream(arrays, "reference", self._video_tensor)
        metadata = {
            "manifest": dict(entry.get("metadata", {})),
            "source": dict(entry["source"]),
            "target_gt": dict(entry["target_gt"]),
            "reference_target": dict(reference_metadata),
            "global_index": global_index,
        }
        if self.sample_format == "generic":
            return {
                "sample_id": entry["id"],
                "source": source,
                "target": target,
                "reference_target": reference,
                "metadata": metadata,
            }
        return {
            "sample_id": entry["id"],
            "video": target["video"],
            "actions": target["actions"],
            "action_mask": target["action_mask"],
            "frame_mask": target["frame_mask"],
            "control_video": source["video"],
            "control_actions": source["actions"],
            "control_action_mask": source["action_mask"],
            "control_frame_mask": source["frame_mask"],
            "reference_video": reference["video"],
            "reference_actions": reference["actions"],
            "reference_action_mask": reference["action_mask"],
            "reference_frame_mask": reference["frame_mask"],
            "metadata": metadata,
        }


def _stack_streams(streams: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        key: torch.stack([stream[key] for stream in streams], dim=0)
        for key in streams[0]
    }


def collate_pairs(batch: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Collate either sample format while keeping heterogeneous metadata as a list."""

    if not batch:
        raise ValueError("cannot collate an empty batch")
    sample_ids = [str(sample["sample_id"]) for sample in batch]
    metadata = [sample["metadata"] for sample in batch]
    if "source" in batch[0]:
        return {
            "sample_id": sample_ids,
            "source": _stack_streams([sample["source"] for sample in batch]),
            "target": _stack_streams([sample["target"] for sample in batch]),
            "reference_target": _stack_streams(
                [sample["reference_target"] for sample in batch]
            ),
            "metadata": metadata,
        }
    tensor_keys = [
        key
        for key, value in batch[0].items()
        if isinstance(value, torch.Tensor)
    ]
    result: dict[str, Any] = {
        key: torch.stack([sample[key] for sample in batch], dim=0)
        for key in tensor_keys
    }
    result["sample_id"] = sample_ids
    result["metadata"] = metadata
    return result


pair_collate_fn = collate_pairs
