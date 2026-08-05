"""Raw manifest schema for cross-embodiment vision/action pairs.

The first version intentionally keeps the on-disk contract small.  Dataset-specific
metadata is retained as an opaque mapping so that a richer robotics schema can be
added without changing the preprocessing core.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "genet.raw-pair/v1"


class SchemaError(ValueError):
    """Raised when a raw JSONL record does not satisfy the v1 contract."""


RAW_PAIR_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": SCHEMA_VERSION,
    "type": "object",
    "required": ["id", "source", "target_gt"],
    "properties": {
        "id": {"type": "string", "minLength": 1},
        "source": {"$ref": "#/$defs/episode"},
        "target_gt": {"$ref": "#/$defs/episode"},
        "reference_target": {"$ref": "#/$defs/episode"},
        "metadata": {"type": "object"},
    },
    "$defs": {
        "episode": {
            "type": "object",
            "required": ["episode_id", "embodiment", "video", "actions"],
            "properties": {
                "episode_id": {"type": "string", "minLength": 1},
                "embodiment": {"type": "string", "minLength": 1},
                "video": {"type": "string", "minLength": 1},
                "actions": {"type": "string", "minLength": 1},
                "start_time": {"type": "number", "minimum": 0},
                "end_time": {"type": "number", "exclusiveMinimum": 0},
                "duration": {"type": "number", "exclusiveMinimum": 0},
                "video_fps": {"type": "number", "exclusiveMinimum": 0},
                "action_fps": {"type": "number", "exclusiveMinimum": 0},
                "action_start_time": {"type": "number", "minimum": 0},
                "action_timestamps": {"type": "string"},
                "video_key": {"type": "string"},
                "action_key": {"type": "string"},
                "timestamp_key": {"type": "string"},
                "metadata": {"type": "object"},
            },
        }
    },
}


def _required_string(data: Mapping[str, Any], key: str, context: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SchemaError(f"{context}.{key} must be a non-empty string")
    return value


def _optional_positive_float(
    data: Mapping[str, Any], key: str, context: str
) -> float | None:
    value = data.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchemaError(f"{context}.{key} must be a number")
    value = float(value)
    if value <= 0:
        raise SchemaError(f"{context}.{key} must be > 0")
    return value


def _resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base_dir / path).resolve()


@dataclass(frozen=True)
class EpisodeRef:
    """A time range from one embodiment's synchronized video/action episode."""

    episode_id: str
    embodiment: str
    video: Path
    actions: Path
    start_time: float = 0.0
    end_time: float | None = None
    video_fps: float | None = None
    action_fps: float | None = None
    action_start_time: float = 0.0
    action_timestamps: Path | None = None
    video_key: str | None = None
    action_key: str | None = None
    timestamp_key: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False)

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any], *, base_dir: Path, context: str
    ) -> EpisodeRef:
        if not isinstance(data, Mapping):
            raise SchemaError(f"{context} must be an object")

        start = data.get("start_time", data.get("clip_start", 0.0))
        if isinstance(start, bool) or not isinstance(start, (int, float)):
            raise SchemaError(f"{context}.start_time must be a number")
        start = float(start)
        if start < 0:
            raise SchemaError(f"{context}.start_time must be >= 0")

        end_value = data.get("end_time")
        duration = data.get("duration")
        if end_value is not None and duration is not None:
            raise SchemaError(f"{context} may set end_time or duration, not both")
        if duration is not None:
            if isinstance(duration, bool) or not isinstance(duration, (int, float)):
                raise SchemaError(f"{context}.duration must be a number")
            if float(duration) <= 0:
                raise SchemaError(f"{context}.duration must be > 0")
            end_value = start + float(duration)
        if end_value is not None:
            if isinstance(end_value, bool) or not isinstance(end_value, (int, float)):
                raise SchemaError(f"{context}.end_time must be a number")
            end_value = float(end_value)
            if end_value <= start:
                raise SchemaError(f"{context}.end_time must be greater than start_time")

        metadata = data.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise SchemaError(f"{context}.metadata must be an object")

        timestamps = data.get("action_timestamps")
        if timestamps is not None and not isinstance(timestamps, str):
            raise SchemaError(f"{context}.action_timestamps must be a path string")
        action_start = data.get("action_start_time", 0.0)
        if isinstance(action_start, bool) or not isinstance(action_start, (int, float)):
            raise SchemaError(f"{context}.action_start_time must be a number")
        action_start = float(action_start)
        if action_start < 0:
            raise SchemaError(f"{context}.action_start_time must be >= 0")

        def optional_string(key: str) -> str | None:
            value = data.get(key)
            if value is not None and (not isinstance(value, str) or not value):
                raise SchemaError(f"{context}.{key} must be a non-empty string")
            return value

        return cls(
            episode_id=_required_string(data, "episode_id", context),
            embodiment=_required_string(data, "embodiment", context),
            video=_resolve_path(_required_string(data, "video", context), base_dir),
            actions=_resolve_path(_required_string(data, "actions", context), base_dir),
            start_time=start,
            end_time=end_value,
            video_fps=_optional_positive_float(data, "video_fps", context),
            action_fps=_optional_positive_float(data, "action_fps", context),
            action_start_time=action_start,
            action_timestamps=(
                _resolve_path(timestamps, base_dir) if timestamps is not None else None
            ),
            video_key=optional_string("video_key"),
            action_key=optional_string("action_key"),
            timestamp_key=optional_string("timestamp_key"),
            metadata=dict(metadata),
        )

    @property
    def pool_key(self) -> tuple[str, str, str, float, float | None]:
        """Stable key used to de-duplicate reference-pool entries."""

        return (
            self.embodiment,
            self.episode_id,
            str(self.video),
            self.start_time,
            self.end_time,
        )

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "embodiment": self.embodiment,
            "video": str(self.video),
            "actions": str(self.actions),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "action_start_time": self.action_start_time,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class PairRecord:
    """One source task paired with its ground-truth target embodiment task."""

    sample_id: str
    source: EpisodeRef
    target_gt: EpisodeRef
    reference_target: EpisodeRef | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False, hash=False)

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any], *, base_dir: Path, context: str
    ) -> PairRecord:
        if not isinstance(data, Mapping):
            raise SchemaError(f"{context} must be an object")
        sample_id = data.get("id", data.get("sample_id"))
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise SchemaError(f"{context}.id must be a non-empty string")
        if "source" not in data or "target_gt" not in data:
            raise SchemaError(f"{context} requires source and target_gt")
        source = EpisodeRef.from_dict(
            data["source"], base_dir=base_dir, context=f"{context}.source"
        )
        target = EpisodeRef.from_dict(
            data["target_gt"], base_dir=base_dir, context=f"{context}.target_gt"
        )
        if source.embodiment == target.embodiment:
            raise SchemaError(
                f"{context} must pair distinct source and target_gt embodiments"
            )
        reference_data = data.get("reference_target")
        reference = (
            EpisodeRef.from_dict(
                reference_data,
                base_dir=base_dir,
                context=f"{context}.reference_target",
            )
            if reference_data is not None
            else None
        )
        if reference is not None:
            if reference.embodiment != target.embodiment:
                raise SchemaError(
                    f"{context}.reference_target embodiment must equal target_gt embodiment"
                )
            if reference.episode_id == target.episode_id:
                raise SchemaError(
                    f"{context}.reference_target must not use the target_gt episode"
                )
        metadata = data.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise SchemaError(f"{context}.metadata must be an object")
        return cls(sample_id, source, target, reference, dict(metadata))


def iter_raw_manifest(path: str | Path) -> Iterator[PairRecord]:
    """Stream and validate a JSONL raw-pair manifest.

    Relative media paths are resolved relative to the JSONL file rather than the
    process working directory.
    """

    manifest_path = Path(path).expanduser().resolve()
    seen_ids: set[str] = set()
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SchemaError(
                    f"{manifest_path}:{line_number}: invalid JSON: {exc.msg}"
                ) from exc
            record = PairRecord.from_dict(
                raw,
                base_dir=manifest_path.parent,
                context=f"{manifest_path}:{line_number}",
            )
            if record.sample_id in seen_ids:
                raise SchemaError(
                    f"{manifest_path}:{line_number}: duplicate id {record.sample_id!r}"
                )
            seen_ids.add(record.sample_id)
            yield record


def read_raw_manifest(path: str | Path) -> list[PairRecord]:
    return list(iter_raw_manifest(path))


def build_reference_candidates(records: Iterable[PairRecord]) -> list[EpisodeRef]:
    """Collect unique target-embodiment clips for the implicit reference pool."""

    candidates: dict[tuple[str, str, str, float, float | None], EpisodeRef] = {}
    for record in records:
        candidates.setdefault(record.target_gt.pool_key, record.target_gt)
        if record.reference_target is not None:
            candidates.setdefault(record.reference_target.pool_key, record.reference_target)
    return list(candidates.values())
