"""End-to-end preprocessing for synchronized cross-embodiment pairs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping, Optional

import numpy as np

from .actions import ActionSeries, load_action_series, resample_actions
from .reference import StatelessReferencePool, stable_index
from .sampling import ShortSequenceError, make_time_grid
from .schema import (
    EpisodeRef,
    build_reference_candidates,
    read_raw_manifest,
)
from .video import DecodedVideo, decode_video, resize_center_crop, sample_video


PROCESSED_FORMAT_VERSION = "genet.processed-pair/v1"


@dataclass(frozen=True)
class PreprocessConfig:
    """Configuration shared by all three streams in a processed pair."""

    num_frames: int = 81
    sample_fps: float = 16.0
    height: int = 192
    width: int = 320
    action_dim: int = 64
    action_resample: str = "linear"
    short_policy: str = "drop"
    truncate_actions: bool = False
    default_video_fps: Optional[float] = 30.0
    default_action_fps: Optional[float] = 30.0
    video_layout: str = "auto"
    reference_seed: int = 0
    randomize_reference_start: bool = True
    validate_wan_frames: bool = True
    wan_frame_offset: int = 1
    wan_temporal_stride: int = 4

    def __post_init__(self) -> None:
        if self.num_frames <= 0:
            raise ValueError("num_frames must be positive")
        if self.sample_fps <= 0:
            raise ValueError("sample_fps must be positive")
        if self.height <= 0 or self.width <= 0:
            raise ValueError("height and width must be positive")
        if self.action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if self.action_resample not in {"linear", "nearest"}:
            raise ValueError("action_resample must be 'linear' or 'nearest'")
        if self.short_policy not in {"drop", "pad"}:
            raise ValueError("short_policy must be 'drop' or 'pad'")
        if self.video_layout.upper() not in {"AUTO", "THWC", "TCHW", "CTHW"}:
            raise ValueError("video_layout must be auto, THWC, TCHW, or CTHW")
        for name in ("default_video_fps", "default_action_fps"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive or null")
        if self.wan_frame_offset < 0 or self.wan_temporal_stride <= 0:
            raise ValueError("Wan frame offset/stride must be non-negative/positive")
        if self.validate_wan_frames and (
            self.num_frames < self.wan_frame_offset
            or (self.num_frames - self.wan_frame_offset) % self.wan_temporal_stride != 0
        ):
            raise ValueError(
                "num_frames violates configured Wan temporal length: expected "
                f"{self.wan_frame_offset} + {self.wan_temporal_stride} * N, got "
                f"{self.num_frames}"
            )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "PreprocessConfig":
        """Build from either a config object or its ``preprocess`` subsection."""

        values: Mapping[str, Any] = data.get("preprocess", data)  # type: ignore[arg-type]
        if not isinstance(values, Mapping):
            raise ValueError("preprocess config must be a JSON object")
        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(values) - known)
        if unknown:
            raise ValueError(f"unknown preprocess config keys: {unknown}")
        return cls(**dict(values))

    @property
    def clip_span_seconds(self) -> float:
        return (self.num_frames - 1) / self.sample_fps


@dataclass(frozen=True)
class PreprocessReport:
    manifest: Path
    index: Path
    stats: Path
    written: int
    dropped: int
    failed: int


@dataclass(frozen=True)
class _ProcessedStream:
    video: np.ndarray
    actions: np.ndarray
    action_mask: np.ndarray
    frame_mask: np.ndarray
    clip_start: float


class _ActionStats:
    def __init__(self, action_dim: int) -> None:
        self.count = np.zeros(action_dim, dtype=np.int64)
        self.total = np.zeros(action_dim, dtype=np.float64)
        self.total_sq = np.zeros(action_dim, dtype=np.float64)

    def update(self, values: np.ndarray, mask: np.ndarray) -> None:
        numeric_mask = mask.astype(np.float64, copy=False)
        self.count += mask.sum(axis=0, dtype=np.int64)
        self.total += (values * numeric_mask).sum(axis=0, dtype=np.float64)
        self.total_sq += ((values.astype(np.float64) ** 2) * numeric_mask).sum(
            axis=0, dtype=np.float64
        )

    def as_dict(self) -> dict[str, Any]:
        safe_count = np.maximum(self.count, 1)
        mean = self.total / safe_count
        variance = np.maximum(self.total_sq / safe_count - mean**2, 0.0)
        mean[self.count == 0] = 0.0
        variance[self.count == 0] = 0.0
        return {
            "count": self.count.tolist(),
            "mean": mean.tolist(),
            "std": np.sqrt(variance).tolist(),
        }


def _stable_fraction(key: str, *, seed: int) -> float:
    denominator = (1 << 32) - 1
    return stable_index(key, denominator + 1, seed=seed) / denominator


def _reference_start(
    ref: EpisodeRef,
    *,
    coverage_start: float,
    coverage_end: float,
    sample_key: str,
    config: PreprocessConfig,
) -> float:
    """Choose a deterministic window wholly inside common vision/action coverage."""

    lower = coverage_start
    upper = coverage_end - config.clip_span_seconds
    if upper <= lower or not config.randomize_reference_start:
        return lower
    fraction = _stable_fraction(
        f"{sample_key}\0{ref.episode_id}\0reference-start",
        seed=config.reference_seed,
    )
    return lower + fraction * (upper - lower)


def _logical_interval_mask(timestamps: np.ndarray, ref: EpisodeRef) -> np.ndarray:
    """Select media observations that belong to an episode's logical range."""

    mask = timestamps >= ref.start_time - 1.0e-9
    if ref.end_time is not None:
        mask &= timestamps <= ref.end_time + 1.0e-9
    return mask


def _restrict_video_to_episode(decoded: DecodedVideo, ref: EpisodeRef) -> DecodedVideo:
    mask = _logical_interval_mask(decoded.timestamps, ref)
    if not bool(mask.any()):
        raise ShortSequenceError(
            f"episode {ref.episode_id!r} has no video frames inside its logical range"
        )
    return DecodedVideo(decoded.frames[mask], decoded.timestamps[mask])


def _restrict_actions_to_episode(series: ActionSeries, ref: EpisodeRef) -> ActionSeries:
    mask = _logical_interval_mask(series.timestamps, ref)
    if not bool(mask.any()):
        raise ShortSequenceError(
            f"episode {ref.episode_id!r} has no actions inside its logical range"
        )
    return ActionSeries(series.values[mask], series.timestamps[mask])


def _process_stream(
    ref: EpisodeRef,
    *,
    config: PreprocessConfig,
    clip_start: Optional[float] = None,
    reference_key: Optional[str] = None,
) -> _ProcessedStream:
    decoded = decode_video(
        ref.video,
        fps=ref.video_fps or config.default_video_fps,
        video_key=ref.video_key,
        layout=config.video_layout,
    )
    # Restrict the candidate observations before nearest sampling/interpolation.
    # Clamping only the query grid is insufficient: a nearest frame or linear
    # interpolation endpoint can otherwise come from an adjacent task segment.
    decoded = _restrict_video_to_episode(decoded, ref)
    action_series = load_action_series(
        ref.actions,
        action_key=ref.action_key,
        timestamp_key=ref.timestamp_key,
        timestamps_path=ref.action_timestamps,
        default_fps=ref.action_fps or config.default_action_fps,
        start_time=ref.action_start_time,
    )
    action_series = _restrict_actions_to_episode(action_series, ref)

    coverage_start = max(
        ref.start_time,
        float(decoded.timestamps[0]),
        float(action_series.timestamps[0]),
    )
    coverage_end = min(
        float(decoded.timestamps[-1]),
        float(action_series.timestamps[-1]),
        ref.end_time if ref.end_time is not None else float("inf"),
    )
    if coverage_end < coverage_start:
        raise ShortSequenceError(
            f"episode {ref.episode_id!r} has no common video/action time coverage"
        )
    if clip_start is None:
        clip_start = coverage_start
    else:
        # A paired stream may request a logical start that precedes its first
        # actual observation. Snap forward instead of treating valid media as a
        # short sequence solely because timestamps are not boundary-aligned.
        clip_start = max(float(clip_start), coverage_start)
    if reference_key is not None:
        clip_start = _reference_start(
            ref,
            coverage_start=coverage_start,
            coverage_end=coverage_end,
            sample_key=reference_key,
            config=config,
        )
    time_grid = make_time_grid(
        start_time=clip_start,
        num_frames=config.num_frames,
        fps=config.sample_fps,
    )
    if time_grid[-1] > coverage_end + 1e-9 and config.short_policy == "drop":
        raise ShortSequenceError(
            f"episode {ref.episode_id!r} common video/action coverage is too short"
        )
    # A logical clip boundary is part of the data contract even when the backing
    # media continues.  Padding must repeat the boundary value, not leak frames or
    # actions from the following task segment.
    sampling_grid = (
        np.minimum(time_grid, ref.end_time)
        if ref.end_time is not None
        else time_grid
    )
    video, frame_mask = sample_video(
        decoded, sampling_grid, short_policy=config.short_policy
    )
    video = resize_center_crop(video, height=config.height, width=config.width)
    actions, action_mask = resample_actions(
        action_series,
        sampling_grid,
        method=config.action_resample,
        action_dim=config.action_dim,
        short_policy=config.short_policy,
        truncate=config.truncate_actions,
    )
    if ref.end_time is not None:
        logical_valid = time_grid <= ref.end_time + 1e-9
        frame_mask &= logical_valid
        action_mask &= logical_valid[:, None]
    return _ProcessedStream(
        video=np.ascontiguousarray(video, dtype=np.uint8),
        actions=np.ascontiguousarray(actions, dtype=np.float32),
        action_mask=np.ascontiguousarray(action_mask, dtype=np.bool_),
        frame_mask=np.ascontiguousarray(frame_mask, dtype=np.bool_),
        clip_start=float(clip_start),
    )


def _sample_filename(sample_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", sample_id).strip("._") or "sample"
    digest = hashlib.blake2b(sample_id.encode("utf-8"), digest_size=6).hexdigest()
    return f"{slug[:80]}-{digest}.npz"


def _atomic_savez(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}-", suffix=".npz", dir=path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}-", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()


def _stream_arrays(prefix: str, stream: _ProcessedStream) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_video": stream.video,
        f"{prefix}_actions": stream.actions,
        f"{prefix}_action_mask": stream.action_mask,
        f"{prefix}_frame_mask": stream.frame_mask,
    }


def _manifest_stream(ref: EpisodeRef, stream: _ProcessedStream) -> dict[str, Any]:
    return {
        "episode_id": ref.episode_id,
        "embodiment": ref.embodiment,
        "clip_start": stream.clip_start,
        "metadata": dict(ref.metadata),
    }


def preprocess_manifest(
    raw_manifest: str | Path,
    output_dir: str | Path,
    *,
    config: PreprocessConfig | Mapping[str, Any] | None = None,
    overwrite: bool = False,
    on_error: str = "raise",
) -> PreprocessReport:
    """Preprocess one raw JSONL manifest into independently loadable NPZ samples.

    ``short_policy='drop'`` always drops short samples.  Other failures are raised
    by default; ``on_error='skip'`` records them in stats and continues.
    """

    if on_error not in {"raise", "skip"}:
        raise ValueError("on_error must be 'raise' or 'skip'")
    if config is None:
        resolved_config = PreprocessConfig()
    elif isinstance(config, PreprocessConfig):
        resolved_config = config
    else:
        resolved_config = PreprocessConfig.from_mapping(config)

    records = read_raw_manifest(raw_manifest)
    output = Path(output_dir).expanduser().resolve()
    manifest_path = output / "manifest.jsonl"
    index_path = output / "index.json"
    stats_path = output / "stats.json"
    existing = [path for path in (manifest_path, index_path, stats_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "processed metadata already exists; pass overwrite=True: "
            + ", ".join(str(path) for path in existing)
        )
    (output / "samples").mkdir(parents=True, exist_ok=True)

    pool = StatelessReferencePool(
        build_reference_candidates(records), seed=resolved_config.reference_seed
    )
    action_stats = {
        role: _ActionStats(resolved_config.action_dim)
        for role in ("source", "target", "reference")
    }
    frame_valid = {role: 0 for role in action_stats}
    frame_total = {role: 0 for role in action_stats}
    processed_records: list[dict[str, Any]] = []
    dropped_errors: list[dict[str, str]] = []
    failed_errors: list[dict[str, str]] = []

    for record in records:
        try:
            reference = record.reference_target or pool.choose(
                embodiment=record.target_gt.embodiment,
                exclude_episode_id=record.target_gt.episode_id,
                sample_key=record.sample_id,
            )
            if reference.episode_id == record.target_gt.episode_id:
                raise ValueError("reference target must exclude the target_gt episode")
            if reference.embodiment != record.target_gt.embodiment:
                raise ValueError("reference target embodiment differs from target_gt")

            source_stream = _process_stream(record.source, config=resolved_config)
            target_stream = _process_stream(record.target_gt, config=resolved_config)
            reference_stream = _process_stream(
                reference,
                config=resolved_config,
                reference_key=record.sample_id,
            )
            streams = {
                "source": source_stream,
                "target": target_stream,
                "reference": reference_stream,
            }
            arrays: dict[str, np.ndarray] = {}
            for role, stream in streams.items():
                arrays.update(_stream_arrays(role, stream))

            relative_npz = Path("samples") / _sample_filename(record.sample_id)
            npz_path = output / relative_npz
            if npz_path.exists() and not overwrite:
                raise FileExistsError(f"processed sample already exists: {npz_path}")
            _atomic_savez(npz_path, arrays)

            manifest_entry = {
                "format_version": PROCESSED_FORMAT_VERSION,
                "id": record.sample_id,
                "npz": relative_npz.as_posix(),
                "source": _manifest_stream(record.source, source_stream),
                "target_gt": _manifest_stream(record.target_gt, target_stream),
                "reference_target": _manifest_stream(reference, reference_stream),
                "shape": {
                    "video": list(source_stream.video.shape),
                    "actions": list(source_stream.actions.shape),
                },
                "metadata": dict(record.metadata),
            }
            processed_records.append(manifest_entry)
            for role, stream in streams.items():
                action_stats[role].update(stream.actions, stream.action_mask)
                frame_valid[role] += int(stream.frame_mask.sum())
                frame_total[role] += int(stream.frame_mask.size)
        except ShortSequenceError as exc:
            dropped_errors.append(
                {"id": record.sample_id, "error": f"{type(exc).__name__}: {exc}"}
            )
        except Exception as exc:
            if on_error == "raise":
                raise RuntimeError(
                    f"failed to preprocess sample {record.sample_id!r}: {exc}"
                ) from exc
            failed_errors.append(
                {"id": record.sample_id, "error": f"{type(exc).__name__}: {exc}"}
            )

    manifest_text = "".join(
        json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n"
        for entry in processed_records
    )
    _atomic_text(manifest_path, manifest_text)

    by_embodiment: dict[str, list[str]] = {}
    by_id: dict[str, dict[str, Any]] = {}
    for line_number, entry in enumerate(processed_records):
        sample_id = entry["id"]
        embodiment = entry["target_gt"]["embodiment"]
        by_embodiment.setdefault(embodiment, []).append(sample_id)
        by_id[sample_id] = {"line": line_number, "npz": entry["npz"]}
    index = {
        "format_version": PROCESSED_FORMAT_VERSION,
        "manifest": manifest_path.name,
        "num_samples": len(processed_records),
        "by_id": by_id,
        "by_target_embodiment": {
            key: value for key, value in sorted(by_embodiment.items())
        },
    }
    _atomic_text(
        index_path,
        json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )

    stats = {
        "format_version": PROCESSED_FORMAT_VERSION,
        "config": asdict(resolved_config),
        "raw_samples": len(records),
        "written_samples": len(processed_records),
        "dropped_short_samples": len(dropped_errors),
        "failed_samples": len(failed_errors),
        "errors": {"dropped": dropped_errors, "failed": failed_errors},
        "actions": {role: value.as_dict() for role, value in action_stats.items()},
        "frame_valid_fraction": {
            role: (frame_valid[role] / frame_total[role] if frame_total[role] else 0.0)
            for role in frame_valid
        },
    }
    _atomic_text(
        stats_path,
        json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return PreprocessReport(
        manifest=manifest_path,
        index=index_path,
        stats=stats_path,
        written=len(processed_records),
        dropped=len(dropped_errors),
        failed=len(failed_errors),
    )
