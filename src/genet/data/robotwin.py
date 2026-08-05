"""Direct RoboTwin-v1 MosaicML Streaming adapter.

The observed RoboTwin cache stores one MDS sample per episode timestep.  This
module validates that concrete contract, reconstructs synchronized frame/state
episodes, and writes the existing ``genet.processed-pair/v1`` format without an
intermediate video or action export.

RoboTwin's ``state`` is a measured joint-position state.  It is exposed through
GenET's generic ``actions`` tensor because that is the model boundary, but the
manifest records the precise signal name and does not claim actuator-command
semantics.
"""

from __future__ import annotations

import importlib.metadata
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from io import BytesIO
from numbers import Integral
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .content import stream_content_sha256
from .directions import require_bidirectional_pairs, summarize_pair_directions
from .preprocess import (
    PROCESSED_FORMAT_VERSION,
    PreprocessConfig,
    PreprocessReport,
    _ActionStats,
    _atomic_savez,
    _atomic_text,
    _sample_filename,
)
from .reference import stable_index
from .video import resize_center_crop

ROBOTWIN_ADAPTER_VERSION = "genet.robotwin-mds/v1"


class RoboTwinSchemaError(ValueError):
    """Raised when an MDS row or adapter contract is inconsistent."""


class RowDataset(Protocol):
    """Minimal random-access surface shared by StreamingDataset and test fakes."""

    def __getitem__(self, index: int) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class RoboTwinContract:
    dataset: str
    schema_version: int
    streaming_version: str
    observed_schema_sha256: str
    default_local_root: str
    columns: Mapping[str, str]
    action_dims: Mapping[str, int]
    cameras: tuple[str, ...]
    window_size: int = 100
    expected_counts: Mapping[str, Mapping[str, Mapping[str, int]]] = field(
        default_factory=dict
    )
    file_sha256: str = ""

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any], *, file_sha256: str = ""
    ) -> RoboTwinContract:
        required = {
            "dataset",
            "schema_version",
            "streaming_version",
            "observed_schema_sha256",
            "default_local_root",
            "expected_counts",
            "columns",
            "action_dims",
            "cameras",
            "window_size",
        }
        missing = sorted(required - set(data))
        if missing:
            raise RoboTwinSchemaError(f"RoboTwin contract is missing keys: {missing}")
        columns = data["columns"]
        action_dims = data["action_dims"]
        cameras = data["cameras"]
        if not isinstance(columns, Mapping) or not isinstance(action_dims, Mapping):
            raise RoboTwinSchemaError("columns and action_dims must be objects")
        if not isinstance(cameras, list) or not cameras:
            raise RoboTwinSchemaError("cameras must be a non-empty array")
        parsed_dims: dict[str, int] = {}
        for embodiment, value in action_dims.items():
            if not isinstance(embodiment, str) or not embodiment:
                raise RoboTwinSchemaError("action_dims keys must be non-empty strings")
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise RoboTwinSchemaError(
                    f"action dimension for {embodiment!r} must be positive"
                )
            parsed_dims[embodiment] = value
        parsed_cameras = tuple(str(value) for value in cameras)
        if any(not value for value in parsed_cameras):
            raise RoboTwinSchemaError("camera names must be non-empty strings")
        digest = str(data["observed_schema_sha256"])
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise RoboTwinSchemaError("observed_schema_sha256 must be lowercase SHA-256")
        window_size = data["window_size"]
        if (
            isinstance(window_size, bool)
            or not isinstance(window_size, int)
            or window_size <= 0
        ):
            raise RoboTwinSchemaError("window_size must be a positive integer")
        expected_counts_raw = data["expected_counts"]
        if not isinstance(expected_counts_raw, Mapping):
            raise RoboTwinSchemaError("expected_counts must be an object")
        expected_counts: dict[str, dict[str, dict[str, int]]] = {}
        for split, by_embodiment in expected_counts_raw.items():
            if split not in {"train", "val"} or not isinstance(
                by_embodiment, Mapping
            ):
                raise RoboTwinSchemaError("expected_counts must define train/val maps")
            expected_counts[split] = {}
            for embodiment, counts in by_embodiment.items():
                if embodiment not in parsed_dims or not isinstance(counts, Mapping):
                    raise RoboTwinSchemaError(
                        f"invalid expected_counts entry {split}/{embodiment}"
                    )
                parsed: dict[str, int] = {}
                for name in ("tasks", "episodes", "samples"):
                    value = counts.get(name)
                    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                        raise RoboTwinSchemaError(
                            f"expected_counts.{split}.{embodiment}.{name} "
                            "must be positive"
                        )
                    parsed[name] = value
                expected_counts[split][embodiment] = parsed
        if set(expected_counts) != {"train", "val"} or any(
            set(expected_counts[split]) != set(parsed_dims)
            for split in ("train", "val")
        ):
            raise RoboTwinSchemaError(
                "expected_counts must cover every embodiment in train and val"
            )
        return cls(
            dataset=str(data["dataset"]),
            schema_version=int(data["schema_version"]),
            streaming_version=str(data["streaming_version"]),
            observed_schema_sha256=digest,
            default_local_root=str(data["default_local_root"]),
            columns=dict(columns),
            action_dims=parsed_dims,
            cameras=parsed_cameras,
            window_size=window_size,
            expected_counts=expected_counts,
            file_sha256=file_sha256,
        )


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_robotwin_contract(path: str | Path) -> RoboTwinContract:
    contract_path = Path(path).expanduser().resolve()
    try:
        raw = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RoboTwinSchemaError(
            f"cannot load RoboTwin contract {contract_path}: {exc}"
        ) from exc
    if not isinstance(raw, Mapping):
        raise RoboTwinSchemaError("RoboTwin contract root must be an object")
    return RoboTwinContract.from_mapping(raw, file_sha256=_file_sha256(contract_path))


@dataclass(frozen=True, order=True)
class EpisodeKey:
    split: str
    task: str
    embodiment: str
    episode_idx: int

    @property
    def pair_key(self) -> tuple[str, int]:
        return (self.task, self.episode_idx)

    @property
    def episode_id(self) -> str:
        return (
            f"robotwin-v1/{self.split}/{self.embodiment}/{self.task}/"
            f"episode{self.episode_idx}"
        )


@dataclass(frozen=True)
class EpisodeRows:
    key: EpisodeKey
    episode_len: int
    row_indices: tuple[int, ...]
    action_dim: int


@dataclass(frozen=True)
class RoboTwinPreprocessConfig:
    """Semantic and sampling choices absent from the structural MDS schema."""

    mds_index_fps: float
    preprocess: PreprocessConfig = field(
        default_factory=lambda: PreprocessConfig(
            default_video_fps=None, default_action_fps=None
        )
    )
    camera: str = "head"
    action_alignment: str = "frame"
    action_signal: str = "joint_position_state"
    window_policy: str = "episode_start"
    clip_stride_native_frames: int = 81
    reference_policy: str = "different_task"
    validate_action_windows: bool = True

    def __post_init__(self) -> None:
        if self.mds_index_fps <= 0 or not math.isfinite(self.mds_index_fps):
            raise ValueError("mds_index_fps must be a finite positive value")
        if self.camera not in {"head", "left", "right"}:
            raise ValueError("camera must be head, left, or right")
        if self.action_alignment != "frame":
            raise ValueError(
                "RoboTwin v1 currently supports only frame-aligned joint state"
            )
        if self.action_signal != "joint_position_state":
            raise ValueError("action_signal must be joint_position_state")
        if self.window_policy not in {"episode_start", "sliding"}:
            raise ValueError("window_policy must be episode_start or sliding")
        if self.clip_stride_native_frames <= 0:
            raise ValueError("clip_stride_native_frames must be positive")
        if self.reference_policy not in {"different_task", "any_task"}:
            raise ValueError("reference_policy must be different_task or any_task")
        if self.preprocess.short_policy != "drop":
            raise ValueError(
                "RoboTwin Cosmos training requires preprocess.short_policy='drop'"
            )
        if not math.isclose(
            self.mds_index_fps,
            self.preprocess.sample_fps,
            rel_tol=0.0,
            abs_tol=1e-12,
        ) and self.preprocess.action_resample != "nearest":
            raise ValueError(
                "non-1:1 RoboTwin FPS conversion cannot linearly interpolate "
                "joint-state channels; set preprocess.action_resample='nearest'"
            )


@dataclass(frozen=True)
class _StreamArrays:
    video: np.ndarray
    actions: np.ndarray
    action_mask: np.ndarray
    frame_mask: np.ndarray
    clip_start: float


def _num_rows(dataset: RowDataset) -> int:
    value = getattr(dataset, "num_samples", None)
    if value is not None:
        result = int(value)
    else:
        try:
            result = len(dataset)  # type: ignore[arg-type]
        except TypeError as exc:
            raise TypeError("row dataset needs num_samples or __len__") from exc
    if result < 0:
        raise ValueError("row dataset length cannot be negative")
    return result


def open_robotwin_mds(
    root: str | Path,
    *,
    split: str,
    embodiment: str,
    expected_streaming_version: str,
) -> RowDataset:
    """Open one local aggregate MDS stream without rank/worker auto-sharding."""

    stream_root = Path(root).expanduser().resolve() / split / embodiment
    if not (stream_root / "index.json").is_file():
        raise FileNotFoundError(
            f"RoboTwin aggregate index does not exist: {stream_root / 'index.json'}"
        )
    try:
        installed = importlib.metadata.version("mosaicml-streaming")
    except importlib.metadata.PackageNotFoundError as exc:
        raise ImportError(
            "RoboTwin MDS support requires the 'robotwin' extra: "
            "python -m pip install -e '.[robotwin]'"
        ) from exc
    if installed != expected_streaming_version:
        raise RuntimeError(
            "MosaicML Streaming version mismatch: contract requires "
            f"{expected_streaming_version}, installed {installed}"
        )
    try:
        from streaming import StreamingDataset  # type: ignore
    except ImportError as exc:  # pragma: no cover - distribution/import mismatch
        raise ImportError("mosaicml-streaming is installed but cannot be imported") from exc
    # Index-driven iteration via range(ds.num_samples) is intentional. Iterating
    # the IterableDataset directly would auto-shard by rank and worker.
    return StreamingDataset(
        local=str(stream_root), remote=None, shuffle=False, batch_size=1
    )


def validate_robotwin_root(
    root: str | Path, contract: RoboTwinContract
) -> tuple[Path, str]:
    """Validate the top-level dataset identity before opening any large stream."""

    dataset_root = Path(root).expanduser().resolve()
    manifest_path = dataset_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"RoboTwin root manifest does not exist: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RoboTwinSchemaError(f"invalid RoboTwin root manifest: {exc}") from exc
    if not isinstance(manifest, Mapping):
        raise RoboTwinSchemaError("RoboTwin root manifest must be an object")
    expected_fields: dict[str, Any] = {
        "schema_version": contract.schema_version,
        "mosaicml_streaming_version": contract.streaming_version,
        "k_max": contract.window_size,
        "action_dims": dict(contract.action_dims),
        "cameras": list(contract.cameras),
        "columns": dict(contract.columns),
        "embodiments": list(contract.action_dims),
    }
    for name, expected in expected_fields.items():
        if manifest.get(name) != expected:
            raise RoboTwinSchemaError(
                f"root manifest {name} differs from the approved contract"
            )
    expected_totals = {
        split: {
            "episodes": sum(
                counts["episodes"]
                for counts in contract.expected_counts.get(split, {}).values()
            ),
            "samples": sum(
                counts["samples"]
                for counts in contract.expected_counts.get(split, {}).values()
            ),
        }
        for split in ("train", "val")
    }
    if contract.expected_counts and manifest.get("totals") != expected_totals:
        raise RoboTwinSchemaError(
            "root manifest totals differ from the approved RoboTwin counts"
        )
    for split in ("train", "val"):
        for embodiment in contract.action_dims:
            index_path = dataset_root / split / embodiment / "index.json"
            if not index_path.is_file():
                raise FileNotFoundError(
                    f"RoboTwin aggregate index does not exist: {index_path}"
                )
    return dataset_root, _file_sha256(manifest_path)


def _require_int(row: Mapping[str, Any], key: str, context: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise RoboTwinSchemaError(f"{context}.{key} must be an integer")
    return int(value)


def _jpeg_bytes(value: Any, *, context: str) -> bytes:
    if isinstance(value, bytes):
        result = value
    elif isinstance(value, (bytearray, memoryview)):
        result = bytes(value)
    else:
        raise RoboTwinSchemaError(f"{context} must contain JPEG bytes")
    if len(result) < 4 or not result.startswith(b"\xff\xd8"):
        raise RoboTwinSchemaError(f"{context} is not a JPEG payload")
    return result


def build_episode_catalog(
    dataset: RowDataset,
    *,
    contract: RoboTwinContract,
    split: str,
    embodiment: str,
    validate_action_windows: bool = True,
) -> dict[tuple[str, int], EpisodeRows]:
    """Validate one embodiment stream and index its contiguous episodes."""

    if embodiment not in contract.action_dims:
        raise RoboTwinSchemaError(f"unknown embodiment {embodiment!r}")
    action_dim = int(contract.action_dims[embodiment])
    required = set(contract.columns)
    camera_fields = tuple(f"{camera}_rgb" for camera in contract.cameras)
    catalog: dict[tuple[str, int], EpisodeRows] = {}
    active_key: EpisodeKey | None = None
    active_indices: list[int] = []
    active_len = 0
    previous_window: np.ndarray | None = None
    closed: set[EpisodeKey] = set()

    def finish_active() -> None:
        nonlocal active_key, active_indices, active_len, previous_window
        if active_key is None:
            return
        if len(active_indices) != active_len:
            raise RoboTwinSchemaError(
                f"{active_key.episode_id} has {len(active_indices)} rows, "
                f"expected {active_len}"
            )
        pair_key = active_key.pair_key
        if pair_key in catalog:
            raise RoboTwinSchemaError(
                f"duplicate task/episode key in {embodiment}: {pair_key}"
            )
        catalog[pair_key] = EpisodeRows(
            key=active_key,
            episode_len=active_len,
            row_indices=tuple(active_indices),
            action_dim=action_dim,
        )
        closed.add(active_key)
        active_key = None
        active_indices = []
        active_len = 0
        previous_window = None

    for row_index in range(_num_rows(dataset)):
        row = dataset[row_index]
        context = f"{split}/{embodiment}[{row_index}]"
        if not isinstance(row, Mapping):
            raise RoboTwinSchemaError(f"{context} must be a mapping")
        missing = sorted(required - set(row))
        if missing:
            raise RoboTwinSchemaError(f"{context} is missing fields: {missing}")
        row_embodiment = row.get("embodiment")
        task = row.get("task")
        if row_embodiment != embodiment:
            raise RoboTwinSchemaError(
                f"{context}.embodiment={row_embodiment!r}, expected {embodiment!r}"
            )
        if not isinstance(task, str) or not task:
            raise RoboTwinSchemaError(f"{context}.task must be a non-empty string")
        episode_idx = _require_int(row, "episode_idx", context)
        episode_len = _require_int(row, "episode_len", context)
        timestep = _require_int(row, "t", context)
        if episode_idx < 0 or episode_len <= 0 or not 0 <= timestep < episode_len:
            raise RoboTwinSchemaError(
                f"{context} has invalid episode_idx/episode_len/t values"
            )
        key = EpisodeKey(split, task, embodiment, episode_idx)
        if key != active_key:
            finish_active()
            if key in closed:
                raise RoboTwinSchemaError(
                    f"{key.episode_id} is not stored as one contiguous row range"
                )
            if timestep != 0:
                raise RoboTwinSchemaError(
                    f"{key.episode_id} starts at t={timestep}, expected 0"
                )
            active_key = key
            active_len = episode_len
        expected_t = len(active_indices)
        if timestep != expected_t:
            raise RoboTwinSchemaError(
                f"{key.episode_id} has t={timestep}, expected {expected_t}"
            )
        if episode_len != active_len:
            raise RoboTwinSchemaError(
                f"{key.episode_id} changes episode_len within the episode"
            )

        state = np.asarray(row["state"])
        endpose = np.asarray(row["endpose"])
        window = np.asarray(row["action_window"])
        if state.dtype != np.float32 or state.shape != (action_dim,):
            raise RoboTwinSchemaError(
                f"{context}.state must be float32[{action_dim}], got "
                f"{state.dtype}{state.shape}"
            )
        if endpose.dtype != np.float32 or endpose.shape != (16,):
            raise RoboTwinSchemaError(
                f"{context}.endpose must be float32[16], got "
                f"{endpose.dtype}{endpose.shape}"
            )
        expected_window = min(contract.window_size, episode_len - timestep)
        if window.dtype != np.float32 or window.shape != (expected_window, action_dim):
            raise RoboTwinSchemaError(
                f"{context}.action_window must be float32"
                f"[{expected_window},{action_dim}], got {window.dtype}{window.shape}"
            )
        if not (
            np.isfinite(state).all()
            and np.isfinite(endpose).all()
            and np.isfinite(window).all()
        ):
            raise RoboTwinSchemaError(f"{context} contains NaN or infinity")
        if validate_action_windows:
            if not np.allclose(state, window[0], rtol=1e-5, atol=1e-6):
                raise RoboTwinSchemaError(
                    f"{context}.action_window[0] does not match state"
                )
            if previous_window is not None:
                overlap = min(len(previous_window) - 1, len(window))
                if overlap and not np.allclose(
                    previous_window[1 : overlap + 1],
                    window[:overlap],
                    rtol=1e-5,
                    atol=1e-6,
                ):
                    raise RoboTwinSchemaError(
                        f"{context}.action_window does not match the previous suffix"
                    )
        for field_name in camera_fields:
            _jpeg_bytes(row[field_name], context=f"{context}.{field_name}")
        active_indices.append(row_index)
        previous_window = window

    finish_active()
    if not catalog:
        raise RoboTwinSchemaError(f"{split}/{embodiment} contains no episodes")
    return catalog


def _decode_jpeg(payload: Any, *, context: str) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Pillow is required for RoboTwin JPEG decoding; install '.[robotwin]'"
        ) from exc
    raw = _jpeg_bytes(payload, context=context)
    try:
        with Image.open(BytesIO(raw)) as image:
            image.load()
            result = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except Exception as exc:
        raise RoboTwinSchemaError(f"cannot decode {context}: {exc}") from exc
    if result.ndim != 3 or result.shape[-1] != 3:
        raise RoboTwinSchemaError(f"{context} did not decode to HWC RGB")
    return result


def _last_start(episode_len: int, config: RoboTwinPreprocessConfig) -> int:
    span = (
        (config.preprocess.num_frames - 1)
        * config.mds_index_fps
        / config.preprocess.sample_fps
    )
    return math.floor((episode_len - 1) - span + 1.0e-9)


def _sample_episode(
    dataset: RowDataset,
    episode: EpisodeRows,
    *,
    start_t: int,
    config: RoboTwinPreprocessConfig,
) -> _StreamArrays:
    last_start = _last_start(episode.episode_len, config)
    if start_t < 0 or start_t > last_start:
        raise ValueError(
            f"invalid start {start_t} for {episode.key.episode_id}; max {last_start}"
        )
    positions = start_t + (
        np.arange(config.preprocess.num_frames, dtype=np.float64)
        * config.mds_index_fps
        / config.preprocess.sample_fps
    )
    lower = np.floor(positions).astype(np.int64)
    upper = np.ceil(positions).astype(np.int64)
    nearest = np.floor(positions + 0.5).astype(np.int64)
    required_t = sorted(set(lower.tolist() + upper.tolist() + nearest.tolist()))
    rows = {
        timestep: dataset[episode.row_indices[timestep]] for timestep in required_t
    }
    camera_field = f"{config.camera}_rgb"
    frames = [
        _decode_jpeg(
            rows[int(timestep)][camera_field],
            context=f"{episode.key.episode_id}/t{int(timestep)}/{camera_field}",
        )
        for timestep in nearest
    ]
    video = resize_center_crop(
        np.stack(frames),
        height=config.preprocess.height,
        width=config.preprocess.width,
    )
    lower_values = np.stack(
        [np.asarray(rows[int(timestep)]["state"], dtype=np.float32) for timestep in lower]
    )
    if config.preprocess.action_resample == "nearest":
        action_values = np.stack(
            [
                np.asarray(rows[int(timestep)]["state"], dtype=np.float32)
                for timestep in nearest
            ]
        )
    else:
        upper_values = np.stack(
            [
                np.asarray(rows[int(timestep)]["state"], dtype=np.float32)
                for timestep in upper
            ]
        )
        alpha = (positions - lower).astype(np.float32)[:, None]
        action_values = lower_values * (1.0 - alpha) + upper_values * alpha
    output_dim = config.preprocess.action_dim
    if episode.action_dim > output_dim:
        if not config.preprocess.truncate_actions:
            raise ValueError(
                f"{episode.key.episode_id} action D={episode.action_dim} exceeds "
                f"configured D={output_dim}"
            )
        action_values = action_values[:, :output_dim]
        real_dim = output_dim
    else:
        real_dim = episode.action_dim
    actions = np.zeros((config.preprocess.num_frames, output_dim), dtype=np.float32)
    actions[:, :real_dim] = action_values[:, :real_dim]
    action_mask = np.zeros_like(actions, dtype=np.bool_)
    action_mask[:, :real_dim] = True
    return _StreamArrays(
        video=np.ascontiguousarray(video, dtype=np.uint8),
        actions=actions,
        action_mask=action_mask,
        frame_mask=np.ones(config.preprocess.num_frames, dtype=np.bool_),
        clip_start=start_t / config.mds_index_fps,
    )


def _arrays_for_role(prefix: str, stream: _StreamArrays) -> dict[str, np.ndarray]:
    return {
        f"{prefix}_video": stream.video,
        f"{prefix}_actions": stream.actions,
        f"{prefix}_action_mask": stream.action_mask,
        f"{prefix}_frame_mask": stream.frame_mask,
    }


def _manifest_stream(
    episode: EpisodeRows,
    stream: _StreamArrays,
    *,
    config: RoboTwinPreprocessConfig,
) -> dict[str, Any]:
    return {
        "content_sha256": stream_content_sha256(
            video=stream.video,
            actions=stream.actions,
            action_mask=stream.action_mask,
            frame_mask=stream.frame_mask,
        ),
        "episode_id": episode.key.episode_id,
        "embodiment": episode.key.embodiment,
        "clip_start": stream.clip_start,
        "metadata": {
            "action_alignment": config.action_alignment,
            "action_signal": config.action_signal,
            "camera": config.camera,
            "episode_idx": episode.key.episode_idx,
            "mds_index_fps": config.mds_index_fps,
            "raw_action_dim": episode.action_dim,
            "split": episode.key.split,
            "task": episode.key.task,
        },
    }


def _reference_episode(
    candidates: Sequence[EpisodeRows],
    *,
    target: EpisodeRows,
    sample_id: str,
    config: RoboTwinPreprocessConfig,
) -> tuple[EpisodeRows, int]:
    eligible = [
        episode
        for episode in candidates
        if episode.key != target.key
        and _last_start(episode.episode_len, config) >= 0
        and (
            config.reference_policy != "different_task"
            or episode.key.task != target.key.task
        )
    ]
    if not eligible:
        raise ValueError(
            f"no eligible reference for target {target.key.episode_id} under "
            f"policy {config.reference_policy}"
        )
    eligible.sort(key=lambda value: value.key)
    episode = eligible[
        stable_index(
            f"{sample_id}\0reference-episode",
            len(eligible),
            seed=config.preprocess.reference_seed,
        )
    ]
    max_start = _last_start(episode.episode_len, config)
    start = stable_index(
        f"{sample_id}\0{episode.key.episode_id}\0reference-start",
        max_start + 1,
        seed=config.preprocess.reference_seed,
    )
    return episode, start


def _clip_starts(
    source: EpisodeRows,
    target: EpisodeRows,
    config: RoboTwinPreprocessConfig,
) -> tuple[int, ...]:
    max_start = min(
        _last_start(source.episode_len, config),
        _last_start(target.episode_len, config),
    )
    if max_start < 0:
        return ()
    if config.window_policy == "episode_start":
        return (0,)
    return tuple(range(0, max_start + 1, config.clip_stride_native_frames))


def _root_manifest_sha256(dataset_root: Path | None) -> str | None:
    if dataset_root is None:
        return None
    path = dataset_root / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"RoboTwin root manifest does not exist: {path}")
    return _file_sha256(path)


def _unique_selection(
    values: Sequence[str] | None, available: tuple[str, ...], *, role: str
) -> tuple[str, ...]:
    result = available if values is None else tuple(values)
    if not result:
        raise ValueError(f"{role}_embodiments cannot be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{role}_embodiments contains duplicate names")
    return result


def _cleanup_unreferenced_samples(
    samples_dir: Path, entries: Sequence[Mapping[str, Any]]
) -> None:
    """Remove old adapter-owned NPZs only after new metadata is committed."""

    referenced_names = {Path(str(entry["npz"])).name for entry in entries}
    for candidate in samples_dir.glob("*.npz"):
        if candidate.name not in referenced_names and (
            candidate.is_file() or candidate.is_symlink()
        ):
            candidate.unlink()


def preprocess_robotwin_datasets(
    datasets: Mapping[str, RowDataset],
    output_dir: str | Path,
    *,
    contract: RoboTwinContract,
    split: str,
    config: RoboTwinPreprocessConfig,
    source_embodiments: Sequence[str] | None = None,
    target_embodiments: Sequence[str] | None = None,
    dataset_root: str | Path | None = None,
    max_samples: int | None = None,
    require_bidirectional: bool = False,
    overwrite: bool = False,
) -> PreprocessReport:
    """Export directed cross-embodiment pairs from random-access MDS readers."""

    if split not in {"train", "val"}:
        raise ValueError("split must be train or val")
    if config.camera not in contract.cameras:
        raise ValueError(
            f"camera {config.camera!r} is not in contract cameras {contract.cameras}"
        )
    if config.preprocess.action_dim < max(contract.action_dims.values()):
        raise ValueError("configured action_dim is smaller than a RoboTwin action width")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("max_samples must be positive or omitted")
    available = tuple(contract.action_dims)
    sources = _unique_selection(source_embodiments, available, role="source")
    targets = _unique_selection(target_embodiments, available, role="target")
    if require_bidirectional:
        expected = set(available)
        if set(sources) != expected or set(targets) != expected:
            raise ValueError(
                "require_bidirectional needs every contract embodiment in both "
                "source and target selections"
            )
        if config.reference_policy != "different_task":
            raise ValueError(
                "require_bidirectional requires reference_policy='different_task'"
            )
        if max_samples is not None:
            raise ValueError(
                "require_bidirectional cannot be combined with max_samples; the global "
                "cap stops in direction order and is only safe for smoke tests"
            )
    requested = set(sources) | set(targets)
    unknown = sorted(requested - set(available))
    if unknown:
        raise ValueError(f"unknown embodiments: {unknown}")
    missing_datasets = sorted(requested - set(datasets))
    if missing_datasets:
        raise ValueError(f"missing MDS readers for embodiments: {missing_datasets}")
    directed_pairs = [
        (source, target)
        for source in sources
        for target in targets
        if source != target
    ]
    if not directed_pairs:
        raise ValueError("source/target selection produces no cross-embodiment pairs")

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

    expected_split: Mapping[str, Mapping[str, int]] | None = None
    if contract.expected_counts:
        expected_split = contract.expected_counts.get(split)
        if not expected_split:
            raise RoboTwinSchemaError(f"contract has no expected counts for {split}")
        missing_expected = sorted(requested - set(expected_split))
        if missing_expected:
            raise RoboTwinSchemaError(
                f"contract has no expected counts for {split}: {missing_expected}"
            )
        for embodiment in sorted(requested):
            observed_samples = _num_rows(datasets[embodiment])
            expected_samples = expected_split[embodiment]["samples"]
            if observed_samples != expected_samples:
                raise RoboTwinSchemaError(
                    f"{split}/{embodiment} sample count {observed_samples} differs "
                    f"from approved count {expected_samples}"
                )

    catalogs = {
        embodiment: build_episode_catalog(
            datasets[embodiment],
            contract=contract,
            split=split,
            embodiment=embodiment,
            validate_action_windows=config.validate_action_windows,
        )
        for embodiment in sorted(requested)
    }
    if expected_split is not None:
        for embodiment, catalog in catalogs.items():
            expected = expected_split[embodiment]
            observed = {
                "samples": _num_rows(datasets[embodiment]),
                "episodes": len(catalog),
                "tasks": len({episode.key.task for episode in catalog.values()}),
            }
            if observed != dict(expected):
                raise RoboTwinSchemaError(
                    f"{split}/{embodiment} counts {observed} differ from approved "
                    f"counts {dict(expected)}"
                )
    references = {
        embodiment: sorted(catalog.values(), key=lambda value: value.key)
        for embodiment, catalog in catalogs.items()
    }
    action_stats = {
        role: _ActionStats(config.preprocess.action_dim)
        for role in ("source", "target", "reference")
    }
    entries: list[dict[str, Any]] = []
    pair_stats: dict[str, Any] = {}
    dropped_short = 0
    planned_samples = 0
    for source_name, target_name in directed_pairs:
        source_catalog = catalogs[source_name]
        target_catalog = catalogs[target_name]
        common = sorted(set(source_catalog) & set(target_catalog))
        pair_name = f"{source_name}->{target_name}"
        pair_dropped = 0
        pair_planned = 0
        for pair_key in common:
            starts = _clip_starts(
                source_catalog[pair_key], target_catalog[pair_key], config
            )
            if starts:
                pair_planned += len(starts)
            else:
                pair_dropped += 1
        planned_samples += pair_planned
        dropped_short += pair_dropped
        pair_stats[pair_name] = {
            "common_episodes": len(common),
            "dropped_short_episodes": pair_dropped,
            "planned_samples": pair_planned,
            "source_only_episodes": len(set(source_catalog) - set(target_catalog)),
            "target_only_episodes": len(set(target_catalog) - set(source_catalog)),
            "written_samples": 0,
        }
    stop = False

    for source_name, target_name in directed_pairs:
        source_catalog = catalogs[source_name]
        target_catalog = catalogs[target_name]
        common = sorted(set(source_catalog) & set(target_catalog))
        pair_name = f"{source_name}->{target_name}"
        for pair_key in common:
            source_episode = source_catalog[pair_key]
            target_episode = target_catalog[pair_key]
            starts = _clip_starts(source_episode, target_episode, config)
            if not starts:
                continue
            for start_t in starts:
                sample_id = (
                    f"robotwin-v1:{split}:{pair_key[0]}:episode{pair_key[1]}:"
                    f"{source_name}->{target_name}:t{start_t}"
                )
                reference_episode, reference_start = _reference_episode(
                    references[target_name],
                    target=target_episode,
                    sample_id=sample_id,
                    config=config,
                )
                source_stream = _sample_episode(
                    datasets[source_name],
                    source_episode,
                    start_t=start_t,
                    config=config,
                )
                target_stream = _sample_episode(
                    datasets[target_name],
                    target_episode,
                    start_t=start_t,
                    config=config,
                )
                reference_stream = _sample_episode(
                    datasets[target_name],
                    reference_episode,
                    start_t=reference_start,
                    config=config,
                )
                arrays: dict[str, np.ndarray] = {}
                arrays.update(_arrays_for_role("source", source_stream))
                arrays.update(_arrays_for_role("target", target_stream))
                arrays.update(_arrays_for_role("reference", reference_stream))
                relative_npz = Path("samples") / _sample_filename(sample_id)
                npz_path = output / relative_npz
                if npz_path.exists() and not overwrite:
                    raise FileExistsError(f"processed sample already exists: {npz_path}")
                _atomic_savez(npz_path, arrays)
                entry = {
                    "format_version": PROCESSED_FORMAT_VERSION,
                    "id": sample_id,
                    "npz": relative_npz.as_posix(),
                    "source": _manifest_stream(
                        source_episode, source_stream, config=config
                    ),
                    "target_gt": _manifest_stream(
                        target_episode, target_stream, config=config
                    ),
                    "reference_target": _manifest_stream(
                        reference_episode, reference_stream, config=config
                    ),
                    "shape": {
                        "video": list(source_stream.video.shape),
                        "actions": list(source_stream.actions.shape),
                    },
                    "metadata": {
                        "adapter_version": ROBOTWIN_ADAPTER_VERSION,
                        "base_window_identity": (
                            f"robotwin-v1:{split}:{pair_key[0]}:episode{pair_key[1]}:"
                            f"t{start_t}"
                        ),
                        "camera": config.camera,
                        "direction": pair_name,
                        "pair_identity": (
                            f"robotwin-v1:{split}:{pair_key[0]}:episode{pair_key[1]}:"
                            f"t{start_t}:{min(source_name, target_name)}<>"
                            f"{max(source_name, target_name)}"
                        ),
                        "pairing_policy": "same_split_task_episode_idx",
                        "reference_policy": config.reference_policy,
                        "split": split,
                        "task_id": pair_key[0],
                    },
                }
                entries.append(entry)
                for role, stream in (
                    ("source", source_stream),
                    ("target", target_stream),
                    ("reference", reference_stream),
                ):
                    action_stats[role].update(stream.actions, stream.action_mask)
                pair_stats[pair_name]["written_samples"] += 1
                if max_samples is not None and len(entries) >= max_samples:
                    stop = True
                    break
            if stop:
                break
        if stop:
            break

    manifest_text = "".join(
        json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n"
        for entry in entries
    )
    _atomic_text(manifest_path, manifest_text)
    by_source: dict[str, list[str]] = {}
    by_target: dict[str, list[str]] = {}
    by_direction: dict[str, list[str]] = {}
    by_id: dict[str, dict[str, Any]] = {}
    for line_number, entry in enumerate(entries):
        sample_id = entry["id"]
        source = entry["source"]["embodiment"]
        target = entry["target_gt"]["embodiment"]
        direction = entry["metadata"]["direction"]
        by_source.setdefault(source, []).append(sample_id)
        by_target.setdefault(target, []).append(sample_id)
        by_direction.setdefault(direction, []).append(sample_id)
        by_id[sample_id] = {"line": line_number, "npz": entry["npz"]}
    direction_summary = (
        require_bidirectional_pairs(
            entries,
            expected_embodiments=available,
        )
        if require_bidirectional
        else summarize_pair_directions(entries)
    )
    index = {
        "format_version": PROCESSED_FORMAT_VERSION,
        "manifest": manifest_path.name,
        "num_samples": len(entries),
        "by_id": by_id,
        "by_pair_direction": {
            key: value for key, value in sorted(by_direction.items())
        },
        "by_source_embodiment": {
            key: value for key, value in sorted(by_source.items())
        },
        "by_target_embodiment": {
            key: value for key, value in sorted(by_target.items())
        },
    }
    _atomic_text(
        index_path,
        json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    root_path = (
        Path(dataset_root).expanduser().resolve() if dataset_root is not None else None
    )
    source_set = set(sources)
    target_set = set(targets)
    contract_set = set(available)
    if source_set == contract_set and target_set == contract_set:
        direction_policy = "all_contract_ordered_distinct"
    elif source_set == target_set:
        direction_policy = "selected_ordered_distinct"
    else:
        direction_policy = "selected_directed_cross_product"
    truncated = len(entries) < planned_samples
    stats = {
        "format_version": PROCESSED_FORMAT_VERSION,
        "adapter_version": ROBOTWIN_ADAPTER_VERSION,
        "config": {
            **asdict(config),
            "preprocess": asdict(config.preprocess),
        },
        "lineage": {
            "contract_file_sha256": contract.file_sha256,
            "observed_schema_sha256": contract.observed_schema_sha256,
            "root_manifest_sha256": _root_manifest_sha256(root_path),
            "dataset_root": str(root_path) if root_path is not None else None,
            "split": split,
            "streaming_version": contract.streaming_version,
        },
        "catalog_episodes": {
            key: len(value) for key, value in sorted(catalogs.items())
        },
        "pair_directions": pair_stats,
        "pairing": {
            "direction_policy": direction_policy,
            "source_embodiments": list(sources),
            "target_embodiments": list(targets),
            "require_bidirectional": require_bidirectional,
            "planned_samples": planned_samples,
            "truncated": truncated,
            "direction_summary": direction_summary,
        },
        "written_samples": len(entries),
        "dropped_short_episodes": dropped_short,
        "max_samples_applied": max_samples,
        "actions": {role: value.as_dict() for role, value in action_stats.items()},
    }
    _atomic_text(
        stats_path,
        json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    if overwrite:
        _cleanup_unreferenced_samples(output / "samples", entries)
    return PreprocessReport(
        manifest=manifest_path,
        index=index_path,
        stats=stats_path,
        written=len(entries),
        dropped=dropped_short,
        failed=0,
    )


def preprocess_robotwin_mds(
    dataset_root: str | Path,
    output_dir: str | Path,
    *,
    contract: RoboTwinContract,
    split: str,
    config: RoboTwinPreprocessConfig,
    source_embodiments: Sequence[str] | None = None,
    target_embodiments: Sequence[str] | None = None,
    max_samples: int | None = None,
    require_bidirectional: bool = False,
    overwrite: bool = False,
) -> PreprocessReport:
    """Open requested local MDS streams and export fixed-length GenET pairs."""

    dataset_root, _ = validate_robotwin_root(dataset_root, contract)
    available = tuple(contract.action_dims)
    sources = _unique_selection(source_embodiments, available, role="source")
    targets = _unique_selection(target_embodiments, available, role="target")
    requested = sorted(set(sources) | set(targets))
    datasets = {
        embodiment: open_robotwin_mds(
            dataset_root,
            split=split,
            embodiment=embodiment,
            expected_streaming_version=contract.streaming_version,
        )
        for embodiment in requested
    }
    return preprocess_robotwin_datasets(
        datasets,
        output_dir,
        contract=contract,
        split=split,
        config=config,
        source_embodiments=sources,
        target_embodiments=targets,
        dataset_root=dataset_root,
        max_samples=max_samples,
        require_bidirectional=require_bidirectional,
        overwrite=overwrite,
    )
