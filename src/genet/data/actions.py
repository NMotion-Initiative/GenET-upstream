"""Action trajectory loading, temporal resampling, and dimension padding."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch

from .sampling import ShortSequenceError, coverage_mask, nearest_indices


@dataclass(frozen=True)
class ActionSeries:
    values: np.ndarray
    timestamps: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        timestamps = np.asarray(self.timestamps)
        if values.ndim != 2:
            raise ValueError(f"actions must have shape [time, dim], got {values.shape}")
        if timestamps.ndim != 1 or len(timestamps) != len(values):
            raise ValueError("action timestamps must have one value per action step")


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # torch < 2.0
        return torch.load(path, map_location="cpu")


def _to_numpy(value: Any, *, context: str) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    try:
        return np.asarray(value)
    except Exception as exc:  # pragma: no cover - defensive adapter boundary
        raise ValueError(f"{context} cannot be converted to an array") from exc


def _select_mapping_value(
    mapping: Mapping[str, Any],
    *,
    requested_key: Optional[str],
    preferred_keys: tuple[str, ...],
    excluded_keys: tuple[str, ...] = (),
    context: str,
) -> Any:
    if requested_key is not None:
        if requested_key not in mapping:
            raise KeyError(f"{context} has no key {requested_key!r}")
        return mapping[requested_key]
    for key in preferred_keys:
        if key in mapping:
            return mapping[key]
    usable = [key for key in mapping if key not in excluded_keys]
    if len(usable) == 1:
        return mapping[usable[0]]
    raise KeyError(
        f"{context} needs an explicit key; available keys: {sorted(mapping)}"
    )


def _load_container(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return {key: archive[key] for key in archive.files}
    if suffix == ".json":
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    if suffix in {".pt", ".pth"}:
        return _torch_load(path)
    raise ValueError(
        f"unsupported action file {path}; expected .npy, .npz, .json, .pt, or .pth"
    )


def load_action_series(
    path: str | Path,
    *,
    action_key: Optional[str] = None,
    timestamp_key: Optional[str] = None,
    timestamps_path: str | Path | None = None,
    default_fps: Optional[float] = None,
    start_time: float = 0.0,
) -> ActionSeries:
    """Load actions and timestamps from common robotics interchange formats.

    JSON/NPZ/PT mappings may contain ``actions`` (or ``data``/``values``) and an
    optional ``timestamps`` array.  A separate timestamp file takes precedence.
    """

    action_path = Path(path)
    if not action_path.is_file():
        raise FileNotFoundError(f"action file does not exist: {action_path}")
    container = _load_container(action_path)
    embedded_timestamps: Any = None
    if isinstance(container, Mapping):
        values_raw = _select_mapping_value(
            container,
            requested_key=action_key,
            preferred_keys=("actions", "action", "data", "values"),
            excluded_keys=("timestamps", "time", "t"),
            context=str(action_path),
        )
        if timestamp_key is not None:
            if timestamp_key not in container:
                raise KeyError(f"{action_path} has no timestamp key {timestamp_key!r}")
            embedded_timestamps = container[timestamp_key]
        else:
            for key in ("timestamps", "time", "t"):
                if key in container:
                    embedded_timestamps = container[key]
                    break
    else:
        if action_key is not None:
            raise KeyError(f"action_key is only valid for mapping/NPZ action files")
        values_raw = container

    values = _to_numpy(values_raw, context=str(action_path))
    if values.ndim == 1:
        values = values[:, None]
    elif values.ndim > 2:
        values = values.reshape(values.shape[0], -1)
    if values.ndim != 2 or not len(values) or values.shape[1] == 0:
        raise ValueError(f"{action_path} contains an empty or invalid action array")
    values = np.asarray(values, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"{action_path} contains NaN or infinite actions")

    if timestamps_path is not None:
        timestamp_container = _load_container(Path(timestamps_path))
        if isinstance(timestamp_container, Mapping):
            timestamps_raw = _select_mapping_value(
                timestamp_container,
                requested_key=timestamp_key,
                preferred_keys=("timestamps", "time", "t", "data", "values"),
                context=str(timestamps_path),
            )
        else:
            timestamps_raw = timestamp_container
    else:
        timestamps_raw = embedded_timestamps

    if timestamps_raw is None:
        if default_fps is None or default_fps <= 0:
            raise ValueError(
                f"{action_path} has no timestamps; provide a positive action_fps"
            )
        timestamps = start_time + np.arange(len(values), dtype=np.float64) / float(
            default_fps
        )
    else:
        timestamps = _to_numpy(timestamps_raw, context="action timestamps")
        timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    if len(timestamps) != len(values):
        raise ValueError(
            f"action timestamp count {len(timestamps)} != action count {len(values)}"
        )
    if not np.isfinite(timestamps).all():
        raise ValueError("action timestamps contain NaN or infinity")
    if len(timestamps) > 1 and not np.all(np.diff(timestamps) > 0):
        raise ValueError("action timestamps must be strictly increasing")
    return ActionSeries(values=values, timestamps=timestamps)


def resample_actions(
    series: ActionSeries,
    time_grid: np.ndarray,
    *,
    method: str = "linear",
    action_dim: int,
    short_policy: str = "drop",
    truncate: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample to a fixed time grid and pad to ``action_dim``.

    The returned mask has shape ``[time, action_dim]``.  It is false for padded
    dimensions and for time steps outside the observed action interval.
    """

    if method not in {"linear", "nearest"}:
        raise ValueError("action resampling method must be 'linear' or 'nearest'")
    if short_policy not in {"drop", "pad"}:
        raise ValueError("short_policy must be 'drop' or 'pad'")
    if action_dim <= 0:
        raise ValueError("action_dim must be positive")
    query = np.asarray(time_grid, dtype=np.float64)
    if query.ndim != 1 or not len(query):
        raise ValueError("time_grid must be a non-empty 1D array")

    source_dim = series.values.shape[1]
    if source_dim > action_dim and not truncate:
        raise ValueError(
            f"source action dim {source_dim} exceeds configured action_dim {action_dim}"
        )
    used_dim = min(source_dim, action_dim)
    valid_time = coverage_mask(series.timestamps, query)
    if short_policy == "drop" and not bool(valid_time.all()):
        raise ShortSequenceError(
            "action trajectory does not cover the requested fixed time grid"
        )

    sampled = np.empty((len(query), used_dim), dtype=np.float32)
    if method == "nearest":
        indices = nearest_indices(series.timestamps, query)
        sampled[...] = series.values[indices, :used_dim]
    else:
        for dimension in range(used_dim):
            sampled[:, dimension] = np.interp(
                query,
                series.timestamps,
                series.values[:, dimension],
            ).astype(np.float32, copy=False)

    output = np.zeros((len(query), action_dim), dtype=np.float32)
    output[:, :used_dim] = sampled
    mask = np.zeros((len(query), action_dim), dtype=np.bool_)
    mask[:, :used_dim] = valid_time[:, None]
    return output, mask
