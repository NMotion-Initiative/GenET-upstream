"""Shared fixed-time-grid sampling utilities."""

from __future__ import annotations

import numpy as np


class ShortSequenceError(ValueError):
    """A stream does not cover the requested fixed-length time grid."""


def make_time_grid(*, start_time: float, num_frames: int, fps: float) -> np.ndarray:
    if start_time < 0:
        raise ValueError("start_time must be >= 0")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if fps <= 0:
        raise ValueError("fps must be positive")
    return start_time + np.arange(num_frames, dtype=np.float64) / float(fps)


def nearest_indices(timestamps: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Return closest monotonic timestamp index for every query timestamp."""

    timestamps = np.asarray(timestamps, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    if timestamps.ndim != 1 or not len(timestamps):
        raise ValueError("timestamps must be a non-empty 1D array")
    right = np.searchsorted(timestamps, query, side="left")
    right = np.clip(right, 0, len(timestamps) - 1)
    left = np.clip(right - 1, 0, len(timestamps) - 1)
    choose_left = np.abs(query - timestamps[left]) <= np.abs(timestamps[right] - query)
    return np.where(choose_left, left, right).astype(np.int64, copy=False)


def coverage_mask(timestamps: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Mark queries covered by the observed timestamp interval."""

    timestamps = np.asarray(timestamps, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    if timestamps.ndim != 1 or not len(timestamps):
        raise ValueError("timestamps must be a non-empty 1D array")
    tolerance = max(np.finfo(np.float64).eps * 16, 1e-9)
    return (query >= timestamps[0] - tolerance) & (
        query <= timestamps[-1] + tolerance
    )
