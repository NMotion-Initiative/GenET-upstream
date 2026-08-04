"""Optional-PyAV video decoding and fixed-grid spatial/temporal sampling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from .sampling import ShortSequenceError, coverage_mask, nearest_indices


class VideoDecodeError(RuntimeError):
    pass


@dataclass(frozen=True)
class DecodedVideo:
    """Decoded RGB frames in ``[time, height, width, channel]`` layout."""

    frames: np.ndarray
    timestamps: np.ndarray

    def __post_init__(self) -> None:
        frames = np.asarray(self.frames)
        timestamps = np.asarray(self.timestamps)
        if frames.ndim != 4 or frames.shape[-1] not in {1, 3, 4}:
            raise ValueError(f"video frames must be THWC, got {frames.shape}")
        if timestamps.ndim != 1 or len(timestamps) != len(frames):
            raise ValueError("video timestamps must have one value per frame")


def _normalise_array_layout(frames: np.ndarray, layout: str = "auto") -> np.ndarray:
    frames = np.asarray(frames)
    if frames.ndim != 4:
        raise ValueError(f"array video must be rank 4, got {frames.shape}")
    layout = layout.upper()
    if layout == "AUTO":
        if frames.shape[-1] in {1, 3, 4}:
            layout = "THWC"
        elif frames.shape[1] in {1, 3, 4}:
            layout = "TCHW"
        elif frames.shape[0] in {1, 3, 4}:
            layout = "CTHW"
        else:
            raise ValueError(
                f"cannot infer array-video layout for {frames.shape}; set video_layout"
            )
    if layout == "THWC":
        result = frames
    elif layout == "TCHW":
        result = np.transpose(frames, (0, 2, 3, 1))
    elif layout == "CTHW":
        result = np.transpose(frames, (1, 2, 3, 0))
    else:
        raise ValueError("video_layout must be auto, THWC, TCHW, or CTHW")
    if result.shape[-1] == 4:
        result = result[..., :3]
    elif result.shape[-1] == 1:
        result = np.repeat(result, 3, axis=-1)
    if np.issubdtype(result.dtype, np.floating):
        if not np.isfinite(result).all():
            raise ValueError("array video contains NaN or infinite pixels")
        maximum = float(np.max(result)) if result.size else 0.0
        if maximum <= 1.0:
            result = result * 255.0
    return np.clip(result, 0, 255).astype(np.uint8, copy=False)


def _decode_array_video(
    path: Path,
    *,
    fps: Optional[float],
    video_key: Optional[str],
    layout: str,
) -> DecodedVideo:
    if path.suffix.lower() == ".npy":
        frames = np.load(path, allow_pickle=False)
        timestamps = None
    else:
        with np.load(path, allow_pickle=False) as archive:
            if video_key is not None:
                if video_key not in archive:
                    raise KeyError(f"{path} has no video key {video_key!r}")
                frames = archive[video_key]
            else:
                keys = [key for key in ("video", "frames", "rgb") if key in archive]
                if keys:
                    frames = archive[keys[0]]
                else:
                    candidates = [key for key in archive.files if key != "timestamps"]
                    if len(candidates) != 1:
                        raise KeyError(
                            f"{path} needs video_key; available keys: {archive.files}"
                        )
                    frames = archive[candidates[0]]
            timestamps = archive["timestamps"] if "timestamps" in archive else None
    frames = _normalise_array_layout(frames, layout)
    if timestamps is None:
        if fps is None or fps <= 0:
            raise ValueError(f"{path} has no timestamps; provide a positive video_fps")
        timestamps = np.arange(len(frames), dtype=np.float64) / float(fps)
    else:
        timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
    return _validated_decoded(frames, timestamps, path)


def _validated_decoded(
    frames: np.ndarray, timestamps: np.ndarray, path: Path
) -> DecodedVideo:
    if not len(frames):
        raise VideoDecodeError(f"video contains no frames: {path}")
    if len(timestamps) != len(frames):
        raise VideoDecodeError(f"timestamp/frame count mismatch in {path}")
    if not np.isfinite(timestamps).all():
        raise VideoDecodeError(f"non-finite video timestamps in {path}")
    if len(timestamps) > 1 and not np.all(np.diff(timestamps) > 0):
        raise VideoDecodeError(f"video timestamps are not strictly increasing: {path}")
    return DecodedVideo(frames=frames, timestamps=timestamps)


def _decode_with_pyav(path: Path) -> DecodedVideo:
    try:
        import av  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "PyAV is required for encoded video files. Install the optional 'preprocess' "
            "dependencies, or preprocess array videos from .npy/.npz."
        ) from exc

    frames: list[np.ndarray] = []
    timestamps: list[float] = []
    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise VideoDecodeError(f"no video stream in {path}")
            stream = container.streams.video[0]
            fallback_fps = float(stream.average_rate) if stream.average_rate else None
            for index, frame in enumerate(container.decode(stream)):
                frames.append(frame.to_ndarray(format="rgb24"))
                if frame.pts is not None and frame.time_base is not None:
                    timestamps.append(float(frame.pts * frame.time_base))
                elif frame.time is not None:
                    timestamps.append(float(frame.time))
                elif fallback_fps:
                    timestamps.append(index / fallback_fps)
                else:
                    raise VideoDecodeError(
                        f"frame timestamps and average frame rate are unavailable in {path}"
                    )
    except VideoDecodeError:
        raise
    except Exception as exc:
        raise VideoDecodeError(f"failed to decode {path}: {exc}") from exc
    if not frames:
        raise VideoDecodeError(f"video contains no decoded frames: {path}")
    return _validated_decoded(
        np.stack(frames).astype(np.uint8, copy=False),
        np.asarray(timestamps, dtype=np.float64),
        path,
    )


def decode_video(
    path: str | Path,
    *,
    fps: Optional[float] = None,
    video_key: Optional[str] = None,
    layout: str = "auto",
) -> DecodedVideo:
    """Decode an array video directly or an encoded video through optional PyAV."""

    video_path = Path(path)
    if not video_path.is_file():
        raise FileNotFoundError(f"video file does not exist: {video_path}")
    if video_path.suffix.lower() in {".npy", ".npz"}:
        return _decode_array_video(
            video_path, fps=fps, video_key=video_key, layout=layout
        )
    return _decode_with_pyav(video_path)


def sample_video(
    decoded: DecodedVideo,
    time_grid: np.ndarray,
    *,
    short_policy: str = "drop",
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-sample RGB frames at exact grid times.

    With ``pad``, boundary frames are repeated and the returned frame mask records
    which time steps were truly covered.  With ``drop``, insufficient coverage
    raises :class:`ShortSequenceError` for the caller to drop the whole pair.
    """

    if short_policy not in {"drop", "pad"}:
        raise ValueError("short_policy must be 'drop' or 'pad'")
    query = np.asarray(time_grid, dtype=np.float64)
    valid = coverage_mask(decoded.timestamps, query)
    if short_policy == "drop" and not bool(valid.all()):
        raise ShortSequenceError("video does not cover the requested fixed time grid")
    indices = nearest_indices(decoded.timestamps, query)
    return decoded.frames[indices], valid.astype(np.bool_, copy=False)


def resize_center_crop(
    frames: np.ndarray, *, height: int, width: int
) -> np.ndarray:
    """Aspect-preserving resize followed by a centered spatial crop."""

    if height <= 0 or width <= 0:
        raise ValueError("height and width must be positive")
    frames = _normalise_array_layout(frames, "THWC")
    source_h, source_w = frames.shape[1:3]
    if source_h == height and source_w == width:
        return np.ascontiguousarray(frames)
    scale = max(height / source_h, width / source_w)
    resized_h = max(height, int(round(source_h * scale)))
    resized_w = max(width, int(round(source_w * scale)))
    tensor = torch.from_numpy(np.ascontiguousarray(frames)).permute(0, 3, 1, 2)
    resized = F.interpolate(
        tensor.float(),
        size=(resized_h, resized_w),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    top = (resized_h - height) // 2
    left = (resized_w - width) // 2
    cropped = resized[:, :, top : top + height, left : left + width]
    return (
        cropped.round()
        .clamp_(0, 255)
        .to(torch.uint8)
        .permute(0, 2, 3, 1)
        .contiguous()
        .cpu()
        .numpy()
    )
