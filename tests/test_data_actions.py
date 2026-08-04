import json
from pathlib import Path

import numpy as np
import pytest
import torch

from genet.data.actions import ActionSeries, load_action_series, resample_actions
from genet.data.sampling import ShortSequenceError
from genet.data.video import DecodedVideo, resize_center_crop, sample_video


@pytest.mark.parametrize("kind", ["npy", "npz", "json", "pt"])
def test_action_loaders(kind: str, tmp_path: Path):
    values = np.arange(6, dtype=np.float32).reshape(3, 2)
    timestamps = np.array([0.0, 0.5, 1.0], dtype=np.float64)
    path = tmp_path / f"actions.{kind}"
    kwargs = {}
    if kind == "npy":
        np.save(path, values)
        kwargs = {"default_fps": 2.0}
    elif kind == "npz":
        np.savez(path, actions=values, timestamps=timestamps)
    elif kind == "json":
        path.write_text(
            json.dumps({"actions": values.tolist(), "timestamps": timestamps.tolist()}),
            encoding="utf-8",
        )
    else:
        torch.save(
            {"actions": torch.from_numpy(values), "timestamps": torch.from_numpy(timestamps)},
            path,
        )
    loaded = load_action_series(path, **kwargs)
    np.testing.assert_array_equal(loaded.values, values)
    np.testing.assert_allclose(loaded.timestamps, timestamps)


def test_linear_nearest_padding_and_short_policy():
    series = ActionSeries(
        values=np.array([[0.0, 2.0], [10.0, 4.0]], dtype=np.float32),
        timestamps=np.array([0.0, 1.0]),
    )
    grid = np.array([0.0, 0.5, 1.0])
    linear, mask = resample_actions(
        series, grid, method="linear", action_dim=4, short_policy="drop"
    )
    np.testing.assert_allclose(linear[:, 0], [0, 5, 10])
    np.testing.assert_allclose(linear[:, 1], [2, 3, 4])
    assert mask[:, :2].all()
    assert not mask[:, 2:].any()

    nearest, _ = resample_actions(
        series, grid, method="nearest", action_dim=2, short_policy="drop"
    )
    np.testing.assert_allclose(nearest[:, 0], [0, 0, 10])
    with pytest.raises(ShortSequenceError):
        resample_actions(
            series,
            np.array([0.0, 2.0]),
            method="linear",
            action_dim=2,
            short_policy="drop",
        )
    padded, padded_mask = resample_actions(
        series,
        np.array([0.0, 2.0]),
        method="linear",
        action_dim=2,
        short_policy="pad",
    )
    np.testing.assert_allclose(padded[-1], series.values[-1])
    assert not padded_mask[-1].any()


def test_video_fixed_grid_padding_and_resize_crop():
    frames = np.zeros((3, 4, 8, 3), dtype=np.uint8)
    frames[:, :, :, 0] = np.arange(3)[:, None, None]
    decoded = DecodedVideo(frames, np.array([0.0, 0.5, 1.0]))
    sampled, mask = sample_video(
        decoded, np.array([0.0, 0.5, 1.5]), short_policy="pad"
    )
    assert mask.tolist() == [True, True, False]
    assert sampled[-1, 0, 0, 0] == 2
    resized = resize_center_crop(sampled, height=4, width=4)
    assert resized.shape == (3, 4, 4, 3)


def test_array_video_npz_timestamps(tmp_path: Path):
    from genet.data.video import decode_video

    frames = np.zeros((5, 3, 4, 3), dtype=np.uint8)
    path = tmp_path / "video.npz"
    np.savez(path, frames=frames, timestamps=np.arange(5) * 0.2)
    decoded = decode_video(path)
    assert decoded.frames.shape == frames.shape
    np.testing.assert_allclose(decoded.timestamps, np.arange(5) * 0.2)
