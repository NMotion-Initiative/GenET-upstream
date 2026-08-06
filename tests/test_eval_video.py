from pathlib import Path

import numpy as np
import pytest
import torch

from genet.config import load_config
from genet.training.eval_video import tensor_to_uint8_thwc, write_video_mp4


def test_stage1_ddp_enables_offline_wandb_and_eval_video() -> None:
    config = load_config(Path("configs/experiments/stage1_control_32gpu_ddp.yaml"))
    assert config.logging.wandb.mode == "offline"
    assert config.logging.wandb.project == "genet"
    assert config.logging.eval_video.enabled is True
    assert config.logging.eval_video.num_samples == 2
    assert "val/manifest.jsonl" in (config.logging.eval_video.manifest or "")


def test_invalid_wandb_mode_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("logging:\n  wandb:\n    mode: sync\n", encoding="utf-8")
    with pytest.raises(ValueError, match="logging.wandb.mode"):
        load_config(path)


def test_tensor_to_uint8_handles_cthw_minus_one_one() -> None:
    video = torch.linspace(-1, 1, 2 * 3 * 4 * 5).reshape(3, 2, 4, 5)
    frames = tensor_to_uint8_thwc(video)
    assert frames.shape == (2, 4, 5, 3)
    assert frames.dtype == np.uint8
    assert frames.min() >= 0 and frames.max() <= 255


def test_write_video_mp4_roundtrip(tmp_path: Path) -> None:
    av = pytest.importorskip("av")
    del av
    video = torch.zeros(3, 4, 16, 16)
    video[:, 1] = 1.0
    path = write_video_mp4(tmp_path / "clip.mp4", video, fps=8.0)
    assert path.is_file()
    assert path.stat().st_size > 0
