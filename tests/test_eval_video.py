from pathlib import Path

import numpy as np
import pytest
import torch

from genet.config import load_config
from genet.training.cosmos import _cosmos_launch_args, _remove_unsupported_callbacks
from genet.training.eval_video import tensor_to_uint8_thwc, write_video_mp4


def test_stage1_ddp_enables_online_wandb_and_eval_video() -> None:
    config = load_config(Path("configs/experiments/stage1_control_32gpu_ddp.yaml"))
    assert config.logging.wandb.mode == "online"
    assert config.logging.wandb.project == "genet"
    assert config.logging.eval_video.enabled is True
    assert config.logging.eval_video.num_samples == 2
    assert "val/manifest.jsonl" in (config.logging.eval_video.manifest or "")


def test_invalid_wandb_mode_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("logging:\n  wandb:\n    mode: sync\n", encoding="utf-8")
    with pytest.raises(ValueError, match="logging.wandb.mode"):
        load_config(path)


def test_remove_unsupported_callbacks_preserves_struct_mode() -> None:
    omegaconf = pytest.importorskip("omegaconf")
    callbacks = omegaconf.OmegaConf.create(
        {
            "every_n_sample_reg": {"enabled": True},
            "every_n_sample_ema": {"enabled": True},
            "dataloader_speed": {"every_n": 100},
            "training_stats": {"enabled": True},
            "ofu": {"every_n": 10},
            "device_monitor": {"every_n": 200},
        }
    )
    omegaconf.OmegaConf.set_struct(callbacks, True)

    _remove_unsupported_callbacks(callbacks)

    assert "every_n_sample_reg" not in callbacks
    assert "every_n_sample_ema" not in callbacks
    assert "dataloader_speed" not in callbacks
    assert "training_stats" not in callbacks
    assert "ofu" not in callbacks
    assert "device_monitor" in callbacks
    assert omegaconf.OmegaConf.is_struct(callbacks) is True


def test_cosmos_launch_args_match_upstream_logging_contract() -> None:
    args = _cosmos_launch_args()
    assert args.config == "genet:cosmos3_edge"
    assert args.opts == []
    assert args.attach_vscode_debugger is False
    assert args.deterministic is False


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
