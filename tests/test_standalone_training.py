import json
from pathlib import Path

import numpy as np
import torch

from genet.config import ProjectConfig
from genet.data.preprocess import PreprocessConfig, preprocess_manifest
from genet.training.distributed import DistributedContext
from genet.training.standalone import run_standalone_training


def _episode(root: Path, episode_id: str, embodiment: str, offset: float) -> dict:
    video = np.full((6, 16, 16, 3), int(offset) % 255, dtype=np.uint8)
    actions = np.stack(
        [np.linspace(offset, offset + 1, 6), np.linspace(0, 1, 6)], axis=1
    ).astype(np.float32)
    video_path = root / f"{episode_id}.video.npy"
    action_path = root / f"{episode_id}.actions.npy"
    np.save(video_path, video)
    np.save(action_path, actions)
    return {
        "episode_id": episode_id,
        "embodiment": embodiment,
        "video": video_path.name,
        "actions": action_path.name,
        "video_fps": 4.0,
        "action_fps": 4.0,
    }


def test_one_step_standalone_training(tmp_path: Path):
    records = [
        {
            "id": "pair-0",
            "source": _episode(tmp_path, "source-0", "franka", 10),
            "target_gt": _episode(tmp_path, "target-0", "ur5", 20),
        },
        {
            "id": "pair-1",
            "source": _episode(tmp_path, "source-1", "franka", 30),
            "target_gt": _episode(tmp_path, "target-1", "ur5", 40),
        },
    ]
    raw_manifest = tmp_path / "raw.jsonl"
    raw_manifest.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    report = preprocess_manifest(
        raw_manifest,
        tmp_path / "processed",
        config=PreprocessConfig(
            num_frames=5,
            sample_fps=4,
            height=16,
            width=16,
            action_dim=4,
        ),
    )

    config = ProjectConfig()
    config.data.manifest = str(report.manifest)
    config.data.num_frames = 5
    config.data.reference_num_frames = 5
    config.data.height = 16
    config.data.width = 16
    config.data.action_dim = 4
    config.loader.num_workers = 0
    config.loader.drop_last = False
    config.loader.pin_memory = False
    config.model.dtype = "float32"
    config.model.hidden_size = 16
    config.model.num_layers = 1
    config.model.num_heads = 4
    config.model.latent_channels = 4
    config.model.patch_size = 1
    config.model.num_embodiments = 4
    config.model.reference.num_heads = 4
    config.train.max_steps = 1
    config.train.warmup_steps = 0
    config.train.log_every = 1
    config.train.stage = "control"
    config.checkpoint.output_dir = str(tmp_path / "output")
    config.checkpoint.save_every = 1
    config.validate(world_size=1)

    run_standalone_training(
        config,
        DistributedContext(
            rank=0, local_rank=0, world_size=1, device=torch.device("cpu")
        ),
    )
    checkpoint = tmp_path / "output" / "step_000000001"
    assert (checkpoint / "COMMITTED").is_file()
    assert (checkpoint / "model.pt").is_file()
