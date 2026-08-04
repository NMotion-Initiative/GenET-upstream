import json
from pathlib import Path

import numpy as np
import pytest
import torch

from genet.data.dataset import ProcessedPairDataset, collate_pairs
from genet.data.preprocess import PreprocessConfig, _process_stream, preprocess_manifest
from genet.data.schema import EpisodeRef
from genet.data.video import decode_video


def _write_episode(root: Path, episode_id: str, color: int, action_base: float) -> dict:
    video = np.full((9, 6, 10, 3), color, dtype=np.uint8)
    actions = np.stack(
        [
            np.linspace(action_base, action_base + 8, 9),
            np.linspace(-action_base, -action_base - 8, 9),
        ],
        axis=1,
    ).astype(np.float32)
    video_path = root / f"{episode_id}.video.npy"
    action_path = root / f"{episode_id}.actions.npy"
    np.save(video_path, video)
    np.save(action_path, actions)
    return {
        "episode_id": episode_id,
        "embodiment": "ur5" if episode_id.startswith("t") else "franka",
        "video": video_path.name,
        "actions": action_path.name,
        "video_fps": 2.0,
        "action_fps": 2.0,
    }


def _build_raw_manifest(tmp_path: Path) -> Path:
    source0 = _write_episode(tmp_path, "s0", 10, 0)
    source1 = _write_episode(tmp_path, "s1", 20, 10)
    target0 = _write_episode(tmp_path, "t0", 100, 20)
    target1 = _write_episode(tmp_path, "t1", 200, 30)
    records = [
        {"id": "pair-0", "source": source0, "target_gt": target0},
        {"id": "pair-1", "source": source1, "target_gt": target1},
    ]
    manifest = tmp_path / "raw.jsonl"
    manifest.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    return manifest


def test_preprocess_outputs_and_dataset_formats(tmp_path: Path):
    raw = _build_raw_manifest(tmp_path)
    output = tmp_path / "processed"
    config = PreprocessConfig(
        num_frames=5,
        sample_fps=2.0,
        height=4,
        width=4,
        action_dim=4,
        reference_seed=123,
        short_policy="drop",
    )
    report = preprocess_manifest(raw, output, config=config)
    assert report.written == 2
    assert report.dropped == report.failed == 0
    assert report.manifest.is_file()
    assert report.index.is_file()
    assert report.stats.is_file()

    entries = [json.loads(line) for line in report.manifest.read_text().splitlines()]
    assert len(entries) == 2
    assert entries[0]["reference_target"]["episode_id"] != entries[0]["target_gt"][
        "episode_id"
    ]
    sample_path = output / entries[0]["npz"]
    with np.load(sample_path, allow_pickle=False) as sample:
        assert sample["source_video"].shape == (5, 4, 4, 3)
        assert sample["target_actions"].shape == (5, 4)
        assert sample["target_action_mask"][:, :2].all()
        assert not sample["target_action_mask"][:, 2:].any()

    stats = json.loads(report.stats.read_text())
    index = json.loads(report.index.read_text())
    assert stats["written_samples"] == 2
    assert index["num_samples"] == 2
    assert index["by_target_embodiment"]["ur5"] == ["pair-0", "pair-1"]

    generic = ProcessedPairDataset(report.manifest, sample_format="generic")
    item = generic[0]
    assert item["source"]["video"].shape == (3, 5, 4, 4)
    assert item["target"]["actions"].shape == (5, 4)
    assert item["target"]["video"].dtype == torch.float32
    assert item["target"]["video"].min() >= -1
    assert item["target"]["video"].max() <= 1

    dynamic = ProcessedPairDataset(
        report.manifest,
        sample_format="cosmos",
        reference_mode="deterministic",
        reference_seed=99,
    )
    dynamic_item = dynamic[0]
    assert dynamic_item["reference_video"].shape == (3, 5, 4, 4)
    assert (
        dynamic_item["metadata"]["reference_target"]["episode_id"]
        != dynamic_item["metadata"]["target_gt"]["episode_id"]
    )
    batch = collate_pairs([dynamic[0], dynamic[1]])
    assert batch["video"].shape == (2, 3, 5, 4, 4)
    assert batch["control_actions"].shape == (2, 5, 4)
    assert batch["sample_id"] == ["pair-0", "pair-1"]


def test_global_rank_sharding_uses_global_manifest_indices(tmp_path: Path):
    raw = _build_raw_manifest(tmp_path)
    report = preprocess_manifest(
        raw,
        tmp_path / "processed",
        config=PreprocessConfig(
            num_frames=5,
            sample_fps=2,
            height=4,
            width=4,
            action_dim=2,
        ),
    )
    rank0 = ProcessedPairDataset(
        report.manifest, shard_by_rank=True, global_rank=0, world_size=2
    )
    rank1 = ProcessedPairDataset(
        report.manifest, shard_by_rank=True, global_rank=1, world_size=2
    )
    assert rank0.global_indices == (0,)
    assert rank1.global_indices == (1,)
    assert rank0[0]["sample_id"] == "pair-0"
    assert rank1[0]["sample_id"] == "pair-1"


def test_logical_episode_range_cannot_sample_adjacent_task_values(tmp_path: Path):
    video = np.stack(
        [np.full((2, 2, 3), value, dtype=np.uint8) for value in (10, 20, 30, 40, 50)]
    )
    actions = np.arange(5, dtype=np.float32)[:, None]
    np.save(tmp_path / "video.npy", video)
    np.save(tmp_path / "actions.npy", actions)
    episode = EpisodeRef(
        episode_id="bounded",
        embodiment="robot",
        video=tmp_path / "video.npy",
        actions=tmp_path / "actions.npy",
        start_time=1.4,
        end_time=3.0,
        video_fps=1.0,
        action_fps=1.0,
    )
    stream = _process_stream(
        episode,
        config=PreprocessConfig(
            num_frames=2,
            sample_fps=1.0,
            height=2,
            width=2,
            action_dim=1,
            short_policy="drop",
            validate_wan_frames=False,
        ),
    )
    # The logical boundary is not aligned to media timestamps. Sampling snaps to
    # the first common in-range observation (t=2) instead of dropping valid data
    # or reading the closer out-of-episode value at t=1.
    assert stream.clip_start == pytest.approx(2.0)
    assert stream.video[0, 0, 0, 0] == 30
    assert stream.actions[0, 0] == pytest.approx(2.0)
    assert stream.frame_mask.all()
    assert stream.action_mask.all()


def test_float_array_video_rejects_non_finite_pixels(tmp_path: Path):
    video = np.zeros((2, 2, 2, 3), dtype=np.float32)
    video[0, 0, 0, 0] = np.nan
    np.save(tmp_path / "bad.npy", video)
    with pytest.raises(ValueError, match="NaN or infinite"):
        decode_video(tmp_path / "bad.npy", fps=1.0)


def test_dataset_rejects_npz_path_outside_processed_root(tmp_path: Path):
    raw = _build_raw_manifest(tmp_path)
    report = preprocess_manifest(
        raw,
        tmp_path / "processed",
        config=PreprocessConfig(
            num_frames=5,
            sample_fps=2,
            height=4,
            width=4,
            action_dim=2,
        ),
    )
    entries = [json.loads(line) for line in report.manifest.read_text().splitlines()]
    entries[0]["npz"] = "../outside.npz"
    report.manifest.write_text(
        "".join(json.dumps(entry) + "\n" for entry in entries),
        encoding="utf-8",
    )
    dataset = ProcessedPairDataset(report.manifest)
    with pytest.raises(ValueError, match="escapes manifest root"):
        dataset[0]
