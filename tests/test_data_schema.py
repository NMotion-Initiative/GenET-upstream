from pathlib import Path

import pytest

from genet.data.reference import NoReferenceCandidateError, StatelessReferencePool
from genet.data.schema import EpisodeRef, PairRecord, SchemaError


def _episode(tmp_path: Path, episode_id: str, embodiment: str = "target") -> dict:
    return {
        "episode_id": episode_id,
        "embodiment": embodiment,
        "video": f"{episode_id}.npy",
        "actions": f"{episode_id}.actions.npy",
        "video_fps": 10,
        "action_fps": 10,
    }


def test_pair_schema_resolves_paths_and_rejects_same_reference_episode(tmp_path: Path):
    raw = {
        "id": "pair-a",
        "source": _episode(tmp_path, "s", "source"),
        "target_gt": _episode(tmp_path, "t"),
        "metadata": {"task": "push"},
    }
    record = PairRecord.from_dict(raw, base_dir=tmp_path, context="test")
    assert record.sample_id == "pair-a"
    assert record.source.video == (tmp_path / "s.npy").resolve()
    assert record.metadata["task"] == "push"

    raw["reference_target"] = _episode(tmp_path, "t")
    with pytest.raises(SchemaError, match="must not use"):
        PairRecord.from_dict(raw, base_dir=tmp_path, context="test")

    raw.pop("reference_target")
    raw["source"] = _episode(tmp_path, "s", "target")
    with pytest.raises(SchemaError, match="distinct source and target_gt"):
        PairRecord.from_dict(raw, base_dir=tmp_path, context="test")


def test_stateless_reference_is_order_independent_and_excludes_target(tmp_path: Path):
    candidates = [
        EpisodeRef(
            episode_id=episode,
            embodiment="ur5",
            video=tmp_path / f"{episode}.npy",
            actions=tmp_path / f"{episode}.actions.npy",
        )
        for episode in ("e0", "e1", "e2")
    ]
    first = StatelessReferencePool(candidates, seed=17).choose(
        embodiment="ur5", exclude_episode_id="e0", sample_key="sample-3"
    )
    second = StatelessReferencePool(list(reversed(candidates)), seed=17).choose(
        embodiment="ur5", exclude_episode_id="e0", sample_key="sample-3"
    )
    assert first.episode_id == second.episode_id
    assert first.episode_id != "e0"

    with pytest.raises(NoReferenceCandidateError):
        StatelessReferencePool(candidates[:1]).choose(
            embodiment="ur5", exclude_episode_id="e0", sample_key="x"
        )


def test_wan_frame_validation_is_configurable():
    from genet.data.preprocess import PreprocessConfig

    with pytest.raises(ValueError, match="Wan temporal length"):
        PreprocessConfig(num_frames=8)
    assert PreprocessConfig(num_frames=8, validate_wan_frames=False).num_frames == 8
