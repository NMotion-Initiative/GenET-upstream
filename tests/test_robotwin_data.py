from __future__ import annotations

import json
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from genet.cli.validate_data import validate_manifest
from genet.data.content import stream_content_sha256
from genet.data.dataset import ProcessedPairDataset
from genet.data.preprocess import PreprocessConfig
from genet.data.robotwin import (
    RoboTwinContract,
    RoboTwinPreprocessConfig,
    RoboTwinSchemaError,
    build_episode_catalog,
    load_robotwin_contract,
    preprocess_robotwin_datasets,
    validate_robotwin_root,
)

ROOT = Path(__file__).resolve().parents[1]


class _Rows:
    def __init__(self, rows: list[dict]):
        self.rows = rows
        self.num_samples = len(rows)

    def __getitem__(self, index: int):
        return self.rows[index]


def _jpeg(color: int) -> bytes:
    output = BytesIO()
    Image.fromarray(np.full((4, 6, 3), color, dtype=np.uint8)).save(
        output, format="JPEG", quality=100, subsampling=0
    )
    return output.getvalue()


def _contract(
    *, expected_counts: dict[str, dict[str, dict[str, int]]] | None = None
) -> RoboTwinContract:
    columns = {
        "action_window": "ndarray:float32",
        "embodiment": "str",
        "endpose": "ndarray:float32",
        "episode_idx": "int",
        "episode_len": "int",
        "head_rgb": "bytes",
        "left_rgb": "bytes",
        "right_rgb": "bytes",
        "state": "ndarray:float32",
        "t": "int",
        "task": "str",
    }
    return RoboTwinContract(
        dataset="robotwin_v1",
        schema_version=1,
        streaming_version="0.13.0",
        observed_schema_sha256="a" * 64,
        default_local_root="/data",
        columns=columns,
        action_dims={"robot-a": 2, "robot-b": 3},
        cameras=("head", "left", "right"),
        window_size=100,
        expected_counts=expected_counts or {},
        file_sha256="b" * 64,
    )


def _actual_counts() -> dict[str, dict[str, dict[str, int]]]:
    return {
        "train": {
            "robot-a": {"tasks": 1, "episodes": 1, "samples": 6},
            "robot-b": {"tasks": 2, "episodes": 2, "samples": 13},
        },
        "val": {
            "robot-a": {"tasks": 1, "episodes": 1, "samples": 1},
            "robot-b": {"tasks": 1, "episodes": 1, "samples": 1},
        },
    }


def _write_valid_root(root: Path, contract: RoboTwinContract) -> dict[str, Any]:
    totals = {
        split: {
            "episodes": sum(
                counts["episodes"]
                for counts in contract.expected_counts[split].values()
            ),
            "samples": sum(
                counts["samples"] for counts in contract.expected_counts[split].values()
            ),
        }
        for split in ("train", "val")
    }
    manifest = {
        "schema_version": contract.schema_version,
        "mosaicml_streaming_version": contract.streaming_version,
        "k_max": contract.window_size,
        "action_dims": dict(contract.action_dims),
        "cameras": list(contract.cameras),
        "columns": dict(contract.columns),
        "embodiments": list(contract.action_dims),
        "totals": totals,
    }
    root.mkdir()
    (root / "manifest.json").write_text(
        json.dumps(manifest) + "\n", encoding="utf-8"
    )
    for split in ("train", "val"):
        for embodiment in contract.action_dims:
            stream = root / split / embodiment
            stream.mkdir(parents=True)
            (stream / "index.json").write_text("{}\n", encoding="utf-8")
    return manifest


def _episode(
    *, embodiment: str, task: str, episode_idx: int, length: int, dim: int, base: float
) -> list[dict]:
    states = np.stack(
        [np.arange(dim, dtype=np.float32) + base + timestep for timestep in range(length)]
    )
    rows = []
    for timestep in range(length):
        payload = _jpeg(int(base + timestep))
        rows.append(
            {
                "action_window": states[timestep : timestep + 100].copy(),
                "embodiment": embodiment,
                "endpose": np.arange(16, dtype=np.float32),
                "episode_idx": episode_idx,
                "episode_len": length,
                "head_rgb": payload,
                "left_rgb": payload,
                "right_rgb": payload,
                "state": states[timestep].copy(),
                "t": timestep,
                "task": task,
            }
        )
    return rows


def _config() -> RoboTwinPreprocessConfig:
    return RoboTwinPreprocessConfig(
        mds_index_fps=2.0,
        preprocess=PreprocessConfig(
            num_frames=5,
            sample_fps=2.0,
            height=4,
            width=6,
            action_dim=4,
            short_policy="drop",
            reference_seed=19,
            default_video_fps=None,
            default_action_fps=None,
        ),
    )


def test_stream_content_hash_is_layout_and_endian_independent() -> None:
    video = np.arange(2 * 2 * 3 * 3, dtype=np.uint8).reshape(2, 2, 3, 3)
    actions = np.arange(8, dtype=np.dtype("<f4")).reshape(2, 4)
    action_mask = np.ones((2, 4), dtype=np.bool_)
    frame_mask = np.ones(2, dtype=np.bool_)
    expected = stream_content_sha256(
        video=video,
        actions=actions,
        action_mask=action_mask,
        frame_mask=frame_mask,
    )
    assert expected == stream_content_sha256(
        video=np.asfortranarray(video),
        actions=actions.astype(np.dtype(">f4")),
        action_mask=np.asfortranarray(action_mask),
        frame_mask=frame_mask.copy(),
    )
    changed = actions.copy()
    changed[0, 0] += 1
    assert expected != stream_content_sha256(
        video=video,
        actions=changed,
        action_mask=action_mask,
        frame_mask=frame_mask,
    )


def test_committed_robotwin_contract_matches_supplied_schema_probe():
    contract = load_robotwin_contract(ROOT / "configs" / "data" / "robotwin_v1.json")
    assert contract.streaming_version == "0.13.0"
    assert contract.observed_schema_sha256 == (
        "e97af74a65748aff2b16ee0d9bd5fd7f2299af14f04a083ea2fef0f1cf435446"
    )
    assert contract.action_dims == {
        "ARX-X5": 14,
        "aloha-agilex": 14,
        "franka-panda": 16,
        "piper": 14,
        "ur5-wsg": 14,
    }
    assert contract.cameras == ("head", "left", "right")
    assert contract.expected_counts["train"]["ARX-X5"] == {
        "tasks": 22,
        "episodes": 2086,
        "samples": 264725,
    }
    assert sum(
        value["samples"] for value in contract.expected_counts["train"].values()
    ) == 1_416_642
    assert sum(
        value["samples"] for value in contract.expected_counts["val"].values()
    ) == 74_854


def test_root_validation_checks_identity_totals_and_every_stream(tmp_path: Path):
    contract = _contract(expected_counts=_actual_counts())
    root = tmp_path / "mds"
    manifest = _write_valid_root(root, contract)

    resolved, digest = validate_robotwin_root(root, contract)
    assert resolved == root.resolve()
    assert len(digest) == 64

    manifest["totals"]["train"]["samples"] += 1
    (root / "manifest.json").write_text(
        json.dumps(manifest) + "\n", encoding="utf-8"
    )
    with pytest.raises(RoboTwinSchemaError, match="totals differ"):
        validate_robotwin_root(root, contract)

    manifest["totals"]["train"]["samples"] -= 1
    (root / "manifest.json").write_text(
        json.dumps(manifest) + "\n", encoding="utf-8"
    )
    (root / "val" / "robot-b" / "index.json").unlink()
    with pytest.raises(FileNotFoundError, match="val/robot-b/index.json"):
        validate_robotwin_root(root, contract)


def test_non_one_to_one_fps_requires_nearest_action_resampling():
    assert RoboTwinPreprocessConfig(mds_index_fps=16.0).preprocess.sample_fps == 16.0
    with pytest.raises(ValueError, match="cannot linearly interpolate"):
        RoboTwinPreprocessConfig(
            mds_index_fps=15.0,
            preprocess=PreprocessConfig(action_resample="linear"),
        )
    config = RoboTwinPreprocessConfig(
        mds_index_fps=15.0,
        preprocess=PreprocessConfig(action_resample="nearest"),
    )
    assert config.preprocess.action_resample == "nearest"


def test_robotwin_direct_export_is_fixed_shape_and_cosmos_valid(tmp_path: Path):
    source_rows = _episode(
        embodiment="robot-a", task="task-a", episode_idx=0, length=6, dim=2, base=10
    )
    target_rows = _episode(
        embodiment="robot-b", task="task-a", episode_idx=0, length=6, dim=3, base=20
    ) + _episode(
        embodiment="robot-b", task="task-b", episode_idx=0, length=7, dim=3, base=40
    )
    root = tmp_path / "mds"
    root.mkdir()
    (root / "manifest.json").write_text('{"version": 1}\n', encoding="utf-8")
    report = preprocess_robotwin_datasets(
        {"robot-a": _Rows(source_rows), "robot-b": _Rows(target_rows)},
        tmp_path / "processed",
        contract=_contract(),
        split="train",
        config=_config(),
        source_embodiments=("robot-a",),
        target_embodiments=("robot-b",),
        dataset_root=root,
    )

    assert report.written == 1
    entry = json.loads(report.manifest.read_text(encoding="utf-8"))
    assert entry["metadata"]["pairing_policy"] == "same_split_task_episode_idx"
    assert entry["source"]["metadata"]["action_signal"] == "joint_position_state"
    assert entry["reference_target"]["metadata"]["task"] == "task-b"
    assert entry["reference_target"]["embodiment"] == "robot-b"
    with np.load(report.manifest.parent / entry["npz"], allow_pickle=False) as arrays:
        assert arrays["source_video"].shape == (5, 4, 6, 3)
        assert arrays["source_actions"].shape == (5, 4)
        assert arrays["source_action_mask"][:, :2].all()
        assert not arrays["source_action_mask"][:, 2:].any()
        assert arrays["target_action_mask"][:, :3].all()
        assert not arrays["target_action_mask"][:, 3:].any()
        assert arrays["reference_frame_mask"].all()
        for manifest_role, array_role in (
            ("source", "source"),
            ("target_gt", "target"),
            ("reference_target", "reference"),
        ):
            assert entry[manifest_role]["content_sha256"] == stream_content_sha256(
                video=arrays[f"{array_role}_video"],
                actions=arrays[f"{array_role}_actions"],
                action_mask=arrays[f"{array_role}_action_mask"],
                frame_mask=arrays[f"{array_role}_frame_mask"],
            )

    validation = validate_manifest(
        report.manifest,
        num_frames=5,
        height=4,
        width=6,
        action_dim=4,
        cosmos=True,
    )
    assert validation["valid"] is True
    dataset = ProcessedPairDataset(report.manifest, reference_mode="stored")
    sample = dataset[0]
    assert sample["target"]["video"].shape == (3, 5, 4, 6)
    assert sample["metadata"]["reference_target"]["metadata"]["task"] == "task-b"

    stats = json.loads(report.stats.read_text(encoding="utf-8"))
    assert stats["lineage"]["contract_file_sha256"] == "b" * 64
    assert len(stats["lineage"]["root_manifest_sha256"]) == 64
    assert stats["pair_directions"]["robot-a->robot-b"]["common_episodes"] == 1
    assert stats["pairing"]["direction_policy"] == "selected_directed_cross_product"
    assert stats["pairing"]["planned_samples"] == 1
    assert stats["pairing"]["truncated"] is False


def test_robotwin_release_export_materializes_balanced_reverse_directions(
    tmp_path: Path,
):
    datasets = {
        "robot-a": _Rows(
            _episode(
                embodiment="robot-a",
                task="task-a",
                episode_idx=0,
                length=6,
                dim=2,
                base=10,
            )
            + _episode(
                embodiment="robot-a",
                task="task-b",
                episode_idx=0,
                length=6,
                dim=2,
                base=20,
            )
        ),
        "robot-b": _Rows(
            _episode(
                embodiment="robot-b",
                task="task-a",
                episode_idx=0,
                length=6,
                dim=3,
                base=30,
            )
            + _episode(
                embodiment="robot-b",
                task="task-b",
                episode_idx=0,
                length=6,
                dim=3,
                base=40,
            )
        ),
    }
    report = preprocess_robotwin_datasets(
        datasets,
        tmp_path / "bidirectional",
        contract=_contract(),
        split="train",
        config=_config(),
        require_bidirectional=True,
    )
    entries = [
        json.loads(line)
        for line in report.manifest.read_text(encoding="utf-8").splitlines()
    ]
    assert len(entries) == 4
    assert {entry["metadata"]["direction"] for entry in entries} == {
        "robot-a->robot-b",
        "robot-b->robot-a",
    }
    assert all(
        entry["source"]["embodiment"] != entry["target_gt"]["embodiment"]
        for entry in entries
    )
    assert all(
        entry["reference_target"]["embodiment"]
        == entry["target_gt"]["embodiment"]
        for entry in entries
    )
    assert all(
        entry["reference_target"]["metadata"]["task"]
        != entry["target_gt"]["metadata"]["task"]
        for entry in entries
    )

    by_identity: dict[str, list[dict]] = {}
    for entry in entries:
        by_identity.setdefault(entry["metadata"]["pair_identity"], []).append(entry)
    assert {len(pair) for pair in by_identity.values()} == {2}
    for forward, reverse in by_identity.values():
        if forward["source"]["embodiment"] == "robot-b":
            forward, reverse = reverse, forward
        assert (
            forward["source"]["content_sha256"]
            == reverse["target_gt"]["content_sha256"]
        )
        assert (
            forward["target_gt"]["content_sha256"]
            == reverse["source"]["content_sha256"]
        )
        forward_npz = report.manifest.parent / forward["npz"]
        reverse_npz = report.manifest.parent / reverse["npz"]
        with np.load(forward_npz, allow_pickle=False) as a, np.load(
            reverse_npz, allow_pickle=False
        ) as b:
            for suffix in ("video", "actions", "action_mask", "frame_mask"):
                np.testing.assert_array_equal(
                    a[f"source_{suffix}"], b[f"target_{suffix}"]
                )
                np.testing.assert_array_equal(
                    a[f"target_{suffix}"], b[f"source_{suffix}"]
                )

    validation = validate_manifest(
        report.manifest,
        cosmos=True,
        require_bidirectional_pairs=True,
    )
    assert validation["valid"] is True
    assert validation["pair_directions"]["bidirectional_complete"] is True
    dataset = ProcessedPairDataset(
        report.manifest, require_bidirectional_pairs=True
    )
    assert dataset.direction_summary["direction_counts"] == {
        "robot-a->robot-b": 2,
        "robot-b->robot-a": 2,
    }
    stats = json.loads(report.stats.read_text(encoding="utf-8"))
    assert stats["pairing"]["direction_policy"] == "all_contract_ordered_distinct"
    assert stats["pairing"]["planned_samples"] == 4
    assert stats["pairing"]["truncated"] is False

    with pytest.raises(ValueError, match="cannot be combined with max_samples"):
        preprocess_robotwin_datasets(
            datasets,
            tmp_path / "truncated",
            contract=_contract(),
            split="train",
            config=_config(),
            max_samples=2,
            require_bidirectional=True,
        )

    capped = preprocess_robotwin_datasets(
        datasets,
        tmp_path / "capped",
        contract=_contract(),
        split="train",
        config=_config(),
        max_samples=1,
    )
    capped_stats = json.loads(capped.stats.read_text(encoding="utf-8"))
    assert capped_stats["pairing"]["direction_policy"] == (
        "all_contract_ordered_distinct"
    )
    assert capped_stats["pairing"]["planned_samples"] == 4
    assert capped_stats["pairing"]["truncated"] is True
    assert set(capped_stats["pair_directions"]) == {
        "robot-a->robot-b",
        "robot-b->robot-a",
    }
    assert capped_stats["pair_directions"]["robot-b->robot-a"][
        "written_samples"
    ] == 0


def test_robotwin_release_requires_full_contract_set_and_different_task_policy(
    tmp_path: Path,
):
    contract = replace(
        _contract(),
        action_dims={"robot-a": 2, "robot-b": 3, "robot-c": 2},
    )
    with pytest.raises(ValueError, match="every contract embodiment"):
        preprocess_robotwin_datasets(
            {"robot-a": _Rows([]), "robot-b": _Rows([])},
            tmp_path / "subset",
            contract=contract,
            split="train",
            config=_config(),
            source_embodiments=("robot-a", "robot-b"),
            target_embodiments=("robot-a", "robot-b"),
            require_bidirectional=True,
        )

    with pytest.raises(ValueError, match="reference_policy='different_task'"):
        preprocess_robotwin_datasets(
            {"robot-a": _Rows([]), "robot-b": _Rows([])},
            tmp_path / "reference-policy",
            contract=_contract(),
            split="train",
            config=replace(_config(), reference_policy="any_task"),
            require_bidirectional=True,
        )


def test_catalog_rejects_noncontiguous_or_inconsistent_action_windows():
    rows = _episode(
        embodiment="robot-a", task="task", episode_idx=0, length=6, dim=2, base=1
    )
    rows[1]["t"] = 2
    with pytest.raises(RoboTwinSchemaError, match="expected 1"):
        build_episode_catalog(
            _Rows(rows),
            contract=_contract(),
            split="train",
            embodiment="robot-a",
        )

    rows = _episode(
        embodiment="robot-a", task="task", episode_idx=0, length=6, dim=2, base=1
    )
    rows[1]["action_window"][0, 0] += 1
    with pytest.raises(RoboTwinSchemaError, match="does not match state"):
        build_episode_catalog(
            _Rows(rows),
            contract=_contract(),
            split="train",
            embodiment="robot-a",
        )


def test_expected_counts_are_enforced_before_export(tmp_path: Path):
    source_rows = _episode(
        embodiment="robot-a", task="task-a", episode_idx=0, length=6, dim=2, base=1
    )
    target_rows = _episode(
        embodiment="robot-b", task="task-a", episode_idx=0, length=6, dim=3, base=2
    ) + _episode(
        embodiment="robot-b", task="task-b", episode_idx=0, length=7, dim=3, base=3
    )
    datasets = {"robot-a": _Rows(source_rows), "robot-b": _Rows(target_rows)}
    counts = _actual_counts()
    counts["train"]["robot-a"]["episodes"] = 2

    with pytest.raises(RoboTwinSchemaError, match="differ from approved counts"):
        preprocess_robotwin_datasets(
            datasets,
            tmp_path / "processed",
            contract=_contract(expected_counts=counts),
            split="train",
            config=_config(),
            source_embodiments=("robot-a",),
            target_embodiments=("robot-b",),
        )


@pytest.mark.parametrize(
    ("source_embodiments", "message"),
    [
        (("robot-a", "robot-a"), "duplicate names"),
        ((), "cannot be empty"),
    ],
)
def test_duplicate_or_empty_embodiment_selection_is_rejected(
    tmp_path: Path, source_embodiments: tuple[str, ...], message: str
):
    with pytest.raises(ValueError, match=message):
        preprocess_robotwin_datasets(
            {"robot-a": _Rows([]), "robot-b": _Rows([])},
            tmp_path / "processed",
            contract=_contract(),
            split="train",
            config=_config(),
            source_embodiments=source_embodiments,
            target_embodiments=("robot-b",),
        )


def test_overwrite_removes_only_unreferenced_sample_npzs(tmp_path: Path):
    datasets = {
        "robot-a": _Rows(
            _episode(
                embodiment="robot-a",
                task="task-a",
                episode_idx=0,
                length=6,
                dim=2,
                base=1,
            )
        ),
        "robot-b": _Rows(
            _episode(
                embodiment="robot-b",
                task="task-a",
                episode_idx=0,
                length=6,
                dim=3,
                base=2,
            )
            + _episode(
                embodiment="robot-b",
                task="task-b",
                episode_idx=0,
                length=7,
                dim=3,
                base=3,
            )
        ),
    }
    output = tmp_path / "processed"
    first = preprocess_robotwin_datasets(
        datasets,
        output,
        contract=_contract(),
        split="train",
        config=replace(
            _config(), window_policy="sliding", clip_stride_native_frames=1
        ),
        source_embodiments=("robot-a",),
        target_embodiments=("robot-b",),
    )
    assert first.written == 2
    first_npzs = set((output / "samples").glob("*.npz"))
    note = output / "samples" / "keep.txt"
    note.write_text("not adapter sample data\n", encoding="utf-8")

    second = preprocess_robotwin_datasets(
        datasets,
        output,
        contract=_contract(),
        split="train",
        config=_config(),
        source_embodiments=("robot-a",),
        target_embodiments=("robot-b",),
        overwrite=True,
    )
    entries = [
        json.loads(line)
        for line in second.manifest.read_text(encoding="utf-8").splitlines()
    ]
    referenced = {output / entry["npz"] for entry in entries}
    assert second.written == 1
    assert first_npzs - referenced
    assert set((output / "samples").glob("*.npz")) == referenced
    assert note.read_text(encoding="utf-8") == "not adapter sample data\n"


def test_reference_selection_is_reproducible_and_never_same_task(tmp_path: Path):
    datasets = {
        "robot-a": _Rows(
            _episode(
                embodiment="robot-a",
                task="task-a",
                episode_idx=0,
                length=6,
                dim=2,
                base=1,
            )
        ),
        "robot-b": _Rows(
            _episode(
                embodiment="robot-b",
                task="task-a",
                episode_idx=0,
                length=6,
                dim=3,
                base=2,
            )
            + _episode(
                embodiment="robot-b",
                task="task-b",
                episode_idx=0,
                length=6,
                dim=3,
                base=3,
            )
            + _episode(
                embodiment="robot-b",
                task="task-c",
                episode_idx=0,
                length=6,
                dim=3,
                base=4,
            )
        ),
    }
    references = []
    for name in ("one", "two"):
        report = preprocess_robotwin_datasets(
            datasets,
            tmp_path / name,
            contract=_contract(),
            split="train",
            config=_config(),
            source_embodiments=("robot-a",),
            target_embodiments=("robot-b",),
        )
        entry = json.loads(report.manifest.read_text(encoding="utf-8"))
        references.append(entry["reference_target"])
    assert references[0] == references[1]
    assert references[0]["metadata"]["task"] != "task-a"
