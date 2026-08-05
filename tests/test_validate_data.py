import json
from pathlib import Path

import numpy as np

from genet.cli.validate_data import main, validate_manifest
from genet.data.content import stream_content_sha256


def _arrays(*, time: int = 5, height: int = 4, width: int = 6, dim: int = 3):
    result = {}
    for role in ("source", "target", "reference"):
        result[f"{role}_video"] = np.zeros(
            (time, height, width, 3), dtype=np.uint8
        )
        result[f"{role}_actions"] = np.zeros((time, dim), dtype=np.float32)
        result[f"{role}_action_mask"] = np.ones((time, dim), dtype=np.bool_)
        result[f"{role}_frame_mask"] = np.ones(time, dtype=np.bool_)
    return result


def _entry(sample_id: str, npz: str) -> dict:
    return {
        "format_version": "genet.processed-pair/v1",
        "id": sample_id,
        "npz": npz,
        "source": {"episode_id": "source-0", "embodiment": "franka"},
        "target_gt": {"episode_id": "target-0", "embodiment": "ur5"},
        "reference_target": {"episode_id": "target-1", "embodiment": "ur5"},
    }


def _bind_strict_metadata(entry: dict, arrays: dict, *, pair_identity: str) -> None:
    entry["metadata"] = {
        "pair_identity": pair_identity,
        "reference_policy": "different_task",
    }
    for metadata_role, array_role, task in (
        ("source", "source", "paired-task"),
        ("target_gt", "target", "paired-task"),
        ("reference_target", "reference", "reference-task"),
    ):
        stream = entry[metadata_role]
        stream["clip_start"] = 0.0
        stream["metadata"] = {"task": task}
        stream["content_sha256"] = stream_content_sha256(
            video=arrays[f"{array_role}_video"],
            actions=arrays[f"{array_role}_actions"],
            action_mask=arrays[f"{array_role}_action_mask"],
            frame_mask=arrays[f"{array_role}_frame_mask"],
        )


def test_validate_manifest_accepts_complete_fixed_shape_dataset(tmp_path: Path):
    samples = tmp_path / "samples"
    samples.mkdir()
    np.savez_compressed(samples / "a.npz", **_arrays())
    np.savez_compressed(samples / "b.npz", **_arrays())
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(_entry("a", "samples/a.npz"))
        + "\n"
        + json.dumps(_entry("b", "samples/b.npz"))
        + "\n",
        encoding="utf-8",
    )

    summary = validate_manifest(
        manifest, num_frames=5, height=4, width=6, action_dim=3
    )
    assert summary["valid"] is True
    assert summary["samples"] == summary["valid_samples"] == 2
    assert summary["invalid_samples"] == summary["error_count"] == 0
    assert summary["expected_num_frames"] == 5

    rank0 = validate_manifest(manifest, num_frames=5, shard_rank=0, shard_world_size=2)
    rank1 = validate_manifest(manifest, num_frames=5, shard_rank=1, shard_world_size=2)
    assert rank0["valid"] is True and rank0["samples"] == 1
    assert rank1["valid"] is True and rank1["samples"] == 1


def test_validate_cli_reports_shape_finite_file_and_metadata_errors(
    tmp_path: Path, capsys
):
    arrays = _arrays()
    arrays["target_actions"][0, 0] = np.nan
    arrays["reference_frame_mask"] = np.ones((5, 1), dtype=np.bool_)
    arrays["source_video"] = np.zeros((4, 4, 6, 3), dtype=np.uint8)
    np.savez_compressed(tmp_path / "bad.npz", **arrays)
    bad_entry = _entry("bad", "bad.npz")
    bad_entry["reference_target"] = {
        "episode_id": "target-0",
        "embodiment": "other-robot",
    }
    missing_entry = _entry("missing", "missing.npz")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(bad_entry) + "\n" + json.dumps(missing_entry) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "validation.json"

    exit_code = main(
        [
            "--manifest",
            str(manifest),
            "--num-frames",
            "5",
            "--output",
            str(output),
        ]
    )
    printed = json.loads(capsys.readouterr().out)
    written = json.loads(output.read_text(encoding="utf-8"))
    codes = {error["code"] for error in printed["errors"]}
    assert exit_code == 1
    assert printed == written
    assert printed["valid"] is False
    assert printed["invalid_samples"] == 2
    assert "reference_embodiment_mismatch" in codes
    assert "reference_episode_collision" in codes
    assert "temporal_length_mismatch" in codes
    assert "non_finite_array" in codes
    assert "invalid_frame_mask_shape" in codes
    assert "npz_missing" in codes


def test_validate_cli_returns_nonzero_for_missing_manifest(tmp_path: Path, capsys):
    exit_code = main(["--manifest", str(tmp_path / "does-not-exist.jsonl")])
    summary = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert summary["error_count"] == 1
    assert summary["errors"][0]["code"] == "manifest_missing"


def test_validate_empty_manifest_is_invalid(tmp_path: Path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("", encoding="utf-8")
    summary = validate_manifest(manifest)
    assert summary["valid"] is False
    assert summary["samples"] == 0
    assert summary["errors"][0]["code"] == "empty_manifest"


def test_validator_rejects_missing_contract_metadata_and_escaping_npz(tmp_path: Path):
    entry = _entry("bad", "../outside.npz")
    del entry["format_version"]
    del entry["source"]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps(entry) + "\n", encoding="utf-8")

    summary = validate_manifest(manifest)
    codes = {error["code"] for error in summary["errors"]}
    assert "unsupported_format_version" in codes
    assert "invalid_pair_metadata" in codes
    assert "unsafe_npz_path" in codes


def test_cosmos_validation_rejects_padding_and_non_uint8_video(tmp_path: Path):
    arrays = _arrays()
    arrays["source_video"] = arrays["source_video"].astype(np.float32)
    arrays["target_frame_mask"][-1] = False
    arrays["reference_action_mask"][-1, 0] = False
    np.savez_compressed(tmp_path / "bad.npz", **arrays)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(_entry("bad", "bad.npz")) + "\n",
        encoding="utf-8",
    )

    summary = validate_manifest(manifest, cosmos=True)
    codes = {error["code"] for error in summary["errors"]}
    assert "cosmos_video_dtype" in codes
    assert "cosmos_temporal_video_padding" in codes
    assert "cosmos_temporal_action_padding" in codes


def test_validator_reports_global_bidirectional_coverage(tmp_path: Path, capsys):
    forward_arrays = _arrays()
    forward_arrays["source_video"].fill(1)
    forward_arrays["target_video"].fill(2)
    forward_arrays["reference_video"].fill(3)
    reverse_arrays = _arrays()
    for suffix in ("video", "actions", "action_mask", "frame_mask"):
        reverse_arrays[f"source_{suffix}"] = forward_arrays[f"target_{suffix}"].copy()
        reverse_arrays[f"target_{suffix}"] = forward_arrays[f"source_{suffix}"].copy()
    reverse_arrays["reference_video"].fill(4)
    np.savez_compressed(tmp_path / "forward.npz", **forward_arrays)
    np.savez_compressed(tmp_path / "reverse.npz", **reverse_arrays)
    forward = _entry("forward", "forward.npz")
    _bind_strict_metadata(
        forward, forward_arrays, pair_identity="window-0:franka<>ur5"
    )
    reverse = {
        **_entry("reverse", "reverse.npz"),
        "source": dict(forward["target_gt"]),
        "target_gt": dict(forward["source"]),
        "reference_target": {
            "episode_id": "source-reference",
            "embodiment": "franka",
        },
        "metadata": dict(forward["metadata"]),
    }
    _bind_strict_metadata(
        reverse, reverse_arrays, pair_identity="window-0:franka<>ur5"
    )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(forward) + "\n" + json.dumps(reverse) + "\n",
        encoding="utf-8",
    )
    valid = validate_manifest(
        manifest,
        cosmos=True,
        require_bidirectional_pairs=True,
        expected_embodiments=["franka", "ur5"],
        shard_rank=1,
        shard_world_size=2,
    )
    assert valid["valid"] is True
    assert valid["samples"] == 1
    assert valid["pair_directions"]["directed_samples"] == 2
    assert valid["pair_directions"]["bidirectional_complete"] is True

    assert (
        main(
            [
                "--manifest",
                str(manifest),
                "--cosmos",
                "--require-bidirectional",
                "--expected-embodiment",
                "franka",
                "--expected-embodiment",
                "ur5",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["valid"] is True

    tampered = {key: value.copy() for key, value in reverse_arrays.items()}
    tampered["target_video"][0, 0, 0, 0] = 99
    np.savez_compressed(tmp_path / "reverse.npz", **tampered)
    invalid_hash = validate_manifest(
        manifest,
        require_bidirectional_pairs=True,
        expected_embodiments=["franka", "ur5"],
    )
    assert "stream_content_hash_mismatch" in {
        error["code"] for error in invalid_hash["errors"]
    }

    manifest.write_text(json.dumps(forward) + "\n", encoding="utf-8")
    invalid = validate_manifest(
        manifest,
        require_bidirectional_pairs=True,
        expected_embodiments=["franka", "ur5"],
    )
    assert invalid["valid"] is False
    assert "bidirectional_pairs_incomplete" in {
        error["code"] for error in invalid["errors"]
    }
