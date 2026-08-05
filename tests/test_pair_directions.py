from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from genet.data.dataset import ProcessedPairDataset
from genet.data.directions import require_bidirectional_pairs, summarize_pair_directions


def _entry(
    sample_id: str,
    *,
    source: str,
    target: str,
    source_episode: str,
    target_episode: str,
    pair_identity: str = "window-0:a<>b",
) -> dict:
    return {
        "format_version": "genet.processed-pair/v1",
        "id": sample_id,
        "npz": f"samples/{sample_id}.npz",
        "source": {
            "embodiment": source,
            "episode_id": source_episode,
            "clip_start": 0.0,
            "content_sha256": source[0] * 64,
            "metadata": {"task": "paired-task"},
        },
        "target_gt": {
            "embodiment": target,
            "episode_id": target_episode,
            "clip_start": 0.0,
            "content_sha256": target[0] * 64,
            "metadata": {"task": "paired-task"},
        },
        "reference_target": {
            "embodiment": target,
            "episode_id": f"reference-{target}",
            "clip_start": 1.0,
            "content_sha256": "f" * 64,
            "metadata": {"task": "reference-task"},
        },
        "metadata": {
            "pair_identity": pair_identity,
            "reference_policy": "different_task",
        },
    }


def _write_manifest(path: Path, entries: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8"
    )
    return path


def test_direction_summary_requires_exact_reciprocal_records() -> None:
    forward = _entry(
        "a-to-b",
        source="a",
        target="b",
        source_episode="a-0",
        target_episode="b-0",
    )
    reverse = _entry(
        "b-to-a",
        source="b",
        target="a",
        source_episode="b-0",
        target_episode="a-0",
    )
    summary = require_bidirectional_pairs(
        [forward, reverse], expected_embodiments=["a", "b"]
    )
    assert summary["directed_samples"] == 2
    assert summary["unique_pair_windows"] == 1
    assert summary["direction_counts"] == {"a->b": 1, "b->a": 1}
    assert summary["source_role_counts"] == summary["target_role_counts"] == {
        "a": 1,
        "b": 1,
    }
    assert summary["bidirectional_complete"] is True

    reverse["target_gt"]["clip_start"] = 0.5
    broken = summarize_pair_directions([forward, reverse], strict=True)
    assert broken["bidirectional_complete"] is False
    assert {error["code"] for error in broken["errors"]} == {
        "reverse_sample_mismatch"
    }


def test_strict_direction_gate_requires_full_graph_and_complete_identity() -> None:
    forward = _entry(
        "a-to-b",
        source="a",
        target="b",
        source_episode="a-0",
        target_episode="b-0",
    )
    reverse = _entry(
        "b-to-a",
        source="b",
        target="a",
        source_episode="b-0",
        target_episode="a-0",
    )
    incomplete = summarize_pair_directions(
        [forward, reverse],
        expected_embodiments=["a", "b", "c"],
        strict=True,
    )
    assert incomplete["bidirectional_complete"] is False
    assert {error["code"] for error in incomplete["errors"]} >= {
        "missing_embodiments",
        "missing_directions",
    }

    b_to_c = _entry(
        "b-to-c",
        source="b",
        target="c",
        source_episode="b-0",
        target_episode="c-0",
        pair_identity="window-0:b<>c",
    )
    c_to_b = _entry(
        "c-to-b",
        source="c",
        target="b",
        source_episode="c-0",
        target_episode="b-0",
        pair_identity="window-0:b<>c",
    )
    missing_pair = summarize_pair_directions(
        [forward, reverse, b_to_c, c_to_b],
        expected_embodiments=["a", "b", "c"],
        strict=True,
    )
    missing_pair_codes = {error["code"] for error in missing_pair["errors"]}
    assert "missing_embodiments" not in missing_pair_codes
    assert "missing_directions" in missing_pair_codes

    missing_clip = copy.deepcopy(forward)
    missing_clip["source"].pop("clip_start")
    with pytest.raises(ValueError, match="source.clip_start is required"):
        require_bidirectional_pairs([missing_clip, reverse])

    same_task_reference = copy.deepcopy(reverse)
    same_task_reference["reference_target"]["metadata"]["task"] = "paired-task"
    with pytest.raises(ValueError, match="must differ"):
        require_bidirectional_pairs([forward, same_task_reference])

    wrong_payload = copy.deepcopy(reverse)
    wrong_payload["source"]["content_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="reverse_sample_mismatch"):
        require_bidirectional_pairs([forward, wrong_payload])


def test_strict_direction_gate_rejects_balanced_duplicate_records() -> None:
    forward = _entry(
        "a-to-b",
        source="a",
        target="b",
        source_episode="a-0",
        target_episode="b-0",
    )
    reverse = _entry(
        "b-to-a",
        source="b",
        target="a",
        source_episode="b-0",
        target_episode="a-0",
    )
    forward_duplicate = copy.deepcopy(forward)
    forward_duplicate["id"] = "a-to-b-copy"
    reverse_duplicate = copy.deepcopy(reverse)
    reverse_duplicate["id"] = "b-to-a-copy"
    with pytest.raises(ValueError, match="duplicate_directed_records"):
        require_bidirectional_pairs(
            [forward, reverse, forward_duplicate, reverse_duplicate]
        )


def test_dataset_bidirectional_gate_fails_before_loading_npz(tmp_path: Path) -> None:
    forward = _entry(
        "a-to-b",
        source="a",
        target="b",
        source_episode="a-0",
        target_episode="b-0",
    )
    manifest = _write_manifest(tmp_path / "manifest.jsonl", [forward])
    with pytest.raises(ValueError, match="complete bidirectional"):
        ProcessedPairDataset(
            manifest,
            require_bidirectional_pairs=True,
            expected_embodiments=["a", "b"],
        )

    forward["target_gt"]["embodiment"] = "a"
    forward["reference_target"]["embodiment"] = "a"
    _write_manifest(manifest, [forward])
    with pytest.raises(ValueError, match="distinct embodiments"):
        ProcessedPairDataset(manifest)
