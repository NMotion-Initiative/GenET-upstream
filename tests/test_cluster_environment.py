from __future__ import annotations

import json
from pathlib import Path

import pytest

from genet.cli.cluster_lock import main
from genet.training.cosmos import _validate_hf_cache_revision
from genet.training.environment import (
    CLUSTER_LOCK_VERSION,
    assert_artifact_bound_to_receipt,
    create_cluster_lock,
    create_cluster_receipt,
    fingerprint_artifact,
    load_cluster_lock,
    load_cluster_receipt,
    verify_cluster_lock,
    write_cluster_lock,
    write_cluster_receipt,
)


def _artifact_tree(root: Path) -> Path:
    root.mkdir(parents=True)
    (root / "manifest.jsonl").write_text('{"sample_id":"one"}\n', encoding="utf-8")
    nested = root / "samples"
    nested.mkdir()
    (nested / "one.bin").write_bytes(b"video-action-pair")
    return root


def test_directory_fingerprint_ignores_absolute_root(tmp_path: Path) -> None:
    first = _artifact_tree(tmp_path / "node-a" / "data")
    second = _artifact_tree(tmp_path / "node-b" / "different-local-path")
    assert fingerprint_artifact(first) == fingerprint_artifact(second)

    first_file = tmp_path / "node-a" / "vae-a.pth"
    second_file = tmp_path / "node-b" / "vae-b.pth"
    first_file.write_bytes(b"same-weights")
    second_file.write_bytes(b"same-weights")
    assert fingerprint_artifact(first_file) == fingerprint_artifact(second_file)


def test_cluster_lock_round_trip_and_detects_content_change(tmp_path: Path) -> None:
    canonical = _artifact_tree(tmp_path / "staging")
    replica = _artifact_tree(tmp_path / "node-0")
    lock = create_cluster_lock({"processed_data": canonical})
    assert lock["format_version"] == CLUSTER_LOCK_VERSION
    lock_path = write_cluster_lock(tmp_path / "release.lock.json", lock)
    loaded = load_cluster_lock(lock_path)
    verified = verify_cluster_lock(loaded, {"processed_data": replica})
    assert verified["processed_data"].file_count == 2
    receipt = create_cluster_receipt(
        lock_path,
        loaded,
        {"processed_data": replica},
        verified,
    )
    receipt_path = write_cluster_receipt(tmp_path / "node-0.receipt.json", receipt)
    loaded_receipt = load_cluster_receipt(receipt_path, lock_path)
    assert loaded_receipt["artifacts"]["processed_data"]["path"] == str(replica.resolve())

    (replica / "samples" / "one.bin").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="do not match"):
        verify_cluster_lock(loaded, {"processed_data": replica})


def test_cluster_lock_accepts_internal_symlinks_and_rejects_escape(tmp_path: Path) -> None:
    first = _artifact_tree(tmp_path / "node-a" / "data")
    second = _artifact_tree(tmp_path / "node-b" / "data")
    (first / "link.bin").symlink_to(first / "samples" / "one.bin")
    (second / "link.bin").symlink_to(second / "samples" / "one.bin")
    assert fingerprint_artifact(first) == fingerprint_artifact(second)

    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"not-contained")
    (first / "escape.bin").symlink_to(outside)
    with pytest.raises(ValueError, match="inside the artifact root"):
        fingerprint_artifact(first)


def test_cluster_lock_cli_create_and_verify(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    canonical = _artifact_tree(tmp_path / "staging")
    replica = _artifact_tree(tmp_path / "node-0")
    lock_path = tmp_path / "release.lock.json"
    assert main(
        [
            "create",
            "--output",
            str(lock_path),
            "--artifact",
            f"processed_data={canonical}",
        ]
    ) == 0
    created = json.loads(capsys.readouterr().out)
    assert created["event"] == "cluster_lock_created"

    assert main(
        [
            "verify",
            "--lock",
            str(lock_path),
            "--artifact",
            f"processed_data={replica}",
            "--receipt",
            str(tmp_path / "node-0.receipt.json"),
        ]
    ) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified["event"] == "cluster_lock_verified"
    assert verified["receipt"] == str((tmp_path / "node-0.receipt.json").resolve())


def test_receipt_binds_runtime_path_without_comparing_node_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = _artifact_tree(tmp_path / "staging")
    replica = _artifact_tree(tmp_path / "node-0")
    other = _artifact_tree(tmp_path / "unverified")
    lock = create_cluster_lock({"processed_data": canonical})
    lock_path = write_cluster_lock(tmp_path / "release.lock.json", lock)
    fingerprints = verify_cluster_lock(lock, {"processed_data": replica})
    receipt = create_cluster_receipt(
        lock_path,
        lock,
        {"processed_data": replica},
        fingerprints,
    )
    receipt_path = write_cluster_receipt(tmp_path / "node-0.receipt.json", receipt)
    monkeypatch.setenv("GENET_CLUSTER_LOCK", str(lock_path))
    monkeypatch.setenv("GENET_CLUSTER_RECEIPT", str(receipt_path))

    assert_artifact_bound_to_receipt(
        "processed_data",
        replica / "manifest.jsonl",
        allow_descendant=True,
    )
    with pytest.raises(RuntimeError, match="is not inside verified artifact"):
        assert_artifact_bound_to_receipt(
            "processed_data",
            other / "manifest.jsonl",
            allow_descendant=True,
        )


def test_hf_cache_revision_is_bound_to_declared_commit(tmp_path: Path) -> None:
    revision = "1" * 40
    reference = (
        tmp_path
        / "hf-cache"
        / "hub"
        / "models--nvidia--Cosmos3-Edge"
        / "refs"
        / "main"
    )
    reference.parent.mkdir(parents=True)
    reference.write_text(revision + "\n", encoding="utf-8")
    _validate_hf_cache_revision(tmp_path / "hf-cache", revision)
    with pytest.raises(ValueError, match="cache revision differs"):
        _validate_hf_cache_revision(tmp_path / "hf-cache", "2" * 40)
