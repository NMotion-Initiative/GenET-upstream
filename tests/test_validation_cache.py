import json
from pathlib import Path

from genet.config import ProjectConfig
from genet.training.cosmos import _validate_local_cosmos_copy
from genet.training.environment import (
    create_cluster_lock,
    create_cluster_receipt,
    verify_cluster_lock,
    write_cluster_lock,
    write_cluster_receipt,
)
from genet.training.validation_cache import (
    build_validation_attestation,
    validation_attestation_matches,
    validation_attestation_path,
    write_validation_attestation,
)


def _strict_attestation(
    tmp_path: Path,
    monkeypatch,
    *,
    manifest_sha256: str = "manifest-a",
    source_sha256: str = "source-a",
    num_frames: int = 81,
):
    processed = tmp_path / "processed"
    processed.mkdir(exist_ok=True)
    manifest = processed / "manifest.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    lock = create_cluster_lock({"processed_data": processed})
    lock_path = write_cluster_lock(tmp_path / "cluster-lock.json", lock)
    fingerprints = verify_cluster_lock(lock, {"processed_data": processed})
    receipt = create_cluster_receipt(
        lock_path,
        lock,
        {"processed_data": processed},
        fingerprints,
    )
    receipt_path = write_cluster_receipt(tmp_path / "node-receipt.json", receipt)

    monkeypatch.setenv("GENET_STRICT_ENV", "1")
    monkeypatch.setenv("GENET_LAUNCH_ID", "a" * 32)
    monkeypatch.setenv("GENET_CLUSTER_LOCK", str(lock_path))
    monkeypatch.setenv("GENET_CLUSTER_RECEIPT", str(receipt_path))
    monkeypatch.setenv("GENET_DATA_ARTIFACT", "processed_data")
    monkeypatch.setenv("GENET_NODE_RUN_ROOT", str(tmp_path / "run"))
    runtime = {
        "cluster_receipt_contract_sha256": "receipt-contract",
        "genet_source_sha256": source_sha256,
        "declared_code_revision": "1" * 40,
    }
    contract = {
        "cosmos": True,
        "num_frames": num_frames,
        "height": 192,
        "width": 320,
        "action_dim": 64,
        "require_bidirectional_pairs": True,
        "expected_embodiments": ["a", "b"],
        "local_world_size": 8,
    }
    expected = build_validation_attestation(
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        distributed_config_fingerprint="config-a",
        validation_contract=contract,
        runtime_signature=runtime,
    )
    assert expected is not None
    return manifest, expected


def test_validation_attestation_round_trip(tmp_path: Path, monkeypatch) -> None:
    _manifest, expected = _strict_attestation(tmp_path, monkeypatch)
    path = validation_attestation_path(expected)
    assert path == (
        tmp_path / "run" / "release" / "data-validation" / f"{'a' * 32}.json"
    )

    write_validation_attestation(path, expected)

    assert validation_attestation_matches(path, expected)


def test_validation_attestation_fails_closed_on_identity_change(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _manifest, expected = _strict_attestation(tmp_path, monkeypatch)
    path = validation_attestation_path(expected)
    assert path is not None
    write_validation_attestation(path, expected)

    _manifest, changed_manifest = _strict_attestation(
        tmp_path,
        monkeypatch,
        manifest_sha256="manifest-b",
    )
    _manifest, changed_source = _strict_attestation(
        tmp_path,
        monkeypatch,
        source_sha256="source-b",
    )
    _manifest, changed_contract = _strict_attestation(
        tmp_path,
        monkeypatch,
        num_frames=49,
    )

    assert not validation_attestation_matches(path, changed_manifest)
    assert not validation_attestation_matches(path, changed_source)
    assert not validation_attestation_matches(path, changed_contract)

    path.write_text("{not-json", encoding="utf-8")
    assert not validation_attestation_matches(path, expected)


def test_validation_attestation_disabled_without_strict_env(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    monkeypatch.delenv("GENET_STRICT_ENV", raising=False)

    assert (
        build_validation_attestation(
            manifest=manifest,
            manifest_sha256="manifest",
            distributed_config_fingerprint="config",
            validation_contract={"cosmos": True},
            runtime_signature={},
        )
        is None
    )


def test_attestation_json_contains_no_unstable_fields(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _manifest, expected = _strict_attestation(tmp_path, monkeypatch)
    rendered = json.dumps(expected, sort_keys=True)
    assert "cache_key" in rendered
    assert "verified_at" not in rendered


def test_cosmos_validation_uses_matching_attestation(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    manifest = processed / "manifest.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    lock = create_cluster_lock({"processed_data": processed})
    lock_path = write_cluster_lock(tmp_path / "cluster-lock.json", lock)
    fingerprints = verify_cluster_lock(lock, {"processed_data": processed})
    receipt = create_cluster_receipt(
        lock_path,
        lock,
        {"processed_data": processed},
        fingerprints,
    )
    receipt_path = write_cluster_receipt(tmp_path / "node-receipt.json", receipt)
    monkeypatch.setenv("GENET_STRICT_ENV", "1")
    monkeypatch.setenv("GENET_LAUNCH_ID", "b" * 32)
    monkeypatch.setenv("GENET_CLUSTER_LOCK", str(lock_path))
    monkeypatch.setenv("GENET_CLUSTER_RECEIPT", str(receipt_path))
    monkeypatch.setenv("GENET_DATA_ARTIFACT", "processed_data")
    monkeypatch.setenv("GENET_NODE_RUN_ROOT", str(tmp_path / "run"))
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    runtime = {
        "cluster_receipt_contract_sha256": "receipt-contract",
        "genet_source_sha256": "source",
        "declared_code_revision": "1" * 40,
    }
    project = ProjectConfig()
    contract = {
        "cosmos": True,
        "num_frames": project.data.num_frames,
        "height": project.data.height,
        "width": project.data.width,
        "action_dim": project.data.action_dim,
        "require_bidirectional_pairs": project.data.require_bidirectional_pairs,
        "expected_embodiments": project.data.expected_embodiments,
        "local_world_size": 1,
    }
    expected = build_validation_attestation(
        manifest=manifest,
        manifest_sha256="manifest",
        distributed_config_fingerprint=project.distributed_fingerprint(),
        validation_contract=contract,
        runtime_signature=runtime,
    )
    assert expected is not None
    path = validation_attestation_path(expected)
    assert path is not None
    write_validation_attestation(path, expected)

    def unexpected_validation(*_args, **_kwargs):
        raise AssertionError("full validation must not run on a cache hit")

    monkeypatch.setattr(
        "genet.cli.validate_data.validate_manifest",
        unexpected_validation,
    )
    _validate_local_cosmos_copy(
        project,
        manifest=manifest,
        manifest_sha256="manifest",
        runtime_signature=runtime,
    )

    assert "cosmos_data_validation_cache_hit" in capsys.readouterr().out
