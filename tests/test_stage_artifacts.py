from __future__ import annotations

import fcntl
import json
import os
import subprocess
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pytest

import genet.cli.stage_artifacts as staging
from genet.cli.stage_artifacts import (
    COSMOS_REQUIRED_MODEL_FILES,
    DCP_MANIFEST_FILENAME,
    STAGE_LOCK_FILENAME,
    _convert_dcp,
    _framework_guard,
    _write_dcp_manifest,
    load_artifact_spec,
    main,
    resolve_stage_paths,
    stage_artifacts,
    validate_cosmos_dcp,
    validate_cosmos_snapshot,
    verify_wan_vae,
)

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "configs" / "checkpoints" / "cosmos3_edge.json"


@pytest.fixture(autouse=True)
def _clean_staging_environment(monkeypatch: pytest.MonkeyPatch):
    for name in (
        "COSMOS_DIR",
        "GENET_COSMOS_REVISION",
        "HF_HOME",
        "HF_HUB_CACHE",
    ):
        monkeypatch.delenv(name, raising=False)


def _write_test_dcp(root: Path, spec, *, shard_count: int = 2) -> Path:
    model = root / "model"
    model.mkdir(parents=True)
    (model / ".metadata").write_bytes(b"metadata")
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    for index in range(shard_count):
        (model / f"weights-{index}.distcp").write_bytes(
            f"weights-{index}".encode()
        )
    (root / ".genet-source.json").write_text(
        json.dumps(
            {
                "cosmos_revision": spec.cosmos_revision,
                "cosmos_framework_revision": spec.cosmos_framework_revision,
            }
        ),
        encoding="utf-8",
    )
    _write_dcp_manifest(root)
    return root


def _write_indexed_snapshot(root: Path) -> Path:
    root.mkdir(parents=True)
    for relative in COSMOS_REQUIRED_MODEL_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.name.endswith(".safetensors") and not path.name.endswith(
            ".index.json"
        ):
            path.write_text("{}\n", encoding="utf-8")
    (root / "model-00001-of-00002.safetensors").write_bytes(b"one")
    (root / "model-00002-of-00002.safetensors").write_bytes(b"two")
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "layer.0": "model-00001-of-00002.safetensors",
                    "layer.1": "model-00002-of-00002.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    transformer = root / "transformer"
    (transformer / "diffusion_pytorch_model-00001-of-00002.safetensors").write_bytes(
        b"transformer-one"
    )
    (transformer / "diffusion_pytorch_model-00002-of-00002.safetensors").write_bytes(
        b"transformer-two"
    )
    (transformer / "diffusion_pytorch_model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "transformer.layer.0": (
                        "diffusion_pytorch_model-00001-of-00002.safetensors"
                    ),
                    "transformer.layer.1": (
                        "diffusion_pytorch_model-00002-of-00002.safetensors"
                    ),
                }
            }
        ),
        encoding="utf-8",
    )
    (root / "vae" / "diffusion_pytorch_model.safetensors").write_bytes(b"vae")
    (root / "vision_encoder" / "model.safetensors").write_bytes(b"vision")
    return root


def test_committed_artifact_spec_has_exact_real_locations():
    spec = load_artifact_spec(SPEC)
    assert spec.cosmos_repo_id == "nvidia/Cosmos3-Edge"
    assert spec.cosmos_revision == "2a00e87e9976dc3ed5533dd18caf4cdbc3a1bcb2"
    assert spec.cosmos_framework_revision == "a904d2d36b774a51dd06ff9ff906816b1a04f579"
    assert spec.wan_repo_id == "Wan-AI/Wan2.2-TI2V-5B"
    assert spec.wan_revision == "921dbaf3f1674a56f47e83fb80a34bac8a8f203e"
    assert spec.wan_sha256 == "20eb789667fa5e60e7516bf509512f6cb61f01b0aa0695eadaea930c13892b36"
    assert spec.wan_size == 2_818_839_170


def test_stage_dry_run_prints_node_local_paths(tmp_path: Path, capsys):
    assert main(["--spec", str(SPEC), "--run-root", str(tmp_path), "--dry-run"]) == 0
    payload = json.loads(capsys.readouterr().out)
    paths = payload["paths"]
    assert paths["wan_vae"] == str(
        tmp_path / "artifacts" / "wan22_vae" / "Wan2.2_VAE.pth"
    )
    assert paths["dcp_root"] == str(tmp_path / "checkpoints" / "Cosmos3-Edge")
    assert paths["cosmos_snapshot"].endswith(
        "/models--nvidia--Cosmos3-Edge/snapshots/"
        "2a00e87e9976dc3ed5533dd18caf4cdbc3a1bcb2"
    )


def test_default_spec_is_independent_of_current_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
):
    monkeypatch.chdir(tmp_path)
    assert main(["--run-root", str(tmp_path / "run"), "--dry-run"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["spec"]["cosmos_repo_id"] == "nvidia/Cosmos3-Edge"


def test_wan_checksum_and_dcp_source_receipt_validation(tmp_path: Path):
    spec = load_artifact_spec(SPEC)
    content = b"tiny-test-vae"
    tiny = replace(spec, wan_size=len(content), wan_sha256=sha256(content).hexdigest())
    vae = tmp_path / "vae.pth"
    vae.write_bytes(content)
    assert verify_wan_vae(vae, tiny)["sha256"] == tiny.wan_sha256
    vae.write_bytes(content + b"bad")
    with pytest.raises(ValueError, match="size mismatch"):
        verify_wan_vae(vae, tiny)

    paths = resolve_stage_paths(tmp_path, spec, cosmos_revision=spec.cosmos_revision)
    _write_test_dcp(paths.dcp_root, spec)
    assert (
        validate_cosmos_dcp(
            paths.dcp_root,
            cosmos_revision=spec.cosmos_revision,
            framework_revision=spec.cosmos_framework_revision,
        )
        == paths.dcp_root
    )


def test_snapshot_validates_every_index_shard_and_rejects_broken_links(
    tmp_path: Path,
):
    snapshot = _write_indexed_snapshot(tmp_path / "snapshot")
    assert validate_cosmos_snapshot(snapshot) == snapshot

    missing = snapshot / "model-00002-of-00002.safetensors"
    missing.unlink()
    with pytest.raises(FileNotFoundError, match="missing shard"):
        validate_cosmos_snapshot(snapshot)

    missing.write_bytes(b"two")
    (snapshot / "broken.bin").symlink_to(snapshot / "missing-blob")
    with pytest.raises(ValueError, match="broken link"):
        validate_cosmos_snapshot(snapshot)


def test_snapshot_rejects_a_partial_official_model_layout(tmp_path: Path):
    snapshot = _write_indexed_snapshot(tmp_path / "snapshot")
    (snapshot / "processor_config.json").unlink()
    with pytest.raises(FileNotFoundError, match="required official model files"):
        validate_cosmos_snapshot(snapshot)


def test_dcp_manifest_detects_recursive_content_changes_and_missing_shards(
    tmp_path: Path,
):
    spec = load_artifact_spec(SPEC)
    dcp = _write_test_dcp(tmp_path / "dcp", spec)
    manifest = json.loads((dcp / DCP_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    recorded_paths = {record["path"] for record in manifest["files"]}
    assert "model/weights-0.distcp" in recorded_paths
    assert ".genet-source.json" in recorded_paths

    (dcp / "model" / "weights-0.distcp").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="digests differ"):
        validate_cosmos_dcp(dcp)

    _write_dcp_manifest(dcp)
    (dcp / "model" / "weights-1.distcp").unlink()
    with pytest.raises(ValueError, match="missing=.*weights-1.distcp"):
        validate_cosmos_dcp(dcp)


def test_hf_cache_paths_must_use_the_versioned_run_root_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    spec = load_artifact_spec(SPEC)
    run_root = tmp_path / "run"
    monkeypatch.setenv("HF_HOME", str(tmp_path / "outside"))
    with pytest.raises(ValueError, match="HF_HOME"):
        resolve_stage_paths(run_root, spec, cosmos_revision=spec.cosmos_revision)

    monkeypatch.setenv("HF_HOME", str(run_root / "hf-cache"))
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "outside-hub"))
    with pytest.raises(ValueError, match="HF_HUB_CACHE"):
        resolve_stage_paths(run_root, spec, cosmos_revision=spec.cosmos_revision)

    monkeypatch.setenv("HF_HOME", str(run_root / "custom-cache"))
    monkeypatch.delenv("HF_HUB_CACHE")
    with pytest.raises(ValueError, match="versioned node-local layout"):
        resolve_stage_paths(run_root, spec, cosmos_revision=spec.cosmos_revision)


def test_stage_rejects_persisted_hf_credentials_before_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    spec = load_artifact_spec(SPEC)
    token = tmp_path / "hf-cache" / "token"
    token.parent.mkdir(parents=True)
    token.write_text("must-not-be-replicated\n", encoding="utf-8")

    def unexpected_download(*args, **kwargs):
        raise AssertionError("snapshot download must not run before credential check")

    monkeypatch.setattr(staging, "_download_snapshot", unexpected_download)
    with pytest.raises(ValueError, match="credential files must not be staged"):
        stage_artifacts(spec, run_root=tmp_path, download_only=True)


def test_mutating_stage_rejects_a_concurrent_run_root_lock(tmp_path: Path):
    spec = load_artifact_spec(SPEC)
    run_root = tmp_path / "run"
    run_root.mkdir()
    descriptor = os.open(
        run_root / STAGE_LOCK_FILENAME,
        os.O_CREAT | os.O_RDWR,
        0o600,
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="another artifact staging process"):
            stage_artifacts(spec, run_root=run_root, download_only=True)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def test_ref_conflict_fails_before_snapshot_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    spec = load_artifact_spec(SPEC)
    paths = resolve_stage_paths(tmp_path, spec, cosmos_revision=spec.cosmos_revision)
    paths.cosmos_ref.parent.mkdir(parents=True)
    paths.cosmos_ref.write_text("f" * 40 + "\n", encoding="utf-8")

    def unexpected_download(*args, **kwargs):
        raise AssertionError("snapshot download must not run before preflight")

    monkeypatch.setattr(staging, "_download_snapshot", unexpected_download)
    with pytest.raises(FileExistsError, match="pass --force"):
        stage_artifacts(spec, run_root=tmp_path, download_only=True)


def test_download_only_cannot_downgrade_a_complete_receipt_before_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    spec = load_artifact_spec(SPEC)
    paths = resolve_stage_paths(tmp_path, spec, cosmos_revision=spec.cosmos_revision)
    paths.receipt.parent.mkdir(parents=True)
    paths.receipt.write_text(
        json.dumps(
            {
                "cosmos3_edge": {
                    "dcp_root": str(paths.dcp_root),
                    "dcp_load_path": str(paths.dcp_root),
                }
            }
        ),
        encoding="utf-8",
    )

    def unexpected_download(*args, **kwargs):
        raise AssertionError("snapshot download must not run before preflight")

    monkeypatch.setattr(staging, "_download_snapshot", unexpected_download)
    with pytest.raises(ValueError, match="cannot replace a complete artifact receipt"):
        stage_artifacts(
            spec,
            run_root=tmp_path,
            download_only=True,
            force=True,
        )


def test_framework_preflight_and_git_checkout_guard_run_before_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    spec = load_artifact_spec(SPEC)

    def unexpected_download(*args, **kwargs):
        raise AssertionError("snapshot download must not run before preflight")

    monkeypatch.setattr(staging, "_download_snapshot", unexpected_download)
    with pytest.raises(RuntimeError, match="GENET_COSMOS_REVISION"):
        stage_artifacts(spec, run_root=tmp_path)

    cosmos_dir = tmp_path / "cosmos"
    (cosmos_dir / ".git").mkdir(parents=True)
    monkeypatch.setenv("GENET_COSMOS_REVISION", spec.cosmos_framework_revision)
    monkeypatch.setenv("COSMOS_DIR", str(cosmos_dir))
    monkeypatch.setattr(
        staging.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, stdout="0" * 40 + "\n"
        ),
    )
    with pytest.raises(RuntimeError, match="Cosmos checkout"):
        _framework_guard(spec)


def test_converter_is_offline_and_writes_a_strict_dcp_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    spec = load_artifact_spec(SPEC)
    paths = resolve_stage_paths(tmp_path, spec, cosmos_revision=spec.cosmos_revision)
    cosmos_dir = tmp_path / "cosmos-without-git"
    cosmos_dir.mkdir()
    monkeypatch.setenv("GENET_COSMOS_REVISION", spec.cosmos_framework_revision)
    monkeypatch.setenv("COSMOS_DIR", str(cosmos_dir))
    observed_environment = {}

    def fake_converter(command, *, check, env):
        assert check is True
        observed_environment.update(env)
        output = Path(command[command.index("-o") + 1])
        model = output / "model"
        model.mkdir(parents=True)
        (model / ".metadata").write_bytes(b"metadata")
        (model / "config.json").write_text("{}\n", encoding="utf-8")
        (model / "weights.distcp").write_bytes(b"weights")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(staging.subprocess, "run", fake_converter)
    assert (
        _convert_dcp(
            spec,
            paths,
            revision=spec.cosmos_revision,
            verify_only=False,
            force=False,
        )
        == paths.dcp_root
    )
    assert observed_environment["HF_HUB_OFFLINE"] == "1"
    assert observed_environment["TRANSFORMERS_OFFLINE"] == "1"
    assert observed_environment["HF_HOME"] == str(paths.hf_home)
    assert observed_environment["HF_HUB_CACHE"] == str(paths.hub_cache)
    assert (paths.dcp_root / DCP_MANIFEST_FILENAME).is_file()
