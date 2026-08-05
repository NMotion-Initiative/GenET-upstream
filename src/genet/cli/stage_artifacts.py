"""Reproducibly stage the exact Cosmos3-Edge and Wan VAE artifacts."""

from __future__ import annotations

import argparse
import errno
import fcntl
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Any

ARTIFACT_FORMAT_VERSION = "genet.model-artifacts/v1"
DCP_MANIFEST_FORMAT_VERSION = "genet.dcp-files/v1"
DCP_MANIFEST_FILENAME = ".genet-files.json"
STAGE_LOCK_FILENAME = ".genet-stage-artifacts.lock"
COSMOS_REQUIRED_MODEL_FILES = (
    "config.json",
    "model.safetensors.index.json",
    "model_index.json",
    "preprocessor_config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
    "video_preprocessor_config.json",
    "vision_encoder/model.safetensors",
)


def _default_spec_path() -> Path:
    relative = Path("configs/checkpoints/cosmos3_edge.json")
    candidates = (
        Path(__file__).resolve().parents[3] / relative,
        Path("/opt/genet") / relative,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


DEFAULT_SPEC = _default_spec_path()


def _full_revision(value: Any, *, field: str) -> str:
    revision = str(value)
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise ValueError(f"{field} must be a full lowercase 40-hex commit")
    return revision


def _sha256_value(value: Any, *, field: str) -> str:
    digest = str(value)
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return digest


@dataclass(frozen=True)
class ArtifactSpec:
    format_version: str
    cosmos_repo_id: str
    cosmos_revision: str
    cosmos_framework_revision: str
    wan_repo_id: str
    wan_revision: str
    wan_filename: str
    wan_sha256: str
    wan_size: int
    spec_sha256: str

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, Any], *, spec_sha256: str
    ) -> ArtifactSpec:
        if data.get("format_version") != ARTIFACT_FORMAT_VERSION:
            raise ValueError(
                f"unsupported artifact spec format {data.get('format_version')!r}"
            )
        cosmos = data.get("cosmos3_edge")
        wan = data.get("wan_vae")
        if not isinstance(cosmos, Mapping) or not isinstance(wan, Mapping):
            raise ValueError("artifact spec needs cosmos3_edge and wan_vae objects")
        wan_size = wan.get("size")
        if isinstance(wan_size, bool) or not isinstance(wan_size, int) or wan_size <= 0:
            raise ValueError("wan_vae.size must be a positive integer")
        values = {
            "cosmos_repo_id": cosmos.get("repo_id"),
            "wan_repo_id": wan.get("repo_id"),
            "wan_filename": wan.get("filename"),
        }
        for field, value in values.items():
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field} must be a non-empty string")
        return cls(
            format_version=ARTIFACT_FORMAT_VERSION,
            cosmos_repo_id=values["cosmos_repo_id"],
            cosmos_revision=_full_revision(
                cosmos.get("revision"), field="cosmos3_edge.revision"
            ),
            cosmos_framework_revision=_full_revision(
                data.get("cosmos_framework_revision"),
                field="cosmos_framework_revision",
            ),
            wan_repo_id=values["wan_repo_id"],
            wan_revision=_full_revision(
                wan.get("revision"), field="wan_vae.revision"
            ),
            wan_filename=values["wan_filename"],
            wan_sha256=_sha256_value(
                wan.get("sha256"), field="wan_vae.sha256"
            ),
            wan_size=wan_size,
            spec_sha256=spec_sha256,
        )


@dataclass(frozen=True)
class StagePaths:
    run_root: Path
    hf_home: Path
    hub_cache: Path
    cosmos_snapshot: Path
    cosmos_ref: Path
    wan_vae: Path
    dcp_root: Path
    receipt: Path


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def load_artifact_spec(path: str | Path) -> ArtifactSpec:
    spec_path = Path(path).expanduser().resolve()
    try:
        raw = json.loads(spec_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read artifact spec {spec_path}: {exc}") from exc
    if not isinstance(raw, Mapping):
        raise ValueError("artifact spec root must be an object")
    return ArtifactSpec.from_mapping(raw, spec_sha256=_file_sha256(spec_path))


def _repo_cache_name(repo_id: str) -> str:
    return "models--" + repo_id.replace("/", "--")


def _require_exact_path(path: Path, expected: Path, *, name: str) -> None:
    if path != expected:
        raise ValueError(
            f"{name} must use the versioned node-local layout: "
            f"{path} != {expected}"
        )


def resolve_stage_paths(
    run_root: str | Path, spec: ArtifactSpec, *, cosmos_revision: str
) -> StagePaths:
    root = Path(run_root).expanduser().resolve()
    hf_home = Path(os.environ.get("HF_HOME", root / "hf-cache")).expanduser().resolve()
    hub_cache = Path(
        os.environ.get("HF_HUB_CACHE", hf_home / "hub")
    ).expanduser().resolve()
    _require_exact_path(hf_home, root / "hf-cache", name="HF_HOME")
    _require_exact_path(hub_cache, hf_home / "hub", name="HF_HUB_CACHE")
    model_cache = hub_cache / _repo_cache_name(spec.cosmos_repo_id)
    return StagePaths(
        run_root=root,
        hf_home=hf_home,
        hub_cache=hub_cache,
        cosmos_snapshot=model_cache / "snapshots" / cosmos_revision,
        cosmos_ref=model_cache / "refs" / "main",
        wan_vae=root / "artifacts" / "wan22_vae" / spec.wan_filename,
        dcp_root=root / "checkpoints" / "Cosmos3-Edge",
        receipt=root / "artifacts" / "ARTIFACTS.json",
    )


@contextmanager
def _exclusive_stage_lock(run_root: Path) -> Iterator[None]:
    """Serialize every mutating publication beneath one node-local run root."""

    run_root.mkdir(parents=True, exist_ok=True)
    lock_path = run_root / STAGE_LOCK_FILENAME
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno not in {errno.EACCES, errno.EAGAIN}:
                raise
            raise RuntimeError(
                f"another artifact staging process holds the lock: {lock_path}"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}-", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}-", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()


def verify_wan_vae(path: str | Path, spec: ArtifactSpec) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.is_file():
        raise FileNotFoundError(f"Wan VAE does not exist: {candidate}")
    size = candidate.stat().st_size
    if size != spec.wan_size:
        raise ValueError(
            f"Wan VAE size mismatch at {candidate}: {size} != {spec.wan_size}"
        )
    digest = _file_sha256(candidate)
    if digest != spec.wan_sha256:
        raise ValueError(
            f"Wan VAE SHA-256 mismatch at {candidate}: {digest} != {spec.wan_sha256}"
        )
    return {"path": str(candidate.resolve()), "size": size, "sha256": digest}


def _validate_snapshot_links(snapshot: Path) -> None:
    for candidate in snapshot.rglob("*"):
        if not candidate.is_symlink():
            continue
        try:
            target = candidate.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ValueError(f"Cosmos snapshot contains a broken link: {candidate}") from exc
        if not target.is_file() and not target.is_dir():
            raise ValueError(
                f"Cosmos snapshot link has an unsupported target: {candidate} -> {target}"
            )


def _indexed_weight_files(index_path: Path) -> set[Path]:
    try:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read Hugging Face weight index {index_path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"Hugging Face weight index must be an object: {index_path}")
    weight_map = payload.get("weight_map")
    if not isinstance(weight_map, Mapping) or not weight_map:
        raise ValueError(f"Hugging Face weight index has no weight_map: {index_path}")
    expected_suffix = (
        ".safetensors"
        if index_path.name.endswith(".safetensors.index.json")
        else ".bin"
    )

    shards: set[Path] = set()
    for parameter, raw_filename in weight_map.items():
        if not isinstance(parameter, str) or not parameter:
            raise ValueError(f"Hugging Face weight index has an invalid parameter name: {index_path}")
        if not isinstance(raw_filename, str) or not raw_filename:
            raise ValueError(
                f"Hugging Face weight index has an invalid shard for {parameter!r}: "
                f"{index_path}"
            )
        filename = PurePosixPath(raw_filename)
        if filename.is_absolute() or ".." in filename.parts:
            raise ValueError(
                f"Hugging Face weight index shard escapes its directory: "
                f"{raw_filename!r} in {index_path}"
            )
        if not filename.name.endswith(expected_suffix):
            raise ValueError(
                f"Hugging Face weight index references a non-{expected_suffix} shard: "
                f"{raw_filename!r} in {index_path}"
            )
        shard = index_path.parent.joinpath(*filename.parts)
        if not shard.is_file():
            raise FileNotFoundError(
                f"Hugging Face weight index references a missing shard: "
                f"{raw_filename!r} in {index_path}"
            )
        shards.add(shard)
    return shards


def validate_cosmos_snapshot(path: str | Path) -> Path:
    snapshot = Path(path).expanduser().resolve()
    if not snapshot.is_dir():
        raise FileNotFoundError(f"Cosmos snapshot does not exist: {snapshot}")
    _validate_snapshot_links(snapshot)
    missing_required = [
        relative
        for relative in COSMOS_REQUIRED_MODEL_FILES
        if not (snapshot / relative).is_file()
    ]
    if missing_required:
        raise FileNotFoundError(
            "Cosmos snapshot is missing required official model files: "
            f"{missing_required} beneath {snapshot}"
        )

    index_paths = sorted(
        set(snapshot.rglob("*.safetensors.index.json"))
        | set(snapshot.rglob("*.bin.index.json"))
    )
    indexed_weights: set[Path] = set()
    for index_path in index_paths:
        if not index_path.is_file():
            raise FileNotFoundError(f"Hugging Face weight index is missing: {index_path}")
        indexed_weights.update(_indexed_weight_files(index_path))

    weights = [
        candidate
        for candidate in (
            list(snapshot.rglob("*.safetensors")) + list(snapshot.rglob("*.bin"))
        )
        if candidate.is_file()
    ]
    if not weights:
        raise FileNotFoundError(f"Cosmos snapshot has no model weight files: {snapshot}")
    if index_paths and not indexed_weights:
        raise ValueError(f"Cosmos snapshot weight indexes reference no shards: {snapshot}")
    return snapshot


def _dcp_file_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    manifest_path = root / DCP_MANIFEST_FILENAME
    for candidate in sorted(
        root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
    ):
        if candidate == manifest_path:
            continue
        relative = candidate.relative_to(root).as_posix()
        if candidate.is_symlink():
            raise ValueError(f"converted DCP must not contain symlinks: {relative}")
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise ValueError(f"converted DCP contains a non-regular file: {relative}")
        records.append(
            {
                "path": relative,
                "size": candidate.stat().st_size,
                "sha256": _file_sha256(candidate),
            }
        )
    return records


def _write_dcp_manifest(path: str | Path) -> Path:
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"converted DCP does not exist: {root}")
    destination = root / DCP_MANIFEST_FILENAME
    _atomic_json(
        destination,
        {
            "format_version": DCP_MANIFEST_FORMAT_VERSION,
            "files": _dcp_file_records(root),
        },
    )
    return destination


def _read_dcp_manifest(root: Path) -> list[dict[str, Any]]:
    manifest_path = root / DCP_MANIFEST_FILENAME
    if manifest_path.is_symlink():
        raise ValueError(f"converted DCP file manifest must not be a symlink: {manifest_path}")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"converted DCP has no file manifest: {manifest_path}")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read converted DCP file manifest: {manifest_path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"converted DCP file manifest must be an object: {manifest_path}")
    if payload.get("format_version") != DCP_MANIFEST_FORMAT_VERSION:
        raise ValueError(f"unsupported converted DCP file manifest: {manifest_path}")
    raw_records = payload.get("files")
    if not isinstance(raw_records, list) or not raw_records:
        raise ValueError(f"converted DCP file manifest has no files: {manifest_path}")

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_record in raw_records:
        if not isinstance(raw_record, Mapping):
            raise ValueError(f"converted DCP file manifest has an invalid record: {manifest_path}")
        relative = raw_record.get("path")
        size = raw_record.get("size")
        digest = raw_record.get("sha256")
        if not isinstance(relative, str) or not relative:
            raise ValueError(f"converted DCP file manifest has an invalid path: {manifest_path}")
        logical_path = PurePosixPath(relative)
        if logical_path.is_absolute() or ".." in logical_path.parts:
            raise ValueError(
                f"converted DCP file manifest path escapes the DCP root: {relative!r}"
            )
        if relative == DCP_MANIFEST_FILENAME or relative in seen:
            raise ValueError(
                f"converted DCP file manifest has a duplicate or self record: {relative!r}"
            )
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(
                f"converted DCP file manifest has an invalid size for {relative!r}"
            )
        normalized_digest = _sha256_value(
            digest, field=f"converted DCP file manifest SHA-256 for {relative!r}"
        )
        seen.add(relative)
        records.append(
            {"path": relative, "size": size, "sha256": normalized_digest}
        )
    return records


def validate_cosmos_dcp(
    path: str | Path,
    *,
    cosmos_revision: str | None = None,
    framework_revision: str | None = None,
) -> Path:
    candidate_root = Path(path).expanduser()
    if candidate_root.is_symlink():
        raise ValueError(f"converted DCP root must not be a symlink: {candidate_root}")
    root = candidate_root.resolve()
    model = root / "model"
    if not (model / ".metadata").is_file():
        raise FileNotFoundError(f"converted DCP has no model/.metadata: {root}")
    if not (model / "config.json").is_file():
        raise FileNotFoundError(f"converted DCP has no model/config.json: {root}")
    if not any(model.rglob("*.distcp")):
        raise FileNotFoundError(f"converted DCP has no .distcp shards: {root}")
    recorded_files = _read_dcp_manifest(root)
    actual_files = _dcp_file_records(root)
    if recorded_files != actual_files:
        recorded_paths = {record["path"] for record in recorded_files}
        actual_paths = {record["path"] for record in actual_files}
        missing = sorted(recorded_paths - actual_paths)
        unexpected = sorted(actual_paths - recorded_paths)
        detail = f"missing={missing}, unexpected={unexpected}"
        if not missing and not unexpected:
            detail = "one or more file sizes or SHA-256 digests differ"
        raise ValueError(f"converted DCP file manifest mismatch: {detail}")
    source_path = root / ".genet-source.json"
    if cosmos_revision is not None or framework_revision is not None:
        if not source_path.is_file():
            raise FileNotFoundError(f"converted DCP has no source receipt: {source_path}")
        try:
            source = json.loads(source_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"cannot read converted DCP source receipt: {source_path}") from exc
        if not isinstance(source, Mapping):
            raise ValueError(f"converted DCP source receipt must be an object: {source_path}")
        if cosmos_revision is not None and source.get("cosmos_revision") != cosmos_revision:
            raise ValueError("converted DCP Cosmos revision differs from requested revision")
        if (
            framework_revision is not None
            and source.get("cosmos_framework_revision") != framework_revision
        ):
            raise ValueError("converted DCP framework revision differs from artifact spec")
    return root


def _resolve_main(spec: ArtifactSpec) -> str:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required; run staging inside the GenET Cosmos image"
        ) from exc
    revision = HfApi().model_info(spec.cosmos_repo_id, revision="main").sha
    return _full_revision(revision, field="resolved Cosmos3-Edge main")


def _download_snapshot(
    spec: ArtifactSpec, paths: StagePaths, revision: str, *, verify_only: bool
) -> Path:
    if verify_only:
        return validate_cosmos_snapshot(paths.cosmos_snapshot)
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required; run staging inside the GenET Cosmos image"
        ) from exc
    downloaded = Path(
        snapshot_download(
            repo_id=spec.cosmos_repo_id,
            revision=revision,
            cache_dir=str(paths.hub_cache),
        )
    ).resolve()
    expected = paths.cosmos_snapshot.resolve()
    if downloaded != expected:
        raise RuntimeError(
            f"Hugging Face returned unexpected snapshot path {downloaded}; expected {expected}"
        )
    return validate_cosmos_snapshot(downloaded)


def _update_main_ref(path: Path, revision: str, *, force: bool) -> None:
    if path.exists():
        current = path.read_text(encoding="utf-8").strip()
        if current == revision:
            return
        if not force:
            raise FileExistsError(
                f"{path} points at {current!r}; pass --force to repoint it to {revision}"
            )
    _atomic_text(path, revision + "\n")


def _require_ref_compatible(path: Path, revision: str, *, force: bool) -> None:
    """Fail before expensive work when publishing would overwrite another ref."""

    if path.is_symlink() and not path.exists():
        if force:
            return
        raise FileNotFoundError(
            f"{path} is a broken symlink; pass --force to replace it"
        )
    if not path.exists():
        return
    current = path.read_text(encoding="utf-8").strip()
    if current != revision and not force:
        raise FileExistsError(
            f"{path} points at {current!r}; pass --force to repoint it to {revision}"
        )


def _stage_wan(
    spec: ArtifactSpec,
    paths: StagePaths,
    *,
    verify_only: bool,
    force: bool,
) -> dict[str, Any]:
    if paths.wan_vae.is_file():
        try:
            return verify_wan_vae(paths.wan_vae, spec)
        except ValueError:
            if not force or verify_only:
                raise
    elif verify_only:
        return verify_wan_vae(paths.wan_vae, spec)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "huggingface_hub is required; run staging inside the GenET Cosmos image"
        ) from exc
    paths.wan_vae.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".wan22-download-", dir=paths.wan_vae.parent
    ) as temporary:
        downloaded = Path(
            hf_hub_download(
                repo_id=spec.wan_repo_id,
                filename=spec.wan_filename,
                revision=spec.wan_revision,
                local_dir=temporary,
            )
        )
        verify_wan_vae(downloaded, spec)
        os.replace(downloaded, paths.wan_vae)
    return verify_wan_vae(paths.wan_vae, spec)


def _framework_guard(spec: ArtifactSpec) -> None:
    observed = os.environ.get("GENET_COSMOS_REVISION")
    if observed != spec.cosmos_framework_revision:
        raise RuntimeError(
            "conversion must run in the pinned GenET image: "
            f"GENET_COSMOS_REVISION={observed!r}, expected "
            f"{spec.cosmos_framework_revision!r}"
        )
    cosmos_dir = Path(os.environ.get("COSMOS_DIR", "/opt/cosmos-framework")).expanduser()
    if not (cosmos_dir / ".git").exists():
        return
    try:
        checkout_revision = subprocess.run(
            ["git", "-C", str(cosmos_dir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"cannot inspect Cosmos checkout at {cosmos_dir}") from exc
    if checkout_revision != spec.cosmos_framework_revision:
        raise RuntimeError(
            f"Cosmos checkout at {cosmos_dir} is {checkout_revision!r}, expected "
            f"{spec.cosmos_framework_revision!r}"
        )


def _convert_dcp(
    spec: ArtifactSpec,
    paths: StagePaths,
    *,
    revision: str,
    verify_only: bool,
    force: bool,
) -> Path:
    if paths.dcp_root.exists():
        try:
            return validate_cosmos_dcp(
                paths.dcp_root,
                cosmos_revision=revision,
                framework_revision=spec.cosmos_framework_revision,
            )
        except (FileNotFoundError, ValueError):
            if not force or verify_only:
                raise
    elif verify_only:
        return validate_cosmos_dcp(
            paths.dcp_root,
            cosmos_revision=revision,
            framework_revision=spec.cosmos_framework_revision,
        )
    _framework_guard(spec)
    paths.dcp_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=".Cosmos3-Edge-dcp-", dir=paths.dcp_root.parent)
    )
    backup: Path | None = None
    try:
        converter_environment = os.environ.copy()
        converter_environment.update(
            {
                "HF_HOME": str(paths.hf_home),
                "HF_HUB_CACHE": str(paths.hub_cache),
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "cosmos_framework.scripts.convert_model_to_dcp",
                "-o",
                str(temporary),
                "--checkpoint-path",
                str(paths.cosmos_snapshot),
            ],
            check=True,
            env=converter_environment,
        )
        _atomic_json(
            temporary / ".genet-source.json",
            {
                "cosmos_repo_id": spec.cosmos_repo_id,
                "cosmos_revision": revision,
                "cosmos_framework_revision": spec.cosmos_framework_revision,
                "snapshot": str(paths.cosmos_snapshot),
            },
        )
        _write_dcp_manifest(temporary)
        validate_cosmos_dcp(
            temporary,
            cosmos_revision=revision,
            framework_revision=spec.cosmos_framework_revision,
        )
        if paths.dcp_root.exists():
            backup = paths.dcp_root.with_name(
                f"{paths.dcp_root.name}.replaced-"
                f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}"
            )
            paths.dcp_root.rename(backup)
            print(f"Preserved replaced DCP at {backup}", file=sys.stderr)
        try:
            temporary.rename(paths.dcp_root)
        except Exception:
            if backup is not None and backup.exists() and not paths.dcp_root.exists():
                backup.rename(paths.dcp_root)
            raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return validate_cosmos_dcp(
        paths.dcp_root,
        cosmos_revision=revision,
        framework_revision=spec.cosmos_framework_revision,
    )


def _forbid_download_only_receipt_downgrade(receipt_path: Path) -> None:
    if not receipt_path.exists():
        return
    if not receipt_path.is_file():
        raise ValueError(f"artifact receipt is not a regular file: {receipt_path}")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot inspect existing artifact receipt: {receipt_path}") from exc
    cosmos = receipt.get("cosmos3_edge") if isinstance(receipt, Mapping) else None
    if isinstance(cosmos, Mapping) and (
        cosmos.get("dcp_root") is not None or cosmos.get("dcp_load_path") is not None
    ):
        raise ValueError(
            "--download-only cannot replace a complete artifact receipt; "
            "retain the converted DCP receipt or stage into a different --run-root"
        )


def _preflight_mutating_stage(
    spec: ArtifactSpec,
    paths: StagePaths,
    *,
    revision: str,
    download_only: bool,
    force: bool,
) -> None:
    _require_ref_compatible(paths.cosmos_ref, revision, force=force)
    if download_only:
        _forbid_download_only_receipt_downgrade(paths.receipt)
    else:
        _framework_guard(spec)

    if paths.wan_vae.exists() or paths.wan_vae.is_symlink():
        try:
            verify_wan_vae(paths.wan_vae, spec)
        except (FileNotFoundError, ValueError):
            if not force:
                raise

    if download_only or (
        not paths.dcp_root.exists() and not paths.dcp_root.is_symlink()
    ):
        return
    if paths.dcp_root.is_symlink():
        raise ValueError(f"converted DCP root must not be a symlink: {paths.dcp_root}")
    try:
        validate_cosmos_dcp(
            paths.dcp_root,
            cosmos_revision=revision,
            framework_revision=spec.cosmos_framework_revision,
        )
    except (FileNotFoundError, ValueError):
        if not force:
            raise


def _reject_persisted_hf_credentials(paths: StagePaths) -> None:
    leaked = [
        candidate
        for candidate in (paths.hf_home / "token", paths.hf_home / "stored_tokens")
        if candidate.exists() or candidate.is_symlink()
    ]
    if leaked:
        raise ValueError(
            "Hugging Face credential files must not be staged or replicated; "
            f"remove {leaked} and pass any short-lived token only through HF_TOKEN"
        )


def _receipt_payload(
    spec: ArtifactSpec,
    paths: StagePaths,
    *,
    revision: str,
    wan: Mapping[str, Any],
    converted: bool,
) -> dict[str, Any]:
    return {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "spec_sha256": spec.spec_sha256,
        "cosmos_framework_revision": spec.cosmos_framework_revision,
        "cosmos3_edge": {
            "repo_id": spec.cosmos_repo_id,
            "revision": revision,
            "snapshot": str(paths.cosmos_snapshot),
            "ref_main": str(paths.cosmos_ref),
            "dcp_root": str(paths.dcp_root) if converted else None,
            "dcp_load_path": str(paths.dcp_root) if converted else None,
        },
        "wan_vae": {
            "repo_id": spec.wan_repo_id,
            "revision": spec.wan_revision,
            "filename": spec.wan_filename,
            **dict(wan),
        },
        "hf_home": str(paths.hf_home),
        "hub_cache": str(paths.hub_cache),
    }


def _stage_artifacts_once(
    spec: ArtifactSpec,
    *,
    paths: StagePaths,
    revision: str,
    download_only: bool = False,
    verify_only: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    if not verify_only:
        _preflight_mutating_stage(
            spec,
            paths,
            revision=revision,
            download_only=download_only,
            force=force,
        )
    snapshot = _download_snapshot(spec, paths, revision, verify_only=verify_only)
    if snapshot != paths.cosmos_snapshot.resolve():
        raise RuntimeError("resolved Cosmos snapshot path changed during staging")
    if verify_only:
        if not paths.cosmos_ref.is_file():
            raise FileNotFoundError(f"Cosmos refs/main is missing: {paths.cosmos_ref}")
        if paths.cosmos_ref.read_text(encoding="utf-8").strip() != revision:
            raise ValueError("Cosmos refs/main does not contain the requested revision")
    else:
        _require_ref_compatible(paths.cosmos_ref, revision, force=force)
    wan = _stage_wan(spec, paths, verify_only=verify_only, force=force)
    converted = not download_only
    if converted:
        _convert_dcp(
            spec,
            paths,
            revision=revision,
            verify_only=verify_only,
            force=force,
        )
    if not verify_only:
        # Publish the mutable alias only after every requested immutable artifact
        # has been verified and (when requested) converted successfully.
        _update_main_ref(paths.cosmos_ref, revision, force=force)
    payload = _receipt_payload(
        spec, paths, revision=revision, wan=wan, converted=converted
    )
    if verify_only:
        if not paths.receipt.is_file():
            raise FileNotFoundError(f"artifact receipt is missing: {paths.receipt}")
        existing = json.loads(paths.receipt.read_text(encoding="utf-8"))
        expected_identity = dict(payload)
        observed_identity = dict(existing)
        expected_identity.pop("created_at", None)
        observed_identity.pop("created_at", None)
        if observed_identity != expected_identity:
            raise ValueError("artifact receipt differs from verified paths or identities")
        return existing
    _atomic_json(paths.receipt, payload)
    return payload


def stage_artifacts(
    spec: ArtifactSpec,
    *,
    run_root: str | Path,
    cosmos_revision: str | None = None,
    resolve_main: bool = False,
    download_only: bool = False,
    verify_only: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    if resolve_main and cosmos_revision is not None:
        raise ValueError("--resolve-main and --cosmos-revision are mutually exclusive")
    if verify_only and resolve_main:
        raise ValueError("--verify-only cannot resolve mutable network main")

    root = Path(run_root).expanduser().resolve()

    def run_once() -> dict[str, Any]:
        revision = (
            _resolve_main(spec)
            if resolve_main
            else _full_revision(
                cosmos_revision or spec.cosmos_revision,
                field="Cosmos3-Edge revision",
            )
        )
        paths = resolve_stage_paths(root, spec, cosmos_revision=revision)
        _reject_persisted_hf_credentials(paths)
        return _stage_artifacts_once(
            spec,
            paths=paths,
            revision=revision,
            download_only=download_only,
            verify_only=verify_only,
            force=force,
        )

    if verify_only:
        return run_once()
    with _exclusive_stage_lock(root):
        return run_once()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="genet-stage-artifacts",
        description=(
            "Download, verify, and convert the exact GenET base artifacts. "
            "Run inside the immutable GenET Cosmos image on one staging node."
        ),
    )
    parser.add_argument(
        "--spec",
        type=Path,
        default=DEFAULT_SPEC,
    )
    parser.add_argument("--run-root", type=Path, default=Path("/mnt/nvme/genet"))
    parser.add_argument("--cosmos-revision", help="Reviewed full 40-hex override")
    parser.add_argument(
        "--resolve-main",
        action="store_true",
        help="Explicitly resolve current upstream main for review and staging",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download/verify the HF snapshot and VAE but do not convert DCP",
    )
    parser.add_argument(
        "--verify-only", action="store_true", help="Use no network and mutate nothing"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace mismatched refs/artifacts while preserving an old DCP backup",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print exact resolved paths and exit"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    spec = load_artifact_spec(args.spec)
    if args.resolve_main and args.dry_run:
        _parser().error("--dry-run cannot resolve network main; pass a reviewed revision")
    revision = _full_revision(
        args.cosmos_revision or spec.cosmos_revision, field="Cosmos3-Edge revision"
    )
    if args.dry_run:
        paths = resolve_stage_paths(args.run_root, spec, cosmos_revision=revision)
        print(
            json.dumps(
                {
                    "spec": asdict(spec),
                    "paths": {key: str(value) for key, value in asdict(paths).items()},
                    "download_only": args.download_only,
                    "verify_only": args.verify_only,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    payload = stage_artifacts(
        spec,
        run_root=args.run_root,
        cosmos_revision=args.cosmos_revision,
        resolve_main=args.resolve_main,
        download_only=args.download_only,
        verify_only=args.verify_only,
        force=args.force,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
