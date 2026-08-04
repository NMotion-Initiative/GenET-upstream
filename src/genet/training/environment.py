"""Reproducibility checks for multi-node training without shared storage."""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from genet.training.distributed import assert_same_across_ranks, raise_if_any_rank_failed, sha256_file

CLUSTER_LOCK_VERSION = "genet.cluster-lock/v1"
CLUSTER_RECEIPT_VERSION = "genet.cluster-lock-receipt/v1"
PINNED_COSMOS_REVISION = "a904d2d36b774a51dd06ff9ff906816b1a04f579"
_SHA256_RE = re.compile(r"^(?:sha256:)?[0-9a-fA-F]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_DISTRIBUTED_ENV_KEYS = {
    "GLOO_SOCKET_IFNAME",
    "GENET_CHECKPOINT_ARTIFACT",
    "GENET_DATA_ARTIFACT",
    "GENET_HF_ARTIFACT",
    "GENET_WAN_VAE_ARTIFACT",
    "HF_HUB_OFFLINE",
    "NCCL_ALGO",
    "NCCL_DEBUG",
    "NCCL_DEBUG_SUBSYS",
    "NCCL_IB_DISABLE",
    "NCCL_IB_GID_INDEX",
    "NCCL_IB_HCA",
    "NCCL_IB_RETRY_CNT",
    "NCCL_IB_SL",
    "NCCL_IB_TC",
    "NCCL_IB_TIMEOUT",
    "NCCL_NET",
    "NCCL_NET_GDR_LEVEL",
    "NCCL_P2P_LEVEL",
    "NCCL_PROTO",
    "NCCL_SHM_DISABLE",
    "NCCL_SOCKET_IFNAME",
    "TORCH_NCCL_ASYNC_ERROR_HANDLING",
    "TORCH_NCCL_BLOCKING_WAIT",
    "TRANSFORMERS_OFFLINE",
    "GENET_STRICT_ENV",
}


@dataclass(frozen=True)
class ArtifactFingerprint:
    kind: str
    sha256: str
    file_count: int
    total_bytes: int


def _hash_record(
    digest: Any,
    relative: str,
    size: int,
    content_hash: str,
    *,
    kind: str = "file",
) -> None:
    payload = json.dumps(
        {"kind": kind, "path": relative, "size": size, "sha256": content_hash},
        sort_keys=True,
        separators=(",", ":"),
    )
    digest.update(payload.encode("utf-8"))
    digest.update(b"\n")


def fingerprint_artifact(path: str | Path) -> ArtifactFingerprint:
    """Hash a file or directory independently of its node-local absolute path."""

    source = Path(path).expanduser()
    if source.is_symlink():
        raise ValueError(f"artifact root must not be a symlink: {source}")
    root = source.resolve()
    if not root.exists():
        raise FileNotFoundError(f"artifact does not exist: {root}")
    if root.is_file():
        size = root.stat().st_size
        content_hash = sha256_file(root)
        aggregate = hashlib.sha256()
        _hash_record(aggregate, ".", size, content_hash)
        return ArtifactFingerprint("file", aggregate.hexdigest(), 1, size)
    if not root.is_dir():
        raise ValueError(f"artifact must be a regular file or directory: {root}")

    aggregate = hashlib.sha256()
    count = 0
    total_bytes = 0
    for candidate in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if candidate.is_symlink():
            try:
                target = candidate.resolve(strict=True)
                target_relative = target.relative_to(root).as_posix()
            except (FileNotFoundError, ValueError) as exc:
                raise ValueError(
                    "artifact symlinks must resolve inside the artifact root: "
                    f"{candidate}"
                ) from exc
            relative = candidate.relative_to(root).as_posix()
            target_bytes = target_relative.encode("utf-8")
            link_hash = hashlib.sha256(b"symlink\0" + target_bytes).hexdigest()
            _hash_record(
                aggregate,
                relative,
                len(target_bytes),
                link_hash,
                kind="symlink",
            )
            count += 1
            total_bytes += len(target_bytes)
            continue
        if not candidate.is_file():
            continue
        relative = candidate.relative_to(root).as_posix()
        size = candidate.stat().st_size
        _hash_record(aggregate, relative, size, sha256_file(candidate))
        count += 1
        total_bytes += size
    return ArtifactFingerprint("directory", aggregate.hexdigest(), count, total_bytes)


def create_cluster_lock(artifacts: Mapping[str, str | Path]) -> dict[str, Any]:
    """Create a deterministic logical-name-to-content-fingerprint contract."""

    if not artifacts:
        raise ValueError("at least one artifact is required")
    fingerprints: dict[str, dict[str, Any]] = {}
    for name, path in sorted(artifacts.items()):
        if not name or any(character.isspace() for character in name):
            raise ValueError(f"artifact name must be non-empty and contain no whitespace: {name!r}")
        fingerprints[name] = asdict(fingerprint_artifact(path))
    return {"format_version": CLUSTER_LOCK_VERSION, "artifacts": fingerprints}


def write_cluster_lock(path: str | Path, lock: Mapping[str, Any]) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def load_cluster_lock(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as handle:
        lock = json.load(handle)
    if not isinstance(lock, dict) or lock.get("format_version") != CLUSTER_LOCK_VERSION:
        raise ValueError(f"unsupported cluster lock format: {source}")
    artifacts = lock.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError(f"cluster lock contains no artifacts: {source}")
    return lock


def create_cluster_receipt(
    lock_path: str | Path,
    lock: Mapping[str, Any],
    artifacts: Mapping[str, str | Path],
    fingerprints: Mapping[str, ArtifactFingerprint],
) -> dict[str, Any]:
    """Record the local paths that were verified against a shared content lock."""

    expected = lock.get("artifacts")
    if not isinstance(expected, dict) or set(expected) != set(artifacts):
        raise ValueError("receipt artifacts must exactly match the cluster lock")
    records: dict[str, dict[str, Any]] = {}
    for name in sorted(expected):
        fingerprint = fingerprints[name]
        record = asdict(fingerprint)
        if record != expected[name]:
            raise ValueError(f"verified fingerprint for {name!r} differs from the lock")
        record["path"] = str(Path(artifacts[name]).expanduser().resolve())
        records[name] = record
    return {
        "format_version": CLUSTER_RECEIPT_VERSION,
        "lock_sha256": sha256_file(lock_path),
        "artifacts": records,
    }


def write_cluster_receipt(path: str | Path, receipt: Mapping[str, Any]) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination


def load_cluster_receipt(
    receipt_path: str | Path,
    lock_path: str | Path,
) -> dict[str, Any]:
    """Validate a node-local receipt without rescanning the large artifacts."""

    receipt_source = Path(receipt_path).expanduser().resolve()
    with receipt_source.open("r", encoding="utf-8") as handle:
        receipt = json.load(handle)
    if not isinstance(receipt, dict) or receipt.get("format_version") != CLUSTER_RECEIPT_VERSION:
        raise ValueError(f"unsupported cluster receipt format: {receipt_source}")
    actual_lock_hash = sha256_file(lock_path)
    if receipt.get("lock_sha256") != actual_lock_hash:
        raise ValueError("cluster receipt was produced from a different lock file")
    lock = load_cluster_lock(lock_path)
    expected = lock["artifacts"]
    records = receipt.get("artifacts")
    if not isinstance(records, dict) or set(records) != set(expected):
        raise ValueError("cluster receipt artifacts differ from the lock")
    for name, expected_fingerprint in expected.items():
        record = records[name]
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ValueError(f"invalid receipt record for artifact {name!r}")
        fingerprint = {key: value for key, value in record.items() if key != "path"}
        if fingerprint != expected_fingerprint:
            raise ValueError(f"receipt fingerprint for {name!r} differs from the lock")
    return receipt


def _receipt_contract_sha256(receipt: Mapping[str, Any]) -> str:
    records = receipt["artifacts"]
    path_independent = {
        name: {key: value for key, value in record.items() if key != "path"}
        for name, record in records.items()
    }
    return _digest_json(
        {"lock_sha256": receipt["lock_sha256"], "artifacts": path_independent}
    )


def assert_artifact_bound_to_receipt(
    logical_name: str,
    actual_path: str | Path,
    *,
    allow_descendant: bool = False,
) -> None:
    """Collectively bind one runtime path to the node-local verified receipt."""

    receipt_path = os.environ.get("GENET_CLUSTER_RECEIPT")
    lock_path = os.environ.get("GENET_CLUSTER_LOCK")
    strict = os.environ.get("GENET_STRICT_ENV", "0").lower() in {"1", "true", "yes", "on"}
    if not receipt_path or not lock_path:
        if strict:
            raise_if_any_rank_failed(
                f"artifact binding {logical_name}",
                "strict mode requires GENET_CLUSTER_LOCK and GENET_CLUSTER_RECEIPT",
            )
        return

    error: str | None = None
    try:
        receipt = load_cluster_receipt(receipt_path, lock_path)
        records = receipt["artifacts"]
        if logical_name not in records:
            raise KeyError(f"cluster receipt has no artifact named {logical_name!r}")
        verified_path = Path(records[logical_name]["path"]).expanduser().resolve()
        runtime_path = Path(actual_path).expanduser().resolve()
        if allow_descendant:
            try:
                runtime_path.relative_to(verified_path)
            except ValueError as exc:
                raise ValueError(
                    f"runtime path {runtime_path} is not inside verified artifact {verified_path}"
                ) from exc
        elif runtime_path != verified_path:
            raise ValueError(
                f"runtime path {runtime_path} is not the verified artifact {verified_path}"
            )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    raise_if_any_rank_failed(f"artifact binding {logical_name}", error)


def verify_cluster_lock(
    lock: Mapping[str, Any], artifacts: Mapping[str, str | Path]
) -> dict[str, ArtifactFingerprint]:
    """Verify that local replicas match every logical artifact in the lock."""

    expected = lock.get("artifacts")
    if not isinstance(expected, dict):
        raise ValueError("cluster lock artifacts must be an object")
    missing = sorted(set(expected) - set(artifacts))
    unexpected = sorted(set(artifacts) - set(expected))
    if missing or unexpected:
        raise ValueError(
            f"artifact mapping differs from lock; missing={missing}, unexpected={unexpected}"
        )

    actual: dict[str, ArtifactFingerprint] = {}
    mismatches: dict[str, dict[str, Any]] = {}
    for name in sorted(expected):
        fingerprint = fingerprint_artifact(artifacts[name])
        actual[name] = fingerprint
        expected_item = expected[name]
        if not isinstance(expected_item, dict):
            raise ValueError(f"invalid fingerprint record for artifact {name!r}")
        actual_item = asdict(fingerprint)
        if actual_item != expected_item:
            mismatches[name] = {"expected": expected_item, "actual": actual_item}
    if mismatches:
        raise ValueError(
            "local artifact replicas do not match the cluster lock: "
            + json.dumps(mismatches, sort_keys=True)
        )
    return actual


def _python_source_sha256(package: str) -> str | None:
    spec = importlib.util.find_spec(package)
    if spec is None or spec.origin is None:
        return None
    package_root = Path(spec.origin).resolve().parent
    digest = hashlib.sha256()
    files = sorted(package_root.rglob("*.py"), key=lambda item: item.relative_to(package_root).as_posix())
    for candidate in files:
        relative = candidate.relative_to(package_root).as_posix()
        _hash_record(digest, relative, candidate.stat().st_size, sha256_file(candidate))
    return digest.hexdigest()


def _git_head(start: Path) -> str | None:
    for parent in (start, *start.parents):
        git_entry = parent / ".git"
        if not git_entry.exists():
            continue
        git_dir = git_entry
        if git_entry.is_file():
            content = git_entry.read_text(encoding="utf-8").strip()
            if not content.startswith("gitdir: "):
                return None
            git_dir = (parent / content.removeprefix("gitdir: ")).resolve()
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
        if not head.startswith("ref: "):
            return head if re.fullmatch(r"[0-9a-fA-F]{40}", head) else None
        reference = head.removeprefix("ref: ")
        loose = git_dir / reference
        if loose.is_file():
            return loose.read_text(encoding="utf-8").strip()
        packed = git_dir / "packed-refs"
        if packed.is_file():
            for line in packed.read_text(encoding="utf-8").splitlines():
                if line.startswith(("#", "^")):
                    continue
                fields = line.split(" ", 1)
                if len(fields) == 2 and fields[1] == reference:
                    return fields[0]
        return None
    return None


def _package_versions() -> list[str]:
    packages: list[str] = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name") or distribution.name
        packages.append(f"{str(name).lower().replace('_', '-')}=={distribution.version}")
    return sorted(packages)


def _digest_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def collect_runtime_signature(*, include_contract: bool = True) -> dict[str, Any]:
    """Collect stable software/GPU identity fields suitable for rank all-gather."""

    genet_spec = importlib.util.find_spec("genet")
    genet_path = Path(genet_spec.origin).resolve() if genet_spec and genet_spec.origin else None
    cosmos_spec = importlib.util.find_spec("cosmos_framework")
    cosmos_path = Path(cosmos_spec.origin).resolve() if cosmos_spec and cosmos_spec.origin else None
    packages = _package_versions()
    cuda_available = torch.cuda.is_available()
    distributed_environment = {
        key: os.environ[key]
        for key in sorted(_DISTRIBUTED_ENV_KEYS)
        if key in os.environ
    }
    nccl_version: Any = torch.cuda.nccl.version() if cuda_available else None
    if isinstance(nccl_version, tuple):
        nccl_version = list(nccl_version)
    signature: dict[str, Any] = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "machine": platform.machine(),
        "libc": platform.libc_ver(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if cuda_available else None,
        "nccl": nccl_version,
        "gpu_name": torch.cuda.get_device_name() if cuda_available else None,
        "gpu_capability": list(torch.cuda.get_device_capability()) if cuda_available else None,
        "packages_sha256": _digest_json(packages),
        "distributed_environment": distributed_environment,
        "genet_source_sha256": _python_source_sha256("genet"),
        "genet_git_revision": _git_head(genet_path) if genet_path else None,
        "cosmos_git_revision": _git_head(cosmos_path) if cosmos_path else None,
        "declared_image_digest": os.environ.get("GENET_IMAGE_DIGEST"),
        "declared_code_revision": os.environ.get("GENET_CODE_REVISION"),
        "declared_cosmos_revision": os.environ.get("GENET_COSMOS_REVISION"),
        "hf_snapshot_revision": os.environ.get("GENET_HF_SNAPSHOT_REVISION"),
    }
    embedded_revision_path = os.environ.get("GENET_BUILD_REVISION_FILE")
    if embedded_revision_path:
        signature["embedded_code_revision"] = (
            Path(embedded_revision_path).expanduser().read_text(encoding="utf-8").strip()
        )
    else:
        signature["embedded_code_revision"] = None
    lock_path = os.environ.get("GENET_CLUSTER_LOCK") if include_contract else None
    signature["cluster_lock_sha256"] = sha256_file(lock_path) if lock_path else None
    receipt_path = os.environ.get("GENET_CLUSTER_RECEIPT") if include_contract else None
    if receipt_path and lock_path:
        receipt = load_cluster_receipt(receipt_path, lock_path)
        signature["cluster_receipt_contract_sha256"] = _receipt_contract_sha256(receipt)
    else:
        signature["cluster_receipt_contract_sha256"] = None
    return signature


def _strict_environment_error(signature: Mapping[str, Any]) -> str | None:
    strict = os.environ.get("GENET_STRICT_ENV", "0").lower() in {"1", "true", "yes", "on"}
    if not strict:
        return None
    required = {
        "GENET_IMAGE_DIGEST": signature.get("declared_image_digest"),
        "GENET_CODE_REVISION": signature.get("declared_code_revision"),
        "GENET_BUILD_REVISION_FILE": signature.get("embedded_code_revision"),
        "GENET_COSMOS_REVISION": signature.get("declared_cosmos_revision"),
        "GENET_CLUSTER_LOCK": signature.get("cluster_lock_sha256"),
        "GENET_CLUSTER_RECEIPT": signature.get("cluster_receipt_contract_sha256"),
        "GENET_HF_SNAPSHOT_REVISION": signature.get("hf_snapshot_revision"),
    }
    missing = sorted(name for name, value in required.items() if not value)
    if missing:
        return f"strict environment preflight requires: {', '.join(missing)}"
    image_digest = str(signature["declared_image_digest"])
    if not _SHA256_RE.fullmatch(image_digest):
        return "GENET_IMAGE_DIGEST must be an immutable sha256 digest"
    if not _REVISION_RE.fullmatch(str(signature["declared_code_revision"])):
        return "GENET_CODE_REVISION must be a full 40-character Git commit"
    if signature["embedded_code_revision"] != signature["declared_code_revision"]:
        return (
            "embedded build revision does not match GENET_CODE_REVISION: "
            f"{signature['embedded_code_revision']} != {signature['declared_code_revision']}"
        )
    if not _REVISION_RE.fullmatch(str(signature["hf_snapshot_revision"])):
        return "GENET_HF_SNAPSHOT_REVISION must be a full 40-character repository commit"
    if signature["declared_cosmos_revision"] != PINNED_COSMOS_REVISION:
        return (
            "GENET_COSMOS_REVISION does not match the pinned Cosmos Framework revision: "
            f"{signature['declared_cosmos_revision']} != {PINNED_COSMOS_REVISION}"
        )
    actual_genet = signature.get("genet_git_revision")
    if actual_genet and signature["declared_code_revision"] != actual_genet:
        return (
            "GENET_CODE_REVISION does not match the checked-out GenET revision: "
            f"{signature['declared_code_revision']} != {actual_genet}"
        )
    actual_cosmos = signature.get("cosmos_git_revision")
    if actual_cosmos and signature["declared_cosmos_revision"] != actual_cosmos:
        return (
            "GENET_COSMOS_REVISION does not match the checked-out Cosmos revision: "
            f"{signature['declared_cosmos_revision']} != {actual_cosmos}"
        )
    return None


def assert_runtime_environment_consistent() -> dict[str, Any]:
    """Fail collectively when ranks run different code, packages, runtimes, or locks."""

    signature: dict[str, Any] | None = None
    error: str | None = None
    try:
        signature = collect_runtime_signature()
        error = _strict_environment_error(signature)
    except Exception as exc:  # keep every rank in the following collective
        error = f"{type(exc).__name__}: {exc}"
    raise_if_any_rank_failed("runtime environment preflight", error)
    assert signature is not None
    assert_same_across_ranks("runtime_environment_signature", signature)
    return signature


def python_runtime_summary() -> dict[str, Any]:
    """Small diagnostic payload used by the lock CLI and scheduler logs."""

    signature = collect_runtime_signature(include_contract=False)
    return {
        "python_executable": sys.executable,
        "python": signature["python"],
        "torch": signature["torch"],
        "cuda_runtime": signature["cuda_runtime"],
        "nccl": signature["nccl"],
        "genet_git_revision": signature["genet_git_revision"],
        "cosmos_git_revision": signature["cosmos_git_revision"],
        "genet_source_sha256": signature["genet_source_sha256"],
        "packages_sha256": signature["packages_sha256"],
    }
