"""Launch-scoped attestations for expensive Cosmos data validation."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from genet.training.distributed import sha256_file
from genet.training.environment import load_cluster_receipt

DATA_VALIDATION_RECEIPT_VERSION = "genet.cosmos-data-validation/v1"


def _digest_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _strict_environment() -> bool:
    return os.environ.get("GENET_STRICT_ENV", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def build_validation_attestation(
    *,
    manifest: str | Path,
    manifest_sha256: str,
    distributed_config_fingerprint: str,
    validation_contract: Mapping[str, Any],
    runtime_signature: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Build the exact receipt expected by dry-run and train in one launch.

    Non-strict/local runs deliberately bypass this optimization. Strict runs
    bind the marker to the already verified cluster receipt and installed
    source identity, so a changed release or config falls back to a full scan.
    """

    if not _strict_environment():
        return None
    launch_id = os.environ.get("GENET_LAUNCH_ID")
    lock_path = os.environ.get("GENET_CLUSTER_LOCK")
    receipt_path = os.environ.get("GENET_CLUSTER_RECEIPT")
    data_name = os.environ.get("GENET_DATA_ARTIFACT", "processed_data")
    if not launch_id or not lock_path or not receipt_path:
        return None

    receipt = load_cluster_receipt(receipt_path, lock_path)
    data_record = receipt["artifacts"].get(data_name)
    if not isinstance(data_record, dict) or not isinstance(data_record.get("path"), str):
        return None

    verified_root = Path(data_record["path"]).expanduser().resolve()
    manifest_path = Path(manifest).expanduser().resolve()
    try:
        manifest_relative = manifest_path.relative_to(verified_root).as_posix()
    except ValueError:
        return None

    processed_fingerprint = {
        key: value for key, value in data_record.items() if key != "path"
    }
    identity = {
        "launch_id": launch_id,
        "lock_sha256": sha256_file(lock_path),
        "cluster_receipt_contract_sha256": runtime_signature.get(
            "cluster_receipt_contract_sha256"
        ),
        "processed_data": processed_fingerprint,
        "processed_data_path": str(verified_root),
        "manifest_relative": manifest_relative,
        "manifest_sha256": manifest_sha256,
        "distributed_config_fingerprint": distributed_config_fingerprint,
        "validation_contract": dict(validation_contract),
        "genet_source_sha256": runtime_signature.get("genet_source_sha256"),
        "declared_code_revision": runtime_signature.get("declared_code_revision"),
    }
    required = (
        "cluster_receipt_contract_sha256",
        "genet_source_sha256",
        "declared_code_revision",
    )
    if any(not identity[name] for name in required):
        return None
    return {
        "format_version": DATA_VALIDATION_RECEIPT_VERSION,
        "cache_key": _digest_json(identity),
        "identity": identity,
    }


def validation_attestation_path(expected: Mapping[str, Any]) -> Path | None:
    launch_id = expected.get("identity", {}).get("launch_id")
    run_root = os.environ.get("GENET_NODE_RUN_ROOT")
    if not isinstance(launch_id, str) or not launch_id or not run_root:
        return None
    root = Path(
        os.environ.get(
            "GENET_DATA_VALIDATION_RECEIPT_DIR",
            str(Path(run_root) / "release" / "data-validation"),
        )
    ).expanduser()
    return root.resolve() / f"{launch_id}.json"


def validation_attestation_matches(
    path: str | Path,
    expected: Mapping[str, Any],
) -> bool:
    try:
        with Path(path).expanduser().open("r", encoding="utf-8") as handle:
            actual = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(actual, dict)
        and actual.get("format_version") == DATA_VALIDATION_RECEIPT_VERSION
        and actual.get("cache_key") == expected.get("cache_key")
        and actual.get("identity") == expected.get("identity")
    )


def write_validation_attestation(
    path: str | Path,
    expected: Mapping[str, Any],
) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(dict(expected), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return destination
