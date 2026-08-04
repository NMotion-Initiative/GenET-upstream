"""Create and verify content locks for node-local training replicas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from genet.training.environment import (
    create_cluster_lock,
    create_cluster_receipt,
    load_cluster_lock,
    python_runtime_summary,
    verify_cluster_lock,
    write_cluster_lock,
    write_cluster_receipt,
)


def _artifact(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name or not raw_path:
        raise argparse.ArgumentTypeError("artifact must use NAME=PATH")
    return name, Path(raw_path)


def _artifacts(values: list[tuple[str, Path]]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for name, path in values:
        if name in result:
            raise ValueError(f"duplicate artifact name: {name}")
        result[name] = path
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="genet-cluster-lock",
        description="Create or verify a content lock for node-local training artifacts.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create", help="Hash a canonical staging copy")
    create.add_argument("--output", type=Path, required=True)
    create.add_argument("--artifact", action="append", type=_artifact, required=True)

    verify = subparsers.add_parser("verify", help="Verify one node-local replica")
    verify.add_argument("--lock", type=Path, required=True)
    verify.add_argument("--artifact", action="append", type=_artifact, required=True)
    verify.add_argument(
        "--receipt",
        type=Path,
        help="Write a node-local path-binding receipt for strict training",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    artifacts = _artifacts(args.artifact)
    if args.command == "create":
        lock = create_cluster_lock(artifacts)
        destination = write_cluster_lock(args.output, lock)
        payload = {
            "event": "cluster_lock_created",
            "lock": str(destination),
            "lock_artifacts": lock["artifacts"],
            "runtime": python_runtime_summary(),
        }
    else:
        lock = load_cluster_lock(args.lock)
        actual = verify_cluster_lock(lock, artifacts)
        receipt_path = None
        if args.receipt is not None:
            receipt = create_cluster_receipt(args.lock, lock, artifacts, actual)
            receipt_path = write_cluster_receipt(args.receipt, receipt)
        payload = {
            "event": "cluster_lock_verified",
            "lock": str(args.lock.expanduser().resolve()),
            "receipt": str(receipt_path) if receipt_path else None,
            "lock_artifacts": {
                name: {
                    "kind": item.kind,
                    "sha256": item.sha256,
                    "file_count": item.file_count,
                    "total_bytes": item.total_bytes,
                }
                for name, item in actual.items()
            },
            "runtime": python_runtime_summary(),
        }
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
