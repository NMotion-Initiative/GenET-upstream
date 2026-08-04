"""One-process-per-node CPU/Gloo preflight before the NCCL training job."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
from datetime import timedelta
from typing import Any, Sequence

import torch
import torch.distributed as dist

from genet.training.distributed import assert_same_across_ranks, destroy_distributed, raise_if_any_rank_failed
from genet.training.environment import assert_runtime_environment_consistent


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="genet-cluster-preflight",
        description="Compare software and hardware across nodes over a CPU/Gloo group.",
    )
    parser.add_argument("--expected-gpus", type=int, required=True)
    return parser


def _driver_version() -> str:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
        timeout=15,
    )
    versions = sorted(set(line.strip() for line in result.stdout.splitlines() if line.strip()))
    if len(versions) != 1:
        raise ValueError(f"expected one NVIDIA driver version per node, got {versions}")
    return versions[0]


def _hardware_signature(expected_gpus: int) -> dict[str, Any]:
    count = torch.cuda.device_count()
    if count != expected_gpus:
        raise ValueError(f"expected {expected_gpus} visible GPUs, found {count}")
    devices = []
    for index in range(count):
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "name": properties.name,
                "capability": [properties.major, properties.minor],
                "total_memory": properties.total_memory,
            }
        )
    return {
        "gpu_count": count,
        "devices": devices,
        "driver": _driver_version() if expected_gpus else None,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.expected_gpus < 0:
        raise ValueError("--expected-gpus must be non-negative")
    if int(os.environ.get("LOCAL_WORLD_SIZE", "1")) != 1:
        raise ValueError("cluster preflight must run with one process per node")

    try:
        timeout_seconds = int(os.environ.get("GENET_PREFLIGHT_TIMEOUT_SECONDS", "120"))
        if timeout_seconds < 10:
            raise ValueError("GENET_PREFLIGHT_TIMEOUT_SECONDS must be at least 10")
        dist.init_process_group(
            backend="gloo",
            init_method="env://",
            timeout=timedelta(seconds=timeout_seconds),
        )
        environment = assert_runtime_environment_consistent()
        hardware: dict[str, Any] | None = None
        error: str | None = None
        try:
            hardware = _hardware_signature(args.expected_gpus)
        except Exception as exc:  # keep every node in the following collective
            error = f"{type(exc).__name__}: {exc}"
        raise_if_any_rank_failed("node hardware preflight", error)
        assert hardware is not None
        assert_same_across_ranks("node_hardware_signature", hardware)

        reports: list[dict[str, Any] | None] = [None] * dist.get_world_size()
        node_id = (
            os.environ.get("GENET_NODE_ID")
            or os.environ.get("SLURMD_NODENAME")
            or socket.gethostname()
        )
        dist.all_gather_object(
            reports,
            {
                "rank": dist.get_rank(),
                "hostname": socket.gethostname(),
                "node_id": node_id,
                "hardware": hardware,
            },
        )
        node_ids = [item["node_id"] for item in reports if item is not None]
        if len(node_ids) != dist.get_world_size() or len(set(node_ids)) != len(node_ids):
            raise RuntimeError(
                "cluster preflight requires one unique physical node per rank; "
                f"got node identities {node_ids}"
            )
        if dist.get_rank() == 0:
            print(
                json.dumps(
                    {
                        "event": "cluster_preflight_passed",
                        "environment": environment,
                        "nodes": reports,
                        "world_size": dist.get_world_size(),
                    },
                    indent=2,
                    sort_keys=True,
                ),
                flush=True,
            )
    finally:
        destroy_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
