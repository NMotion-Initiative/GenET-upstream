"""One-process-per-GPU NCCL correctness and bandwidth preflight."""

from __future__ import annotations

import argparse
import json
import os
import socket
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist

_MIB = 1024 * 1024


@dataclass(frozen=True)
class PreflightConfig:
    expected_nnodes: int
    expected_local_world_size: int
    buffer_mib: int
    warmup_iterations: int
    iterations: int
    timeout_seconds: int

    @property
    def expected_world_size(self) -> int:
        return self.expected_nnodes * self.expected_local_world_size

    @property
    def buffer_bytes(self) -> int:
        return self.buffer_mib * _MIB


@dataclass(frozen=True)
class LaunchEnvironment:
    rank: int
    local_rank: int
    world_size: int
    local_world_size: int


def _integer(value: str, *, name: str, minimum: int) -> int:
    if not value.isascii() or not value.isdecimal():
        raise argparse.ArgumentTypeError(f"{name} must be a base-10 integer, got {value!r}")
    parsed = int(value)
    if parsed < minimum:
        comparator = "positive" if minimum == 1 else f"at least {minimum}"
        raise argparse.ArgumentTypeError(f"{name} must be {comparator}, got {parsed}")
    return parsed


def _positive_integer(name: str):
    return lambda value: _integer(value, name=name, minimum=1)


def _non_negative_integer(name: str):
    return lambda value: _integer(value, name=name, minimum=0)


def _timeout(value: str) -> int:
    return _integer(value, name="--timeout-seconds", minimum=10)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="genet-nccl-preflight",
        description="Validate one-process-per-GPU multi-node NCCL all-reduce correctness and bandwidth.",
    )
    parser.add_argument(
        "--expected-nnodes",
        type=_positive_integer("--expected-nnodes"),
        required=True,
    )
    parser.add_argument(
        "--expected-local-world-size",
        type=_positive_integer("--expected-local-world-size"),
        required=True,
    )
    parser.add_argument(
        "--buffer-mib",
        type=_positive_integer("--buffer-mib"),
        default=64,
        help="Size of the float32 all-reduce buffer on each rank (default: 64 MiB).",
    )
    parser.add_argument(
        "--warmup-iterations",
        type=_non_negative_integer("--warmup-iterations"),
        default=3,
    )
    parser.add_argument(
        "--iterations",
        type=_positive_integer("--iterations"),
        default=10,
    )
    parser.add_argument("--timeout-seconds", type=_timeout, default=180)
    return parser


def _parse_environment_integer(name: str) -> int:
    raw = os.environ.get(name)
    if raw is None:
        raise ValueError(f"torchrun did not set required environment variable {name}")
    if not raw.isascii() or not raw.isdecimal():
        raise ValueError(f"{name} must be a non-negative base-10 integer, got {raw!r}")
    return int(raw)


def _validate_launch_environment(config: PreflightConfig) -> LaunchEnvironment:
    environment = LaunchEnvironment(
        rank=_parse_environment_integer("RANK"),
        local_rank=_parse_environment_integer("LOCAL_RANK"),
        world_size=_parse_environment_integer("WORLD_SIZE"),
        local_world_size=_parse_environment_integer("LOCAL_WORLD_SIZE"),
    )
    if environment.world_size != config.expected_world_size:
        raise ValueError(
            "WORLD_SIZE does not match the requested topology: "
            f"expected {config.expected_world_size}, got {environment.world_size}"
        )
    if environment.local_world_size != config.expected_local_world_size:
        raise ValueError(
            "LOCAL_WORLD_SIZE does not match --expected-local-world-size: "
            f"expected {config.expected_local_world_size}, got {environment.local_world_size}"
        )
    if not 0 <= environment.rank < environment.world_size:
        raise ValueError(f"RANK must be in [0, WORLD_SIZE), got {environment.rank}")
    if not 0 <= environment.local_rank < environment.local_world_size:
        raise ValueError(
            "LOCAL_RANK must be in [0, LOCAL_WORLD_SIZE), "
            f"got {environment.local_rank}"
        )
    expected_local_rank = environment.rank % environment.local_world_size
    if environment.local_rank != expected_local_rank:
        raise ValueError(
            "torchrun rank layout is not contiguous by node: "
            f"rank {environment.rank} has LOCAL_RANK={environment.local_rank}, "
            f"expected {expected_local_rank}"
        )
    return environment


def _rank_metadata(environment: LaunchEnvironment) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(environment.local_rank)
    node_id = os.environ.get("GENET_NODE_ID", "").strip() or socket.gethostname()
    return {
        "rank": environment.rank,
        "local_rank": environment.local_rank,
        "node_id": node_id,
        "hostname": socket.gethostname(),
        "device": {
            "name": properties.name,
            "capability": [properties.major, properties.minor],
            "total_memory_bytes": properties.total_memory,
        },
    }


def _gather_and_validate_topology(
    metadata: dict[str, Any],
    config: PreflightConfig,
) -> list[dict[str, Any]]:
    gathered: list[dict[str, Any] | None] = [None] * config.expected_world_size
    dist.all_gather_object(gathered, metadata)
    if any(item is None for item in gathered):
        raise RuntimeError("NCCL topology gather returned an incomplete rank list")
    ranks = [item for item in gathered if item is not None]
    if sorted(item["rank"] for item in ranks) != list(range(config.expected_world_size)):
        raise RuntimeError("NCCL topology gather returned duplicate or missing global ranks")

    by_node: dict[str, list[dict[str, Any]]] = {}
    for item in ranks:
        by_node.setdefault(item["node_id"], []).append(item)
    if len(by_node) != config.expected_nnodes:
        raise RuntimeError(
            f"expected {config.expected_nnodes} unique GENET_NODE_ID values, "
            f"got {sorted(by_node)}"
        )
    expected_local_ranks = list(range(config.expected_local_world_size))
    for node_id, node_ranks in by_node.items():
        actual_local_ranks = sorted(item["local_rank"] for item in node_ranks)
        if actual_local_ranks != expected_local_ranks:
            raise RuntimeError(
                f"node {node_id!r} has local ranks {actual_local_ranks}; "
                f"expected {expected_local_ranks}"
            )
    return ranks


def _all_reduce_once(buffer: torch.Tensor, rank_value: float) -> float:
    buffer.fill_(rank_value)
    dist.barrier()
    torch.cuda.synchronize(buffer.device)
    started = time.perf_counter()
    dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize(buffer.device)
    return (time.perf_counter() - started) * 1000.0


def _verify_all_reduce(buffer: torch.Tensor, environment: LaunchEnvironment) -> float:
    rank_value = float(environment.rank + 1)
    buffer.fill_(rank_value)
    dist.barrier()
    dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
    expected = float(environment.world_size * (environment.world_size + 1) // 2)
    local_max_error = torch.max(torch.abs(buffer - expected))
    dist.all_reduce(local_max_error, op=dist.ReduceOp.MAX)
    max_error = float(local_max_error.item())
    if max_error != 0.0:
        raise RuntimeError(
            "NCCL all-reduce produced incorrect values: "
            f"expected every element to equal {expected}, max absolute error was {max_error}"
        )
    return expected


def _benchmark(
    buffer: torch.Tensor,
    environment: LaunchEnvironment,
    config: PreflightConfig,
) -> list[float]:
    rank_value = float(environment.rank + 1)
    for _ in range(config.warmup_iterations):
        _all_reduce_once(buffer, rank_value)
    return [_all_reduce_once(buffer, rank_value) for _ in range(config.iterations)]


def _gather_timings(local_timings_ms: list[float], environment: LaunchEnvironment) -> list[list[float]] | None:
    device = torch.device("cuda", environment.local_rank)
    local = torch.tensor(local_timings_ms, dtype=torch.float64, device=device)
    gathered = [torch.empty_like(local) for _ in range(environment.world_size)]
    dist.all_gather(gathered, local)
    if environment.rank != 0:
        return None
    return [tensor.cpu().tolist() for tensor in gathered]


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _build_report(
    config: PreflightConfig,
    ranks: list[dict[str, Any]],
    timings_by_rank_ms: list[list[float]],
    expected_sum: float,
) -> dict[str, Any]:
    if len(timings_by_rank_ms) != config.expected_world_size:
        raise RuntimeError("timing gather returned the wrong number of ranks")
    if any(len(item) != config.iterations for item in timings_by_rank_ms):
        raise RuntimeError("timing gather returned the wrong number of iterations")
    iteration_max_ms = [
        max(rank_timings[iteration] for rank_timings in timings_by_rank_ms)
        for iteration in range(config.iterations)
    ]
    if any(value <= 0.0 for value in iteration_max_ms):
        raise RuntimeError(f"NCCL timing must be positive, got {iteration_max_ms}")
    median_ms = statistics.median(iteration_max_ms)
    algorithmic_bandwidth_gb_per_second = config.buffer_bytes / (median_ms / 1000.0) / 1e9
    bus_bandwidth_gb_per_second = (
        algorithmic_bandwidth_gb_per_second
        * 2.0
        * (config.expected_world_size - 1)
        / config.expected_world_size
    )
    nodes: dict[str, list[dict[str, Any]]] = {}
    for item in ranks:
        nodes.setdefault(item["node_id"], []).append(item)
    node_report = [
        {
            "node_id": node_id,
            "hostname": sorted({item["hostname"] for item in node_ranks}),
            "ranks": sorted(item["rank"] for item in node_ranks),
            "devices": [
                item["device"]
                for item in sorted(node_ranks, key=lambda rank: rank["local_rank"])
            ],
        }
        for node_id, node_ranks in sorted(nodes.items())
    ]
    return {
        "event": "nccl_preflight_passed",
        "format_version": 1,
        "backend": "nccl",
        "world_size": config.expected_world_size,
        "nnodes": config.expected_nnodes,
        "local_world_size": config.expected_local_world_size,
        "buffer_bytes": config.buffer_bytes,
        "buffer_mib": config.buffer_mib,
        "dtype": "float32",
        "warmup_iterations": config.warmup_iterations,
        "iterations": config.iterations,
        "timeout_seconds": config.timeout_seconds,
        "correctness": {"expected_sum": expected_sum, "max_abs_error": 0.0},
        "timing_ms": {
            "per_iteration_rank_max": iteration_max_ms,
            "min": min(iteration_max_ms),
            "median": median_ms,
            "p95": _percentile(iteration_max_ms, 0.95),
            "max": max(iteration_max_ms),
        },
        "algorithmic_bandwidth_GBps": algorithmic_bandwidth_gb_per_second,
        "bus_bandwidth_GBps": bus_bandwidth_gb_per_second,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "nodes": node_report,
    }


def _cleanup_distributed() -> None:
    errors: list[Exception] = []
    if dist.is_initialized():
        try:
            dist.destroy_process_group()
        except Exception as exc:  # pragma: no cover - requires a broken NCCL runtime
            errors.append(exc)
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception as exc:  # pragma: no cover - requires a broken CUDA runtime
            errors.append(exc)
    if errors:
        raise RuntimeError(f"failed to clean up NCCL preflight: {errors}")


def _run(config: PreflightConfig) -> dict[str, Any] | None:
    environment = _validate_launch_environment(config)
    if dist.is_initialized():
        raise RuntimeError("a process group is already initialized before the NCCL preflight")
    if not dist.is_nccl_available():
        raise RuntimeError("this PyTorch build does not provide the NCCL backend")
    if not torch.cuda.is_available():
        raise RuntimeError("NCCL preflight requires CUDA")
    visible_devices = torch.cuda.device_count()
    if visible_devices != environment.local_world_size:
        raise RuntimeError(
            f"expected {environment.local_world_size} visible CUDA devices, found {visible_devices}"
        )
    torch.cuda.set_device(environment.local_rank)
    device = torch.device("cuda", environment.local_rank)

    primary_error: BaseException | None = None
    try:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=timedelta(seconds=config.timeout_seconds),
        )
        metadata = _rank_metadata(environment)
        ranks = _gather_and_validate_topology(metadata, config)
        element_size = torch.empty((), dtype=torch.float32).element_size()
        if config.buffer_bytes % element_size != 0:
            raise ValueError("the requested buffer size is not divisible by the float32 element size")
        buffer = torch.empty(
            config.buffer_bytes // element_size,
            dtype=torch.float32,
            device=device,
        )
        expected_sum = _verify_all_reduce(buffer, environment)
        local_timings_ms = _benchmark(buffer, environment, config)
        timings_by_rank_ms = _gather_timings(local_timings_ms, environment)
        dist.barrier()
        if environment.rank != 0:
            return None
        assert timings_by_rank_ms is not None
        return _build_report(config, ranks, timings_by_rank_ms, expected_sum)
    except BaseException as exc:
        primary_error = exc
        raise
    finally:
        try:
            _cleanup_distributed()
        except Exception:
            if primary_error is None:
                raise


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = PreflightConfig(
        expected_nnodes=args.expected_nnodes,
        expected_local_world_size=args.expected_local_world_size,
        buffer_mib=args.buffer_mib,
        warmup_iterations=args.warmup_iterations,
        iterations=args.iterations,
        timeout_seconds=args.timeout_seconds,
    )
    report = _run(config)
    if report is not None:
        print(json.dumps(report, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("NCCL preflight interrupted", file=sys.stderr, flush=True)
        raise SystemExit(130) from None
