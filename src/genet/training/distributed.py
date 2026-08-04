"""Small, explicit distributed runtime used by the standalone trainer."""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def node_rank(self) -> int:
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "1"))
        return self.rank // local_world_size


def initialize_distributed() -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")
    return DistributedContext(rank=rank, local_rank=local_rank, world_size=world_size, device=device)


def seed_everything(seed: int, rank: int = 0) -> None:
    seed = int(seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def barrier() -> None:
    if dist.is_initialized():
        dist.barrier()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_same_across_ranks(name: str, value: Any) -> None:
    """Fail before training if local data/config fingerprints differ."""

    if not dist.is_initialized():
        return
    serialized = json.dumps(value, sort_keys=True, default=str)
    gathered: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, serialized)
    if len(set(gathered)) != 1:
        by_rank = {rank: item for rank, item in enumerate(gathered)}
        raise RuntimeError(f"{name} differs across ranks: {by_rank}")


def raise_if_any_rank_failed(name: str, error: str | None) -> None:
    """Turn asymmetric node-local preflight errors into a global failure."""

    if not dist.is_initialized():
        if error is not None:
            raise RuntimeError(f"{name} failed: {error}")
        return
    gathered: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, error)
    failures = {rank: item for rank, item in enumerate(gathered) if item is not None}
    if failures:
        raise RuntimeError(f"{name} failed on rank(s): {failures}")


def destroy_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
