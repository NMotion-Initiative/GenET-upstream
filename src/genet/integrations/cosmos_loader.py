"""Resume-aware finite-map loading for the pinned Cosmos packing stack.

The upstream ``PackingDataLoader`` caches an iterator over its wrapped loader.
That is appropriate for the framework's infinite datasets, but a normal finite
``Dataset`` would otherwise be exhausted after one local epoch. These thin
wrappers preserve upstream packing while adding deterministic epoch shuffling,
infinite cycling, and exact consumed-micro-batch restoration.
"""

from __future__ import annotations

from collections.abc import Iterator, Sized
from typing import Any

import torch
from torch.utils.data import Sampler

_COSMOS_LOADER_IMPORT_ERROR: ImportError | None = None
try:  # Optional on lightweight preprocessing/test machines.
    from cosmos_framework.data.generator.joint_dataloader import (
        PackingDataLoader as _PackingDataLoader,
        RankPartitionedDataLoader as _RankPartitionedDataLoader,
    )

    COSMOS_LOADER_AVAILABLE = True
except ImportError as exc:  # pragma: no cover - depends on NVIDIA environment
    _COSMOS_LOADER_IMPORT_ERROR = exc
    _PackingDataLoader = object  # type: ignore[assignment,misc]
    _RankPartitionedDataLoader = object  # type: ignore[assignment,misc]
    COSMOS_LOADER_AVAILABLE = False


class DeterministicEpochSampler(Sampler[int | tuple[int, int]]):
    """Shuffle one finite local shard and restore by consumed-item count."""

    def __init__(
        self,
        data_source: Sized,
        *,
        seed: int,
        emit_epoch: bool = False,
    ) -> None:
        self.data_source = data_source
        self.seed = int(seed)
        self.emit_epoch = bool(emit_epoch)
        self.epoch = 0
        self.offset = 0
        if len(self.data_source) < 1:
            raise ValueError("a rank-local Cosmos dataset must contain at least one sample")

    def set_start_iteration(self, consumed_items: int) -> None:
        if consumed_items < 0:
            raise ValueError("consumed_items must be non-negative")
        size = len(self.data_source)
        if size < 1:
            raise ValueError("cannot position an empty sampler")
        self.epoch, self.offset = divmod(int(consumed_items), size)

    def __iter__(self) -> Iterator[int | tuple[int, int]]:
        size = len(self.data_source)
        if size < 1:
            return
        current_epoch = self.epoch
        generator = torch.Generator()
        generator.manual_seed(self.seed + current_epoch)
        order = torch.randperm(size, generator=generator).tolist()
        start = self.offset
        self.epoch += 1
        self.offset = 0
        for index in order[start:]:
            yield (current_epoch, index) if self.emit_epoch else index

    def __len__(self) -> int:
        return max(len(self.data_source) - self.offset, 0)


class CosmosInfiniteRankPartitionedDataLoader(_RankPartitionedDataLoader):  # type: ignore[misc]
    """Rank-partition upstream, then cycle a deterministic local sampler."""

    def __init__(
        self,
        datasets: dict[str, dict[str, Any]],
        *,
        seed: int,
        **dataloader_kwargs: Any,
    ) -> None:
        if not COSMOS_LOADER_AVAILABLE:  # pragma: no cover - NVIDIA-only path
            raise RuntimeError(
                "Cosmos loader integration is unavailable; install the pinned "
                "cosmos-framework environment"
            ) from _COSMOS_LOADER_IMPORT_ERROR
        if int(dataloader_kwargs.get("batch_size", 1)) != 1:
            raise ValueError("GenET Cosmos loading currently requires batch_size=1")
        if dataloader_kwargs.get("shuffle", False):
            raise ValueError("shuffle is owned by DeterministicEpochSampler")
        super().__init__(datasets, **dataloader_kwargs)
        rank_seed = int(seed) + int(self.dataset.shard_rank)
        self._genet_sampler = DeterministicEpochSampler(
            self.dataset,
            seed=rank_seed,
            emit_epoch=True,
        )
        self._worker_seed = int(seed) + 1_000_000 + int(self.dataset.shard_rank)
        self._worker_generator = torch.Generator()
        self.dataloader.generator = self._worker_generator
        # DataLoader reads indices from ``batch_sampler``. Replacing this leaf
        # preserves upstream-created workers, collation, pinning, and overrides.
        self.dataloader.batch_sampler.sampler = self._genet_sampler

    def set_start_iteration(self, consumed_micro_batches: int) -> None:
        self._genet_sampler.set_start_iteration(consumed_micro_batches)

    def __iter__(self):
        while True:
            # Keep DataLoader worker/base-seed draws off the model's global RNG.
            # Re-seeding from the sampler epoch makes iterator reconstruction on
            # exact resume independent of prewarm and worker lifecycle details.
            self._worker_generator.manual_seed(
                self._worker_seed + self._genet_sampler.epoch
            )
            yielded = False
            for batch in self.dataloader:
                yielded = True
                yield batch
            if not yielded:
                raise RuntimeError("rank-local Cosmos dataloader produced no batches")


class CosmosResumeAwarePackingDataLoader(_PackingDataLoader):  # type: ignore[misc]
    """Forward upstream resume iteration to the finite-map rank loader."""

    def set_start_iteration(self, iteration: int) -> None:
        if not COSMOS_LOADER_AVAILABLE:  # pragma: no cover - NVIDIA-only path
            raise RuntimeError(
                "Cosmos loader integration is unavailable; install the pinned "
                "cosmos-framework environment"
            ) from _COSMOS_LOADER_IMPORT_ERROR
        super().set_start_iteration(iteration)
        if len(self.dataloader_list) != 1:
            raise RuntimeError("GenET expects exactly one wrapped Cosmos dataloader")
        inner = self.dataloader_list[0]
        setter = getattr(inner, "set_start_iteration", None)
        if not callable(setter):
            raise TypeError("wrapped Cosmos dataloader is not resume-aware")
        setter(int(iteration))
        # Upstream pre-warms before checkpoint loading. Discard that speculative
        # sample and recreate the iterator at the restored position.
        self.buffers[0].clear()
        old_iterator = self.dataloaders[0]
        close = getattr(old_iterator, "close", None)
        if callable(close):
            close()
        self.dataloaders[0] = iter(inner)


__all__ = [
    "COSMOS_LOADER_AVAILABLE",
    "CosmosInfiniteRankPartitionedDataLoader",
    "CosmosResumeAwarePackingDataLoader",
    "DeterministicEpochSampler",
]
