"""Transactional checkpoints and no-shared-filesystem DCP consolidation."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from genet.training.distributed import sha256_file


@dataclass
class TrainerState:
    step: int
    epoch: int
    batches_in_epoch: int


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _safe_manifest_path(root: Path, relative: str) -> Path:
    """Resolve an archive entry beneath ``root`` without traversal/symlinks."""

    if not isinstance(relative, str) or not relative:
        raise ValueError("checkpoint manifest paths must be non-empty strings")
    raw = Path(relative)
    if raw.is_absolute() or any(part in {"", ".", ".."} for part in raw.parts):
        raise ValueError(f"unsafe checkpoint manifest path: {relative!r}")
    resolved_root = root.resolve()
    candidate = (resolved_root / raw).resolve()
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"checkpoint manifest path escapes root: {relative!r}") from exc
    return candidate


class CheckpointManager:
    """Single-file checkpoint manager for the standalone/DDP backend."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        trainer_state: TrainerState,
        config: dict[str, Any],
        rng_states: dict[int, dict[str, Any]] | None = None,
    ) -> Path:
        final = self.root / f"step_{trainer_state.step:09d}"
        staging = self.root / f"step_{trainer_state.step:09d}.incomplete"
        if final.exists() or staging.exists():
            raise FileExistsError(f"Checkpoint destination already exists: {final} / {staging}")
        staging.mkdir(parents=False)
        model_state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        torch.save(model_state, staging / "model.pt")
        torch.save(optimizer.state_dict(), staging / "optimizer.pt")
        torch.save(scheduler.state_dict(), staging / "scheduler.pt")
        if rng_states is None:
            rng_states = {0: capture_rng_state()}
        torch.save({"by_rank": rng_states}, staging / "rng.pt")
        _atomic_json(staging / "trainer.json", asdict(trainer_state))
        _atomic_json(staging / "config.json", config)
        manifest = {
            path.name: {"sha256": sha256_file(path), "size": path.stat().st_size}
            for path in sorted(staging.iterdir())
            if path.is_file()
        }
        _atomic_json(staging / "MANIFEST.json", manifest)
        (staging / "COMMITTED").write_text(f"committed_at={time.time()}\n", encoding="utf-8")
        os.replace(staging, final)
        _atomic_json(self.root / "latest.json", {"path": final.name, "step": trainer_state.step})
        return final

    def load(
        self,
        checkpoint: str | Path,
        *,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        weights_only: bool = False,
        rank: int = 0,
    ) -> TrainerState:
        path = Path(checkpoint)
        verify_committed_checkpoint(path)
        target = model.module if hasattr(model, "module") else model
        target.load_state_dict(
            torch.load(path / "model.pt", map_location="cpu", weights_only=True),
            strict=not weights_only,
        )
        if not weights_only:
            if optimizer is None or scheduler is None:
                raise ValueError("optimizer and scheduler are required for an exact resume")
            optimizer.load_state_dict(torch.load(path / "optimizer.pt", map_location="cpu", weights_only=True))
            scheduler.load_state_dict(torch.load(path / "scheduler.pt", map_location="cpu", weights_only=True))
            rng = torch.load(path / "rng.pt", map_location="cpu", weights_only=True)
            if "by_rank" in rng:
                by_rank = rng["by_rank"]
                local = by_rank.get(rank, by_rank.get(str(rank)))
                if local is None:
                    raise KeyError(f"checkpoint has no RNG state for global rank {rank}")
                restore_rng_state(local)
            else:  # Compatibility with the initial standalone checkpoint format.
                torch.set_rng_state(rng["torch"])
                if torch.cuda.is_available() and rng["cuda"] is not None:
                    torch.cuda.set_rng_state_all(rng["cuda"])
        with (path / "trainer.json").open("r", encoding="utf-8") as handle:
            return TrainerState(**json.load(handle))


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, torch, and the current rank's CUDA RNG."""

    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": torch.from_numpy(numpy_state[1].copy()),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["bit_generator"],
            numpy_state["state"].cpu().numpy(),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"])


def verify_committed_checkpoint(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not (path / "COMMITTED").is_file():
        raise ValueError(f"Checkpoint is not committed: {path}")
    with (path / "MANIFEST.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    for relative, metadata in manifest.items():
        candidate = _safe_manifest_path(path, relative)
        if not candidate.is_file():
            raise FileNotFoundError(f"Missing checkpoint file: {candidate}")
        if candidate.stat().st_size != metadata["size"] or sha256_file(candidate) != metadata["sha256"]:
            raise ValueError(f"Checkpoint checksum mismatch: {candidate}")
    return manifest


def create_node_manifest(checkpoint_dir: str | Path, node_rank: int, output: str | Path) -> Path:
    checkpoint_dir = Path(checkpoint_dir).resolve()
    files = []
    for path in sorted(checkpoint_dir.rglob("*")):
        if path.is_file():
            files.append(
                {
                    "path": str(path.relative_to(checkpoint_dir)),
                    "size": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    payload = {"node_rank": int(node_rank), "checkpoint": checkpoint_dir.name, "files": files}
    output = Path(output)
    _atomic_json(output, payload)
    return output


def consolidate_node_archives(
    archive_dir: str | Path,
    output_dir: str | Path,
    *,
    expected_nodes: int,
) -> Path:
    """Merge ``node_XX`` uploads, verify checksums, then create COMMITTED.

    File names emitted by PyTorch DCP are global-rank unique. Duplicate paths
    are accepted only when their bytes are identical (for metadata replicated
    by a transport); conflicting duplicates abort the commit.
    """

    archive_dir = Path(archive_dir)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    # ``.incomplete`` is the transport/upload directory used by node
    # publishers. Keep consolidation staging distinct so archive_dir may be
    # ``<output>.incomplete`` without colliding with this transaction.
    staging = output_dir.with_name(output_dir.name + ".consolidating")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    combined: dict[str, dict[str, Any]] = {}
    try:
        for node_rank in range(expected_nodes):
            node_dir = archive_dir / f"node_{node_rank:02d}"
            manifest_path = node_dir / "NODE_MANIFEST.json"
            done_path = node_dir / "NODE_DONE"
            if not manifest_path.is_file() or not done_path.is_file():
                raise FileNotFoundError(f"Node {node_rank} upload is incomplete under {node_dir}")
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            if int(manifest["node_rank"]) != node_rank:
                raise ValueError(f"Wrong node_rank in {manifest_path}")
            for item in manifest["files"]:
                relative = item["path"]
                if relative in {"COMMITTED", "MANIFEST.json"}:
                    raise ValueError(f"Reserved checkpoint transport path: {relative}")
                source = _safe_manifest_path(node_dir / "files", relative)
                if not source.is_file() or source.stat().st_size != item["size"]:
                    raise ValueError(f"Missing/short uploaded shard: {source}")
                if sha256_file(source) != item["sha256"]:
                    raise ValueError(f"Uploaded shard checksum mismatch: {source}")
                if relative in combined:
                    if combined[relative]["sha256"] != item["sha256"]:
                        raise ValueError(f"Conflicting duplicate DCP path: {relative}")
                    continue
                destination = _safe_manifest_path(staging, relative)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                combined[relative] = item
        if not any(path.name == ".metadata" for path in staging.rglob(".metadata")):
            raise ValueError("Consolidated DCP has no .metadata; coordinator upload is missing")
        _atomic_json(staging / "MANIFEST.json", combined)
        digest = hashlib.sha256(json.dumps(combined, sort_keys=True).encode()).hexdigest()
        (staging / "COMMITTED").write_text(f"manifest_sha256={digest}\n", encoding="utf-8")
        os.replace(staging, output_dir)
        return output_dir
    except Exception:
        # Keep staging for forensic inspection; never label it committed.
        raise
