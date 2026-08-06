"""Sparse GenET-conditioned eval video dumps during Cosmos training.

The upstream ``every_n_sample_*`` callbacks are removed for GenET because they
do not preserve Source/Reference condition keys. This callback keeps those keys
and writes local MP4s (optionally logging them to W&B).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

_EVERY_N_IMPORT_ERROR: ImportError | None = None
try:
    from cosmos_framework.callbacks.every_n import EveryN as _EveryN
except ImportError as exc:  # pragma: no cover - NVIDIA container only
    _EVERY_N_IMPORT_ERROR = exc
    _EveryN = object  # type: ignore[misc,assignment]


def tensor_to_uint8_thwc(video: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert CTHW / TCHW / THWC video tensors to uint8 THWC."""

    array = video.detach().float().cpu().numpy() if isinstance(video, torch.Tensor) else np.asarray(video)
    if array.ndim != 4:
        raise ValueError(f"video must be rank-4, got {array.shape}")
    if array.shape[0] in {1, 3, 4} and array.shape[1] not in {1, 3, 4}:
        array = np.transpose(array, (1, 2, 3, 0))  # CTHW -> THWC
    elif array.shape[1] in {1, 3, 4}:
        array = np.transpose(array, (0, 2, 3, 1))  # TCHW -> THWC
    if array.shape[-1] == 4:
        array = array[..., :3]
    elif array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] != 3:
        raise ValueError(f"expected 3-channel RGB after layout fix, got {array.shape}")
    if np.issubdtype(array.dtype, np.floating):
        maximum = float(np.max(array)) if array.size else 0.0
        minimum = float(np.min(array)) if array.size else 0.0
        if minimum < -0.01:
            # Common VAE decode range [-1, 1].
            array = (array + 1.0) * 0.5
        elif maximum <= 1.0 + 1e-3:
            pass
        else:
            array = array / 255.0
        array = np.clip(array, 0.0, 1.0) * 255.0
    return np.clip(array, 0, 255).astype(np.uint8, copy=False)


def write_video_mp4(path: str | Path, video: torch.Tensor | np.ndarray, *, fps: float) -> Path:
    """Encode an RGB video tensor/array to MP4 via PyAV."""

    if fps <= 0:
        raise ValueError("fps must be positive")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frames = tensor_to_uint8_thwc(video)
    try:
        import av
    except ImportError as exc:  # pragma: no cover - optional on light hosts
        raise RuntimeError(
            "writing eval MP4 requires PyAV; install GenET with the 'preprocess' extra"
        ) from exc

    container = av.open(str(destination), mode="w")
    try:
        # PyAV expects an int or fractions.Fraction, not a bare float.
        rate = int(round(fps)) if abs(fps - round(fps)) < 1e-6 else fps
        if isinstance(rate, float):
            from fractions import Fraction

            rate = Fraction(rate).limit_denominator(1000)
        stream = container.add_stream("libx264", rate=rate)
        stream.width = int(frames.shape[2])
        stream.height = int(frames.shape[1])
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "23", "preset": "veryfast"}
        for frame in frames:
            packet_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
            for packet in stream.encode(packet_frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()
    return destination


def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device, non_blocking=True)
        elif isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
            moved[key] = [item.to(device, non_blocking=True) for item in value]
        else:
            moved[key] = value
    return moved


def _extract_vision_sample(sample: Any) -> torch.Tensor:
    if isinstance(sample, dict):
        if "vision" in sample:
            return sample["vision"]
        if "video" in sample:
            return sample["video"]
    if isinstance(sample, torch.Tensor):
        return sample
    raise TypeError(f"unsupported generate_samples_from_batch return type: {type(sample)!r}")


class GenETEvalVideoCallback(_EveryN):  # type: ignore[misc]
    """Every-N callback that dumps GenET-conditioned inference videos."""

    def __init__(
        self,
        every_n: int = 1000,
        *,
        num_samples: int = 2,
        manifest: str | None = None,
        output_dir: str,
        fps: float = 16.0,
        log_to_wandb: bool = True,
        run_at_start: bool = False,
        num_sampling_steps: int = 35,
        embodiment_map: dict[str, int] | None = None,
        dataset_kwargs: dict[str, Any] | None = None,
        packing_kwargs: dict[str, Any] | None = None,
        loader_kwargs: dict[str, Any] | None = None,
        barrier_after_run: bool = True,
    ) -> None:
        if _EVERY_N_IMPORT_ERROR is not None:  # pragma: no cover
            raise RuntimeError(
                "GenETEvalVideoCallback requires cosmos-framework EveryN"
            ) from _EVERY_N_IMPORT_ERROR
        super().__init__(
            every_n=every_n,
            step_size=1,
            barrier_after_run=barrier_after_run,
            run_at_start=run_at_start,
        )
        self.num_samples = int(num_samples)
        self.manifest = manifest
        self.output_dir = Path(output_dir)
        self.fps = float(fps)
        self.log_to_wandb = bool(log_to_wandb)
        self.num_sampling_steps = int(num_sampling_steps)
        self.embodiment_map = dict(embodiment_map or {})
        self.dataset_kwargs = dict(dataset_kwargs or {})
        self.packing_kwargs = dict(packing_kwargs or {})
        self.loader_kwargs = dict(loader_kwargs or {})
        self._val_batches: list[dict[str, Any]] = []

    def on_train_start(self, model: Any, iteration: int = 0) -> None:
        del model, iteration
        if not self.manifest:
            return
        if not self.embodiment_map:
            raise ValueError("eval_video.manifest requires embodiment_map")
        from genet.integrations.cosmos_data import CosmosProcessedPairDataset
        from genet.integrations.cosmos_loader import (
            CosmosInfiniteRankPartitionedDataLoader,
            CosmosResumeAwarePackingDataLoader,
        )
        from cosmos_framework.utils.lazy_config import LazyCall as L
        from cosmos_framework.utils.lazy_config import instantiate

        manifest = Path(self.manifest).expanduser().resolve()
        if not manifest.is_file():
            raise FileNotFoundError(f"eval_video.manifest does not exist: {manifest}")

        worker_kwargs = {
            "batch_size": 1,
            "in_order": True,
            "num_workers": 0,
            "persistent_workers": False,
            "pin_memory": False,
            "drop_last": False,
            "sampler": None,
            **self.loader_kwargs,
        }
        loader = instantiate(
            L(CosmosResumeAwarePackingDataLoader)(
                **self.packing_kwargs,
                dataloader=L(CosmosInfiniteRankPartitionedDataLoader)(
                    datasets={
                        "genet_pairs": {
                            "ratio": 1,
                            "dataset": L(CosmosProcessedPairDataset)(
                                manifest=str(manifest),
                                embodiment_map=self.embodiment_map,
                                **self.dataset_kwargs,
                            ),
                        }
                    },
                    seed=0,
                    **worker_kwargs,
                ),
            )
        )
        batches: list[dict[str, Any]] = []
        iterator = iter(loader)
        for _ in range(self.num_samples):
            batch = next(iterator)
            cpu_batch = {
                key: (
                    value.detach().cpu()
                    if isinstance(value, torch.Tensor)
                    else (
                        [item.detach().cpu() for item in value]
                        if isinstance(value, list)
                        and value
                        and isinstance(value[0], torch.Tensor)
                        else value
                    )
                )
                for key, value in batch.items()
            }
            batches.append(cpu_batch)
        self._val_batches = batches

    def every_n_impl(
        self,
        trainer: Any,
        model: Any,
        data_batch: dict[str, Any],
        output_batch: dict[str, Any],
        loss: torch.Tensor,
        iteration: int,
    ) -> None:
        del trainer, output_batch, loss
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        world = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        # Only rank 0 encodes MP4s; other ranks still run sampling so collectives stay aligned.
        batches = self._val_batches if self._val_batches else [data_batch]
        batches = batches[: self.num_samples]
        device = next(model.parameters()).device
        step_dir = self.output_dir / f"step_{iteration:08d}"
        if rank == 0:
            step_dir.mkdir(parents=True, exist_ok=True)

        logged: dict[str, Any] = {}
        for index, batch in enumerate(batches):
            # Replicate the same batch on every rank to keep sampler collectives happy.
            local_batch = _move_batch_to_device(batch, device)
            with torch.no_grad():
                # Cosmos requires one seed per sample; the eval loader is built
                # with batch_size=1, so each batch carries exactly one sample.
                sample = model.generate_samples_from_batch(
                    local_batch,
                    num_steps=self.num_sampling_steps,
                    seed=[int(iteration) + index],
                )
                vision = _extract_vision_sample(sample)
                if isinstance(vision, (list, tuple)):
                    vision = vision[0]
                decoded = model.decode(vision)
                if isinstance(decoded, (list, tuple)):
                    decoded = decoded[0]
                if decoded.ndim == 5:
                    decoded = decoded[0]
            if rank != 0:
                continue
            sample_id = f"sample{index}"
            path = write_video_mp4(
                step_dir / f"{sample_id}_pred.mp4",
                decoded,
                fps=self.fps,
            )
            logged[f"eval_video/{sample_id}"] = str(path)
            target = local_batch.get("video")
            if isinstance(target, torch.Tensor):
                gt = target[0] if target.ndim == 5 else target
                gt_path = write_video_mp4(
                    step_dir / f"{sample_id}_gt.mp4",
                    gt,
                    fps=self.fps,
                )
                logged[f"eval_video/{sample_id}_gt"] = str(gt_path)

        if rank == 0 and self.log_to_wandb:
            try:
                import wandb
            except ImportError:
                wandb = None  # type: ignore[assignment]
            if wandb is not None and wandb.run is not None:
                media = {
                    key: wandb.Video(path, fps=self.fps, format="mp4")
                    for key, path in logged.items()
                    if path.endswith(".mp4")
                }
                if media:
                    wandb.log(media, step=iteration)

        if world > 1:
            torch.distributed.barrier()
