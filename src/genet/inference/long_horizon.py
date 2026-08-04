"""Transactional rolling-window generation for long video/action trajectories.

The training contract remains a fixed clip.  This module repeatedly invokes a
single-window sampler, conditions each later window on the accepted target
prefix, commits only the new suffix, and rolls back video and action together
when every candidate at a boundary fails quality checks.

Tensor conventions are deliberately explicit:

* video: ``[C,T,H,W]``; uint8 or normalized floating point,
* action: ``[T,D]`` for frame alignment or ``[T-1,D]`` for transition alignment,
* all temporal intervals are half-open and video/action commits are atomic.
"""

from __future__ import annotations

import dataclasses
import fcntl
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import torch
import yaml

LONG_GENERATION_FORMAT = "genet.long-generation/v1"


@dataclass(frozen=True)
class SamplingConfig:
    guidance: float = 1.5
    num_steps: int = 35
    shift: float = 5.0
    sigma_max: float = 80.0
    normalize_cfg: bool = False
    has_negative_prompt: bool = False

    def validate(self) -> None:
        if not isinstance(self.normalize_cfg, bool) or not isinstance(
            self.has_negative_prompt, bool
        ):
            raise ValueError("sampling boolean flags must be bool")
        numeric = (int, float)
        if (
            isinstance(self.guidance, bool)
            or not isinstance(self.guidance, numeric)
            or not math.isfinite(self.guidance)
            or self.guidance < 0
        ):
            raise ValueError("sampling.guidance must be non-negative")
        if (
            isinstance(self.num_steps, bool)
            or not isinstance(self.num_steps, int)
            or self.num_steps < 1
        ):
            raise ValueError("sampling.num_steps must be positive")
        if (
            any(
                isinstance(value, bool)
                or not isinstance(value, numeric)
                or not math.isfinite(value)
                for value in (self.shift, self.sigma_max)
            )
            or self.shift <= 0
            or self.sigma_max <= 0
        ):
            raise ValueError("sampling.shift and sampling.sigma_max must be positive")


@dataclass(frozen=True)
class RecoveryConfig:
    max_attempts_per_chunk: int = 3
    rollback_depth: int = 1
    rollback_chunks: int = 1
    max_total_rollbacks: int = 8
    resume: bool = True

    def validate(self) -> None:
        if not isinstance(self.resume, bool):
            raise ValueError("recovery.resume must be bool")
        integer_values = {
            "max_attempts_per_chunk": self.max_attempts_per_chunk,
            "rollback_depth": self.rollback_depth,
            "rollback_chunks": self.rollback_chunks,
            "max_total_rollbacks": self.max_total_rollbacks,
        }
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in integer_values.values()
        ):
            raise ValueError("recovery counts must be integers")
        if self.max_attempts_per_chunk < 1:
            raise ValueError("recovery.max_attempts_per_chunk must be positive")
        if self.rollback_depth < 0 or self.rollback_chunks < 0:
            raise ValueError("rollback_depth and rollback_chunks must be non-negative")
        if self.rollback_chunks > self.rollback_depth:
            raise ValueError("rollback_chunks cannot exceed rollback_depth")
        if self.max_total_rollbacks < 0:
            raise ValueError("recovery.max_total_rollbacks must be non-negative")


@dataclass(frozen=True)
class ContinuityConfig:
    require_finite: bool = True
    # Normalized to [0,1] regardless of uint8 or [-1,1] video representation.
    max_video_prefix_mae: float | None = 0.20
    # Relative to max(mean(abs(context)), 1), so this is also safe near zero.
    max_action_prefix_relative_mae: float | None = 1.0e-4
    # Boundary metrics are always recorded. Absolute gates remain disabled until
    # embodiment-specific normalization/physics limits are supplied.
    max_video_boundary_mae: float | None = None
    max_action_boundary_relative_jump: float | None = None

    def validate(self) -> None:
        if not isinstance(self.require_finite, bool):
            raise ValueError("continuity.require_finite must be bool")
        for name, value in dataclasses.asdict(self).items():
            if name == "require_finite" or value is None:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"continuity.{name} must be non-negative or null")


@dataclass(frozen=True)
class LongHorizonConfig:
    chunk_frames: int = 81
    overlap_frames: int = 17
    temporal_compression_factor: int = 4
    action_alignment: Literal["frame", "transition"] = "frame"
    base_seed: int = 42
    store_video_dtype: Literal["float16", "float32", "uint8"] = "float16"
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    continuity: ContinuityConfig = field(default_factory=ContinuityConfig)

    def validate(self) -> None:
        for name in (
            "chunk_frames",
            "overlap_frames",
            "temporal_compression_factor",
            "base_seed",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer")
        if self.chunk_frames < 1:
            raise ValueError("chunk_frames must be positive")
        if self.temporal_compression_factor < 1:
            raise ValueError("temporal_compression_factor must be positive")
        if (self.chunk_frames - 1) % self.temporal_compression_factor:
            raise ValueError(
                "chunk_frames must equal 1 + N * temporal_compression_factor"
            )
        if not 0 <= self.overlap_frames < self.chunk_frames:
            raise ValueError("overlap_frames must satisfy 0 <= overlap < chunk_frames")
        if self.overlap_frames and (
            (self.overlap_frames - 1) % self.temporal_compression_factor
        ):
            raise ValueError(
                "non-zero overlap_frames must equal 1 + N * temporal_compression_factor"
            )
        if self.action_alignment not in {"frame", "transition"}:
            raise ValueError("action_alignment must be 'frame' or 'transition'")
        if self.action_alignment == "transition" and self.overlap_frames == 0:
            raise ValueError(
                "transition-aligned generation requires overlap_frames >= 1 so "
                "the action crossing each chunk boundary is generated exactly once"
            )
        if self.store_video_dtype not in {"float16", "float32", "uint8"}:
            raise ValueError("store_video_dtype must be float16, float32, or uint8")
        self.sampling.validate()
        self.recovery.validate()
        self.continuity.validate()

    @property
    def stride_frames(self) -> int:
        return self.chunk_frames - self.overlap_frames

    @property
    def chunk_action_steps(self) -> int:
        return self.chunk_frames if self.action_alignment == "frame" else self.chunk_frames - 1

    def action_steps_for_video_frames(self, frames: int) -> int:
        return frames if self.action_alignment == "frame" else max(frames - 1, 0)

    @property
    def overlap_action_steps(self) -> int:
        return self.action_steps_for_video_frames(self.overlap_frames)

    @property
    def overlap_latent_frames(self) -> int:
        if self.overlap_frames == 0:
            return 0
        return 1 + (self.overlap_frames - 1) // self.temporal_compression_factor

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def fingerprint(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "LongHorizonConfig":
        values = dict(raw)
        allowed = {field.name for field in dataclasses.fields(cls)}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"unknown long-horizon config keys: {', '.join(unknown)}")
        nested: tuple[tuple[str, type[Any]], ...] = (
            ("sampling", SamplingConfig),
            ("recovery", RecoveryConfig),
            ("continuity", ContinuityConfig),
        )
        for name, nested_type in nested:
            item = values.get(name)
            if item is not None:
                if not isinstance(item, Mapping):
                    raise ValueError(f"{name} must be a mapping")
                nested_allowed = {field.name for field in dataclasses.fields(nested_type)}
                nested_unknown = sorted(set(item) - nested_allowed)
                if nested_unknown:
                    raise ValueError(
                        f"unknown {name} config keys: {', '.join(nested_unknown)}"
                    )
                values[name] = nested_type(**dict(item))
        result = cls(**values)
        result.validate()
        return result


def load_long_horizon_config(path: str | Path) -> LongHorizonConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("long-horizon YAML must contain a mapping")
    return LongHorizonConfig.from_mapping(raw)


@dataclass(frozen=True)
class RunIdentity:
    source_id: str
    model_id: str
    reference_id: str
    normalization_id: str = "none"
    code_revision: str = "unknown"

    def validate(self) -> None:
        for field_info in dataclasses.fields(self):
            value = getattr(self, field_info.name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"identity.{field_info.name} must be a non-empty string")


@dataclass(frozen=True)
class SourceWindow:
    video: torch.Tensor
    action: torch.Tensor
    valid_video_frames: int
    valid_action_steps: int


class WindowSource(Protocol):
    @property
    def num_video_frames(self) -> int: ...

    def read_window(self, start_frame: int, config: LongHorizonConfig) -> SourceWindow: ...


class TensorWindowSource:
    """In-memory/memmap-friendly source reader with repeat padding at the tail."""

    def __init__(self, video: torch.Tensor, action: torch.Tensor, *, alignment: str = "frame") -> None:
        if video.ndim != 4:
            raise ValueError("source video must have shape [C,T,H,W]")
        if action.ndim != 2:
            raise ValueError("source action must have shape [T,D] or [T-1,D]")
        if video.shape[1] < 1:
            raise ValueError("source video must contain at least one frame")
        if alignment not in {"frame", "transition"}:
            raise ValueError("alignment must be frame or transition")
        expected_action = video.shape[1] if alignment == "frame" else video.shape[1] - 1
        if action.shape[0] != expected_action:
            raise ValueError(
                f"source action length {action.shape[0]} does not match {alignment} "
                f"alignment expectation {expected_action}"
            )
        self.video = video
        self.action = action
        self.alignment = alignment

    @property
    def num_video_frames(self) -> int:
        return int(self.video.shape[1])

    @staticmethod
    def _repeat_pad(tensor: torch.Tensor, *, axis: int, length: int) -> torch.Tensor:
        current = tensor.shape[axis]
        if current == length:
            return tensor
        if current > length:
            return tensor.narrow(axis, 0, length)
        if current == 0:
            shape = list(tensor.shape)
            shape[axis] = length
            return tensor.new_zeros(shape)
        last = tensor.narrow(axis, current - 1, 1)
        repeats = [1] * tensor.ndim
        repeats[axis] = length - current
        return torch.cat([tensor, last.repeat(*repeats)], dim=axis)

    def read_window(self, start_frame: int, config: LongHorizonConfig) -> SourceWindow:
        if config.action_alignment != self.alignment:
            raise ValueError("source alignment differs from long-horizon config")
        if not 0 <= start_frame < self.num_video_frames:
            raise IndexError(start_frame)
        video_end = min(start_frame + config.chunk_frames, self.num_video_frames)
        video = self.video[:, start_frame:video_end]
        valid_video = int(video.shape[1])
        video = self._repeat_pad(video, axis=1, length=config.chunk_frames)

        action_start = start_frame
        requested_action = config.chunk_action_steps
        action_end = min(action_start + requested_action, self.action.shape[0])
        action = self.action[action_start:action_end]
        valid_action = int(action.shape[0])
        action = self._repeat_pad(action, axis=0, length=requested_action)
        return SourceWindow(video, action, valid_video, valid_action)


@dataclass(frozen=True)
class ChunkRequest:
    window_start_frame: int
    committed_frames: int
    context_video: torch.Tensor | None
    context_action: torch.Tensor | None
    source_video: torch.Tensor
    source_action: torch.Tensor
    new_video_frames: int
    new_action_steps: int
    valid_window_frames: int
    valid_window_action_steps: int
    attempt_id: int
    seed: int
    rollback_count: int
    config: LongHorizonConfig

    @property
    def context_frames(self) -> int:
        return 0 if self.context_video is None else int(self.context_video.shape[1])

    @property
    def context_action_steps(self) -> int:
        return 0 if self.context_action is None else int(self.context_action.shape[0])

    @property
    def condition_video_latent_indexes(self) -> tuple[int, ...]:
        if self.context_frames == 0:
            return ()
        count = 1 + (self.context_frames - 1) // self.config.temporal_compression_factor
        return tuple(range(count))

    @property
    def condition_action_indexes(self) -> tuple[int, ...]:
        return tuple(range(self.context_action_steps))


@dataclass
class ChunkOutput:
    video: torch.Tensor
    action: torch.Tensor
    metadata: dict[str, Any] = field(default_factory=dict)


class WindowSampler(Protocol):
    def __call__(self, request: ChunkRequest) -> ChunkOutput: ...


@dataclass(frozen=True)
class QualityReport:
    accepted: bool
    metrics: dict[str, float]
    failures: tuple[str, ...] = ()


def _video_unit_mae(left: torch.Tensor, right: torch.Tensor) -> float:
    def normalized(value: torch.Tensor) -> torch.Tensor:
        result = value.detach().to(device="cpu", dtype=torch.float32)
        return result.div(127.5).sub(1.0) if value.dtype == torch.uint8 else result

    return float((normalized(left) - normalized(right)).abs().mean().item() / 2.0)


class DefaultContinuityEvaluator:
    """Schema-agnostic hard gates plus diagnostic seam metrics."""

    def __init__(
        self,
        config: ContinuityConfig,
        *,
        target_action_dim: int | None = None,
    ) -> None:
        config.validate()
        if target_action_dim is not None and (
            isinstance(target_action_dim, bool)
            or not isinstance(target_action_dim, int)
            or target_action_dim < 1
        ):
            raise ValueError("target_action_dim must be positive when provided")
        self.config = config
        self.target_action_dim = target_action_dim

    def __call__(self, request: ChunkRequest, output: ChunkOutput) -> QualityReport:
        failures: list[str] = []
        metrics: dict[str, float] = {}
        expected_video = (
            request.source_video.shape[0],
            request.config.chunk_frames,
            request.source_video.shape[2],
            request.source_video.shape[3],
        )
        video_is_tensor = isinstance(output.video, torch.Tensor)
        action_is_tensor = isinstance(output.action, torch.Tensor)
        if not video_is_tensor:
            failures.append(f"video_type:{type(output.video).__name__}")
        elif tuple(output.video.shape) != tuple(expected_video):
            failures.append(
                f"video_shape:{tuple(output.video.shape)}!={tuple(expected_video)}"
            )
        if not action_is_tensor:
            failures.append(f"action_type:{type(output.action).__name__}")
        elif output.action.ndim != 2 or output.action.shape[0] != request.config.chunk_action_steps:
            failures.append(
                f"action_shape:{tuple(output.action.shape)}!="
                f"({request.config.chunk_action_steps},D)"
            )
        elif (
            self.target_action_dim is not None
            and output.action.shape[1] != self.target_action_dim
        ):
            failures.append(
                f"action_width:{output.action.shape[1]}!={self.target_action_dim}"
            )
        if (
            request.context_action is not None
            and action_is_tensor
            and output.action.ndim == 2
            and output.action.shape[1] != request.context_action.shape[1]
        ):
            failures.append(
                f"action_width:{output.action.shape[1]}!={request.context_action.shape[1]}"
            )
        if self.config.require_finite:
            if video_is_tensor and not bool(torch.isfinite(output.video).all()):
                failures.append("video_non_finite")
            if action_is_tensor and not bool(torch.isfinite(output.action).all()):
                failures.append("action_non_finite")

        if not failures and request.context_video is not None:
            context_frames = request.context_frames
            video_prefix_mae = _video_unit_mae(
                output.video[:, :context_frames], request.context_video
            )
            metrics["video_prefix_mae"] = video_prefix_mae
            threshold = self.config.max_video_prefix_mae
            if threshold is not None and video_prefix_mae > threshold:
                failures.append("video_prefix_mae")
            if context_frames < output.video.shape[1]:
                metrics["video_boundary_mae"] = _video_unit_mae(
                    request.context_video[:, -1],
                    output.video[:, context_frames],
                )
                boundary_threshold = self.config.max_video_boundary_mae
                if (
                    boundary_threshold is not None
                    and metrics["video_boundary_mae"] > boundary_threshold
                ):
                    failures.append("video_boundary_mae")

        if (
            not failures
            and request.context_action is not None
            and request.context_action_steps > 0
        ):
            context_steps = request.context_action_steps
            prefix = output.action[:context_steps].detach().to(torch.float32)
            expected = request.context_action.detach().to(torch.float32)
            denominator = max(float(expected.abs().mean().item()), 1.0)
            relative_mae = float((prefix - expected).abs().mean().item() / denominator)
            metrics["action_prefix_relative_mae"] = relative_mae
            threshold = self.config.max_action_prefix_relative_mae
            if threshold is not None and relative_mae > threshold:
                failures.append("action_prefix_relative_mae")
            if context_steps and context_steps < output.action.shape[0]:
                jump_scale = max(
                    float(request.context_action.detach().to(torch.float32).abs().mean().item()),
                    1.0,
                )
                metrics["action_boundary_relative_jump"] = float(
                    (
                        output.action[context_steps].to(torch.float32)
                        - request.context_action[-1].to(torch.float32)
                    )
                    .abs()
                    .mean()
                    .item()
                    / jump_scale
                )
                boundary_threshold = self.config.max_action_boundary_relative_jump
                if (
                    boundary_threshold is not None
                    and metrics["action_boundary_relative_jump"] > boundary_threshold
                ):
                    failures.append("action_boundary_relative_jump")
        return QualityReport(not failures, metrics, tuple(failures))


@dataclass(frozen=True)
class LongHorizonResult:
    output_dir: Path
    total_video_frames: int
    total_action_steps: int
    chunks: int
    rollbacks: int
    status: str


@dataclass(frozen=True)
class _Snapshot:
    committed_frames: int
    chunks: int
    rollbacks: int
    parent_digest: str
    context_video: torch.Tensor | None
    context_action: torch.Tensor | None
    status: str
    failure: str | None


def _fsync_directory(path: Path) -> None:
    """Persist a rename in ``path`` before publishing dependent state."""

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}-", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        _fsync_directory(path.parent)
    finally:
        temporary = Path(temporary_name)
        if temporary.exists():
            temporary.unlink()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_clone(value: Any, *, label: str) -> Any:
    """Validate journal metadata before any candidate artifact is published."""

    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} must contain only finite JSON values") from error


class _RunLock:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("a+", encoding="utf-8")

    def __enter__(self) -> "_RunLock":
        try:
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self._handle.close()
            raise
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        self._handle.close()


class _Coordinator:
    def __init__(self) -> None:
        self.enabled = torch.distributed.is_available() and torch.distributed.is_initialized()
        self.rank = torch.distributed.get_rank() if self.enabled else 0
        self.world_size = torch.distributed.get_world_size() if self.enabled else 1

    @property
    def is_primary(self) -> bool:
        return self.rank == 0

    def broadcast(self, value: Any) -> Any:
        if not self.enabled:
            return value
        payload = [value if self.is_primary else None]
        torch.distributed.broadcast_object_list(payload, src=0)
        return payload[0]

    @staticmethod
    def _error_payload(error: BaseException) -> dict[str, str]:
        return {
            "type": type(error).__name__,
            "message": str(error),
        }

    @staticmethod
    def _raise_remote(stage: str, error: Mapping[str, str]) -> None:
        raise RuntimeError(
            f"distributed long-generation stage {stage!r} failed on a peer: "
            f"{error.get('type', 'Exception')}: {error.get('message', '')}"
        )

    def run_primary(self, stage: str, operation: Callable[[], Any]) -> Any:
        """Run a rank-zero mutation and broadcast either its value or failure."""

        if not self.enabled:
            return operation()
        packet: dict[str, Any] | None = None
        if self.is_primary:
            try:
                packet = {"ok": True, "value": operation()}
            except BaseException as error:
                packet = {"ok": False, "error": self._error_payload(error)}
        packet = self.broadcast(packet)
        if not isinstance(packet, dict):
            raise RuntimeError("distributed primary stage returned an invalid packet")
        if not packet["ok"]:
            self._raise_remote(stage, packet["error"])
        return packet["value"]

    def run_all(self, stage: str, operation: Callable[[], Any]) -> Any:
        """Run local preparation/sampling and make Python-side failures collective."""

        if not self.enabled:
            return operation()
        value: Any = None
        local_error: dict[str, str] | None = None
        try:
            value = operation()
        except BaseException as error:
            local_error = self._error_payload(error)
        errors: list[dict[str, str] | None] = [None] * self.world_size
        torch.distributed.all_gather_object(errors, local_error)
        for peer_rank, peer_error in enumerate(errors):
            if peer_error is not None:
                self._raise_remote(f"{stage}/rank-{peer_rank}", peer_error)
        return value

    def require_equal(self, stage: str, value: Any) -> None:
        if not self.enabled:
            return
        values: list[Any] = [None] * self.world_size
        torch.distributed.all_gather_object(values, value)
        if any(item != values[0] for item in values[1:]):
            raise RuntimeError(
                f"distributed long-generation stage {stage!r} differs across ranks: "
                f"{values!r}"
            )


class GenerationJournal:
    """Single-writer immutable-chunk journal with recoverable rollback."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        config: LongHorizonConfig,
        identity: RunIdentity,
        total_frames: int,
        resume: bool,
    ) -> None:
        config.validate()
        identity.validate()
        if total_frames < 1:
            raise ValueError("total_frames must be positive")
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_dir = self.output_dir / "chunks"
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.output_dir / "RUN.json"
        self.config = config
        self.identity = identity
        self.total_frames = int(total_frames)
        self.state: dict[str, Any]
        if self.manifest_path.exists():
            if not resume:
                raise FileExistsError(
                    f"long-generation state exists; enable resume: {self.manifest_path}"
                )
            self.state = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            self._validate_loaded()
            self._verify_active_chunks()
        else:
            self.state = {
                "format_version": LONG_GENERATION_FORMAT,
                "config": config.as_dict(),
                "config_fingerprint": config.fingerprint(),
                "identity": dataclasses.asdict(identity),
                "total_frames": self.total_frames,
                "status": "running",
                "failure": None,
                "committed_frames": 0,
                "finalized_frames": 0,
                "total_rollbacks": 0,
                "attempt_counters": {},
                "attempts": [],
                "active_chunks": [],
                "superseded_chunks": [],
                "output_contract": None,
            }
            self._write()

    def _validate_loaded(self) -> None:
        expected = {
            "format_version": LONG_GENERATION_FORMAT,
            "config": self.config.as_dict(),
            "config_fingerprint": self.config.fingerprint(),
            "identity": dataclasses.asdict(self.identity),
            "total_frames": self.total_frames,
        }
        for key, value in expected.items():
            if self.state.get(key) != value:
                raise ValueError(
                    f"resume state {key} mismatch: {self.state.get(key)!r} != {value!r}"
                )
        required = {
            "status",
            "failure",
            "committed_frames",
            "finalized_frames",
            "total_rollbacks",
            "attempt_counters",
            "attempts",
            "active_chunks",
            "superseded_chunks",
            "output_contract",
        }
        missing = sorted(required - self.state.keys())
        if missing:
            raise ValueError(f"resume state is missing keys: {', '.join(missing)}")
        for key in ("attempts", "active_chunks", "superseded_chunks"):
            if not isinstance(self.state[key], list):
                raise ValueError(f"resume state {key} must be a list")
        if not isinstance(self.state["attempt_counters"], Mapping):
            raise ValueError("resume state attempt_counters must be a mapping")
        if int(self.state["total_rollbacks"]) < 0:
            raise ValueError("resume state total_rollbacks must be non-negative")
        status = self.state["status"]
        if status not in {"running", "complete", "failed"}:
            raise ValueError(f"resume state has invalid status {status!r}")
        committed = int(self.state["committed_frames"])
        if not 0 <= committed <= self.total_frames:
            raise ValueError("resume state committed_frames lies outside the run")
        if status == "complete" and committed != self.total_frames:
            raise ValueError("complete resume state does not cover the requested timeline")
        if status == "complete" and int(self.state["finalized_frames"]) != self.total_frames:
            raise ValueError("complete resume state is not fully finalized")
        if status == "failed" and not self.state.get("failure"):
            raise ValueError("failed resume state is missing its failure reason")

    def _verify_active_chunks(self) -> None:
        cursor = 0
        action_steps = 0
        chunk_boundaries = {0}
        contract = self.state.get("output_contract")
        for record in self.state["active_chunks"]:
            relative_path = Path(record["path"])
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(
                    f"active chunk path must be canonical and relative: {relative_path}"
                )
            path = (self.output_dir / relative_path).resolve()
            try:
                path.relative_to(self.chunks_dir)
            except ValueError as error:
                raise ValueError(f"active chunk is outside the immutable chunk store: {path}") from error
            if not path.is_file() or _sha256_file(path) != record["sha256"]:
                raise ValueError(f"active long-generation chunk is missing/corrupt: {path}")
            if record["start_frame"] != cursor:
                raise ValueError("active chunk timeline is not contiguous")
            end_frame = int(record["end_frame"])
            if not cursor < end_frame <= self.total_frames:
                raise ValueError("active chunk has an invalid video interval")
            contribution_frames = end_frame - cursor
            contribution_actions = int(record["action_steps"])
            video_shape = record.get("video_shape")
            action_shape = record.get("action_shape")
            if not isinstance(contract, Mapping):
                raise ValueError("active chunks require an output_contract")
            if video_shape != [
                int(contract["video_channels"]),
                contribution_frames,
                int(contract["video_height"]),
                int(contract["video_width"]),
            ]:
                raise ValueError("active chunk video shape metadata is inconsistent")
            if action_shape != [contribution_actions, int(contract["action_dim"])]:
                raise ValueError("active chunk action shape metadata is inconsistent")
            accepted = any(
                item.get("status") == "accepted"
                and int(item["window_start"]) == int(record["window_start"])
                and int(item["attempt_id"]) == int(record["attempt_id"])
                and int(item["seed"]) == int(record["seed"])
                for item in self.state["attempts"]
            )
            if not accepted:
                raise ValueError("active chunk has no matching accepted attempt")
            cursor = end_frame
            chunk_boundaries.add(cursor)
            action_steps += contribution_actions
        if cursor != int(self.state["committed_frames"]):
            raise ValueError("committed_frames does not match active chunk timeline")
        expected_actions = self.config.action_steps_for_video_frames(cursor)
        if action_steps != expected_actions:
            raise ValueError(
                "active action timeline does not match the committed video timeline: "
                f"{action_steps} != {expected_actions}"
            )
        finalized = int(self.state["finalized_frames"])
        if not 0 <= finalized <= cursor:
            raise ValueError("finalized_frames lies outside the committed timeline")
        if finalized not in chunk_boundaries:
            raise ValueError("finalized_frames is not an active chunk boundary")

    def _write(self) -> None:
        _atomic_json(self.manifest_path, self.state)

    def allocate_attempt(self, window_start: int, parent_digest: str) -> tuple[int, int]:
        key = str(window_start)
        attempt_id = int(self.state["attempt_counters"].get(key, 0))
        self.state["attempt_counters"][key] = attempt_id + 1
        identity_payload = json.dumps(
            dataclasses.asdict(self.identity), sort_keys=True, separators=(",", ":")
        )
        identity_digest = hashlib.sha256(identity_payload.encode("utf-8")).hexdigest()
        seed_payload = (
            f"{LONG_GENERATION_FORMAT}\0{self.config.base_seed}\0{window_start}\0"
            f"{identity_digest}\0{parent_digest}\0{attempt_id}\0"
            f"{self.state['total_rollbacks']}"
        )
        seed = int.from_bytes(
            hashlib.blake2b(seed_payload.encode("utf-8"), digest_size=8).digest(),
            "little",
        ) % (2**31 - 1)
        self.state["attempts"].append(
            {
                "window_start": window_start,
                "attempt_id": attempt_id,
                "seed": seed,
                "parent_digest": parent_digest,
                "rollback_count": int(self.state["total_rollbacks"]),
                "status": "started",
            }
        )
        self._write()
        return attempt_id, seed

    def remaining_attempts(self, window_start: int, parent_digest: str) -> int:
        rollback_count = int(self.state["total_rollbacks"])
        consumed = sum(
            1
            for item in self.state["attempts"]
            if int(item["window_start"]) == window_start
            and item["parent_digest"] == parent_digest
            and int(item.get("rollback_count", 0)) == rollback_count
        )
        return max(self.config.recovery.max_attempts_per_chunk - consumed, 0)

    def mark_failed(self, reason: str) -> None:
        self.state["status"] = "failed"
        self.state["failure"] = str(reason)
        self._write()

    def _finish_attempt(
        self,
        window_start: int,
        attempt_id: int,
        *,
        status: str,
        report: QualityReport,
        write: bool = True,
    ) -> None:
        for item in reversed(self.state["attempts"]):
            if item["window_start"] == window_start and item["attempt_id"] == attempt_id:
                item["status"] = status
                item["metrics"] = _json_clone(report.metrics, label="quality metrics")
                item["failures"] = list(report.failures)
                if write:
                    self._write()
                return
        raise RuntimeError("attempt record disappeared before completion")

    def record_rejected(
        self, window_start: int, attempt_id: int, report: QualityReport
    ) -> None:
        self._finish_attempt(window_start, attempt_id, status="rejected", report=report)

    def _stored_video(self, video: torch.Tensor) -> np.ndarray:
        array = video.detach().cpu().numpy()
        if video.dtype == torch.uint8 and self.config.store_video_dtype != "uint8":
            array = array.astype(np.float32) / 127.5 - 1.0
        if self.config.store_video_dtype == "float16":
            return array.astype(np.float16)
        if self.config.store_video_dtype == "float32":
            return array.astype(np.float32)
        if video.dtype == torch.uint8:
            return array.astype(np.uint8)
        # Normalized [-1,1] -> uint8 only when explicitly requested.
        return np.rint((np.clip(array, -1.0, 1.0) + 1.0) * 127.5).astype(np.uint8)

    def commit(
        self,
        request: ChunkRequest,
        output: ChunkOutput,
        report: QualityReport,
    ) -> None:
        if not report.accepted:
            raise ValueError("cannot commit a rejected candidate")
        if not isinstance(output.video, torch.Tensor) or output.video.ndim != 4:
            raise ValueError("committed video must be a [C,T,H,W] tensor")
        if not isinstance(output.action, torch.Tensor) or output.action.ndim != 2:
            raise ValueError("committed action must be a [T,D] tensor")
        metadata = _json_clone(output.metadata, label="chunk metadata")
        metrics = _json_clone(report.metrics, label="quality metrics")
        current_head = int(self.state["committed_frames"])
        if request.committed_frames != current_head:
            raise RuntimeError(
                f"stale candidate starts at {request.committed_frames}, journal head is "
                f"{current_head}"
            )
        end_frame = current_head + request.new_video_frames
        if request.new_video_frames < 1 or end_frame > self.total_frames:
            raise RuntimeError("candidate video suffix falls outside the requested timeline")
        expected_new_actions = self.config.action_steps_for_video_frames(
            end_frame
        ) - self.config.action_steps_for_video_frames(current_head)
        if request.new_action_steps != expected_new_actions:
            raise RuntimeError(
                "candidate action suffix does not match its global video interval: "
                f"{request.new_action_steps} != {expected_new_actions}"
            )
        matching_attempt = next(
            (
                item
                for item in reversed(self.state["attempts"])
                if int(item["window_start"]) == request.window_start_frame
                and int(item["attempt_id"]) == request.attempt_id
            ),
            None,
        )
        if matching_attempt is None or matching_attempt.get("status") != "started":
            raise RuntimeError("candidate does not match an active allocated attempt")
        if int(matching_attempt["seed"]) != request.seed:
            raise RuntimeError("candidate seed differs from its allocated attempt")
        current_parent = (
            self.state["active_chunks"][-1]["sha256"]
            if self.state["active_chunks"]
            else "root"
        )
        if matching_attempt["parent_digest"] != current_parent:
            raise RuntimeError("candidate parent differs from the current journal head")
        expected_window_start = current_head - (
            0 if current_head == 0 else self.config.overlap_frames
        )
        if request.window_start_frame != expected_window_start:
            raise RuntimeError("candidate window start differs from the current journal head")
        expected_video_shape = (
            request.source_video.shape[0],
            self.config.chunk_frames,
            request.source_video.shape[2],
            request.source_video.shape[3],
        )
        if tuple(output.video.shape) != expected_video_shape:
            raise ValueError(
                "candidate video does not match the complete source-window geometry"
            )
        if output.action.shape[0] != self.config.chunk_action_steps:
            raise ValueError("candidate action does not contain one complete model window")
        context_frames = request.context_frames
        context_actions = request.context_action_steps
        video = output.video[:, context_frames : context_frames + request.new_video_frames]
        action = output.action[
            context_actions : context_actions + request.new_action_steps
        ]
        if video.shape[1] != request.new_video_frames:
            raise RuntimeError("candidate did not contain the requested new video suffix")
        if action.shape[0] != request.new_action_steps:
            raise RuntimeError("candidate did not contain the requested new action suffix")
        contract = {
            "video_channels": int(output.video.shape[0]),
            "video_height": int(output.video.shape[2]),
            "video_width": int(output.video.shape[3]),
            "action_dim": int(output.action.shape[1]),
        }
        if self.state.get("output_contract") is None:
            self.state["output_contract"] = contract
        elif self.state["output_contract"] != contract:
            raise RuntimeError(
                f"candidate output contract changed: {contract!r} != "
                f"{self.state['output_contract']!r}"
            )

        sequence = len(self.state["active_chunks"]) + len(self.state["superseded_chunks"])
        chunk_name = (
            f"chunk-{sequence:06d}-{request.committed_frames:09d}-"
            f"{request.seed:010d}.npz"
        )
        destination = self.chunks_dir / chunk_name
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.stem}-", suffix=".npz", dir=self.chunks_dir
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            np.savez_compressed(
                temporary,
                video=self._stored_video(video),
                action=action.detach().cpu().numpy().astype(np.float32),
            )
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, destination)
            _fsync_directory(self.chunks_dir)
        finally:
            if temporary.exists():
                temporary.unlink()
        checksum = _sha256_file(destination)
        record = {
            "chunk_id": destination.stem,
            "path": str(destination.relative_to(self.output_dir)),
            "sha256": checksum,
            "window_start": request.window_start_frame,
            "start_frame": request.committed_frames,
            "end_frame": end_frame,
            "action_steps": request.new_action_steps,
            "video_shape": list(video.shape),
            "action_shape": list(action.shape),
            "attempt_id": request.attempt_id,
            "seed": request.seed,
            "metrics": metrics,
            "metadata": metadata,
        }
        self.state["active_chunks"].append(record)
        self.state["committed_frames"] = record["end_frame"]
        self._recompute_finalized(write=False)
        self._finish_attempt(
            request.window_start_frame,
            request.attempt_id,
            status="accepted",
            report=report,
            write=False,
        )
        self._write()

    def _recompute_finalized(self, *, write: bool = True) -> None:
        active = self.state["active_chunks"]
        depth = self.config.recovery.rollback_depth
        if depth == 0 and active:
            finalized = active[-1]["end_frame"]
        elif len(active) > depth:
            finalized = active[-depth - 1]["end_frame"]
        else:
            finalized = 0
        self.state["finalized_frames"] = max(
            int(self.state.get("finalized_frames", 0)), int(finalized)
        )
        if write:
            self._write()

    def can_rollback(self, count: int) -> bool:
        if count < 1:
            return False
        active = self.state["active_chunks"]
        if count > len(active):
            return False
        first_removed = active[-count]
        return int(first_removed["start_frame"]) >= int(self.state["finalized_frames"])

    def rollback(self, count: int) -> None:
        if count < 1:
            raise ValueError("rollback count must be positive")
        active = self.state["active_chunks"]
        if not self.can_rollback(count):
            raise RuntimeError("rollback would cross the finalized boundary")
        for _ in range(count):
            record = active.pop()
            superseded = dict(record)
            superseded["status"] = "superseded"
            self.state["superseded_chunks"].append(superseded)
        self.state["committed_frames"] = active[-1]["end_frame"] if active else 0
        self.state["total_rollbacks"] += 1
        # Candidate files are immutable and remain at their original paths. The
        # atomic manifest switch is therefore valid both before and after a crash;
        # optional garbage collection can happen only after the run completes.
        self._recompute_finalized(write=False)
        self._write()

    @staticmethod
    def _read_chunk(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
        with np.load(path, allow_pickle=False) as archive:
            video = torch.from_numpy(np.array(archive["video"], copy=True))
            action = torch.from_numpy(np.array(archive["action"], copy=True))
        return video, action.to(torch.float32)

    def tail(self, frames: int, action_steps: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if frames == 0:
            return None, None
        videos: list[torch.Tensor] = []
        actions: list[torch.Tensor] = []
        video_count = 0
        action_count = 0
        for record in reversed(self.state["active_chunks"]):
            video, action = self._read_chunk(self.output_dir / record["path"])
            videos.append(video)
            actions.append(action)
            video_count += video.shape[1]
            action_count += action.shape[0]
            if video_count >= frames and action_count >= action_steps:
                break
        if video_count < frames or action_count < action_steps:
            raise RuntimeError("active journal does not contain enough rollback context")
        video = torch.cat(list(reversed(videos)), dim=1)[:, -frames:]
        all_actions = torch.cat(list(reversed(actions)), dim=0)
        action = all_actions[-action_steps:] if action_steps else all_actions[:0]
        return video, action

    def snapshot(self) -> _Snapshot:
        committed = int(self.state["committed_frames"])
        context_frames = min(self.config.overlap_frames, committed)
        context_actions = self.config.action_steps_for_video_frames(context_frames)
        video, action = self.tail(context_frames, context_actions) if context_frames else (None, None)
        active = self.state["active_chunks"]
        parent = active[-1]["sha256"] if active else "root"
        return _Snapshot(
            committed_frames=committed,
            chunks=len(active),
            rollbacks=int(self.state["total_rollbacks"]),
            parent_digest=parent,
            context_video=video,
            context_action=action,
            status=str(self.state["status"]),
            failure=(
                str(self.state["failure"])
                if self.state.get("failure") is not None
                else None
            ),
        )

    def finalize(self) -> None:
        if int(self.state["committed_frames"]) != self.total_frames:
            raise RuntimeError("cannot finalize an incomplete long-generation run")
        self.state["status"] = "complete"
        self.state["finalized_frames"] = self.total_frames
        self._write()

    def iter_chunks(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Stream canonical active contributions without loading the full run."""

        for record in self.state["active_chunks"]:
            yield self._read_chunk(self.output_dir / record["path"])

    def materialize(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Materialize a small run; production encoders should use ``iter_chunks``."""

        chunks = list(self.iter_chunks())
        videos = [video for video, _action in chunks]
        actions = [action for _video, action in chunks]
        if not videos:
            raise RuntimeError("long-generation run has no active chunks")
        return torch.cat(videos, dim=1), torch.cat(actions, dim=0)


QualityEvaluator = Callable[[ChunkRequest, ChunkOutput], QualityReport]


class LongHorizonGenerator:
    def __init__(
        self,
        config: LongHorizonConfig,
        sampler: WindowSampler,
        *,
        evaluator: QualityEvaluator | None = None,
        target_action_dim: int | None = None,
    ) -> None:
        config.validate()
        self.config = config
        self.sampler = sampler
        inferred_action_dim = getattr(sampler, "target_action_dim", None)
        resolved_action_dim = (
            target_action_dim if target_action_dim is not None else inferred_action_dim
        )
        self.evaluator = evaluator or DefaultContinuityEvaluator(
            config.continuity,
            target_action_dim=resolved_action_dim,
        )

    def _request(
        self,
        source: WindowSource,
        snapshot: _Snapshot,
        *,
        attempt_id: int,
        seed: int,
    ) -> ChunkRequest:
        context_frames = 0 if snapshot.committed_frames == 0 else self.config.overlap_frames
        if context_frames > snapshot.committed_frames:
            raise RuntimeError("committed output is shorter than configured rolling context")
        window_start = snapshot.committed_frames - context_frames
        source_window = source.read_window(window_start, self.config)
        if not isinstance(source_window, SourceWindow):
            raise TypeError("WindowSource.read_window must return SourceWindow")
        if (
            not isinstance(source_window.video, torch.Tensor)
            or source_window.video.ndim != 4
            or source_window.video.shape[1] != self.config.chunk_frames
        ):
            raise ValueError("source window video must have shape [C,chunk_frames,H,W]")
        if (
            not isinstance(source_window.action, torch.Tensor)
            or source_window.action.ndim != 2
            or source_window.action.shape[0] != self.config.chunk_action_steps
        ):
            raise ValueError("source window action must have shape [chunk_action_steps,D]")
        remaining = source.num_video_frames - snapshot.committed_frames
        new_video = min(self.config.chunk_frames - context_frames, remaining)
        valid_window_video = context_frames + new_video
        valid_window_action = self.config.action_steps_for_video_frames(valid_window_video)
        if source_window.valid_video_frames != valid_window_video:
            raise ValueError(
                "source window valid video length differs from the target timeline"
            )
        if source_window.valid_action_steps != valid_window_action:
            raise ValueError(
                "source window valid action length differs from the target timeline"
            )
        context_actions = self.config.action_steps_for_video_frames(context_frames)
        new_action = valid_window_action - context_actions
        return ChunkRequest(
            window_start_frame=window_start,
            committed_frames=snapshot.committed_frames,
            context_video=snapshot.context_video,
            context_action=snapshot.context_action,
            source_video=source_window.video,
            source_action=source_window.action,
            new_video_frames=new_video,
            new_action_steps=new_action,
            valid_window_frames=valid_window_video,
            valid_window_action_steps=valid_window_action,
            attempt_id=attempt_id,
            seed=seed,
            rollback_count=snapshot.rollbacks,
            config=self.config,
        )

    def generate(
        self,
        source: WindowSource,
        *,
        output_dir: str | Path,
        identity: RunIdentity,
        resume: bool | None = None,
    ) -> LongHorizonResult:
        coordinator = _Coordinator()
        resolved_resume = self.config.recovery.resume if resume is None else bool(resume)
        journal: GenerationJournal | None = None
        lock: _RunLock | None = None

        def local_descriptor() -> dict[str, Any]:
            self.config.validate()
            identity.validate()
            frames = int(source.num_video_frames)
            if frames < 1:
                raise ValueError("source contains no video frames")
            return {
                "config_fingerprint": self.config.fingerprint(),
                "identity": dataclasses.asdict(identity),
                "source_frames": frames,
            }

        descriptor = coordinator.run_all("startup-validation", local_descriptor)
        coordinator.require_equal("startup-contract", descriptor)
        total_frames = int(descriptor["source_frames"])

        def initialize_primary() -> _Snapshot:
            nonlocal journal, lock
            lock = _RunLock(Path(output_dir).expanduser().resolve() / "RUN.lock")
            try:
                lock.__enter__()
                journal = GenerationJournal(
                    output_dir,
                    config=self.config,
                    identity=identity,
                    total_frames=total_frames,
                    resume=resolved_resume,
                )
                return journal.snapshot()
            except BaseException:
                if lock is not None:
                    lock.__exit__(None, None, None)
                    lock = None
                raise

        try:
            snapshot = coordinator.run_primary("journal-initialize", initialize_primary)
            assert isinstance(snapshot, _Snapshot)
            if snapshot.status == "complete":
                return LongHorizonResult(
                    Path(output_dir).expanduser().resolve(),
                    snapshot.committed_frames,
                    self.config.action_steps_for_video_frames(snapshot.committed_frames),
                    snapshot.chunks,
                    snapshot.rollbacks,
                    snapshot.status,
                )
            if snapshot.status == "failed":
                raise RuntimeError(
                    "long-generation run is terminally failed; start a new output "
                    f"directory ({snapshot.failure or 'unknown failure'})"
                )

            while snapshot.committed_frames < total_frames:
                accepted = False
                window_start = snapshot.committed_frames - (
                    0 if snapshot.committed_frames == 0 else self.config.overlap_frames
                )
                remaining_attempts = coordinator.run_primary(
                    "attempt-budget",
                    lambda: journal.remaining_attempts(
                        window_start, snapshot.parent_digest
                    )
                    if journal is not None
                    else 0,
                )
                for _ in range(int(remaining_attempts)):
                    allocation = coordinator.run_primary(
                        "attempt-allocate",
                        lambda: journal.allocate_attempt(
                            window_start,
                            snapshot.parent_digest,
                        )
                        if journal is not None
                        else None,
                    )
                    attempt_id, seed = allocation
                    request = coordinator.run_all(
                        "window-prepare",
                        lambda: self._request(
                            source,
                            snapshot,
                            attempt_id=int(attempt_id),
                            seed=int(seed),
                        ),
                    )
                    coordinator.require_equal(
                        "window-shape",
                        {
                            "source_video": tuple(request.source_video.shape),
                            "source_action": tuple(request.source_action.shape),
                            "source_video_dtype": str(request.source_video.dtype),
                            "source_action_dtype": str(request.source_action.dtype),
                            "valid_video": request.valid_window_frames,
                            "valid_action": request.valid_window_action_steps,
                        },
                    )
                    output = coordinator.run_all(
                        "window-sample", lambda: self.sampler(request)
                    )

                    def candidate_descriptor() -> dict[str, Any]:
                        if not isinstance(output, ChunkOutput):
                            raise TypeError("WindowSampler must return ChunkOutput")
                        if not isinstance(output.video, torch.Tensor):
                            raise TypeError("candidate video must be a tensor")
                        if not isinstance(output.action, torch.Tensor):
                            raise TypeError("candidate action must be a tensor")
                        return {
                            "video": tuple(output.video.shape),
                            "action": tuple(output.action.shape),
                            "video_dtype": str(output.video.dtype),
                            "action_dtype": str(output.action.dtype),
                        }

                    candidate_shape = coordinator.run_all(
                        "candidate-validation", candidate_descriptor
                    )
                    coordinator.require_equal(
                        "candidate-shape",
                        candidate_shape,
                    )
                    report = coordinator.run_primary(
                        "quality-evaluation",
                        lambda: self.evaluator(request, output),
                    )
                    assert isinstance(report, QualityReport)
                    if report.accepted:
                        def commit_primary() -> _Snapshot:
                            assert journal is not None
                            journal.commit(request, output, report)
                            return journal.snapshot()

                        snapshot = coordinator.run_primary(
                            "candidate-commit", commit_primary
                        )
                        accepted = True
                        break
                    coordinator.run_primary(
                        "candidate-reject",
                        lambda: journal.record_rejected(
                            request.window_start_frame, attempt_id, report
                        )
                        if journal is not None
                        else None,
                    )

                if accepted:
                    continue

                rollback_chunks = self.config.recovery.rollback_chunks
                can_rollback = coordinator.run_primary(
                    "rollback-check",
                    lambda: bool(
                        journal is not None
                        and rollback_chunks > 0
                        and snapshot.rollbacks
                        < self.config.recovery.max_total_rollbacks
                        and journal.can_rollback(rollback_chunks)
                    ),
                )
                if not can_rollback:
                    reason = (
                        "all candidates failed and the configured rollback budget is exhausted"
                    )
                    coordinator.run_primary(
                        "mark-terminal-failure",
                        lambda: journal.mark_failed(reason)
                        if journal is not None
                        else None,
                    )
                    raise RuntimeError(reason)

                def rollback_primary() -> _Snapshot:
                    assert journal is not None
                    journal.rollback(rollback_chunks)
                    return journal.snapshot()

                snapshot = coordinator.run_primary("rollback", rollback_primary)
                assert isinstance(snapshot, _Snapshot)

            def finalize_primary() -> _Snapshot:
                assert journal is not None
                journal.finalize()
                return journal.snapshot()

            snapshot = coordinator.run_primary("finalize", finalize_primary)
            assert isinstance(snapshot, _Snapshot)
            return LongHorizonResult(
                Path(output_dir).expanduser().resolve(),
                snapshot.committed_frames,
                self.config.action_steps_for_video_frames(snapshot.committed_frames),
                snapshot.chunks,
                snapshot.rollbacks,
                snapshot.status,
            )
        finally:
            if lock is not None:
                lock.__exit__(None, None, None)


@dataclass
class LongGenerationJob:
    source: WindowSource
    sampler: WindowSampler
    identity: RunIdentity
    evaluator: QualityEvaluator | None = None
    target_action_dim: int | None = None


__all__ = [
    "ChunkOutput",
    "ChunkRequest",
    "ContinuityConfig",
    "DefaultContinuityEvaluator",
    "GenerationJournal",
    "LONG_GENERATION_FORMAT",
    "LongGenerationJob",
    "LongHorizonConfig",
    "LongHorizonGenerator",
    "LongHorizonResult",
    "QualityReport",
    "RecoveryConfig",
    "RunIdentity",
    "SamplingConfig",
    "SourceWindow",
    "TensorWindowSource",
    "WindowSampler",
    "WindowSource",
    "load_long_horizon_config",
]
