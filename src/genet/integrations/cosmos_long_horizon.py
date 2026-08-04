"""Cosmos3-Edge single-window bridge for long video/action continuation.

The rolling state machine is framework-independent.  This module only converts
one :class:`ChunkRequest` into the exact one-sample batch layout consumed by the
pinned ``OmniMoTModel`` API, then decodes the returned Wan latent.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import torch

from genet.inference.long_horizon import ChunkOutput, ChunkRequest
from genet.integrations.cosmos_data import make_cosmos_sequence_plan


def _single(value: Any, *, key: str) -> Any:
    while isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"template {key} must describe exactly one sample")
        value = value[0]
    if value is None:
        raise KeyError(f"template is missing required {key}")
    return value


def _tensor(value: Any, *, key: str) -> torch.Tensor:
    value = _single(value, key=key)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"template {key} must be a tensor")
    return value.detach().clone()


def _scalar_int(value: Any, *, key: str) -> int:
    tensor = _tensor(value, key=key)
    if tensor.numel() != 1:
        raise ValueError(f"template {key} must be scalar")
    return int(tensor.item())


def _normalized_video(video: torch.Tensor) -> torch.Tensor:
    value = video.detach().to(device="cpu")
    if value.dtype == torch.uint8:
        return value.to(torch.float32).div(127.5).sub(1.0)
    if not torch.is_floating_point(value):
        raise TypeError(f"video must be uint8 or floating point, got {value.dtype}")
    value = value.to(torch.float32)
    if not bool(torch.isfinite(value).all()):
        raise ValueError("video contains NaN or infinity")
    if value.numel() and (float(value.min()) < -1.0001 or float(value.max()) > 1.0001):
        raise ValueError("floating video must be normalized to [-1,1]")
    return value


def _model_action(
    action: torch.Tensor,
    *,
    model_dim: int,
    raw_dim: int,
    processing_record: Any | None = None,
) -> torch.Tensor:
    if action.ndim != 2:
        raise ValueError("action must have shape [T,D]")
    if action.shape[1] != raw_dim:
        if action.shape[1] == model_dim and processing_record is None:
            return action.to(torch.float32)
        raise ValueError(
            f"external action width {action.shape[1]} does not match raw width {raw_dim}"
        )
    value = action.to(torch.float32)
    normalizer = getattr(processing_record, "action_normalizer", None)
    if normalizer is not None:
        value = normalizer.normalize_action(value)
    if value.shape[1] > model_dim:
        raise ValueError(f"action width {value.shape[1]} exceeds model width {model_dim}")
    if value.shape[1] < model_dim:
        value = torch.cat(
            [value, value.new_zeros(value.shape[0], model_dim - value.shape[1])],
            dim=1,
        )
    return value


def _context_target_video(request: ChunkRequest) -> torch.Tensor:
    source = _normalized_video(request.source_video)
    target = torch.zeros_like(source)
    if request.context_video is not None:
        context = _normalized_video(request.context_video)
        if context.shape[0] != target.shape[0] or context.shape[2:] != target.shape[2:]:
            raise ValueError("target context video geometry differs from source window")
        target[:, : request.context_frames] = context
        if request.context_frames < target.shape[1]:
            target[:, request.context_frames :] = context[:, -1:].expand(
                -1, target.shape[1] - request.context_frames, -1, -1
            )
    return target


def build_long_horizon_cosmos_batch(
    request: ChunkRequest,
    template_sample: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one already-batched Cosmos sample without mutating the template."""

    target_template_action = _tensor(template_sample.get("action"), key="action")
    source_template_action = _tensor(
        template_sample.get("source_action"), key="source_action"
    )
    target_model_dim = int(target_template_action.shape[-1])
    source_model_dim = int(source_template_action.shape[-1])
    target_raw_dim = _scalar_int(template_sample.get("raw_action_dim"), key="raw_action_dim")
    source_raw_dim = _scalar_int(
        template_sample.get("source_raw_action_dim"), key="source_raw_action_dim"
    )
    processing_record = _single(
        template_sample.get("action_processing_record"), key="action_processing_record"
    )

    target_action = torch.zeros(
        request.config.chunk_action_steps,
        target_model_dim,
        dtype=torch.float32,
    )
    if request.context_action is not None and request.context_action_steps > 0:
        model_context = _model_action(
            request.context_action,
            model_dim=target_model_dim,
            raw_dim=target_raw_dim,
            processing_record=processing_record,
        )
        target_action[: request.context_action_steps] = model_context
        if request.context_action_steps < target_action.shape[0]:
            target_action[request.context_action_steps :] = model_context[-1:].expand(
                target_action.shape[0] - request.context_action_steps, -1
            )

    source_action = _model_action(
        request.source_action,
        model_dim=source_model_dim,
        raw_dim=source_raw_dim,
    )
    if source_action.shape[0] != request.config.chunk_action_steps:
        raise ValueError("source action window length differs from configured chunk")

    target_video = _context_target_video(request).unsqueeze(0)
    source_video = request.source_video.detach().clone()
    reference_video = _tensor(
        template_sample.get("reference_video"), key="reference_video"
    )
    reference_action = _tensor(
        template_sample.get("reference_action"), key="reference_action"
    )
    caption = _single(template_sample.get("ai_caption", ""), key="ai_caption")
    if not isinstance(caption, str):
        raise TypeError("template ai_caption must be a string")

    plan = make_cosmos_sequence_plan(
        action_start_frame_offset=0 if request.config.action_alignment == "frame" else 1,
        condition_frame_indexes_vision=request.condition_video_latent_indexes,
        condition_frame_indexes_action=request.condition_action_indexes,
    )
    batch: dict[str, Any] = {
        "dataset_name": "genet_long_horizon",
        "video": [target_video],
        "action": [target_action],
        "domain_id": [_tensor(template_sample.get("domain_id"), key="domain_id")],
        "sequence_plan": [plan],
        "ai_caption": [caption],
        "image_size": [_tensor(template_sample.get("image_size"), key="image_size")],
        "conditioning_fps": [
            _tensor(template_sample.get("conditioning_fps"), key="conditioning_fps")
        ],
        "raw_action_dim": [torch.tensor(target_raw_dim, dtype=torch.long)],
        "action_processing_record": [processing_record],
        "is_preprocessed": True,
        "source_video": [source_video],
        "source_action": [source_action],
        "source_domain_id": [
            _tensor(template_sample.get("source_domain_id"), key="source_domain_id")
        ],
        "source_image_size": [
            _tensor(template_sample.get("source_image_size"), key="source_image_size")
        ],
        "source_conditioning_fps": [
            _tensor(
                template_sample.get("source_conditioning_fps"),
                key="source_conditioning_fps",
            )
        ],
        "source_raw_action_dim": [torch.tensor(source_raw_dim, dtype=torch.long)],
        "reference_video": [reference_video],
        "reference_action": [reference_action],
        "reference_domain_id": [
            _tensor(template_sample.get("reference_domain_id"), key="reference_domain_id")
        ],
        "reference_image_size": [
            _tensor(template_sample.get("reference_image_size"), key="reference_image_size")
        ],
        "reference_conditioning_fps": [
            _tensor(
                template_sample.get("reference_conditioning_fps"),
                key="reference_conditioning_fps",
            )
        ],
        "reference_raw_action_dim": [
            _tensor(
                template_sample.get("reference_raw_action_dim"),
                key="reference_raw_action_dim",
            )
        ],
    }
    negative = template_sample.get("neg_ai_caption")
    if negative is not None:
        negative = _single(negative, key="neg_ai_caption")
        if not isinstance(negative, str):
            raise TypeError("template neg_ai_caption must be a string")
        batch["neg_ai_caption"] = [negative]
    return batch


def _place_preprocessed_videos(batch: dict[str, Any], model: Any) -> None:
    """Move floating video inputs because upstream only moves uint8 inputs."""

    tensor_kwargs = getattr(model, "tensor_kwargs", None)
    if not isinstance(tensor_kwargs, Mapping):
        return
    device = tensor_kwargs.get("device")
    dtype = tensor_kwargs.get("dtype")
    if device is None:
        return
    if not isinstance(dtype, torch.dtype):
        dtype = torch.bfloat16 if torch.device(device).type == "cuda" else torch.float32
    for key in ("video", "source_video", "reference_video"):
        values = batch.get(key)
        if not isinstance(values, list):
            continue
        batch[key] = [
            value.to(device=device, dtype=dtype)
            if isinstance(value, torch.Tensor) and torch.is_floating_point(value)
            else value
            for value in values
        ]


class CosmosLongHorizonSampler:
    """Invoke the public Cosmos sampler once and return decoded canonical tensors."""

    def __init__(
        self,
        model: Any,
        template_sample: Mapping[str, Any],
        *,
        use_source_condition: bool = True,
        use_reference_condition: bool = True,
    ) -> None:
        self.model = model
        self.template_sample = copy.copy(dict(template_sample))
        self.use_source_condition = bool(use_source_condition)
        self.use_reference_condition = bool(use_reference_condition)
        self.target_action_dim = _scalar_int(
            self.template_sample.get("raw_action_dim"), key="raw_action_dim"
        )

    def __call__(self, request: ChunkRequest) -> ChunkOutput:
        batch = build_long_horizon_cosmos_batch(request, self.template_sample)
        if (
            request.config.sampling.has_negative_prompt
            and "neg_ai_caption" not in batch
        ):
            raise ValueError(
                "sampling.has_negative_prompt=true requires template neg_ai_caption"
            )
        _place_preprocessed_videos(batch, self.model)
        fixed_sampler = getattr(self.model, "fixed_step_sampler", None)
        sampler = fixed_sampler if fixed_sampler is not None else None
        guidance = 1.0 if fixed_sampler is not None else request.config.sampling.guidance
        outputs = self.model.generate_samples_from_batch(
            batch,
            sampler=sampler,
            guidance=guidance,
            seed=[request.seed],
            n_sample=1,
            has_negative_prompt=request.config.sampling.has_negative_prompt,
            num_steps=request.config.sampling.num_steps,
            shift=request.config.sampling.shift,
            sigma_max=request.config.sampling.sigma_max,
            normalize_cfg=request.config.sampling.normalize_cfg,
            use_source_condition=self.use_source_condition,
            use_reference_condition=self.use_reference_condition,
        )
        vision_values = outputs.get("vision")
        action_values = outputs.get("action")
        if not isinstance(vision_values, list) or len(vision_values) != 1:
            raise RuntimeError("Cosmos long-horizon sampling expected one vision latent")
        if not isinstance(action_values, list) or len(action_values) != 1:
            raise RuntimeError("Cosmos long-horizon sampling expected one action trajectory")
        decoded = self.model.decode(vision_values[0])
        if isinstance(decoded, (list, tuple)):
            if len(decoded) != 1:
                raise RuntimeError("Cosmos decoder returned multiple videos for one chunk")
            decoded = decoded[0]
        if not isinstance(decoded, torch.Tensor):
            raise TypeError("Cosmos decoder did not return a tensor")
        if decoded.ndim == 5 and decoded.shape[0] == 1:
            decoded = decoded.squeeze(0)
        if decoded.ndim != 4:
            raise ValueError(f"decoded video must be [C,T,H,W], got {tuple(decoded.shape)}")
        action = action_values[0]
        if not isinstance(action, torch.Tensor):
            raise TypeError("Cosmos generated action is not a tensor")
        return ChunkOutput(
            video=decoded.detach().clamp(-1, 1).to(device="cpu", dtype=torch.float32),
            action=action.detach().to(device="cpu", dtype=torch.float32),
            metadata={"seed": request.seed},
        )


__all__ = [
    "CosmosLongHorizonSampler",
    "build_long_horizon_cosmos_batch",
]
