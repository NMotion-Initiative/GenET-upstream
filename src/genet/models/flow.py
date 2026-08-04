"""Rectified-flow utilities shared by the toy and Cosmos backends."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FlowBatch:
    noisy: torch.Tensor
    target_velocity: torch.Tensor
    sigma: torch.Tensor
    noise: torch.Tensor


def sample_logit_normal_sigma(
    batch_size: int,
    *,
    device: torch.device | str,
    generator: torch.Generator | None = None,
    mean: float = 0.0,
    std: float = 1.0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sample RF interpolation levels in ``(0, 1)``."""

    logits = torch.randn(batch_size, device=device, generator=generator, dtype=dtype)
    return torch.sigmoid(logits.mul(std).add(mean))


def interpolate_rectified_flow(
    clean: torch.Tensor,
    sigma: torch.Tensor,
    *,
    noise: torch.Tensor | None = None,
    generator: torch.Generator | None = None,
) -> FlowBatch:
    """Build ``x_t=(1-sigma)x_0+sigma*epsilon`` and target ``epsilon-x_0``.

    ``sigma`` is one scalar per batch item and is broadcast over all remaining
    dimensions.  Keeping this helper modality-agnostic guarantees video and
    action use exactly the same sampled time when the config requests a shared
    schedule.
    """

    if clean.ndim < 2:
        raise ValueError(f"clean must have a batch dimension, got {tuple(clean.shape)}")
    if sigma.shape != (clean.shape[0],):
        raise ValueError(f"sigma must have shape ({clean.shape[0]},), got {tuple(sigma.shape)}")
    if noise is None:
        noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
    if noise.shape != clean.shape:
        raise ValueError(f"noise shape {tuple(noise.shape)} != clean shape {tuple(clean.shape)}")
    view = (clean.shape[0],) + (1,) * (clean.ndim - 1)
    sigma_view = sigma.to(device=clean.device, dtype=clean.dtype).view(view)
    noisy = (1.0 - sigma_view) * clean + sigma_view * noise
    return FlowBatch(noisy=noisy, target_velocity=noise - clean, sigma=sigma, noise=noise)


def masked_token_mean(loss: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Mean a per-element loss over valid tokens, not padded channels."""

    if mask is None:
        return loss.mean()
    expanded = mask.to(device=loss.device, dtype=loss.dtype)
    while expanded.ndim < loss.ndim:
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.expand_as(loss)
    denominator = expanded.sum().clamp_min(1.0)
    return (loss * expanded).sum() / denominator

