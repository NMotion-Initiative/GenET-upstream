"""Zero-initialized source-control adapters.

The design follows the safe initialization invariant used by ControlNet/VACE:
at construction the adapter is an exact no-op, so loading it beside a frozen
pretrained generator cannot perturb the base model before the first update.
"""

from __future__ import annotations

import torch
from torch import nn


class SourceVideoControl(nn.Module):
    """Project aligned source latents into the target noisy-latent stream."""

    def __init__(self, channels: int, *, zero_init: bool = True, scale: float = 1.0) -> None:
        super().__init__()
        self.in_norm = nn.GroupNorm(num_groups=1, num_channels=channels)
        self.act = nn.SiLU()
        self.proj = nn.Conv3d(channels, channels, kernel_size=1)
        self.scale = float(scale)
        if zero_init:
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

    def forward(self, noisy_target: torch.Tensor, source: torch.Tensor, enabled: bool = True) -> torch.Tensor:
        if noisy_target.shape != source.shape:
            raise ValueError(
                "Source and target video latents must be aligned before control injection: "
                f"{tuple(source.shape)} != {tuple(noisy_target.shape)}"
            )
        # Cosmos stores each packed sample as [C,T,H,W], while the standalone
        # trainer uses [B,C,T,H,W]. Conv3d/GroupNorm require the latter.
        squeeze_batch = source.ndim == 4
        if squeeze_batch:
            source_for_control = source.unsqueeze(0)
        elif source.ndim == 5:
            source_for_control = source
        else:
            raise ValueError(
                "video latents must have shape [C,T,H,W] or [B,C,T,H,W], "
                f"got {tuple(source.shape)}"
            )
        source_for_control = source_for_control.to(dtype=self.proj.weight.dtype)
        residual = self.proj(self.act(self.in_norm(source_for_control)))
        if squeeze_batch:
            residual = residual.squeeze(0)
        residual = residual.to(dtype=noisy_target.dtype)
        # Keep the adapter in the autograd/FSDP graph even for a dropped CFG
        # condition.  Multiplication by zero is functionally an exact bypass,
        # but avoids a rank occasionally having no trainable path at all.
        condition_gate = residual.new_tensor(float(enabled))
        return noisy_target + condition_gate * self.scale * residual


class SourceActionControl(nn.Module):
    """Map source actions into the padded target action space with domain cues."""

    def __init__(
        self,
        action_dim: int,
        num_embodiments: int,
        domain_embedding_dim: int = 32,
        *,
        zero_init: bool = True,
        scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.domain_embedding = nn.Embedding(num_embodiments, domain_embedding_dim)
        self.in_proj = nn.Linear(action_dim + 2 * domain_embedding_dim, action_dim * 2)
        self.out_proj = nn.Linear(action_dim * 2, action_dim)
        self.act = nn.SiLU()
        self.scale = float(scale)
        if zero_init:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        noisy_target: torch.Tensor,
        source: torch.Tensor,
        source_domain_id: torch.Tensor,
        target_domain_id: torch.Tensor,
        *,
        output_mask: torch.Tensor | None = None,
        enabled: bool = True,
    ) -> torch.Tensor:
        if noisy_target.shape != source.shape:
            raise ValueError(
                "Source and target actions must have equal padded shapes: "
                f"{tuple(source.shape)} != {tuple(noisy_target.shape)}"
            )
        batch, steps, _ = source.shape
        source = source.to(dtype=self.in_proj.weight.dtype)
        src_domain = self.domain_embedding(source_domain_id).unsqueeze(1).expand(batch, steps, -1)
        tgt_domain = self.domain_embedding(target_domain_id).unsqueeze(1).expand(batch, steps, -1)
        residual = self.out_proj(self.act(self.in_proj(torch.cat([source, src_domain, tgt_domain], dim=-1))))
        residual = residual.to(dtype=noisy_target.dtype)
        if output_mask is not None:
            mask = output_mask.to(device=residual.device, dtype=residual.dtype)
            if mask.ndim == 2:
                mask = mask.unsqueeze(-1)
            residual = residual * mask
        condition_gate = residual.new_tensor(float(enabled))
        return noisy_target + condition_gate * self.scale * residual


class SourceControlAdapter(nn.Module):
    """Synchronized vision/action source conditioning."""

    def __init__(
        self,
        latent_channels: int,
        action_dim: int,
        num_embodiments: int,
        domain_embedding_dim: int = 32,
        *,
        zero_init: bool = True,
        vision_scale: float = 1.0,
        action_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.video = SourceVideoControl(latent_channels, zero_init=zero_init, scale=vision_scale)
        self.action = SourceActionControl(
            action_dim,
            num_embodiments,
            domain_embedding_dim,
            zero_init=zero_init,
            scale=action_scale,
        )
