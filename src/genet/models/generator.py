"""Runnable cross-embodiment generator core.

The lightweight backend is deliberately real rather than a shape-only mock:
it trains a rectified-flow video/action denoiser end to end and exercises the
same source-control and reference-attention modules installed into Cosmos3.
It is intended for CPU CI, one-batch overfit checks, and data-contract bringup;
production training selects the Cosmos adapter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from genet.config import ModelConfig
from genet.models.control import SourceControlAdapter
from genet.models.reference_attention import RoutedReferenceCrossAttention


@dataclass
class GeneratorOutput:
    video_velocity: torch.Tensor
    action_velocity: torch.Tensor


class CausalVideoTokenizer(nn.Module):
    """Small frozen tokenizer preserving Wan's ``1 + 4N`` time convention."""

    def __init__(self, latent_channels: int, temporal_factor: int = 4, spatial_factor: int = 16) -> None:
        super().__init__()
        self.temporal_factor = temporal_factor
        self.spatial_factor = spatial_factor
        self.channel_proj = nn.Conv3d(3, latent_channels, kernel_size=1, bias=False)
        nn.init.orthogonal_(self.channel_proj.weight.view(latent_channels, 3))
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def encode(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5 or video.shape[1] != 3:
            raise ValueError(f"video must be [B,3,T,H,W], got {tuple(video.shape)}")
        if (video.shape[2] - 1) % self.temporal_factor:
            raise ValueError(
                f"video frames must be 1 + N*{self.temporal_factor}, got {video.shape[2]}"
            )
        if video.dtype == torch.uint8:
            video = video.float().div(127.5).sub(1.0)
        else:
            video = video.float()
        video = video.to(dtype=self.channel_proj.weight.dtype)
        first = video[:, :, :1]
        tail = video[:, :, 1:]
        first = F.avg_pool3d(first, kernel_size=(1, self.spatial_factor, self.spatial_factor))
        if tail.shape[2]:
            tail = F.avg_pool3d(
                tail,
                kernel_size=(self.temporal_factor, self.spatial_factor, self.spatial_factor),
                stride=(self.temporal_factor, self.spatial_factor, self.spatial_factor),
            )
            compressed = torch.cat([first, tail], dim=2)
        else:
            compressed = first
        return self.channel_proj(compressed)


class DomainAwareProjection(nn.Module):
    """Per-embodiment affine projection, matching Cosmos' boundary semantics."""

    def __init__(self, input_dim: int, output_dim: int, num_embodiments: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embodiments, output_dim, input_dim))
        self.bias = nn.Parameter(torch.zeros(num_embodiments, output_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, values: torch.Tensor, domain_id: torch.Tensor) -> torch.Tensor:
        if values.shape[0] != domain_id.shape[0]:
            raise ValueError("domain_id must have one value per batch item")
        weight = self.weight[domain_id]
        bias = self.bias[domain_id]
        return torch.einsum("bti,boi->bto", values, weight) + bias.unsqueeze(1)


class ToyMoTBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, reference_attention: RoutedReferenceCrossAttention) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.self_attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.reference_attention = reference_attention

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        vision_tokens: int,
        reference: torch.Tensor | None,
        reference_mask: torch.Tensor | None,
        use_reference: bool = True,
    ) -> torch.Tensor:
        normalized = self.norm1(tokens)
        attended, _ = self.self_attention(normalized, normalized, normalized, need_weights=False)
        tokens = tokens + attended
        if reference is not None:
            vision_candidate = self.reference_attention(
                tokens[:, :vision_tokens], reference, route_index=0, reference_mask=reference_mask
            )
            action_candidate = self.reference_attention(
                tokens[:, vision_tokens:], reference, route_index=1, reference_mask=reference_mask
            )
            # As with source control, execute both projections on a dropped
            # condition and apply an exact zero gate.  This keeps shared/dual
            # ablations and DDP parameter usage stable across ranks.
            reference_gate = tokens.new_tensor(float(use_reference))
            vision_query = tokens[:, :vision_tokens]
            action_query = tokens[:, vision_tokens:]
            vision = vision_query + reference_gate * (vision_candidate - vision_query)
            action = action_query + reference_gate * (action_candidate - action_query)
            tokens = torch.cat([vision, action], dim=1)
        return tokens + self.mlp(self.norm2(tokens))


class CrossEmbodimentGenerator(nn.Module):
    """Joint video/action rectified-flow denoiser used by the standalone trainer."""

    def __init__(self, config: ModelConfig, action_dim: int = 64, temporal_factor: int = 4) -> None:
        super().__init__()
        self.config = config
        self.action_dim = action_dim
        self.patch_size = config.patch_size
        self.tokenizer = CausalVideoTokenizer(config.latent_channels, temporal_factor=temporal_factor)
        self.control_adapter = SourceControlAdapter(
            latent_channels=config.latent_channels,
            action_dim=action_dim,
            num_embodiments=config.num_embodiments,
            domain_embedding_dim=config.source_control.domain_embedding_dim,
            zero_init=config.source_control.zero_init,
            vision_scale=config.source_control.vision_scale,
            action_scale=config.source_control.action_scale,
        )
        patch_dim = config.latent_channels * config.patch_size**2
        self.video_in = nn.Linear(patch_dim, config.hidden_size)
        self.video_out = nn.Linear(config.hidden_size, patch_dim)
        self.action_in = DomainAwareProjection(action_dim, config.hidden_size, config.num_embodiments)
        self.action_out = DomainAwareProjection(config.hidden_size, action_dim, config.num_embodiments)
        self.reference_video_in = nn.Linear(patch_dim, config.hidden_size)
        self.reference_action_in = DomainAwareProjection(action_dim, config.hidden_size, config.num_embodiments)
        self.video_modality = nn.Parameter(torch.zeros(config.hidden_size))
        self.action_modality = nn.Parameter(torch.zeros(config.hidden_size))
        self.time_mlp = nn.Sequential(
            nn.Linear(1, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        blocks: list[ToyMoTBlock] = []
        for layer_index in range(config.num_layers):
            ref_attn = RoutedReferenceCrossAttention(
                config.hidden_size,
                config.reference.num_heads,
                projection_mode=config.reference.projection_mode,
                dropout=config.reference.dropout,
                gate_init=config.reference.gate_init,
            )
            if layer_index % config.reference.inject_every_n_layers:
                # Keep module/parameter count fixed while disabling injection at this layer.
                ref_attn.gates.requires_grad_(False)
                with torch.no_grad():
                    ref_attn.gates.fill_(-20.0)
            blocks.append(ToyMoTBlock(config.hidden_size, config.num_heads, ref_attn))
        self.blocks = nn.ModuleList(blocks)
        self.final_norm = nn.LayerNorm(config.hidden_size)

    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        return self.tokenizer.encode(video)

    def _patchify(self, latent: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int, int, int, int]]:
        batch, channels, frames, height, width = latent.shape
        patch = self.patch_size
        padded_h = math.ceil(height / patch) * patch
        padded_w = math.ceil(width / patch) * patch
        latent = F.pad(latent, (0, padded_w - width, 0, padded_h - height))
        tokens = latent.reshape(batch, channels, frames, padded_h // patch, patch, padded_w // patch, patch)
        tokens = torch.einsum("bcthpwq->bthwpqc", tokens).reshape(batch, -1, channels * patch * patch)
        return tokens, (batch, channels, frames, height, width)

    def _unpatchify(self, tokens: torch.Tensor, shape: tuple[int, int, int, int, int]) -> torch.Tensor:
        batch, channels, frames, height, width = shape
        patch = self.patch_size
        padded_h = math.ceil(height / patch) * patch
        padded_w = math.ceil(width / patch) * patch
        values = tokens.reshape(batch, frames, padded_h // patch, padded_w // patch, patch, patch, channels)
        values = torch.einsum("bthwpqc->bcthpwq", values)
        values = values.reshape(batch, channels, frames, padded_h, padded_w)
        return values[:, :, :, :height, :width]

    def _reference_tokens(
        self,
        reference_video: torch.Tensor | None,
        reference_action: torch.Tensor | None,
        reference_domain_id: torch.Tensor,
    ) -> torch.Tensor | None:
        parts: list[torch.Tensor] = []
        if self.config.reference.use_video and reference_video is not None:
            tokens, _ = self._patchify(reference_video)
            parts.append(self.reference_video_in(tokens) + self.video_modality)
        if self.config.reference.use_action and reference_action is not None:
            parts.append(self.reference_action_in(reference_action, reference_domain_id) + self.action_modality)
        return torch.cat(parts, dim=1) if parts else None

    def forward(
        self,
        *,
        noisy_target_video: torch.Tensor,
        noisy_target_action: torch.Tensor,
        sigma: torch.Tensor,
        source_video: torch.Tensor,
        source_action: torch.Tensor,
        source_domain_id: torch.Tensor,
        target_domain_id: torch.Tensor,
        reference_video: torch.Tensor | None,
        reference_action: torch.Tensor | None,
        reference_domain_id: torch.Tensor,
        target_action_mask: torch.Tensor | None = None,
        reference_mask: torch.Tensor | None = None,
        use_source: bool = True,
        use_reference: bool = True,
    ) -> GeneratorOutput:
        if self.config.source_control.enabled:
            noisy_target_video = self.control_adapter.video(noisy_target_video, source_video, enabled=use_source)
            noisy_target_action = self.control_adapter.action(
                noisy_target_action,
                source_action,
                source_domain_id,
                target_domain_id,
                output_mask=target_action_mask,
                enabled=use_source,
            )
        video_patches, video_shape = self._patchify(noisy_target_video)
        vision_tokens = self.video_in(video_patches) + self.video_modality
        action_tokens = self.action_in(noisy_target_action, target_domain_id) + self.action_modality
        timestep = self.time_mlp(sigma.float().unsqueeze(-1)).to(dtype=vision_tokens.dtype).unsqueeze(1)
        vision_tokens = vision_tokens + timestep
        action_tokens = action_tokens + timestep
        reference = None
        if self.config.reference.enabled:
            reference = self._reference_tokens(reference_video, reference_action, reference_domain_id)
        tokens = torch.cat([vision_tokens, action_tokens], dim=1)
        num_vision_tokens = vision_tokens.shape[1]
        for block in self.blocks:
            tokens = block(
                tokens,
                vision_tokens=num_vision_tokens,
                reference=reference,
                reference_mask=reference_mask,
                use_reference=use_reference,
            )
        tokens = self.final_norm(tokens)
        video_velocity = self._unpatchify(self.video_out(tokens[:, :num_vision_tokens]), video_shape)
        action_velocity = self.action_out(tokens[:, num_vision_tokens:], target_domain_id)
        return GeneratorOutput(video_velocity=video_velocity, action_velocity=action_velocity)
