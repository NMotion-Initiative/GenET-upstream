"""Reference-target cross attention and the shared-vs-dual ablation."""

from __future__ import annotations

import copy
import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

ProjectionMode = Literal["shared", "dual", "dual_tied"]
Route = Literal["vision", "action", "ar", "dm"]


class KVProjection(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.key = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.key(reference), self.value(reference)


class RoutedReferenceProjector(nn.Module):
    """The only module whose sharing changes in the main ablation.

    ``shared`` and ``dual_tied`` route both consumers through one KV projector.
    ``dual`` owns two identically initialized copies.  Query/output projections,
    gates, token counts, and injection layers live outside this class and are
    therefore identical across the comparison.
    """

    def __init__(self, hidden_size: int, mode: ProjectionMode = "shared") -> None:
        super().__init__()
        self.mode: ProjectionMode = mode
        self.route_a = KVProjection(hidden_size)
        if mode == "dual":
            self.route_b: KVProjection | None = copy.deepcopy(self.route_a)
        else:
            self.route_b = None

    def project(self, reference: torch.Tensor, route_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if route_index not in (0, 1):
            raise ValueError(f"route_index must be 0 or 1, got {route_index}")
        module = self.route_b if route_index == 1 and self.route_b is not None else self.route_a
        return module(reference)

    def copy_shared_to_dual(self) -> None:
        """Initialize a dual route from route A for fair stage transitions."""

        if self.route_b is None:
            raise ValueError("copy_shared_to_dual requires projection_mode='dual'")
        self.route_b.load_state_dict(self.route_a.state_dict())


class RoutedReferenceCrossAttention(nn.Module):
    """Cross-attend two fixed consumer routes to one reference token set."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        *,
        projection_mode: ProjectionMode = "shared",
        dropout: float = 0.0,
        gate_init: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.dropout = float(dropout)
        self.projector = RoutedReferenceProjector(hidden_size, projection_mode)
        self.query = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(2)])
        self.output = nn.ModuleList([nn.Linear(hidden_size, hidden_size, bias=False) for _ in range(2)])
        self.norm_query = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(2)])
        self.norm_reference = nn.LayerNorm(hidden_size)
        self.gates = nn.Parameter(torch.full((2,), float(gate_init)))

    def _reshape_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, length, _ = tensor.shape
        return tensor.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        reference: torch.Tensor,
        *,
        route_index: int,
        reference_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if query.numel() == 0:
            return query
        q = self.query[route_index](self.norm_query[route_index](query))
        k, v = self.projector.project(self.norm_reference(reference), route_index)
        q = self._reshape_heads(q)
        k = self._reshape_heads(k)
        v = self._reshape_heads(v)
        attn_mask = None
        if reference_mask is not None:
            valid = reference_mask.to(device=q.device, dtype=torch.bool)
            if valid.shape != reference.shape[:2]:
                raise ValueError(
                    f"reference_mask shape {tuple(valid.shape)} must equal {tuple(reference.shape[:2])}"
                )
            attn_mask = valid[:, None, None, :].expand(-1, self.num_heads, query.shape[1], -1)
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            scale=1.0 / math.sqrt(self.head_dim),
        )
        attended = attended.transpose(1, 2).reshape(query.shape)
        delta = self.output[route_index](attended)
        return query + torch.tanh(self.gates[route_index]) * delta


def migrate_reference_projectors(
    state_dict: dict[str, torch.Tensor],
    *,
    prefix: str,
    destination_mode: ProjectionMode,
) -> dict[str, torch.Tensor]:
    """Copy shared route-A tensors into missing dual route-B tensors.

    This returns a new mapping and never silently averages dual weights back
    into a shared projector.  A dual-to-shared migration should be an explicit
    experiment decision because it is not function preserving.
    """

    migrated = dict(state_dict)
    if destination_mode != "dual":
        return migrated
    marker = f"{prefix}projector.route_a."
    for key, value in list(state_dict.items()):
        if key.startswith(marker):
            suffix = key[len(marker) :]
            route_b_key = f"{prefix}projector.route_b.{suffix}"
            migrated.setdefault(route_b_key, value.clone())
    return migrated

