"""Multi-stage freezing policy and optimizer groups."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn


_CONTROL_MARKERS = ("control_adapter", "source_control", "cross_embodiment_adapter.source_control")
_REFERENCE_MARKERS = ("reference_", "reference_layers", "cross_embodiment_adapter.reference")
_COSMOS_BOUNDARY_MARKERS = (
    "moe_gen",
    "time_embedder",
    "vae2llm",
    "llm2vae",
    "action2llm",
    "llm2action",
    "k_norm_und_for_gen",
)


def _contains(name: str, markers: Iterable[str]) -> bool:
    return any(marker in name for marker in markers)


def configure_trainable_stage(model: nn.Module, stage: str) -> dict[str, int]:
    """Apply the documented stage policy and return parameter counts.

    Stage ``reference`` keeps the learned source-control branch trainable while
    adding reference modules.  Stage ``joint`` also enables the generator and
    action boundary modules, but always leaves the video tokenizer frozen.
    """

    if stage not in {"control", "reference", "joint"}:
        raise ValueError(f"Unknown training stage: {stage}")
    model_config = getattr(model, "config", None)
    source_config = getattr(model_config, "source_control", None)
    reference_config = getattr(model_config, "reference", None)
    source_enabled = bool(getattr(source_config, "enabled", True))
    reference_enabled = bool(getattr(reference_config, "enabled", True))

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for name, parameter in model.named_parameters():
        trainable = source_enabled and _contains(name, _CONTROL_MARKERS)
        if stage in {"reference", "joint"}:
            trainable = trainable or (
                reference_enabled and _contains(name, _REFERENCE_MARKERS)
            )
        if stage == "joint":
            is_tokenizer = name.startswith("tokenizer.") or ".tokenizer_vision_gen." in name
            if not is_tokenizer:
                # The toy backbone has no Cosmos marker; joint means all non-tokenizer
                # parameters. Cosmos stays selective to protect the reasoner tower.
                is_cosmos_model = "language_model.model.layers" in name or "moe_gen" in name
                trainable = trainable or (not is_cosmos_model) or _contains(name, _COSMOS_BOUNDARY_MARKERS)
        parameter.requires_grad_(trainable)
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in model.parameters())
    if trainable_count == 0:
        raise RuntimeError(f"Stage {stage!r} selected no trainable parameters")
    return {"trainable": trainable_count, "total": total_count}


def build_optimizer(
    model: nn.Module,
    *,
    new_module_lr: float,
    base_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    new_parameters: list[nn.Parameter] = []
    base_parameters: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if _contains(name, (*_CONTROL_MARKERS, *_REFERENCE_MARKERS)):
            new_parameters.append(parameter)
        else:
            base_parameters.append(parameter)
    groups = []
    if new_parameters:
        groups.append({"params": new_parameters, "lr": new_module_lr, "name": "adapters"})
    if base_parameters:
        groups.append({"params": base_parameters, "lr": base_lr, "name": "base"})
    if not groups:
        raise RuntimeError("No trainable optimizer parameters")
    return torch.optim.AdamW(groups, betas=(0.9, 0.95), eps=1.0e-8, weight_decay=weight_decay)


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    warmup_steps: int,
    max_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    import math

    def scale(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)
