"""Optional adapter that installs GenET conditioning into Cosmos3-Edge.

This module contains no copied Cosmos source.  It imports the pinned upstream
package at runtime and attaches registered PyTorch modules plus decoder-layer
hooks.  Base parameter names stay unchanged, which allows loading an official
Cosmos3-Edge DCP with non-strict handling only for the new adapter keys.

Current production constraints are explicit:

* one packed sample per rank (the official vision/action recipes already use 1),
* two-way attention and context-parallel degree 1,
* Wan VAE and reference encoder frozen during the initial stages.
"""

from __future__ import annotations

import contextlib
import math
import types
from dataclasses import dataclass
from typing import Any, Iterator

import torch
from torch import nn

from genet.config import ModelConfig, ReferenceConfig, SourceControlConfig
from genet.models.control import SourceControlAdapter
from genet.models.reference_attention import RoutedReferenceCrossAttention

_COSMOS_IMPORT_ERROR: ImportError | None = None

try:  # The standalone data/model tests do not install the full Cosmos stack.
    from cosmos_framework.model.generator.omni_mot_model import OmniMoTModel as _OmniMoTModel
    from cosmos_framework.data.generator.sequence_packing.runtime import (
        from_und_gen_splits,
        get_gen_seq,
        get_und_seq,
    )

    COSMOS_AVAILABLE = True
except ImportError as exc:  # pragma: no cover - exercised on lightweight developer machines
    _COSMOS_IMPORT_ERROR = exc
    _OmniMoTModel = nn.Module  # type: ignore[assignment,misc]
    COSMOS_AVAILABLE = False


def _require_cosmos() -> None:
    if not COSMOS_AVAILABLE:
        detail = ""
        if _COSMOS_IMPORT_ERROR is not None:
            missing = getattr(_COSMOS_IMPORT_ERROR, "name", None)
            detail = (
                f" Missing Python module {missing!r}."
                if missing
                else f" Import failed with: {_COSMOS_IMPORT_ERROR}."
            )
        raise RuntimeError(
            "Cosmos backend is not installed. Run scripts/bootstrap_cosmos.sh and install the pinned "
            "Cosmos Framework environment, or use model.backend=toy for smoke tests."
            + detail
        ) from _COSMOS_IMPORT_ERROR


def model_config_from_mapping(raw: dict[str, Any]) -> ModelConfig:
    """Construct the adapter subset of :class:`ModelConfig` from Hydra data."""

    values = dict(raw)
    source = values.pop("source_control", {})
    reference = values.pop("reference", {})
    # Parallel topology is owned by the upstream OmniMoT config, not this adapter.
    values.pop("parallelism", None)
    allowed = {field.name for field in __import__("dataclasses").fields(ModelConfig)}
    values = {key: value for key, value in values.items() if key in allowed}
    values["source_control"] = SourceControlConfig(**source)
    values["reference"] = ReferenceConfig(**reference)
    return ModelConfig(**values)


@dataclass
class CosmosConditionState:
    source_vision: list[torch.Tensor]
    source_action: list[torch.Tensor] | None
    source_domain_id: list[torch.Tensor] | None
    target_raw_action_dim: list[torch.Tensor] | None
    reference_vision: list[torch.Tensor]
    reference_action: list[torch.Tensor] | None
    reference_domain_id: list[torch.Tensor] | None
    use_source: bool = True
    use_reference: bool = True
    vision_token_count: int = 0
    action_token_count: int = 0
    reference_tokens: torch.Tensor | None = None


def _axis_sinusoidal_encoding(
    coordinates: torch.Tensor,
    width: int,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Encode one coordinate axis without adding checkpoint parameters."""

    if width <= 0:
        return torch.empty(
            (coordinates.numel(), 0),
            device=coordinates.device,
            dtype=dtype,
        )
    frequency_count = (width + 1) // 2
    denominator = max(frequency_count - 1, 1)
    exponent = torch.arange(
        frequency_count,
        device=coordinates.device,
        dtype=torch.float32,
    )
    inverse_frequency = torch.exp(-math.log(10_000.0) * exponent / denominator)
    angles = coordinates.to(torch.float32).reshape(-1, 1) * inverse_frequency.reshape(1, -1)
    encoded = torch.stack((angles.sin(), angles.cos()), dim=-1).flatten(1)
    return encoded[:, :width].to(dtype=dtype)


def _reference_coordinate_encoding(
    time: torch.Tensor,
    vertical: torch.Tensor,
    horizontal: torch.Tensor,
    *,
    hidden_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    if not (time.shape == vertical.shape == horizontal.shape):
        raise ValueError("reference coordinate tensors must have identical shapes")
    widths = [hidden_size // 3] * 3
    for index in range(hidden_size % 3):
        widths[index] += 1
    return torch.cat(
        [
            _axis_sinusoidal_encoding(axis, width, dtype=dtype)
            for axis, width in zip((time, vertical, horizontal), widths, strict=True)
        ],
        dim=-1,
    )


def _reference_video_position_encoding(
    shapes: list[tuple[int, int, int]],
    *,
    patch_size: int,
    hidden_size: int,
    temporal_extent: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Match Cosmos patch order (t, h, w) with fixed 3-D sinusoidal positions."""

    if patch_size <= 0:
        raise ValueError("latent patch size must be positive")
    chunks: list[torch.Tensor] = []
    for temporal, height, width in shapes:
        height_patches = (height + patch_size - 1) // patch_size
        width_patches = (width + patch_size - 1) // patch_size
        t_axis = torch.linspace(0.0, temporal_extent, temporal, device=device)
        y_axis = torch.arange(height_patches, device=device, dtype=torch.float32)
        x_axis = torch.arange(width_patches, device=device, dtype=torch.float32)
        time, vertical, horizontal = torch.meshgrid(
            t_axis,
            y_axis,
            x_axis,
            indexing="ij",
        )
        chunks.append(
            _reference_coordinate_encoding(
                time.reshape(-1),
                vertical.reshape(-1),
                horizontal.reshape(-1),
                hidden_size=hidden_size,
                dtype=dtype,
            )
        )
    return torch.cat(chunks, dim=0)


def _reference_action_position_encoding(
    length: int,
    *,
    hidden_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    time = torch.arange(length, device=device, dtype=torch.float32)
    spatial = torch.zeros_like(time)
    return _reference_coordinate_encoding(
        time,
        spatial,
        spatial,
        hidden_size=hidden_size,
        dtype=dtype,
    )


class CosmosAdapterController(nn.Module):
    """Own new parameters and inject their residuals into an instantiated VFM net."""

    def __init__(self, net: nn.Module, config: ModelConfig, action_dim: int = 64) -> None:
        super().__init__()
        _require_cosmos()
        if config.reference.routing not in {"vision_action", "ar_dm"}:
            raise ValueError(f"Unsupported Cosmos reference routing: {config.reference.routing}")
        self.config = config
        self.net_ref = [net]  # non-registered reference; net owns this controller
        latent_channels = int(getattr(net, "latent_channel", config.latent_channels))
        native_action_dim = int(getattr(net, "action_dim", action_dim))
        native_num_domains = int(
            getattr(net, "num_embodiment_domains", config.num_embodiments)
        )
        self.source_control = SourceControlAdapter(
            latent_channels,
            native_action_dim,
            native_num_domains,
            config.source_control.domain_embedding_dim,
            zero_init=config.source_control.zero_init,
            vision_scale=config.source_control.vision_scale,
            action_scale=config.source_control.action_scale,
        )
        language_layers = net.language_model.model.layers
        hidden_size = int(net.hidden_size)
        self.reference_layers = nn.ModuleList(
            [
                RoutedReferenceCrossAttention(
                    hidden_size,
                    config.reference.num_heads,
                    projection_mode=config.reference.projection_mode,
                    dropout=config.reference.dropout,
                    gate_init=config.reference.gate_init,
                )
                for _ in language_layers
            ]
            if config.reference.enabled
            else []
        )
        self._state: CosmosConditionState | None = None
        self._condition_active = False
        self._hook_handles = []
        for index, layer in enumerate(language_layers):
            self._hook_handles.append(layer.register_forward_hook(self._make_layer_hook(index)))

    def reset_parameters(self) -> None:
        """Materialize deterministic adapter initialization after Cosmos' meta build.

        Cosmos constructs the VFM network on the meta device, attaches FSDP, then
        calls ``to_empty`` followed by its own targeted ``init_weights`` method.
        The upstream initializer cannot know about these extra modules, so merely
        relying on their constructors would leave uninitialized storage after
        materialization.  Reset every leaf module here, then re-apply the two
        experiment invariants: zero control outputs and identically initialized
        dual reference projections.
        """

        for name, module in self.named_modules():
            if not name or "route_b" in name.split("."):
                continue
            reset = getattr(module, "reset_parameters", None)
            if callable(reset):
                reset()

        if self.config.source_control.zero_init:
            nn.init.zeros_(self.source_control.video.proj.weight)
            nn.init.zeros_(self.source_control.video.proj.bias)
            nn.init.zeros_(self.source_control.action.out_proj.weight)
            nn.init.zeros_(self.source_control.action.out_proj.bias)

        for layer in self.reference_layers:
            nn.init.constant_(layer.gates, float(self.config.reference.gate_init))
            if layer.projector.route_b is not None:
                layer.projector.route_b.load_state_dict(layer.projector.route_a.state_dict())

    @property
    def net(self) -> nn.Module:
        return self.net_ref[0]

    def _make_layer_hook(self, layer_index: int):
        def hook(_module: nn.Module, _inputs: tuple[Any, ...], output: tuple[Any, ...]):
            state = self._state
            if (
                state is None
                or not self.config.reference.enabled
                or state.reference_tokens is None
                or layer_index % self.config.reference.inject_every_n_layers
            ):
                return output
            hidden_pack, metadata, kv_to_store = output
            und = get_und_seq(hidden_pack)
            gen = get_gen_seq(hidden_pack)
            ref = state.reference_tokens
            adapter = self.reference_layers[layer_index]

            def apply_reference(query: torch.Tensor, route_index: int) -> torch.Tensor:
                candidate = adapter(query.unsqueeze(0), ref, route_index=route_index).squeeze(0)
                # Execute both branches even under CFG dropout so FSDP observes
                # every adapter parameter on every step.  A zero scalar keeps
                # the dropped condition exactly function-neutral with zero grads.
                gate = query.new_tensor(float(state.use_reference))
                return query + gate * (candidate - query)

            if self.config.reference.routing == "ar_dm":
                valid_und = int(hidden_pack.get("_num_causal_tokens", und.shape[0]))
                valid_gen = int(hidden_pack.get("_num_full_tokens", gen.shape[0]))
                und_valid = apply_reference(und[:valid_und], route_index=0)
                gen_valid = apply_reference(gen[:valid_gen], route_index=1)
                und = torch.cat([und_valid, und[valid_und:]], dim=0)
                gen = torch.cat([gen_valid, gen[valid_gen:]], dim=0)
            else:
                vision_end = state.vision_token_count
                action_end = vision_end + state.action_token_count
                valid_gen = int(hidden_pack.get("_num_full_tokens", gen.shape[0]))
                if action_end > valid_gen:
                    raise RuntimeError(
                        f"Adapter token ranges ({vision_end}+{state.action_token_count}) exceed valid gen tokens "
                        f"({valid_gen}); keep batch_size=1, two-way attention, and CP=1."
                    )
                vision = apply_reference(gen[:vision_end], route_index=0)
                action = apply_reference(gen[vision_end:action_end], route_index=1)
                gen = torch.cat([vision, action, gen[action_end:]], dim=0)
            return from_und_gen_splits(und, gen, hidden_pack), metadata, kv_to_store

        return hook

    def _build_reference_tokens(self, state: CosmosConditionState) -> torch.Tensor | None:
        if not self.config.reference.enabled:
            return None
        if len(state.reference_vision) != 1:
            raise ValueError("Cosmos adapter currently requires one packed sample per rank")
        parts: list[torch.Tensor] = []
        action_length = (
            int(state.reference_action[0].shape[0])
            if state.reference_action
            else None
        )
        if self.config.reference.use_video:
            shapes: list[tuple[int, int, int]] = []
            for latent in state.reference_vision:
                if latent.ndim not in (4, 5):
                    raise ValueError(
                        "reference vision latents must be [C,T,H,W] or [B,C,T,H,W], "
                        f"got {tuple(latent.shape)}"
                    )
                shapes.append(tuple(int(value) for value in latent.shape[-3:]))
            packed, _ = self.net.patchify_and_pack_latents(state.reference_vision, shapes)
            vision_tokens = self.net.vae2llm(
                packed.to(dtype=self.net.vae2llm.weight.dtype)
            )
            temporal_extent = float(
                max(action_length - 1, 0)
                if action_length is not None
                else max(shapes[0][0] - 1, 0)
            )
            vision_positions = _reference_video_position_encoding(
                shapes,
                patch_size=int(getattr(self.net, "latent_patch_size", 1)),
                hidden_size=vision_tokens.shape[-1],
                temporal_extent=temporal_extent,
                device=vision_tokens.device,
                dtype=vision_tokens.dtype,
            )
            if vision_positions.shape != vision_tokens.shape:
                raise RuntimeError(
                    "reference video position shape does not match Cosmos patch tokens: "
                    f"{tuple(vision_positions.shape)} != {tuple(vision_tokens.shape)}"
                )
            parts.append(vision_tokens + vision_positions)
        if self.config.reference.use_action and state.reference_action:
            if not hasattr(self.net, "action2llm"):
                raise RuntimeError("Reference actions requested but Cosmos net was built with action_gen=False")
            domains = state.reference_domain_id
            if not domains:
                raise ValueError("reference_domain_id is required with reference actions")
            action_shapes = [(int(action.shape[0]),) for action in state.reference_action]
            packed_action, per_token_domain = self.net.pack_action(
                state.reference_action,
                action_shapes,
                domains,
            )
            action_dtype = next(self.net.action2llm.parameters()).dtype
            packed_action = packed_action.to(dtype=action_dtype)
            action_tokens = self.net.action2llm(packed_action, per_token_domain)
            action_tokens = action_tokens + self.net.action_modality_embed.view(1, -1)
            action_positions = _reference_action_position_encoding(
                action_tokens.shape[0],
                hidden_size=action_tokens.shape[-1],
                device=action_tokens.device,
                dtype=action_tokens.dtype,
            )
            parts.append(action_tokens + action_positions)
        return torch.cat(parts, dim=0).unsqueeze(0) if parts else None

    @contextlib.contextmanager
    def condition(self, packed_seq: Any, state: CosmosConditionState) -> Iterator[None]:
        """Temporarily inject source residuals and expose reference KV to hooks."""

        if self._condition_active:
            raise RuntimeError("Nested Cosmos adapter contexts are not supported")
        # Full activation checkpointing replays decoder layers during backward,
        # after the root forward method has returned.  Keep the previous step's
        # reference state alive through that replay and release it only when the
        # next forward begins.
        if self._state is not None:
            self._state.reference_tokens = None
            self._state = None
        if getattr(self.net, "video_temporal_causal", False):
            raise ValueError("GenET Cosmos adapter requires video_temporal_causal=False")
        if packed_seq.vision is None or not packed_seq.vision.tokens:
            raise ValueError("Target packed sequence has no vision tokens")
        original_vision = packed_seq.vision.tokens
        original_action = packed_seq.action.tokens if packed_seq.action is not None else None
        state.vision_token_count = len(packed_seq.vision.sequence_indexes)
        state.action_token_count = (
            len(packed_seq.action.sequence_indexes) if packed_seq.action is not None else 0
        )
        state.reference_tokens = self._build_reference_tokens(state)
        failed = True
        try:
            if self.config.source_control.enabled:
                if len(original_vision) != len(state.source_vision):
                    raise ValueError("source/target packed vision item counts differ")
                packed_seq.vision.tokens = [
                    target
                    + target.new_tensor(float(state.use_source))
                    * (self.source_control.video(target, source) - target)
                    for target, source in zip(original_vision, state.source_vision, strict=True)
                ]
                if original_action is not None:
                    if not state.source_action or not state.source_domain_id:
                        raise ValueError("source action/domain conditions are required for an action target")
                    target_domains = packed_seq.action.domain_id
                    target_raw_dims = state.target_raw_action_dim
                    if target_raw_dims is None:
                        target_raw_dims = [
                            torch.tensor(target.shape[-1], device=target.device)
                            for target in original_action
                        ]
                    packed_seq.action.tokens = [
                        target
                        + target.new_tensor(float(state.use_source))
                        * (
                            self.source_control.action(
                                target.unsqueeze(0),
                                source.unsqueeze(0),
                                source_domain.reshape(1),
                                target_domain.reshape(1),
                                output_mask=(
                                    torch.arange(target.shape[-1], device=target.device)
                                    < target_raw_dim.to(device=target.device)
                                ).view(1, 1, -1),
                            ).squeeze(0)
                            - target
                        )
                        for target, source, source_domain, target_domain, target_raw_dim in zip(
                            original_action,
                            state.source_action,
                            state.source_domain_id,
                            target_domains,
                            target_raw_dims,
                            strict=True,
                        )
                    ]
            self._state = state
            self._condition_active = True
            yield
            failed = False
        finally:
            self._condition_active = False
            if failed:
                self._state = None
                state.reference_tokens = None
            packed_seq.vision.tokens = original_vision
            if packed_seq.action is not None:
                packed_seq.action.tokens = original_action

    def clear_condition_state(self) -> None:
        """Release reference activations after training replay or sampling."""

        if self._state is not None:
            self._state.reference_tokens = None
        self._state = None
        self._condition_active = False


def install_cosmos_adapter(net: nn.Module, config: ModelConfig, action_dim: int = 64) -> CosmosAdapterController:
    """Attach adapters without nesting the base net or renaming base weights."""

    _require_cosmos()
    if hasattr(net, "cross_embodiment_adapter"):
        raise ValueError("A cross-embodiment adapter is already installed")

    # ``OmniMoTModel.build_net`` leaves its ``torch.device('meta')`` context
    # immediately before this integration hook.  Match the base network's
    # device so FSDP never sees a mixed meta/CPU module tree.
    try:
        adapter_device = next(net.parameters()).device
    except StopIteration:  # pragma: no cover - a real Cosmos VFM always has parameters
        adapter_device = torch.device("cpu")
    with torch.device(adapter_device):
        controller = CosmosAdapterController(net, config, action_dim=action_dim)
    net.add_module("cross_embodiment_adapter", controller)

    original_init_weights = getattr(net, "init_weights", None)
    if callable(original_init_weights):

        def init_weights_with_adapter(module: nn.Module, *args: Any, **kwargs: Any):
            result = original_init_weights(*args, **kwargs)
            module.cross_embodiment_adapter.reset_parameters()
            return result

        net.init_weights = types.MethodType(init_weights_with_adapter, net)  # type: ignore[method-assign]

    def forward_with_conditions(
        module: nn.Module,
        *,
        packed_seq: Any,
        state: CosmosConditionState,
        memory: Any = None,
        video_temporal_causal: bool | None = None,
    ):
        # This method is registered as an FSDP forward method after upstream
        # parallelization.  Source/reference projections therefore execute while
        # the root parameters are unsharded, in the same lifecycle as VFM forward.
        with module.cross_embodiment_adapter.condition(packed_seq, state):
            return module.forward(
                packed_seq=packed_seq,
                memory=memory,
                video_temporal_causal=video_temporal_causal,
            )

    net._genet_forward_with_conditions = types.MethodType(forward_with_conditions, net)  # type: ignore[attr-defined]
    return controller


class CosmosCrossEmbodimentModel(_OmniMoTModel):  # type: ignore[misc]
    """Upstream OmniMoT training model with Source/Reference batch support.

    The dataset keeps target ground truth in the normal Cosmos keys
    (``video``, ``action``, ``domain_id``) and adds prefixed auxiliary keys.
    This subclass lets the upstream training step own VAE encoding, RF noising,
    loss, FSDP, EMA, and DCP while only overriding auxiliary encoding/denoise.
    """

    def __init__(
        self,
        *args: Any,
        cross_embodiment_config: ModelConfig | dict[str, Any],
        copy_shared_reference_to_dual_on_warm_start: bool = False,
        initialize_ema_from_regular_on_warm_start: bool = True,
        condition_dropout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        _require_cosmos()
        adapter_config = (
            cross_embodiment_config
            if isinstance(cross_embodiment_config, ModelConfig)
            else model_config_from_mapping(cross_embodiment_config)
        )
        if adapter_config.parallelism.context_parallel_shard_degree != 1:
            raise ValueError("GenET condition state currently requires context_parallel_shard_degree=1")
        if not 0.0 <= float(condition_dropout) < 1.0:
            raise ValueError("condition_dropout must be in [0, 1)")
        # ``super().__init__`` calls build_net(), which in turn invokes our
        # install_attention_dispatch() hook before FSDP/activation-checkpoint
        # transforms.  Keep the config available during that call without
        # registering it as a torch child.
        object.__setattr__(self, "_genet_adapter_config", adapter_config)
        object.__setattr__(
            self,
            "_genet_copy_shared_reference_to_dual_on_warm_start",
            bool(copy_shared_reference_to_dual_on_warm_start),
        )
        object.__setattr__(
            self,
            "_genet_initialize_ema_from_regular_on_warm_start",
            bool(initialize_ema_from_regular_on_warm_start),
        )
        object.__setattr__(self, "_genet_condition_dropout", float(condition_dropout))
        object.__setattr__(self, "_genet_warm_start_copy_applied", False)
        super().__init__(*args, **kwargs)
        controller = self.net.cross_embodiment_adapter
        # The controller is already registered below self.net. Keep only a plain
        # convenience reference here so state_dict keys are not duplicated.
        object.__setattr__(self, "_genet_controller", controller)
        self._genet_condition_state: CosmosConditionState | None = None
        self._genet_inference_condition_state: CosmosConditionState | None = None

    @staticmethod
    def _should_copy_reference_on_load(
        *,
        enabled: bool,
        has_resumable_checkpoint: bool,
        has_load_path: bool,
    ) -> bool:
        """Only migrate a warm start; never overwrite an exact resume."""

        return enabled and has_load_path and not has_resumable_checkpoint

    @staticmethod
    def _should_initialize_ema_on_load(
        *,
        enabled: bool,
        has_resumable_checkpoint: bool,
        has_load_path: bool,
    ) -> bool:
        return enabled and has_load_path and not has_resumable_checkpoint

    @staticmethod
    def _copy_reference_route_a_to_b(net: nn.Module) -> int:
        controller = getattr(net, "cross_embodiment_adapter", None)
        if controller is None:
            raise RuntimeError("Cosmos network has no cross_embodiment_adapter")
        copied = 0
        for layer in controller.reference_layers:
            if layer.projector.route_b is not None:
                layer.projector.copy_shared_to_dual()
                copied += 1
        return copied

    def load_pretrained_model_if_needed(
        self,
        *,
        has_resumable_checkpoint: bool,
        has_load_path: bool,
    ) -> None:
        """Run the optional shared→dual migration after upstream DCP loading."""

        super().load_pretrained_model_if_needed(
            has_resumable_checkpoint=has_resumable_checkpoint,
            has_load_path=has_load_path,
        )
        should_initialize_ema = self._should_initialize_ema_on_load(
            enabled=self._genet_initialize_ema_from_regular_on_warm_start,
            has_resumable_checkpoint=has_resumable_checkpoint,
            has_load_path=has_load_path,
        )
        if should_initialize_ema and self.config.ema.enabled:
            # Warm-start intentionally skips net_ema.* in DCP. Reset EMA from
            # the now-loaded regular network instead of leaving its pre-load
            # random/base construction behind.
            self.net_ema_worker.copy_to(src_model=self.net, tgt_model=self.net_ema)
        should_copy = self._should_copy_reference_on_load(
            enabled=self._genet_copy_shared_reference_to_dual_on_warm_start,
            has_resumable_checkpoint=has_resumable_checkpoint,
            has_load_path=has_load_path,
        )
        if not should_copy or self._genet_warm_start_copy_applied:
            return
        self._copy_reference_route_a_to_b(self.net)
        if self.config.ema.enabled:
            self._copy_reference_route_a_to_b(self.net_ema)
        object.__setattr__(self, "_genet_warm_start_copy_applied", True)

    def install_attention_dispatch(self, net: nn.Module) -> None:
        """Install GenET before upstream parallelizes the complete module tree."""

        super().install_attention_dispatch(net)
        install_cosmos_adapter(
            net,
            self._genet_adapter_config,
            action_dim=int(self.config.max_action_dim),
        )

    def build_net(self, dtype: torch.dtype, *, lora_enabled: bool | None = None) -> nn.Module:
        """Register the condition-aware call as a root FSDP forward method."""

        net = super().build_net(dtype=dtype, lora_enabled=lora_enabled)
        from torch.distributed.fsdp import register_fsdp_forward_method

        register_fsdp_forward_method(net, "_genet_forward_with_conditions")
        return net

    @staticmethod
    def _as_sample_list(
        value: Any,
        *,
        item_ndim: int,
        key: str,
    ) -> list[torch.Tensor] | None:
        """Normalize collated/packed auxiliary values to unbatched samples.

        The pinned ``PackingDataLoader`` represents non-standard tensor keys as
        ``list[Tensor(1, ...)]``.  A regular PyTorch collate instead produces one
        ``Tensor(B, ...)``.  Cosmos tokenizers want a list of tensors with the
        leading batch dimension removed in both cases.
        """

        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.ndim == item_ndim:
                return [value]
            if value.ndim == item_ndim + 1:
                return list(value.unbind(0))
            raise ValueError(
                f"{key} must have item rank {item_ndim} or batched rank {item_ndim + 1}, "
                f"got shape {tuple(value.shape)}"
            )
        if isinstance(value, (list, tuple)):
            samples: list[torch.Tensor] = []
            for item in value:
                while isinstance(item, (list, tuple)) and len(item) == 1:
                    item = item[0]
                if not isinstance(item, torch.Tensor):
                    raise TypeError(
                        f"{key} entries must be tensors, got {type(item).__name__}"
                    )
                if item.ndim == item_ndim + 1 and item.shape[0] == 1:
                    item = item.squeeze(0)
                if item.ndim != item_ndim:
                    raise ValueError(
                        f"{key} entries must have rank {item_ndim} after removing the "
                        f"PackingDataLoader batch axis, got shape {tuple(item.shape)}"
                    )
                samples.append(item)
            return samples or None
        raise TypeError(
            f"Expected tensor/list auxiliary batch value for {key}, got {type(value).__name__}"
        )

    @staticmethod
    def _get_auxiliary_value(data_batch: dict[str, Any], prefix: str, field: str) -> Any:
        aliases = {
            ("source", "video"): ("source_video", "control_video"),
            ("source", "action"): (
                "source_action",
                "source_actions",
                "control_action",
                "control_actions",
            ),
            ("source", "domain_id"): ("source_domain_id", "control_domain_id"),
            ("source", "raw_action_dim"): (
                "source_raw_action_dim",
                "control_raw_action_dim",
            ),
            ("source", "image_size"): ("source_image_size", "control_image_size"),
            ("source", "conditioning_fps"): (
                "source_conditioning_fps",
                "control_conditioning_fps",
            ),
            ("reference", "video"): ("reference_video",),
            ("reference", "action"): ("reference_action", "reference_actions"),
            ("reference", "domain_id"): ("reference_domain_id",),
            ("reference", "raw_action_dim"): ("reference_raw_action_dim",),
            ("reference", "image_size"): ("reference_image_size",),
            ("reference", "conditioning_fps"): ("reference_conditioning_fps",),
        }
        for key in aliases[(prefix, field)]:
            if key in data_batch and data_batch[key] is not None:
                return data_batch[key]
        return None

    @staticmethod
    def _is_preprocessed_video(video: list[torch.Tensor], *, key: str) -> bool:
        floating = [torch.is_floating_point(item) for item in video]
        if all(floating):
            return True
        if not any(floating) and all(item.dtype == torch.uint8 for item in video):
            return False
        dtypes = sorted({str(item.dtype) for item in video})
        raise TypeError(
            f"{key} must be uniformly uint8 pixels or floating [-1, 1] tensors; got {dtypes}"
        )

    def _encode_auxiliary(self, data_batch: dict[str, Any], prefix: str, iteration: int):
        video_value = self._get_auxiliary_value(data_batch, prefix, "video")
        video = self._as_sample_list(
            video_value,
            item_ndim=4,
            key=f"{prefix}_video",
        )
        if video is None:
            accepted = "source_video/control_video" if prefix == "source" else "reference_video"
            raise KeyError(f"Missing required auxiliary video batch key ({accepted})")
        action = self._as_sample_list(
            self._get_auxiliary_value(data_batch, prefix, "action"),
            item_ndim=2,
            key=f"{prefix}_action",
        )
        domains = self._as_sample_list(
            self._get_auxiliary_value(data_batch, prefix, "domain_id"),
            item_ndim=0,
            key=f"{prefix}_domain_id",
        )
        if action is not None and domains is None:
            raise KeyError(f"{prefix}_domain_id is required when {prefix}_action is present")
        if action is not None and len(action) != len(video):
            raise ValueError(
                f"{prefix}_action sample count {len(action)} != {prefix}_video count {len(video)}"
            )
        if domains is not None and action is not None and len(domains) != len(action):
            raise ValueError(
                f"{prefix}_domain_id count {len(domains)} != {prefix}_action count {len(action)}"
            )
        aux: dict[str, Any] = {
            self.input_video_key: video,
            self.input_caption_key: data_batch.get(self.input_caption_key, [""] * len(video)),
            "is_preprocessed": self._is_preprocessed_video(video, key=f"{prefix}_video"),
        }
        conditioning_fps = self._get_auxiliary_value(data_batch, prefix, "conditioning_fps")
        if conditioning_fps is None:
            conditioning_fps = data_batch.get("conditioning_fps")
        if conditioning_fps is not None:
            aux["conditioning_fps"] = conditioning_fps
        image_size = self._as_sample_list(
            self._get_auxiliary_value(data_batch, prefix, "image_size"),
            item_ndim=1,
            key=f"{prefix}_image_size",
        )
        if image_size is not None:
            aux["image_size"] = image_size
        if action is not None:
            aux["action"] = action
            aux["domain_id"] = domains
            raw_action_dim = self._as_sample_list(
                self._get_auxiliary_value(data_batch, prefix, "raw_action_dim"),
                item_ndim=0,
                key=f"{prefix}_raw_action_dim",
            )
            if raw_action_dim is None:
                raw_action_dim = [
                    torch.tensor(item.shape[-1], dtype=torch.long, device=item.device)
                    for item in action
                ]
            aux["raw_action_dim"] = raw_action_dim
        return self.get_data_and_condition(aux, iteration=iteration, retain_raw_state_vision=False)

    def _build_condition_state(
        self,
        data_batch: dict[str, Any],
        *,
        iteration: int,
    ) -> CosmosConditionState:
        source = self._encode_auxiliary(data_batch, "source", iteration)
        reference = (
            self._encode_auxiliary(data_batch, "reference", iteration)
            if self._genet_adapter_config.reference.enabled
            else None
        )
        return CosmosConditionState(
            source_vision=source.x0_tokens_vision,
            source_action=source.x0_tokens_action,
            source_domain_id=source.action_domain_id,
            target_raw_action_dim=self._as_sample_list(
                data_batch.get("raw_action_dim"),
                item_ndim=0,
                key="raw_action_dim",
            ),
            reference_vision=reference.x0_tokens_vision if reference is not None else [],
            reference_action=reference.x0_tokens_action if reference is not None else None,
            reference_domain_id=reference.action_domain_id if reference is not None else None,
        )

    def _prepare_training_data(self, data_batch: dict[str, Any], iteration: int):
        result = super()._prepare_training_data(data_batch, iteration)
        if self.parallel_dims is not None and self.parallel_dims.cp_enabled:
            raise ValueError("GenET Cosmos training currently requires context parallel degree 1")
        self._genet_condition_state = self._build_condition_state(
            data_batch,
            iteration=iteration,
        )
        if self._genet_condition_dropout > 0.0:
            # Exactly two draws in a stable source/reference order makes the
            # stochastic conditioning schedule identical for shared and dual
            # reference-projection ablations given the same training seed.
            keep = torch.rand(2) >= self._genet_condition_dropout
            self._genet_condition_state.use_source = bool(keep[0].item())
            self._genet_condition_state.use_reference = bool(keep[1].item())
        return result

    @torch.no_grad()
    def generate_samples_from_batch(
        self,
        data_batch: dict[str, Any],
        net: nn.Module | None = None,
        *args: Any,
        use_source_condition: bool = True,
        use_reference_condition: bool = True,
        **kwargs: Any,
    ):
        """Run upstream joint video/action sampling with GenET conditions.

        For ordinary generation the dataset's SequencePlan marks every target
        token as noisy, so Target ``video``/``action`` values provide shapes only.
        Rolling long-horizon inference may instead mark a clean Target
        video/action prefix; upstream sampling then clamps those indexed tokens
        while denoising the suffix. Source and reference tensors are encoded once
        before each sampling loop, and ``denoise`` reuses their condition state
        for every solver/CFG forward.
        """

        if self._genet_inference_condition_state is not None:
            raise RuntimeError("Nested GenET sampling calls are not supported")
        if self.parallel_dims is not None and self.parallel_dims.cp_enabled:
            raise ValueError("GenET Cosmos inference currently requires context parallel degree 1")
        target_net = self.net if net is None else net
        target_controller = getattr(target_net, "cross_embodiment_adapter", None)
        if target_controller is None or not callable(
            getattr(target_controller, "clear_condition_state", None)
        ):
            raise TypeError(
                "sampling net must carry a GenET cross_embodiment_adapter; use the "
                "regular net/ema_scope or install the adapter on the explicit net"
            )
        state = self._build_condition_state(data_batch, iteration=0)
        state.use_source = bool(use_source_condition)
        state.use_reference = bool(use_reference_condition)
        self._genet_inference_condition_state = state
        try:
            return super().generate_samples_from_batch(
                data_batch,
                net,
                *args,
                **kwargs,
            )
        finally:
            self._genet_inference_condition_state = None
            target_controller.clear_condition_state()

    def denoise(self, net=None, data_batch_packed=None, memory=None, video_temporal_causal=None):
        if (
            self._genet_condition_state is not None
            and self._genet_inference_condition_state is not None
        ):
            raise RuntimeError("Training and inference condition states cannot overlap")
        training_state = self._genet_condition_state
        state = training_state or self._genet_inference_condition_state
        if state is None:
            raise RuntimeError("No GenET conditions were prepared for this forward pass")
        target_net = net or self.net
        if data_batch_packed is None:
            raise ValueError("data_batch_packed is required")
        try:
            out_net = target_net._genet_forward_with_conditions(
                packed_seq=data_batch_packed,
                state=state,
                memory=memory,
                video_temporal_causal=video_temporal_causal,
            )
        finally:
            if training_state is not None:
                self._genet_condition_state = None

        output_dict = {"preds_vision": out_net["preds_vision"]}
        if self.config.action_gen and "preds_action" in out_net:
            output_dict["preds_action"] = out_net["preds_action"]
        if self.config.sound_gen and "preds_sound" in out_net:
            output_dict["preds_sound"] = out_net["preds_sound"]
        output_dict.update(
            (key, value) for key, value in out_net.items() if "lbl_metadata_" in key
        )
        return output_dict
