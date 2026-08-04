import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

import genet.models.cosmos_adapter as cosmos_adapter
import genet.integrations.cosmos_data as cosmos_data
from genet.config import ModelConfig, ReferenceConfig
from genet.integrations.cosmos_data import (
    COSMOS_OFFICIAL_BATCH_KEYS,
    CosmosProcessedPairDataset,
)


def _write_processed_manifest(root: Path, count: int = 5) -> Path:
    manifest = root / "manifest.jsonl"
    entries = []
    for index in range(count):
        arrays = {}
        for role, pixel, action_base in (
            ("source", 10 + index, 10.0 + index),
            ("target", 100 + index, 20.0 + index),
            ("reference", 150 + index, 30.0 + index),
        ):
            arrays[f"{role}_video"] = np.full((5, 4, 6, 3), pixel, dtype=np.uint8)
            action = np.zeros((5, 4), dtype=np.float32)
            action[:, 0] = np.arange(5, dtype=np.float32) + action_base
            action[:, 1] = -action[:, 0]
            arrays[f"{role}_actions"] = action
            action_mask = np.zeros((5, 4), dtype=np.bool_)
            action_mask[:, :2] = True
            arrays[f"{role}_action_mask"] = action_mask
            arrays[f"{role}_frame_mask"] = np.ones(5, dtype=np.bool_)
        sample_path = root / f"sample-{index}.npz"
        np.savez_compressed(sample_path, **arrays)
        entries.append(
            {
                "id": f"pair-{index}",
                "format_version": "genet.processed-pair/v1",
                "npz": sample_path.name,
                "source": {
                    "episode_id": f"source-{index}",
                    "embodiment": "franka",
                    "metadata": {"task": "pick the red cube"},
                },
                "target_gt": {
                    "episode_id": f"target-{index}",
                    "embodiment": "ur5",
                    "metadata": {},
                },
                "reference_target": {
                    "episode_id": f"reference-{index}",
                    "embodiment": "ur5",
                    "metadata": {},
                },
                "metadata": {"task_id": f"pick-{index}"},
            }
        )
    manifest.write_text(
        "".join(json.dumps(entry) + "\n" for entry in entries),
        encoding="utf-8",
    )
    return manifest


def test_cosmos_dataset_emits_official_and_auxiliary_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    tokenizer = object()
    monkeypatch.setattr(
        cosmos_data,
        "_lazy_instantiate",
        lambda _config: SimpleNamespace(tokenizer=tokenizer),
    )
    monkeypatch.setattr(cosmos_data, "_add_special_tokens", lambda value: (value, 1))
    monkeypatch.setattr(
        cosmos_data,
        "_tokenize_caption",
        lambda caption, _tokenizer, **_kwargs: [11, len(caption) + 12],
    )
    dataset = CosmosProcessedPairDataset(
        _write_processed_manifest(tmp_path),
        embodiment_map={"franka": 3, "ur5": 7},
        fps=8.0,
        reference_mode="stored",
        reference_seed=12,
        action_alignment="frame",
        tokenizer_config={"_target_": "fake"},
        max_action_dim=6,
    )

    sample = dataset[0]
    assert COSMOS_OFFICIAL_BATCH_KEYS <= sample.keys()
    assert sample["sample_id"] == "pair-0"
    assert sample["video"].shape == (3, 5, 4, 6)
    assert sample["video"].dtype == torch.uint8
    assert sample["text_token_ids"].tolist() == [11, 18]
    assert sample["action"].shape == (5, 6)
    assert sample["source_action"].shape == (5, 6)
    assert sample["reference_action"].shape == (5, 6)
    assert sample["raw_action_dim"].item() == 2
    assert sample["action_processing_record"].raw_action_dim == 2
    assert sample["action_processing_record"].action_normalizer is None
    assert sample["source_raw_action_dim"].item() == 2
    assert sample["domain_id"].item() == 7
    assert sample["source_domain_id"].item() == 3
    assert sample["reference_domain_id"].item() == 7
    assert sample["conditioning_fps"].item() == 8.0
    assert sample["image_size"].tolist() == [4.0, 6.0, 4.0, 6.0]
    assert sample["ai_caption"] == "pick-0"
    assert sample["sequence_plan"].has_vision
    assert sample["sequence_plan"].has_action
    assert sample["sequence_plan"].condition_frame_indexes_vision == []
    assert sample["sequence_plan"].condition_frame_indexes_action == []
    assert sample["sequence_plan"].action_start_frame_offset == 0


def test_dynamic_loader_sharding_and_transition_alignment(tmp_path: Path):
    dataset = CosmosProcessedPairDataset(
        _write_processed_manifest(tmp_path),
        embodiment_map={"franka": 0, "ur5": 1},
        fps=16,
        action_alignment="transition",
        max_action_dim=4,
    )
    assert len(dataset) == 5

    # RankPartitionedDataLoader assigns these after LazyCall instantiation.
    dataset.shard_world_size = 2
    dataset.shard_rank = 1
    assert len(dataset) == 2
    assert [dataset[i]["sample_id"] for i in range(len(dataset))] == ["pair-1", "pair-3"]
    sample = dataset[0]
    assert sample["action"].shape == (4, 4)
    assert sample["source_action"].shape == (4, 4)
    assert sample["action"][0, 0].item() == pytest.approx(22.0)
    assert sample["sequence_plan"].action_start_frame_offset == 1

    dataset.shard_rank = 0
    assert len(dataset) == 2
    assert [dataset[i]["sample_id"] for i in range(len(dataset))] == [
        "pair-0",
        "pair-2",
    ]


def test_epoch_tag_rotates_which_global_tail_sample_is_omitted(tmp_path: Path):
    manifest = _write_processed_manifest(tmp_path, count=5)
    rank_samples: list[set[str]] = []
    for epoch in (0, 1):
        observed: set[str] = set()
        for rank in (0, 1):
            dataset = CosmosProcessedPairDataset(
                manifest,
                embodiment_map={"franka": 0, "ur5": 1},
                fps=8,
                shard_seed=0,
            )
            dataset.shard_world_size = 2
            dataset.shard_rank = rank
            observed.update(
                dataset[(epoch, index)]["sample_id"] for index in range(len(dataset))
            )
        rank_samples.append(observed)

    assert rank_samples[0] == {"pair-0", "pair-1", "pair-2", "pair-3"}
    assert rank_samples[1] == {"pair-1", "pair-2", "pair-3", "pair-4"}


def test_missing_embodiment_mapping_fails_with_role_context(tmp_path: Path):
    dataset = CosmosProcessedPairDataset(
        _write_processed_manifest(tmp_path, count=1),
        embodiment_map={"ur5": 1},
        fps=8,
    )
    with pytest.raises(KeyError, match="source embodiment 'franka'"):
        dataset[0]


def test_adapter_unwraps_packing_dataloader_singleton_axes():
    packed_video = [torch.zeros(1, 3, 5, 4, 6)]
    packed_action = [torch.zeros(1, 5, 8)]
    packed_domain = [torch.tensor([2])]

    video = cosmos_adapter.CosmosCrossEmbodimentModel._as_sample_list(
        packed_video, item_ndim=4, key="source_video"
    )
    action = cosmos_adapter.CosmosCrossEmbodimentModel._as_sample_list(
        packed_action, item_ndim=2, key="source_action"
    )
    domain = cosmos_adapter.CosmosCrossEmbodimentModel._as_sample_list(
        packed_domain, item_ndim=0, key="source_domain_id"
    )
    assert video is not None and video[0].shape == (3, 5, 4, 6)
    assert action is not None and action[0].shape == (5, 8)
    assert domain is not None and domain[0].shape == torch.Size([])


class _StubLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Linear(8, 8)])


class _StubCosmosNet(nn.Module):
    latent_channel = 2
    action_dim = 4
    hidden_size = 8

    def __init__(self) -> None:
        super().__init__()
        self.language_model = _StubLanguageModel()
        self.base_init_calls = 0

    def init_weights(self, _buffer_device=None):
        self.base_init_calls += 1

    def forward(self, **kwargs):
        return kwargs

    def patchify_and_pack_latents(self, latents, shapes):
        assert all(latent.ndim == 4 for latent in latents)
        expected = [tuple(int(value) for value in latent.shape[-3:]) for latent in latents]
        assert shapes == expected
        token_count = sum(t * h * w for t, h, w in shapes)
        return torch.randn(token_count, 2), shapes


def test_stub_install_reapplies_zero_and_dual_initialization(monkeypatch):
    monkeypatch.setattr(cosmos_adapter, "COSMOS_AVAILABLE", True)
    net = _StubCosmosNet()
    config = ModelConfig(
        num_embodiments=8,
        reference=ReferenceConfig(projection_mode="dual", num_heads=2),
    )
    controller = cosmos_adapter.install_cosmos_adapter(net, config, action_dim=4)
    with torch.no_grad():
        controller.source_control.video.proj.weight.fill_(1)
        controller.reference_layers[0].projector.route_a.key.weight.fill_(2)
        controller.reference_layers[0].projector.route_b.key.weight.fill_(3)

    net.init_weights(None)
    assert net.base_init_calls == 1
    assert torch.count_nonzero(controller.source_control.video.proj.weight) == 0
    route_a = controller.reference_layers[0].projector.route_a.state_dict()
    route_b = controller.reference_layers[0].projector.route_b.state_dict()
    assert route_a.keys() == route_b.keys()
    assert all(torch.equal(route_a[key], route_b[key]) for key in route_a)
    assert hasattr(net, "_genet_forward_with_conditions")


def test_shared_and_dual_reset_keep_common_parameter_initialization_identical(monkeypatch):
    monkeypatch.setattr(cosmos_adapter, "COSMOS_AVAILABLE", True)
    shared = cosmos_adapter.install_cosmos_adapter(
        _StubCosmosNet(),
        ModelConfig(
            num_embodiments=8,
            reference=ReferenceConfig(projection_mode="shared", num_heads=2),
        ),
        action_dim=4,
    )
    dual = cosmos_adapter.install_cosmos_adapter(
        _StubCosmosNet(),
        ModelConfig(
            num_embodiments=8,
            reference=ReferenceConfig(projection_mode="dual", num_heads=2),
        ),
        action_dim=4,
    )
    torch.manual_seed(1234)
    shared.reset_parameters()
    torch.manual_seed(1234)
    dual.reset_parameters()
    shared_state = shared.state_dict()
    dual_state = dual.state_dict()
    for key, value in shared_state.items():
        assert key in dual_state
        assert torch.equal(value, dual_state[key]), key


def test_dropped_source_condition_keeps_adapter_parameters_in_graph(monkeypatch):
    monkeypatch.setattr(cosmos_adapter, "COSMOS_AVAILABLE", True)
    net = _StubCosmosNet()
    config = ModelConfig(
        num_embodiments=8,
        reference=ReferenceConfig(enabled=False, num_heads=2),
    )
    controller = cosmos_adapter.install_cosmos_adapter(net, config, action_dim=4)
    target_video = torch.randn(1, 2, 3, 2, 2)
    target_action = torch.randn(3, 4)
    packed = SimpleNamespace(
        vision=SimpleNamespace(tokens=[target_video], sequence_indexes=[0]),
        action=SimpleNamespace(
            tokens=[target_action],
            sequence_indexes=[0],
            domain_id=[torch.tensor(1)],
        ),
    )
    state = cosmos_adapter.CosmosConditionState(
        source_vision=[torch.randn_like(target_video)],
        source_action=[torch.randn_like(target_action)],
        source_domain_id=[torch.tensor(0)],
        target_raw_action_dim=[torch.tensor(4)],
        reference_vision=[torch.randn_like(target_video)],
        reference_action=[torch.randn_like(target_action)],
        reference_domain_id=[torch.tensor(1)],
        use_source=False,
        use_reference=False,
    )
    with controller.condition(packed, state):
        controlled_video = packed.vision.tokens[0]
        controlled_action = packed.action.tokens[0]
        assert torch.equal(controlled_video, target_video)
        assert torch.equal(controlled_action, target_action)
        loss = controlled_video.sum() + controlled_action.sum()
    loss.backward()
    assert controller.source_control.video.proj.weight.grad is not None
    assert controller.source_control.action.out_proj.weight.grad is not None


def test_cosmos_source_action_residual_respects_target_raw_dim(monkeypatch):
    monkeypatch.setattr(cosmos_adapter, "COSMOS_AVAILABLE", True)
    net = _StubCosmosNet()
    config = ModelConfig(
        num_embodiments=8,
        reference=ReferenceConfig(enabled=False, num_heads=2),
    )
    controller = cosmos_adapter.install_cosmos_adapter(net, config, action_dim=4)
    with torch.no_grad():
        controller.source_control.action.out_proj.weight.zero_()
        controller.source_control.action.out_proj.bias.fill_(1.0)
    target_video = torch.randn(1, 2, 3, 2, 2)
    target_action = torch.zeros(3, 4)
    packed = SimpleNamespace(
        vision=SimpleNamespace(tokens=[target_video], sequence_indexes=[0]),
        action=SimpleNamespace(
            tokens=[target_action],
            sequence_indexes=[0],
            domain_id=[torch.tensor(1)],
        ),
    )
    state = cosmos_adapter.CosmosConditionState(
        source_vision=[torch.randn_like(target_video)],
        source_action=[torch.randn_like(target_action)],
        source_domain_id=[torch.tensor(0)],
        target_raw_action_dim=[torch.tensor(2)],
        reference_vision=[],
        reference_action=None,
        reference_domain_id=None,
    )
    with controller.condition(packed, state):
        controlled = packed.action.tokens[0]
        assert torch.equal(controlled[:, :2], torch.ones_like(controlled[:, :2]))
        assert torch.count_nonzero(controlled[:, 2:]) == 0


def test_reference_builder_accepts_cosmos_unbatched_latent(monkeypatch):
    monkeypatch.setattr(cosmos_adapter, "COSMOS_AVAILABLE", True)
    net = _StubCosmosNet()
    config = ModelConfig(
        num_embodiments=8,
        reference=ReferenceConfig(use_action=False, num_heads=2),
    )
    controller = cosmos_adapter.install_cosmos_adapter(net, config, action_dim=4)
    # The real Cosmos net projects a patch width into hidden_size. Keep the stub
    # projection shape-compatible with the synthetic packed tokens above.
    net.vae2llm = nn.Linear(2, net.hidden_size)
    state = cosmos_adapter.CosmosConditionState(
        source_vision=[],
        source_action=None,
        source_domain_id=None,
        target_raw_action_dim=None,
        reference_vision=[torch.randn(2, 3, 4, 5)],
        reference_action=None,
        reference_domain_id=None,
    )
    tokens = controller._build_reference_tokens(state)
    assert tokens is not None
    assert tokens.shape == (1, 3 * 4 * 5, net.hidden_size)


def test_reference_position_encoding_preserves_time_and_space() -> None:
    video = cosmos_adapter._reference_video_position_encoding(
        [(3, 2, 2)],
        patch_size=1,
        hidden_size=8,
        temporal_extent=4.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    action = cosmos_adapter._reference_action_position_encoding(
        5,
        hidden_size=8,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert video.shape == (12, 8)
    assert action.shape == (5, 8)
    assert not torch.equal(video[0], video[1])  # horizontal position
    assert not torch.equal(video[0], video[4])  # temporal position
    assert torch.allclose(video[0], action[0])
    assert torch.allclose(video[8], action[4])


def test_dropped_reference_condition_keeps_both_dual_routes_in_graph(monkeypatch):
    monkeypatch.setattr(cosmos_adapter, "COSMOS_AVAILABLE", True)
    monkeypatch.setattr(cosmos_adapter, "get_und_seq", lambda pack: pack["causal_seq"], raising=False)
    monkeypatch.setattr(cosmos_adapter, "get_gen_seq", lambda pack: pack["full_only_seq"], raising=False)

    def rebuild(und, gen, original):
        result = dict(original)
        result["causal_seq"] = und
        result["full_only_seq"] = gen
        return result

    monkeypatch.setattr(cosmos_adapter, "from_und_gen_splits", rebuild, raising=False)
    net = _StubCosmosNet()
    config = ModelConfig(
        num_embodiments=8,
        reference=ReferenceConfig(
            projection_mode="dual",
            routing="ar_dm",
            num_heads=2,
        ),
    )
    controller = cosmos_adapter.install_cosmos_adapter(net, config, action_dim=4)
    state = cosmos_adapter.CosmosConditionState(
        source_vision=[],
        source_action=None,
        source_domain_id=None,
        target_raw_action_dim=None,
        reference_vision=[],
        reference_action=None,
        reference_domain_id=None,
        use_reference=False,
        reference_tokens=torch.randn(1, 2, 8),
    )
    controller._state = state
    hidden_pack = {
        "causal_seq": torch.randn(3, 8),
        "full_only_seq": torch.randn(4, 8),
        "_num_causal_tokens": 3,
        "_num_full_tokens": 4,
    }
    result, _, _ = controller._make_layer_hook(0)(None, (), (hidden_pack, None, None))
    assert torch.equal(result["causal_seq"], hidden_pack["causal_seq"])
    assert torch.equal(result["full_only_seq"], hidden_pack["full_only_seq"])
    (result["causal_seq"].sum() + result["full_only_seq"].sum()).backward()
    projector = controller.reference_layers[0].projector
    assert projector.route_a.key.weight.grad is not None
    assert projector.route_b.key.weight.grad is not None


def test_warm_start_copy_gate_never_overwrites_exact_resume(monkeypatch):
    assert cosmos_adapter.CosmosCrossEmbodimentModel._should_copy_reference_on_load(
        enabled=True,
        has_resumable_checkpoint=False,
        has_load_path=True,
    )
    assert not cosmos_adapter.CosmosCrossEmbodimentModel._should_copy_reference_on_load(
        enabled=True,
        has_resumable_checkpoint=True,
        has_load_path=True,
    )
    assert cosmos_adapter.CosmosCrossEmbodimentModel._should_initialize_ema_on_load(
        enabled=True,
        has_resumable_checkpoint=False,
        has_load_path=True,
    )
    assert not cosmos_adapter.CosmosCrossEmbodimentModel._should_initialize_ema_on_load(
        enabled=False,
        has_resumable_checkpoint=False,
        has_load_path=True,
    )

    monkeypatch.setattr(cosmos_adapter, "COSMOS_AVAILABLE", True)
    net = _StubCosmosNet()
    config = ModelConfig(
        num_embodiments=8,
        reference=ReferenceConfig(projection_mode="dual", num_heads=2),
    )
    controller = cosmos_adapter.install_cosmos_adapter(net, config, action_dim=4)
    route_a = controller.reference_layers[0].projector.route_a
    route_b = controller.reference_layers[0].projector.route_b
    with torch.no_grad():
        route_a.key.weight.fill_(4)
        route_b.key.weight.fill_(9)
    copied = cosmos_adapter.CosmosCrossEmbodimentModel._copy_reference_route_a_to_b(net)
    assert copied == 1
    assert torch.equal(route_a.key.weight, route_b.key.weight)


def test_missing_cosmos_dependency_error_preserves_import_cause(monkeypatch):
    error = ImportError("No module named 'cosmos_framework'", name="cosmos_framework")
    monkeypatch.setattr(cosmos_adapter, "COSMOS_AVAILABLE", False)
    monkeypatch.setattr(cosmos_adapter, "_COSMOS_IMPORT_ERROR", error)
    with pytest.raises(RuntimeError, match="Missing Python module 'cosmos_framework'") as caught:
        cosmos_adapter._require_cosmos()
    assert caught.value.__cause__ is error
