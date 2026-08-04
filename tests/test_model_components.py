import torch

from genet.config import ModelConfig, ReferenceConfig
from genet.models.control import SourceControlAdapter
from genet.models.generator import CrossEmbodimentGenerator
from genet.models.reference_attention import RoutedReferenceCrossAttention, migrate_reference_projectors
from genet.training.stages import configure_trainable_stage


def test_zero_initialized_source_control_is_exact_noop() -> None:
    adapter = SourceControlAdapter(4, 6, 3, zero_init=True)
    target_video = torch.randn(2, 4, 3, 2, 2)
    source_video = torch.randn_like(target_video)
    target_action = torch.randn(2, 5, 6)
    source_action = torch.randn_like(target_action)
    actual_video = adapter.video(target_video, source_video)
    actual_action = adapter.action(
        target_action,
        source_action,
        torch.tensor([0, 1]),
        torch.tensor([1, 2]),
    )
    assert torch.equal(actual_video, target_video)
    assert torch.equal(actual_action, target_action)

    (actual_video.sum() + actual_action.sum()).backward()
    assert adapter.video.proj.weight.grad is not None
    assert adapter.action.out_proj.weight.grad is not None


def test_dropped_source_control_is_noop_but_keeps_autograd_path() -> None:
    adapter = SourceControlAdapter(4, 6, 3, zero_init=False)
    target_video = torch.randn(2, 4, 3, 2, 2)
    source_video = torch.randn_like(target_video)
    target_action = torch.randn(2, 5, 6)
    source_action = torch.randn_like(target_action)
    actual_video = adapter.video(target_video, source_video, enabled=False)
    actual_action = adapter.action(
        target_action,
        source_action,
        torch.tensor([0, 1]),
        torch.tensor([1, 2]),
        enabled=False,
    )
    assert torch.equal(actual_video, target_video)
    assert torch.equal(actual_action, target_action)

    (actual_video.sum() + actual_action.sum()).backward()
    assert adapter.video.proj.weight.grad is not None
    assert adapter.action.out_proj.weight.grad is not None
    assert torch.count_nonzero(adapter.video.proj.weight.grad) == 0
    assert torch.count_nonzero(adapter.action.out_proj.weight.grad) == 0


def test_source_video_control_accepts_cosmos_unbatched_latent() -> None:
    adapter = SourceControlAdapter(4, 6, 3, zero_init=True).to(dtype=torch.bfloat16)
    target = torch.randn(4, 3, 2, 2, dtype=torch.bfloat16)
    # Upstream VAE x0 is fp32 while the noised target/model run in bf16.
    source = torch.randn(4, 3, 2, 2, dtype=torch.float32)
    actual = adapter.video(target, source)
    assert actual.shape == target.shape
    assert torch.equal(actual, target)
    actual.sum().backward()
    assert adapter.video.proj.weight.grad is not None


def test_source_action_control_masks_target_padding_channels() -> None:
    adapter = SourceControlAdapter(4, 6, 3, zero_init=False)
    target = torch.zeros(1, 5, 6)
    source = torch.randn_like(target)
    output_mask = torch.zeros_like(target, dtype=torch.bool)
    output_mask[:, :, :2] = True
    actual = adapter.action(
        target,
        source,
        torch.tensor([0]),
        torch.tensor([1]),
        output_mask=output_mask,
    )
    assert torch.count_nonzero(actual[:, :, 2:]) == 0


def test_shared_and_dual_start_functionally_identical() -> None:
    torch.manual_seed(7)
    shared = RoutedReferenceCrossAttention(16, 4, projection_mode="shared", gate_init=0.5)
    torch.manual_seed(7)
    dual = RoutedReferenceCrossAttention(16, 4, projection_mode="dual", gate_init=0.5)
    query = torch.randn(2, 3, 16)
    reference = torch.randn(2, 5, 16)
    for route in (0, 1):
        expected = shared(query, reference, route_index=route)
        actual = dual(query, reference, route_index=route)
        assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_shared_checkpoint_can_seed_dual_route() -> None:
    state = {
        "blocks.0.projector.route_a.key.weight": torch.randn(4, 4),
        "blocks.0.projector.route_a.value.weight": torch.randn(4, 4),
    }
    migrated = migrate_reference_projectors(state, prefix="blocks.0.", destination_mode="dual")
    assert torch.equal(
        migrated["blocks.0.projector.route_b.key.weight"],
        state["blocks.0.projector.route_a.key.weight"],
    )


def test_toy_generator_joint_video_action_forward_and_stages() -> None:
    config = ModelConfig(
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        latent_channels=8,
        patch_size=2,
        num_embodiments=4,
        reference=ReferenceConfig(num_heads=4, projection_mode="dual", gate_init=0.0),
    )
    model = CrossEmbodimentGenerator(config, action_dim=6)
    video = torch.randint(0, 256, (1, 3, 17, 32, 32), dtype=torch.uint8)
    latent = model.encode_video(video)
    action = torch.randn(1, 16, 6)
    output = model(
        noisy_target_video=latent,
        noisy_target_action=action,
        sigma=torch.tensor([0.5]),
        source_video=latent,
        source_action=action,
        source_domain_id=torch.tensor([0]),
        target_domain_id=torch.tensor([1]),
        reference_video=latent,
        reference_action=action,
        reference_domain_id=torch.tensor([1]),
    )
    assert output.video_velocity.shape == latent.shape
    assert output.action_velocity.shape == action.shape

    counts = configure_trainable_stage(model, "control")
    assert 0 < counts["trainable"] < counts["total"]
    assert all(
        ("control_adapter" in name) == parameter.requires_grad
        for name, parameter in model.named_parameters()
    )
