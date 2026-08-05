from pathlib import Path

import pytest

from genet.config import load_config

EXPECTED_ROBOTWIN_EMBODIMENTS = [
    "ARX-X5",
    "aloha-agilex",
    "franka-panda",
    "piper",
    "ur5-wsg",
]

PRODUCTION_EXPERIMENTS = [
    "stage1_control_32gpu.yaml",
    "stage2_no_ref_8gpu.yaml",
    "stage2_shared_8gpu.yaml",
    "stage2_dual_8gpu.yaml",
    "stage3_shared_32gpu.yaml",
    "stage3_dual_32gpu.yaml",
]


def test_config_inheritance_and_fingerprint() -> None:
    config = load_config(Path("configs/experiments/stage2_dual_8gpu.yaml"))
    assert config.model.reference.projection_mode == "dual"
    assert config.model.parallelism.data_parallel_replicate_degree == 1
    assert config.data.require_bidirectional_pairs is True
    assert len(config.fingerprint()) == 64
    local_path_variant = load_config(Path("configs/experiments/stage2_dual_8gpu.yaml"))
    local_path_variant.data.manifest = "/another-node/data/manifest.jsonl"
    local_path_variant.checkpoint.output_dir = "/another-node/output"
    assert local_path_variant.fingerprint() != config.fingerprint()
    assert local_path_variant.distributed_fingerprint() == config.distributed_fingerprint()
    local_path_variant.checkpoint.resume = "/node-local/resume"
    assert local_path_variant.distributed_fingerprint() != config.distributed_fingerprint()


@pytest.mark.parametrize("experiment", PRODUCTION_EXPERIMENTS)
def test_production_experiments_pin_bidirectional_robotwin_contract(
    experiment: str,
) -> None:
    config = load_config(Path("configs/experiments") / experiment)
    assert config.data.require_bidirectional_pairs is True
    assert config.data.reference_mode == "stored"
    assert config.data.expected_embodiments == EXPECTED_ROBOTWIN_EMBODIMENTS


@pytest.mark.parametrize(
    ("yaml", "message"),
    [
        (
            "data:\n  expected_embodiments: [franka-panda, franka-panda]\n",
            "must not contain duplicates",
        ),
        (
            "data:\n  expected_embodiments: [franka-panda, '']\n",
            "only non-empty strings",
        ),
        (
            "data:\n  require_bidirectional_pairs: true\n",
            "requires non-empty data.expected_embodiments",
        ),
        (
            "data:\n"
            "  expected_embodiments: [franka-panda, ur5-wsg]\n"
            "  require_bidirectional_pairs: true\n"
            "  reference_mode: deterministic\n",
            "requires data.reference_mode='stored'",
        ),
    ],
)
def test_invalid_bidirectional_data_contract_is_rejected(
    tmp_path: Path, yaml: str, message: str
) -> None:
    path = tmp_path / "bad-bidirectional.yaml"
    path.write_text(yaml, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_config(path)


def test_wan_frame_invariant(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("data:\n  num_frames: 80\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"1 \+ N"):
        load_config(path)


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("train:\n  learning_rate_typo: 1e-4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="learning_rate_typo"):
        load_config(path)


@pytest.mark.parametrize(
    ("yaml", "message"),
    [
        ("model:\n  reference:\n    inject_every_n_layers: 0\n", "inject_every_n_layers"),
        ("data:\n  reference_mode: worker_random\n", "reference_mode"),
        ("train:\n  warmup_steps: -1\n", "warmup_steps"),
    ],
)
def test_invalid_values_are_rejected(
    tmp_path: Path, yaml: str, message: str
) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(yaml, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_config(path)


def test_cosmos_micro_batch_is_explicitly_limited(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        "model:\n  backend: cosmos3_edge\nloader:\n  micro_batch_size: 2\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="micro_batch_size=1"):
        load_config(path)
