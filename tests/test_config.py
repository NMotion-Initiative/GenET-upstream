from pathlib import Path

import pytest

from genet.config import load_config


def test_config_inheritance_and_fingerprint() -> None:
    config = load_config(Path("configs/experiments/stage2_dual_8gpu.yaml"))
    assert config.model.reference.projection_mode == "dual"
    assert config.model.parallelism.data_parallel_replicate_degree == 1
    assert len(config.fingerprint()) == 64
    local_path_variant = load_config(Path("configs/experiments/stage2_dual_8gpu.yaml"))
    local_path_variant.data.manifest = "/another-node/data/manifest.jsonl"
    local_path_variant.checkpoint.output_dir = "/another-node/output"
    assert local_path_variant.fingerprint() != config.fingerprint()
    assert local_path_variant.distributed_fingerprint() == config.distributed_fingerprint()
    local_path_variant.checkpoint.resume = "/node-local/resume"
    assert local_path_variant.distributed_fingerprint() != config.distributed_fingerprint()


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
