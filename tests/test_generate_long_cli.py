import json
import sys
import types
from pathlib import Path

import pytest
import torch

from genet.cli.generate_long import main
from genet.inference.long_horizon import (
    ChunkOutput,
    LongGenerationJob,
    LongHorizonConfig,
    RunIdentity,
    TensorWindowSource,
)


def _write_config(path: Path, *, resume: bool = True, extra: str = "") -> Path:
    path.write_text(
        "\n".join(
            [
                "chunk_frames: 5",
                "overlap_frames: 1",
                "temporal_compression_factor: 4",
                "action_alignment: frame",
                "store_video_dtype: float32",
                "recovery:",
                f"  resume: {'true' if resume else 'false'}",
                extra,
            ]
        ),
        encoding="utf-8",
    )
    return path


def _job(frames: int = 3) -> LongGenerationJob:
    source = TensorWindowSource(
        torch.zeros(3, frames, 2, 2, dtype=torch.uint8),
        torch.zeros(frames, 2),
        alignment="frame",
    )

    def sampler(request):
        video = torch.zeros(3, request.config.chunk_frames, 2, 2)
        action = torch.zeros(request.config.chunk_action_steps, 2)
        if request.context_video is not None:
            video[:, : request.context_frames] = request.context_video
        if request.context_action is not None:
            action[: request.context_action_steps] = request.context_action
        return ChunkOutput(video, action)

    return LongGenerationJob(
        source=source,
        sampler=sampler,
        identity=RunIdentity(
            source_id="source-test",
            model_id="model-test",
            reference_id="reference-test",
            code_revision="test",
        ),
    )


def test_dry_run_validates_config_without_loading_factory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    config_path = _write_config(tmp_path / "long.yaml", resume=False)
    output = tmp_path / "does-not-exist"
    assert (
        main(
            [
                "--config",
                str(config_path),
                "--factory",
                "module_that_must_not_be_imported:build",
                "--output",
                str(output),
                "--resume",
                "--dry-run",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "dry-run"
    assert payload["resume"] is True
    assert payload["stride_frames"] == 4
    assert payload["overlap_latent_frames"] == 1
    assert not output.exists()


def test_config_is_a_strict_long_horizon_mapping(tmp_path: Path):
    config_path = _write_config(tmp_path / "invalid.yaml", extra="unknown_key: 1")
    with pytest.raises(ValueError, match="unknown long-horizon config keys"):
        main(
            [
                "--config",
                str(config_path),
                "--factory",
                "unused:factory",
                "--output",
                str(tmp_path / "output"),
                "--dry-run",
            ]
        )


def test_callable_factory_receives_config_and_runs_job(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
):
    seen: dict[str, object] = {}
    module = types.ModuleType("genet_test_long_factory")

    def build(*, config: LongHorizonConfig):
        seen["config"] = config
        return _job()

    module.build = build  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    config_path = _write_config(tmp_path / "long.yaml")
    output = tmp_path / "generated"
    assert (
        main(
            [
                "--config",
                str(config_path),
                "--factory",
                f"{module.__name__}:build",
                "--output",
                str(output),
                "--no-resume",
            ]
        )
        == 0
    )
    assert isinstance(seen["config"], LongHorizonConfig)
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "complete"
    assert payload["total_video_frames"] == 3
    assert payload["total_action_steps"] == 3
    assert (output / "RUN.json").is_file()


def test_factory_may_resolve_directly_to_a_job(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
):
    module = types.ModuleType("genet_test_direct_long_job")
    module.job = _job(1)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    config_path = _write_config(tmp_path / "long.yaml")
    output = tmp_path / "direct"
    assert (
        main(
            [
                "--config",
                str(config_path),
                "--factory",
                f"{module.__name__}:job",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["total_video_frames"] == 1


def test_factory_result_must_be_a_long_generation_job(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = types.ModuleType("genet_test_bad_long_factory")
    module.build = lambda: object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    config_path = _write_config(tmp_path / "long.yaml")
    with pytest.raises(TypeError, match="expected LongGenerationJob"):
        main(
            [
                "--config",
                str(config_path),
                "--factory",
                f"{module.__name__}:build",
                "--output",
                str(tmp_path / "bad"),
            ]
        )
