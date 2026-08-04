import json
import math
from pathlib import Path

import pytest
import torch

from genet.inference.long_horizon import (
    ChunkOutput,
    ContinuityConfig,
    GenerationJournal,
    LongHorizonConfig,
    LongHorizonGenerator,
    QualityReport,
    RecoveryConfig,
    RunIdentity,
    TensorWindowSource,
)
from genet.integrations.cosmos_data import CosmosActionProcessingRecord
from genet.integrations.cosmos_long_horizon import (
    CosmosLongHorizonSampler,
    build_long_horizon_cosmos_batch,
)


IDENTITY = RunIdentity(
    source_id="source-sha",
    model_id="model-sha",
    reference_id="reference-sha",
    code_revision="test",
)


class _TimelineSampler:
    def __init__(self, *, interrupt_on_call: int | None = None) -> None:
        self.requests = []
        self.interrupt_on_call = interrupt_on_call

    def __call__(self, request):
        self.requests.append(request)
        if self.interrupt_on_call == len(self.requests):
            raise KeyboardInterrupt("simulated crash")
        cfg = request.config
        indexes = torch.arange(
            request.window_start_frame,
            request.window_start_frame + cfg.chunk_frames,
            dtype=torch.float32,
        )
        values = (indexes.remainder(200) / 100.0 - 1.0).view(1, -1, 1, 1)
        video = values.expand(request.source_video.shape[0], -1, *request.source_video.shape[2:]).clone()
        action = indexes.view(-1, 1).expand(-1, 2).clone()
        if cfg.action_alignment == "transition":
            action = action[:-1]
        if request.context_video is not None:
            video[:, : request.context_frames] = request.context_video
        if request.context_action is not None:
            action[: request.context_action_steps] = request.context_action
        return ChunkOutput(video, action, {"window_start": request.window_start_frame})


def _config(**kwargs) -> LongHorizonConfig:
    values = {
        "chunk_frames": 81,
        "overlap_frames": 17,
        "temporal_compression_factor": 4,
        "store_video_dtype": "float32",
        "recovery": RecoveryConfig(
            max_attempts_per_chunk=3,
            rollback_depth=1,
            rollback_chunks=1,
            max_total_rollbacks=4,
        ),
        "continuity": ContinuityConfig(
            max_video_prefix_mae=1.0e-6,
            max_action_prefix_relative_mae=1.0e-6,
        ),
    }
    values.update(kwargs)
    result = LongHorizonConfig(**values)
    result.validate()
    return result


def _source(frames: int, *, alignment: str = "frame") -> TensorWindowSource:
    video = torch.zeros(3, frames, 2, 2, dtype=torch.uint8)
    action_steps = frames if alignment == "frame" else max(frames - 1, 0)
    action = torch.zeros(action_steps, 2)
    return TensorWindowSource(video, action, alignment=alignment)


@pytest.mark.parametrize("frames", [1, 17, 81, 82, 145, 200])
def test_long_horizon_has_no_gaps_or_duplicate_overlap(tmp_path: Path, frames: int):
    config = _config()
    sampler = _TimelineSampler()
    source = _source(frames)
    output_dir = tmp_path / f"run-{frames}"
    result = LongHorizonGenerator(config, sampler).generate(
        source,
        output_dir=output_dir,
        identity=IDENTITY,
    )

    journal = GenerationJournal(
        output_dir,
        config=config,
        identity=IDENTITY,
        total_frames=frames,
        resume=True,
    )
    video, action = journal.materialize()
    assert video.shape == (3, frames, 2, 2)
    assert action.shape == (frames, 2)
    assert torch.equal(action[:, 0], torch.arange(frames, dtype=torch.float32))
    expected_chunks = 1 if frames <= 81 else 1 + math.ceil((frames - 81) / 64)
    assert result.chunks == expected_chunks
    assert result.status == "complete"
    starts = [request.window_start_frame for request in sampler.requests]
    assert starts == [0] + [64 * index for index in range(1, expected_chunks)]
    if len(sampler.requests) > 1:
        assert sampler.requests[1].condition_video_latent_indexes == tuple(range(5))
        assert sampler.requests[1].condition_action_indexes == tuple(range(17))


def test_transition_aligned_actions_stitch_to_video_minus_one(tmp_path: Path):
    config = _config(action_alignment="transition")
    sampler = _TimelineSampler()
    source = _source(145, alignment="transition")
    output_dir = tmp_path / "transition"
    LongHorizonGenerator(config, sampler).generate(
        source, output_dir=output_dir, identity=IDENTITY
    )
    journal = GenerationJournal(
        output_dir,
        config=config,
        identity=IDENTITY,
        total_frames=145,
        resume=True,
    )
    _, action = journal.materialize()
    assert action.shape == (144, 2)
    assert torch.equal(action[:, 0], torch.arange(144, dtype=torch.float32))
    assert sampler.requests[1].condition_action_indexes == tuple(range(16))


@pytest.mark.parametrize("frames", [1, 2, 81, 82, 145])
def test_transition_alignment_supports_one_frame_overlap(
    tmp_path: Path, frames: int
):
    config = _config(action_alignment="transition", overlap_frames=1)
    sampler = _TimelineSampler()
    output_dir = tmp_path / f"transition-one-{frames}"
    LongHorizonGenerator(config, sampler).generate(
        _source(frames, alignment="transition"),
        output_dir=output_dir,
        identity=IDENTITY,
    )
    journal = GenerationJournal(
        output_dir,
        config=config,
        identity=IDENTITY,
        total_frames=frames,
        resume=True,
    )
    _, action = journal.materialize()
    assert action.shape == (max(frames - 1, 0), 2)
    assert torch.equal(action[:, 0], torch.arange(max(frames - 1, 0), dtype=torch.float32))
    if frames > 81:
        assert sampler.requests[1].context_action is not None
        assert sampler.requests[1].context_action.shape == (0, 2)
        assert sampler.requests[1].condition_action_indexes == ()


def test_rejected_future_chunk_rolls_back_video_and_action_together(tmp_path: Path):
    config = _config()
    sampler = _TimelineSampler()
    default = LongHorizonGenerator(config, sampler).evaluator

    def evaluator(request, output):
        if request.window_start_frame == 128 and request.rollback_count == 0:
            return QualityReport(False, {"forced": 1.0}, ("forced_seam_failure",))
        return default(request, output)

    output_dir = tmp_path / "rollback"
    result = LongHorizonGenerator(config, sampler, evaluator=evaluator).generate(
        _source(200), output_dir=output_dir, identity=IDENTITY
    )
    state = json.loads((output_dir / "RUN.json").read_text(encoding="utf-8"))
    assert result.rollbacks == 1
    assert len(state["superseded_chunks"]) == 1
    assert (output_dir / state["superseded_chunks"][0]["path"]).is_file()
    assert [request.window_start_frame for request in sampler.requests].count(128) == 4
    journal = GenerationJournal(
        output_dir,
        config=config,
        identity=IDENTITY,
        total_frames=200,
        resume=True,
    )
    _, action = journal.materialize()
    assert torch.equal(action[:, 0], torch.arange(200, dtype=torch.float32))


def test_resume_continues_from_atomic_chunk_boundary(tmp_path: Path):
    config = _config()
    output_dir = tmp_path / "resume"
    interrupted = _TimelineSampler(interrupt_on_call=3)
    with pytest.raises(KeyboardInterrupt, match="simulated crash"):
        LongHorizonGenerator(config, interrupted).generate(
            _source(200), output_dir=output_dir, identity=IDENTITY
        )
    state = json.loads((output_dir / "RUN.json").read_text(encoding="utf-8"))
    assert state["committed_frames"] == 145
    assert state["status"] == "running"

    resumed = _TimelineSampler()
    result = LongHorizonGenerator(config, resumed).generate(
        _source(200), output_dir=output_dir, identity=IDENTITY, resume=True
    )
    assert result.status == "complete"
    assert resumed.requests[0].window_start_frame == 128
    journal = GenerationJournal(
        output_dir,
        config=config,
        identity=IDENTITY,
        total_frames=200,
        resume=True,
    )
    _, action = journal.materialize()
    assert torch.equal(action[:, 0], torch.arange(200, dtype=torch.float32))


def test_config_rejects_temporally_misaligned_overlap():
    with pytest.raises(ValueError, match="overlap_frames"):
        LongHorizonConfig(overlap_frames=16).validate()
    with pytest.raises(ValueError, match="rollback_chunks"):
        LongHorizonConfig(
            recovery=RecoveryConfig(rollback_depth=0, rollback_chunks=1)
        ).validate()
    with pytest.raises(ValueError, match="transition-aligned"):
        LongHorizonConfig(
            overlap_frames=0,
            action_alignment="transition",
        ).validate()


def test_failed_regeneration_cannot_cross_finalized_boundary(tmp_path: Path):
    config = _config()
    sampler = _TimelineSampler()
    default = LongHorizonGenerator(config, sampler).evaluator

    def evaluator(request, output):
        if request.window_start_frame == 128 or (
            request.window_start_frame == 64 and request.rollback_count > 0
        ):
            return QualityReport(False, {"forced": 1.0}, ("forced_failure",))
        return default(request, output)

    output_dir = tmp_path / "terminal-finalized"
    with pytest.raises(RuntimeError, match="rollback budget is exhausted"):
        LongHorizonGenerator(config, sampler, evaluator=evaluator).generate(
            _source(200), output_dir=output_dir, identity=IDENTITY
        )
    state = json.loads((output_dir / "RUN.json").read_text(encoding="utf-8"))
    assert state["status"] == "failed"
    assert state["total_rollbacks"] == 1
    assert state["committed_frames"] == 81
    assert state["finalized_frames"] == 81

    resumed = _TimelineSampler()
    with pytest.raises(RuntimeError, match="terminally failed"):
        LongHorizonGenerator(config, resumed).generate(
            _source(200), output_dir=output_dir, identity=IDENTITY, resume=True
        )
    assert resumed.requests == []


def test_started_attempt_consumes_retry_budget_after_resume(tmp_path: Path):
    config = _config(
        recovery=RecoveryConfig(
            max_attempts_per_chunk=3,
            rollback_depth=0,
            rollback_chunks=0,
            max_total_rollbacks=0,
        )
    )
    output_dir = tmp_path / "attempt-budget"
    with pytest.raises(KeyboardInterrupt, match="simulated crash"):
        LongHorizonGenerator(
            config, _TimelineSampler(interrupt_on_call=1)
        ).generate(_source(20), output_dir=output_dir, identity=IDENTITY)

    resumed = _TimelineSampler()

    def reject(_request, _output):
        return QualityReport(False, {}, ("forced_failure",))

    with pytest.raises(RuntimeError, match="rollback budget is exhausted"):
        LongHorizonGenerator(config, resumed, evaluator=reject).generate(
            _source(20), output_dir=output_dir, identity=IDENTITY, resume=True
        )
    assert len(resumed.requests) == 2
    state = json.loads((output_dir / "RUN.json").read_text(encoding="utf-8"))
    assert len(state["attempts"]) == 3
    assert state["status"] == "failed"


def test_resume_rejects_chunk_path_escape(tmp_path: Path):
    config = _config()
    output_dir = tmp_path / "path-escape"
    LongHorizonGenerator(config, _TimelineSampler()).generate(
        _source(1), output_dir=output_dir, identity=IDENTITY
    )
    manifest_path = output_dir / "RUN.json"
    state = json.loads(manifest_path.read_text(encoding="utf-8"))
    state["active_chunks"][0]["path"] = "../outside.npz"
    manifest_path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ValueError, match="active chunk"):
        GenerationJournal(
            output_dir,
            config=config,
            identity=IDENTITY,
            total_frames=1,
            resume=True,
        )


def test_first_window_target_action_width_is_enforced(tmp_path: Path):
    config = _config(
        recovery=RecoveryConfig(
            max_attempts_per_chunk=1,
            rollback_depth=0,
            rollback_chunks=0,
            max_total_rollbacks=0,
        )
    )
    with pytest.raises(RuntimeError, match="rollback budget is exhausted"):
        LongHorizonGenerator(
            config,
            _TimelineSampler(),
            target_action_dim=3,
        ).generate(
            _source(1), output_dir=tmp_path / "action-width", identity=IDENTITY
        )


def test_non_json_metadata_is_rejected_before_chunk_publish(tmp_path: Path):
    class BadMetadataSampler(_TimelineSampler):
        def __call__(self, request):
            output = super().__call__(request)
            output.metadata = {"not_finite": float("nan")}
            return output

    output_dir = tmp_path / "bad-metadata"
    with pytest.raises(TypeError, match="finite JSON"):
        LongHorizonGenerator(_config(), BadMetadataSampler()).generate(
            _source(1), output_dir=output_dir, identity=IDENTITY
        )
    assert not list((output_dir / "chunks").glob("*.npz"))


def _template() -> dict:
    return {
        "video": torch.zeros(3, 81, 2, 2, dtype=torch.uint8),
        "action": torch.zeros(81, 64),
        "domain_id": torch.tensor(1),
        "ai_caption": "pick object",
        "image_size": torch.tensor([2.0, 2.0, 2.0, 2.0]),
        "conditioning_fps": torch.tensor(16.0),
        "raw_action_dim": torch.tensor(2),
        "action_processing_record": CosmosActionProcessingRecord(2),
        "source_video": torch.zeros(3, 81, 2, 2, dtype=torch.uint8),
        "source_action": torch.zeros(81, 64),
        "source_domain_id": torch.tensor(0),
        "source_image_size": torch.tensor([2.0, 2.0, 2.0, 2.0]),
        "source_conditioning_fps": torch.tensor(16.0),
        "source_raw_action_dim": torch.tensor(2),
        "reference_video": torch.zeros(3, 81, 2, 2, dtype=torch.uint8),
        "reference_action": torch.zeros(81, 64),
        "reference_domain_id": torch.tensor(1),
        "reference_image_size": torch.tensor([2.0, 2.0, 2.0, 2.0]),
        "reference_conditioning_fps": torch.tensor(16.0),
        "reference_raw_action_dim": torch.tensor(2),
    }


def _continuation_request():
    config = _config()
    source = _source(145).read_window(64, config)
    from genet.inference.long_horizon import ChunkRequest

    return ChunkRequest(
        window_start_frame=64,
        committed_frames=81,
        context_video=torch.full((3, 17, 2, 2), 0.25),
        context_action=torch.arange(34, dtype=torch.float32).reshape(17, 2),
        source_video=source.video,
        source_action=source.action[:, :2],
        new_video_frames=64,
        new_action_steps=64,
        valid_window_frames=81,
        valid_window_action_steps=81,
        attempt_id=0,
        seed=123,
        rollback_count=0,
        config=config,
    )


def test_cosmos_long_batch_sets_joint_clean_prefix():
    request = _continuation_request()
    batch = build_long_horizon_cosmos_batch(request, _template())
    assert batch["video"][0].shape == (1, 3, 81, 2, 2)
    assert torch.equal(batch["video"][0][0, :, :17], request.context_video)
    assert torch.equal(batch["action"][0][:17, :2], request.context_action)
    assert torch.count_nonzero(batch["action"][0][:, 2:]) == 0
    plan = batch["sequence_plan"][0]
    assert plan.condition_frame_indexes_vision == list(range(5))
    assert plan.condition_frame_indexes_action == list(range(17))
    assert plan.action_start_frame_offset == 0


def test_cosmos_transition_batch_handles_zero_action_context():
    from genet.inference.long_horizon import ChunkRequest

    config = _config(action_alignment="transition", overlap_frames=1)
    source = _source(82, alignment="transition").read_window(80, config)
    request = ChunkRequest(
        window_start_frame=80,
        committed_frames=81,
        context_video=torch.zeros(3, 1, 2, 2),
        context_action=torch.empty(0, 2),
        source_video=source.video,
        source_action=source.action,
        new_video_frames=1,
        new_action_steps=1,
        valid_window_frames=2,
        valid_window_action_steps=1,
        attempt_id=0,
        seed=456,
        rollback_count=0,
        config=config,
    )
    template = _template()
    template["action"] = torch.zeros(80, 64)
    template["source_action"] = torch.zeros(80, 64)
    batch = build_long_horizon_cosmos_batch(request, template)
    assert batch["action"][0].shape == (80, 64)
    assert torch.count_nonzero(batch["action"][0]) == 0
    plan = batch["sequence_plan"][0]
    assert plan.condition_frame_indexes_vision == [0]
    assert plan.condition_frame_indexes_action == []
    assert plan.action_start_frame_offset == 1


class _FakeCosmosModel:
    fixed_step_sampler = None

    def __init__(self) -> None:
        self.batch = None
        self.kwargs = None

    def generate_samples_from_batch(self, batch, **kwargs):
        self.batch = batch
        self.kwargs = kwargs
        return {
            "vision": [torch.zeros(1, 2, 21, 1, 1)],
            "action": [torch.zeros(81, 2)],
        }

    def decode(self, _latent):
        return torch.zeros(1, 3, 81, 2, 2)


def test_cosmos_sampler_uses_list_seed_and_decodes_one_window():
    model = _FakeCosmosModel()
    model.tensor_kwargs = {"device": torch.device("cpu"), "dtype": torch.float64}
    output = CosmosLongHorizonSampler(model, _template())(_continuation_request())
    assert output.video.shape == (3, 81, 2, 2)
    assert output.action.shape == (81, 2)
    assert model.kwargs["seed"] == [123]
    assert model.kwargs["n_sample"] == 1
    assert model.batch["video"][0].dtype == torch.float64
    assert model.batch["source_video"][0].dtype == torch.uint8
