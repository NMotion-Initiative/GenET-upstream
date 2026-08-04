import datetime
import json
import multiprocessing
import os
import socket
from pathlib import Path

import pytest
import torch

from genet.inference.long_horizon import (
    ChunkOutput,
    LongHorizonConfig,
    LongHorizonGenerator,
    RecoveryConfig,
    RunIdentity,
    TensorWindowSource,
)


pytestmark = pytest.mark.skipif(
    os.environ.get("GENET_RUN_DISTRIBUTED_TESTS") != "1",
    reason="set GENET_RUN_DISTRIBUTED_TESTS=1 on a host that permits Gloo sockets",
)


def _distributed_worker(
    rank: int,
    world_size: int,
    init_file: str,
    output_dir: str,
    result_dir: str,
    mode: str,
) -> None:
    loopback = next(
        (name for _index, name in socket.if_nameindex() if name.startswith("lo")),
        None,
    )
    if loopback is not None:
        os.environ.setdefault("GLOO_SOCKET_IFNAME", loopback)
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=10),
    )
    try:
        frames = 3 if mode == "length_mismatch" and rank == 1 else 2
        source = TensorWindowSource(
            torch.zeros(3, frames, 2, 2, dtype=torch.uint8),
            torch.zeros(frames, 2),
        )
        config = LongHorizonConfig(
            chunk_frames=5,
            overlap_frames=1,
            temporal_compression_factor=4,
            store_video_dtype="float32",
            recovery=RecoveryConfig(
                max_attempts_per_chunk=1,
                rollback_depth=0,
                rollback_chunks=0,
                max_total_rollbacks=0,
            ),
        )

        def sampler(request):
            metadata = {"bad": float("nan")} if mode == "commit_failure" else {}
            return ChunkOutput(
                torch.zeros(3, request.config.chunk_frames, 2, 2),
                torch.zeros(request.config.chunk_action_steps, 2),
                metadata,
            )

        generator = LongHorizonGenerator(config, sampler, target_action_dim=2)
        identity = RunIdentity("source", "model", "reference", code_revision="test")
        try:
            generator.generate(
                source,
                output_dir=output_dir,
                identity=identity,
                resume=False,
            )
            result = {"ok": True, "message": ""}
        except Exception as error:
            result = {"ok": False, "message": str(error)}
        Path(result_dir, f"rank-{rank}.json").write_text(
            json.dumps(result), encoding="utf-8"
        )
    finally:
        torch.distributed.destroy_process_group()


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("length_mismatch", "differs across ranks"),
        ("journal_failure", "journal-initialize"),
        ("commit_failure", "candidate-commit"),
    ],
)
def test_distributed_failures_reach_every_rank(
    tmp_path: Path, mode: str, expected: str
):
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("torch.distributed Gloo is unavailable")
    output_dir = tmp_path / "output"
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    if mode == "journal_failure":
        output_dir.mkdir()
        (output_dir / "RUN.json").write_text("{}", encoding="utf-8")

    context = multiprocessing.get_context("spawn")
    processes = [
        context.Process(
            target=_distributed_worker,
            args=(
                rank,
                2,
                str(tmp_path / "gloo-init"),
                str(output_dir),
                str(result_dir),
                mode,
            ),
        )
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=15)
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            pytest.fail("distributed long-generation worker hung")
        assert process.exitcode == 0

    results = [
        json.loads((result_dir / f"rank-{rank}.json").read_text(encoding="utf-8"))
        for rank in range(2)
    ]
    assert all(not result["ok"] for result in results)
    assert all(expected in result["message"] for result in results)
