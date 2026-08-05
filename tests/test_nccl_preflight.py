from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from genet.cli import nccl_preflight
from genet.cli.nccl_preflight import PreflightConfig

ROOT = Path(__file__).resolve().parents[1]


def _config(**overrides: int) -> PreflightConfig:
    values = {
        "expected_nnodes": 4,
        "expected_local_world_size": 8,
        "buffer_mib": 64,
        "warmup_iterations": 3,
        "iterations": 3,
        "timeout_seconds": 180,
    }
    values.update(overrides)
    return PreflightConfig(**values)


def _set_torchrun_environment(monkeypatch: pytest.MonkeyPatch, *, rank: int = 9) -> None:
    monkeypatch.setenv("RANK", str(rank))
    monkeypatch.setenv("LOCAL_RANK", str(rank % 8))
    monkeypatch.setenv("WORLD_SIZE", "32")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")


def test_launch_environment_requires_exact_4x8_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_torchrun_environment(monkeypatch)
    environment = nccl_preflight._validate_launch_environment(_config())
    assert environment.rank == 9
    assert environment.local_rank == 1
    assert environment.world_size == 32

    monkeypatch.setenv("WORLD_SIZE", "31")
    with pytest.raises(ValueError, match="WORLD_SIZE does not match"):
        nccl_preflight._validate_launch_environment(_config())

    monkeypatch.setenv("WORLD_SIZE", "32")
    monkeypatch.setenv("LOCAL_RANK", "2")
    with pytest.raises(ValueError, match="rank layout is not contiguous"):
        nccl_preflight._validate_launch_environment(_config())


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--expected-nnodes", "0", "--expected-local-world-size", "8"], "must be positive"),
        (["--expected-nnodes", "4", "--expected-local-world-size", "-8"], "base-10 integer"),
        (
            [
                "--expected-nnodes",
                "4",
                "--expected-local-world-size",
                "8",
                "--buffer-mib",
                "0",
            ],
            "must be positive",
        ),
        (
            [
                "--expected-nnodes",
                "4",
                "--expected-local-world-size",
                "8",
                "--iterations",
                "0",
            ],
            "must be positive",
        ),
        (
            [
                "--expected-nnodes",
                "4",
                "--expected-local-world-size",
                "8",
                "--timeout-seconds",
                "9",
            ],
            "at least 10",
        ),
    ],
)
def test_parser_strictly_rejects_invalid_parameters(
    arguments: list[str],
    message: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit, match="2"):
        nccl_preflight._parser().parse_args(arguments)
    assert message in capsys.readouterr().err


def test_rank_zero_report_uses_slowest_rank_per_iteration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(nccl_preflight.torch, "__version__", "test-torch")
    config = _config(expected_nnodes=1, expected_local_world_size=2, buffer_mib=1)
    ranks = [
        {
            "rank": rank,
            "local_rank": rank,
            "node_id": "node-0",
            "hostname": "host-0",
            "device": {
                "name": "gpu",
                "capability": [9, 0],
                "total_memory_bytes": 80 * 1024**3,
            },
        }
        for rank in range(2)
    ]
    report = nccl_preflight._build_report(
        config,
        ranks,
        [[1.0, 4.0, 3.0], [2.0, 3.0, 6.0]],
        expected_sum=3.0,
    )
    assert report["event"] == "nccl_preflight_passed"
    assert report["world_size"] == 2
    assert report["correctness"] == {"expected_sum": 3.0, "max_abs_error": 0.0}
    assert report["timing_ms"]["per_iteration_rank_max"] == [2.0, 4.0, 6.0]
    assert report["timing_ms"]["median"] == 4.0
    assert report["algorithmic_bandwidth_GBps"] == pytest.approx(0.262144)
    assert report["bus_bandwidth_GBps"] == pytest.approx(0.262144)
    json.dumps(report)


def test_run_destroys_initialized_process_group_after_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_torchrun_environment(monkeypatch)
    state = {"initialized": False, "destroyed": False, "cache_emptied": False}

    monkeypatch.setattr(nccl_preflight.dist, "is_initialized", lambda: state["initialized"])
    monkeypatch.setattr(nccl_preflight.dist, "is_nccl_available", lambda: True)

    def initialize(**_: object) -> None:
        state["initialized"] = True

    def destroy() -> None:
        state["destroyed"] = True
        state["initialized"] = False

    monkeypatch.setattr(nccl_preflight.dist, "init_process_group", initialize)
    monkeypatch.setattr(nccl_preflight.dist, "destroy_process_group", destroy)
    monkeypatch.setattr(nccl_preflight.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(nccl_preflight.torch.cuda, "device_count", lambda: 8)
    monkeypatch.setattr(nccl_preflight.torch.cuda, "set_device", lambda _device: None)
    monkeypatch.setattr(
        nccl_preflight.torch.cuda,
        "empty_cache",
        lambda: state.__setitem__("cache_emptied", True),
    )
    monkeypatch.setattr(
        nccl_preflight,
        "_rank_metadata",
        lambda _environment: (_ for _ in ()).throw(RuntimeError("metadata failed")),
    )

    with pytest.raises(RuntimeError, match="metadata failed"):
        nccl_preflight._run(_config())
    assert state == {"initialized": False, "destroyed": True, "cache_emptied": True}


def test_nccl_shell_launches_one_process_per_gpu(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    torchrun = fake_bin / "torchrun"
    torchrun.write_text(
        "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\"\n",
        encoding="utf-8",
    )
    torchrun.chmod(0o755)
    environment = dict(os.environ)
    environment.update(
        {
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "NNODES": "4",
            "NPROC_PER_NODE": "8",
            "NODE_RANK": "2",
            "MASTER_ADDR": "10.1.2.3",
            "NCCL_PREFLIGHT_PORT": "29498",
            "GENET_NCCL_PREFLIGHT_BUFFER_MIB": "128",
            "GENET_NCCL_PREFLIGHT_WARMUP_ITERATIONS": "2",
            "GENET_NCCL_PREFLIGHT_ITERATIONS": "7",
            "GENET_NCCL_PREFLIGHT_TIMEOUT_SECONDS": "240",
        }
    )
    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / "preflight_nccl.sh")],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    arguments = result.stdout.splitlines()
    assert "--nnodes=4" in arguments
    assert "--nproc-per-node=8" in arguments
    assert "--node-rank=2" in arguments
    assert "--master-addr=10.1.2.3" in arguments
    assert "--master-port=29498" in arguments
    assert "genet.cli.nccl_preflight" in arguments
    assert arguments[arguments.index("--buffer-mib") + 1] == "128"
    assert arguments[arguments.index("--iterations") + 1] == "7"
    assert arguments[arguments.index("--timeout-seconds") + 1] == "240"


def test_nccl_shell_rejects_invalid_values() -> None:
    environment = dict(os.environ)
    environment.update(
        {
            "NNODES": "4",
            "NPROC_PER_NODE": "8",
            "NODE_RANK": "4",
            "MASTER_ADDR": "10.1.2.3",
            "NCCL_PREFLIGHT_PORT": "29498",
        }
    )
    result = subprocess.run(
        ["bash", str(ROOT / "scripts" / "preflight_nccl.sh")],
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 2
    assert "NODE_RANK must be in [0, NNODES)" in result.stderr


def test_nccl_shell_parses() -> None:
    subprocess.run(
        ["bash", "-n", str(ROOT / "scripts" / "preflight_nccl.sh")],
        check=True,
    )
