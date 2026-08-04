from __future__ import annotations

import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = (
    "build_hyperbolic_image.sh",
    "check_hyperbolic_host.sh",
    "launch_cluster_ssh.sh",
    "launch_roce.sh",
    "preflight_roce.sh",
    "pull_hyperbolic_image.sh",
    "run_hyperbolic_container.sh",
)
IMAGE_REF = "registry.example/genet@sha256:" + "a" * 64


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def _fake_host_tools(tmp_path: Path) -> tuple[dict[str, str], Path]:
    binary_dir = tmp_path / "bin"
    binary_dir.mkdir()
    docker_log = tmp_path / "docker.log"
    _write_executable(
        binary_dir / "docker",
        """#!/usr/bin/env bash
set -eu
if [[ "${1:-}" == container && "${2:-}" == inspect ]]; then
  exit 1
fi
printf '%q ' "$@" >> "${FAKE_DOCKER_LOG}"
printf '\n' >> "${FAKE_DOCKER_LOG}"
if [[ "${1:-}" == image && "${2:-}" == inspect && "${3:-}" == --format ]]; then
  printf 'sha256:%064d\n' 0
fi
""",
    )
    real_find = subprocess.run(
        ["sh", "-c", "command -v find"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _write_executable(
        binary_dir / "find",
        f"""#!/usr/bin/env bash
set -eu
if [[ "${{1:-}}" == /dev/infiniband ]]; then
  printf '/dev/null\\0'
  exit 0
fi
exec {real_find!s} "$@"
""",
    )
    cache = tmp_path / "cache"
    run_root = tmp_path / "run"
    cache.mkdir()
    run_root.mkdir()
    environment = dict(os.environ)
    environment.update(
        {
            "PATH": f"{binary_dir}:{environment['PATH']}",
            "FAKE_DOCKER_LOG": str(docker_log),
            "NODE_RANK": "0",
            "GENET_NODE_CACHE": str(cache),
            "GENET_NODE_RUN_ROOT": str(run_root),
        }
    )
    return environment, docker_log


def test_hyperbolic_shell_scripts_parse() -> None:
    for script_name in SCRIPTS:
        subprocess.run(
            ["bash", "-n", str(ROOT / "scripts" / script_name)],
            check=True,
        )


def test_container_runner_accepts_inherited_environment(tmp_path: Path) -> None:
    environment, docker_log = _fake_host_tools(tmp_path)
    subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "run_hyperbolic_container.sh"),
            "-",
            IMAGE_REF,
            "python",
            "-V",
        ],
        check=True,
        env=environment,
    )
    invocation = docker_log.read_text(encoding="utf-8")
    assert "run" in invocation
    assert "--pull missing" in invocation
    assert "ai.genet.launch-id=manual" in invocation
    assert "readonly" in invocation
    assert "/dev/null:/dev/null" in invocation
    assert IMAGE_REF in invocation
    assert "python -V" in invocation


def test_single_controller_dry_run_fans_out_locally(tmp_path: Path) -> None:
    environment, docker_log = _fake_host_tools(tmp_path)
    hostfile = tmp_path / "hosts.txt"
    hostfile.write_text("local\n", encoding="utf-8")
    env_file = tmp_path / "job.env"
    env_file.write_text(
        "\n".join(
            (
                "export NNODES=1",
                "export NPROC_PER_NODE=8",
                "export MASTER_ADDR=127.0.0.1",
                "export MASTER_PORT=29500",
                "export PREFLIGHT_PORT=29499",
                f"export GENET_NODE_CACHE={environment['GENET_NODE_CACHE']}",
                f"export GENET_NODE_RUN_ROOT={environment['GENET_NODE_RUN_ROOT']}",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    logs = tmp_path / "logs"
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "launch_cluster_ssh.sh"),
            "--hosts",
            str(hostfile),
            "--env",
            str(env_file),
            "--image",
            IMAGE_REF,
            "--config",
            "configs/experiments/stage1_control_32gpu.yaml",
            "--run-id",
            "test-launch",
            "--log-dir",
            str(logs),
            "--skip-host-check",
            "--dry-run-only",
            "--",
            "--manifest",
            f"{environment['GENET_NODE_CACHE']}/manifest.jsonl",
        ],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
        timeout=20,
    )
    assert "Dry-run-only launch completed" in result.stdout
    invocations = docker_log.read_text(encoding="utf-8").splitlines()
    assert len(invocations) == 4
    assert sum("image inspect" in line for line in invocations) == 2
    assert any("scripts/preflight_roce.sh" in line for line in invocations)
    assert any("scripts/launch_roce.sh" in line and "--dry-run" in line for line in invocations)
    assert (logs / "image-pull-rank-0.log").is_file()
    assert (logs / "preflight-rank-0.log").is_file()
    assert (logs / "dry-run-rank-0.log").is_file()


def test_training_launcher_rejects_inherited_hf_token(tmp_path: Path) -> None:
    environment, _ = _fake_host_tools(tmp_path)
    environment["HF_TOKEN"] = "must-not-cross-ssh"
    hostfile = tmp_path / "hosts.txt"
    hostfile.write_text("local\n", encoding="utf-8")
    env_file = tmp_path / "job.env"
    env_file.write_text(
        "\n".join(
            (
                "export NNODES=1",
                "export NPROC_PER_NODE=8",
                "export MASTER_ADDR=127.0.0.1",
                "export MASTER_PORT=29500",
                "export PREFLIGHT_PORT=29499",
                f"export GENET_NODE_CACHE={environment['GENET_NODE_CACHE']}",
                f"export GENET_NODE_RUN_ROOT={environment['GENET_NODE_RUN_ROOT']}",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "launch_cluster_ssh.sh"),
            "--hosts",
            str(hostfile),
            "--env",
            str(env_file),
            "--image",
            IMAGE_REF,
            "--config",
            "configs/experiments/stage1_control_32gpu.yaml",
            "--log-dir",
            str(tmp_path / "logs"),
            "--dry-run-only",
        ],
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 2
    assert "unset HF_TOKEN before training" in result.stderr
