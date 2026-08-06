import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "commit_cluster_dcp_ssh.sh"
IMAGE = f"registry.example/genet@sha256:{'a' * 64}"


def test_commit_script_requires_all_arguments() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "Usage:" in result.stderr


def test_commit_script_rejects_mutable_image(tmp_path: Path) -> None:
    hosts = tmp_path / "hosts"
    hosts.write_text("local\n", encoding="utf-8")
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            str(hosts),
            "registry.example/genet:latest",
            str(tmp_path / "output"),
            str(tmp_path / "committed"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "IMAGE_REF must be immutable" in result.stderr


def test_commit_script_rejects_unsafe_destination(tmp_path: Path) -> None:
    hosts = tmp_path / "hosts"
    hosts.write_text("local\n", encoding="utf-8")
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            str(hosts),
            IMAGE,
            str(tmp_path / "output"),
            "/",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "Refusing unsafe COMMITTED_ROOT" in result.stderr


def test_commit_script_requires_separate_output_and_archive(tmp_path: Path) -> None:
    hosts = tmp_path / "hosts"
    hosts.write_text("local\n", encoding="utf-8")
    output = tmp_path / "output"
    result = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            str(hosts),
            IMAGE,
            str(output),
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "COMMITTED_ROOT must differ" in result.stderr
