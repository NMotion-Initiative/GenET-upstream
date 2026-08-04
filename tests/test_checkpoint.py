import json
from pathlib import Path

import pytest
import torch

from genet.training.checkpoint import (
    CheckpointManager,
    TrainerState,
    consolidate_node_archives,
    create_node_manifest,
    verify_committed_checkpoint,
)


def test_transactional_checkpoint_round_trip(tmp_path: Path) -> None:
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    manager = CheckpointManager(tmp_path)
    checkpoint = manager.save(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        trainer_state=TrainerState(step=3, epoch=1, batches_in_epoch=2),
        config={"name": "test"},
    )
    verify_committed_checkpoint(checkpoint)
    state = manager.load(checkpoint, model=model, optimizer=optimizer, scheduler=scheduler)
    assert state.step == 3


def test_dcp_node_consolidation_requires_every_node(tmp_path: Path) -> None:
    # Match the documented transport/final naming pair. Consolidation's own
    # transaction directory must not collide with the upload directory.
    archive = tmp_path / "consolidated.incomplete"
    for rank in range(2):
        files = archive / f"node_{rank:02d}" / "files"
        files.mkdir(parents=True)
        (files / f"__{rank}_0.distcp").write_bytes(f"rank-{rank}".encode())
        if rank == 0:
            (files / ".metadata").write_bytes(b"metadata")
        create_node_manifest(files, rank, archive / f"node_{rank:02d}" / "NODE_MANIFEST.json")
        (archive / f"node_{rank:02d}" / "NODE_DONE").write_text("done\n", encoding="utf-8")

    output = consolidate_node_archives(archive, tmp_path / "consolidated", expected_nodes=2)
    assert (output / "COMMITTED").is_file()
    manifest = json.loads((output / "MANIFEST.json").read_text(encoding="utf-8"))
    assert "__0_0.distcp" in manifest
    assert "__1_0.distcp" in manifest


def test_dcp_consolidation_rejects_manifest_path_traversal(tmp_path: Path) -> None:
    node = tmp_path / "archive" / "node_00"
    files = node / "files"
    files.mkdir(parents=True)
    (files / ".metadata").write_bytes(b"metadata")
    payload = {
        "node_rank": 0,
        "checkpoint": "iter",
        "files": [{"path": "../../escape.distcp", "size": 0, "sha256": ""}],
    }
    (node / "NODE_MANIFEST.json").write_text(json.dumps(payload), encoding="utf-8")
    (node / "NODE_DONE").write_text("done\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unsafe checkpoint manifest path"):
        consolidate_node_archives(
            tmp_path / "archive",
            tmp_path / "consolidated",
            expected_nodes=1,
        )
