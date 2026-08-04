"""Checkpoint archive utility."""

from __future__ import annotations

import argparse
from pathlib import Path

from genet.training.checkpoint import (
    consolidate_node_archives,
    create_node_manifest,
    verify_committed_checkpoint,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GenET checkpoint verification/consolidation")
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest = subparsers.add_parser("manifest", help="hash one node's local DCP files")
    manifest.add_argument("--checkpoint-dir", required=True)
    manifest.add_argument("--node-rank", required=True, type=int)
    manifest.add_argument("--output", required=True)

    verify = subparsers.add_parser("verify", help="verify a committed standalone or consolidated checkpoint")
    verify.add_argument("--checkpoint-dir", required=True)

    consolidate = subparsers.add_parser("consolidate", help="merge node_XX DCP uploads and commit atomically")
    consolidate.add_argument("--archive-dir", required=True)
    consolidate.add_argument("--output-dir", required=True)
    consolidate.add_argument("--expected-nodes", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "manifest":
        output = create_node_manifest(args.checkpoint_dir, args.node_rank, args.output)
        print(output)
    elif args.command == "verify":
        path = Path(args.checkpoint_dir)
        manifest = verify_committed_checkpoint(path)
        print(f"verified {path}: {len(manifest)} files")
    elif args.command == "consolidate":
        output = consolidate_node_archives(
            args.archive_dir,
            args.output_dir,
            expected_nodes=args.expected_nodes,
        )
        print(output)


if __name__ == "__main__":
    main()

