"""Unified GenET training entry point."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from genet.config import load_config
from genet.training.distributed import destroy_distributed, initialize_distributed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="genet-train",
        description="Train the cross-embodiment video/action generator.",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, help="Override data.manifest")
    parser.add_argument("--output-dir", type=Path, help="Override checkpoint.output_dir")
    parser.add_argument("--max-steps", type=int, help="Override train.max_steps")
    parser.add_argument("--resume", type=Path, help="Exact resume from a committed run checkpoint")
    parser.add_argument("--warm-start", type=Path, help="Load model weights without trainer state")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config/data/model construction without optimizer steps",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    if args.manifest is not None:
        config.data.manifest = str(args.manifest)
    if args.output_dir is not None:
        config.checkpoint.output_dir = str(args.output_dir)
    if args.max_steps is not None:
        config.train.max_steps = args.max_steps
    if args.resume is not None:
        config.checkpoint.resume = str(args.resume)
        config.checkpoint.warm_start = None
    if args.warm_start is not None:
        config.checkpoint.warm_start = str(args.warm_start)
        config.checkpoint.resume = None

    if config.model.backend == "cosmos3_edge":
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        config.validate(world_size=world_size)
        try:
            from genet.training.cosmos import run_cosmos_training

            run_cosmos_training(config, dry_run=args.dry_run)
        finally:
            destroy_distributed()
    else:
        context = initialize_distributed()
        try:
            config.validate(world_size=context.world_size)
            from genet.training.standalone import run_standalone_training

            run_standalone_training(config, context, dry_run=args.dry_run)
        finally:
            destroy_distributed()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
