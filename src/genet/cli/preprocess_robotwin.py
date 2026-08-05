"""Command-line entry point for the fixed RoboTwin-v1 MDS schema."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from genet.data.preprocess import PreprocessConfig
from genet.data.robotwin import (
    RoboTwinPreprocessConfig,
    load_robotwin_contract,
    preprocess_robotwin_mds,
)


def _default_schema_path() -> Path:
    relative = Path("configs/data/robotwin_v1.json")
    for candidate in (
        Path(__file__).resolve().parents[3] / relative,
        Path("/opt/genet") / relative,
    ):
        if candidate.is_file():
            return candidate
    return Path(__file__).resolve().parents[3] / relative


DEFAULT_SCHEMA = _default_schema_path()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="genet-preprocess-robotwin",
        description=(
            "Validate a local RoboTwin-v1 MosaicML Streaming cache and export "
            "fixed-length cross-embodiment GenET pairs. Run this once before "
            "creating the immutable processed-data cluster lock."
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("/mnt/nvme/mds-cache/robotwin_v1"),
        help="RoboTwin root containing manifest.json and train/val",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--schema",
        type=Path,
        default=DEFAULT_SCHEMA,
        help="Versioned structural schema contract",
    )
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument(
        "--mds-index-fps",
        type=float,
        required=True,
        help=(
            "Declared cadence of the integer MDS index. RoboTwin MDS has no "
            "timestamps; 16 means one row per GenET 16 Hz index, not a recovered "
            "physical clock."
        ),
    )
    parser.add_argument(
        "--source-embodiment",
        action="append",
        dest="source_embodiments",
        help="Repeat to restrict source embodiments; default is all five",
    )
    parser.add_argument(
        "--target-embodiment",
        action="append",
        dest="target_embodiments",
        help="Repeat to restrict target embodiments; default is all five",
    )
    parser.add_argument("--camera", choices=("head", "left", "right"), default="head")
    parser.add_argument(
        "--window-policy",
        choices=("episode_start", "sliding"),
        default="episode_start",
        help=(
            "Use episode_start for production. sliding is experimental until "
            "cross-embodiment task-phase retiming is reviewed."
        ),
    )
    parser.add_argument("--clip-stride-native-frames", type=int, default=81)
    parser.add_argument(
        "--reference-policy",
        choices=("different_task", "any_task"),
        default="different_task",
    )
    parser.add_argument("--reference-seed", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--sample-fps", type=float, default=16.0)
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--action-dim", type=int, default=64)
    parser.add_argument(
        "--action-resample", choices=("linear", "nearest"), default="linear"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help=(
            "Cap written samples only after every requested aggregate stream has "
            "been fully scanned and validated"
        ),
    )
    parser.add_argument(
        "--require-bidirectional",
        action="store_true",
        help=(
            "Production guard: require every contract embodiment in both roles, "
            "different_task references, every reverse direction, and no --max-samples"
        ),
    )
    parser.add_argument(
        "--skip-action-window-validation",
        action="store_true",
        help=(
            "Skip state/window and adjacent-window checks. This weakens schema "
            "validation and is not recommended for release preprocessing."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the resolved semantic configuration and exit",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    contract = load_robotwin_contract(args.schema)
    preprocess = PreprocessConfig(
        num_frames=args.num_frames,
        sample_fps=args.sample_fps,
        height=args.height,
        width=args.width,
        action_dim=args.action_dim,
        action_resample=args.action_resample,
        short_policy="drop",
        reference_seed=args.reference_seed,
        default_video_fps=None,
        default_action_fps=None,
    )
    config = RoboTwinPreprocessConfig(
        mds_index_fps=args.mds_index_fps,
        preprocess=preprocess,
        camera=args.camera,
        window_policy=args.window_policy,
        clip_stride_native_frames=args.clip_stride_native_frames,
        reference_policy=args.reference_policy,
        validate_action_windows=not args.skip_action_window_validation,
    )
    if args.print_config:
        print(
            json.dumps(
                {
                    "contract": asdict(contract),
                    "root": str(args.root.expanduser().resolve()),
                    "split": args.split,
                    "source_embodiments": args.source_embodiments,
                    "target_embodiments": args.target_embodiments,
                    "require_bidirectional": args.require_bidirectional,
                    "config": asdict(config),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    report = preprocess_robotwin_mds(
        args.root,
        args.output,
        contract=contract,
        split=args.split,
        config=config,
        source_embodiments=args.source_embodiments,
        target_embodiments=args.target_embodiments,
        max_samples=args.max_samples,
        require_bidirectional=args.require_bidirectional,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "manifest": str(report.manifest),
                "index": str(report.index),
                "stats": str(report.stats),
                "written": report.written,
                "dropped": report.dropped,
                "failed": report.failed,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
