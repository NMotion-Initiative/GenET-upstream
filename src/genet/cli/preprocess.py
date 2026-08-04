"""Command-line entry point for GenET dataset preprocessing."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Sequence

from genet.data.preprocess import PreprocessConfig, preprocess_manifest
from genet.data.schema import RAW_PAIR_JSON_SCHEMA


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="genet-preprocess",
        description=(
            "Convert raw source/target vision-action JSONL pairs into fixed-shape "
            "compressed samples."
        ),
    )
    parser.add_argument("--manifest", type=Path, help="Raw pair JSONL manifest")
    parser.add_argument("--output", type=Path, help="Processed output directory")
    parser.add_argument(
        "--config",
        type=Path,
        help="JSON config; accepts either the preprocess object or a preprocess key",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--on-error",
        choices=("raise", "skip"),
        default="raise",
        help="Whether non-short input failures abort the run",
    )
    parser.add_argument("--num-frames", type=int)
    parser.add_argument("--sample-fps", type=float)
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--action-dim", type=int)
    parser.add_argument("--reference-seed", type=int)
    parser.add_argument("--short-policy", choices=("drop", "pad"))
    parser.add_argument("--action-resample", choices=("linear", "nearest"))
    parser.add_argument(
        "--disable-wan-frame-validation",
        action="store_true",
        help="Allow frame counts other than offset + stride*N",
    )
    parser.add_argument(
        "--print-raw-schema",
        action="store_true",
        help="Print the v1 raw-pair JSON Schema and exit",
    )
    return parser


def _load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return asdict(PreprocessConfig())
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError("config root must be a JSON object")
    values = raw.get("preprocess", raw)
    if not isinstance(values, dict):
        raise ValueError("config preprocess section must be a JSON object")
    return dict(values)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.print_raw_schema:
        print(json.dumps(RAW_PAIR_JSON_SCHEMA, indent=2, sort_keys=True))
        return 0
    if args.manifest is None or args.output is None:
        _parser().error("--manifest and --output are required unless printing the schema")

    values = _load_config(args.config)
    for key in (
        "num_frames",
        "sample_fps",
        "height",
        "width",
        "action_dim",
        "reference_seed",
        "short_policy",
        "action_resample",
    ):
        value = getattr(args, key)
        if value is not None:
            values[key] = value
    if args.disable_wan_frame_validation:
        values["validate_wan_frames"] = False
    config = PreprocessConfig.from_mapping(values)
    report = preprocess_manifest(
        args.manifest,
        args.output,
        config=config,
        overwrite=args.overwrite,
        on_error=args.on_error,
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
