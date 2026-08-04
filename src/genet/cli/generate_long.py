"""Command-line entry point for transactional long-horizon generation."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from genet.inference.long_horizon import (
    LongGenerationJob,
    LongHorizonConfig,
    LongHorizonGenerator,
    load_long_horizon_config,
)


def _factory_spec(value: str) -> str:
    module_name, separator, attribute_name = value.partition(":")
    if not separator or not module_name.strip() or not attribute_name.strip():
        raise argparse.ArgumentTypeError(
            "factory must use the form 'module:function'"
        )
    return f"{module_name.strip()}:{attribute_name.strip()}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="genet-generate-long",
        description=(
            "Generate a long, synchronized target video/action trajectory with "
            "transactional rolling context and rollback."
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="YAML mapping accepted by LongHorizonConfig",
    )
    parser.add_argument(
        "--factory",
        type=_factory_spec,
        required=True,
        help=(
            "Import path module:function. The referenced value may be a "
            "LongGenerationJob or a callable that constructs one."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Transactional run directory",
    )
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help="Resume a compatible incomplete or completed run",
    )
    resume.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Require a new output run",
    )
    parser.set_defaults(resume=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the long-horizon plan without importing the factory",
    )
    return parser


def _load_factory(spec: str) -> Any:
    module_name, attribute_path = spec.split(":", 1)
    module = importlib.import_module(module_name)
    value: Any = module
    for component in attribute_path.split("."):
        if not component:
            raise ValueError(f"invalid empty attribute in factory spec {spec!r}")
        try:
            value = getattr(value, component)
        except AttributeError as error:
            raise AttributeError(
                f"factory {spec!r} has no attribute component {component!r}"
            ) from error
    return value


def _invoke_factory(factory: Callable[..., Any], config: LongHorizonConfig) -> Any:
    """Call a job factory without hiding a TypeError raised inside the factory."""

    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return factory(config)

    for args, kwargs in (
        ((), {"config": config}),
        ((config,), {}),
        ((), {}),
    ):
        try:
            signature.bind(*args, **kwargs)
        except TypeError:
            continue
        return factory(*args, **kwargs)
    raise TypeError(
        "long-generation factory must accept a LongHorizonConfig as a keyword or "
        "positional argument, or accept no arguments"
    )


def _build_job(spec: str, config: LongHorizonConfig) -> LongGenerationJob:
    candidate = _load_factory(spec)
    if isinstance(candidate, LongGenerationJob):
        return candidate
    if not callable(candidate):
        raise TypeError(
            f"factory {spec!r} resolved to {type(candidate).__name__}, expected "
            "LongGenerationJob or callable"
        )
    result = _invoke_factory(candidate, config)
    if not isinstance(result, LongGenerationJob):
        raise TypeError(
            f"factory {spec!r} returned {type(result).__name__}, expected "
            "LongGenerationJob"
        )
    return result


def _plan_payload(
    *,
    config: LongHorizonConfig,
    factory: str,
    output: Path,
    resume: bool,
) -> dict[str, Any]:
    return {
        "status": "dry-run",
        "factory": factory,
        "output_dir": str(output),
        "resume": resume,
        "chunk_frames": config.chunk_frames,
        "overlap_frames": config.overlap_frames,
        "overlap_latent_frames": config.overlap_latent_frames,
        "stride_frames": config.stride_frames,
        "chunk_action_steps": config.chunk_action_steps,
        "config": config.as_dict(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_long_horizon_config(args.config)
    output = args.output.expanduser().resolve()
    effective_resume = config.recovery.resume if args.resume is None else bool(args.resume)

    if args.dry_run:
        print(
            json.dumps(
                _plan_payload(
                    config=config,
                    factory=args.factory,
                    output=output,
                    resume=effective_resume,
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    job = _build_job(args.factory, config)
    generator = LongHorizonGenerator(
        config,
        job.sampler,
        evaluator=job.evaluator,
        target_action_dim=job.target_action_dim,
    )
    result = generator.generate(
        job.source,
        output_dir=output,
        identity=job.identity,
        resume=args.resume,
    )
    print(
        json.dumps(
            {
                "status": result.status,
                "output_dir": str(result.output_dir),
                "total_video_frames": result.total_video_frames,
                "total_action_steps": result.total_action_steps,
                "chunks": result.chunks,
                "rollbacks": result.rollbacks,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
