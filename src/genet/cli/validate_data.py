"""Validate processed GenET manifests and per-sample NPZ payloads."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from genet.data.content import stream_content_sha256
from genet.data.directions import summarize_pair_directions

_ROLES = ("source", "target", "reference")
_SUFFIXES = ("video", "actions", "action_mask", "frame_mask")
_FORMAT_VERSION = "genet.processed-pair/v1"


def _iter_manifest_entries(path: Path) -> Iterator[Mapping[str, Any]]:
    """Stream lightweight metadata for global direction checks."""

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            entry = json.loads(line)
            if not isinstance(entry, Mapping):
                raise ValueError("manifest entry must be an object")
            yield entry


def _issue(
    issues: list[dict[str, Any]],
    *,
    code: str,
    message: str,
    line: int | None = None,
    sample_id: str | None = None,
    path: Path | None = None,
) -> None:
    item: dict[str, Any] = {"code": code, "message": message}
    if line is not None:
        item["line"] = line
    if sample_id is not None:
        item["id"] = sample_id
    if path is not None:
        item["path"] = str(path)
    issues.append(item)


def _is_numeric(array: np.ndarray) -> bool:
    return np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.bool_
    )


def _check_finite(
    array: np.ndarray,
    *,
    name: str,
    issues: list[dict[str, Any]],
    line: int,
    sample_id: str,
    path: Path,
) -> None:
    if not _is_numeric(array):
        _issue(
            issues,
            code="non_numeric_array",
            message=f"{name} must be numeric, got dtype {array.dtype}",
            line=line,
            sample_id=sample_id,
            path=path,
        )
        return
    if not bool(np.isfinite(array).all()):
        _issue(
            issues,
            code="non_finite_array",
            message=f"{name} contains NaN or infinity",
            line=line,
            sample_id=sample_id,
            path=path,
        )


def _check_stream_content_hashes(
    entry: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    *,
    required: bool,
    issues: list[dict[str, Any]],
    line: int,
    sample_id: str,
    path: Path,
) -> None:
    """Bind manifest stream identities to the decompressed NPZ payload."""

    for metadata_role, array_role in (
        ("source", "source"),
        ("target_gt", "target"),
        ("reference_target", "reference"),
    ):
        required_arrays = tuple(
            f"{array_role}_{suffix}" for suffix in _SUFFIXES
        )
        if any(name not in arrays for name in required_arrays):
            continue
        stream = entry.get(metadata_role)
        expected = stream.get("content_sha256") if isinstance(stream, Mapping) else None
        if expected is None and not required:
            continue
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
        ):
            _issue(
                issues,
                code="stream_content_hash_missing",
                message=(
                    f"{metadata_role}.content_sha256 must be a lowercase SHA-256"
                ),
                line=line,
                sample_id=sample_id,
                path=path,
            )
            continue
        actual = stream_content_sha256(
            video=arrays[f"{array_role}_video"],
            actions=arrays[f"{array_role}_actions"],
            action_mask=arrays[f"{array_role}_action_mask"],
            frame_mask=arrays[f"{array_role}_frame_mask"],
        )
        if actual != expected:
            _issue(
                issues,
                code="stream_content_hash_mismatch",
                message=(
                    f"{metadata_role}.content_sha256 does not match the NPZ payload"
                ),
                line=line,
                sample_id=sample_id,
                path=path,
            )


def _check_binary_mask(
    array: np.ndarray,
    *,
    name: str,
    issues: list[dict[str, Any]],
    line: int,
    sample_id: str,
    path: Path,
) -> None:
    if not _is_numeric(array) or not bool(np.isfinite(array).all()):
        return
    if not bool(np.logical_or(array == 0, array == 1).all()):
        _issue(
            issues,
            code="non_binary_mask",
            message=f"{name} must contain only boolean/0/1 values",
            line=line,
            sample_id=sample_id,
            path=path,
        )


def _check_reference_metadata(
    entry: Mapping[str, Any],
    *,
    issues: list[dict[str, Any]],
    line: int,
    sample_id: str,
) -> None:
    streams = {
        role: entry.get(role)
        for role in ("source", "target_gt", "reference_target")
    }
    invalid = [role for role, value in streams.items() if not isinstance(value, Mapping)]
    if invalid:
        _issue(
            issues,
            code="invalid_pair_metadata",
            message=(
                "source, target_gt, and reference_target metadata must be objects; "
                f"invalid: {', '.join(invalid)}"
            ),
            line=line,
            sample_id=sample_id,
        )
        return
    source = streams["source"]
    target = streams["target_gt"]
    reference = streams["reference_target"]
    assert isinstance(source, Mapping)
    assert isinstance(target, Mapping)
    assert isinstance(reference, Mapping)
    source_embodiment = source.get("embodiment")
    source_episode = source.get("episode_id")
    target_embodiment = target.get("embodiment")
    reference_embodiment = reference.get("embodiment")
    target_episode = target.get("episode_id")
    reference_episode = reference.get("episode_id")
    missing = [
        name
        for name, value in (
            ("source.embodiment", source_embodiment),
            ("source.episode_id", source_episode),
            ("target_gt.embodiment", target_embodiment),
            ("reference_target.embodiment", reference_embodiment),
            ("target_gt.episode_id", target_episode),
            ("reference_target.episode_id", reference_episode),
        )
        if not isinstance(value, str) or not value
    ]
    if missing:
        _issue(
            issues,
            code="incomplete_pair_metadata",
            message=f"missing non-empty metadata fields: {', '.join(missing)}",
            line=line,
            sample_id=sample_id,
        )
        return
    if source_embodiment == target_embodiment:
        _issue(
            issues,
            code="same_embodiment_pair",
            message="source and target_gt must use distinct embodiments",
            line=line,
            sample_id=sample_id,
        )
    if target_embodiment != reference_embodiment:
        _issue(
            issues,
            code="reference_embodiment_mismatch",
            message=(
                f"target embodiment {target_embodiment!r} != reference embodiment "
                f"{reference_embodiment!r}"
            ),
            line=line,
            sample_id=sample_id,
        )
    if target_episode == reference_episode:
        _issue(
            issues,
            code="reference_episode_collision",
            message="reference_target must not use the target_gt episode",
            line=line,
            sample_id=sample_id,
        )


def _validate_arrays(
    arrays: Mapping[str, np.ndarray],
    *,
    expected_t: int | None,
    expected_height: int | None,
    expected_width: int | None,
    expected_action_dim: int | None,
    cosmos: bool,
    issues: list[dict[str, Any]],
    line: int,
    sample_id: str,
    path: Path,
) -> int | None:
    role_shapes: dict[str, dict[str, tuple[int, ...]]] = {}
    observed_t = expected_t
    for role in _ROLES:
        role_shapes[role] = {}
        for suffix in _SUFFIXES:
            name = f"{role}_{suffix}"
            if name not in arrays:
                _issue(
                    issues,
                    code="missing_array",
                    message=f"NPZ is missing {name}",
                    line=line,
                    sample_id=sample_id,
                    path=path,
                )
                continue
            array = arrays[name]
            role_shapes[role][suffix] = tuple(array.shape)
            _check_finite(
                array,
                name=name,
                issues=issues,
                line=line,
                sample_id=sample_id,
                path=path,
            )

        video = arrays.get(f"{role}_video")
        actions = arrays.get(f"{role}_actions")
        action_mask = arrays.get(f"{role}_action_mask")
        frame_mask = arrays.get(f"{role}_frame_mask")

        if video is not None:
            if video.ndim != 4 or video.shape[-1] != 3:
                _issue(
                    issues,
                    code="invalid_video_shape",
                    message=(
                        f"{role}_video must have shape [T,H,W,3], got "
                        f"{tuple(video.shape)}"
                    ),
                    line=line,
                    sample_id=sample_id,
                    path=path,
                )
            else:
                if observed_t is None:
                    observed_t = int(video.shape[0])
                if video.shape[0] != observed_t:
                    _issue(
                        issues,
                        code="temporal_length_mismatch",
                        message=(
                            f"{role}_video T={video.shape[0]} but expected T={observed_t}"
                        ),
                        line=line,
                        sample_id=sample_id,
                        path=path,
                    )
                if expected_height is not None and video.shape[1] != expected_height:
                    _issue(
                        issues,
                        code="height_mismatch",
                        message=(
                            f"{role}_video H={video.shape[1]} but expected "
                            f"H={expected_height}"
                        ),
                        line=line,
                        sample_id=sample_id,
                        path=path,
                    )
                if expected_width is not None and video.shape[2] != expected_width:
                    _issue(
                        issues,
                        code="width_mismatch",
                        message=(
                            f"{role}_video W={video.shape[2]} but expected "
                            f"W={expected_width}"
                        ),
                        line=line,
                        sample_id=sample_id,
                        path=path,
                    )
                if cosmos and video.dtype != np.uint8:
                    _issue(
                        issues,
                        code="cosmos_video_dtype",
                        message=(
                            f"{role}_video must be uint8 for Cosmos worker-side "
                            f"normalization, got {video.dtype}"
                        ),
                        line=line,
                        sample_id=sample_id,
                        path=path,
                    )

        if actions is not None:
            if actions.ndim != 2:
                _issue(
                    issues,
                    code="invalid_action_shape",
                    message=f"{role}_actions must have shape [T,D], got {tuple(actions.shape)}",
                    line=line,
                    sample_id=sample_id,
                    path=path,
                )
            else:
                if observed_t is None:
                    observed_t = int(actions.shape[0])
                if actions.shape[0] != observed_t:
                    _issue(
                        issues,
                        code="temporal_length_mismatch",
                        message=(
                            f"{role}_actions T={actions.shape[0]} but expected T={observed_t}"
                        ),
                        line=line,
                        sample_id=sample_id,
                        path=path,
                    )
                if expected_action_dim is not None and actions.shape[1] != expected_action_dim:
                    _issue(
                        issues,
                        code="action_dim_mismatch",
                        message=(
                            f"{role}_actions D={actions.shape[1]} but expected "
                            f"D={expected_action_dim}"
                        ),
                        line=line,
                        sample_id=sample_id,
                        path=path,
                    )

        if action_mask is not None:
            if actions is None or action_mask.shape != actions.shape:
                _issue(
                    issues,
                    code="invalid_action_mask_shape",
                    message=(
                        f"{role}_action_mask shape {tuple(action_mask.shape)} must equal "
                        f"{role}_actions shape "
                        f"{tuple(actions.shape) if actions is not None else '<missing>'}"
                    ),
                    line=line,
                    sample_id=sample_id,
                    path=path,
                )
            _check_binary_mask(
                action_mask,
                name=f"{role}_action_mask",
                issues=issues,
                line=line,
                sample_id=sample_id,
                path=path,
            )
            if (
                cosmos
                and actions is not None
                and action_mask.shape == actions.shape
                and _is_numeric(action_mask)
                and bool(np.isfinite(action_mask).all())
                and bool(np.logical_or(action_mask == 0, action_mask == 1).all())
            ):
                mask = action_mask.astype(np.bool_, copy=False)
                active = mask.any(axis=0)
                raw_dim = int(active.sum())
                expected = np.arange(mask.shape[1]) < raw_dim
                if raw_dim <= 0 or not bool(np.array_equal(active, expected)):
                    _issue(
                        issues,
                        code="cosmos_action_mask_not_prefix",
                        message=(
                            f"{role}_action_mask must describe a non-empty contiguous "
                            "prefix of real channels"
                        ),
                        line=line,
                        sample_id=sample_id,
                        path=path,
                    )
                elif not bool(mask[:, :raw_dim].all()):
                    _issue(
                        issues,
                        code="cosmos_temporal_action_padding",
                        message=(
                            f"{role}_action_mask contains temporally padded real channels; "
                            "Cosmos production requires short_policy='drop'"
                        ),
                        line=line,
                        sample_id=sample_id,
                        path=path,
                    )

        if frame_mask is not None:
            required_shape = (observed_t,) if observed_t is not None else None
            if frame_mask.ndim != 1 or (
                required_shape is not None and frame_mask.shape != required_shape
            ):
                _issue(
                    issues,
                    code="invalid_frame_mask_shape",
                    message=(
                        f"{role}_frame_mask must have shape [T]"
                        f"{f'={required_shape}' if required_shape is not None else ''}, got "
                        f"{tuple(frame_mask.shape)}"
                    ),
                    line=line,
                    sample_id=sample_id,
                    path=path,
                )
            _check_binary_mask(
                frame_mask,
                name=f"{role}_frame_mask",
                issues=issues,
                line=line,
                sample_id=sample_id,
                path=path,
            )
            if (
                cosmos
                and frame_mask.ndim == 1
                and _is_numeric(frame_mask)
                and bool(np.isfinite(frame_mask).all())
                and not bool(frame_mask.astype(np.bool_, copy=False).all())
            ):
                _issue(
                    issues,
                    code="cosmos_temporal_video_padding",
                    message=(
                        f"{role}_frame_mask contains padding; Cosmos production "
                        "requires short_policy='drop'"
                    ),
                    line=line,
                    sample_id=sample_id,
                    path=path,
                )

    complete_video_shapes = [
        role_shapes[role].get("video")
        for role in _ROLES
        if role_shapes[role].get("video") is not None
    ]
    if len(complete_video_shapes) == len(_ROLES) and len(set(complete_video_shapes)) != 1:
        _issue(
            issues,
            code="stream_video_shape_mismatch",
            message=f"source/target/reference video shapes differ: {complete_video_shapes}",
            line=line,
            sample_id=sample_id,
            path=path,
        )
    complete_action_shapes = [
        role_shapes[role].get("actions")
        for role in _ROLES
        if role_shapes[role].get("actions") is not None
    ]
    if len(complete_action_shapes) == len(_ROLES) and len(set(complete_action_shapes)) != 1:
        _issue(
            issues,
            code="stream_action_shape_mismatch",
            message=f"source/target/reference action shapes differ: {complete_action_shapes}",
            line=line,
            sample_id=sample_id,
            path=path,
        )
    return observed_t


def validate_manifest(
    manifest: str | Path,
    *,
    num_frames: int | None = None,
    height: int | None = None,
    width: int | None = None,
    action_dim: int | None = None,
    cosmos: bool = False,
    require_bidirectional_pairs: bool = False,
    expected_embodiments: Sequence[str] | None = None,
    shard_rank: int = 0,
    shard_world_size: int = 1,
) -> dict[str, Any]:
    """Return a JSON-serializable validation summary without mutating the dataset.

    ``shard_rank``/``shard_world_size`` are an internal production-preflight
    optimization: local ranks on every node divide the expensive NPZ reads. The
    normal CLI validates the complete manifest with their default values.
    """

    manifest_path = Path(manifest).expanduser().resolve()
    if manifest_path.is_dir():
        manifest_path = manifest_path / "manifest.jsonl"
    issues: list[dict[str, Any]] = []
    if not manifest_path.is_file():
        _issue(
            issues,
            code="manifest_missing",
            message="processed manifest does not exist",
            path=manifest_path,
        )
        return {
            "valid": False,
            "manifest": str(manifest_path),
            "samples": 0,
            "valid_samples": 0,
            "invalid_samples": 0,
            "expected_num_frames": num_frames,
            "pair_directions": None,
            "error_count": len(issues),
            "errors": issues,
        }
    for name, value in (
        ("num_frames", num_frames),
        ("height", height),
        ("width", width),
        ("action_dim", action_dim),
    ):
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive")
    if shard_world_size <= 0:
        raise ValueError("shard_world_size must be positive")
    if not 0 <= shard_rank < shard_world_size:
        raise ValueError("shard_rank must satisfy 0 <= shard_rank < shard_world_size")

    direction_summary: dict[str, Any] | None = None
    try:
        direction_summary = summarize_pair_directions(
            _iter_manifest_entries(manifest_path),
            expected_embodiments=expected_embodiments,
            strict=require_bidirectional_pairs,
        )
    except (OSError, ValueError) as exc:
        if require_bidirectional_pairs:
            _issue(
                issues,
                code="bidirectional_metadata_invalid",
                message=f"cannot validate pair directions: {exc}",
                path=manifest_path,
            )
    else:
        if require_bidirectional_pairs and not direction_summary["bidirectional_complete"]:
            _issue(
                issues,
                code="bidirectional_pairs_incomplete",
                message=(
                    "manifest does not contain balanced exact Source/Target reverse "
                    f"records: {direction_summary['errors']}"
                ),
                path=manifest_path,
            )

    total = 0
    entry_index = 0
    invalid_lines: set[int] = set()
    seen_ids: set[str] = set()
    expected_t = num_frames
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            selected = entry_index % shard_world_size == shard_rank
            entry_index += 1
            if not selected:
                continue
            total += 1
            issue_start = len(issues)
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as exc:
                _issue(
                    issues,
                    code="invalid_manifest_json",
                    message=f"invalid JSON: {exc.msg}",
                    line=line_number,
                )
                invalid_lines.add(line_number)
                continue
            if not isinstance(entry, Mapping):
                _issue(
                    issues,
                    code="invalid_manifest_entry",
                    message="manifest entry must be an object",
                    line=line_number,
                )
                invalid_lines.add(line_number)
                continue
            sample_id = entry.get("id")
            if not isinstance(sample_id, str) or not sample_id:
                sample_id = f"<line-{line_number}>"
                _issue(
                    issues,
                    code="missing_sample_id",
                    message="manifest entry requires a non-empty id",
                    line=line_number,
                )
            elif sample_id in seen_ids:
                _issue(
                    issues,
                    code="duplicate_sample_id",
                    message=f"duplicate sample id {sample_id!r}",
                    line=line_number,
                    sample_id=sample_id,
                )
            else:
                seen_ids.add(sample_id)

            version = entry.get("format_version")
            if version != _FORMAT_VERSION:
                _issue(
                    issues,
                    code="unsupported_format_version",
                    message=f"unsupported format_version {version!r}",
                    line=line_number,
                    sample_id=sample_id,
                )
            _check_reference_metadata(
                entry,
                issues=issues,
                line=line_number,
                sample_id=sample_id,
            )

            raw_npz = entry.get("npz")
            if not isinstance(raw_npz, str) or not raw_npz:
                _issue(
                    issues,
                    code="missing_npz_path",
                    message="manifest entry requires a non-empty npz path",
                    line=line_number,
                    sample_id=sample_id,
                )
            else:
                raw_path = Path(raw_npz)
                unsafe = raw_path.is_absolute()
                npz_path = (manifest_path.parent / raw_path).resolve()
                if not unsafe:
                    try:
                        npz_path.relative_to(manifest_path.parent)
                    except ValueError:
                        unsafe = True
                if unsafe:
                    _issue(
                        issues,
                        code="unsafe_npz_path",
                        message="npz path must stay beneath the processed manifest root",
                        line=line_number,
                        sample_id=sample_id,
                        path=npz_path,
                    )
                elif not npz_path.is_file():
                    _issue(
                        issues,
                        code="npz_missing",
                        message="sample NPZ does not exist",
                        line=line_number,
                        sample_id=sample_id,
                        path=npz_path,
                    )
                else:
                    try:
                        with np.load(npz_path, allow_pickle=False) as archive:
                            arrays = {key: archive[key] for key in archive.files}
                        _check_stream_content_hashes(
                            entry,
                            arrays,
                            required=require_bidirectional_pairs,
                            issues=issues,
                            line=line_number,
                            sample_id=sample_id,
                            path=npz_path,
                        )
                        expected_t = _validate_arrays(
                            arrays,
                            expected_t=expected_t,
                            expected_height=height,
                            expected_width=width,
                            expected_action_dim=action_dim,
                            cosmos=cosmos,
                            issues=issues,
                            line=line_number,
                            sample_id=sample_id,
                            path=npz_path,
                        )
                    except Exception as exc:
                        _issue(
                            issues,
                            code="npz_read_error",
                            message=f"cannot read NPZ: {type(exc).__name__}: {exc}",
                            line=line_number,
                            sample_id=sample_id,
                            path=npz_path,
                        )
            if len(issues) != issue_start:
                invalid_lines.add(line_number)

    if total == 0:
        _issue(
            issues,
            code="empty_manifest",
            message="processed manifest contains no samples",
            path=manifest_path,
        )
    invalid_samples = len(invalid_lines)
    return {
        "valid": not issues,
        "manifest": str(manifest_path),
        "samples": total,
        "valid_samples": total - invalid_samples,
        "invalid_samples": invalid_samples,
        "expected_num_frames": expected_t,
        "pair_directions": direction_summary,
        "error_count": len(issues),
        "errors": issues,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="genet-validate-data",
        description="Validate a processed GenET manifest and every referenced NPZ sample.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Processed manifest.jsonl or its containing directory",
    )
    parser.add_argument("--num-frames", type=int, help="Require this fixed T")
    parser.add_argument("--height", type=int)
    parser.add_argument("--width", type=int)
    parser.add_argument("--action-dim", type=int)
    parser.add_argument(
        "--cosmos",
        action="store_true",
        help=(
            "Also require the production Cosmos contract: uint8 video, no "
            "temporal padding, and contiguous action-channel prefixes"
        ),
    )
    parser.add_argument(
        "--require-bidirectional",
        action="store_true",
        help=(
            "Require every cross-embodiment sample to have an exact reverse record "
            "and equal Source/Target role marginals"
        ),
    )
    parser.add_argument(
        "--expected-embodiment",
        action="append",
        dest="expected_embodiments",
        help=(
            "Repeat for every embodiment required in the complete directed graph; "
            "canonical RoboTwin v1 training uses all five"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optionally also write the JSON summary to this path",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = validate_manifest(
            args.manifest,
            num_frames=args.num_frames,
            height=args.height,
            width=args.width,
            action_dim=args.action_dim,
            cosmos=args.cosmos,
            require_bidirectional_pairs=args.require_bidirectional,
            expected_embodiments=args.expected_embodiments,
        )
    except ValueError as exc:
        summary = {
            "valid": False,
            "manifest": str(args.manifest.expanduser().resolve()),
            "samples": 0,
            "valid_samples": 0,
            "invalid_samples": 0,
            "expected_num_frames": args.num_frames,
            "pair_directions": None,
            "error_count": 1,
            "errors": [{"code": "invalid_arguments", "message": str(exc)}],
        }
    rendered = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    print(rendered, end="")
    if args.output is not None:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    return 0 if summary["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
