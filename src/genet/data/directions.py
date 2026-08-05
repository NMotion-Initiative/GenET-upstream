"""Direction coverage accounting for processed cross-embodiment pairs."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

_MAX_REVERSE_EXAMPLES = 20
_SHA256_LENGTH = 64


def _required_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _clip_token(value: Any, *, field: str, required: bool = False) -> str:
    if value is None:
        if required:
            raise ValueError(f"{field} is required for strict bidirectional data")
        return "<missing>"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number when present")
    try:
        numeric = float(value)
    except OverflowError as exc:
        raise ValueError(f"{field} must be finite") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{field} must be finite")
    if numeric == 0.0:
        numeric = 0.0
    return format(numeric, ".17g")


def _content_hash(
    stream: Mapping[str, Any], *, field: str, required: bool
) -> str:
    value = stream.get("content_sha256")
    if value is None and not required:
        return ""
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field}.content_sha256 must be a lowercase SHA-256")
    return value


def _sample_signature(
    entry: Mapping[str, Any], *, strict: bool = False
) -> tuple[str, ...]:
    source = entry.get("source")
    target = entry.get("target_gt")
    if not isinstance(source, Mapping) or not isinstance(target, Mapping):
        raise ValueError("source and target_gt metadata must be objects")
    metadata = entry.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("entry metadata must be an object")
    pair_identity = metadata.get("pair_identity", "")
    if pair_identity is None:
        pair_identity = ""
    if not isinstance(pair_identity, str):
        raise ValueError("metadata.pair_identity must be a string when present")
    if strict and not pair_identity:
        raise ValueError(
            "metadata.pair_identity is required for strict bidirectional data"
        )
    if strict:
        reference = entry.get("reference_target")
        if not isinstance(reference, Mapping):
            raise ValueError("reference_target metadata must be an object")
        target_embodiment = _required_text(
            target.get("embodiment"), field="target_gt.embodiment"
        )
        reference_embodiment = _required_text(
            reference.get("embodiment"), field="reference_target.embodiment"
        )
        if reference_embodiment != target_embodiment:
            raise ValueError(
                "reference_target.embodiment must match target_gt.embodiment"
            )
        if metadata.get("reference_policy") != "different_task":
            raise ValueError(
                "metadata.reference_policy must be 'different_task' for strict "
                "bidirectional data"
            )
        target_metadata = target.get("metadata")
        reference_metadata = reference.get("metadata")
        if not isinstance(target_metadata, Mapping) or not isinstance(
            reference_metadata, Mapping
        ):
            raise ValueError(
                "target_gt.metadata and reference_target.metadata must be objects"
            )
        target_task = _required_text(
            target_metadata.get("task"), field="target_gt.metadata.task"
        )
        reference_task = _required_text(
            reference_metadata.get("task"), field="reference_target.metadata.task"
        )
        if target_task == reference_task:
            raise ValueError(
                "reference_target.metadata.task must differ from target_gt.metadata.task"
            )
        _content_hash(reference, field="reference_target", required=True)
    return (
        _required_text(source.get("embodiment"), field="source.embodiment"),
        _required_text(target.get("embodiment"), field="target_gt.embodiment"),
        _required_text(source.get("episode_id"), field="source.episode_id"),
        _required_text(target.get("episode_id"), field="target_gt.episode_id"),
        _clip_token(
            source.get("clip_start"), field="source.clip_start", required=strict
        ),
        _clip_token(
            target.get("clip_start"), field="target_gt.clip_start", required=strict
        ),
        pair_identity,
        _content_hash(source, field="source", required=strict),
        _content_hash(target, field="target_gt", required=strict),
    )


def _reverse_signature(signature: tuple[str, ...]) -> tuple[str, ...]:
    (
        source,
        target,
        source_episode,
        target_episode,
        source_clip,
        target_clip,
        identity,
        source_hash,
        target_hash,
    ) = signature
    return (
        target,
        source,
        target_episode,
        source_episode,
        target_clip,
        source_clip,
        identity,
        target_hash,
        source_hash,
    )


def _expected_embodiment_tuple(
    values: Iterable[str] | None,
) -> tuple[str, ...] | None:
    if values is None:
        return None
    if isinstance(values, (str, bytes)):
        raise ValueError("expected_embodiments must be an iterable of names")
    result = tuple(values)
    if not result:
        raise ValueError("expected_embodiments cannot be empty")
    if any(not isinstance(value, str) or not value for value in result):
        raise ValueError("expected_embodiments must contain non-empty strings")
    if len(set(result)) != len(result):
        raise ValueError("expected_embodiments contains duplicate names")
    return result


def summarize_pair_directions(
    entries: Iterable[Mapping[str, Any]],
    *,
    expected_embodiments: Iterable[str] | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Summarize role use and prove exact reciprocal sample coverage.

    A reciprocal record swaps the Source/Target embodiments, episode IDs, and
    clip starts while retaining the optional order-independent pair identity.
    References are deliberately excluded from the reciprocal identity because
    each direction uses its own current-Target reference. Strict mode still
    binds that Reference to a content hash and the different-task policy.
    """

    expected = _expected_embodiment_tuple(expected_embodiments)
    signatures: Counter[tuple[str, ...]] = Counter()
    directions: Counter[tuple[str, str]] = Counter()
    source_roles: Counter[str] = Counter()
    target_roles: Counter[str] = Counter()
    for entry in entries:
        signature = _sample_signature(entry, strict=strict)
        signatures[signature] += 1
        source, target = signature[:2]
        directions[(source, target)] += 1
        source_roles[source] += 1
        target_roles[target] += 1

    errors: list[dict[str, Any]] = []
    if not signatures:
        errors.append({"code": "empty_manifest"})
    observed_embodiments = set(source_roles) | set(target_roles)
    if expected is not None:
        expected_set = set(expected)
        missing = sorted(expected_set - observed_embodiments)
        unexpected = sorted(observed_embodiments - expected_set)
        if missing:
            errors.append({"code": "missing_embodiments", "embodiments": missing})
        if unexpected:
            errors.append(
                {"code": "unexpected_embodiments", "embodiments": unexpected}
            )
        missing_directions = [
            f"{source}->{target}"
            for source in expected
            for target in expected
            if source != target and directions.get((source, target), 0) == 0
        ]
        if missing_directions:
            errors.append(
                {
                    "code": "missing_directions",
                    "directions": missing_directions,
                }
            )
    unordered_directions = {
        tuple(sorted((source, target))) for source, target in directions
    }
    for source, target in sorted(unordered_directions):
        count = directions.get((source, target), 0)
        reverse_count = directions.get((target, source), 0)
        if source == target:
            errors.append(
                {
                    "code": "same_embodiment_pair",
                    "source": source,
                    "target": target,
                    "samples": count,
                }
            )
            continue
        if count != reverse_count:
            errors.append(
                {
                    "code": "direction_count_mismatch",
                    "source": source,
                    "target": target,
                    "samples": count,
                    "reverse_samples": reverse_count,
                }
            )

    embodiments = sorted(observed_embodiments)
    for embodiment in embodiments:
        source_count = source_roles.get(embodiment, 0)
        target_count = target_roles.get(embodiment, 0)
        if source_count != target_count:
            errors.append(
                {
                    "code": "role_marginal_mismatch",
                    "embodiment": embodiment,
                    "source_samples": source_count,
                    "target_samples": target_count,
                }
            )

    reverse_mismatches = 0
    reverse_examples: list[dict[str, Any]] = []
    canonical_signatures = {
        min(signature, _reverse_signature(signature)) for signature in signatures
    }
    for signature in sorted(canonical_signatures):
        reverse = _reverse_signature(signature)
        count = signatures.get(signature, 0)
        reverse_count = signatures.get(reverse, 0)
        if count == reverse_count:
            continue
        reverse_mismatches += abs(count - reverse_count)
        if len(reverse_examples) < _MAX_REVERSE_EXAMPLES:
            reverse_examples.append(
                {
                    "source": signature[0],
                    "target": signature[1],
                    "source_episode": signature[2],
                    "target_episode": signature[3],
                    "source_clip_start": signature[4],
                    "target_clip_start": signature[5],
                    "pair_identity": signature[6] or None,
                    "samples": count,
                    "reverse_samples": reverse_count,
                }
            )
    if reverse_mismatches:
        errors.append(
            {
                "code": "reverse_sample_mismatch",
                "mismatched_records": reverse_mismatches,
                "examples": reverse_examples,
            }
        )

    if strict:
        duplicate_records = sum(count - 1 for count in signatures.values() if count > 1)
        if duplicate_records:
            duplicate_examples = []
            for signature, count in sorted(signatures.items()):
                if count <= 1:
                    continue
                duplicate_examples.append(
                    {
                        "source": signature[0],
                        "target": signature[1],
                        "source_episode": signature[2],
                        "target_episode": signature[3],
                        "source_clip_start": signature[4],
                        "target_clip_start": signature[5],
                        "pair_identity": signature[6],
                        "samples": count,
                    }
                )
                if len(duplicate_examples) >= _MAX_REVERSE_EXAMPLES:
                    break
            errors.append(
                {
                    "code": "duplicate_directed_records",
                    "duplicate_records": duplicate_records,
                    "examples": duplicate_examples,
                }
            )

    return {
        "directed_samples": sum(signatures.values()),
        "unique_pair_windows": len(canonical_signatures),
        "embodiments": embodiments,
        "expected_embodiments": list(expected) if expected is not None else None,
        "direction_counts": {
            f"{source}->{target}": count
            for (source, target), count in sorted(directions.items())
        },
        "source_role_counts": dict(sorted(source_roles.items())),
        "target_role_counts": dict(sorted(target_roles.items())),
        "bidirectional_complete": not errors,
        "errors": errors,
    }


def require_bidirectional_pairs(
    entries: Iterable[Mapping[str, Any]],
    *,
    expected_embodiments: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Return the direction summary or reject an incomplete/imbalanced manifest."""

    summary = summarize_pair_directions(
        entries,
        expected_embodiments=expected_embodiments,
        strict=True,
    )
    if not summary["bidirectional_complete"]:
        raise ValueError(
            "processed manifest is not a complete bidirectional cross-embodiment "
            f"dataset: {summary['errors']}"
        )
    return summary


__all__ = ["require_bidirectional_pairs", "summarize_pair_directions"]
