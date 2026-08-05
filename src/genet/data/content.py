"""Deterministic content identities for processed vision/action streams."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

STREAM_CONTENT_HASH_VERSION = "genet.stream-content-sha256/v1"


def _canonical_array(value: Any, *, dtype: np.dtype[Any]) -> np.ndarray:
    """Return C-contiguous bytes with an architecture-independent dtype."""

    return np.ascontiguousarray(np.asarray(value, dtype=dtype))


def stream_content_sha256(
    *,
    video: Any,
    actions: Any,
    action_mask: Any,
    frame_mask: Any,
) -> str:
    """Hash one processed stream, including field names, dtypes, and shapes.

    The canonical representation fixes video to uint8, actions to little-endian
    float32, and masks to bool before hashing.  Length-prefixed JSON headers keep
    field and shape boundaries unambiguous while raw C-order bytes avoid format-
    or compression-dependent identities.
    """

    arrays = (
        ("video", _canonical_array(video, dtype=np.dtype("u1"))),
        ("actions", _canonical_array(actions, dtype=np.dtype("<f4"))),
        ("action_mask", _canonical_array(action_mask, dtype=np.dtype("?"))),
        ("frame_mask", _canonical_array(frame_mask, dtype=np.dtype("?"))),
    )
    digest = hashlib.sha256()
    digest.update(STREAM_CONTENT_HASH_VERSION.encode("ascii"))
    digest.update(b"\0")
    for name, array in arrays:
        header = json.dumps(
            {
                "dtype": array.dtype.str,
                "name": name,
                "shape": list(array.shape),
            },
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
        digest.update(len(header).to_bytes(8, "little", signed=False))
        digest.update(header)
        payload = memoryview(array).cast("B")
        digest.update(len(payload).to_bytes(8, "little", signed=False))
        digest.update(payload)
    return digest.hexdigest()


__all__ = ["STREAM_CONTENT_HASH_VERSION", "stream_content_sha256"]
