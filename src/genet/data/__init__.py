"""Data contracts and preprocessing for GenET cross-embodiment training."""

from .actions import ActionSeries, load_action_series, resample_actions
from .dataset import ProcessedPairDataset, collate_pairs, pair_collate_fn
from .preprocess import (
    PROCESSED_FORMAT_VERSION,
    PreprocessConfig,
    PreprocessReport,
    preprocess_manifest,
)
from .reference import NoReferenceCandidateError, StatelessReferencePool, stable_index
from .robotwin import (
    ROBOTWIN_ADAPTER_VERSION,
    RoboTwinContract,
    RoboTwinPreprocessConfig,
    load_robotwin_contract,
    preprocess_robotwin_mds,
    validate_robotwin_root,
)
from .sampling import ShortSequenceError, make_time_grid
from .schema import (
    RAW_PAIR_JSON_SCHEMA,
    SCHEMA_VERSION,
    EpisodeRef,
    PairRecord,
    SchemaError,
    iter_raw_manifest,
    read_raw_manifest,
)
from .video import DecodedVideo, VideoDecodeError, decode_video, resize_center_crop

__all__ = [
    "ActionSeries",
    "DecodedVideo",
    "EpisodeRef",
    "NoReferenceCandidateError",
    "PROCESSED_FORMAT_VERSION",
    "PairRecord",
    "PreprocessConfig",
    "PreprocessReport",
    "ProcessedPairDataset",
    "RAW_PAIR_JSON_SCHEMA",
    "ROBOTWIN_ADAPTER_VERSION",
    "RoboTwinContract",
    "RoboTwinPreprocessConfig",
    "SCHEMA_VERSION",
    "SchemaError",
    "ShortSequenceError",
    "StatelessReferencePool",
    "VideoDecodeError",
    "collate_pairs",
    "decode_video",
    "iter_raw_manifest",
    "load_action_series",
    "load_robotwin_contract",
    "make_time_grid",
    "pair_collate_fn",
    "preprocess_manifest",
    "preprocess_robotwin_mds",
    "read_raw_manifest",
    "resample_actions",
    "resize_center_crop",
    "stable_index",
    "validate_robotwin_root",
]
