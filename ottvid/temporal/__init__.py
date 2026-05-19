"""OT temporal merge — Sinkhorn-based content-adaptive cross-frame compression."""
from .ot_merge import (
    ot_temporal_merge,
    OTMergeConfig,
    OTMergeOutput,
    compute_pair_alpha_from_pre,
)

__all__ = [
    "ot_temporal_merge",
    "OTMergeConfig",
    "OTMergeOutput",
    "compute_pair_alpha_from_pre",
]

