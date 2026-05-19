"""Unified config for the full compression pipeline: spatial + temporal + PPE.

Fixed (no longer user-tunable):
  spatial_method   = "spat"               (saliency-weighted token coverage)
  ot_mass_source   = "loo"                (terminal Leave-One-Out)
  ot_alpha_method  = "position_aligned"   (same-position cosine)
  ot_matching_mode = "many_to_one"        (one anchor absorbs many)
  ot_locality_scale (λ)  = 1.0            (cost = α·feat + (1-α)·cent)
  temporal_ppe_mode = "anchor"            (RoPE = anchor frame position)
"""

from typing import Optional
from dataclasses import dataclass, field

import torch


@dataclass
class PipelineConfig:
    # ========================
    # Global
    # ========================
    total_retention_ratio: float = field(default=0.25)  # final token survival rate (r_total)

    # ========================
    # Spatial compression (Stage 1)  — fixed to spat
    # ========================
    # Primary knob (paper notation, γ): split between spatial/temporal stages.
    #   spatial_retention = total_retention_ratio ** (1 - temporal_share)
    #   temporal_retention = total_retention_ratio ** temporal_share
    # γ=0 → pure spatial (no temporal merge needed); γ=1 → pure temporal.
    temporal_share: float = field(default=0.3)
    # Override: if set, takes precedence over temporal_share derivation.
    spatial_retention: Optional[float] = field(default=None)  # r_s; None → derive from γ
    # r_t = total_retention_ratio / spatial_retention (computed at runtime)

    def __post_init__(self):
        if self.spatial_retention is None:
            self.spatial_retention = (
                self.total_retention_ratio ** (1.0 - self.temporal_share)
            )

    # ========================
    # Temporal compression (Stage 2)  — OT with fixed mass=LOO, α=position_aligned,
    # matching=many_to_one, λ=1.0
    # ========================
    enable_temporal: bool = field(default=False)
    ot_budget_temperature: float = field(default=0.2)
    ot_sinkhorn_eps: float = field(default=0.1)
    ot_sinkhorn_iters: int = field(default=50)
    # Mass softmax temperature: None = uniform; float τ → m_i ∝ exp(-s_i/τ)
    ot_mass_tau: Optional[float] = field(default=None)
    # Strong-prune threshold on pair cost; None disables.
    ot_cost_threshold: Optional[float] = field(default=None)

    # ========================
    # Runtime metadata (filled during forward)
    # ========================
    visual_token_start_index: Optional[int] = field(default=None)
    visual_token_length: Optional[int] = field(default=None)
    temporal_ranges: Optional[torch.Tensor] = field(default=None)  # (N, 2) [frame_min, frame_max]

