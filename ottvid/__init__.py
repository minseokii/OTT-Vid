"""OTT-Vid: Optimal Transport Temporal Token Compression for Video LLMs.

Training-free, plug-and-play compression for Qwen2.5-VL / LLaVA-OV / LLaVA-Video.

Pipeline:
    1. Spatial compression (per-frame, saliency-weighted token coverage).
    2. OT temporal merge (Sinkhorn) with terminal-LOO mass, position-aligned α,
       many-to-one matching, λ=1.0 cost, anchor RoPE — all fixed.

User-facing knobs:
    total_retention_ratio (r), temporal_share (γ),
    ot_mass_tau (τ_m), ot_budget_temperature (τ_b),
    ot_cost_threshold (θ), ot_sinkhorn_eps, ot_sinkhorn_iters.

The per-frame spatial retention is derived as r_s = r ** (1 - γ).

Usage (Qwen2.5-VL):
    from ottvid import apply_ottvid
    model = apply_ottvid(model, total_retention_ratio=0.10,
                              temporal_share=0.3,
                              enable_temporal=True,
                              ot_mass_tau=0.3, ot_budget_temperature=0.3,
                              ot_cost_threshold=0.3,
                              ot_sinkhorn_eps=0.01, ot_sinkhorn_iters=200)

Usage (LLaVA-Video / LLaVA-OneVision):
    from ottvid import apply_ottvid_llava_vid
    model = apply_ottvid_llava_vid(model, total_retention_ratio=0.10,
                                        temporal_share=0.3,
                                        enable_temporal=True,
                                        ot_mass_tau=0.3,
                                        ot_budget_temperature=0.3,
                                        ot_cost_threshold=0.3)
"""

import types
from typing import Optional

import torch.nn as nn

from .config import PipelineConfig
from .backbones.llava_vid import apply_ottvid_llava_vid  # noqa: F401

__version__ = "0.1.0"


def apply_ottvid(
    model: nn.Module,
    # Global
    total_retention_ratio: float = 0.25,
    # Spatial / Temporal split (paper notation γ; r_s = r ** (1 - γ))
    temporal_share: float = 0.3,
    spatial_retention: Optional[float] = None,  # advanced override; None → derive from γ
    # Temporal
    enable_temporal: bool = False,
    ot_budget_temperature: float = 0.2,
    ot_sinkhorn_eps: float = 0.1,
    ot_sinkhorn_iters: int = 50,
    ot_mass_tau: Optional[float] = None,
    ot_cost_threshold: Optional[float] = None,
) -> nn.Module:
    """Apply OTT-Vid compression to a Qwen2.5-VL model.

    Monkey-patches model forward methods. Training-free, plug-and-play.
    """
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VLForConditionalGeneration,
        Qwen2_5_VLModel,
        Qwen2_5_VLVisionBlock,
        Qwen2_5_VisionTransformerPretrainedModel,
    )

    from .backbones.qwen2_5_vl import (
        Qwen2_5_VLModel_forward,
        Qwen2_5_VLModel_get_video_features,
        Qwen2_5_VLVisionAttention_forward,
        Qwen2_5_VLVisionBlock_forward,
        Qwen2_5_VisionTransformerPretrainedModel_forward,
        Qwen2_5_VLForConditionalGeneration_generate,
    )

    if type(model) is not Qwen2_5_VLForConditionalGeneration:
        raise NotImplementedError(
            f"apply_ottvid does not support {type(model).__name__}; "
            f"use apply_ottvid_llava_vid for LLaVA-OV / LLaVA-Video."
        )

    # Patch vision encoder (extract attention weights as saliency)
    attn_cls = type(model.model.visual.blocks[0].attn)
    attn_cls.forward = Qwen2_5_VLVisionAttention_forward
    Qwen2_5_VLVisionBlock.forward = Qwen2_5_VLVisionBlock_forward
    Qwen2_5_VisionTransformerPretrainedModel.forward = (
        Qwen2_5_VisionTransformerPretrainedModel_forward
    )
    Qwen2_5_VLModel.get_video_features = Qwen2_5_VLModel_get_video_features
    Qwen2_5_VLModel.forward = Qwen2_5_VLModel_forward
    Qwen2_5_VLForConditionalGeneration.generate_ori = (
        Qwen2_5_VLForConditionalGeneration.generate
    )
    Qwen2_5_VLForConditionalGeneration.generate = (
        Qwen2_5_VLForConditionalGeneration_generate
    )

    config = PipelineConfig(
        total_retention_ratio=total_retention_ratio,
        temporal_share=temporal_share,
        spatial_retention=spatial_retention,
        enable_temporal=enable_temporal,
        ot_budget_temperature=ot_budget_temperature,
        ot_sinkhorn_eps=ot_sinkhorn_eps,
        ot_sinkhorn_iters=ot_sinkhorn_iters,
        ot_mass_tau=ot_mass_tau,
        ot_cost_threshold=ot_cost_threshold,
    )

    setattr(model, "pipeline_config", config)
    setattr(model.model, "pipeline_config", config)

    # Fix accelerate hooks (device_map="auto")
    def _fix_hook(module, func):
        if hasattr(module, '_old_forward'):
            module._old_forward = types.MethodType(func, module)

    _fix_hook(model.model.visual, Qwen2_5_VisionTransformerPretrainedModel_forward)
    _fix_hook(model.model, Qwen2_5_VLModel_forward)
    for blk in model.model.visual.blocks:
        _fix_hook(blk, Qwen2_5_VLVisionBlock_forward)
        _fix_hook(blk.attn, Qwen2_5_VLVisionAttention_forward)

    return model


__all__ = ["apply_ottvid", "apply_ottvid_llava_vid", "PipelineConfig"]

