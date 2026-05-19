"""Monkey-patched forward methods for Qwen2.5-VL: spatial + temporal compression.

Vision encoder patches are identical to spatial_merge/modeling_qwen2_5_vl.py.
Model forward runs: spatial compression → (optional) OT temporal merge → position filter.
"""

import os
from typing import Callable, Optional, Union, Tuple, List

import torch
import torch.nn as nn


def _profile_mark(label: str):
    """Env-gated stage memory probe (PROFILE_FWD=1). Reports peak alloc since last reset."""
    if not os.environ.get("PROFILE_FWD"):
        return
    torch.cuda.synchronize()
    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    cur_mb = torch.cuda.memory_allocated() / 1024**2
    print(f"[FWD-PROFILE] {label:<35} cur={cur_mb:8.1f} MB  peak_in_stage={peak_mb:8.1f} MB", flush=True)
    torch.cuda.reset_peak_memory_stats()
from torch.nn import functional as F
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    apply_multimodal_rotary_pos_emb,
    apply_rotary_pos_emb_vision,
    eager_attention_forward,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2_5_VLModel,
    Qwen2_5_VLVisionAttention,
    Qwen2_5_VLVisionBlock,
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLModelOutputWithPast,
)

from ..config import PipelineConfig


# ============================================================
# Spatial compression: fixed to spatial compression
# ============================================================

def _run_spatial(video_embeds, saliency, video_grid_thw, sms, cfg):
    """Run spatial compression. Returns (merged_tokens, keep_indices).

    Pass-through when spatial_retention=1.0 (no spatial compression).
    """
    if cfg.spatial_retention >= 1.0:
        N = video_embeds.shape[0]
        return video_embeds, torch.arange(N, device=video_embeds.device)

    from ..spatial.facility_location import facility_location_video
    return facility_location_video(
        video_embeds=video_embeds, saliency=saliency,
        video_grid_thw=video_grid_thw, retention_ratio=cfg.spatial_retention,
        spatial_merge_size=sms,
        saliency_mode="wi", redundancy_lambda=0.0,
    )


@torch.no_grad()
def _batched_terminal_leave_one_out_mass(
    pre_tokens: torch.Tensor,   # (T, N, D)
    saliency: torch.Tensor,     # (T, N)
    selected: torch.Tensor,     # (T, K) — per-frame local indices (padded/sorted)
    tau: float,
) -> torch.Tensor:
    """Batched terminal LOO mass across T frames. Returns (T, K) normalized mass."""
    T, N, D = pre_tokens.shape
    K = selected.shape[1]
    t_norm = F.normalize(pre_tokens.float(), p=2, dim=-1)
    sim = torch.bmm(t_norm, t_norm.transpose(1, 2))          # (T, N, N)
    w = saliency.float()
    w = w / (w.sum(dim=-1, keepdim=True) + 1e-8)             # (T, N)
    idx = selected.unsqueeze(1).expand(-1, N, -1)            # (T, N, K)
    S = torch.gather(sim, 2, idx)                            # (T, N, K)
    top2 = S.topk(min(2, K), dim=2)
    best_sim = top2.values[..., 0]                           # (T, N)
    if K >= 2:
        second_sim = top2.values[..., 1]                     # (T, N)
    else:
        second_sim = torch.zeros_like(best_sim)
    best_idx = top2.indices[..., 0]                          # (T, N)
    diff = (best_sim - second_sim).clamp_min_(0) * w         # (T, N)
    contribs = torch.zeros(T, K, device=pre_tokens.device, dtype=sim.dtype)
    contribs.scatter_add_(1, best_idx, diff)                 # (T, K)
    s = contribs / (contribs.max(dim=1, keepdim=True).values + 1e-8)
    logits = -s / tau
    logits = logits - logits.max(dim=1, keepdim=True).values
    m = torch.exp(logits)
    return m / m.sum(dim=1, keepdim=True)


@torch.no_grad()
def _batched_terminal_leave_one_out_mass_wj(
    pre_tokens: torch.Tensor,   # (T, N, D)
    sel_saliency: torch.Tensor, # (T, K) saliency at selected positions (w_j)
    selected: torch.Tensor,     # (T, K) per-frame local indices
    tau: float,
) -> torch.Tensor:
    """LOO with uniform coverage; external w_j multiplication.

    contrib_j = w_j · Σᵢ max(0, sim(i,j) - 2nd_max_sim(i, S\{j}))
    """
    T, N, D = pre_tokens.shape
    K = selected.shape[1]
    t_norm = F.normalize(pre_tokens.float(), p=2, dim=-1)
    sim = torch.bmm(t_norm, t_norm.transpose(1, 2))
    idx = selected.unsqueeze(1).expand(-1, N, -1)
    S = torch.gather(sim, 2, idx)
    top2 = S.topk(min(2, K), dim=2)
    best_sim = top2.values[..., 0]
    second_sim = top2.values[..., 1] if K >= 2 else torch.zeros_like(best_sim)
    best_idx = top2.indices[..., 0]
    diff = (best_sim - second_sim).clamp_min_(0)
    contribs = torch.zeros(T, K, device=pre_tokens.device, dtype=sim.dtype)
    contribs.scatter_add_(1, best_idx, diff)
    w_j = sel_saliency.float()
    w_j = w_j / (w_j.sum(dim=-1, keepdim=True) + 1e-8)
    contribs = contribs * w_j
    s = contribs / (contribs.max(dim=1, keepdim=True).values + 1e-8)
    logits = -s / tau
    logits = logits - logits.max(dim=1, keepdim=True).values
    m = torch.exp(logits)
    return m / m.sum(dim=1, keepdim=True)


@torch.no_grad()
def _batched_saliency_mass(
    saliency: torch.Tensor,     # (T, N) — full per-token saliency
    selected: torch.Tensor,     # (T, K) — local indices kept by spatial compression
    tau: float,
) -> torch.Tensor:
    """Mass from raw saliency (e.g. cls attention), same normalize+softmax pipeline.

    s_j = saliency[selected_j]; normalize by max, then m ∝ exp(-s/τ).
    """
    sal_sel = torch.gather(saliency.float(), 1, selected)        # (T, K)
    s = sal_sel / (sal_sel.max(dim=1, keepdim=True).values + 1e-8)
    logits = -s / tau
    logits = logits - logits.max(dim=1, keepdim=True).values
    m = torch.exp(logits)
    return m / m.sum(dim=1, keepdim=True)


@torch.no_grad()
def _batched_adts_isolation_mass(
    selected_tokens: torch.Tensor,   # (T, K, D) — post-spatial selected tokens
    sel_saliency: torch.Tensor,      # (T, K) — saliency at selected positions
    tau: float,
    reduction: str = "min",
) -> torch.Tensor:
    """ADTS-native LOO mass: isolation within the selected set, saliency-calibrated.

    reduction="min" mirrors ADTS's own FPS greedy gain:
      iso_j = sal[j] · min_{k ∈ S\{j}} (1 − cos(v_j, v_k))
    reduction="mean" (soft variant, less outlier-sensitive for small K):
      iso_j = sal[j] · mean_{k ∈ S\{j}} (1 − cos(v_j, v_k))
    mass ∝ exp(−iso/τ)  — high iso (unique) → low mass (keep);
                         low iso (central/duplicate) → high mass (merge target).
    """
    T, K, _ = selected_tokens.shape
    t_norm = F.normalize(selected_tokens.float(), p=2, dim=-1)
    sim = torch.bmm(t_norm, t_norm.transpose(1, 2))                   # (T, K, K)
    dist = 1.0 - sim                                                  # ∈ [0, 2]
    # cal[t, i, j] = dist[t, i, j] · sal[t, j]  (column j scaled by j's saliency)
    cal = dist * sel_saliency.float().unsqueeze(1)                    # (T, K, K)
    diag = torch.eye(K, dtype=torch.bool, device=cal.device).unsqueeze(0)
    if reduction == "min":
        cal = cal.masked_fill(diag, float('inf'))
        iso = cal.min(dim=1).values                                   # (T, K)
    elif reduction == "mean":
        # Zero the diagonal, then divide by (K - 1) to exclude self.
        cal = cal.masked_fill(diag, 0.0)
        iso = cal.sum(dim=1) / max(K - 1, 1)                          # (T, K)
    else:
        raise ValueError(f"Unknown reduction: {reduction}")
    s = iso / (iso.max(dim=1, keepdim=True).values + 1e-8)
    logits = -s / tau
    logits = logits - logits.max(dim=1, keepdim=True).values
    m = torch.exp(logits)
    return m / m.sum(dim=1, keepdim=True)


@torch.no_grad()
def _terminal_leave_one_out_mass(
    pre_tokens: torch.Tensor,   # (N, D) — full frame features before spatial compression
    saliency: torch.Tensor,     # (N,)   — per-token saliency weights
    selected: torch.Tensor,     # (K,)   — local indices into the N tokens
    tau: float,
) -> torch.Tensor:
    """Terminal leave-one-out contribution → softmax(-s/τ) mass on selected tokens.

    contrib_j = Σᵢ wᵢ · max(0, sim(i,j) - max_{k∈S, k≠j} sim(i,k))
    """
    t_norm = F.normalize(pre_tokens.float(), p=2, dim=-1)
    sim = t_norm @ t_norm.T                               # (N, N)
    w = saliency.float()
    w = w / (w.sum() + 1e-8)                              # (N,)
    S = sim[:, selected]                                  # (N, K)
    K = selected.numel()
    # O(N·K) via top-2 per row: only i whose best match in S is j contributes
    # to j's contrib; that contribution equals (best_sim - second_best_sim).
    top2 = S.topk(min(2, K), dim=1)
    best_sim = top2.values[:, 0]                          # (N,)
    if K >= 2:
        second_sim = top2.values[:, 1]                    # (N,)
    else:
        second_sim = torch.zeros_like(best_sim)
    best_idx = top2.indices[:, 0]                         # (N,) — local idx into S (0..K-1)
    diff = (best_sim - second_sim).clamp_min_(0) * w      # (N,)
    contribs = torch.zeros(K, device=pre_tokens.device, dtype=sim.dtype)
    contribs.scatter_add_(0, best_idx, diff)
    # normalize contribs to [0, 1] for τ interpretability, then softmax(-s/τ).
    s = contribs / (contribs.max() + 1e-8)
    logits = -s / tau
    logits = logits - logits.max()
    m = torch.exp(logits)
    return m / m.sum()


def _run_temporal(video_embeds, keep_indices, video_grid_thw, sms, cfg,
                  pre_video_embeds=None, saliency=None):
    """Run OT temporal merge on spatially compressed tokens.

    Returns (merged_tokens, keep_global_indices, temporal_ranges).
    """
    from ..temporal.ot_merge import (
        OTMergeConfig, compute_pair_alpha_from_pre, ot_temporal_merge,
    )

    _DEBUG = os.environ.get("PROFILE_TEMP")
    if _DEBUG:
        torch.cuda.synchronize()
        _ev0 = torch.cuda.Event(enable_timing=True); _ev0.record()

    T = video_grid_thw[0, 0].item()
    H = video_grid_thw[0, 1].item() // sms
    W = video_grid_thw[0, 2].item() // sms
    K_original = H * W
    device = video_embeds.device

    # Split spatially compressed tokens back into per-frame lists
    tokens_per_frame = keep_indices.shape[0] // T  # approximate (may vary)
    # Use keep_indices to figure out frame boundaries
    frame_boundaries = []
    for t in range(T):
        frame_start = t * K_original
        frame_end = (t + 1) * K_original
        mask = (keep_indices >= frame_start) & (keep_indices < frame_end)
        frame_boundaries.append(mask)

    frame_tokens = [video_embeds[mask] for mask in frame_boundaries]
    frame_local_indices = [keep_indices[mask] - t * K_original for t, mask in enumerate(frame_boundaries)]

    if _DEBUG:
        _ev1 = torch.cuda.Event(enable_timing=True); _ev1.record()

    # Spatial positions for each surviving token
    ys = torch.linspace(0, 1, H, device=device)
    xs = torch.linspace(0, 1, W, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    full_pos = torch.stack([gy.flatten(), gx.flatten()], dim=-1)  # (K, 2)

    frame_pos = [full_pos[idx] for idx in frame_local_indices]

    # Mass: uniform by default; non-uniform LOO when ot_mass_tau is set.
    # LOO = terminal leave-one-out contribution on pre-compression features.
    if cfg.ot_mass_tau is None or pre_video_embeds is None or saliency is None:
        frame_mass = [torch.ones(ft.shape[0], device=device) for ft in frame_tokens]
    else:
        pre_per_frame = pre_video_embeds.view(T, K_original, -1)
        sal_per_frame = saliency.view(T, K_original)
        ks = [idx.numel() for idx in frame_local_indices]
        uniform_k = len(set(ks)) == 1 and ks[0] > 0
        if uniform_k:
            selected_stack = torch.stack(frame_local_indices)       # (T, K)
            mass_batched = _batched_terminal_leave_one_out_mass(
                pre_per_frame, sal_per_frame, selected_stack, cfg.ot_mass_tau,
            )
            frame_mass = [mass_batched[t] for t in range(T)]
        else:
            frame_mass = [
                _terminal_leave_one_out_mass(
                    pre_per_frame[t], sal_per_frame[t],
                    frame_local_indices[t], cfg.ot_mass_tau,
                )
                for t in range(T)
            ]

    # α = position_aligned (same-grid-position cosine).
    pair_alpha = None
    if pre_video_embeds is not None:
        H_pre = video_grid_thw[0, 1].item() // sms
        W_pre = video_grid_thw[0, 2].item() // sms
        pre_per_frame_full = pre_video_embeds.view(T, K_original, -1)
        pair_alpha = compute_pair_alpha_from_pre(
            pre_per_frame_full, H=H_pre, W=W_pre,
        )

    # Compute total merge budget
    total_input = sum(ft.shape[0] for ft in frame_tokens)
    target_final = int(round(T * K_original * cfg.total_retention_ratio))
    target_remove = max(0, total_input - target_final)

    ot_cfg = OTMergeConfig(
        total_merge_budget=target_remove,
        sinkhorn_eps=cfg.ot_sinkhorn_eps,
        sinkhorn_iters=cfg.ot_sinkhorn_iters,
        budget_temperature=cfg.ot_budget_temperature,
        cost_threshold=cfg.ot_cost_threshold,
    )

    if _DEBUG:
        _ev2 = torch.cuda.Event(enable_timing=True); _ev2.record()
        torch.cuda.synchronize()

    result = ot_temporal_merge(
        frame_tokens=frame_tokens,
        frame_spatial_pos=frame_pos,
        frame_mass=frame_mass,
        pair_alpha=pair_alpha,
        config=ot_cfg,
    )

    if _DEBUG:
        torch.cuda.synchronize()
        _ev3 = torch.cuda.Event(enable_timing=True); _ev3.record()

    # Reconstruct global indices and temporal ranges
    all_tokens = []
    all_global_indices = []
    all_fmin = []
    all_fmax = []
    for t in range(T):
        surv = result.survivor_indices[t]
        if surv.numel() == 0:
            continue
        all_tokens.append(result.tokens[t])
        # Map back to original global indices
        original_local = frame_local_indices[t][surv]
        all_global_indices.append(original_local + t * K_original)
        all_fmin.append(result.frame_min[t])
        all_fmax.append(result.frame_max[t])

    merged_tokens = torch.cat(all_tokens, dim=0)
    merged_global_indices = torch.cat(all_global_indices, dim=0)
    temporal_ranges = torch.stack([torch.cat(all_fmin), torch.cat(all_fmax)], dim=1)

    # Sort by global index
    sort_order = merged_global_indices.argsort()
    out = (
        merged_tokens[sort_order],
        merged_global_indices[sort_order],
        temporal_ranges[sort_order],
    )

    if _DEBUG:
        _ev4 = torch.cuda.Event(enable_timing=True); _ev4.record()
        torch.cuda.synchronize()
        print(f"[TEMP-PROFILE] frame_split={_ev0.elapsed_time(_ev1):.2f}ms  "
              f"mass+pos={_ev1.elapsed_time(_ev2):.2f}ms  "
              f"OT_call={_ev2.elapsed_time(_ev3):.2f}ms  "
              f"reconstruct={_ev3.elapsed_time(_ev4):.2f}ms", flush=True)

    return out


# ============================================================
# Vision encoder patches (identical to spatial_merge version)
# ============================================================

def Qwen2_5_VLVisionAttention_forward(
    self: Qwen2_5_VLVisionAttention,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    return_logits: bool = False,
    **kwargs,
) -> torch.Tensor:
    seq_length = hidden_states.shape[0]
    query_states, key_states, value_states = (
        self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    )
    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)

    query_states = query_states.transpose(0, 1).unsqueeze(0)
    key_states = key_states.transpose(0, 1).unsqueeze(0)
    value_states = value_states.transpose(0, 1).unsqueeze(0)

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    assert self.config._attn_implementation == "flash_attention_2"
    max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
    attn_output, _ = attention_interface(
        self, query_states, key_states, value_states,
        attention_mask=None, scaling=self.scaling,
        dropout=0.0 if not self.training else self.attention_dropout,
        cu_seq_lens_q=cu_seqlens, cu_seq_lens_k=cu_seqlens,
        max_length_q=max_seqlen, max_length_k=max_seqlen,
        is_causal=False, **kwargs,
    )

    attn_weights = None
    if return_logits:
        num_frames = cu_seqlens.shape[0] - 1
        q, k = query_states.squeeze(0), key_states.squeeze(0)
        q, k = q.transpose(0, 1), k.transpose(0, 1)
        q = q.reshape(num_frames, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        k = k.reshape(num_frames, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3).contiguous()
        # Per-frame chunked computation: each frame's attention is independent
        # (cu_seqlens isolates frames). Compute mean-of-mean per-frame and stack.
        # Mathematically identical to batched matmul + softmax + mean.mean —
        # but avoids materializing the (F, H, S, S) intermediate tensor,
        # cutting peak transient memory from ~8 GB to ~0.2 GB.
        max_seqlen_q = q.shape[2]
        # PRECISION: keep softmax + mean in fp32, cast to bf16 only at the end.
        # Trade-off: more accurate saliency vs cache (bf16-mean) bit-exactness.
        # Live mode now diverges from cached saliency by ULP-level — re-extract
        # cache if strict bit-exactness is required.
        attn_weights = q.new_empty((num_frames, max_seqlen_q), dtype=q.dtype)
        scale = self.head_dim ** -0.5
        for f in range(num_frames):
            af = torch.matmul(q[f], k[f].transpose(-1, -2)) * scale  # (H, S, S) bf16
            af = nn.functional.softmax(af, dim=-1, dtype=torch.float32)  # keep fp32
            attn_weights[f] = af.mean(0).mean(0).to(q.dtype)  # fp32 mean → cast at end
    attn_output = attn_output.reshape(seq_length, -1).contiguous()
    attn_output = self.proj(attn_output)
    return attn_output, attn_weights


def Qwen2_5_VLVisionBlock_forward(
    self: Qwen2_5_VLVisionBlock,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: Optional[torch.Tensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    return_logits: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    residual = hidden_states
    hidden_states, attn_weights = self.attn(
        self.norm1(hidden_states), cu_seqlens=cu_seqlens,
        rotary_pos_emb=rotary_pos_emb, position_embeddings=position_embeddings,
        return_logits=return_logits,
    )
    hidden_states = residual + hidden_states
    residual = hidden_states
    hidden_states = self.mlp(self.norm2(hidden_states))
    hidden_states = residual + hidden_states
    return hidden_states, attn_weights


def Qwen2_5_VisionTransformerPretrainedModel_forward(
    self: Qwen2_5_VisionTransformerPretrainedModel,
    hidden_states: torch.Tensor,
    grid_thw: torch.Tensor,
) -> torch.Tensor:
    hidden_states = self.patch_embed(hidden_states)
    rotary_pos_emb = self.rot_pos_emb(grid_thw)
    window_index, cu_window_seqlens = self.get_window_index(grid_thw)
    cu_window_seqlens = torch.tensor(
        cu_window_seqlens, device=hidden_states.device,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
    hidden_states = hidden_states[window_index, :, :]
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
    rotary_pos_emb = rotary_pos_emb[window_index, :, :]
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0, dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    num_blocks = len(self.blocks)
    for layer_num, blk in enumerate(self.blocks):
        if layer_num in self.fullatt_block_indexes:
            cu_seqlens_now = cu_seqlens
        else:
            cu_seqlens_now = cu_window_seqlens
        return_logits = (num_blocks - 1) == layer_num
        hidden_states, attn_weights = blk(
            hidden_states, cu_seqlens=cu_seqlens_now,
            position_embeddings=position_embeddings, return_logits=return_logits,
        )

    hidden_states = self.merger(hidden_states)
    reverse_indices = torch.argsort(window_index)
    hidden_states = hidden_states[reverse_indices, :]

    num_frames = grid_thw[0][0].item()
    seq_len = attn_weights.shape[-1] // 4
    attn_weights = attn_weights.view(num_frames, seq_len, -1).mean(-1).view(-1)
    attn_weights = attn_weights[reverse_indices].view(num_frames, -1)

    object.__setattr__(self, '_spatial_merge_saliency', attn_weights)
    return hidden_states, attn_weights


def Qwen2_5_VLModel_get_video_features(
    self: Qwen2_5_VLModel,
    pixel_values_videos: torch.FloatTensor,
    video_grid_thw: Optional[torch.LongTensor] = None,
):
    pixel_values_videos = pixel_values_videos.type(self.visual.dtype)
    result = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
    if isinstance(result, tuple):
        video_embeds, saliency = result
    else:
        video_embeds = result
        saliency = getattr(self.visual, '_spatial_merge_saliency', None)
    split_sizes = (video_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
    video_embeds = torch.split(video_embeds, split_sizes)
    return video_embeds, saliency


# ============================================================
# Main model forward: spatial → temporal → position filter
# ============================================================

def Qwen2_5_VLModel_forward(
    self: Qwen2_5_VLModel,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    rope_deltas: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    **kwargs: Unpack[TransformersKwargs],
) -> Union[tuple, Qwen2_5_VLModelOutputWithPast]:

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    if pixel_values is not None:
        image_embeds = self.get_image_features(pixel_values, image_grid_thw)
        image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    _profile_mark("BEFORE vision encoder")
    has_video = False
    if pixel_values_videos is not None:
        video_embeds = self.get_video_features(pixel_values_videos, video_grid_thw)
        _profile_mark("AFTER vision encoder")
        if isinstance(video_embeds, tuple):
            video_embeds, saliency = video_embeds
        else:
            saliency = object.__getattribute__(self.visual, '_spatial_merge_saliency')
        video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        _, video_mask = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
        )
        n_video_tokens = video_embeds.shape[0]
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
        has_video = True
        _profile_mark("AFTER scatter to inputs_embeds")

    batch_size, seq_length = inputs_embeds.shape[:2]
    if position_ids is None:
        position_ids = cache_position.view(1, 1, -1).expand(3, batch_size, -1)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + seq_length, device=inputs_embeds.device
        )

    if rope_deltas is not None:
        if not hasattr(self, "rope_deltas"):
            self.rope_deltas = rope_deltas
        if position_ids.shape[-1] > 1:
            delta = (cache_position[0] + self.rope_deltas).to(inputs_embeds.device)
        else:
            delta = torch.zeros((batch_size, seq_length), device=inputs_embeds.device)
        delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=1)
        position_ids = position_ids + delta.to(position_ids.device)

    # =============================================
    # Compression pipeline: spatial → temporal
    # =============================================
    if has_video and position_ids.shape[-1] > 1:
        cfg: PipelineConfig = getattr(self, "pipeline_config")
        sms = self.visual.spatial_merge_size

        _profile_mark("BEFORE spatial (video_embeds ready)")

        # Latency probe (env-gated)
        if os.environ.get("PROFILE_FWD"):
            torch.cuda.synchronize()
            _t0 = torch.cuda.Event(enable_timing=True); _t0.record()

        # --- Stage 1: Spatial compression ---
        pre_video_embeds = video_embeds  # keep reference for terminal-LOO mass
        pre_saliency = saliency
        merged_tokens, keep_indices = _run_spatial(
            video_embeds, saliency, video_grid_thw, sms, cfg
        )

        if os.environ.get("PROFILE_FWD"):
            _t1 = torch.cuda.Event(enable_timing=True); _t1.record()
            torch.cuda.synchronize()
            print(f"[FWD-LATENCY] spatial = {_t0.elapsed_time(_t1):.2f} ms", flush=True)

        _profile_mark("AFTER spatial")

        temporal_ranges = None

        # --- Stage 2: OT Temporal merge (optional) ---
        if os.environ.get("PROFILE_FWD"):
            torch.cuda.synchronize()
            _t0 = torch.cuda.Event(enable_timing=True); _t0.record()

        if cfg.enable_temporal:
            merged_tokens, keep_indices, temporal_ranges = _run_temporal(
                merged_tokens, keep_indices, video_grid_thw, sms, cfg,
                pre_video_embeds=pre_video_embeds, saliency=pre_saliency,
            )
            cfg.temporal_ranges = temporal_ranges

        # COMPRESSION_DUMP: env-gated dump of compression-stage outputs for bit-exact diagnostics
        if os.environ.get("COMPRESSION_DUMP"):
            import hashlib
            sample_idx = getattr(self, '_compression_dump_idx', 0)
            self._compression_dump_idx = sample_idx + 1
            if sample_idx < 10:  # only first 10 samples
                # Hash compression output: merged_tokens (concat list), keep_indices, video_embeds, saliency input
                def _t2bytes(x):
                    # bf16/fp16 → cast to fp32 first (numpy doesn't support bf16)
                    if x.dtype in (torch.bfloat16, torch.float16):
                        x = x.float()
                    return x.cpu().contiguous().numpy().tobytes()
                def _h(t):
                    if t is None: return 'None'
                    if isinstance(t, list):
                        if len(t) == 0: return 'empty_list'
                        if isinstance(t[0], torch.Tensor):
                            return hashlib.sha256(b''.join(_t2bytes(x) for x in t)).hexdigest()[:16]
                    if isinstance(t, torch.Tensor):
                        return hashlib.sha256(_t2bytes(t)).hexdigest()[:16]
                    return str(t)[:30]
                _ve = pre_video_embeds.cpu() if pre_video_embeds is not None else None
                _sal = pre_saliency.cpu() if pre_saliency is not None else None
                print(f"[CDUMP {sample_idx:02d}] "
                      f"video_embeds_in={_h(_ve) if _ve is not None else 'None'} shape={tuple(_ve.shape) if _ve is not None else None} "
                      f"sal_in={_h(_sal) if _sal is not None else 'None'} "
                      f"merged_n={[t.shape[0] for t in merged_tokens] if isinstance(merged_tokens, list) else 'tensor'} "
                      f"merged_h={_h(merged_tokens)} "
                      f"keep_h={_h(keep_indices)}",
                      flush=True)

        if os.environ.get("PROFILE_FWD"):
            _t1 = torch.cuda.Event(enable_timing=True); _t1.record()
            torch.cuda.synchronize()
            print(f"[FWD-LATENCY] temporal = {_t0.elapsed_time(_t1):.2f} ms", flush=True)

        _profile_mark("AFTER temporal")

        if os.environ.get("PROFILE_FWD"):
            torch.cuda.synchronize()
            _t0 = torch.cuda.Event(enable_timing=True); _t0.record()
        del video_embeds, saliency
        # Removed torch.cuda.empty_cache() — costs ~15ms and is not needed
        # (PyTorch caching allocator auto-reuses freed memory in next allocation).
        if os.environ.get("PROFILE_FWD"):
            _t1 = torch.cuda.Event(enable_timing=True); _t1.record()
            torch.cuda.synchronize()
            print(f"[FWD-LATENCY] del = {_t0.elapsed_time(_t1):.2f} ms", flush=True)
        _profile_mark("AFTER del")

        # --- Position filtering (FlashVID pattern) ---
        if os.environ.get("PROFILE_FWD"):
            torch.cuda.synchronize()
            _t0 = torch.cuda.Event(enable_timing=True); _t0.record()
        visual_start_index = torch.where(input_ids[0] == self.config.video_token_id)[0][0].item()
        visual_end_index = visual_start_index + n_video_tokens

        cfg.visual_token_start_index = visual_start_index
        cfg.visual_token_length = merged_tokens.shape[0]

        bsz, _, hidden_size = inputs_embeds.shape
        keep_visual_global_indices = keep_indices.to(input_ids.device) + visual_start_index
        inputs_embeds.scatter_(
            dim=1,
            index=keep_visual_global_indices.unsqueeze(0).unsqueeze(-1).expand(bsz, -1, hidden_size),
            src=merged_tokens.view(-1, hidden_size).unsqueeze(0).to(inputs_embeds.dtype),
        )
        global_indices = torch.arange(input_ids.shape[-1], device=input_ids.device)
        keep_global_indices = torch.cat([
            global_indices[:visual_start_index],
            keep_visual_global_indices,
            global_indices[visual_end_index:],
        ]).sort().values

        inputs_embeds = torch.gather(
            inputs_embeds, dim=1,
            index=keep_global_indices.view(1, -1, 1).expand(bsz, -1, hidden_size),
        )
        del merged_tokens
        position_ids = position_ids[:, :, keep_global_indices]
        if os.environ.get("PROFILE_FWD"):
            _t1 = torch.cuda.Event(enable_timing=True); _t1.record()
            torch.cuda.synchronize()
            print(f"[FWD-LATENCY] position_filter = {_t0.elapsed_time(_t1):.2f} ms", flush=True)
        _profile_mark("AFTER position filter (LLM input ready)")
        attention_mask = attention_mask[:, keep_global_indices]
        cache_position = cache_position[keep_global_indices]

        # TODO: PPE (temporal_ppe_mode) — modify position_ids based on temporal_ranges

    outputs = self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        cache_position=cache_position,
        **kwargs,
    )

    return Qwen2_5_VLModelOutputWithPast(
        last_hidden_state=outputs.last_hidden_state,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=self.rope_deltas if hasattr(self, "rope_deltas") else rope_deltas,
    )


@torch.no_grad()
def Qwen2_5_VLForConditionalGeneration_generate(
    self: Qwen2_5_VLForConditionalGeneration,
    **kwargs,
):
    cfg: PipelineConfig = getattr(self, "pipeline_config")
    visual_token_start_index = torch.where(kwargs["input_ids"][0] == self.config.video_token_id)[0][0].item()
    visual_token_length = torch.where(kwargs["input_ids"][0] == self.config.video_token_id)[0].shape[0]
    cfg.visual_token_start_index = visual_token_start_index
    cfg.visual_token_length = visual_token_length
    return self.generate_ori(**kwargs)

