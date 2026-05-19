"""Monkey-patched LLaVA-Video forward: spatial + OT temporal compression at pooled (F,169,D).

Fair-comparison insertion point (identical to FlashVID/HoliTom/UniComp/FastVID):
    encode_images -> 2dPool (729->169) -> [compression here] -> mm_newline_position=frame -> LLM

Design notes:
- Saliency = last-layer SigLIP attention mean(head).mean(query) — matches FlashVID.
- Supports spatial_method in {"none", "topk"} for now (expand as needed).
- Temporal merge uses existing OT pipeline (temporal_merge.ot_temporal_merge).
- Because OT merge crosses frame boundaries, we use `mm_newline_position=frame` and
  group merged tokens by their anchor frame (frame index of the "survivor" token).
- LLaVA uses 1D RoPE on flat sequence, so `temporal_ppe_mode` is a no-op.
"""

from typing import List, Optional, Tuple

import math
import torch
import torch.nn as nn
import types

from ..config import PipelineConfig


# ==========================================================================
# Compression entry
# ==========================================================================

@torch.no_grad()
def _run_spatial_llava(
    features: torch.Tensor,      # (F, K, D) pooled features
    saliency: torch.Tensor,      # (F, K)
    cfg: PipelineConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (spatial_features (F, Ks, D), spatial_saliency (F, Ks), keep_idx_per_frame (F, Ks)).

    keep_idx_per_frame: local index into original 169 positions (for possible re-gridding).
    Supports: spat.
    """
    F, K, D = features.shape
    if cfg.spatial_retention >= 1.0:
        idx = torch.arange(K, device=features.device).unsqueeze(0).expand(F, -1)
        return features, saliency, idx

    Ks = max(1, int(round(K * cfg.spatial_retention)))

    # spatial selection (fixed).
    from ..spatial.facility_location import facility_location_batched
    top_idx = facility_location_batched(
        frame_tokens=features, saliency=saliency, num_tokens_to_keep=Ks,
        saliency_mode="wi",
        redundancy_lambda=0.0,
    )                                                       # (F, Ks) already sorted

    gather_exp = top_idx.unsqueeze(-1).expand(-1, -1, D)        # (F, Ks, D)
    feat_sel = torch.gather(features, 1, gather_exp)            # (F, Ks, D)
    sal_sel = torch.gather(saliency, 1, top_idx)                # (F, Ks)
    return feat_sel, sal_sel, top_idx


@torch.no_grad()
def _run_temporal_llava(
    features: torch.Tensor,             # (F, Ks, D)
    saliency: torch.Tensor,             # (F, Ks)
    pre_features_full: torch.Tensor,    # (F, K, D) full pre-compression (for α)
    keep_idx_per_frame: torch.Tensor,   # (F, Ks) local grid indices (0..K-1)
    cfg: PipelineConfig,
    grid_hw: Tuple[int, int] = (13, 13),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run OT temporal merge across frames.

    Returns:
        merged_features: (N, D) flat concatenation over host frames
        anchor_frames:   (N,) host frame index of each survivor
        original_grid_pos: (N,) original 13x13 grid position of each survivor
                           (lineage-preserving: survivor's seat in its host frame).
    """
    from ..temporal.ot_merge import (
        ot_temporal_merge, OTMergeConfig, compute_pair_alpha_from_pre,
    )

    F_, Ks, D = features.shape
    H, W = grid_hw
    device = features.device

    # Spatial positions in [0,1]^2 for each retained token
    h_idx = (keep_idx_per_frame // W).float() / max(1, H - 1)
    w_idx = (keep_idx_per_frame % W).float() / max(1, W - 1)
    centroids = torch.stack([h_idx, w_idx], dim=-1)          # (F, Ks, 2)

    # Pair α (position_aligned) from pre-compression features (full grid).
    pair_alpha = compute_pair_alpha_from_pre(
        pre_features=pre_features_full,
    )

    # Convert (F, Ks, ...) tensors to lists as ot_temporal_merge expects
    frame_tokens = [features[t] for t in range(F_)]
    frame_pos = [centroids[t] for t in range(F_)]

    # Mass: uniform by default; terminal LOO softmax(-s/τ) when ot_mass_tau is set.
    if cfg.ot_mass_tau is None:
        frame_mass = [torch.ones(Ks, device=device) / Ks for _ in range(F_)]
    else:
        from .qwen2_5_vl import _batched_terminal_leave_one_out_mass
        K_full = pre_features_full.shape[1]
        if saliency.shape == (F_, K_full):
            sal_full = saliency
        else:
            # Selected-only saliency: scatter into full grid; others = per-frame mean.
            mean_sal = saliency.mean(dim=1, keepdim=True).expand(-1, K_full).clone()
            mean_sal.scatter_(1, keep_idx_per_frame, saliency)
            sal_full = mean_sal
        mass_b = _batched_terminal_leave_one_out_mass(
            pre_features_full, sal_full, keep_idx_per_frame, cfg.ot_mass_tau,
        )
        frame_mass = [mass_b[t] for t in range(F_)]

    # Compute total_merge_budget: how many tokens to remove across all pairs.
    # total_retention / spatial_retention = fraction of (F*Ks) to keep in temporal stage.
    total_initial = F_ * Ks
    keep_target = max(1, int(round(total_initial *
                                   (cfg.total_retention_ratio / max(cfg.spatial_retention, 1e-6)))))
    total_merge_budget = max(0, total_initial - keep_target)

    ot_cfg = OTMergeConfig(
        total_merge_budget=total_merge_budget,
        sinkhorn_eps=cfg.ot_sinkhorn_eps,
        sinkhorn_iters=cfg.ot_sinkhorn_iters,
        budget_temperature=cfg.ot_budget_temperature,
        cost_threshold=cfg.ot_cost_threshold,
    )

    out = ot_temporal_merge(
        frame_tokens=frame_tokens,
        frame_spatial_pos=frame_pos,
        frame_mass=frame_mass,
        pair_alpha=pair_alpha,
        config=ot_cfg,
    )

    # Flatten survivors: host frame = frame t for any token in out.tokens[t]
    merged_features = torch.cat(out.tokens, dim=0).to(features.dtype)     # (N, D)
    anchor_frames = torch.cat([
        torch.full((out.tokens[t].shape[0],), t, dtype=torch.long, device=device)
        for t in range(F_)
    ])
    # Each survivor in frame t has a local index in out.survivor_indices[t] (0..Ks-1);
    # map back to the original 13x13 grid position via keep_idx_per_frame.
    original_grid_pos = torch.cat([
        keep_idx_per_frame[t][out.survivor_indices[t]]
        for t in range(F_)
    ])
    return merged_features, anchor_frames, original_grid_pos


# ==========================================================================
# LLaVA-Video monkey-patch targets
# ==========================================================================

def build_encode_images_patch(pipeline_cfg: PipelineConfig):
    """Closure: patched encode_images that returns (features, saliency) from SigLIP."""
    def encode_images(self, images: torch.Tensor):
        # Vision tower now returns (features, saliency) per siglip_encoder.py patch.
        image_features, saliency = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features, saliency
    return encode_images


def build_prepare_patch(pipeline_cfg: PipelineConfig):
    """Closure: patched prepare_inputs_labels_for_multimodal that inserts compression."""
    from llava.constants import IMAGE_TOKEN_INDEX, IGNORE_INDEX

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, modalities=["image"], image_sizes=None,
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if isinstance(modalities, str):
            modalities = [modalities]

        # Standard video batching path (mirrors FlashVID/HoliTom)
        if not (isinstance(images, list) or images.ndim == 5):
            image_features = self.encode_images(images)
            # fall through: no compression path
            raise NotImplementedError("Non-video path not patched for compression.")

        if isinstance(images, list):
            images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
        video_idx_in_batch = [i for i, m in enumerate(modalities) if m == "video"]
        images_list = [img if img.ndim == 4 else img.unsqueeze(0) for img in images]

        concat_images = torch.cat(images_list, dim=0)
        split_sizes = [img.shape[0] for img in images_list]
        encoded_image_features, raw_saliency = self.encode_images(concat_images)  # (sum_F, 729, D), (sum_F, 729)

        encoded_image_features = torch.split(encoded_image_features, split_sizes)
        raw_saliency = torch.split(raw_saliency, split_sizes)

        image_features_out: List[torch.Tensor] = []
        assert len(encoded_image_features) == 1, "Batch size 1 only."

        for idx, (feat, sal) in enumerate(zip(encoded_image_features, raw_saliency)):
            if idx not in video_idx_in_batch:
                image_features_out.append(feat)
                continue

            # 2D pool 729 -> 169 (follow LLaVA-Video default)
            pooled_feat = self.get_2dPool(feat)                       # (F, 169, D)
            pooled_sal = self.get_2dPool(sal.unsqueeze(-1)).squeeze(-1)  # (F, 169)

            # Mine pipeline handles Ours (OT) + per-frame compression methods only.
            # Baseline methods (holitom/unicomp/fastvid/flashvid) use author forks.
            spatial_feat, spatial_sal, keep_idx = _run_spatial_llava(
                pooled_feat, pooled_sal, pipeline_cfg,
            )  # (F, Ks, D), (F, Ks), (F, Ks)

            if pipeline_cfg.enable_temporal:
                merged_feat, anchor_frames, grid_pos = _run_temporal_llava(
                    spatial_feat, spatial_sal, pooled_feat, keep_idx, pipeline_cfg,
                )
            else:
                F_, Ks, D = spatial_feat.shape
                merged_feat = spatial_feat.reshape(F_ * Ks, D)
                anchor_frames = torch.arange(F_, device=feat.device).repeat_interleave(Ks)
                grid_pos = keep_idx.reshape(-1)  # (F*Ks,) in 0..168

            # Newline insertion — branch on config.mm_newline_position to match
            # each backbone's learned distribution.
            mm_newline = getattr(self.config, "mm_newline_position", "one_token")
            newline_token = self.model.image_newline[None].to(merged_feat.device)
            K_pooled = pooled_feat.shape[1]
            side = int(round(K_pooled ** 0.5))
            H_grid = W_grid = side if side * side == K_pooled else 13
            F_total = pooled_feat.shape[0]

            if mm_newline == "grid":
                # HoliTom/UniComp/FastVID compatible: per-frame row-wise newline
                per_frame_chunks: List[torch.Tensor] = []
                for f_idx in range(F_total):
                    mask = (anchor_frames == f_idx)
                    chunk = merged_feat[mask]
                    pos = grid_pos[mask]
                    if chunk.shape[0] == 0:
                        per_frame_chunks.append(newline_token)
                        continue
                    pos_sorted, sort_idx = pos.sort()
                    chunk_sorted = chunk[sort_idx]
                    rows = pos_sorted // W_grid
                    segments: List[torch.Tensor] = []
                    cur = 0
                    while cur < chunk_sorted.shape[0]:
                        row_id = rows[cur].item()
                        end = cur
                        while end < chunk_sorted.shape[0] and rows[end].item() == row_id:
                            end += 1
                        segments.append(chunk_sorted[cur:end])
                        segments.append(newline_token)
                        cur = end
                    per_frame_chunks.append(torch.cat(segments, dim=0))
                final_feat = torch.cat(per_frame_chunks, dim=0)

            elif mm_newline == "frame":
                # FlashVID compatible: one newline per frame at frame end
                per_frame_chunks = []
                for f_idx in range(F_total):
                    mask = (anchor_frames == f_idx)
                    chunk = merged_feat[mask]
                    pos = grid_pos[mask]
                    if chunk.shape[0] == 0:
                        per_frame_chunks.append(newline_token)
                        continue
                    _, sort_idx = pos.sort()
                    chunk_sorted = chunk[sort_idx]
                    per_frame_chunks.append(torch.cat([chunk_sorted, newline_token], dim=0))
                final_feat = torch.cat(per_frame_chunks, dim=0)

            elif mm_newline == "one_token":
                # LLaVA-OV default: flatten all tokens (frame order preserved), one
                # trailing newline for the entire video.
                ordered_chunks: List[torch.Tensor] = []
                for f_idx in range(F_total):
                    mask = (anchor_frames == f_idx)
                    chunk = merged_feat[mask]
                    pos = grid_pos[mask]
                    if chunk.shape[0] == 0:
                        continue
                    _, sort_idx = pos.sort()
                    ordered_chunks.append(chunk[sort_idx])
                final_feat = torch.cat(ordered_chunks, dim=0)
                final_feat = torch.cat([final_feat, newline_token], dim=0)

            elif mm_newline == "no_token":
                ordered_chunks = []
                for f_idx in range(F_total):
                    mask = (anchor_frames == f_idx)
                    chunk = merged_feat[mask]
                    pos = grid_pos[mask]
                    if chunk.shape[0] == 0:
                        continue
                    _, sort_idx = pos.sort()
                    ordered_chunks.append(chunk[sort_idx])
                final_feat = torch.cat(ordered_chunks, dim=0)

            else:
                raise ValueError(f"Unknown mm_newline_position: {mm_newline}")

            image_features_out.append(final_feat)

        # ----- rest: identical to stock LLaVA prepare -----
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        input_ids_list = [cur[cur_am] for cur, cur_am in zip(input_ids, attention_mask)]
        labels_list = [cur[cur_am] for cur, cur_am in zip(labels, attention_mask)]

        new_input_embeds, new_labels = [], []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids_list):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features_out[cur_image_idx]
                cur_embed = self.get_model().embed_tokens(cur_input_ids)
                new_input_embeds.append(torch.cat([cur_embed, cur_image_features[0:0]], dim=0))
                new_labels.append(labels_list[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            input_ids_chunks, labels_chunks = [], []
            for i in range(len(image_token_indices) - 1):
                input_ids_chunks.append(cur_input_ids[image_token_indices[i] + 1: image_token_indices[i + 1]])
                labels_chunks.append(labels_list[batch_idx][image_token_indices[i] + 1: image_token_indices[i + 1]])
            split_sizes_ = [x.shape[0] for x in labels_chunks]
            cur_embed = self.get_model().embed_tokens(torch.cat(input_ids_chunks))
            cur_embed_no_im = torch.split(cur_embed, split_sizes_, dim=0)

            merged_embeds, merged_labels = [], []
            for i in range(num_images + 1):
                merged_embeds.append(cur_embed_no_im[i])
                merged_labels.append(labels_chunks[i])
                if i < num_images:
                    feats = image_features_out[cur_image_idx]
                    cur_image_idx += 1
                    merged_embeds.append(feats)
                    merged_labels.append(torch.full((feats.shape[0],), IGNORE_INDEX,
                                                    device=labels_chunks[i].device,
                                                    dtype=labels_chunks[i].dtype))
            merged_embeds = [e.to(self.device) for e in merged_embeds]
            new_input_embeds.append(torch.cat(merged_embeds))
            new_labels.append(torch.cat(merged_labels))

        # Pad to max_len
        tokenizer_max_len = getattr(self.config, "tokenizer_model_max_length", None)
        new_input_embeds = [x[:tokenizer_max_len] for x in new_input_embeds]
        new_labels = [x[:tokenizer_max_len] for x in new_labels]

        max_len = max(x.shape[0] for x in new_input_embeds)
        bsz = len(new_input_embeds)
        padded_embeds = []
        padded_labels = torch.full((bsz, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        new_attention_mask = torch.zeros((bsz, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        new_position_ids = torch.zeros((bsz, max_len), dtype=position_ids.dtype, device=position_ids.device)
        pad_side = getattr(self.config, "tokenizer_padding_side", "right")
        for i, (emb, lbl) in enumerate(zip(new_input_embeds, new_labels)):
            L = emb.shape[0]
            if pad_side == "left":
                padded_embeds.append(torch.cat([torch.zeros((max_len - L, emb.shape[1]), dtype=emb.dtype, device=emb.device), emb], dim=0))
                if L > 0:
                    padded_labels[i, -L:] = lbl
                    new_attention_mask[i, -L:] = True
                    new_position_ids[i, -L:] = torch.arange(0, L, dtype=new_position_ids.dtype, device=emb.device)
            else:
                padded_embeds.append(torch.cat([emb, torch.zeros((max_len - L, emb.shape[1]), dtype=emb.dtype, device=emb.device)], dim=0))
                if L > 0:
                    padded_labels[i, :L] = lbl
                    new_attention_mask[i, :L] = True
                    new_position_ids[i, :L] = torch.arange(0, L, dtype=new_position_ids.dtype, device=emb.device)
        new_input_embeds = torch.stack(padded_embeds, dim=0)

        ret_labels = None if _labels is None else padded_labels
        ret_attn = None if _attention_mask is None else new_attention_mask.to(_attention_mask.dtype)
        ret_pos = None if _position_ids is None else new_position_ids
        return None, ret_pos, ret_attn, past_key_values, new_input_embeds, ret_labels

    return prepare_inputs_labels_for_multimodal


# ==========================================================================
# Entry point: apply_ottvid_llava_vid
# ==========================================================================

def apply_ottvid_llava_vid(model, **kwargs) -> nn.Module:
    """Install pipeline compression on a LLaVA-Video (or LLaVA-OV) model.

    Expected model: LlavaQwenForCausalLM or similar LlavaMetaForCausalLM subclass.
    """
    # Build config from kwargs (same keys as PipelineConfig)
    cfg_kwargs = {k: v for k, v in kwargs.items() if k in PipelineConfig.__dataclass_fields__}
    cfg = PipelineConfig(**cfg_kwargs)

    # 1. Patch SigLIP attention last layer to emit attn weights
    from llava.model.multimodal_encoder.siglip_encoder import (
        SigLipAttention, SigLipVisionTower,
    )
    from .siglip_encoder import SigLipAttention_forward, SigLipVisionTower_forward

    # Mark last encoder layer's attention
    vision_tower = model.get_vision_tower()
    last_attn = vision_tower.vision_tower.vision_model.encoder.layers[-1].self_attn
    last_attn.is_last_layer = True
    # Bind forwards via monkey-patch on class or instance
    SigLipAttention.forward = SigLipAttention_forward
    SigLipVisionTower.forward = SigLipVisionTower_forward

    # 2. Patch encode_images to return (features, saliency)
    model.encode_images = types.MethodType(build_encode_images_patch(cfg), model)

    # 3. Patch prepare_inputs_labels_for_multimodal to insert compression
    model.prepare_inputs_labels_for_multimodal = types.MethodType(
        build_prepare_patch(cfg), model,
    )

    # Stash config for inspection
    model._pipeline_config = cfg
    return model

