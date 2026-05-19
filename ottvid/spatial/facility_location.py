"""Weighted Facility Location based spatial token pruning.

Supports 4 saliency modes:
  "none":  gain[j] = Σᵢ relu(sim(i,j) - current_max[i])
  "wj":    gain[j] = w_j · Σᵢ relu(sim(i,j) - current_max[i])
  "wi":    gain[j] = Σᵢ wᵢ · relu(sim(i,j) - current_max[i])
  "wij":   gain[j] = w_j · Σᵢ wᵢ · relu(sim(i,j) - current_max[i])
"""

import math
from typing import Tuple

import torch
import torch.nn.functional as F


@torch.no_grad()
def _greedy_facility_location(
    sim_matrix: torch.Tensor,
    weights: torch.Tensor,
    K: int,
    saliency_mode: str = "wi",
    redundancy_lambda: float = 0.0,
) -> torch.Tensor:
    """Optimized greedy facility location with configurable saliency.

    Args:
        sim_matrix: (N, N) pairwise cosine similarity.
        weights: (N,) per-token saliency weights (sum to 1).
        K: number of tokens to select.
        saliency_mode: "none", "wj", "wi", or "wij".
        redundancy_lambda: penalty weight for similarity to already-selected tokens.
            gain = coverage_gain - lambda * max_sim(j, selected). Default 0 (no penalty).

    Returns:
        selected: (K,) indices of selected tokens.
    """
    N = sim_matrix.shape[0]
    device = sim_matrix.device
    K = min(K, N)

    current_max_sim = torch.zeros(N, device=device)
    selected = torch.zeros(K, dtype=torch.long, device=device)
    selected_mask = torch.zeros(N, dtype=torch.bool, device=device)
    max_sim_to_selected = torch.zeros(N, device=device)

    diff_buf = torch.empty_like(sim_matrix)  # (N, N)

    # Precompute uniform weights for "none" and "wj" modes
    ones = torch.ones(N, device=device) / N

    for k in range(K):
        torch.sub(sim_matrix, current_max_sim.unsqueeze(1), out=diff_buf)
        diff_buf.clamp_(min=0)

        if saliency_mode == "none":
            # gain[j] = Σᵢ relu(sim(i,j) - current_max[i])
            gains = torch.mv(diff_buf.t(), ones)
        elif saliency_mode == "wj":
            # gain[j] = w_j · Σᵢ relu(sim(i,j) - current_max[i])
            coverage = torch.mv(diff_buf.t(), ones)
            gains = weights * coverage
        elif saliency_mode == "wi":
            # gain[j] = Σᵢ wᵢ · relu(sim(i,j) - current_max[i])
            gains = torch.mv(diff_buf.t(), weights)
        elif saliency_mode == "wij":
            # gain[j] = w_j · Σᵢ wᵢ · relu(sim(i,j) - current_max[i])
            coverage_weighted = torch.mv(diff_buf.t(), weights)
            gains = weights * coverage_weighted
        else:
            raise ValueError(f"Unknown saliency_mode: {saliency_mode}")

        if redundancy_lambda > 0 and k > 0:
            gains = gains - redundancy_lambda * max_sim_to_selected

        gains[selected_mask] = -float('inf')

        best = gains.argmax()
        selected[k] = best
        selected_mask[best] = True

        torch.max(current_max_sim, sim_matrix[:, best], out=current_max_sim)
        if redundancy_lambda > 0:
            torch.max(max_sim_to_selected, sim_matrix[:, best], out=max_sim_to_selected)

    return selected


@torch.no_grad()
def _greedy_facility_location_batched(
    sim_matrix: torch.Tensor,   # (T, N, N)
    weights: torch.Tensor,      # (T, N)
    K: int,
    saliency_mode: str = "wi",
    redundancy_lambda: float = 0.0,
) -> torch.Tensor:
    """Batched greedy FL across T frames. Returns (T, K) selected indices."""
    Tb, N, _ = sim_matrix.shape
    device = sim_matrix.device
    K = min(K, N)

    current_max = torch.zeros(Tb, N, device=device)
    selected = torch.zeros(Tb, K, dtype=torch.long, device=device)
    mask = torch.zeros(Tb, N, dtype=torch.bool, device=device)
    max_sim_to_selected = torch.zeros(Tb, N, device=device)
    ones_w = torch.full((Tb, N), 1.0 / N, device=device)
    neg_inf = torch.tensor(-float('inf'), device=device)

    for k in range(K):
        diff = (sim_matrix - current_max.unsqueeze(2)).clamp_min(0)  # (T, N, N)
        if saliency_mode == "none":
            gains = torch.bmm(diff.transpose(1, 2), ones_w.unsqueeze(2)).squeeze(2)
        elif saliency_mode == "wj":
            cov = torch.bmm(diff.transpose(1, 2), ones_w.unsqueeze(2)).squeeze(2)
            gains = weights * cov
        elif saliency_mode == "wi":
            gains = torch.bmm(diff.transpose(1, 2), weights.unsqueeze(2)).squeeze(2)
        elif saliency_mode == "wij":
            cov_w = torch.bmm(diff.transpose(1, 2), weights.unsqueeze(2)).squeeze(2)
            gains = weights * cov_w
        else:
            raise ValueError(f"Unknown saliency_mode: {saliency_mode}")

        if redundancy_lambda > 0 and k > 0:
            gains = gains - redundancy_lambda * max_sim_to_selected
        gains = torch.where(mask, neg_inf, gains)

        best = gains.argmax(dim=1)                     # (T,)
        selected[:, k] = best
        mask.scatter_(1, best.unsqueeze(1), True)
        # sim_matrix[b, :, best[b]] → (T, N)
        sim_at_best = torch.gather(
            sim_matrix, 2, best.view(-1, 1, 1).expand(-1, N, 1)
        ).squeeze(2)
        torch.max(current_max, sim_at_best, out=current_max)
        if redundancy_lambda > 0:
            torch.max(max_sim_to_selected, sim_at_best, out=max_sim_to_selected)

    return selected


@torch.no_grad()
def facility_location_batched(
    frame_tokens: torch.Tensor,   # (T, N, D)
    saliency: torch.Tensor,       # (T, N)
    num_tokens_to_keep: int,
    saliency_mode: str = "wi",
    redundancy_lambda: float = 0.0,
) -> torch.Tensor:
    """Batched spatial across T frames. Returns (T, K) sorted selected indices."""
    Tb, N, D = frame_tokens.shape
    K = min(num_tokens_to_keep, N)
    tokens_norm = F.normalize(frame_tokens.float(), p=2, dim=-1)
    sim = torch.bmm(tokens_norm, tokens_norm.transpose(1, 2))  # (T, N, N)
    w = saliency.float()
    w = w / (w.sum(dim=-1, keepdim=True) + 1e-8)
    sel = _greedy_facility_location_batched(
        sim, w, K, saliency_mode=saliency_mode,
        redundancy_lambda=redundancy_lambda,
    )
    # Sort to match per-frame API
    return sel.sort(dim=1).values


@torch.no_grad()
def facility_location_per_frame(
    frame_tokens: torch.Tensor,
    saliency: torch.Tensor,
    H: int,
    W: int,
    num_tokens_to_keep: int,
    saliency_mode: str = "wi",
    redundancy_lambda: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply Weighted Facility Location pruning to a single frame.

    Args:
        frame_tokens: (N, D) visual tokens.
        saliency: (N,) per-token saliency scores.
        H, W: spatial grid dimensions.
        num_tokens_to_keep: target number of tokens.
        saliency_mode: "none", "wj", "wi", or "wij".
        redundancy_lambda: penalty for similarity to already-selected tokens.

    Returns:
        selected_tokens: (K, D)
        selected_indices: (K,)
    """
    N, D = frame_tokens.shape
    num_tokens_to_keep = min(num_tokens_to_keep, N)

    tokens_norm = F.normalize(frame_tokens, p=2, dim=-1)
    sim_matrix = torch.mm(tokens_norm, tokens_norm.t()).float()

    weights = saliency.float()
    weights = weights / (weights.sum() + 1e-8)

    selected_indices = _greedy_facility_location(
        sim_matrix, weights, num_tokens_to_keep,
        saliency_mode=saliency_mode, redundancy_lambda=redundancy_lambda,
    )
    selected_indices = selected_indices.sort().values
    selected_tokens = frame_tokens[selected_indices]

    return selected_tokens, selected_indices


@torch.no_grad()
def facility_location_video(
    video_embeds: torch.Tensor,
    saliency: torch.Tensor,
    video_grid_thw: torch.Tensor,
    retention_ratio: float = 0.25,
    spatial_merge_size: int = 2,
    saliency_mode: str = "wi",
    redundancy_lambda: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply Facility Location pruning across all frames.

    Args:
        video_embeds: (total_tokens, D) flat post-merge visual tokens.
        saliency: (num_frames, tokens_per_frame) saliency scores.
        video_grid_thw: (num_videos, 3) tensor with [T, H, W].
        retention_ratio: fraction of tokens to keep per frame.
        spatial_merge_size: vision encoder's spatial merge size.
        saliency_mode: "none", "wj", "wi", or "wij".
        redundancy_lambda: penalty for similarity to already-selected tokens.

    Returns:
        selected_tokens: (total_selected, D)
        keep_indices: (total_selected,) global indices.
    """
    num_frames = video_grid_thw[0, 0].item()
    H_grid = video_grid_thw[0, 1].item()
    W_grid = video_grid_thw[0, 2].item()
    H_post = H_grid // spatial_merge_size
    W_post = W_grid // spatial_merge_size
    tokens_per_frame = H_post * W_post

    num_tokens_to_keep = max(1, int(tokens_per_frame * retention_ratio))

    frame_tokens_list = video_embeds.view(num_frames, tokens_per_frame, -1)

    # Batched spatial across frames
    selected_idx_batched = facility_location_batched(
        frame_tokens=frame_tokens_list,
        saliency=saliency,
        num_tokens_to_keep=num_tokens_to_keep,
        saliency_mode=saliency_mode,
        redundancy_lambda=redundancy_lambda,
    )  # (T, K)

    # Gather selected tokens, build global indices
    T, K = selected_idx_batched.shape
    selected = frame_tokens_list.gather(
        1, selected_idx_batched.unsqueeze(-1).expand(-1, -1, frame_tokens_list.shape[-1])
    ).reshape(T * K, -1)
    offsets = (torch.arange(T, device=selected_idx_batched.device) * tokens_per_frame).unsqueeze(1)
    global_idx = (selected_idx_batched + offsets).reshape(T * K)

    n_total = num_frames * tokens_per_frame
    lam_str = f", λ={redundancy_lambda}" if redundancy_lambda > 0 else ""
    print(f"[FL] {n_total} -> {selected.shape[0]} tokens "
          f"({selected.shape[0]/n_total*100:.1f}% retained{lam_str})")
    return selected, global_idx

