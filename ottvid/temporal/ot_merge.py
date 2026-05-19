"""OT-based adaptive temporal token merging.

Stage 2 of the proposed pipeline. Assumes Stage 1 (spatial compression) has
already produced a set of K tokens per frame, each carrying its own spatial
position (h, w) and an optional mass (number of original tokens it represents).

Pipeline (single phase, no OT recomputation):
    1. For every adjacent frame pair (t, t+1) compute (batched across pairs):
           α_t        — frame-level dynamic weight (semantic difference)
           C_t        — cost matrix  α·feature_dist + (1-α)·centroid_dist
           T_t        — mass-weighted Sinkhorn transport plan
           W_t = Σ T_t · C_t   (Wasserstein cost)
    2. Allocate per-pair merge budget from W (low W → easy pair → more merges).
    3. For each pair, take the top-budget greedy one-to-one matches on the
       mass-normalized transport T_t / (m_a · m_bᵀ).
    4. Build absorption graph (union-find), compute per-component mass-weighted
       centroids in one batched pass. No sequential state mutation.
"""

from __future__ import annotations

import math
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_pair_alpha_from_pre(
    pre_features: torch.Tensor,           # (T, N, D), pre-spatial-compression
    H: int = None,
    W: int = None,
) -> torch.Tensor:
    """Per-pair α from same-position cosine across consecutive frames.

    α = 0.5 + 0.5 · (1 − mean_n cos(f_{t,n}, f_{t+1,n})), ∈ [0.5, 1.0].
    Vectorized across T-1 pairs in one batched op.
    """
    feat_n = F.normalize(pre_features, dim=-1)  # (T, N, D)
    sim = (feat_n[:-1] * feat_n[1:]).sum(-1).mean(-1)  # (T-1,)
    alpha_raw = (1.0 - sim).clamp(0.0, 1.0)
    return 0.5 + 0.5 * alpha_raw


def _batched_pair_cost(
    feat_a: torch.Tensor,   # (P, K, D)
    feat_b: torch.Tensor,   # (P, K, D)
    pos_a: torch.Tensor,    # (P, K, 2) or (K, 2) broadcastable
    pos_b: torch.Tensor,    # (P, K, 2) or (K, 2)
    alpha: torch.Tensor,    # (P,)
) -> torch.Tensor:
    """Batched cost matrices for P pairs.

    C = α · feat_dist + (1 − α) · cent_dist,   C ∈ [0, 1]
    (Locality scale λ is fixed to 1.0.)
    """
    a_n = F.normalize(feat_a, dim=-1)
    b_n = F.normalize(feat_b, dim=-1)
    feat_dist = 1.0 - torch.bmm(a_n, b_n.transpose(1, 2)).clamp(min=0.0)  # (P, K, K)

    # Spatial distance
    if pos_a.dim() == 2:
        pos_a = pos_a.unsqueeze(0).expand(feat_a.shape[0], -1, -1)
    if pos_b.dim() == 2:
        pos_b = pos_b.unsqueeze(0).expand(feat_b.shape[0], -1, -1)
    diff = pos_a.unsqueeze(2) - pos_b.unsqueeze(1)  # (P, K, K, 2)
    cent_dist = diff.norm(dim=-1) / math.sqrt(2.0)  # (P, K, K)

    alpha_3d = alpha.view(-1, 1, 1)  # (P, 1, 1)
    return alpha_3d * feat_dist + (1.0 - alpha_3d) * cent_dist


def _batched_sinkhorn(
    a: torch.Tensor,   # (P, K)
    b: torch.Tensor,   # (P, K)
    C: torch.Tensor,   # (P, K, K)
    eps: float = 0.1,
    iters: int = 50,
) -> torch.Tensor:
    """Batched Sinkhorn for P pairs. Returns (P, K, K) transport plans.

    Auto-switches to log-domain for eps < 0.05 to avoid exp(-C/eps) underflow.
    """
    if os.environ.get("DUMP_OT_INPUTS"):
        _dump_dir = "/tmp/ot_dumps"
        os.makedirs(_dump_dir, exist_ok=True)
        _existing = len([f for f in os.listdir(_dump_dir) if f.endswith(".pt")])
        _max_dump = int(os.environ.get("DUMP_OT_MAX", "50"))
        if _existing < _max_dump:
            torch.save({"mass_a": a.cpu(), "mass_b": b.cpu(), "C": C.cpu(), "eps": eps, "iters": iters},
                       f"{_dump_dir}/{_existing:04d}.pt")
            if _existing % 10 == 0:
                print(f"[DUMP_OT] {_existing+1}/{_max_dump} mass_a={a.shape} C={C.shape}", flush=True)
    a = a / a.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    b = b / b.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    if eps < 0.05:
        log_a = a.clamp_min(1e-30).log()
        log_b = b.clamp_min(1e-30).log()
        log_K = -C / eps
        log_u = torch.zeros_like(log_a)
        log_v = torch.zeros_like(log_b)
        for _ in range(iters):
            log_v = log_b - torch.logsumexp(log_K + log_u.unsqueeze(-1), dim=-2)
            log_u = log_a - torch.logsumexp(log_K + log_v.unsqueeze(-2), dim=-1)
        return (log_u.unsqueeze(-1) + log_K + log_v.unsqueeze(-2)).exp()

    K_mat = torch.exp(-C / eps)
    u = torch.ones_like(a)
    for _ in range(iters):
        v = b / torch.bmm(K_mat.transpose(1, 2), u.unsqueeze(2)).squeeze(2).clamp_min(1e-12)
        u = a / torch.bmm(K_mat, v.unsqueeze(2)).squeeze(2).clamp_min(1e-12)
    return u.unsqueeze(2) * K_mat * v.unsqueeze(1)


def _greedy_match(
    score: torch.Tensor,   # (Na, Nb), higher = stronger correspondence
    budget: int,
    safety: int = 5,
) -> List[Tuple[int, int]]:
    """Greedy many-to-one matching, picking top-`budget` pairs.

    Each absorber-side index j can be claimed at most once; absorbed-side i may
    repeat. Uses torch.topk to fetch `budget * safety` candidates in a single
    GPU call, then runs the greedy collision check on CPU over the candidate
    list. Falls back to fetching more candidates if the initial batch is
    insufficient (rare).
    """
    if budget <= 0:
        return []
    Na, Nb = score.shape
    flat = score.reshape(-1)
    total = flat.shape[0]

    used_b: set = set()
    matches: List[Tuple[int, int]] = []
    fetched = 0

    while len(matches) < budget and fetched < total:
        k = min(budget * safety, total - fetched)
        if k <= 0:
            break
        _, indices = torch.topk(flat, k)
        candidates = indices.tolist()               # single GPU→CPU sync
        flat[indices] = -float('inf')               # mask fetched entries
        fetched += k

        for idx in candidates:
            i, j = divmod(idx, Nb)
            if j in used_b:
                continue
            used_b.add(j)
            matches.append((i, j))
            if len(matches) >= budget:
                break

    return matches


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclass
class OTMergeConfig:
    total_merge_budget: int = 0          # # tokens to remove across all pairs
    sinkhorn_eps: float = 0.1
    sinkhorn_iters: int = 50
    budget_temperature: float = 0.2      # softmax temperature over -W
    cost_threshold: Optional[float] = None  # If set, matches with C[i,j] > threshold are
                                            # "strong-pruned": the absorbed token and its
                                            # entire subtree are removed from the output.


@dataclass
class OTMergeOutput:
    """Output of OT temporal merge.

    All fields are length-T lists. Frame t contains only the tokens that
    survived (i.e. were not absorbed). Each surviving token carries:
        tokens     — feature vector
        spatial_pos — mass-weighted centroid in [0, 1]
        mass        — total number of original tokens it represents
        frame_min   — earliest original frame index it covers
        frame_max   — latest   original frame index it covers
        survivor_indices — indices into the *input* frame_tokens[t] array
                           that survived (useful for mapping back to original
                           global indices tracked by the caller).
    """
    tokens: List[torch.Tensor]
    spatial_pos: List[torch.Tensor]
    mass: List[torch.Tensor]
    frame_min: List[torch.Tensor]
    frame_max: List[torch.Tensor]
    survivor_indices: List[torch.Tensor]


class _StageProfiler:
    """Stage-level GPU timer + peak memory tracker, env-gated by PROFILE_OT."""
    def __init__(self):
        self.enabled = bool(os.environ.get("PROFILE_OT"))
        if not self.enabled:
            return
        self.events = []
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self._start = torch.cuda.Event(enable_timing=True); self._start.record()
        self._cur_peak_baseline = torch.cuda.max_memory_allocated()

    def mark(self, label: str):
        if not self.enabled: return
        e = torch.cuda.Event(enable_timing=True); e.record()
        torch.cuda.synchronize()  # for accurate per-stage peak read
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        torch.cuda.reset_peak_memory_stats()
        self.events.append((label, e, peak_mb))

    def report(self):
        if not self.enabled or not self.events: return
        torch.cuda.synchronize()
        prev = self._start
        print(f"\n[OT-PROFILE] stage breakdown:", flush=True)
        total = 0.0
        for label, e, peak_mb in self.events:
            dt = prev.elapsed_time(e)
            total += dt
            print(f"  {label:<30} {dt:7.2f} ms   peak_in_stage={peak_mb:8.1f} MB", flush=True)
            prev = e
        print(f"  {'TOTAL':<30} {total:7.2f} ms", flush=True)


@torch.no_grad()
def ot_temporal_merge(
    frame_tokens: List[torch.Tensor],          # T × (Nt, D)
    frame_spatial_pos: List[torch.Tensor],     # T × (Nt, 2) in [0,1]
    frame_mass: Optional[List[torch.Tensor]] = None,  # T × (Nt,)
    pair_alpha: Optional[torch.Tensor] = None,        # (T-1,) precomputed α
    config: Optional[OTMergeConfig] = None,
) -> OTMergeOutput:
    """OT-based temporal merging across adjacent frame pairs.

    Inputs are lists (length T = #frames) so that frames may have different
    token counts (Stage 1 spatial merge does not guarantee uniform N).

    Args:
        frame_tokens:      per-frame token features.
        frame_spatial_pos: per-frame spatial positions, normalized to [0, 1].
        frame_mass:        per-frame token mass (default: ones).
        pair_alpha:        optional (T-1,) tensor of pre-computed α values.
        config:            OT merge configuration.

    Returns:
        OTMergeOutput with per-frame surviving tokens, spatial positions,
        masses, and frame_min / frame_max temporal spans.
    """
    cfg = config or OTMergeConfig()
    T = len(frame_tokens)
    device = frame_tokens[0].device
    prof = _StageProfiler()

    if frame_mass is None:
        frame_mass = [torch.ones(x.shape[0], device=device) for x in frame_tokens]

    # Check if all frames have uniform token count (enables batched path).
    token_counts = [f.shape[0] for f in frame_tokens]
    uniform_K = all(c == token_counts[0] for c in token_counts)

    orig_feat = [x.float() for x in frame_tokens]
    orig_pos = [x.float() for x in frame_spatial_pos]
    orig_mass = [m.float() for m in frame_mass]
    D = orig_feat[0].shape[-1]

    # Lineage: each token starts as covering only its own frame.
    orig_fmin = [torch.full((f.shape[0],), t, dtype=torch.long, device=device)
                 for t, f in enumerate(orig_feat)]
    orig_fmax = [torch.full((f.shape[0],), t, dtype=torch.long, device=device)
                 for t, f in enumerate(orig_feat)]

    if T < 2 or cfg.total_merge_budget <= 0:
        return OTMergeOutput(
            tokens=orig_feat, spatial_pos=orig_pos, mass=orig_mass,
            frame_min=orig_fmin, frame_max=orig_fmax,
            survivor_indices=[torch.arange(f.shape[0], device=device) for f in orig_feat],
        )

    # ------------------------------------------------------------------
    # 1) OT computation — batched if uniform K, else sequential fallback.
    # ------------------------------------------------------------------
    if uniform_K:
        K = token_counts[0]
        P = T - 1  # number of pairs

        feat_stack = torch.stack(orig_feat)  # (T, K, D)
        feat_a = feat_stack[:-1]             # (P, K, D)
        feat_b = feat_stack[1:]              # (P, K, D)

        mass_stack = torch.stack(orig_mass)  # (T, K)
        mass_a = mass_stack[:-1]             # (P, K)
        mass_b = mass_stack[1:]              # (P, K)

        if os.environ.get("DEBUG_MASS"):
            mf = mass_stack.detach().float().cpu()
            uniq = mf[0].unique().numel()
            print(f"[MASS-stats] T={T} K={K} per-frame-uniq={uniq} "
                  f"min={mf.min().item():.4e} max={mf.max().item():.4e} "
                  f"std={mf.std().item():.4e} (std=0 → uniform)", flush=True)

        # Positions: if all frames share the same spatial grid, broadcast.
        pos_stack = torch.stack(orig_pos)    # (T, K, 2)
        pos_a = pos_stack[:-1]               # (P, K, 2)
        pos_b = pos_stack[1:]                # (P, K, 2)

        prof.mark("setup_stack_pos_mass")

        if pair_alpha is None:
            pair_alpha = compute_pair_alpha_from_pre(feat_stack)
        prof.mark("compute_pair_alpha")

        C_batch = _batched_pair_cost(feat_a, feat_b, pos_a, pos_b,
                                     pair_alpha)  # (P, K, K)
        prof.mark("batched_pair_cost")

        T_batch = _batched_sinkhorn(mass_a, mass_b, C_batch,
                                    eps=cfg.sinkhorn_eps,
                                    iters=cfg.sinkhorn_iters)         # (P, K, K)
        prof.mark(f"batched_sinkhorn (iters={cfg.sinkhorn_iters})")

        W_tensor = (T_batch * C_batch).sum(dim=(1, 2))               # (P,)
        prof.mark("compute_W")

        pair_T_list = [T_batch[t] for t in range(P)]
        pair_C_list = [C_batch[t] for t in range(P)]
    else:
        # Fallback: sequential (frames have different token counts).
        pair_T_list = []
        pair_C_list = []
        pair_W_list = []
        for t in range(T - 1):
            if pair_alpha is not None:
                alpha = pair_alpha[t]
            else:
                alpha = compute_pair_alpha_from_pre(
                    torch.stack([orig_feat[t], orig_feat[t + 1]]),
                )[0]
            # Single-pair cost (unbatched). λ (locality_scale) fixed to 1.0.
            a_n = F.normalize(orig_feat[t], dim=-1)
            b_n = F.normalize(orig_feat[t + 1], dim=-1)
            feat_dist = 1.0 - (a_n @ b_n.T).clamp(min=0.0)
            diff = orig_pos[t].unsqueeze(1) - orig_pos[t + 1].unsqueeze(0)
            cent_dist = diff.norm(dim=-1) / math.sqrt(2.0)
            C = alpha * feat_dist + (1.0 - alpha) * cent_dist

            # Single-pair Sinkhorn
            a = orig_mass[t] / orig_mass[t].sum().clamp_min(1e-12)
            b = orig_mass[t + 1] / orig_mass[t + 1].sum().clamp_min(1e-12)
            K_mat = torch.exp(-C / cfg.sinkhorn_eps)
            u = torch.ones_like(a)
            for _ in range(cfg.sinkhorn_iters):
                v = b / (K_mat.t() @ u).clamp_min(1e-12)
                u = a / (K_mat @ v).clamp_min(1e-12)
            Tmat = u.unsqueeze(1) * K_mat * v.unsqueeze(0)

            pair_T_list.append(Tmat)
            pair_C_list.append(C)
            pair_W_list.append((Tmat * C).sum())

        W_tensor = torch.stack(pair_W_list)

    if os.environ.get("DEBUG_W"):
        Wf = W_tensor.detach().float().cpu()
        print(f"[W-stats] T={T} P={Wf.numel()} min={Wf.min().item():.4f} "
              f"max={Wf.max().item():.4f} mean={Wf.mean().item():.4f} "
              f"median={Wf.median().item():.4f}", flush=True)

    # ------------------------------------------------------------------
    # 2) Dynamic budget allocation: low W (easy/static pair) → more merges.
    #    Iterative softmax-preserving redistribution: when a pair hits its
    #    per-frame cap (token_counts[t+1]-1), freeze it and redistribute
    #    overflow over remaining active pairs via re-normalized softmax.
    #    Repeat until no overflow OR no active pair remains. Preserves the
    #    τ_w temperature semantics within the active set at each iteration.
    # ------------------------------------------------------------------
    P = T - 1
    caps_t = torch.tensor(
        [max(0, int(token_counts[t + 1]) - 1) for t in range(P)],
        dtype=torch.long, device=W_tensor.device,
    )
    budget_t = torch.zeros(P, dtype=torch.long, device=W_tensor.device)
    active = torch.ones(P, dtype=torch.bool, device=W_tensor.device)
    remaining = int(cfg.total_merge_budget)
    n_iters = 0
    debug = bool(os.environ.get("DEBUG_BUDGET"))
    if debug:
        wf0 = torch.softmax(-W_tensor / cfg.budget_temperature, dim=0)
        wf0c = wf0.detach().float().cpu()
        bf0c = (wf0 * cfg.total_merge_budget).detach().float().cpu()
        print(f"[BUDGET-stats] P={P} τ={cfg.budget_temperature} "
              f"weights min={wf0c.min().item():.4e} max={wf0c.max().item():.4e} "
              f"std={wf0c.std().item():.4e} "
              f"budget min={bf0c.min().item():.2f} max={bf0c.max().item():.2f}",
              flush=True)
    while remaining > 0 and active.any():
        n_iters += 1
        W_active = W_tensor[active]
        weights = torch.softmax(-W_active / cfg.budget_temperature, dim=0)
        bf = weights * remaining
        b_int = bf.floor().long()
        leftover_round = remaining - int(b_int.sum().item())
        if leftover_round > 0:
            frac = bf - bf.floor()
            top = torch.topk(frac, k=leftover_round).indices
            b_int[top] += 1
        # Apply clipping against cap, accumulate overflow for next iter.
        room = caps_t[active] - budget_t[active]
        applied = torch.minimum(b_int, room)
        overflow = b_int - applied
        # Update budget on active subset
        active_idx = torch.nonzero(active, as_tuple=True)[0]
        budget_t[active_idx] = budget_t[active_idx] + applied
        remaining = int(overflow.sum().item())
        # Freeze pairs that fully reached cap
        newly_frozen_local = (budget_t[active_idx] >= caps_t[active_idx])
        active[active_idx[newly_frozen_local]] = False
        if debug:
            print(f"[BUDGET-redist] iter={n_iters} active={int(active.sum().item())} "
                  f"applied={int(applied.sum().item())} overflow={remaining}",
                  flush=True)
        if int(applied.sum().item()) == 0 and remaining > 0:
            # No further capacity (shouldn't happen unless caps total < total_budget).
            break
    if debug and remaining > 0:
        print(f"[BUDGET-redist] WARNING unused={remaining} (caps total < total_merge_budget)",
              flush=True)
    budgets = budget_t.cpu().tolist()
    prof.mark("budget_allocation")

    # ------------------------------------------------------------------
    # 3) Greedy matching per pair using mass-normalized transport plan.
    # ------------------------------------------------------------------
    pair_matches: List[List[Tuple[int, int]]] = []
    for t in range(T - 1):
        Tmat = pair_T_list[t]
        pair_matches.append(_greedy_match(Tmat, budgets[t]))
    prof.mark(f"greedy_match (P={T-1})")

    # ------------------------------------------------------------------
    # 4) Batched merge via union-find.
    #    Build absorption graph, find connected components, compute
    #    per-component mass-weighted centroids in one pass from originals.
    #    Mathematically identical to sequential reverse-order absorption
    #    (mass-weighted mean is associative).
    # ------------------------------------------------------------------

    # Flatten all tokens into a single dimension for vectorized aggregation.
    frame_off = [0]
    for c in token_counts:
        frame_off.append(frame_off[-1] + c)
    total_tokens = frame_off[-1]

    # Build parent_flat: default self, overwritten by matches.
    parent_flat = torch.arange(total_tokens, dtype=torch.long, device=device)
    # Collect all (child_fi, parent_fi) pairs across T-1 pair_matches.
    # pair_matches[t] is List[(i, j)] meaning (t+1, j) absorbed by (t, i).
    # Threshold split: if C[i,j] > cost_threshold, the child is marked as a
    # "pruned root" (no union edge) so its entire subtree is removed later.
    thr = cfg.cost_threshold

    # Vectorized: collect all (pair_idx, i, j, child_fi, parent_fi) on CPU first
    # (pair_matches is already CPU ints), then do threshold comparison in one
    # batched GPU op instead of per-match .item() sync (~1200 syncs → 1 sync).
    all_t: List[int] = []
    all_i: List[int] = []
    all_j: List[int] = []
    all_child: List[int] = []
    all_parent: List[int] = []
    for t in range(T - 1):
        if not pair_matches[t]:
            continue
        off_t = frame_off[t]
        off_t1 = frame_off[t + 1]
        for i, j in pair_matches[t]:
            all_t.append(t); all_i.append(i); all_j.append(j)
            all_child.append(off_t1 + j); all_parent.append(off_t + i)

    pruned_root_flat: List[int] = []
    if all_child:
        if thr is not None:
            if os.environ.get("USE_OLD_UNIONFIND"):
                # OLD path: per-match .item() sync (verification only, ~1200 syncs).
                child_list_old: List[int] = []
                par_list_old: List[int] = []
                for k_idx in range(len(all_t)):
                    t_, i_, j_ = all_t[k_idx], all_i[k_idx], all_j[k_idx]
                    Ct = pair_C_list[t_]
                    if Ct[i_, j_].item() > thr:
                        pruned_root_flat.append(all_child[k_idx])
                    else:
                        child_list_old.append(all_child[k_idx])
                        par_list_old.append(all_parent[k_idx])
                if child_list_old:
                    c_t = torch.tensor(child_list_old, dtype=torch.long, device=device)
                    p_t = torch.tensor(par_list_old, dtype=torch.long, device=device)
                    parent_flat[c_t] = p_t
            else:
                # Batched cost lookup on GPU, then comparison, then single sync.
                t_idx = torch.tensor(all_t, dtype=torch.long, device=device)
                i_idx = torch.tensor(all_i, dtype=torch.long, device=device)
                j_idx = torch.tensor(all_j, dtype=torch.long, device=device)
                child_t = torch.tensor(all_child, dtype=torch.long, device=device)
                parent_t = torch.tensor(all_parent, dtype=torch.long, device=device)
                # If pair_C_list came from C_batch (uniform path), it's stack-able.
                # Otherwise (sequential path), C tensors may have different shapes.
                try:
                    C_stack = torch.stack(pair_C_list, dim=0)  # (P, K, K)
                    cost_vals = C_stack[t_idx, i_idx, j_idx]    # (M,)
                except RuntimeError:
                    # Non-uniform path: index per-pair (still batched per pair)
                    cost_vals = torch.empty(len(all_t), device=device, dtype=pair_C_list[0].dtype)
                    for k_idx in range(len(all_t)):
                        cost_vals[k_idx] = pair_C_list[all_t[k_idx]][all_i[k_idx], all_j[k_idx]]
                keep_mask = cost_vals <= thr  # (M,) bool
                keep_mask_cpu = keep_mask.tolist()  # single sync
                kept_child = child_t[keep_mask]
                kept_parent = parent_t[keep_mask]
                parent_flat[kept_child] = kept_parent
                for k_idx, keep in enumerate(keep_mask_cpu):
                    if not keep:
                        pruned_root_flat.append(all_child[k_idx])
        else:
            child_t = torch.tensor(all_child, dtype=torch.long, device=device)
            parent_t = torch.tensor(all_parent, dtype=torch.long, device=device)
            parent_flat[child_t] = parent_t
    prof.mark("build_parent_flat (cost_thr)")

    # Pointer-doubling union-find: root[i] = fixed point of parent[i].
    # Max chain length ≤ T (sequential absorption across frames), so
    # ceil(log2(T)) iterations suffice. Use a generous 20 for safety.
    root_flat = parent_flat
    for _ in range(20):
        nxt = root_flat[root_flat]
        if torch.equal(nxt, root_flat):
            break
        root_flat = nxt
    prof.mark("pointer_doubling")

    orig_feat_cat = torch.cat(orig_feat, dim=0)     # (total, D)
    orig_pos_cat = torch.cat(orig_pos, dim=0)       # (total, 2)
    orig_mass_cat = torch.cat(orig_mass, dim=0)     # (total,)
    orig_fmin_cat = torch.cat(orig_fmin, dim=0)     # (total,)
    orig_fmax_cat = torch.cat(orig_fmax, dim=0)     # (total,)

    # Uniform averaging (paper formula): per-cluster count-based mean.
    # Mass field is kept as sum for downstream metadata but does NOT weight feature/pos.
    ones_cat = torch.ones_like(orig_mass_cat)

    new_feat_num = torch.zeros_like(orig_feat_cat)
    new_pos_num = torch.zeros_like(orig_pos_cat)
    new_mass = torch.zeros_like(orig_mass_cat)
    new_count = torch.zeros_like(orig_mass_cat)
    new_feat_num.index_add_(0, root_flat, orig_feat_cat)
    new_pos_num.index_add_(0, root_flat, orig_pos_cat)
    new_mass.index_add_(0, root_flat, orig_mass_cat)
    new_count.index_add_(0, root_flat, ones_cat)

    # fmin/fmax via scatter_reduce with sentinel init.
    big = torch.iinfo(orig_fmin_cat.dtype).max
    neg = torch.iinfo(orig_fmax_cat.dtype).min
    new_fmin_buf = torch.full_like(orig_fmin_cat, big)
    new_fmax_buf = torch.full_like(orig_fmax_cat, neg)
    new_fmin_buf.scatter_reduce_(0, root_flat, orig_fmin_cat, reduce='amin', include_self=True)
    new_fmax_buf.scatter_reduce_(0, root_flat, orig_fmax_cat, reduce='amax', include_self=True)
    prof.mark("aggregate (index_add+scatter)")

    # Write back at root positions; non-root positions get original values
    # (they will be filtered out by absorbed mask below).
    is_root = (root_flat == torch.arange(total_tokens, device=device))
    safe_count = new_count.clamp_min(1.0).unsqueeze(-1)

    result_feat_flat = orig_feat_cat.clone()
    result_pos_flat = orig_pos_cat.clone()
    result_mass_flat = orig_mass_cat.clone()
    result_fmin_flat = orig_fmin_cat.clone()
    result_fmax_flat = orig_fmax_cat.clone()

    result_feat_flat[is_root] = (new_feat_num / safe_count)[is_root]
    result_pos_flat[is_root] = (new_pos_num / safe_count)[is_root]
    result_mass_flat[is_root] = new_mass[is_root]
    result_fmin_flat[is_root] = new_fmin_buf[is_root]
    result_fmax_flat[is_root] = new_fmax_buf[is_root]

    # Absorbed = not the root of itself
    absorbed_flat = ~is_root

    # Strong-prune: tokens whose root is a pruned root are removed from output
    # (same treatment as absorbed — excluded via the ~survivor mask below).
    if pruned_root_flat:
        pr_mask = torch.zeros(total_tokens, dtype=torch.bool, device=device)
        pr_t = torch.tensor(pruned_root_flat, dtype=torch.long, device=device)
        pr_mask[pr_t] = True
        pruned_subtree_flat = pr_mask[root_flat]
        absorbed_flat = absorbed_flat | pruned_subtree_flat

    # Split back per-frame
    result_feat = [result_feat_flat[frame_off[t]:frame_off[t+1]] for t in range(T)]
    result_pos = [result_pos_flat[frame_off[t]:frame_off[t+1]] for t in range(T)]
    result_mass = [result_mass_flat[frame_off[t]:frame_off[t+1]] for t in range(T)]
    result_fmin = [result_fmin_flat[frame_off[t]:frame_off[t+1]] for t in range(T)]
    result_fmax = [result_fmax_flat[frame_off[t]:frame_off[t+1]] for t in range(T)]
    absorbed = [absorbed_flat[frame_off[t]:frame_off[t+1]] for t in range(T)]

    out = OTMergeOutput(
        tokens=[result_feat[t][~absorbed[t]] for t in range(T)],
        spatial_pos=[result_pos[t][~absorbed[t]] for t in range(T)],
        mass=[result_mass[t][~absorbed[t]] for t in range(T)],
        frame_min=[result_fmin[t][~absorbed[t]] for t in range(T)],
        frame_max=[result_fmax[t][~absorbed[t]] for t in range(T)],
        survivor_indices=[(~absorbed[t]).nonzero(as_tuple=True)[0] for t in range(T)],
    )
    prof.mark("write_back + per_frame_split")
    prof.report()
    return out

