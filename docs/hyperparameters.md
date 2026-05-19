# OTT-Vid Hyperparameters

## User-facing knobs (5 functional + 2 numerical)

| Code variable | Paper symbol | Default | Description |
|---|---|---|---|
| `total_retention_ratio` | r | 0.10 | Final survival ratio of all video tokens. |
| `spatial_retention` | r_s | 0.20 (γ=0.3) | Per-frame retention after spatial compression. Usually derived: `r_s = r^(1-γ)` where γ = `temporal_share`. |
| `ot_mass_tau` | τ_m | 0.3 | Mass softmax temperature; salient tokens get low mass (preserved). |
| `ot_budget_temperature` | τ_b | 0.3 | Budget softmax temperature; low-W (static) pairs get larger merge budget. |
| `ot_cost_threshold` | θ | 0.3 | Strong-prune threshold on pair cost `C`; matches with `C > θ` discard the absorbed token. |
| `ot_sinkhorn_eps` | ε | 0.01 | Sinkhorn entropy regularizer. |
| `ot_sinkhorn_iters` | n_iters | 200 | Sinkhorn fixed-point iterations. |

## Fixed (paper config)

The following are *hardcoded* inside the package and not exposed as variables.
Changing them requires editing the source.

| Component | Fixed value | Rationale |
|---|---|---|
| `spatial_method` | `spat` | Single best spatial selector across benchmarks |
| `spat_saliency_mode` | `wi` | sample-side weight; matches OT mass |
| `spat_redundancy_lambda` | 0.0 | No additional FPS-style penalty needed |
| `ot_alpha_method` | `position_aligned` | Same-position cosine across consecutive frames |
| `ot_mass_source` | `loo` | Terminal Leave-One-Out (frame-level marginal gain) |
| `ot_matching_mode` | `many_to_one` | One anchor absorbs multiple absorbed tokens |
| `ot_locality_scale` (λ) | 1.0 | Cost: `C = α·feat + (1-α)·cent` |
| `temporal_ppe_mode` | `anchor` | Merged token inherits its anchor frame's M-RoPE position |

## OT cost formula

```
C[i,j] = α · feat_dist(i,j) + (1 − α) · cent_dist(i,j) ∈ [0, 1]
  feat_dist = 1 − max(0, cos(a_i, b_j))
  cent_dist = || pos_i − pos_j ||₂ / √2
  α         = 0.5 + 0.5 · (1 − mean cosine sim at same position) ∈ [0.5, 1.0]
```

## Sinkhorn-OT

```
T = Sinkhorn(μ, ν, C, ε = 0.01, iters = 200)
  μ_i = softmax(−s_i / τ_m)         # source mass (LOO over pre-compression tokens)
  ν_j = softmax(−s_j / τ_m)         # target mass
W[pair] = Σ T · C                    # pair cost summary
budget[pair] ∝ softmax(−W / τ_b)     # iteratively redistributed under per-frame caps
```

## Strong pruning

When the matched pair (i, j) has `C[i, j] > θ`:
- The absorbed token (and its entire downstream subtree in the union-find) is
  discarded from the final output.
- Retention is slightly over-compressed (conservative).

## Mass — Terminal LOO

```
s_j = Σᵢ w_i · max(0, sim(i, j) − 2nd_max sim(i, S \ {j}))     # marginal gain
m_j = softmax(−s_j / τ_m)
```

Salient tokens get low mass (less likely to be transported away in OT).

## Sweep ranges studied in the paper

- **r**: 0.01, 0.05, 0.10, 0.15, 0.20, 0.25
- **γ** (temporal share): 0.0, 0.1, …, 1.0 (sweet spot 0.3)
- **τ_m**: 0.1, 0.2, 0.3, 0.4, 0.5
- **τ_b**: 0.10, 0.15, 0.25, 0.30, 1.0
- **θ**: 0.0, 0.3, 1.0

