# lmms-eval Integration

These wrapper files extend [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)
with OTT-Vid compression. They replace (or augment) the stock model wrappers in
your lmms-eval clone.

## Files

- `qwen2_5_vl.py` — Qwen2.5-VL wrapper (`enable_pipeline=True`).
- `llava_onevision.py` — LLaVA-OneVision (1.0/baseline) wrapper.
- `llava_vid.py` — LLaVA-Video wrapper.

All three share the same `enable_pipeline=True` + `pipeline_*` argument
convention. See the argument cheat-sheet at the bottom.

## Installation

1. Clone the upstream lmms-eval and `pip install -e .` it.
2. Make sure `ottvid` is importable (install OTT-Vid root with
   `pip install -e .` or add to `PYTHONPATH`).
3. Copy these files into your lmms-eval clone (note: `models/simple/`, not just
   `models/` — recent lmms-eval splits wrappers into `simple/` vs `chat/`):
   ```bash
   cp qwen2_5_vl.py      <lmms-eval>/lmms_eval/models/simple/qwen2_5_vl.py
   cp llava_onevision.py <lmms-eval>/lmms_eval/models/simple/llava_onevision.py
   cp llava_vid.py       <lmms-eval>/lmms_eval/models/simple/llava_vid.py
   ```
   (Or symlink them.)

   The wrappers register as `qwen2_5_vl` / `llava_onevision` / `llava_vid`;
   call with `python -m lmms_eval --model qwen2_5_vl ...`. The `chat/` subdir
   contains chat-format variants (`qwen2_5_vl_chat`, `llava_onevision1_5`) —
   not used by OTT-Vid.

## Usage (Qwen2.5-VL, MVBench, r=0.10)

```bash
S1="pretrained=$MODEL,attn_implementation=flash_attention_2,\
max_num_frames=32,max_pixels=1605632,min_pixels=200704"

HP="enable_pipeline=True,\
pipeline_total_retention=0.10,pipeline_temporal_share=0.3,\
pipeline_enable_temporal=True,\
pipeline_ot_mass_tau=0.3,pipeline_ot_cost_threshold=0.3,\
pipeline_ot_budget_temperature=0.3,\
pipeline_ot_sinkhorn_eps=0.01,pipeline_ot_sinkhorn_iters=200"

python -m lmms_eval --model qwen2_5_vl --model_args "${S1},${HP}" \
  --tasks mvbench --batch_size 1
```

## Argument cheat-sheet (all three wrappers share these)

| Arg | Default | Notes |
|-----|---------|-------|
| `enable_pipeline` | `False` | Set `True` to activate OTT-Vid |
| `pipeline_total_retention` | 0.25 | r |
| `pipeline_temporal_share` | None | γ (if None, derived from `pipeline_spatial_retention`) |
| `pipeline_spatial_retention` | None | r_s override; usually leave unset |
| `pipeline_enable_temporal` | True | Stage-2 OT merge on/off |
| `pipeline_ot_mass_tau` | None | τ_m |
| `pipeline_ot_budget_temperature` | 0.2 | τ_b |
| `pipeline_ot_cost_threshold` | None | θ (strong-prune; e.g. 0.3) |
| `pipeline_ot_sinkhorn_eps` | 0.1 | ε |
| `pipeline_ot_sinkhorn_iters` | 50 | n_iters |
| `stage2_subsample` (Qwen only) | False | False = exactly `max_num_frames` (linspace) — VQA. True = fps=2 decoding up to 768 frames then linspace-subsample — VTG (Charades-/ActivityNet-TimeLens). |

