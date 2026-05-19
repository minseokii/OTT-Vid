#!/bin/bash
# Reproduce: Qwen2.5-VL-7B + OTT-Vid on MVBench at r=0.10.
# Requires the unified dispatcher in `lmms_eval_wrappers/qwen2_5_vl.py` to be
# installed into your lmms-eval clone (see lmms_eval_wrappers/README.md).
set -u

# Pre-flight: ensure lmms-eval + ottvid are importable.
python -c "import lmms_eval" 2>/dev/null || {
    echo "ERROR: lmms_eval not installed." >&2
    echo "  git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git" >&2
    echo "  cd lmms-eval && pip install -e ." >&2
    echo "  cp OTT-Vid/lmms_eval_wrappers/qwen2_5_vl.py lmms-eval/lmms_eval/models/simple/qwen2_5_vl.py" >&2
    exit 1
}
python -c "import ottvid" 2>/dev/null || {
    echo "ERROR: ottvid not installed.  Run \`cd OTT-Vid && pip install -e .\`" >&2
    exit 1
}

MODEL=${MODEL:-/path/to/Qwen2.5-VL-7B-Instruct}
OUT=${OUT:-logs/qwen_mvbench_ottvid}
mkdir -p "$OUT"

S1="pretrained=$MODEL,attn_implementation=flash_attention_2,\
max_num_frames=32,max_pixels=1605632,min_pixels=200704"

HP="enable_pipeline=True,\
pipeline_total_retention=0.10,pipeline_temporal_share=0.3,\
pipeline_enable_temporal=True,\
pipeline_ot_mass_tau=0.3,pipeline_ot_cost_threshold=0.3,\
pipeline_ot_budget_temperature=0.3,\
pipeline_ot_sinkhorn_eps=0.01,pipeline_ot_sinkhorn_iters=200"

env CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python -m lmms_eval --model qwen2_5_vl --model_args "${S1},${HP}" \
  --tasks mvbench --batch_size 1 --log_samples --output_path "$OUT"

