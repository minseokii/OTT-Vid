# Installation

## Requirements (tested versions)

- Python 3.10
- CUDA-capable GPU (~24 GB VRAM for Qwen2.5-VL-7B at 32 frames)
- PyTorch 2.7.0+cu128
- transformers 4.57.3
- flash-attn 2.8.3
- decord 0.6.0
- qwen-vl-utils 0.0.14

## Install

```bash
git clone https://github.com/<your-username>/OTT-Vid.git
cd OTT-Vid
pip install -e .
```

This installs the `ottvid` package and pulls in core dependencies (PyTorch,
transformers, safetensors, decord, numpy).

### Optional: visualization

```bash
pip install -e ".[viz]"     # matplotlib, Pillow
```

### Manual dependencies

If you don't use pip:

```
torch==2.7.0         transformers==4.57.3
safetensors>=0.4     decord==0.6.0
numpy>=1.24
```

For Qwen2.5-VL specifically, also install `qwen-vl-utils`:

```bash
pip install qwen-vl-utils
```

## Optional dependencies (for benchmarks)

If you want to run lmms-eval benchmarks (MVBench / Video-MME / MLVU / LVB /
Charades-TL / ActivityNet-TL):

```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git
cd lmms-eval && pip install -e .

# Then drop in OTT-Vid wrappers:
cp ../OTT-Vid/lmms_eval_wrappers/qwen2_5_vl.py     lmms_eval/models/qwen2_5_vl.py
cp ../OTT-Vid/lmms_eval_wrappers/llava_onevision.py lmms_eval/models/llava_onevision.py
```

See `lmms_eval_wrappers/README.md` for details.

## Verify install

```python
from ottvid import apply_ottvid, apply_ottvid_llava_vid, PipelineConfig
print("OK")
```

