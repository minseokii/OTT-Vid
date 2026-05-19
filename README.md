# OTT-Vid: Optimal Transport Temporal Token Compression for Video Large Language Models

<div align="center">

<a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License"></a> &nbsp;
<a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-3776ab.svg" alt="Python"></a> &nbsp;
<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.0%2B-DF3411.svg" alt="PyTorch"></a> &nbsp;
<a href="https://huggingface.co/docs/transformers/index"><img src="https://img.shields.io/badge/transformers-4.55%2B-FFD21E.svg" alt="transformers"></a>

</div>

<p align="center">
  <a href="mailto:louis0503@yonsei.ac.kr">Minseok Kang</a><sup>1</sup>,
  Minhyeok Lee<sup>1</sup>,
  Jungho Lee<sup>1</sup>,
  Minjung Kim<sup>2</sup>,
  Donghyeong Kim<sup>1</sup>,
  Dayeon Lee<sup>1</sup>,
  Heeseung Choi<sup>3</sup>,
  Ig-Jae Kim<sup>3</sup>,
  Sangyoun Lee<sup>1†</sup>
  <br>
  <sup>1</sup>Yonsei University &nbsp;
  <sup>2</sup>LG Electronics &nbsp;
  <sup>3</sup>KIST
  <br>
  <sub><sup>†</sup>Corresponding author</sub>
</p>

Training-free, plug-and-play video token compression for Video-LLMs (Qwen2.5-VL,
LLaVA-OneVision, LLaVA-Video). Operates on vision-encoder features —
no fine-tuning, no architectural changes.

## 🔖 Table of Contents

1. [Highlights](#-highlights)
2. [Method](#-method)
3. [Installation](#-installation)
4. [Quickstart](#-quickstart)
5. [Supported Models](#-supported-models)
6. [Supported Datasets](#-supported-datasets)
7. [Evaluation](#-evaluation)
8. [Acknowledgement](#-acknowledgement)
9. [Citation](#-citation)

## ✨ Highlights

1. **Two-stage spatiotemporal compression** — per-frame spatial selection
   followed by OT (Sinkhorn) cross-frame merging with content-adaptive budget
   allocation.
2. **Strong-prune via cost threshold** — matches with high OT cost are
   completely dropped (the absorbed token and its downstream subtree), giving
   robust compression at extreme retention ratios (r ≤ 0.10).
3. **Training-free, plug-and-play** — single function call
   `apply_ottvid(model, ...)` monkey-patches three backbones:
   Qwen2.5-VL, LLaVA-OneVision, LLaVA-Video.

## 📋 Method

Two stages, both fixed-implementation:

1. **Spatial Compression** — Per-frame, select a token subset that best
   "covers" the frame in feature space, saliency-weighted.
2. **Temporal (OT merge)** — Sinkhorn-OT between consecutive frames' kept
   tokens.
   - Cost: `C = α · feat_dist + (1 − α) · cent_dist`, `α ∈ [0.5, 1.0]` from
     same-position cosine similarity.
   - Source mass: terminal Leave-One-Out (low mass = salient = preserved).
   - Budget: `softmax(−W / τ_b)` redistributes merges to static pairs.
   - Strong prune: discard absorbed tokens whose match cost exceeds `θ`.

## 📦 Installation

```bash
# 1. Install OTT-Vid package
git clone https://github.com/minseokii/OTT-Vid.git
cd OTT-Vid && pip install -e . && cd ..

# 2. (For benchmarks) install lmms-eval + drop in OTT-Vid wrappers
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git
cd lmms-eval && pip install -e . && cd ..
cp OTT-Vid/lmms_eval_wrappers/qwen2_5_vl.py      lmms-eval/lmms_eval/models/simple/qwen2_5_vl.py
cp OTT-Vid/lmms_eval_wrappers/llava_onevision.py lmms-eval/lmms_eval/models/simple/llava_onevision.py
cp OTT-Vid/lmms_eval_wrappers/llava_vid.py       lmms-eval/lmms_eval/models/simple/llava_vid.py

# 3. Qwen-specific helper (only for Qwen2.5-VL backbone)
pip install qwen-vl-utils

# 4. Flash-attention (optional but recommended for Qwen2.5-VL)
pip install flash-attn --no-build-isolation
```

Tested versions: PyTorch 2.7.0+cu128, transformers 4.57.3, flash-attn 2.8.3,
decord 0.6.0, qwen-vl-utils 0.0.14.
See [`docs/installation.md`](docs/installation.md) for details.

## 🚀 Quickstart

OTT-Vid is plug-and-play — wrap the model once with `apply_ottvid()`.

### Qwen2.5-VL

```python
import torch
from transformers import AutoModelForImageTextToText
from ottvid import apply_ottvid

model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
).eval()

model = apply_ottvid(
    model,
    total_retention_ratio=0.10,    # final 10 % token survival
    temporal_share=0.3,            # γ — splits r into r_s = r**(1-γ), r_t = r**γ
    enable_temporal=True,
    ot_mass_tau=0.3,
    ot_budget_temperature=0.3,
    ot_cost_threshold=0.3,
    ot_sinkhorn_eps=0.01,
    ot_sinkhorn_iters=200,
)
# Use `model.generate(...)` as usual.
```

### LLaVA-OneVision / LLaVA-Video

```python
from ottvid import apply_ottvid_llava_vid

model = apply_ottvid_llava_vid(
    model,
    total_retention_ratio=0.10,
    temporal_share=0.3,
    enable_temporal=True,
    ot_mass_tau=0.3,
    ot_budget_temperature=0.3,
    ot_cost_threshold=0.3,
)
```

A complete single-video example is in
[`examples/inference_qwen.py`](examples/inference_qwen.py):

```bash
python examples/inference_qwen.py \
    --model /path/to/Qwen2.5-VL-7B-Instruct \
    --video /path/to/video.mp4 \
    --question "Describe this video." \
    --total_retention 0.10
```

## 🤖 Supported Models

| Backbone | HF Model card |
|---|---|
| Qwen2.5-VL-7B-Instruct | [`Qwen/Qwen2.5-VL-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) |
| LLaVA-OneVision-7B | [`lmms-lab/llava-onevision-qwen2-7b-ov`](https://huggingface.co/lmms-lab/llava-onevision-qwen2-7b-ov) |
| LLaVA-Video-7B | [`lmms-lab/LLaVA-Video-7B-Qwen2`](https://huggingface.co/lmms-lab/LLaVA-Video-7B-Qwen2) |

## 📚 Supported Datasets

All datasets are pulled by lmms-eval at first run.

| Benchmark | Task | HF Dataset |
|---|---|---|
| MVBench | VQA (4-way MC, 20 subtasks) | [`OpenGVLab/MVBench`](https://huggingface.co/datasets/OpenGVLab/MVBench) |
| Video-MME | VQA (4-way MC) | [`lmms-lab/Video-MME`](https://huggingface.co/datasets/lmms-lab/Video-MME) |
| MLVU (dev) | VQA (4-way MC) | [`MLVU/MVLU`](https://huggingface.co/datasets/MLVU/MVLU) |
| LongVideoBench | VQA (5-way MC) | [`longvideobench/LongVideoBench`](https://huggingface.co/datasets/longvideobench/LongVideoBench) |
| Charades-TimeLens | VTG (mIoU) | [`TencentARC/TimeLens-Bench`](https://huggingface.co/datasets/TencentARC/TimeLens-Bench) |
| ActivityNet-TimeLens | VTG (mIoU) | [`TencentARC/TimeLens-Bench`](https://huggingface.co/datasets/TencentARC/TimeLens-Bench) |

## 📊 Evaluation

We use [LMMs-Eval](https://github.com/EvolvingLMMs-Lab/lmms-eval). Sample
launchers are in `scripts/`.

### Qwen2.5-VL on MVBench

```bash
export MODEL=/path/to/Qwen2.5-VL-7B-Instruct
bash scripts/eval_qwen_mvbench.sh
```

### Default config (paper, r=0.10)

```
total_retention_ratio = 0.10
temporal_share        = 0.3   → spatial_retention ≈ 0.20
ot_mass_tau           = 0.3
ot_budget_temperature = 0.3
ot_cost_threshold     = 0.3
ot_sinkhorn_eps       = 0.01
ot_sinkhorn_iters     = 200
```

## 👏 Acknowledgement

This project builds upon excellent open-source efforts:
[FastV](https://github.com/pkunlp-icler/FastV),
[VisionZip](https://github.com/dvlab-research/VisionZip),
[PruneVID](https://github.com/visual-ai/prunevid),
[FastVID](https://github.com/LunarShen/FastVID),
[FlashVID](https://github.com/Fanziyang-v/FlashVID),
[HoliTom](https://github.com/cokeshao/HoliTom),
[UniComp](https://github.com/jzcrooks/UniComp),
[LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT),
[Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL),
[LMMs-Eval](https://github.com/EvolvingLMMs-Lab/lmms-eval).
Thanks for their excellent work!

## 📜 Citation

```bibtex
@article{kang2026ott,
  title   = {OTT-Vid: Optimal Transport Temporal Token Compression for Video Large Language Models},
  author  = {Kang, Minseok and Lee, Minhyeok and Lee, Jungho and Kim, Minjung and Kim, Donghyeong and Lee, Dayeon and Choi, Heeseung and Kim, Ig-jae and Lee, Sangyoun},
  journal = {arXiv preprint arXiv:2605.11803},
  year    = {2026}
}
```

## License

MIT (see [LICENSE](LICENSE)).

