"""Minimal single-video inference example with OTT-Vid + Qwen2.5-VL.

Usage:
    python examples/inference_qwen.py \\
        --model /path/to/Qwen2.5-VL-7B-Instruct \\
        --video /path/to/video.mp4 \\
        --question "What is happening in this video?"
"""
import argparse
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from ottvid import apply_ottvid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Qwen2.5-VL model path")
    parser.add_argument("--video", required=True, help="Path to mp4")
    parser.add_argument("--question", default="Describe this video.")
    parser.add_argument("--max_num_frames", type=int, default=32)
    parser.add_argument("--max_pixels", type=int, default=1605632)
    parser.add_argument("--min_pixels", type=int, default=200704)
    parser.add_argument("--total_retention", type=float, default=0.10)
    parser.add_argument("--temporal_share", type=float, default=0.3,
                        help="γ; spatial_retention = r^(1-γ)")
    parser.add_argument("--mass_tau", type=float, default=0.3)
    parser.add_argument("--budget_tau", type=float, default=0.3)
    parser.add_argument("--cost_threshold", type=float, default=0.3)
    args = parser.parse_args()

    # 1) Load model
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model,
                                              max_pixels=args.max_pixels,
                                              min_pixels=args.min_pixels)

    # 2) Install OTT-Vid compression
    model = apply_ottvid(
        model,
        total_retention_ratio=args.total_retention,
        temporal_share=args.temporal_share,
        enable_temporal=True,
        ot_mass_tau=args.mass_tau,
        ot_budget_temperature=args.budget_tau,
        ot_cost_threshold=args.cost_threshold,
        ot_sinkhorn_eps=0.01,
        ot_sinkhorn_iters=200,
    )
    print(f"OTT-Vid installed: r={args.total_retention}, γ={args.temporal_share}")

    # 3) Build chat-format prompt
    msg = [{
        "role": "user",
        "content": [
            {"type": "video", "video": args.video,
             "max_pixels": args.max_pixels, "min_pixels": args.min_pixels,
             "nframes": args.max_num_frames},
            {"type": "text", "text": args.question},
        ],
    }]
    text = processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)

    from qwen_vl_utils import process_vision_info
    _, video_inputs, video_kwargs = process_vision_info([msg], return_video_kwargs=True)
    inputs = processor(text=[text], videos=video_inputs,
                       padding=True, return_tensors="pt", **video_kwargs)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # 4) Generate
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    answer = processor.batch_decode(out_ids[:, inputs["input_ids"].shape[1]:],
                                     skip_special_tokens=True)[0]
    print(f"\nQ: {args.question}\nA: {answer}")


if __name__ == "__main__":
    main()

