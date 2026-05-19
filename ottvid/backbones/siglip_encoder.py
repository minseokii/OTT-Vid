"""SigLIP vision encoder hook for LLaVA-Video / LLaVA-OneVision.

Extracts per-token "pseudo-CLS" saliency by averaging last-layer attention weights
over heads and queries (identical to FlashVID approach; CLS token absent in SigLIP).
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


@torch.no_grad()
def SigLipVisionTower_forward(self, images: torch.Tensor):
    """Return (features, attention_weights) from last SigLIP block.

    attention_weights: (B, num_heads, q_len, k_len). Downstream code averages
    over heads + queries to get per-token saliency.
    """
    image_forward_outs = self.vision_tower(
        images.to(device=self.device, dtype=self.dtype),
        output_attentions=True,
        output_hidden_states=True,
    )
    image_features = image_forward_outs.hidden_states[-1].to(images.dtype)
    cls_attentions = image_forward_outs.attentions[-1].to(images.dtype)
    # Aggregate to per-token saliency (B, N) following FlashVID convention.
    saliency = cls_attentions.mean(1).mean(1)  # head & query average
    return image_features, saliency


def SigLipAttention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    output_attentions: Optional[bool] = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """SigLIP attention forward with optional attn_weights output on last layer only.

    Bounded memory: only materializes attn_weights tensor when `is_last_layer` is set.
    """
    batch_size, q_len, _ = hidden_states.size()
    q = self.q_proj(hidden_states).view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    k = self.k_proj(hidden_states).view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    v = self.v_proj(hidden_states).view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)

    attn_weights = torch.matmul(q, k.transpose(2, 3)) * self.scale
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    # Keep bf16 for attention; FlashVID's fp32 cast was only required for saliency
    # extraction and caused OOM on LVB with retention 0.25.
    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, v)

    attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, q_len, self.embed_dim)
    attn_output = self.out_proj(attn_output)

    if getattr(self, "is_last_layer", False):
        return attn_output, attn_weights  # caller averages head & query
    return attn_output, None

