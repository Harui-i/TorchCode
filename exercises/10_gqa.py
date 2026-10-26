# 元データ: duoan/TorchCode (https://github.com/duoan/TorchCode)
# 採点: uv run python scripts/judge.py check gqa

#
# # 🔴 Hard: Grouped Query Attention (GQA)
#
# Implement **Grouped Query Attention** — used in LLaMA 2, Mistral, etc. to reduce KV cache size.
#
# Like MHA, but with **fewer KV heads** than Q heads. Each group of Q heads shares the same K/V head.
#
# ### Signature
# ```python
# class GroupQueryAttention:
#     def __init__(self, d_model: int, num_heads: int, num_kv_heads: int): ...
#     def forward(self, x) -> torch.Tensor:  # self-attention
# ```
#
# ### Requirements
# - `self.W_q`: `nn.Linear(d_model, d_model)` — full Q projection
# - `self.W_k`: `nn.Linear(d_model, num_kv_heads * d_k)` — reduced K projection
# - `self.W_v`: `nn.Linear(d_model, num_kv_heads * d_k)` — reduced V projection
# - `self.W_o`: `nn.Linear(d_model, d_model)` — output projection
# - `d_k = d_model // num_heads`
# - Expand KV heads with `repeat_interleave` to match Q heads
# - When `num_kv_heads == num_heads`, should behave like standard MHA

import torch
import torch.nn as nn
import math

# ✏️ YOUR IMPLEMENTATION HERE

class GroupQueryAttention:
    def __init__(self, d_model, num_heads, num_kv_heads):
        pass  # Initialize projections

    def forward(self, x):
        pass  # Self-attention with grouped KV

if __name__ == "__main__":
    # 🧪 Debug
    torch.manual_seed(0)
    gqa = GroupQueryAttention(d_model=32, num_heads=8, num_kv_heads=2)
    print("W_q shape:", gqa.W_q.weight.shape)  # (32, 32)
    print("W_k shape:", gqa.W_k.weight.shape)  # (8, 32)  — only 2 KV heads * d_k=4

    x = torch.randn(2, 6, 32)
    out = gqa.forward(x)
    print("Output shape:", out.shape)           # (2, 6, 32)
