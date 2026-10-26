# 元データ: duoan/TorchCode (https://github.com/duoan/TorchCode)
# 採点: uv run python scripts/judge.py check cross_attention

#
# # 🟠 Medium: Multi-Head Cross-Attention
#
# Implement **multi-head cross-attention** (encoder-decoder attention).
#
# ### Signature
# ```python
# class MultiHeadCrossAttention(nn.Module):
#     def __init__(self, d_model: int, num_heads: int): ...
#     def forward(self, x_q: Tensor, x_kv: Tensor) -> Tensor:
#         # x_q: (B, S_q, D) — decoder queries
#         # x_kv: (B, S_kv, D) — encoder keys/values
# ```
#
# ### Key Differences from Self-Attention
# - Q comes from the decoder, K and V come from the encoder
# - No causal mask (all encoder positions visible)

import torch
import torch.nn as nn
import math

# ✏️ YOUR IMPLEMENTATION HERE

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        pass  # W_q, W_k, W_v, W_o

    def forward(self, x_q, x_kv):
        pass  # Q from x_q, K/V from x_kv, no causal mask

if __name__ == "__main__":
    # 🧪 Debug
    attn = MultiHeadCrossAttention(64, 4)
    x_q = torch.randn(2, 6, 64)
    x_kv = torch.randn(2, 10, 64)
    print('Output:', attn(x_q, x_kv).shape)
