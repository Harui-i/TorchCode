# 元データ: duoan/TorchCode (https://github.com/duoan/TorchCode)
# 採点: uv run python scripts/judge.py check attention

#
# # 🔴 Hard: Softmax Attention
#
# Implement the core attention mechanism used in Transformers.
#
# $$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
#
# ### Signature
# ```python
# def scaled_dot_product_attention(
#     Q: torch.Tensor,  # (batch, seq_q, d_k)
#     K: torch.Tensor,  # (batch, seq_k, d_k)
#     V: torch.Tensor,  # (batch, seq_k, d_v)
# ) -> torch.Tensor:   # (batch, seq_q, d_v)
#     ...
# ```
#
# ### Rules
# - Do **NOT** use `F.scaled_dot_product_attention`
# - You **may** use `torch.softmax` and `torch.bmm`
# - Must support autograd
# - Must handle cross-attention (seq_q ≠ seq_k)

import torch
import math

# ✏️ YOUR IMPLEMENTATION HERE

def scaled_dot_product_attention(Q, K, V):
    pass  # Replace this

if __name__ == "__main__":
    # 🧪 Debug
    torch.manual_seed(42)
    Q = torch.randn(2, 4, 8)
    K = torch.randn(2, 4, 8)
    V = torch.randn(2, 4, 8)

    out = scaled_dot_product_attention(Q, K, V)
    print("Output shape:", out.shape)          # should be (2, 4, 8)
    print("Has NaN?    ", torch.isnan(out).any().item())  # should be False
    print("Has Inf?    ", torch.isinf(out).any().item())  # should be False

    # Cross-attention: seq_q != seq_k
    Q2 = torch.randn(1, 3, 16)
    K2 = torch.randn(1, 5, 16)
    V2 = torch.randn(1, 5, 32)
    out2 = scaled_dot_product_attention(Q2, K2, V2)
    print("Cross-attn shape:", out2.shape)     # should be (1, 3, 32)
