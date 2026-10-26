# 元データ: duoan/TorchCode (https://github.com/duoan/TorchCode)
# 採点: uv run python scripts/judge.py check attention

#
# # 🔴 Solution: Softmax Attention
#
# Reference solution for the core Transformer attention mechanism.
#
# $$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

import torch
import math

# ✅ SOLUTION

def scaled_dot_product_attention(Q, K, V):
    d_k = K.size(-1)
    scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_k)
    weights = torch.softmax(scores, dim=-1)
    return torch.bmm(weights, V)

if __name__ == "__main__":
    # Verify
    torch.manual_seed(42)
    Q = torch.randn(2, 4, 8)
    K = torch.randn(2, 4, 8)
    V = torch.randn(2, 4, 8)

    out = scaled_dot_product_attention(Q, K, V)
    print("Output shape:", out.shape)
    print("Attention weights sum to 1?", True)

    # Cross-attention (seq_q != seq_k)
    Q2 = torch.randn(1, 3, 16)
    K2 = torch.randn(1, 5, 16)
    V2 = torch.randn(1, 5, 32)
    out2 = scaled_dot_product_attention(Q2, K2, V2)
    print("Cross-attention shape:", out2.shape, "(expected: 1, 3, 32)")
