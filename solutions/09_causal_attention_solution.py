# 元データ: duoan/TorchCode (https://github.com/duoan/TorchCode)
# 採点: uv run python scripts/judge.py check causal_attention

#
# # 🔴 Solution: Causal Self-Attention
#
# Reference solution — softmax attention with an upper-triangular mask.

import torch
import math

# ✅ SOLUTION

def causal_attention(Q, K, V):
    d_k = K.size(-1)
    scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_k)
    S = scores.size(-1)
    mask = torch.triu(torch.ones(S, S, device=scores.device, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask.unsqueeze(0), float('-inf'))
    weights = torch.softmax(scores, dim=-1)
    return torch.bmm(weights, V)

if __name__ == "__main__":
    # Verify
    torch.manual_seed(0)
    Q = torch.randn(1, 4, 8)
    K = torch.randn(1, 4, 8)
    V = torch.randn(1, 4, 8)
    out = causal_attention(Q, K, V)
    print("Pos 0 == V[0]?", torch.allclose(out[:, 0], V[:, 0], atol=1e-5))
