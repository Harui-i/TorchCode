# 元データ: duoan/TorchCode (https://github.com/duoan/TorchCode)
# 採点: uv run python scripts/judge.py check sliding_window

#
# # 🔴 Solution: Sliding Window Attention
#
# Reference solution — softmax attention with a band mask.

import torch
import math

# ✅ SOLUTION

def sliding_window_attention(Q, K, V, window_size):
    d_k = K.size(-1)
    scores = torch.bmm(Q, K.transpose(1, 2)) / math.sqrt(d_k)
    S = Q.size(1)
    idx = torch.arange(S, device=Q.device)
    mask = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() > window_size
    scores = scores.masked_fill(mask.unsqueeze(0), float('-inf'))
    weights = torch.softmax(scores, dim=-1)
    return torch.bmm(weights, V)

Q=torch.randn(1,6,8); K=torch.randn(1,6,8); V=torch.randn(1,6,8)
print('window=0==V?', torch.allclose(sliding_window_attention(Q,K,V,0), V, atol=1e-5))
