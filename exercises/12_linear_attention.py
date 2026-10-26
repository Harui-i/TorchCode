# 元データ: duoan/TorchCode (https://github.com/duoan/TorchCode)
# 採点: uv run python scripts/judge.py check linear_attention

#
# # 🔴 Hard: Linear Self-Attention
#
# Implement **Linear Attention** — O(S·D²) instead of O(S²·D), enabling efficient long-sequence processing.
#
# Replace softmax with a **kernel feature map** $\phi$:
#
# $$\text{LinearAttn}(Q,K,V) = \frac{\phi(Q) \left(\phi(K)^T V\right)}{\phi(Q) \cdot \sum \phi(K)}$$
#
# ### Feature map
# Use $\phi(x) = \text{elu}(x) + 1$ (ensures non-negative features).
#
# ### Signature
# ```python
# def linear_attention(Q, K, V):
#     # Q: (B, S, D_k), K: (B, S, D_k), V: (B, S, D_v)
#     # Returns: (B, S, D_v)
# ```
#
# ### Key insight
# Instead of computing the $S \times S$ attention matrix, compute $\phi(K)^T V$ first (a $D_k \times D_v$ matrix), then multiply by $\phi(Q)$.
#
# ### Rules
# - Must use a feature map (NOT softmax)
# - Must be O(S·D²) — should run fast on long sequences
# - You **may** use `F.elu`

import torch
import torch.nn.functional as F

# ✏️ YOUR IMPLEMENTATION HERE

def linear_attention(Q, K, V):
    pass  # Replace this

if __name__ == "__main__":
    # 🧪 Debug
    Q = torch.randn(1, 8, 16)
    K = torch.randn(1, 8, 16)
    V = torch.randn(1, 8, 32)
    out = linear_attention(Q, K, V)
    print("Output shape:", out.shape)   # (1, 8, 32)
    print("Has NaN?", torch.isnan(out).any().item())
