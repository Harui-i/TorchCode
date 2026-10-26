# 元データ: duoan/TorchCode (https://github.com/duoan/TorchCode)
# 採点: uv run python scripts/judge.py check mha

#
# # 🔴 Hard: Multi-Head Attention
#
# Implement **Multi-Head Attention** from scratch — the core building block of the Transformer.
#
# $$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$
# $$\text{head}_i = \text{Attention}(Q W_i^Q,\; K W_i^K,\; V W_i^V)$$
#
# ### Signature
# ```python
# class MultiHeadAttention:
#     def __init__(self, d_model: int, num_heads: int): ...
#     def forward(self, Q, K, V) -> torch.Tensor: ...
# ```
#
# ### Requirements
# - Use `nn.Linear(d_model, d_model)` for `self.W_q`, `self.W_k`, `self.W_v`, `self.W_o`
# - `d_k = d_model // num_heads` per head
# - `forward(Q, K, V)`: Q is `(B, seq_q, d_model)`, K/V are `(B, seq_k, d_model)`
# - Must support **cross-attention** (`seq_q != seq_k`)
# - Do **NOT** use `torch.nn.MultiheadAttention`
# - You **may** use `torch.softmax` and `torch.matmul`
#
# ### Steps
# 1. Project: `q = self.W_q(Q)`, `k = self.W_k(K)`, `v = self.W_v(V)`
# 2. Reshape to `(B, num_heads, seq, d_k)`
# 3. Scaled dot-product attention per head
# 4. Concat heads → `(B, seq_q, d_model)`
# 5. Output projection: `self.W_o(concat)`

import torch
import torch.nn as nn
import math

# ✏️ YOUR IMPLEMENTATION HERE

class MultiHeadAttention:
    def __init__(self, d_model: int, num_heads: int):
        pass  # Initialize W_q, W_k, W_v, W_o

    def forward(self, Q, K, V):
        pass  # Implement multi-head attention

if __name__ == "__main__":
    # 🧪 Debug
    torch.manual_seed(0)
    mha = MultiHeadAttention(d_model=32, num_heads=4)
    print("W_q type:", type(mha.W_q))          # should be nn.Linear
    print("W_q.weight shape:", mha.W_q.weight.shape)  # (32, 32)

    x = torch.randn(2, 6, 32)
    out = mha.forward(x, x, x)
    print("Output shape:", out.shape)          # (2, 6, 32)

    # Cross-attention
    Q = torch.randn(1, 3, 32)
    K = torch.randn(1, 7, 32)
    V = torch.randn(1, 7, 32)
    out2 = mha.forward(Q, K, V)
    print("Cross-attn shape:", out2.shape)     # (1, 3, 32)
