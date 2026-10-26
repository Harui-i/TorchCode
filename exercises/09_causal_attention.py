# 元データ: duoan/TorchCode (https://github.com/duoan/TorchCode)
# 採点: uv run python scripts/judge.py check causal_attention

#
# # 🔴 Hard: Causal Self-Attention
#
# Implement **causal (masked) self-attention** — the attention used in GPT-style decoders.
#
# Same as softmax attention, but each position can **only attend to itself and earlier positions** (no peeking at future tokens).
#
# $$\text{scores}_{ij} = \begin{cases} \frac{Q_i \cdot K_j}{\sqrt{d_k}} & \text{if } j \le i \\ -\infty & \text{if } j > i \end{cases}$$
#
# ### Signature
# ```python
# def causal_attention(Q, K, V):
#     # Q, K, V: (batch, seq, d) → output: (batch, seq, d_v)
# ```
#
# ### Rules
# - Do **NOT** use `F.scaled_dot_product_attention`
# - Position $i$ can only attend to positions $\le i$
# - You **may** use `torch.softmax`, `torch.bmm`, `torch.triu`

import torch
import math

# ✏️ YOUR IMPLEMENTATION HERE

def causal_attention(Q, K, V):
    pass  # Replace this

if __name__ == "__main__":
    # 🧪 Debug
    torch.manual_seed(0)
    Q = torch.randn(1, 4, 8)
    K = torch.randn(1, 4, 8)
    V = torch.randn(1, 4, 8)
    out = causal_attention(Q, K, V)
    print("Output shape:", out.shape)          # (1, 4, 8)
    print("Pos 0 == V[0]?", torch.allclose(out[:, 0], V[:, 0], atol=1e-5))  # should be True
