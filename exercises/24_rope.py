# 元データ: duoan/TorchCode (https://github.com/duoan/TorchCode)
# 採点: uv run python scripts/judge.py check rope

#
# # 🔴 Hard: Rotary Position Embedding (RoPE)
#
# Implement **RoPE** — the position encoding used in LLaMA, GPT-NeoX, and most modern LLMs.
#
# ### Signature
# ```python
# def apply_rope(q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
#     # q, k: (B, S, D) where D is even
#     # Returns rotated (q, k) with same shape
# ```
#
# ### Key Idea
# Split each vector into consecutive pairs. Rotate each pair by `θ = pos / 10000^(2i/D)`:
# ```
# [x_0, x_1] → [x_0*cosθ - x_1*sinθ, x_0*sinθ + x_1*cosθ]
# ```
# This makes `dot(q_rot[i], k_rot[j])` depend only on `i - j` (relative position).

import torch
import math

# ✏️ YOUR IMPLEMENTATION HERE

def apply_rope(q, k):
    # 1. Compute position angles
    # 2. Split into even/odd pairs
    # 3. Apply rotation
    pass

if __name__ == "__main__":
    # 🧪 Debug
    q = torch.randn(1, 8, 16)
    k = torch.randn(1, 8, 16)
    qr, kr = apply_rope(q, k)
    print('Shape preserved:', qr.shape == q.shape)
    print('Norm preserved:', torch.allclose(q.norm(dim=-1), qr.norm(dim=-1), atol=1e-4))
