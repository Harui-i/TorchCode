# 元データ: duoan/TorchCode (https://github.com/duoan/TorchCode)
# 採点: uv run python scripts/judge.py check sliding_window

#
# # 🔴 Hard: Sliding Window Attention
#
# Implement **Sliding Window Attention** — used in Longformer, Mistral, etc. for efficient long-context processing.
#
# Each position $i$ can only attend to positions $j$ where $|i - j| \le w$ (the window size).
#
# ### Signature
# ```python
# def sliding_window_attention(Q, K, V, window_size):
#     # Q, K, V: (batch, seq, d) → output: (batch, seq, d_v)
#     # window_size: int — position i attends to [i-w, i+w]
# ```
#
# ### Rules
# - Do **NOT** use sparse attention libraries
# - Mask positions outside the window with `-inf`
# - `window_size=0`: only self — output should equal V
# - `window_size >= seq_len`: equivalent to full attention

import torch
import math

# ✏️ YOUR IMPLEMENTATION HERE

def sliding_window_attention(Q, K, V, window_size):
    pass  # Replace this

if __name__ == "__main__":
    # 🧪 Debug
    Q = torch.randn(1, 6, 8)
    K = torch.randn(1, 6, 8)
    V = torch.randn(1, 6, 8)

    out = sliding_window_attention(Q, K, V, window_size=1)
    print("Output shape:", out.shape)  # (1, 6, 8)

    # window=0 should return V
    out0 = sliding_window_attention(Q, K, V, window_size=0)
    print("window=0 == V?", torch.allclose(out0, V, atol=1e-5))
