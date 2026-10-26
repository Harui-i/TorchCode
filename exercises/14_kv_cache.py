# 元データ: duoan/TorchCode (https://github.com/duoan/TorchCode)
# 採点: uv run python scripts/judge.py check kv_cache

#
# # 🔴 Hard: KV Cache Attention
#
# Implement **multi-head attention with KV caching** for efficient autoregressive generation.
#
# During LLM inference, recomputing all key/value projections at every step is wasteful.
# A **KV cache** stores previously computed K and V tensors so only the new token(s) need projection.
#
# ### Signature
# ```python
# class KVCacheAttention(nn.Module):
#     def __init__(self, d_model: int, num_heads: int): ...
#     def forward(self, x: torch.Tensor, cache=None) -> tuple[torch.Tensor, tuple]:
#         # x: (B, S_new, D) — new tokens
#         # cache: None or (K_past, V_past) each (B, num_heads, S_past, d_k)
#         # Returns: (output, (K_all, V_all))
# ```
#
# ### Requirements
# - Inherit from `nn.Module`
# - `self.W_q`, `self.W_k`, `self.W_v`, `self.W_o`: `nn.Linear` projections
# - When `cache=None` (prefill): apply **causal mask**, return all K/V as cache
# - When `cache` provided (decode): concat new K/V with cached, no causal mask needed for single-token decode
# - Incremental decode must produce **identical** results to full forward pass
#
# ### Key Idea
# ```
# Prefill:  [t0 t1 t2 t3] → full causal attention → cache = (K_{0:3}, V_{0:3})
# Decode:   [t4]           → Q=t4, K/V=cache+t4  → cache = (K_{0:4}, V_{0:4})
# Decode:   [t5]           → Q=t5, K/V=cache+t5  → cache = (K_{0:5}, V_{0:5})
# ```

import torch
import torch.nn as nn
import math

# ✏️ YOUR IMPLEMENTATION HERE

class KVCacheAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        pass  # Initialize W_q, W_k, W_v, W_o

    def forward(self, x, cache=None):
        # 1. Project Q, K, V from x
        # 2. Reshape to multi-head: (B, num_heads, S, d_k)
        # 3. If cache exists, concat new K/V with cached K/V
        # 4. Compute attention (causal mask needed during prefill)
        # 5. Return (output, (K_all, V_all))
        pass

if __name__ == "__main__":
    # 🧪 Debug
    torch.manual_seed(0)
    attn = KVCacheAttention(d_model=64, num_heads=4)
    x = torch.randn(1, 6, 64)

    # Full forward
    full_out, _ = attn(x)
    print("Full output shape:", full_out.shape)  # (1, 6, 64)

    # Incremental: prefill 4, decode 1, decode 1
    out1, cache = attn(x[:, :4])
    print("Cache K shape:", cache[0].shape)  # (1, 4, 4, 16)
    out2, cache = attn(x[:, 4:5], cache=cache)
    out3, cache = attn(x[:, 5:6], cache=cache)
    inc_out = torch.cat([out1, out2, out3], dim=1)
    print("Match:", torch.allclose(full_out, inc_out, atol=1e-5))
