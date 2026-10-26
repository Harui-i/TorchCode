# 元データ: duoan/TorchCode (https://github.com/duoan/TorchCode)
# 採点: uv run python scripts/judge.py check gpt2_block

#
# # 🔴 Hard: GPT-2 Transformer Block
#
# Implement a full **GPT-2 style Transformer block** — combining everything you've learned.
#
# ### Architecture (Pre-Norm)
# ```
# x = x + causal_self_attention(ln1(x))
# x = x + mlp(ln2(x))
# ```
#
# ### Signature
# ```python
# class GPT2Block(nn.Module):
#     def __init__(self, d_model: int, num_heads: int): ...
#     def forward(self, x: torch.Tensor) -> torch.Tensor: ...
# ```
#
# ### Requirements
# - Inherit from `nn.Module`
# - `self.ln1`, `self.ln2`: `nn.LayerNorm(d_model)`
# - `self.W_q`, `self.W_k`, `self.W_v`, `self.W_o`: `nn.Linear` for attention
# - `self.mlp`: `nn.Sequential(Linear(d, 4d), GELU(), Linear(4d, d))`
# - Attention must be **causal** (mask future positions)
# - Pre-norm architecture (LayerNorm *before* attention and MLP)
# - Residual connections around both attention and MLP

import torch
import torch.nn as nn
import math

# ✏️ YOUR IMPLEMENTATION HERE

class GPT2Block(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        pass  # Initialize layers

    def forward(self, x):
        pass  # Pre-norm + causal attention + MLP with residuals

if __name__ == "__main__":
    # 🧪 Debug
    torch.manual_seed(0)
    block = GPT2Block(d_model=64, num_heads=4)
    x = torch.randn(2, 8, 64)
    out = block(x)
    print("Output shape:", out.shape)           # (2, 8, 64)
    print("Is nn.Module?", isinstance(block, nn.Module))
    print("Params:", sum(p.numel() for p in block.parameters()))
