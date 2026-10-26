# 元データ: duoan/TorchCode (https://github.com/duoan/TorchCode)
# 採点: uv run python scripts/judge.py check mlp

#
# # 🟠 Medium: SwiGLU MLP
#
# Implement the **SwiGLU MLP** (feed-forward network) used in modern LLMs like LLaMA.
#
# $$\text{SwiGLU}(x) = \text{down\_proj}\big(\text{SiLU}(\text{gate\_proj}(x)) \odot \text{up\_proj}(x)\big)$$
#
# where $\text{SiLU}(x) = x \cdot \sigma(x)$
#
# ### Signature
# ```python
# class SwiGLUMLP(nn.Module):
#     def __init__(self, d_model: int, d_ff: int): ...
#     def forward(self, x: torch.Tensor) -> torch.Tensor: ...
# ```
#
# ### Requirements
# - Inherit from `nn.Module`
# - `self.gate_proj`: `nn.Linear(d_model, d_ff)`
# - `self.up_proj`: `nn.Linear(d_model, d_ff)`
# - `self.down_proj`: `nn.Linear(d_ff, d_model)`
# - Activation: **SiLU** (a.k.a. Swish) — `F.silu` or implement as `x * torch.sigmoid(x)`
#
# ### Why SwiGLU?
# Unlike the classic `Linear → ReLU/GELU → Linear` FFN, SwiGLU uses a **gating mechanism**:
# the gate projection controls information flow, while the up projection provides the content.
# This consistently outperforms standard FFNs in practice (PaLM, LLaMA, Mistral all use it).

import torch
import torch.nn as nn
import torch.nn.functional as F

# ✏️ YOUR IMPLEMENTATION HERE

class SwiGLUMLP(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        pass  # Initialize gate_proj, up_proj, down_proj

    def forward(self, x):
        pass  # down_proj(silu(gate_proj(x)) * up_proj(x))

if __name__ == "__main__":
    # 🧪 Debug
    mlp = SwiGLUMLP(d_model=64, d_ff=128)
    x = torch.randn(2, 8, 64)
    out = mlp(x)
    print("Output shape:", out.shape)  # (2, 8, 64)
    print("Params:", sum(p.numel() for p in mlp.parameters()))
