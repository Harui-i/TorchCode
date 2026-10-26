# 元データ: duoan/TorchCode (https://github.com/duoan/TorchCode)
# 採点: uv run python scripts/judge.py check rmsnorm

#
# # 🟡 Medium: Implement RMSNorm
#
# Implement **Root Mean Square Layer Normalization** — the normalization used in LLaMA, Gemma, etc.
#
# $$\text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \cdot w, \quad \text{RMS}(x) = \sqrt{\frac{1}{d}\sum x_i^2 + \epsilon}$$
#
# ### Signature
# ```python
# def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
#     # Normalize over the last dimension. No mean subtraction (unlike LayerNorm).
# ```
#
# ### Rules
# - Do **NOT** use any built-in norm layers
# - Normalize over `dim=-1`
# - Must support autograd

import torch

# ✏️ YOUR IMPLEMENTATION HERE

def rms_norm(x, weight, eps=1e-6):
    pass  # Replace this

if __name__ == "__main__":
    # 🧪 Debug
    x = torch.randn(2, 8)
    w = torch.ones(8)
    out = rms_norm(x, w)
    print("Output shape:", out.shape)
    print("RMS of output:", out.pow(2).mean(dim=-1).sqrt())  # should be ~1
