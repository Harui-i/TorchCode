# 元データ: duoan/TorchCode (https://github.com/duoan/TorchCode)
# 採点: uv run python scripts/judge.py check layernorm

#
# # 🟡 Solution: Implement LayerNorm
#
# Reference solution for Layer Normalization.
#
# $$\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

import torch

# ✅ SOLUTION

def my_layer_norm(x, gamma, beta, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    return gamma * x_norm + beta

if __name__ == "__main__":
    # Verify
    x = torch.randn(2, 8)
    gamma = torch.ones(8)
    beta = torch.zeros(8)
    out = my_layer_norm(x, gamma, beta)
    ref = torch.nn.functional.layer_norm(x, [8], gamma, beta)
    print("Match ref?", torch.allclose(out, ref, atol=1e-4))
