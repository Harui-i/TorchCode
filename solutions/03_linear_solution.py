# 元データ: duoan/TorchCode (https://github.com/duoan/TorchCode)
# 採点: uv run python scripts/judge.py check linear

#
# # 🟡 Solution: Simple Linear Layer
#
# Reference solution for a fully-connected linear layer: **y = xW^T + b**

import torch
import math

# ✅ SOLUTION

class SimpleLinear:
    def __init__(self, in_features: int, out_features: int):
        self.weight = torch.randn(out_features, in_features) * (1 / math.sqrt(in_features))
        self.weight.requires_grad_(True)
        self.bias = torch.zeros(out_features, requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T + self.bias

if __name__ == "__main__":
    # Verify
    layer = SimpleLinear(8, 4)
    print("W shape:", layer.weight.shape)
    print("b shape:", layer.bias.shape)
    x = torch.randn(2, 8)
    print("Output shape:", layer.forward(x).shape)
