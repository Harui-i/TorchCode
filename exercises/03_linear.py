# 元データ: duoan/TorchCode (https://github.com/duoan/TorchCode)
# 採点: uv run python scripts/judge.py check linear

#
# # 🟡 Medium: Simple Linear Layer
#
# Implement a fully-connected linear layer: **y = xW^T + b**
#
# ### Signature
# ```python
# class SimpleLinear:
#     def __init__(self, in_features: int, out_features: int): ...
#     def forward(self, x: torch.Tensor) -> torch.Tensor: ...
# ```
#
# ### Requirements
# - `self.weight`: shape `(out_features, in_features)`, init with `randn * (1/√in_features)`
# - `self.bias`: shape `(out_features,)`, init as zeros
# - Both must have `requires_grad=True`
# - `forward(x)` computes `x @ W^T + b`
# - Do **NOT** use `torch.nn.Linear`

import torch
import math

# ✏️ YOUR IMPLEMENTATION HERE


class SimpleLinear:
    def __init__(self, in_features: int, out_features: int):
        self.weight = torch.randn(size=(out_features, in_features)) / math.sqrt(
            in_features
        )
        self.weight.requires_grad = True
        self.bias = torch.zeros(out_features)
        self.bias.requires_grad = True
        pass  # Initialize weight and bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T + self.bias
        pass  # Compute y = x @ W^T + b


if __name__ == "__main__":
    # 🧪 Debug
    layer = SimpleLinear(8, 4)
    print("W shape:", layer.weight.shape)  # should be (4, 8)
    print("b shape:", layer.bias.shape)  # should be (4,)

    x = torch.randn(2, 8)
    y = layer.forward(x)
    print("Output shape:", y.shape)  # should be (2, 4)
