# 元データ: duoan/TorchCode (https://github.com/duoan/TorchCode)
# 採点: uv run python scripts/judge.py check softmax

#
# # 🟢 Solution: Implement Softmax
#
# Reference solution for the numerically-stable Softmax function.
#
# $$\text{softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}$$

import torch

# ✅ SOLUTION

def my_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x_max = x.max(dim=dim, keepdim=True).values
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(dim=dim, keepdim=True)

if __name__ == "__main__":
    # Verify
    x = torch.tensor([1.0, 2.0, 3.0])
    print("Output:", my_softmax(x, dim=-1))
    print("Sum:   ", my_softmax(x, dim=-1).sum())
    print("Ref:   ", torch.softmax(x, dim=-1))
