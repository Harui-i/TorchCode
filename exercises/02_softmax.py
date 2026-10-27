# 元データ: duoan/TorchCode (https://github.com/duoan/TorchCode)
# 採点: uv run python scripts/judge.py check softmax

#
# # 🟢 Easy: Implement Softmax
#
# Implement the **Softmax** function from scratch.
#
# $$\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$
#
# ### Signature
# ```python
# def my_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
#     ...
# ```
#
# ### Rules
# - Do **NOT** use `torch.softmax`, `F.softmax`, or `torch.nn.Softmax`
# - Must be **numerically stable** (hint: subtract `max` before `exp`)
#
# ### Example
# ```
# Input:  tensor([1., 2., 3.])
# Output: tensor([0.0900, 0.2447, 0.6652])  # sums to 1.0
# ```

import torch

# ✏️ YOUR IMPLEMENTATION HERE


def my_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    x = x - x.max(dim=dim, keepdim=True).values
    return x.exp() / x.exp().sum(dim=dim, keepdim=True)


if __name__ == "__main__":
    # 🧪 Debug
    x = torch.tensor([1.0, 2.0, 3.0])
    print("Output:", my_softmax(x, dim=-1))
    print("Sum:   ", my_softmax(x, dim=-1).sum())  # should be ~1.0
    print("Ref:   ", torch.softmax(x, dim=-1))
