# 元データ: duoan/TorchCode (https://github.com/duoan/TorchCode)
# 採点: uv run python scripts/judge.py check relu

#
# # 🟢 Easy: Implement ReLU
#
# Implement the **ReLU** (Rectified Linear Unit) activation function from scratch.
#
# $$\text{ReLU}(x) = \max(0, x)$$
#
# ### Signature
# ```python
# def relu(x: torch.Tensor) -> torch.Tensor:
#     ...
# ```
#
# ### Rules
# - Do **NOT** use `torch.relu`, `F.relu`, `torch.clamp`, or any built-in activation
# - Must support autograd (gradients should flow back)
#
# ### Example
# ```
# Input:  tensor([-2., -1., 0., 1., 2.])
# Output: tensor([ 0.,  0., 0., 1., 2.])
# ```

import torch

# ✏️ YOUR IMPLEMENTATION HERE

def relu(x: torch.Tensor) -> torch.Tensor:
    return x * (x > 0.0)

if __name__ == "__main__":
    # 🧪 Test your implementation (feel free to add more debug prints)
    x = torch.tensor([-2., -1., 0., 1., 2.])
    print("Input: ", x)
    print("Output:", relu(x))
    print("Shape: ", relu(x).shape)

# 進捗確認: ./torchcode status
