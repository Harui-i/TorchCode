# 元データ: duoan/TorchCode (https://github.com/duoan/TorchCode)
# 採点: uv run python scripts/judge.py check relu

#
# # 🟢 Solution: Implement ReLU
#
# Reference solution for the ReLU activation function.
#
# $$\text{ReLU}(x) = \max(0, x)$$

import torch

# ✅ SOLUTION

def relu(x: torch.Tensor) -> torch.Tensor:
    return x * (x > 0).float()

if __name__ == "__main__":
    # Verify
    x = torch.tensor([-2., -1., 0., 1., 2.])
    print("Input: ", x)
    print("Output:", relu(x))
