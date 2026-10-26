# 元データ: duoan/TorchCode (https://github.com/duoan/TorchCode)
# 採点: uv run python scripts/judge.py check gelu

#
# # 🟢 Easy: GELU Activation
#
# Implement the **GELU** (Gaussian Error Linear Unit) activation.
#
# $$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot 0.5 \cdot (1 + \text{erf}(x / \sqrt{2}))$$
#
# ### Signature
# ```python
# def my_gelu(x: Tensor) -> Tensor: ...
# ```
#
# ### Rules
# - Do NOT use `F.gelu`, `nn.GELU`, or `torch.nn.functional.gelu`
# - Use `torch.erf` for the exact version

import torch
import math

# ✏️ YOUR IMPLEMENTATION HERE

def my_gelu(x):
    pass

if __name__ == "__main__":
    # 🧪 Debug
    x = torch.tensor([-2., -1., 0., 1., 2.])
    print('Output:', my_gelu(x))
    print('Ref:   ', torch.nn.functional.gelu(x))
