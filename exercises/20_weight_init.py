# 元データ: duoan/TorchCode (https://github.com/duoan/TorchCode)
# 採点: uv run python scripts/judge.py check weight_init

#
# # 🟢 Easy: Kaiming Initialization
#
# Implement **Kaiming (He) normal initialization** for weight tensors.
#
# $$W \sim \mathcal{N}(0, \text{std}^2) \quad \text{where} \quad \text{std} = \sqrt{\frac{2}{\text{fan\_in}}}$$
#
# ### Signature
# ```python
# def kaiming_init(weight: Tensor) -> Tensor:
#     # Initialize weight in-place with Kaiming normal
#     # fan_in = weight.shape[1]
#     # Returns the weight tensor
# ```

import torch
import math

# ✏️ YOUR IMPLEMENTATION HERE

def kaiming_init(weight):
    pass  # fill with normal(0, sqrt(2/fan_in))

if __name__ == "__main__":
    # 🧪 Debug
    import math
    w = torch.empty(256, 512)
    kaiming_init(w)
    print(f'Mean: {w.mean():.4f} (expect ~0)')
    print(f'Std:  {w.std():.4f} (expect {math.sqrt(2/512):.4f})')
