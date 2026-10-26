# 元データ: duoan/TorchCode (https://github.com/duoan/TorchCode)
# 採点: uv run python scripts/judge.py check conv2d

#
# # 🟠 Medium: 2D Convolution
#
# Implement **2D convolution** from scratch.
#
# ### Signature
# ```python
# def my_conv2d(x, weight, bias=None, stride=1, padding=0):
#     # x: (B, C_in, H, W), weight: (C_out, C_in, kH, kW)
#     # Returns: (B, C_out, H_out, W_out)
# ```
#
# ### Rules
# - Do NOT use `F.conv2d` or `nn.Conv2d`
# - Support `stride` and `padding` parameters
# - `F.pad` for zero-padding is allowed

import torch
import torch.nn.functional as F

# ✏️ YOUR IMPLEMENTATION HERE

def my_conv2d(x, weight, bias=None, stride=1, padding=0):
    pass  # extract patches, apply kernel, handle stride/padding

if __name__ == "__main__":
    # 🧪 Debug
    x = torch.randn(1, 3, 8, 8)
    w = torch.randn(16, 3, 3, 3)
    print('Output:', my_conv2d(x, w).shape)
    print('Match:', torch.allclose(my_conv2d(x, w), F.conv2d(x, w), atol=1e-4))
