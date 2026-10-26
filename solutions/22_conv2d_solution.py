# 元データ: duoan/TorchCode (https://github.com/duoan/TorchCode)
# 採点: uv run python scripts/judge.py check conv2d

#
# # Solution: 2D Convolution
#
# Reference solution.

import torch
import torch.nn.functional as F

# ✅ SOLUTION

def my_conv2d(x, weight, bias=None, stride=1, padding=0):
    if padding > 0:
        x = F.pad(x, [padding] * 4)
    B, C_in, H, W = x.shape
    C_out, _, kH, kW = weight.shape
    H_out = (H - kH) // stride + 1
    W_out = (W - kW) // stride + 1
    patches = x.unfold(2, kH, stride).unfold(3, kW, stride)
    out = torch.einsum('bihwjk,oijk->bohw', patches, weight)
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1)
    return out

# Demo
x = torch.randn(1, 3, 8, 8)
w = torch.randn(16, 3, 3, 3)
print('Output:', my_conv2d(x, w).shape)
print('Match:', torch.allclose(my_conv2d(x, w), F.conv2d(x, w), atol=1e-4))
