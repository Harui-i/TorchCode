# 元データ: duoan/TorchCode (https://github.com/duoan/TorchCode)
# 採点: uv run python scripts/judge.py check int8_quantization

#
# # 🔴 Hard: INT8 Quantized Linear
#
# Implement a **post-training quantized linear layer** using INT8 weights.
#
# ### Signature
# ```python
# class Int8Linear(nn.Module):
#     def __init__(self, weight: Tensor, bias: Tensor = None): ...
#     def forward(self, x: Tensor) -> Tensor: ...
# ```
#
# ### Quantization (per-channel)
# 1. `scale = weight.abs().max(dim=1) / 127`
# 2. `weight_int8 = round(weight / scale).clamp(-128, 127).to(int8)`
# 3. Store as `register_buffer` (not trainable)
# 4. Forward: dequantize (`int8.float() * scale`) then matmul

import torch
import torch.nn as nn

# ✏️ YOUR IMPLEMENTATION HERE

class Int8Linear(nn.Module):
    def __init__(self, weight, bias=None):
        super().__init__()
        pass  # quantize weight, register buffers

    def forward(self, x):
        pass  # dequantize and matmul

if __name__ == "__main__":
    # 🧪 Debug
    w = torch.randn(8, 4)
    q = Int8Linear(w)
    x = torch.randn(2, 4)
    print('Output:', q(x).shape)
    print('dtype:', q.weight_int8.dtype)
    print('Max quant error:', (w - q.weight_int8.float() * q.scale).abs().max().item())
