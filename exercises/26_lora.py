# 元データ: duoan/TorchCode (https://github.com/duoan/TorchCode)
# 採点: uv run python scripts/judge.py check lora

#
# # 🟠 Medium: LoRA (Low-Rank Adaptation)
#
# Implement **LoRA** — parameter-efficient fine-tuning for large models.
#
# $$h = W_0 x + \frac{\alpha}{r} B A x$$
#
# ### Signature
# ```python
# class LoRALinear(nn.Module):
#     def __init__(self, in_features, out_features, rank, alpha=1.0): ...
#     def forward(self, x: Tensor) -> Tensor: ...
# ```
#
# ### Requirements
# - `self.linear`: frozen `nn.Linear` (weight & bias `requires_grad=False`)
# - `self.lora_A`: `nn.Parameter(rank, in_features)` — random init
# - `self.lora_B`: `nn.Parameter(out_features, rank)` — **zero** init
# - Scaling: `alpha / rank`

import torch
import torch.nn as nn

# ✏️ YOUR IMPLEMENTATION HERE

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank, alpha=1.0):
        super().__init__()
        pass  # frozen linear + lora_A + lora_B

    def forward(self, x):
        pass  # base + lora

if __name__ == "__main__":
    # 🧪 Debug
    layer = LoRALinear(16, 8, rank=4)
    x = torch.randn(2, 16)
    print('Output:', layer(x).shape)
    print('Trainable:', sum(p.numel() for p in layer.parameters() if p.requires_grad))
    print('Total:    ', sum(p.numel() for p in layer.parameters()))
