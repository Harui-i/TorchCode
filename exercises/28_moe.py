# 元データ: duoan/TorchCode (https://github.com/duoan/TorchCode)
# 採点: uv run python scripts/judge.py check moe

#
# # 🔴 Hard: Mixture of Experts (MoE)
#
# Implement a **Mixture of Experts** layer (Mixtral / Switch Transformer style).
#
# ### Signature
# ```python
# class MixtureOfExperts(nn.Module):
#     def __init__(self, d_model, d_ff, num_experts, top_k=2): ...
#     def forward(self, x: Tensor) -> Tensor:
#         # x: (B, S, D) -> (B, S, D)
# ```
#
# ### Architecture
# - `self.router`: `nn.Linear(d_model, num_experts)` — gating network
# - `self.experts`: `nn.ModuleList` of MLPs `(Linear→ReLU→Linear)`
# - For each token: select top-k experts, compute weighted sum of their outputs

import torch
import torch.nn as nn

# ✏️ YOUR IMPLEMENTATION HERE

class MixtureOfExperts(nn.Module):
    def __init__(self, d_model, d_ff, num_experts, top_k=2):
        super().__init__()
        pass  # router + experts

    def forward(self, x):
        pass  # route tokens to top-k experts

if __name__ == "__main__":
    # 🧪 Debug
    moe = MixtureOfExperts(32, 64, num_experts=4, top_k=2)
    x = torch.randn(2, 8, 32)
    print('Output:', moe(x).shape)
    print('Params:', sum(p.numel() for p in moe.parameters()))
