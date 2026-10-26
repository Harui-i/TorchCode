# 元データ: duoan/TorchCode (https://github.com/duoan/TorchCode)
# 採点: uv run python scripts/judge.py check ppo_loss

#
# # 🔴 Hard: PPO Clipped Loss
#
# Implement the **PPO (Proximal Policy Optimization)** **clipped surrogate loss**.
#
# Given:
# - `new_logps`: current policy log-probs $(B,)$
# - `old_logps`: old policy log-probs $(B,)$
# - `advantages`: advantage estimates $(B,)$
#
# Define the ratio
#
# $$ r_i = \exp(\text{new\_logps}_i - \text{old\_logps}_i). $$
#
# Then compute
# - $L^{\text{unclipped}}_i = r_i A_i$
# - $L^{\text{clipped}}_i = \operatorname{clip}(r_i, 1-\epsilon, 1+\epsilon) A_i$
#
# The loss is the negative batch mean of the elementwise minimum:
#
# $$
# \mathcal{L}_\text{PPO} = -\mathbb{E}_i\big[\min(L^{\text{unclipped}}_i, L^{\text{clipped}}_i)\big].
# $$
#
# Implementation notes: detach `old_logps` and `advantages` so gradients only flow through `new_logps`.
#
# ### Signature
# ```python
# from torch import Tensor
#
# def ppo_loss(new_logps: Tensor, old_logps: Tensor, advantages: Tensor,
#              clip_ratio: float = 0.2) -> Tensor:
#     """PPO clipped surrogate loss over a batch."""
# ```

import torch
import torch.nn.functional as F
from torch import Tensor

# ✏️ YOUR IMPLEMENTATION HERE

def ppo_loss(new_logps: Tensor, old_logps: Tensor, advantages: Tensor,
             clip_ratio: float = 0.2) -> Tensor:
    pass  # -mean(min(r * adv, clamp(r, 1-clip, 1+clip) * adv)) with gradients only through new_logps

if __name__ == "__main__":
    # 🧪 Debug
    new_logps = torch.tensor([0.0, -0.2, -0.4, -0.6])
    old_logps = torch.tensor([0.0, -0.1, -0.5, -0.5])
    advantages = torch.tensor([1.0, -1.0, 0.5, -0.5])
    print('Loss:', ppo_loss(new_logps, old_logps, advantages, clip_ratio=0.2))
