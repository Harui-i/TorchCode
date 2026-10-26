# 元データ: duoan/TorchCode (https://github.com/duoan/TorchCode)
# 採点: uv run python scripts/judge.py check grpo_loss

#
# # Solution: GRPO (Group Relative Policy Optimization) Loss
#
# Reference solution.

import torch
import torch.nn.functional as F
from torch import Tensor

# ✅ SOLUTION

def grpo_loss(logps: Tensor, rewards: Tensor, group_ids: Tensor,
              eps: float = 1e-5) -> Tensor:
    """Group Relative Policy Optimization (GRPO) loss.

    logps: (B,) policy log-probs for each sampled response
    rewards: (B,) scalar rewards for each response
    group_ids: (B,) integers, same id = same prompt/group
    returns: scalar loss (Tensor)
    """
    # Compute per-group normalized advantages A_i
    unique_ids = group_ids.unique()
    advantages = torch.empty_like(rewards)
    for gid in unique_ids:
        mask = group_ids == gid
        r_g = rewards[mask]
        mean_g = r_g.mean()
        std_g = r_g.std(unbiased=False)
        advantages[mask] = (r_g - mean_g) / (std_g + eps)

    # Stop gradient through advantages
    advantages_detached = advantages.detach()

    # GRPO objective: -E[A_i * logpi_i]
    return -(advantages_detached * logps).mean()

# Demo
logps = torch.tensor([0.0, -0.5, -1.0, -1.5])
rewards = torch.tensor([1.0, 0.8, 0.2, 0.0])
group_ids = torch.tensor([0, 0, 1, 1])
print('Loss:', grpo_loss(logps, rewards, group_ids).item())
