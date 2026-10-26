# 元データ: duoan/TorchCode (https://github.com/duoan/TorchCode)
# 採点: uv run python scripts/judge.py check grpo_loss

#
# # 🔴 Hard: GRPO Loss
#
# Implement the **Group Relative Policy Optimization (GRPO)** loss — a group-wise, baseline-subtracted REINFORCE objective commonly used in RLAIF (reinforcement learning from AI feedback).
#
# Given a batch of log-probabilities, scalar rewards, and group ids (one group per prompt), define the within-group normalized advantages:
#
# $$A_i = \frac{r_i - \bar r_{g(i)}}{\text{std}_{g(i)} + \epsilon}$$
#
# where \(\bar r_{g(i)}\) and \(\text{std}_{g(i)}\) are the mean and standard deviation of rewards in the group of example \(i\).
#
# The GRPO loss is then the negative advantage-weighted log-probability:
#
# $$\mathcal{L}_{\text{GRPO}} = -\mathbb{E}_i \big[\,\text{stop\_grad}(A_i)\, \log \pi_\theta(y_i)\big].$$
#
# ### Signature
# ```python
# from torch import Tensor
#
# def grpo_loss(logps: Tensor, rewards: Tensor, group_ids: Tensor,
#               eps: float = 1e-5) -> Tensor:
#     """GRPO loss over a batch.
#
#     logps: (B,) policy log-probs for each sampled response
#     rewards: (B,) scalar rewards for each response
#     group_ids: (B,) integers, same id = same prompt/group
#     returns: scalar loss (Tensor)
#     """
# ```

import torch
import torch.nn.functional as F

# ✏️ YOUR IMPLEMENTATION HERE

from torch import Tensor

def grpo_loss(logps: Tensor, rewards: Tensor, group_ids: Tensor,
              eps: float = 1e-5) -> Tensor:
    pass  # compute normalized advantages per group and return -mean(adv.detach() * logps)

if __name__ == "__main__":
    # 🧪 Debug
    logps = torch.tensor([0.0, -0.5, -1.0, -1.5])
    rewards = torch.tensor([1.0, 0.8, 0.2, 0.0])
    group_ids = torch.tensor([0, 0, 1, 1])
    print('Loss:', grpo_loss(logps, rewards, group_ids).item())
