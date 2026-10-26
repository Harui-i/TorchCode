# 元データ: duoan/TorchCode (https://github.com/duoan/TorchCode)
# 採点: uv run python scripts/judge.py check dpo_loss

#
# # Solution: DPO (Direct Preference Optimization) Loss
#
# Reference solution.

import torch
import torch.nn.functional as F

# ✅ SOLUTION

def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    chosen_rewards = beta * (policy_chosen_logps - ref_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - ref_rejected_logps)
    return -F.logsigmoid(chosen_rewards - rejected_rewards).mean()

# Demo
chosen = torch.tensor([0.0, 0.0])
rejected = torch.tensor([-5.0, -5.0])
ref_c = torch.tensor([-1.0, -1.0])
ref_r = torch.tensor([-1.0, -1.0])
print('Loss:', dpo_loss(chosen, rejected, ref_c, ref_r, beta=0.1).item())
