# 元データ: duoan/TorchCode (https://github.com/duoan/TorchCode)
# 採点: uv run python scripts/judge.py check dpo_loss

#
# # 🔴 Hard: DPO Loss
#
# Implement the **Direct Preference Optimization** loss — the standard loss for LLM alignment.
#
# $$\mathcal{L}_{\text{DPO}} = -\log \sigma\Big(\beta \big[\log\frac{\pi(y_w)}{\text{ref}(y_w)} - \log\frac{\pi(y_l)}{\text{ref}(y_l)}\big]\Big)$$
#
# ### Signature
# ```python
# def dpo_loss(policy_chosen_logps, policy_rejected_logps,
#              ref_chosen_logps, ref_rejected_logps, beta=0.1) -> Tensor:
#     # All inputs: (B,) log-probabilities
#     # Returns: scalar loss
# ```

import torch
import torch.nn.functional as F

# ✏️ YOUR IMPLEMENTATION HERE

def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta=0.1):
    pass  # -log(sigmoid(beta * (chosen_reward - rejected_reward)))

if __name__ == "__main__":
    # 🧪 Debug
    chosen = torch.tensor([0.0, 0.0])
    rejected = torch.tensor([-5.0, -5.0])
    ref_c = torch.tensor([-1.0, -1.0])
    ref_r = torch.tensor([-1.0, -1.0])
    print('Loss:', dpo_loss(chosen, rejected, ref_c, ref_r, beta=0.1).item())
