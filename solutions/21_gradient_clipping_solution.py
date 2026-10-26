# 元データ: duoan/TorchCode (https://github.com/duoan/TorchCode)
# 採点: uv run python scripts/judge.py check gradient_clipping

#
# # Solution: Gradient Norm Clipping
#
# Reference solution.

import torch

# ✅ SOLUTION

def clip_grad_norm(parameters, max_norm):
    parameters = [p for p in parameters if p.grad is not None]
    total_norm = torch.sqrt(sum(p.grad.norm() ** 2 for p in parameters))
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.mul_(clip_coef)
    return total_norm.item()

# Demo
p = torch.randn(100, requires_grad=True)
(p * 10).sum().backward()
print('Before:', p.grad.norm().item())
orig = clip_grad_norm([p], max_norm=1.0)
print('After: ', p.grad.norm().item())
print('Returned:', orig)
