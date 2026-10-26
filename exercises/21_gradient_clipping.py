# 元データ: duoan/TorchCode (https://github.com/duoan/TorchCode)
# 採点: uv run python scripts/judge.py check gradient_clipping

#
# # 🟢 Easy: Gradient Norm Clipping
#
# Implement **gradient norm clipping** — a training stability technique.
#
# ### Signature
# ```python
# def clip_grad_norm(parameters, max_norm: float) -> float:
#     # Clip gradients in-place so total norm <= max_norm
#     # Returns the original (unclipped) total norm
# ```
#
# ### Algorithm
# 1. Compute total norm: `sqrt(sum(p.grad.norm()^2 for p in parameters))`
# 2. If total > max_norm: scale all grads by `max_norm / total`
# 3. Return original total norm

import torch

# ✏️ YOUR IMPLEMENTATION HERE

def clip_grad_norm(parameters, max_norm):
    pass  # compute total norm, clip if needed, return original norm

if __name__ == "__main__":
    # 🧪 Debug
    p = torch.randn(100, requires_grad=True)
    (p * 10).sum().backward()
    print('Before:', p.grad.norm().item())
    orig = clip_grad_norm([p], max_norm=1.0)
    print('After: ', p.grad.norm().item())
    print('Original norm:', orig)
