# 元データ: duoan/TorchCode (https://github.com/duoan/TorchCode)
# 採点: uv run python scripts/judge.py check adam

#
# # 🟠 Medium: Adam Optimizer
#
# Implement the **Adam** optimizer from scratch.
#
# ### Signature
# ```python
# class MyAdam:
#     def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8): ...
#     def step(self): ...
#     def zero_grad(self): ...
# ```
#
# ### Algorithm (per parameter)
# ```
# m = β1 * m + (1-β1) * grad
# v = β2 * v + (1-β2) * grad²
# m̂ = m / (1 - β1ᵗ)    # bias correction
# v̂ = v / (1 - β2ᵗ)
# p -= lr * m̂ / (√v̂ + ε)
# ```

import torch

# ✏️ YOUR IMPLEMENTATION HERE

class MyAdam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        pass  # store params, init m and v to zeros

    def step(self):
        pass  # update params using Adam rule

    def zero_grad(self):
        pass  # zero all gradients

if __name__ == "__main__":
    # 🧪 Debug
    torch.manual_seed(0)
    w = torch.randn(4, 3, requires_grad=True)
    opt = MyAdam([w], lr=0.01)
    for i in range(5):
        loss = (w ** 2).sum()
        loss.backward()
        opt.step()
        opt.zero_grad()
        print(f'Step {i}: loss={loss.item():.4f}')
