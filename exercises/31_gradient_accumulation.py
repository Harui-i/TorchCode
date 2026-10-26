# 元データ: duoan/TorchCode (https://github.com/duoan/TorchCode)
# 採点: uv run python scripts/judge.py check gradient_accumulation

#
# # 🟢 Easy: Gradient Accumulation
#
# Implement a **training step with gradient accumulation** — simulating large batches with limited memory.
#
# ### Signature
# ```python
# def accumulated_step(model, optimizer, loss_fn, micro_batches) -> float:
#     # micro_batches: list of (input, target) tuples
#     # Returns: average loss (float)
# ```
#
# ### Algorithm
# 1. `optimizer.zero_grad()`
# 2. For each `(x, y)` in micro_batches: `loss = loss_fn(model(x), y) / len(micro_batches)`, then `loss.backward()`
# 3. `optimizer.step()`
# 4. Return total accumulated loss
#
# The key insight: dividing each loss by `n` before backward makes accumulated gradients equal to a single large-batch gradient.

import torch
import torch.nn as nn

# ✏️ YOUR IMPLEMENTATION HERE

def accumulated_step(model, optimizer, loss_fn, micro_batches):
    pass  # zero_grad, loop (forward, scale loss, backward), step

if __name__ == "__main__":
    # 🧪 Debug
    model = nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    loss = accumulated_step(model, opt, nn.MSELoss(),
        [(torch.randn(2, 4), torch.randn(2, 2)) for _ in range(4)])
    print('Loss:', loss)
