# 元データ: duoan/TorchCode (https://github.com/duoan/TorchCode)
# 採点: uv run python scripts/judge.py check gradient_accumulation

#
# # Solution: Gradient Accumulation
#
# Reference solution.

import torch
import torch.nn as nn

# ✅ SOLUTION

def accumulated_step(model, optimizer, loss_fn, micro_batches):
    optimizer.zero_grad()
    total_loss = 0.0
    n = len(micro_batches)
    for x, y in micro_batches:
        loss = loss_fn(model(x), y) / n
        loss.backward()
        total_loss += loss.item()
    optimizer.step()
    return total_loss

# Demo
model = nn.Linear(4, 2)
opt = torch.optim.SGD(model.parameters(), lr=0.01)
loss = accumulated_step(model, opt, nn.MSELoss(),
    [(torch.randn(2, 4), torch.randn(2, 2)) for _ in range(4)])
print('Accumulated loss:', loss)
