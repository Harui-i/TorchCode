# 元データ: duoan/TorchCode (https://github.com/duoan/TorchCode)
# 採点: uv run python scripts/judge.py check cosine_lr

#
# # Solution: Cosine LR Scheduler with Warmup
#
# Reference solution.

import math

# ✅ SOLUTION

def cosine_lr_schedule(step, total_steps, warmup_steps, max_lr, min_lr=0.0):
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    if step >= total_steps:
        return min_lr
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1.0 + math.cos(math.pi * progress))

# Demo
lrs = [cosine_lr_schedule(i, 100, 10, 0.001) for i in range(101)]
print(f'Start: {lrs[0]:.6f}, Warmup end: {lrs[10]:.6f}, Mid: {lrs[55]:.6f}, End: {lrs[100]:.6f}')
