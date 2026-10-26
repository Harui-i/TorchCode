# 元データ: duoan/TorchCode (https://github.com/duoan/TorchCode)
# 採点: uv run python scripts/judge.py check cosine_lr

#
# # 🟠 Medium: Cosine LR Scheduler with Warmup
#
# Implement a **cosine learning rate schedule** with linear warmup.
#
# ### Signature
# ```python
# def cosine_lr_schedule(step, total_steps, warmup_steps, max_lr, min_lr=0.0) -> float:
# ```
#
# ### Schedule
# ```
# step < warmup:  lr = max_lr * step / warmup_steps  (linear ramp)
# step >= warmup: lr = min_lr + 0.5*(max_lr-min_lr)*(1 + cos(π * progress))
# ```
# where `progress = (step - warmup) / (total - warmup)`

import math

# ✏️ YOUR IMPLEMENTATION HERE

def cosine_lr_schedule(step, total_steps, warmup_steps, max_lr, min_lr=0.0):
    pass  # warmup then cosine decay

if __name__ == "__main__":
    # 🧪 Debug
    lrs = [cosine_lr_schedule(i, 100, 10, 0.001) for i in range(101)]
    print(f'Start: {lrs[0]:.6f}')
    print(f'Warmup end: {lrs[10]:.6f}')
    print(f'Mid: {lrs[55]:.6f}')
    print(f'End: {lrs[100]:.6f}')
