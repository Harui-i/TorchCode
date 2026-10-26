# 元データ: duoan/TorchCode (https://github.com/duoan/TorchCode)
# 採点: uv run python scripts/judge.py check cross_entropy

#
# # 🟢 Easy: Cross-Entropy Loss
#
# Implement **cross-entropy loss** from scratch.
#
# $$\text{CE}(x, y) = -\log\frac{e^{x_y}}{\sum_j e^{x_j}}$$
#
# ### Signature
# ```python
# def cross_entropy_loss(logits: Tensor, targets: Tensor) -> Tensor:
#     # logits: (B, C) float, targets: (B,) long indices
#     # Returns: scalar loss (mean over batch)
# ```
#
# ### Rules
# - Do NOT use `F.cross_entropy` or `nn.CrossEntropyLoss`
# - Must be numerically stable (use logsumexp trick)

import torch

# ✏️ YOUR IMPLEMENTATION HERE

def cross_entropy_loss(logits, targets):
    pass  # log_probs = logits - logsumexp(...)

if __name__ == "__main__":
    # 🧪 Debug
    logits = torch.randn(4, 10)
    targets = torch.randint(0, 10, (4,))
    print('Loss:', cross_entropy_loss(logits, targets))
    print('Ref: ', torch.nn.functional.cross_entropy(logits, targets))
