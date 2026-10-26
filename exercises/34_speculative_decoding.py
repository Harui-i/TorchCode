# 元データ: duoan/TorchCode (https://github.com/duoan/TorchCode)
# 採点: uv run python scripts/judge.py check speculative_decoding

#
# # 🔴 Hard: Speculative Decoding
#
# Implement the **acceptance/rejection step** of speculative decoding — a technique for accelerating LLM inference.
#
# ### Signature
# ```python
# def speculative_decode(target_probs, draft_probs, draft_tokens) -> list[int]:
#     # target_probs: (K, V) from target (large) model
#     # draft_probs: (K, V) from draft (small) model
#     # draft_tokens: (K,) tokens sampled by draft model
#     # Returns: list of accepted tokens (1 to K)
# ```
#
# ### Algorithm
# For each position i = 0, ..., K-1:
# 1. `ratio = target_probs[i, token_i] / draft_probs[i, token_i]`
# 2. Accept with probability `min(1, ratio)`
# 3. If rejected: sample from `normalize(max(0, target - draft))`, append, and stop

import torch

# ✏️ YOUR IMPLEMENTATION HERE

def speculative_decode(target_probs, draft_probs, draft_tokens):
    pass  # accept/reject loop

if __name__ == "__main__":
    # 🧪 Debug
    torch.manual_seed(0)
    probs = torch.softmax(torch.randn(4, 10), dim=-1)
    tokens = torch.tensor([2, 5, 1, 8])
    print('Perfect draft:', speculative_decode(probs, probs, tokens))
    target = torch.softmax(torch.randn(4, 10), dim=-1)
    draft = torch.softmax(torch.randn(4, 10), dim=-1)
    print('Random draft:', speculative_decode(target, draft, tokens))
