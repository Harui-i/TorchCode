# 元データ: duoan/TorchCode (https://github.com/duoan/TorchCode)
# 採点: uv run python scripts/judge.py check flash_attention

#
# # 🔴 Hard: Flash Attention (Tiled)
#
# Implement **tiled attention with online softmax** — the core idea behind Flash Attention.
#
# ### Signature
# ```python
# def flash_attention(Q, K, V, block_size=32) -> Tensor:
#     # Q, K, V: (B, S, D)
#     # Returns: (B, S, D) — same as standard attention
# ```
#
# ### Key Insight
# Instead of materializing the full S×S attention matrix, process in blocks:
# 1. For each Q-block, iterate over K/V blocks
# 2. Use **online softmax**: track running `max` and `sum`
# 3. Rescale accumulator when max changes: `acc *= exp(old_max - new_max)`
# 4. Final: `output = acc / row_sum`
#
# Must give **identical** results to standard softmax attention.

import torch
import math

# ✏️ YOUR IMPLEMENTATION HERE

def flash_attention(Q, K, V, block_size=32):
    # Process Q in blocks, iterate K/V blocks with online softmax
    pass

if __name__ == "__main__":
    # 🧪 Debug
    import math
    Q, K, V = torch.randn(1, 8, 4), torch.randn(1, 8, 4), torch.randn(1, 8, 4)
    out = flash_attention(Q, K, V, block_size=4)
    scores = torch.bmm(Q, K.transpose(1,2)) / math.sqrt(4)
    ref = torch.bmm(torch.softmax(scores, dim=-1), V)
    print('Match:', torch.allclose(out, ref, atol=1e-4))
