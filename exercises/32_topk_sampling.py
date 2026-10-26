# 元データ: duoan/TorchCode (https://github.com/duoan/TorchCode)
# 採点: uv run python scripts/judge.py check topk_sampling

#
# # 🟠 Medium: Top-k / Top-p (Nucleus) Sampling
#
# Implement **sampling with top-k and top-p filtering** — the standard LLM decoding strategy.
#
# ### Signature
# ```python
# def sample_top_k_top_p(logits, top_k=0, top_p=1.0, temperature=1.0) -> int:
#     # logits: (V,) unnormalized log-probabilities
#     # Returns: sampled token index
# ```
#
# ### Algorithm
# 1. Scale by temperature: `logits /= temperature`
# 2. Top-k: keep only top-k logits, set rest to `-inf`
# 3. Top-p: sort by prob, mask tokens where cumulative prob exceeds p
# 4. Sample from filtered distribution

import torch

# ✏️ YOUR IMPLEMENTATION HERE

def sample_top_k_top_p(logits, top_k=0, top_p=1.0, temperature=1.0):
    pass  # temperature, top-k filter, top-p filter, sample

if __name__ == "__main__":
    # 🧪 Debug
    logits = torch.tensor([1.0, 5.0, 2.0, 0.5])
    print('top_k=1:', sample_top_k_top_p(logits.clone(), top_k=1))
    print('top_p=0.5:', sample_top_k_top_p(logits.clone(), top_p=0.5))
    print('temp=0.01:', sample_top_k_top_p(logits.clone(), temperature=0.01))
