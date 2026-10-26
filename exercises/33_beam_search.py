# 元データ: duoan/TorchCode (https://github.com/duoan/TorchCode)
# 採点: uv run python scripts/judge.py check beam_search

#
# # 🟠 Medium: Beam Search Decoding
#
# Implement **beam search** — the classic decoding algorithm for sequence generation.
#
# ### Signature
# ```python
# def beam_search(log_prob_fn, start_token, max_len, beam_width, eos_token) -> list[int]:
#     # log_prob_fn: takes token list, returns (V,) log-probabilities
#     # Returns: best sequence (list of ints)
# ```
#
# ### Algorithm
# 1. Start with `[(0.0, [start_token])]`
# 2. Each step: expand each beam with top-k next tokens
# 3. Keep top `beam_width` beams by total log-probability
# 4. Stop when best beam ends with `eos_token` or `max_len` reached

import torch

# ✏️ YOUR IMPLEMENTATION HERE

def beam_search(log_prob_fn, start_token, max_len, beam_width, eos_token):
    pass  # maintain beams, expand, prune, return best

if __name__ == "__main__":
    # 🧪 Debug
    def simple_fn(tokens):
        lp = torch.full((5,), -10.0)
        lp[min(len(tokens), 4)] = 0.0
        return lp
    seq = beam_search(simple_fn, start_token=0, max_len=5, beam_width=2, eos_token=4)
    print('Sequence:', seq)
