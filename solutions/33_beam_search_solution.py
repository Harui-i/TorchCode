# 元データ: duoan/TorchCode (https://github.com/duoan/TorchCode)
# 採点: uv run python scripts/judge.py check beam_search

#
# # Solution: Beam Search Decoding
#
# Reference solution.

import torch

# ✅ SOLUTION

def beam_search(log_prob_fn, start_token, max_len, beam_width, eos_token):
    beams = [(0.0, [start_token])]
    completed = []
    for _ in range(max_len):
        candidates = []
        for score, seq in beams:
            if seq[-1] == eos_token:
                completed.append((score, seq))
                continue
            log_probs = log_prob_fn(torch.tensor(seq))
            topk_lp, topk_idx = log_probs.topk(beam_width)
            for j in range(beam_width):
                candidates.append((score + topk_lp[j].item(), seq + [topk_idx[j].item()]))
        if not candidates:
            break
        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:beam_width]
    all_seqs = completed + beams
    all_seqs.sort(key=lambda x: x[0], reverse=True)
    return all_seqs[0][1]

# Demo
def simple_fn(tokens):
    lp = torch.full((5,), -10.0)
    lp[min(len(tokens), 4)] = 0.0
    return lp
seq = beam_search(simple_fn, start_token=0, max_len=5, beam_width=2, eos_token=4)
print('Sequence:', seq)
