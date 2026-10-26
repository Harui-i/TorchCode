# 元データ: duoan/TorchCode (https://github.com/duoan/TorchCode)
# 採点: uv run python scripts/judge.py check bpe

#
# # 🔴 Hard: Byte-Pair Encoding (BPE)
#
# Implement a simple **BPE tokenizer** — the foundation of GPT/LLaMA tokenization.
#
# ### Signature
# ```python
# class SimpleBPE:
#     def __init__(self): ...
#     def train(self, corpus: list[str], num_merges: int): ...
#     def encode(self, text: str) -> list[str]: ...
# ```
#
# ### Algorithm (training)
# 1. Split each word into characters + `</w>` end marker
# 2. Count all adjacent pairs across the corpus
# 3. Merge the most frequent pair into a single token
# 4. Repeat for `num_merges` iterations

# No imports needed

# ✏️ YOUR IMPLEMENTATION HERE

class SimpleBPE:
    def __init__(self):
        self.merges = []

    def train(self, corpus, num_merges):
        pass  # iteratively find & merge most frequent pairs

    def encode(self, text):
        pass  # apply learned merges to split text

if __name__ == "__main__":
    # 🧪 Debug
    bpe = SimpleBPE()
    bpe.train(['low', 'low', 'low', 'lower', 'newest', 'widest'], num_merges=10)
    print('Merges:', bpe.merges[:5])
    print('Encode:', bpe.encode('low lower'))
