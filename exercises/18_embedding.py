# 元データ: duoan/TorchCode (https://github.com/duoan/TorchCode)
# 採点: uv run python scripts/judge.py check embedding

#
# # 🟢 Easy: Embedding Layer
#
# Implement an **embedding lookup table** from scratch.
#
# ### Signature
# ```python
# class MyEmbedding(nn.Module):
#     def __init__(self, num_embeddings: int, embedding_dim: int): ...
#     def forward(self, indices: Tensor) -> Tensor: ...
# ```
#
# ### Rules
# - `self.weight`: `nn.Parameter` of shape `(num_embeddings, embedding_dim)`
# - Forward: index into weight matrix — `weight[indices]`
# - Do NOT use `nn.Embedding`

import torch
import torch.nn as nn

# ✏️ YOUR IMPLEMENTATION HERE

class MyEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        pass

    def forward(self, indices):
        pass

if __name__ == "__main__":
    # 🧪 Debug
    emb = MyEmbedding(10, 4)
    idx = torch.tensor([0, 3, 7])
    print('Output shape:', emb(idx).shape)
    print('Matches manual:', torch.equal(emb(idx)[0], emb.weight[0]))
