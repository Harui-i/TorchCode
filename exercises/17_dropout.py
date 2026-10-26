# 元データ: duoan/TorchCode (https://github.com/duoan/TorchCode)
# 採点: uv run python scripts/judge.py check dropout

#
# # 🟢 Easy: Implement Dropout
#
# Implement **Dropout** regularization from scratch.
#
# ### Signature
# ```python
# class MyDropout(nn.Module):
#     def __init__(self, p: float = 0.5): ...
#     def forward(self, x: Tensor) -> Tensor: ...
# ```
#
# ### Rules
# - During **training**: zero each element with probability `p`, scale remaining by `1/(1-p)`
# - During **eval**: return input unchanged (identity)
# - Do NOT use `nn.Dropout` or `F.dropout`

import torch
import torch.nn as nn

# ✏️ YOUR IMPLEMENTATION HERE

class MyDropout(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        pass

    def forward(self, x):
        pass

if __name__ == "__main__":
    # 🧪 Debug
    d = MyDropout(p=0.5)
    d.train()
    x = torch.ones(10)
    print('Train:', d(x))
    d.eval()
    print('Eval: ', d(x))
