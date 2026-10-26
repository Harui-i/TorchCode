# 元データ: duoan/TorchCode (https://github.com/duoan/TorchCode)
# 採点: uv run python scripts/judge.py check vit_patch

#
# # 🟠 Medium: ViT Patch Embedding
#
# Implement the **patch embedding** layer from Vision Transformer (ViT).
#
# ### Signature
# ```python
# class PatchEmbedding(nn.Module):
#     def __init__(self, img_size, patch_size, in_channels, embed_dim): ...
#     def forward(self, x: Tensor) -> Tensor:
#         # x: (B, C, H, W)
#         # Returns: (B, num_patches, embed_dim)
# ```
#
# ### Algorithm
# 1. Reshape image into non-overlapping patches: `(B, C, H, W)` → `(B, N, C*P*P)`
# 2. Project each patch: `nn.Linear(C*P*P, embed_dim)`
# 3. `num_patches = (img_size // patch_size) ** 2`

import torch
import torch.nn as nn

# ✏️ YOUR IMPLEMENTATION HERE

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        pass  # self.num_patches, self.proj

    def forward(self, x):
        pass  # reshape to patches, project

if __name__ == "__main__":
    # 🧪 Debug
    pe = PatchEmbedding(32, 8, 3, 64)
    x = torch.randn(2, 3, 32, 32)
    print('Output:', pe(x).shape)
    print('Patches:', pe.num_patches)
