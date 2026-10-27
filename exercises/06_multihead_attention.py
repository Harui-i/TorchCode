# 元データ: duoan/TorchCode (https://github.com/duoan/TorchCode)
# 採点: uv run python scripts/judge.py check mha

#
# # 🔴 Hard: Multi-Head Attention
#
# Implement **Multi-Head Attention** from scratch — the core building block of the Transformer.
#
# $$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$
# $$\text{head}_i = \text{Attention}(Q W_i^Q,\; K W_i^K,\; V W_i^V)$$
#
# ### Signature
# ```python
# class MultiHeadAttention:
#     def __init__(self, d_model: int, num_heads: int): ...
#     def forward(self, Q, K, V) -> torch.Tensor: ...
# ```
#
# ### Requirements
# - Use `nn.Linear(d_model, d_model)` for `self.W_q`, `self.W_k`, `self.W_v`, `self.W_o`
# - `d_k = d_model // num_heads` per head
# - `forward(Q, K, V)`: Q is `(B, seq_q, d_model)`, K/V are `(B, seq_k, d_model)`
# - Must support **cross-attention** (`seq_q != seq_k`)
# - Do **NOT** use `torch.nn.MultiheadAttention`
# - You **may** use `torch.softmax` and `torch.matmul`
#
# ### Steps
# 1. Project: `q = self.W_q(Q)`, `k = self.W_k(K)`, `v = self.W_v(V)`
# 2. Reshape to `(B, num_heads, seq, d_k)`
# 3. Scaled dot-product attention per head
# 4. Concat heads → `(B, seq_q, d_model)`
# 5. Output projection: `self.W_o(concat)`

import torch
import torch.nn as nn
import math

# ✏️ YOUR IMPLEMENTATION HERE


class MultiHeadAttention:
    def __init__(self, d_model: int, num_heads: int):
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        pass  # Initialize W_q, W_k, W_v, W_o

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
        assert K.shape == V.shape
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)

        B = Q.size(0)
        seq_q = Q.size(1)
        seq_k = K.size(1)
        d_k = Q.size(-1) // self.num_heads
        d_model = self.d_model
        # Q: (B, seq_q, d_model)
        # K V: (B, seq_k, d_model)
        # これを(B, num_heads, seq, d_k) にreshapeするのどうやるんだ
        # このreshapeあってる？
        Q = torch.reshape(Q, torch.Size([B, seq_q, self.num_heads, d_k])).transpose(
            1, 2
        )
        K = torch.reshape(K, torch.Size([B, seq_k, self.num_heads, d_k])).transpose(
            1, 2
        )
        V = torch.reshape(V, torch.Size([B, seq_k, self.num_heads, d_k])).transpose(
            1, 2
        )

        # Q: (B, num_heads, seq_q, d_k)
        # K: (B, num_heads, seq_k, d_k)
        # V: (B, num_heads, seq_k, d_k)

        # step3 scaled dot-product attention per head
        # torchにdot-product attentionってないんかね
        # torch.matmulとtorch.softmaxを使うって書いてあるし、自前実装するんだろうな
        # torch.bmmじゃないんだ

        # score = torch.matmul(Q, K.transpose(-1, -2))  # [B, num_heads, seq_q, seq_k]
        # weight = torch.softmax(score, dim=-1)
        # out = torch.matmul(weight, V)  # [B. num_heads, seq_q, d_k]
        # out = out.reshape(Q.size(0), Q.size(2), self.d_model)  # [B, seq_q, d_model]

        """
        なんかあわない 数値が違うんだろうな hintをみた

        Reshpae to (B, heads, S, d_k), SDPA per head,  concat, output projection と書いてあった

        reshapeはできてる気がする. SDPAってなんすか→ScaledDotProductAttentionらしい
        """
        print(f"{Q.shape=}")
        print(f"{K.shape=}")
        print(f"{V.shape=}")
        out = nn.functional.scaled_dot_product_attention(
            Q, K, V
        )  # [B, num_heads, seq_q, d_model]
        print(f"SPDA out shape: {out.shape}")
        assert out.shape == Q.shape, "out.shape == Q.shape"

        out = out.transpose(1, 2)  # [B, seq_q, num_heads, d_k]
        out = out.reshape([B, seq_q, d_model])

        # out = out.reshape(
        #    Q.size(0), Q.size(2), self.d_model
        # )  # これはshapeは合うが数値は合わない
        # これが合わない理由はたぶんconcatとreshapeが違うから。reshapeするというのがどういう操作なのかよくわかってないｗ
        # で、concatするっていっても,,,,てかreshapeとconcatが違うなら
        # Q = torch.reshape(Q, torch.Size([V.siez(0), num_heads, V.size(1), d_k])) の操作も違うんじゃね！？ｗ
        # torch.catはtensorsをいくつかくっつけるやつだった。今回のは一つのtensor軸を潰すみたいな感じだよな
        # torch.catはtorch.split, torch.chunkの逆操作らしい
        # squeezeか？→squeezeはtensor shapeにある1を消すやつだった　違う. どっちかっていうとcat,split,chunkのほうが近い

        print(f"{out.shape=}")

        assert out.shape == torch.Size([B, seq_q, self.d_model]), (
            "out.shape == [B, seq_q, d_model]"
        )
        return self.W_o(out)  # こういうのもprojっていうんかな


if __name__ == "__main__":
    # 🧪 Debug
    torch.manual_seed(0)
    mha = MultiHeadAttention(d_model=32, num_heads=4)
    print("W_q type:", type(mha.W_q))  # should be nn.Linear
    print("W_q.weight shape:", mha.W_q.weight.shape)  # (32, 32)

    x = torch.randn(2, 6, 32)
    out = mha.forward(x, x, x)
    print("Output shape:", out.shape)  # (2, 6, 32)

    # Cross-attention
    Q = torch.randn(1, 3, 32)
    K = torch.randn(1, 7, 32)
    V = torch.randn(1, 7, 32)
    out2 = mha.forward(Q, K, V)
    print("Cross-attn shape:", out2.shape)  # (1, 3, 32)
