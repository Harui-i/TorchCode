# 元データ: duoan/TorchCode (https://github.com/duoan/TorchCode)
# 採点: uv run python scripts/judge.py check attention

#
# # 🔴 Hard: Softmax Attention
#
# Implement the core attention mechanism used in Transformers.
#
# $$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
#
# ### Signature
# ```python
# def scaled_dot_product_attention(
#     Q: torch.Tensor,  # (batch, seq_q, d_k)
#     K: torch.Tensor,  # (batch, seq_k, d_k)
#     V: torch.Tensor,  # (batch, seq_k, d_v)
# ) -> torch.Tensor:   # (batch, seq_q, d_v)
#     ...
# ```
#
# ### Rules
# - Do **NOT** use `F.scaled_dot_product_attention`
# - You **may** use `torch.softmax` and `torch.bmm`
# - Must support autograd
# - Must handle cross-attention (seq_q ≠ seq_k)

import torch
import math

# ✏️ YOUR IMPLEMENTATION HERE


def scaled_dot_product_attention(Q, K, V):
    """
        torch.softmaxはつかっていいんだ
        torch.bm　なんだそれ
        seq_q != seq_kの場合も考慮しなきゃいけないのか

        とはいえまずはQK^Tのところを扱うか
        Q @ K.T と書いたらエラー;;
        /home/harui/learn/torch-learn/exercises/05_attention.py:34: UserWarning: The use of `x.T` on tensors of dimension other than 2 to reverse their shape is deprecated and it will throw an error in a future release. Consider `x.mT` to transpose batches of matrices or `x.permute(*torch.arange(x.ndim - 1, -1, -1))` to reverse the dimensions of a tensor. (Triggered internally at /__w/pytorch/pytorch/aten/src/ATen/native/TensorShape.cpp:4320.)
      x = Q @ K.T
    Traceback (most recent call last):
      File "/home/harui/learn/torch-learn/exercises/05_attention.py", line 46, in <module>
        out = scaled_dot_product_attention(Q, K, V)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
      File "/home/harui/learn/torch-learn/exercises/05_attention.py", line 34, in scaled_dot_product_attention
        x = Q @ K.T
            ~~^~~~~
    RuntimeError: The size of tensor a (2) must match the size of tensor b (8) at non-singleton dimension 0


        警告は、x.Tを2次元Tensor以外で使うのはDeprecatedだよという話か

        Q @ K.Tは 2と8のdimがあってないというお話だった。
        Q K:  (2,4,8) を与えていて、

        K.T のshapeは (8,4,2) だった。 このエラーの意味はよくわからないけど、まあtransposeを使うべきじゃないということはわかった

    torch.bmmは？
    https://docs.pytorch.org/docs/2.14/generated/torch.bmm.html#torch.bmm

    batch matrix matrix prodcutらしい

    x = torch.bmm(Q,K_transposed)みたいな形にしたい
    Kをどう変形すればいいんだ? permuteとか?
    →うまくいったｗ！

    """
    d_k = Q.shape[-1]
    # 模範解答はK.size(-1)としていた へー

    # 模範解答はpermuteではなくてK.transpose(1,2)をしていた
    # 模範解答はこれをscoresとよんで,それをsoftmaxしたものをweightsとよんでいた
    S = torch.bmm(Q, torch.permute(K, (0, 2, 1))) / math.sqrt(d_k)
    # S: (batch, seq_q, seq_k)
    # softmaxはshape-preserving. dimはどうするべきだ？ queryのdimとおなじにすればいいけど、、、
    # # queryのdimだからdim=-2 か
    # S = torch.softmax(S, dim=-2)  # batch, seq=q, seq=k
    # ↑ちがいますね　vを同重みづけるかなんだからdim=-1でsoftmaxするべき
    S = torch.softmax(S, dim=-1)
    # V: (batch, seq_k, d)
    return torch.bmm(S, V)


if __name__ == "__main__":
    # 🧪 Debug
    torch.manual_seed(42)
    Q = torch.randn(2, 4, 8)
    K = torch.randn(2, 4, 8)
    V = torch.randn(2, 4, 8)

    out = scaled_dot_product_attention(Q, K, V)
    print("Output shape:", out.shape)  # should be (2, 4, 8)
    print("Has NaN?    ", torch.isnan(out).any().item())  # should be False
    print("Has Inf?    ", torch.isinf(out).any().item())  # should be False

    # Cross-attention: seq_q != seq_k
    Q2 = torch.randn(1, 3, 16)
    K2 = torch.randn(1, 5, 16)
    V2 = torch.randn(1, 5, 32)
    out2 = scaled_dot_product_attention(Q2, K2, V2)
    print("Cross-attn shape:", out2.shape)  # should be (1, 3, 32)
