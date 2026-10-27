# 元データ: duoan/TorchCode (https://github.com/duoan/TorchCode)
# 採点: uv run python scripts/judge.py check batchnorm

#
# # 🟡 Medium: Implement BatchNorm
#
# Implement **Batch Normalization** with both **training** and **inference** behavior.
#
# In training mode, use **batch statistics** and update running estimates:
#
# $$\text{BN}(x) = \gamma \cdot \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} + \beta$$
#
# where $\mu_B$ and $\sigma_B^2$ are the mean and variance computed **across the batch** (dim=0).
#
# In inference mode, use the provided **running mean/var** instead of current batch stats.
#
# ### Signature
# ```python
# def my_batch_norm(
#     x: torch.Tensor,
#     gamma: torch.Tensor,
#     beta: torch.Tensor,
#     running_mean: torch.Tensor,
#     running_var: torch.Tensor,
#     eps: float = 1e-5,
#     momentum: float = 0.1,
#     training: bool = True,
# ) -> torch.Tensor:
#     # x: (N, D) — normalize each feature across all samples in the batch
#     # running_mean, running_var: updated in-place during training; used as-is during inference
# ```
#
# ### Rules
# - Do **NOT** use `F.batch_norm`, `nn.BatchNorm1d`, etc.
# - Compute batch mean and variance over `dim=0` with `unbiased=False`
# - Update running stats like PyTorch: `running = (1 - momentum) * running + momentum * batch_stat`
# - Use `running_mean` / `running_var` for inference when `training=False`
# - Must support autograd w.r.t. `x`, `gamma`, `beta` (running statistics should be treated as buffers, not parameters requiring gradients)

import torch

# ✏️ YOUR IMPLEMENTATION HERE


def my_batch_norm(
    x: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    eps=1e-5,
    momentum=0.1,
    training=True,
):
    # x: [b, d]
    #
    assert running_mean.requires_grad is False
    assert running_mean.requires_grad is False

    if not training:
        # inference
        x = gamma * (x - running_mean) / (running_var + eps).sqrt() + beta
        return x
    else:
        # ここでxがgradをrequireするときbatch_...も計算グラフにのっちゃう
        batch_sigma_squared = x.var(dim=0, unbiased=False)  # [d]
        batch_mean = x.mean(dim=0)  # [d]
        # training
        x = gamma * (x - batch_mean) / (batch_sigma_squared + eps).sqrt() + beta

        # detachについて: https://docs.pytorch.org/docs/2.14/generated/torch.Tensor.detach.html#torch.Tensor.detach
        # 計算ぐらふから取り除かれて、勾配を要求しなくなる

        # update batch statistics
        # 問題文にこうかいてあるので. へー！　batch normalizationって訓練時の統計量更新をするんだ…
        # Q: でもなんで？running_mean, running_varを訓練時はつかってなくね?
        # A: 推論時に使うために、訓練時に蓄積しておくらしい！　へ〜〜〜〜
        # running_mean, running_var: updated in-place during training; used as-is during inference
        # ここなんか running_mean = ...みたいに書いてたら同じrunning_meanじゃなくてなんか別のオブジェクトが作られていた？のか破壊的変更を出せなかった　なんでだろ
        # shapeがいっしょじゃなかったからかな？

        # """
        # ok
        running_mean *= 1.0 - momentum
        running_mean += momentum * batch_mean.detach()
        running_var *= 1.0 - momentum
        running_var += momentum * batch_sigma_squared.detach()
        # """

        """
        # ng
        running_mean = (1.0 - momentum) * running_mean + momentum * batch_mean
        running_var = (1.0 - momentum) * running_var + momentum * batch_sigma_squared
        """
        # ちなみにin-place更新をするために、模範解答は
        # running_mean.mul_(1-momentum).add_(momentum * batch_mean.detach())としていた。

        assert running_mean.requires_grad is False
        assert running_var.requires_grad is False

        return x


if __name__ == "__main__":
    # 🧪 Debug
    x = torch.randn(8, 4)
    gamma = torch.ones(4)
    beta = torch.zeros(4)

    # Running stats typically live on the same device and shape as features
    running_mean = torch.zeros(4)
    running_var = torch.ones(4)

    # Training mode: uses batch stats and updates running_mean / running_var
    out_train = my_batch_norm(x, gamma, beta, running_mean, running_var, training=True)
    print("[Train] Output shape:", out_train.shape)
    print("[Train] Column means:", out_train.mean(dim=0))  # should be ~0
    print("[Train] Column stds: ", out_train.std(dim=0))  # should be ~1
    print("Updated running_mean:", running_mean)
    print("Updated running_var:", running_var)

    # Inference mode: uses running_mean / running_var only
    out_eval = my_batch_norm(x, gamma, beta, running_mean, running_var, training=False)
    print("[Eval] Output shape:", out_eval.shape)
