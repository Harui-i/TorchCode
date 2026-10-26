# 元データ: duoan/TorchCode (https://github.com/duoan/TorchCode)
# 採点: uv run python scripts/judge.py check linear_regression

#
# # 🟡 Medium: Linear Regression
#
# Implement **linear regression** using three different approaches — all in pure PyTorch.
#
# Given data `X` of shape `(N, D)` and targets `y` of shape `(N,)`, find weight `w` of shape `(D,)` and bias `b` (scalar) such that:
#
# $$\hat{y} = Xw + b$$
#
# ### Signature
# ```python
# class LinearRegression:
#     def closed_form(self, X: Tensor, y: Tensor) -> tuple[Tensor, Tensor]: ...
#     def gradient_descent(self, X: Tensor, y: Tensor, lr=0.01, steps=1000) -> tuple[Tensor, Tensor]: ...
#     def nn_linear(self, X: Tensor, y: Tensor, lr=0.01, steps=1000) -> tuple[Tensor, Tensor]: ...
# ```
#
# All methods return `(w, b)` where `w` has shape `(D,)` and `b` has shape `()`.
#
# ### Method 1 — Closed-Form (Normal Equation)
# Augment X with a ones column, then solve:
#
# $$\theta = (X_{aug}^T X_{aug})^{-1} X_{aug}^T y$$
#
# Or use `torch.linalg.lstsq` / `torch.linalg.solve`.
#
# ### Method 2 — Gradient Descent from Scratch
# Initialize `w` and `b` to zeros. Repeat for `steps` iterations:
# ```
# pred = X @ w + b
# error = pred - y
# grad_w = (2/N) * X^T @ error
# grad_b = (2/N) * error.sum()
# w -= lr * grad_w
# b -= lr * grad_b
# ```
#
# ### Method 3 — PyTorch nn.Linear
# Create `nn.Linear(D, 1)`, use `nn.MSELoss` and an optimizer (e.g., `torch.optim.SGD`).
# After training, extract `w` and `b` from the layer.
#
# ### Rules
# - All inputs and outputs must be **PyTorch tensors**
# - Do **NOT** use numpy or sklearn
# - `closed_form` must not use iterative optimization
# - `gradient_descent` must manually compute gradients (no `autograd`)
# - `nn_linear` should use `torch.nn.Linear` and `loss.backward()`

import torch
import torch.nn as nn

# ✏️ YOUR IMPLEMENTATION HERE

class LinearRegression:
    def closed_form(self, X: torch.Tensor, y: torch.Tensor):
        """Normal equation: w = (X^T X)^{-1} X^T y"""
        pass  # Return (w, b)

    def gradient_descent(self, X: torch.Tensor, y: torch.Tensor,
                         lr: float = 0.01, steps: int = 1000):
        """Manual gradient descent loop"""
        pass  # Return (w, b)

    def nn_linear(self, X: torch.Tensor, y: torch.Tensor,
                  lr: float = 0.01, steps: int = 1000):
        """Train nn.Linear with autograd"""
        pass  # Return (w, b)

if __name__ == "__main__":
    # 🧪 Debug
    torch.manual_seed(42)
    X = torch.randn(100, 3)
    true_w = torch.tensor([2.0, -1.0, 0.5])
    y = X @ true_w + 3.0

    model = LinearRegression()

    w_cf, b_cf = model.closed_form(X, y)
    print(f"Closed-form:  w={w_cf}, b={b_cf.item():.4f}")

    w_gd, b_gd = model.gradient_descent(X, y, lr=0.05, steps=2000)
    print(f"Grad descent: w={w_gd}, b={b_gd.item():.4f}")

    w_nn, b_nn = model.nn_linear(X, y, lr=0.05, steps=2000)
    print(f"nn.Linear:    w={w_nn}, b={b_nn.item():.4f}")

    print(f"\nTrue:         w={true_w}, b=3.0")
