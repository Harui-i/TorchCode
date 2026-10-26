# 元データ: duoan/TorchCode (https://github.com/duoan/TorchCode)
# 採点: uv run python scripts/judge.py check linear_regression

#
# # 🟡 Solution: Linear Regression
#
# Reference solution demonstrating closed-form, gradient descent, and nn.Linear approaches.

import torch
import torch.nn as nn

# ✅ SOLUTION

class LinearRegression:
    def closed_form(self, X: torch.Tensor, y: torch.Tensor):
        """Normal equation via augmented matrix."""
        N, D = X.shape
        # Augment X with ones column for bias
        X_aug = torch.cat([X, torch.ones(N, 1)], dim=1)  # (N, D+1)
        # Solve (X^T X) theta = X^T y
        theta = torch.linalg.lstsq(X_aug, y).solution      # (D+1,)
        w = theta[:D]
        b = theta[D]
        return w.detach(), b.detach()

    def gradient_descent(self, X: torch.Tensor, y: torch.Tensor,
                         lr: float = 0.01, steps: int = 1000):
        """Manual gradient computation — no autograd."""
        N, D = X.shape
        w = torch.zeros(D)
        b = torch.tensor(0.0)

        for _ in range(steps):
            pred = X @ w + b          # (N,)
            error = pred - y           # (N,)
            grad_w = (2.0 / N) * (X.T @ error)  # (D,)
            grad_b = (2.0 / N) * error.sum()     # scalar
            w = w - lr * grad_w
            b = b - lr * grad_b

        return w, b

    def nn_linear(self, X: torch.Tensor, y: torch.Tensor,
                  lr: float = 0.01, steps: int = 1000):
        """PyTorch nn.Linear with autograd training loop."""
        N, D = X.shape
        layer = nn.Linear(D, 1)
        optimizer = torch.optim.SGD(layer.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        for _ in range(steps):
            optimizer.zero_grad()
            pred = layer(X).squeeze(-1)  # (N,)
            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

        w = layer.weight.data.squeeze(0)  # (D,)
        b = layer.bias.data.squeeze(0)    # scalar ()
        return w, b

if __name__ == "__main__":
    # Verify
    torch.manual_seed(42)
    X = torch.randn(100, 3)
    true_w = torch.tensor([2.0, -1.0, 0.5])
    y = X @ true_w + 3.0

    model = LinearRegression()
    for name, method in [("Closed-form", model.closed_form),
                          ("Grad Descent", lambda X, y: model.gradient_descent(X, y, lr=0.05, steps=2000)),
                          ("nn.Linear", lambda X, y: model.nn_linear(X, y, lr=0.05, steps=2000))]:
        w, b = method(X, y)
        print(f"{name:13s}  w={w.tolist()}  b={b.item():.4f}")
    print(f"{'True':13s}  w={true_w.tolist()}  b=3.0000")
