# 元データ: duoan/TorchCode (https://github.com/duoan/TorchCode)
# 採点: uv run python scripts/judge.py check opd_loss

#
# # 🔴 Hard: OPD Loss
#
# Implement the **On-Policy Distillation (OPD)** loss used to distill one or more teacher policies into a student policy on trajectories sampled from the student.
#
# For each token position, OPD minimizes a weighted reverse KL from the student distribution to each teacher distribution:
#
# $$\mathcal{L}_{\text{OPD}} = \sum_j w_j\,D_{\text{KL}}\big(\pi_\theta(\cdot \mid x)\;||\;\pi_{E_j}(\cdot \mid x)\big)$$
#
# where
#
# $$D_{\text{KL}}(p||q) = \sum_v p(v)\,[\log p(v) - \log q(v)].$$
#
# ### Signature
# ```python
# from torch import Tensor
#
# def opd_loss(student_logits: Tensor,
#              teacher_logits: Tensor,
#              teacher_weights: Tensor | None = None,
#              mask: Tensor | None = None,
#              temperature: float = 1.0) -> Tensor:
#     """OPD reverse-KL distillation loss.
#
#     student_logits: (..., V) logits from the student policy
#     teacher_logits: (..., V) for one teacher, or (T, ..., V) for T teachers
#     teacher_weights: optional (T,) weights, normalized internally
#     mask: optional (...) token mask, where 1 = include and 0 = ignore
#     temperature: softmax temperature used for distillation
#     returns: scalar loss (Tensor)
#     """
# ```

import torch
import torch.nn.functional as F
from torch import Tensor

# ✏️ YOUR IMPLEMENTATION HERE

def opd_loss(student_logits: Tensor,
             teacher_logits: Tensor,
             teacher_weights: Tensor | None = None,
             mask: Tensor | None = None,
             temperature: float = 1.0) -> Tensor:
    pass  # reverse KL: sum_v p_student * (log p_student - log p_teacher)

if __name__ == "__main__":
    # 🧪 Debug
    student_logits = torch.tensor([[[2.0, 0.0, -1.0], [0.5, 1.0, -0.5]]])
    teacher_logits = torch.tensor([[[1.0, 1.5, -0.5], [0.0, 2.0, -1.0]]])
    print('Loss:', opd_loss(student_logits, teacher_logits).item())
