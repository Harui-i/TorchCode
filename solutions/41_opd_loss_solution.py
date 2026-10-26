# 元データ: duoan/TorchCode (https://github.com/duoan/TorchCode)
# 採点: uv run python scripts/judge.py check opd_loss

#
# # Solution: OPD (On-Policy Distillation) Loss
#
# Reference solution.

import torch
import torch.nn.functional as F
from torch import Tensor

# ✅ SOLUTION

def opd_loss(student_logits: Tensor,
             teacher_logits: Tensor,
             teacher_weights: Tensor | None = None,
             mask: Tensor | None = None,
             temperature: float = 1.0) -> Tensor:
    """On-Policy Distillation (OPD) reverse-KL loss.

    student_logits: (..., V) logits from the student policy
    teacher_logits: (..., V) for one teacher, or (T, ..., V) for T teachers
    teacher_weights: optional (T,) weights, normalized internally
    mask: optional (...) token mask, where 1 = include and 0 = ignore
    temperature: softmax temperature used for distillation
    returns: scalar loss (Tensor)
    """
    if teacher_logits.dim() == student_logits.dim():
        teacher_logits = teacher_logits.unsqueeze(0)
    elif teacher_logits.dim() != student_logits.dim() + 1:
        raise ValueError('teacher_logits must have shape (..., V) or (T, ..., V)')

    teacher_logits = teacher_logits.detach()
    t = float(temperature)

    student_logp = F.log_softmax(student_logits / t, dim=-1)
    student_prob = student_logp.exp()
    teacher_logp = F.log_softmax(teacher_logits / t, dim=-1)

    per_teacher_kl = (
        student_prob.unsqueeze(0) * (student_logp.unsqueeze(0) - teacher_logp)
    ).sum(dim=-1)

    num_teachers = per_teacher_kl.shape[0]
    if teacher_weights is None:
        weights = torch.full(
            (num_teachers,),
            1.0 / num_teachers,
            dtype=per_teacher_kl.dtype,
            device=per_teacher_kl.device,
        )
    else:
        weights = teacher_weights.to(dtype=per_teacher_kl.dtype, device=per_teacher_kl.device)
        weights = weights / weights.sum()

    view_shape = (num_teachers,) + (1,) * (per_teacher_kl.dim() - 1)
    per_token_kl = (weights.view(view_shape) * per_teacher_kl).sum(dim=0)

    if mask is not None:
        mask = mask.to(dtype=per_token_kl.dtype, device=per_token_kl.device)
        loss = (per_token_kl * mask).sum() / mask.sum().clamp_min(1.0)
    else:
        loss = per_token_kl.mean()

    return loss * (t ** 2)

# Demo
student_logits = torch.tensor([[[2.0, 0.0, -1.0], [0.5, 1.0, -0.5]]])
teacher_logits = torch.tensor([[[1.0, 1.5, -0.5], [0.0, 2.0, -1.0]]])
print('Loss:', opd_loss(student_logits, teacher_logits).item())
