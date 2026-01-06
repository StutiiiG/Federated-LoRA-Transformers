from __future__ import annotations
import torch
import torch.nn.functional as F

def kl_alignment_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    KL( softmax(teacher/T) || softmax(student/T) ) * T^2 (standard distillation form)
    """
    s = F.log_softmax(student_logits / temperature, dim=-1)
    t = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(s, t, reduction="batchmean") * (temperature ** 2)

