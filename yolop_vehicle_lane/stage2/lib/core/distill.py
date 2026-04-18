"""
Detection-branch knowledge distillation from a frozen Stage1 teacher.

The Stage2 student may add classes (motorcycle + bicycle) on top of
Stage1's single merged 'vehicle' class. We distill the **shared**
signal — objectness + vehicle-class logits + box regression — and
leave the new student-only classes alone.

Usage:
    teacher = build_teacher(stage1_ckpt)   # set eval(), no grad
    distiller = DetectionDistillLoss(teacher_nc=1, student_nc=3, ...)
    total_distill = distiller(student_det_out, teacher_det_out)

The teacher forward is expected to be the same YOLOP-style list of
3 grid tensors with shape [B, na, H, W, 5 + teacher_nc].
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectionDistillLoss(nn.Module):
    """Shared-signal KD for YOLOP-style detection heads.

    Loss components (each gated by its own weight):
      * `box`: smooth-L1 between student/teacher (tx,ty,tw,th).
      * `obj`: BCE-with-logits between student/teacher objectness.
      * `vehicle_cls`: BCE-with-logits on the shared vehicle class
        (src row = teacher row 0, dst row = student row 0).

    No distillation on student's extra classes (motorcycle / bicycle)
    because the teacher does not have them.
    """

    def __init__(self,
                 teacher_nc: int = 1,
                 student_nc: int = 3,
                 box_weight: float = 1.0,
                 obj_weight: float = 1.0,
                 vehicle_cls_weight: float = 1.0,
                 temperature: float = 1.0):
        super().__init__()
        self.teacher_nc = int(teacher_nc)
        self.student_nc = int(student_nc)
        self.box_weight = float(box_weight)
        self.obj_weight = float(obj_weight)
        self.vehicle_cls_weight = float(vehicle_cls_weight)
        self.T = float(temperature)

    def forward(self,
                student_det: List[torch.Tensor],
                teacher_det: List[torch.Tensor]) -> torch.Tensor:
        """Both args are YOLOP's training-mode list-of-3-scales output."""
        box_losses = []
        obj_losses = []
        cls_losses = []
        for s, t in zip(student_det, teacher_det):
            # s: [B, na, H, W, 5 + student_nc]
            # t: [B, na, H, W, 5 + teacher_nc]
            s_box = s[..., 0:4]
            t_box = t[..., 0:4]
            box_losses.append(F.smooth_l1_loss(s_box, t_box))

            s_obj = s[..., 4]
            t_obj = t[..., 4].sigmoid()
            obj_losses.append(F.binary_cross_entropy_with_logits(s_obj, t_obj))

            # vehicle class row (teacher row 0, student row 0).
            if self.teacher_nc >= 1 and self.student_nc >= 1:
                s_cls_v = s[..., 5]
                t_cls_v = t[..., 5].sigmoid()
                cls_losses.append(F.binary_cross_entropy_with_logits(s_cls_v, t_cls_v))

        total = (
            self.box_weight         * torch.stack(box_losses).mean() +
            self.obj_weight         * torch.stack(obj_losses).mean() +
            (self.vehicle_cls_weight * torch.stack(cls_losses).mean()
             if cls_losses else torch.zeros((), device=student_det[0].device))
        )
        return total


@torch.no_grad()
def freeze_as_teacher(model: nn.Module) -> nn.Module:
    """Put `model` in eval mode, disable grads on every parameter, and
    move it to the same device as the caller expects.
    """
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
