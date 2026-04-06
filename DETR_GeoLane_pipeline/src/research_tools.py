"""Research helpers for diagnosing multi-task conflict before full training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch


@dataclass
class GradientConflictSummary:
    det_norm: float
    lane_norm: float
    cosine: float
    shared_params: int

    def to_dict(self) -> Dict[str, float]:
        return {
            "det_grad_norm": self.det_norm,
            "lane_grad_norm": self.lane_norm,
            "grad_cosine": self.cosine,
            "shared_params": float(self.shared_params),
        }


def _flatten_grads(named_params, prefix_filters: List[str]) -> torch.Tensor:
    chunks = []
    count = 0
    for name, p in named_params:
        if p.grad is None:
            continue
        if not any(key in name for key in prefix_filters):
            continue
        chunks.append(p.grad.detach().reshape(-1))
        count += p.numel()
    if not chunks:
        return torch.zeros(1)
    return torch.cat(chunks)


def gradient_conflict_summary(model, criterion, outputs, batch, shared_prefixes=None) -> Dict[str, float]:
    if shared_prefixes is None:
        shared_prefixes = ["backbone", "det_adapters", "lane_adapters", "det_proj", "lane_proj"]

    model.zero_grad(set_to_none=True)
    det_gt = criterion._prepare_det_gt(outputs, batch)
    det_loss, _ = criterion.det_loss(outputs, det_gt["classes"], det_gt["boxes"])
    det_loss.backward(retain_graph=True)
    det_vec = _flatten_grads(model.named_parameters(), shared_prefixes)
    det_norm = float(det_vec.norm().item())

    model.zero_grad(set_to_none=True)
    lane_loss, _ = criterion.lane_loss(
        outputs,
        batch["lane_existence"],
        batch["lane_points"],
        batch["lane_visibility"],
        batch["lane_type"],
        batch["has_lanes"],
    )
    lane_loss.backward(retain_graph=True)
    lane_vec = _flatten_grads(model.named_parameters(), shared_prefixes)
    lane_norm = float(lane_vec.norm().item())

    denom = (det_vec.norm() * lane_vec.norm()).clamp(min=1e-8)
    cosine = float(torch.dot(det_vec, lane_vec).div(denom).item())
    model.zero_grad(set_to_none=True)
    return GradientConflictSummary(det_norm, lane_norm, cosine, int(det_vec.numel())).to_dict()


def feature_branch_gap(model, images: torch.Tensor) -> Dict[str, float]:
    model.eval()
    with torch.no_grad():
        fpn = model.backbone(images)
        if hasattr(model, '_task_adapt'):
            det_fpn, lane_fpn = model._task_adapt(fpn)
        else:
            det_fpn, lane_fpn = fpn, fpn
    out = {}
    for idx, (d, l) in enumerate(zip(det_fpn, lane_fpn)):
        out[f"p{idx+3}_cosine"] = float(torch.nn.functional.cosine_similarity(d.flatten(1), l.flatten(1), dim=1).mean().item())
        out[f"p{idx+3}_l2_gap"] = float((d - l).pow(2).mean().sqrt().item())
    return out
