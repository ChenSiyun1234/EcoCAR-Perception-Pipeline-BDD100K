from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch


ROOT = Path(__file__).resolve().parents[2]
RMT_METRIC_SOURCES = [
    ROOT / 'stage2/vendor/RMT-PPAD/ultralytics/models/mtdetr/val.py',
    ROOT / 'stage2/vendor/RMT-PPAD/ultralytics/utils/metrics.py',
    ROOT / 'stage2/vendor/RMT-PPAD/ultralytics/models/rtdetr/val.py',
]
CLRKD_METRIC_SOURCES = [
    ROOT / 'stage2/vendor/CLRKDNet/clrkd/datasets/bdd100k_curve.py',
    ROOT / 'stage2/vendor/CLRKDNet/clrkd/models/utils/dynamic_assign.py',
]


def metric_source_report() -> Dict[str, object]:
    return {
        'rmt_metric_sources': [str(p) for p in RMT_METRIC_SOURCES if p.exists()],
        'clrkd_metric_sources': [str(p) for p in CLRKD_METRIC_SOURCES if p.exists()],
        'rmt_sources_missing': [str(p) for p in RMT_METRIC_SOURCES if not p.exists()],
        'clrkd_sources_missing': [str(p) for p in CLRKD_METRIC_SOURCES if not p.exists()],
    }


def cxcywh_to_xyxy(box: torch.Tensor) -> torch.Tensor:
    x, y, w, h = box.unbind(-1)
    x1 = x - 0.5 * w
    y1 = y - 0.5 * h
    x2 = x + 0.5 * w
    y2 = y + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1).clamp(0.0, 1.0)


def box_iou_xyxy(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.numel() == 0 or b.numel() == 0:
        return a.new_zeros((a.shape[0], b.shape[0]))
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = ((a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0))[:, None]
    area_b = ((b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0))[None, :]
    return inter / (area_a + area_b - inter + 1e-6)


@dataclass
class VehicleDetectionMetric:
    conf_threshold: float = 0.05
    iou_threshold: float = 0.5
    max_preds: int = 300

    @torch.no_grad()
    def __call__(self, pred: Dict[str, torch.Tensor], targets: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        obj = pred['obj_logits'].sigmoid()
        cls = pred['cls_logits'].sigmoid().max(dim=-1).values
        score = obj * cls
        boxes = cxcywh_to_xyxy(pred['box_pred'])
        device = boxes.device
        all_rows: List[Tuple[float, int, int]] = []
        total_gt = 0
        matched_flags: List[torch.Tensor] = []
        for b, labels in enumerate(targets):
            labels = labels.to(device)
            if labels.numel() > 0:
                valid = (labels[:, 1:5] >= 0).all(dim=1) & (labels[:, 1:5] <= 1).all(dim=1) & (labels[:, 3:5] > 0).all(dim=1)
                gt = cxcywh_to_xyxy(labels[valid, 1:5])
            else:
                gt = boxes.new_zeros((0, 4))
            total_gt += int(gt.shape[0])
            matched_flags.append(torch.zeros((gt.shape[0],), dtype=torch.bool, device=device))
            keep = torch.nonzero(score[b] >= self.conf_threshold, as_tuple=False).flatten()
            if keep.numel() == 0:
                continue
            keep_score = score[b, keep]
            order = torch.argsort(keep_score, descending=True)[: self.max_preds]
            keep = keep[order]
            keep_score = keep_score[order]
            ious = box_iou_xyxy(boxes[b, keep], gt)
            for local_i, q in enumerate(keep.tolist()):
                best_iou = 0.0
                best_gt = -1
                if gt.numel() > 0:
                    vals = ious[local_i]
                    best_iou_t, best_gt_t = vals.max(dim=0)
                    best_iou = float(best_iou_t.item())
                    best_gt = int(best_gt_t.item())
                all_rows.append((float(keep_score[local_i].item()), b, best_gt if best_iou >= self.iou_threshold else -1))
        if total_gt == 0:
            z = boxes.new_zeros(())
            return {'det/metric_map50': z, 'det/metric_precision50': z, 'det/metric_recall50': z, 'det/metric_num_gt': z, 'det/metric_num_pred': z}
        all_rows.sort(key=lambda x: x[0], reverse=True)
        tp_vals = []
        fp_vals = []
        for _score, b, gt_idx in all_rows:
            if gt_idx >= 0 and not bool(matched_flags[b][gt_idx].item()):
                matched_flags[b][gt_idx] = True
                tp_vals.append(1.0)
                fp_vals.append(0.0)
            else:
                tp_vals.append(0.0)
                fp_vals.append(1.0)
        if not tp_vals:
            z = boxes.new_zeros(())
            return {
                'det/metric_map50': z,
                'det/metric_precision50': z,
                'det/metric_recall50': z,
                'det/metric_num_gt': torch.tensor(float(total_gt), device=device),
                'det/metric_num_pred': z,
            }
        tp = torch.tensor(tp_vals, device=device).cumsum(0)
        fp = torch.tensor(fp_vals, device=device).cumsum(0)
        recall = tp / max(float(total_gt), 1.0)
        precision = tp / (tp + fp + 1e-6)
        mrec = torch.cat([torch.zeros(1, device=device), recall, torch.ones(1, device=device)])
        mpre = torch.cat([torch.ones(1, device=device), precision, torch.zeros(1, device=device)])
        for i in range(mpre.numel() - 2, -1, -1):
            mpre[i] = torch.maximum(mpre[i], mpre[i + 1])
        idx = torch.nonzero(mrec[1:] != mrec[:-1], as_tuple=False).flatten()
        ap = ((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum()
        return {
            'det/metric_map50': ap.detach(),
            'det/metric_precision50': precision[-1].detach(),
            'det/metric_recall50': recall[-1].detach(),
            'det/metric_num_gt': torch.tensor(float(total_gt), device=device),
            'det/metric_num_pred': torch.tensor(float(len(tp_vals)), device=device),
        }


@dataclass
class CLRKDStyleLaneMetric:
    score_threshold: float = 0.5
    line_iou_threshold: float = 0.5
    radius: float = 0.015

    @torch.no_grad()
    def __call__(self, pred: Dict[str, torch.Tensor], matched_target: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        from stage2.fusion.losses import _line_iou_1d

        score = pred['cls_logits'].sigmoid()
        pred_exist = score >= self.score_threshold
        gt_exist = matched_target['existence'] > 0.5
        vis = matched_target['visibility']
        line_iou = _line_iou_1d(pred['coord_pred'][..., 0], matched_target['points'][..., 0], vis, radius=self.radius)
        tp = ((line_iou >= self.line_iou_threshold) & pred_exist & gt_exist).float().sum()
        fp = (pred_exist & ~((line_iou >= self.line_iou_threshold) & gt_exist)).float().sum()
        fn = (gt_exist & ~((line_iou >= self.line_iou_threshold) & pred_exist)).float().sum()
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * precision * recall / (precision + recall + 1e-6)
        return {
            'lane/clrkd_style_f1': f1.detach(),
            'lane/clrkd_style_precision': precision.detach(),
            'lane/clrkd_style_recall': recall.detach(),
            'lane/clrkd_style_tp': tp.detach(),
            'lane/clrkd_style_fp': fp.detach(),
            'lane/clrkd_style_fn': fn.detach(),
        }
