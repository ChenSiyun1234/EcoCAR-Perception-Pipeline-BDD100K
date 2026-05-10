from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DetectionLossConfig:
    obj_loss_weight: float = 1.0
    cls_loss_weight: float = 1.0
    bbox_loss_weight: float = 5.0
    giou_loss_weight: float = 2.0
    dn_loss_weight: float = 0.0


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 3):
        super().__init__()
        layers = []
        last = in_dim
        for _ in range(max(1, num_layers - 1)):
            layers.append(nn.Linear(last, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            last = hidden_dim
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SinePositionEmbedding2D(nn.Module):
    def __init__(self, num_pos_feats: int = 128, temperature: int = 10000):
        super().__init__()
        self.num_pos_feats = int(num_pos_feats)
        self.temperature = int(temperature)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _c, h, w = x.shape
        y_embed = torch.linspace(0, 1, h, device=x.device, dtype=x.dtype).view(1, h, 1).expand(b, h, w)
        x_embed = torch.linspace(0, 1, w, device=x.device, dtype=x.dtype).view(1, 1, w).expand(b, h, w)
        dim_t = torch.arange(self.num_pos_feats, dtype=x.dtype, device=x.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='floor') / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


def read_yolo_label(path: Path) -> torch.Tensor:
    rows: List[List[float]] = []
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                rows.append([float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
            except ValueError:
                continue
    if not rows:
        return torch.zeros((0, 5), dtype=torch.float32)
    return torch.tensor(rows, dtype=torch.float32)


def cxcywh_to_xyxy(box: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = box.unbind(-1)
    x1 = cx - w * 0.5
    y1 = cy - h * 0.5
    x2 = cx + w * 0.5
    y2 = cy + h * 0.5
    return torch.stack([x1, y1, x2, y2], dim=-1)


def generalized_iou(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
    a = cxcywh_to_xyxy(box_a)
    b = cxcywh_to_xyxy(box_b)
    x1 = torch.maximum(a[..., 0], b[..., 0])
    y1 = torch.maximum(a[..., 1], b[..., 1])
    x2 = torch.minimum(a[..., 2], b[..., 2])
    y2 = torch.minimum(a[..., 3], b[..., 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_a = (a[..., 2] - a[..., 0]).clamp(min=0) * (a[..., 3] - a[..., 1]).clamp(min=0)
    area_b = (b[..., 2] - b[..., 0]).clamp(min=0) * (b[..., 3] - b[..., 1]).clamp(min=0)
    union = area_a + area_b - inter + 1e-7
    iou = inter / union
    cx1 = torch.minimum(a[..., 0], b[..., 0])
    cy1 = torch.minimum(a[..., 1], b[..., 1])
    cx2 = torch.maximum(a[..., 2], b[..., 2])
    cy2 = torch.maximum(a[..., 3], b[..., 3])
    c_area = (cx2 - cx1).clamp(min=0) * (cy2 - cy1).clamp(min=0) + 1e-7
    return iou - (c_area - union) / c_area


class SimpleVehicleDetectionHead(nn.Module):
    def __init__(self, in_channels: int = 128, hidden_dim: int = 128, num_classes: int = 1):
        super().__init__()
        self.num_classes = int(num_classes)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.obj_head = nn.Conv2d(hidden_dim, 1, 1)
        self.box_head = nn.Conv2d(hidden_dim, 4, 1)
        self.cls_head = nn.Conv2d(hidden_dim, self.num_classes, 1)

    def forward(self, feats: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = self.stem(feats[0])
        return {'obj_logits': self.obj_head(x), 'box_raw': self.box_head(x), 'cls_logits': self.cls_head(x)}


class SimpleVehicleDetectionLoss(nn.Module):
    def __init__(self, cfg: Optional[DetectionLossConfig] = None):
        super().__init__()
        self.cfg = cfg or DetectionLossConfig()

    def forward(self, pred: Dict[str, torch.Tensor], targets: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        cfg = self.cfg
        obj_logits = pred['obj_logits']
        box_raw = pred['box_raw']
        cls_logits = pred['cls_logits']
        bsz, _one, h, w = obj_logits.shape
        device = obj_logits.device
        obj_target = torch.zeros((bsz, 1, h, w), device=device)
        box_target = torch.zeros((bsz, 4, h, w), device=device)
        cls_target = torch.zeros((bsz, cls_logits.shape[1], h, w), device=device)
        positive = torch.zeros((bsz, 1, h, w), device=device)
        for b, labels in enumerate(targets):
            labels = labels.to(device)
            if labels.numel() == 0:
                continue
            valid = (labels[:, 1:5] >= 0).all(dim=1) & (labels[:, 1:5] <= 1).all(dim=1) & (labels[:, 3:5] > 0).all(dim=1)
            labels = labels[valid]
            for row in labels:
                gx = int(torch.floor(row[1].clamp(0, 0.9999) * w).item())
                gy = int(torch.floor(row[2].clamp(0, 0.9999) * h).item())
                obj_target[b, 0, gy, gx] = 1.0
                positive[b, 0, gy, gx] = 1.0
                box_target[b, :, gy, gx] = row[1:5]
                class_id = int(row[0].item())
                if 0 <= class_id < cls_logits.shape[1]:
                    cls_target[b, class_id, gy, gx] = 1.0
        obj_loss = F.binary_cross_entropy_with_logits(obj_logits, obj_target)
        box_pred = torch.sigmoid(box_raw)
        if positive.sum() > 0:
            pred_pos = box_pred.permute(0, 2, 3, 1)[positive[:, 0] > 0.5]
            tgt_pos = box_target.permute(0, 2, 3, 1)[positive[:, 0] > 0.5]
            bbox_loss = F.smooth_l1_loss(pred_pos, tgt_pos, beta=0.05)
            giou_loss = (1.0 - generalized_iou(pred_pos, tgt_pos)).mean()
            cls_pos_logits = cls_logits.permute(0, 2, 3, 1)[positive[:, 0] > 0.5]
            cls_pos_target = cls_target.permute(0, 2, 3, 1)[positive[:, 0] > 0.5]
            cls_loss = F.binary_cross_entropy_with_logits(cls_pos_logits, cls_pos_target)
        else:
            bbox_loss = obj_logits.new_zeros(())
            giou_loss = obj_logits.new_zeros(())
            cls_loss = obj_logits.new_zeros(())
        total = cfg.obj_loss_weight * obj_loss + cfg.cls_loss_weight * cls_loss + cfg.bbox_loss_weight * bbox_loss + cfg.giou_loss_weight * giou_loss
        return total, {'det/obj': obj_loss.detach(), 'det/cls': cls_loss.detach(), 'det/bbox': bbox_loss.detach(), 'det/giou': giou_loss.detach(), 'det/total': total.detach(), 'det/positives': positive.sum().detach()}


class DETRVehicleDetectionHead(nn.Module):
    def __init__(self, in_channels: int = 128, hidden_dim: int = 256, num_queries: int = 100, num_classes: int = 1, num_decoder_layers: int = 3, num_heads: int = 8, dim_feedforward: int = 1024):
        super().__init__()
        self.num_queries = int(num_queries)
        self.num_classes = int(num_classes)
        self.hidden_dim = int(hidden_dim)
        self.proj = nn.ModuleList([nn.Conv2d(in_channels, hidden_dim, 1) for _ in range(3)])
        self.pos = SinePositionEmbedding2D(hidden_dim // 2)
        self.level_embed = nn.Parameter(torch.randn(3, hidden_dim) * 0.02)
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=dim_feedforward, dropout=0.0, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.query_embed = nn.Parameter(torch.randn(self.num_queries, hidden_dim) * 0.02)
        self.obj_head = nn.Linear(hidden_dim, 1)
        self.cls_head = nn.Linear(hidden_dim, self.num_classes)
        self.box_head = MLP(hidden_dim, hidden_dim, 4, num_layers=3)

    def forward(self, feats: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        memories = []
        for i, feat in enumerate(feats[:3]):
            j = min(i, len(self.proj) - 1)
            p = self.proj[j](feat)
            pos = self.pos(p)
            if pos.shape[1] != p.shape[1]:
                pos = pos[:, :p.shape[1]]
            tokens = (p + pos + self.level_embed[j].view(1, -1, 1, 1)).flatten(2).transpose(1, 2)
            memories.append(tokens)
        memory = torch.cat(memories, dim=1)
        q = self.query_embed.unsqueeze(0).expand(memory.shape[0], -1, -1)
        z = self.decoder(q, memory)
        return {'obj_logits': self.obj_head(z).squeeze(-1), 'cls_logits': self.cls_head(z), 'box_pred': torch.sigmoid(self.box_head(z))}


def _hungarian(cost: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if cost.numel() == 0:
        device = cost.device
        return torch.zeros((0,), dtype=torch.long, device=device), torch.zeros((0,), dtype=torch.long, device=device)
    try:
        from scipy.optimize import linear_sum_assignment
        row, col = linear_sum_assignment(cost.detach().cpu().numpy())
        return torch.as_tensor(row, dtype=torch.long, device=cost.device), torch.as_tensor(col, dtype=torch.long, device=cost.device)
    except Exception:
        q, n = cost.shape
        used_q = torch.zeros((q,), dtype=torch.bool, device=cost.device)
        used_g = torch.zeros((n,), dtype=torch.bool, device=cost.device)
        pairs_q = []
        pairs_g = []
        for _ in range(min(q, n)):
            masked = cost.clone()
            masked[used_q, :] = float('inf')
            masked[:, used_g] = float('inf')
            flat = torch.argmin(masked)
            val = masked.flatten()[flat]
            if not torch.isfinite(val):
                break
            qi = flat // n
            gi = flat % n
            pairs_q.append(qi)
            pairs_g.append(gi)
            used_q[qi] = True
            used_g[gi] = True
        if not pairs_q:
            device = cost.device
            return torch.zeros((0,), dtype=torch.long, device=device), torch.zeros((0,), dtype=torch.long, device=device)
        return torch.stack(pairs_q).long(), torch.stack(pairs_g).long()


class DETRVehicleDetectionLoss(nn.Module):
    def __init__(self, cfg: Optional[DetectionLossConfig] = None, no_object_weight: float = 0.25, cost_cls: float = 1.0):
        super().__init__()
        self.cfg = cfg or DetectionLossConfig()
        self.no_object_weight = float(no_object_weight)
        self.cost_cls = float(cost_cls)

    def forward(self, pred: Dict[str, torch.Tensor], targets: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        cfg = self.cfg
        obj_logits = pred['obj_logits']
        cls_logits = pred['cls_logits']
        box_pred = pred['box_pred']
        bsz, _num_queries = obj_logits.shape
        device = obj_logits.device
        obj_target = torch.zeros_like(obj_logits)
        cls_target = torch.zeros_like(cls_logits)
        matched_pred_boxes = []
        matched_gt_boxes = []
        matched_cls_logits = []
        matched_cls_targets = []
        total_gt = 0
        for b in range(bsz):
            labels = targets[b].to(device)
            if labels.numel() == 0:
                continue
            valid = (labels[:, 1:5] >= 0).all(dim=1) & (labels[:, 1:5] <= 1).all(dim=1) & (labels[:, 3:5] > 0).all(dim=1)
            labels = labels[valid]
            if labels.numel() == 0:
                continue
            gt_cls = labels[:, 0].long().clamp(0, cls_logits.shape[-1] - 1)
            gt_box = labels[:, 1:5]
            total_gt += int(gt_box.shape[0])
            cls_prob = cls_logits[b].sigmoid()[:, gt_cls]
            cls_cost = -cls_prob
            l1 = torch.cdist(box_pred[b], gt_box, p=1)
            giou = generalized_iou(box_pred[b].unsqueeze(1), gt_box.unsqueeze(0))
            cost = self.cost_cls * cls_cost + cfg.bbox_loss_weight * l1 + cfg.giou_loss_weight * (1.0 - giou)
            pq, pg = _hungarian(cost)
            if pq.numel() == 0:
                continue
            obj_target[b, pq] = 1.0
            cls_target[b, pq, gt_cls[pg]] = 1.0
            matched_pred_boxes.append(box_pred[b, pq])
            matched_gt_boxes.append(gt_box[pg])
            matched_cls_logits.append(cls_logits[b, pq])
            matched_cls_targets.append(cls_target[b, pq])
        weight = torch.full_like(obj_target, self.no_object_weight)
        weight[obj_target > 0.5] = 1.0
        obj_loss = F.binary_cross_entropy_with_logits(obj_logits, obj_target, weight=weight)
        if matched_pred_boxes:
            pred_box = torch.cat(matched_pred_boxes, dim=0)
            gt_box = torch.cat(matched_gt_boxes, dim=0)
            bbox_loss = F.l1_loss(pred_box, gt_box)
            giou_loss = (1.0 - generalized_iou(pred_box, gt_box)).mean()
            cls_loss = F.binary_cross_entropy_with_logits(torch.cat(matched_cls_logits, dim=0), torch.cat(matched_cls_targets, dim=0))
        else:
            bbox_loss = obj_logits.new_zeros(())
            giou_loss = obj_logits.new_zeros(())
            cls_loss = obj_logits.new_zeros(())
        dn_loss = obj_logits.new_zeros(())
        total = cfg.obj_loss_weight * obj_loss + cfg.cls_loss_weight * cls_loss + cfg.bbox_loss_weight * bbox_loss + cfg.giou_loss_weight * giou_loss + cfg.dn_loss_weight * dn_loss
        return total, {'det/obj': obj_loss.detach(), 'det/cls': cls_loss.detach(), 'det/bbox_l1': bbox_loss.detach(), 'det/giou': giou_loss.detach(), 'det/dn': dn_loss.detach(), 'det/total': total.detach(), 'det/positives': torch.tensor(float(total_gt), device=device)}
