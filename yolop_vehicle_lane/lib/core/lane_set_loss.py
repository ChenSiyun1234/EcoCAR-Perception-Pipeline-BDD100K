"""
DETR-style lane set-prediction loss (Hungarian matching + curve geometry).

[SOURCE] This module is vendored from
    DETR_GeoLane_pipeline/src/losses.py
    (classes LaneLoss, LaneLossScheduler + batched curve helpers).
The detection half of that file is NOT vendored here; YOLOPv2-DETRLane
keeps the YOLOP-style anchor-based detection loss for the vehicle head.

Operates on point-based lane labels instead of binary masks. A prediction
is a set of `Q` lane queries, each producing `P` 2-D points plus
existence / visibility / type logits. The loss is Hungarian-matched
against GT polylines using a combined geometric + soft-raster cost.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


# ──────────────────────────────────────────────────────────────────────
# Geometry helpers
# ──────────────────────────────────────────────────────────────────────

def _grid_xy(height: int, width: int, device, dtype) -> torch.Tensor:
    ys = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype)
    xs = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([xx, yy], dim=-1).reshape(-1, 2)


def _batched_prepare_curves(points: torch.Tensor,
                            visibility: torch.Tensor | None = None):
    m, p, _ = points.shape
    device, dtype = points.device, points.dtype
    if visibility is None:
        return points, torch.ones(m, p, device=device, dtype=dtype)
    raw_mask = (visibility > 0.5)
    valid_counts = raw_mask.sum(dim=-1)
    fallback = valid_counts < 2
    point_mask = raw_mask.to(dtype)
    if fallback.any():
        point_mask = point_mask.clone()
        point_mask[fallback] = 1.0
    return points, point_mask


def _batched_resample_polyline(points: torch.Tensor,
                               visibility: torch.Tensor | None,
                               num: int) -> torch.Tensor:
    curves, point_mask = _batched_prepare_curves(points, visibility)
    m, p, _ = curves.shape
    device, dtype = curves.device, curves.dtype
    seg_valid = point_mask[:, :-1] * point_mask[:, 1:]
    seg_vec = curves[:, 1:] - curves[:, :-1]
    seg_len = torch.norm(seg_vec, dim=-1) * seg_valid
    cum = torch.cat(
        [torch.zeros(m, 1, device=device, dtype=dtype), seg_len.cumsum(dim=-1)], dim=-1)
    total = cum[:, -1]
    base_t = (torch.zeros(1, device=device, dtype=dtype)
              if num == 1 else torch.linspace(0.0, 1.0, num, device=device, dtype=dtype))
    t = base_t.unsqueeze(0) * total.unsqueeze(1)
    idx = torch.searchsorted(cum.contiguous(), t.contiguous(), right=True) - 1
    idx = idx.clamp(min=0, max=p - 2)
    gather_idx = idx.unsqueeze(-1).expand(-1, -1, 2)
    left = torch.gather(curves[:, :-1], 1, gather_idx)
    right = torch.gather(curves[:, 1:], 1, gather_idx)
    left_t = torch.gather(cum[:, :-1], 1, idx)
    right_t = torch.gather(cum[:, 1:], 1, idx)
    alpha = ((t - left_t) / (right_t - left_t).clamp(min=1e-8)).unsqueeze(-1)
    out = left + alpha * (right - left)
    degenerate = total < 1e-8
    if degenerate.any():
        first_point = curves[:, :1, :].expand(-1, num, -1)
        out = torch.where(degenerate.view(-1, 1, 1), first_point, out)
    return out


def _pairwise_point_to_polyline_distance(points_a: torch.Tensor,
                                         points_b: torch.Tensor) -> torch.Tensor:
    qa, r, _ = points_a.shape
    qb = points_b.shape[0]
    if r == 0:
        return torch.zeros(qa, qb, device=points_a.device, dtype=points_a.dtype)
    if r == 1:
        diff = points_a[:, None, :, :] - points_b[None, :, :1, :]
        return torch.norm(diff, dim=-1).mean(dim=-1)
    seg_a = points_b[:, :-1, :]
    seg_b = points_b[:, 1:, :]
    ab = seg_b - seg_a
    denom = (ab * ab).sum(dim=-1).clamp(min=1e-8)
    pts = points_a[:, None, :, None, :]
    seg_a_e = seg_a[None, :, None, :, :]
    ab_e = ab[None, :, None, :, :]
    ap = pts - seg_a_e
    t = (ap * ab_e).sum(dim=-1) / denom[None, :, None, :]
    t = t.clamp(0.0, 1.0)
    proj = seg_a_e + t.unsqueeze(-1) * ab_e
    dist = torch.norm(pts - proj, dim=-1)
    min_dist = dist.min(dim=-1).values
    return min_dist.mean(dim=-1)


def _batched_polyline_tangents(points: torch.Tensor) -> torch.Tensor:
    if points.shape[1] < 2:
        return torch.zeros_like(points)
    fwd = torch.zeros_like(points)
    fwd[:, 1:-1] = points[:, 2:] - points[:, :-2]
    fwd[:, 0] = points[:, 1] - points[:, 0]
    fwd[:, -1] = points[:, -1] - points[:, -2]
    norm = torch.norm(fwd, dim=-1, keepdim=True).clamp(min=1e-8)
    return fwd / norm


def _pairwise_curve_to_curve_distance(pred_points: torch.Tensor,
                                      gt_points: torch.Tensor,
                                      gt_visibility: torch.Tensor | None = None,
                                      resample_n: int = 96) -> Dict[str, torch.Tensor]:
    pred_rs = _batched_resample_polyline(pred_points, None, resample_n)
    gt_rs = _batched_resample_polyline(gt_points, gt_visibility, resample_n)
    d_pg = _pairwise_point_to_polyline_distance(pred_rs, gt_rs)
    d_gp = _pairwise_point_to_polyline_distance(gt_rs, pred_rs).transpose(0, 1)
    sym_dist = 0.5 * (d_pg + d_gp)
    pred_tan = _batched_polyline_tangents(pred_rs)
    gt_tan = _batched_polyline_tangents(gt_rs)
    tan_align = 1.0 - (pred_tan[:, None] * gt_tan[None, :]).sum(dim=-1).abs().mean(dim=-1)
    pred_second = pred_rs[:, 2:] - 2 * pred_rs[:, 1:-1] + pred_rs[:, :-2]
    gt_second = gt_rs[:, 2:] - 2 * gt_rs[:, 1:-1] + gt_rs[:, :-2]
    curvature_gap = F.smooth_l1_loss(
        pred_second[:, None].expand(-1, gt_second.shape[0], -1, -1),
        gt_second[None].expand(pred_second.shape[0], -1, -1, -1),
        reduction='none',
    ).mean(dim=(-1, -2))
    return {'sym_dist': sym_dist, 'tan': tan_align, 'curvature': curvature_gap}


def _batched_soft_polyline_mask(points: torch.Tensor,
                                visibility: torch.Tensor | None = None,
                                height: int = 72,
                                width: int = 128,
                                thickness: float = 0.03,
                                sharpness: float = 80.0,
                                grid: torch.Tensor | None = None) -> torch.Tensor:
    curves, point_mask = _batched_prepare_curves(points, visibility)
    m, p, _ = curves.shape
    device, dtype = curves.device, curves.dtype
    if grid is None:
        grid = _grid_xy(height, width, device, dtype)
    else:
        grid = grid.to(device=device, dtype=dtype)
    if p == 1:
        dist = torch.norm(grid[None, :, :] - curves[:, :1, :], dim=-1)
        return torch.sigmoid((thickness - dist) * sharpness).view(m, height, width)
    seg_a = curves[:, :-1, :]
    seg_b = curves[:, 1:, :]
    ab = seg_b - seg_a
    denom = (ab * ab).sum(dim=-1).clamp(min=1e-8)
    seg_valid = (point_mask[:, :-1] * point_mask[:, 1:]) > 0.5
    pts = grid[None, :, None, :]
    seg_a_e = seg_a[:, None, :, :]
    ab_e = ab[:, None, :, :]
    ap = pts - seg_a_e
    t = ((ap * ab_e).sum(dim=-1) / denom[:, None, :]).clamp(0.0, 1.0)
    proj = seg_a_e + t.unsqueeze(-1) * ab_e
    dist = torch.norm(pts - proj, dim=-1)
    inf = torch.full_like(dist, float('inf'))
    dist = torch.where(seg_valid[:, None, :], dist, inf)
    min_dist = dist.min(dim=-1).values
    no_valid = ~seg_valid.any(dim=-1)
    if no_valid.any():
        point_dist = torch.norm(grid[None, :, None, :] - curves[:, None, :, :], dim=-1)
        point_dist = torch.where(
            point_mask[:, None, :] > 0.5, point_dist, inf[:, :, :point_mask.shape[1]])
        fallback = point_dist.min(dim=-1).values
        min_dist = torch.where(no_valid[:, None], fallback, min_dist)
    return torch.sigmoid((thickness - min_dist) * sharpness).view(m, height, width)


# ──────────────────────────────────────────────────────────────────────
# Curriculum schedule
# ──────────────────────────────────────────────────────────────────────

class LaneLossScheduler:
    """Cosine schedule for geometric vs raster loss weights.
    Early: heavy raster (soft IoU) → stable optimization.
    Late: heavy geometry (curve distance) → sharp polylines.
    """

    def __init__(self,
                 total_epochs: int = 50,
                 geom_start: float = 0.70,
                 geom_end: float = 1.00,
                 raster_start: float = 1.00,
                 raster_end: float = 0.15,
                 start_ratio: float = 0.25,
                 end_ratio: float = 0.75):
        self.total_epochs = max(int(total_epochs), 1)
        self.geom_start, self.geom_end = float(geom_start), float(geom_end)
        self.raster_start, self.raster_end = float(raster_start), float(raster_end)
        self.start_ratio = float(start_ratio)
        self.end_ratio = float(end_ratio)

    @staticmethod
    def _cosine(a: float, b: float, t: float) -> float:
        t = 0.5 * (1.0 - math.cos(math.pi * max(0.0, min(1.0, t))))
        return a + (b - a) * t

    def get(self, epoch: int) -> Tuple[float, float]:
        if self.total_epochs <= 1:
            return self.geom_end, self.raster_end
        x = epoch / float(self.total_epochs - 1)
        if x <= self.start_ratio:
            return self.geom_start, self.raster_start
        if x >= self.end_ratio:
            return self.geom_end, self.raster_end
        t = (x - self.start_ratio) / max(self.end_ratio - self.start_ratio, 1e-8)
        return (self._cosine(self.geom_start, self.geom_end, t),
                self._cosine(self.raster_start, self.raster_end, t))


# ──────────────────────────────────────────────────────────────────────
# LaneLoss — Hungarian-matched set prediction loss for polylines
# ──────────────────────────────────────────────────────────────────────

class LaneSetLoss(nn.Module):
    """Hungarian-matched loss between predicted lane queries and GT polylines.

    Output dict produced by LaneSetHead must contain:
        'exist_logits'  : [B, Q, 1]
        'pred_points'   : [B, Q, P, 2]
        'vis_logits'    : [B, Q, P]     (optional)
        'type_logits'   : [B, Q, C]
        'aux_outputs'   : list of dicts with the same keys (optional)

    GT tensors:
        existence  : [B, M]      1 if GT lane slot is used
        points     : [B, M, P, 2]
        visibility : [B, M, P]
        lane_type  : [B, M]
        has_lanes  : [B]         1 if the frame has any lanes at all
    """

    def __init__(self,
                 num_lane_types: int = 7,
                 exist_weight: float = 2.0,
                 pts_weight: float = 5.0,
                 type_weight: float = 1.0,
                 tangent_weight: float = 1.0,
                 curvature_weight: float = 0.5,
                 overlap_weight: float = 2.0,
                 vis_weight: float = 0.5,
                 match_resample_n: int = 64,
                 loss_resample_n: int = 96,
                 raster_h: int = 72,
                 raster_w: int = 128,
                 raster_thickness: float = 0.03,
                 aux_weight: float = 0.5):
        super().__init__()
        self.num_lane_types = num_lane_types
        self.exist_weight = exist_weight
        self.pts_weight = pts_weight
        self.type_weight = type_weight
        self.tangent_weight = tangent_weight
        self.curvature_weight = curvature_weight
        self.overlap_weight = overlap_weight
        self.vis_weight = vis_weight
        self.match_resample_n = match_resample_n
        self.loss_resample_n = loss_resample_n
        self.raster_h = raster_h
        self.raster_w = raster_w
        self.raster_thickness = raster_thickness
        self.aux_weight = aux_weight
        self._grid_cache: Dict[tuple, torch.Tensor] = {}
        self.geom_runtime_scale = 1.0
        self.raster_runtime_scale = 1.0

    def set_runtime_scales(self, geom_scale: float = 1.0, raster_scale: float = 1.0):
        self.geom_runtime_scale = float(geom_scale)
        self.raster_runtime_scale = float(raster_scale)

    def _get_raster_grid(self, device, dtype) -> torch.Tensor:
        key = (device.type, getattr(device, 'index', None), str(dtype), self.raster_h, self.raster_w)
        grid = self._grid_cache.get(key)
        if grid is None:
            grid = _grid_xy(self.raster_h, self.raster_w, device, dtype)
            self._grid_cache[key] = grid
        return grid

    @torch.no_grad()
    def _hungarian_match(self, pred_pts, pred_exist, gt_pts, gt_exist, gt_vis=None):
        b = pred_pts.shape[0]
        matches = []
        grid = self._get_raster_grid(pred_pts.device, pred_pts.dtype)
        for bi in range(b):
            gt_mask = gt_exist[bi] > 0.5
            n_gt = int(gt_mask.sum().item())
            if n_gt == 0:
                matches.append(([], []))
                continue
            gt_pts_i = gt_pts[bi][gt_mask]
            gt_vis_i = gt_vis[bi][gt_mask] if gt_vis is not None else None
            gt_indices = torch.where(gt_mask)[0]

            geom = _pairwise_curve_to_curve_distance(
                pred_pts[bi], gt_pts_i, gt_vis_i, self.match_resample_n)
            pred_masks = _batched_soft_polyline_mask(
                pred_pts[bi], None, self.raster_h, self.raster_w,
                self.raster_thickness, grid=grid).flatten(1)
            gt_masks = _batched_soft_polyline_mask(
                gt_pts_i, gt_vis_i, self.raster_h, self.raster_w,
                self.raster_thickness, grid=grid).flatten(1)
            inter = torch.minimum(pred_masks[:, None], gt_masks[None]).sum(-1)
            union = torch.maximum(pred_masks[:, None], gt_masks[None]).sum(-1).clamp(min=1e-6)
            overlap_cost = 1.0 - inter / union

            cost = geom['sym_dist'] + 0.35 * geom['tan'] + 0.20 * geom['curvature'] + 0.75 * overlap_cost
            cost = cost - 0.25 * pred_exist[bi, :, 0].sigmoid().unsqueeze(1)

            pi, gi = linear_sum_assignment(cost.detach().cpu().numpy())
            matches.append((pi.tolist(), gt_indices[gi].tolist()))
        return matches

    def _loss_single(self, outputs, gt_exist, gt_points, gt_vis, gt_type, has_lanes):
        pred_exist = outputs['exist_logits']
        pred_pts = outputs['pred_points']
        pred_type = outputs['type_logits']
        pred_vis = outputs.get('vis_logits')
        b, q = pred_exist.shape[:2]
        device = pred_exist.device

        lane_mask = has_lanes > 0.5
        if not lane_mask.any():
            zero = torch.zeros((), device=device)
            return zero, {k: 0.0 for k in
                          ['lane_exist', 'lane_curve', 'lane_tangent', 'lane_curvature',
                           'lane_overlap', 'lane_type', 'lane_vis']}

        matches = self._hungarian_match(pred_pts, pred_exist, gt_points, gt_exist, gt_vis)
        exist_target = torch.zeros(b, q, device=device)
        total_curve = torch.zeros((), device=device)
        total_tangent = torch.zeros((), device=device)
        total_curvature = torch.zeros((), device=device)
        total_overlap = torch.zeros((), device=device)
        total_type = torch.zeros((), device=device)
        total_vis = torch.zeros((), device=device)
        n_matched = 0
        grid = self._get_raster_grid(pred_pts.device, pred_pts.dtype)

        for bi, (pi, gi) in enumerate(matches):
            if not lane_mask[bi] or len(pi) == 0:
                continue
            pi_t = torch.as_tensor(pi, dtype=torch.long, device=device)
            gi_t = torch.as_tensor(gi, dtype=torch.long, device=device)
            exist_target[bi, pi_t] = 1.0

            pred_pts_i = pred_pts[bi, pi_t]
            gt_pts_i = gt_points[bi, gi_t]
            gt_vis_i = gt_vis[bi, gi_t] if gt_vis is not None else None

            geom = _pairwise_curve_to_curve_distance(
                pred_pts_i, gt_pts_i, gt_vis_i, self.loss_resample_n)
            diag = torch.arange(len(pi), device=device)
            total_curve += geom['sym_dist'][diag, diag].sum()
            total_tangent += geom['tan'][diag, diag].sum()
            total_curvature += geom['curvature'][diag, diag].sum()

            pred_masks = _batched_soft_polyline_mask(
                pred_pts_i, None, self.raster_h, self.raster_w,
                self.raster_thickness, grid=grid).flatten(1)
            gt_masks = _batched_soft_polyline_mask(
                gt_pts_i, gt_vis_i, self.raster_h, self.raster_w,
                self.raster_thickness, grid=grid).flatten(1)
            inter = (pred_masks * gt_masks).sum(-1)
            union = (pred_masks + gt_masks - pred_masks * gt_masks).sum(-1).clamp(min=1e-6)
            dice = 1.0 - (2.0 * inter + 1e-6) / (pred_masks.sum(-1) + gt_masks.sum(-1) + 1e-6)
            iou = 1.0 - inter / union
            total_overlap += (0.5 * (iou + dice)).sum()

            total_type += F.cross_entropy(
                pred_type[bi, pi_t], gt_type[bi, gi_t].to(device), reduction='sum')
            if pred_vis is not None and gt_vis is not None:
                total_vis += F.binary_cross_entropy_with_logits(
                    pred_vis[bi, pi_t], gt_vis[bi, gi_t].to(device), reduction='sum')
            n_matched += len(pi)

        exist_loss = F.binary_cross_entropy_with_logits(
            pred_exist[lane_mask, :, 0], exist_target[lane_mask],
            pos_weight=torch.tensor(3.0, device=device))
        n = max(n_matched, 1)
        curve_loss = total_curve / n
        tangent_loss = total_tangent / n
        curvature_loss = total_curvature / n
        overlap_loss = total_overlap / n
        type_loss = total_type / n
        vis_loss = total_vis / n

        total = (
            self.exist_weight * exist_loss +
            self.geom_runtime_scale * self.pts_weight * curve_loss +
            self.geom_runtime_scale * self.tangent_weight * tangent_loss +
            self.geom_runtime_scale * self.curvature_weight * curvature_loss +
            self.raster_runtime_scale * self.overlap_weight * overlap_loss +
            self.type_weight * type_loss +
            self.vis_weight * vis_loss
        )
        info = {
            'lane_exist': float(exist_loss.item()),
            'lane_curve': float(curve_loss.item()),
            'lane_tangent': float(tangent_loss.item()),
            'lane_curvature': float(curvature_loss.item()),
            'lane_overlap': float(overlap_loss.item()),
            'lane_type': float(type_loss.item()),
            'lane_vis': float(vis_loss.item()),
        }
        return total, info

    def forward(self, outputs, gt_exist, gt_points, gt_vis, gt_type, has_lanes):
        total, info = self._loss_single(outputs, gt_exist, gt_points, gt_vis, gt_type, has_lanes)
        aux_outputs = outputs.get('aux_outputs', [])
        if aux_outputs:
            aux_total = torch.zeros((), device=gt_points.device)
            for aux in aux_outputs:
                aux_loss, _ = self._loss_single(aux, gt_exist, gt_points, gt_vis, gt_type, has_lanes)
                aux_total += aux_loss
            total = total + self.aux_weight * aux_total / max(len(aux_outputs), 1)
        return total, info
