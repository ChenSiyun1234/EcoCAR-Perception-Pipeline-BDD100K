"""
Multi-task loss for YOLOPv2-DETRLane.

Combines:
  * YOLOP anchor-based detection loss (CIoU + obj + cls) — the exact
    detection half of `lib/core/loss.py::MultiHeadLoss`, without
    touching that file.
  * LaneSetLoss (Hungarian + curve + raster) from `lane_set_loss.py`.

The two losses are weighted by `cfg.LOSS.DET_TASK_WEIGHT` and
`cfg.LOSS.LANE_TASK_WEIGHT` respectively so the ratio can be tuned
without re-weighting individual sub-terms.

A `LaneLossScheduler` is used to interpolate geom vs raster scales
across epochs (same curriculum DETR_GeoLane uses, which literature on
gradient-conflict mitigation in MTL supports: stable supervision first,
sharp structured supervision later).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

from .general import bbox_iou
from .postprocess import build_targets
from .loss import smooth_BCE, FocalLoss
from .lane_set_loss import LaneSetLoss, LaneLossScheduler


class DETRLaneMultiHeadLoss(nn.Module):
    """Detection (YOLOP-style) + Lane (DETR-style) multi-task loss."""

    def __init__(self, cfg, device):
        super().__init__()
        self.cfg = cfg
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cfg.LOSS.CLS_POS_WEIGHT])).to(device)
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([cfg.LOSS.OBJ_POS_WEIGHT])).to(device)
        if getattr(cfg.LOSS, 'FL_GAMMA', 0.0) > 0:
            BCEcls = FocalLoss(BCEcls, cfg.LOSS.FL_GAMMA)
            BCEobj = FocalLoss(BCEobj, cfg.LOSS.FL_GAMMA)
        self.BCEcls = BCEcls
        self.BCEobj = BCEobj
        self.lambdas = cfg.LOSS.MULTI_HEAD_LAMBDA or [1.0, 1.0, 1.0]

        lane_cfg = getattr(cfg, 'LANE', None)
        lane_kw = {}
        if lane_cfg is not None:
            lane_kw = dict(
                num_lane_types=int(getattr(lane_cfg, 'NUM_TYPES', 7)),
                exist_weight=float(getattr(lane_cfg, 'EXIST_WEIGHT', 2.0)),
                pts_weight=float(getattr(lane_cfg, 'PTS_WEIGHT', 5.0)),
                type_weight=float(getattr(lane_cfg, 'TYPE_WEIGHT', 1.0)),
                tangent_weight=float(getattr(lane_cfg, 'TANGENT_WEIGHT', 1.0)),
                curvature_weight=float(getattr(lane_cfg, 'CURVATURE_WEIGHT', 0.5)),
                overlap_weight=float(getattr(lane_cfg, 'OVERLAP_WEIGHT', 2.0)),
                vis_weight=float(getattr(lane_cfg, 'VIS_WEIGHT', 0.5)),
                raster_h=int(getattr(lane_cfg, 'RASTER_H', 72)),
                raster_w=int(getattr(lane_cfg, 'RASTER_W', 128)),
                raster_thickness=float(getattr(lane_cfg, 'RASTER_THICKNESS', 0.03)),
                aux_weight=float(getattr(lane_cfg, 'AUX_WEIGHT', 0.5)),
            )
        self.lane_loss = LaneSetLoss(**lane_kw).to(device)
        self.lane_scheduler = LaneLossScheduler(
            total_epochs=int(cfg.TRAIN.END_EPOCH),
            geom_start=float(getattr(lane_cfg, 'GEOM_WARMUP_SCALE', 0.70) if lane_cfg else 0.70),
            geom_end=float(getattr(lane_cfg, 'GEOM_FINAL_SCALE', 1.00) if lane_cfg else 1.00),
            raster_start=float(getattr(lane_cfg, 'RASTER_START_SCALE', 1.00) if lane_cfg else 1.00),
            raster_end=float(getattr(lane_cfg, 'RASTER_FINAL_SCALE', 0.15) if lane_cfg else 0.15),
        )
        self.det_weight = float(getattr(cfg.LOSS, 'DET_TASK_WEIGHT', 1.0))
        self.lane_weight = float(getattr(cfg.LOSS, 'LANE_TASK_WEIGHT', 1.0))

    def set_epoch(self, epoch: int):
        g, r = self.lane_scheduler.get(epoch)
        self.lane_loss.set_runtime_scales(geom_scale=g, raster_scale=r)

    # ── Detection half (transplanted from loss.py::MultiHeadLoss) ─────
    def _det_loss(self, det_preds, det_targets, model) -> Tuple[torch.Tensor, dict]:
        cfg = self.cfg
        device = det_targets.device
        lcls = torch.zeros(1, device=device)
        lbox = torch.zeros(1, device=device)
        lobj = torch.zeros(1, device=device)

        tcls, tbox, indices, anchors = build_targets(cfg, det_preds, det_targets, model)
        cp, cn = smooth_BCE(eps=0.0)
        no = len(det_preds)
        balance = [4.0, 1.0, 0.4] if no == 3 else [4.0, 1.0, 0.4, 0.1]

        for i, pi in enumerate(det_preds):
            b, a, gj, gi = indices[i]
            tobj = torch.zeros_like(pi[..., 0], device=device)
            n = b.shape[0]
            if n:
                ps = pi[b, a, gj, gi]
                pxy = ps[:, :2].sigmoid() * 2. - 0.5
                pwh = (ps[:, 2:4].sigmoid() * 2) ** 2 * anchors[i]
                pbox = torch.cat((pxy, pwh), 1).to(device)
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=False, CIoU=True)
                lbox += (1.0 - iou).mean()
                tobj[b, a, gj, gi] = (1.0 - model.gr) + model.gr * iou.detach().clamp(0).type(tobj.dtype)
                if model.nc > 1:
                    t = torch.full_like(ps[:, 5:], cn, device=device)
                    t[range(n), tcls[i]] = cp
                    lcls += self.BCEcls(ps[:, 5:], t)
            lobj += self.BCEobj(pi[..., 4], tobj) * balance[i]

        s = 3 / no
        lcls *= cfg.LOSS.CLS_GAIN * s * self.lambdas[0]
        lobj *= cfg.LOSS.OBJ_GAIN * s * (1.4 if no == 4 else 1.0) * self.lambdas[1]
        lbox *= cfg.LOSS.BOX_GAIN * s * self.lambdas[2]
        total = lbox + lobj + lcls
        return total, {
            'det_box': float(lbox.item()),
            'det_obj': float(lobj.item()),
            'det_cls': float(lcls.item()),
        }

    # ── Combined forward ──────────────────────────────────────────────
    def forward(self, model_out: dict, target: dict, model) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            model_out: dict with keys
                'det_out': IDetect output during training -> list of 3 tensors
                'lane'   : LaneSetHead output dict
            target: dict with keys 'det', 'lane_existence',
                'lane_points', 'lane_visibility', 'lane_type', 'has_lanes'.
            model: has `.nc` and `.gr` attributes.
        Returns:
            total_loss, info_dict
        """
        det_out = model_out['det_out']
        if isinstance(det_out, tuple):  # (infer, train_out) in eval mode
            det_train = det_out[1]
        else:
            det_train = det_out

        det_total, det_info = self._det_loss(det_train, target['det'], model)
        lane_total, lane_info = self.lane_loss(
            model_out['lane'],
            target['lane_existence'],
            target['lane_points'],
            target['lane_visibility'],
            target['lane_type'],
            target['has_lanes'],
        )
        total = self.det_weight * det_total + self.lane_weight * lane_total

        info = {
            **det_info, **lane_info,
            'det_total': float(det_total.item()),
            'lane_total': float(lane_total.item()),
            'det_weight': self.det_weight,
            'lane_weight': self.lane_weight,
        }
        return total, info


def get_loss_detrlane(cfg, device):
    return DETRLaneMultiHeadLoss(cfg, device)
