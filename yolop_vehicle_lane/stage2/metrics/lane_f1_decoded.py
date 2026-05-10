"""Lane-line F1 evaluator with proper top-K decode + lane-NMS.

Replaces the eval-time hardcoded `score_threshold = 0.5` bug in
`CLRKDStyleLaneMetric` (see metrics/original_metric_adapters.py:138). That
metric reported F1 = 0 across every Exp2 run because our models' max sigmoid
score has been ~0.15 -- well below 0.5 -- so its `pred_exist` was all-False
by construction.

This metric instead:
- Decodes top-K lanes per image by descending score (no threshold floor by
  default; ranking is what matters).
- Applies lane-NMS by LineIoU (CLRKDNet-standard).
- Greedily assigns surviving predictions to GT lanes by descending LineIoU.
  A prediction is TP if its LineIoU with an unassigned GT exceeds the
  threshold.
- Reads the *original* target['existence'] / target['points'], not the
  matched_target produced by FusionLaneLoss.match_targets, so it measures
  prediction-vs-GT in the same way as CLRKDNet's published F1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F

from stage2.fusion.lane_decode import decode_top_k_lanes, lane_nms
from stage2.fusion.losses import _compute_lineiou_target, _line_iou_1d


@torch.no_grad()
def _mask_consistency_score(
    mask_logit: torch.Tensor,
    coord_pred: torch.Tensor,
) -> torch.Tensor:
    """Mean sigmoid(mask) sampled along each prior's predicted curve.

    Args
    ----
    mask_logit: (B, 1, H, W) auxiliary lane segmentation logits.
    coord_pred: (B, P, N, 2) per-prior curve coordinates in normalized [0, 1]
        (x, y) order, matching the head's output convention.

    Returns: (B, P) score in [0, 1].

    The intuition: the auxiliary segmentation mask is trained on a deterministic
    per-pixel target (no per-prior matching instability) so its sigmoid is a
    much cleaner geometric verifier than the per-prior cls head, which has been
    collapsing to near-uniform output across NB39/NB40/NB41/NB42. For each
    prior we sample the mask sigmoid at the prior's 72 (x, y) curve points;
    priors that pass through high-mask regions get high score, priors that
    drift through background get low score. Inference-time only -- the metric
    consumes the existing `mask_logit` field already emitted by CLRKDLaneHead
    when mask_aux=True.
    """
    if mask_logit is None or mask_logit.dim() != 4:
        raise RuntimeError(
            f"mask_logit expected (B, 1, H, W); got {None if mask_logit is None else tuple(mask_logit.shape)}"
        )
    if coord_pred.dim() != 4 or coord_pred.shape[-1] != 2:
        raise RuntimeError(
            f"coord_pred expected (B, P, N, 2); got {tuple(coord_pred.shape)}"
        )
    bsz, num_priors, num_points, _ = coord_pred.shape
    # F.grid_sample wants normalized coords in [-1, 1] with order (x, y).
    grid = (coord_pred * 2.0 - 1.0).clamp(-1.0, 1.0)            # (B, P, N, 2)
    sampled = F.grid_sample(
        mask_logit.float(), grid.float(),
        mode='bilinear', padding_mode='border', align_corners=False,
    )                                                            # (B, 1, P, N)
    score = torch.sigmoid(sampled).squeeze(1).mean(dim=-1)       # (B, P)
    return score.clamp(0.0, 1.0)


@dataclass
class LaneF1DecodedMetric:
    top_k: int = 4
    line_iou_threshold: float = 0.5
    lane_nms_iou: float = 0.5
    radius: float = 0.015
    score_threshold: float = 0.0
    # Ranking score sources:
    # 'cls' (default): sigmoid(cls_logit). Has been collapsing to near-uniform
    #   on the 192-anchor head across Exp2II/KK/LL/MM (pos-neg gap < 0.01).
    # 'oracle_iou': ground-truth LineIoU (Exp2O diagnostic; F1 ceiling the
    #   geometry can support independent of cls).
    # 'cls_x_iou' (Exp2MM): sigmoid(cls_logit) * sigmoid(iou_logit) from the
    #   dual-score head.
    # 'mask_consistency' (Exp2NN): mean sigmoid(mask) sampled along each
    #   prior's predicted curve. Bypasses cls entirely and uses the auxiliary
    #   segmentation mask as a per-prior geometric verifier. The mask has no
    #   per-prior matching instability, so its sigmoid is a much cleaner
    #   ranking signal than the degenerate cls head.
    # 'cls_x_mask' (Exp2NN): sigmoid(cls_logit) * mask_consistency. Hedge:
    #   keep any cls signal that does separate priors and combine with the
    #   robust mask verifier.
    score_source: str = 'cls'
    # Suffix appended to all output keys, so multiple metric instances can run
    # side by side in val without colliding (cls vs oracle vs cls_x_iou).
    key_suffix: str = ''

    @torch.no_grad()
    def __call__(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        if self.score_source == 'oracle_iou':
            scores_override = _compute_lineiou_target(
                pred['coord_pred'], target['points'], target['visibility'],
                radius=self.radius, target_pow=1.0,
            )
        elif self.score_source == 'cls_x_iou':
            iou_logit = pred.get('iou_logits')
            if iou_logit is None:
                # Fall back to plain cls scoring if the head did not emit
                # iou_logits (back-compat with non-dual_score heads).
                scores_override = None
            else:
                scores_override = (
                    torch.sigmoid(pred['cls_logits']) * torch.sigmoid(iou_logit)
                ).clamp(0.0, 1.0)
        elif self.score_source == 'mask_consistency':
            mask_logit = pred.get('mask_logit')
            if mask_logit is None:
                scores_override = None
            else:
                scores_override = _mask_consistency_score(mask_logit, pred['coord_pred'])
        elif self.score_source == 'cls_x_mask':
            mask_logit = pred.get('mask_logit')
            if mask_logit is None:
                scores_override = None
            else:
                mask_score = _mask_consistency_score(mask_logit, pred['coord_pred'])
                scores_override = (torch.sigmoid(pred['cls_logits']) * mask_score).clamp(0.0, 1.0)
        else:
            scores_override = None
        decoded = decode_top_k_lanes(
            pred, top_k=self.top_k, score_threshold=self.score_threshold,
            scores_override=scores_override,
        )
        decoded = lane_nms(decoded, line_iou_threshold=self.lane_nms_iou, radius=self.radius)

        existence = target['existence']               # (B, L)
        points_gt = target['points']                  # (B, L, N, 2)
        vis = target['visibility']                    # (B, L, N)
        bsz = existence.shape[0]
        device = existence.device

        tp_total = torch.zeros((), device=device)
        fp_total = torch.zeros((), device=device)
        fn_total = torch.zeros((), device=device)
        pred_count_total = torch.zeros((), device=device)
        score_sum = torch.zeros((), device=device)
        score_n = torch.zeros((), device=device)

        for b in range(bsz):
            gt_mask = existence[b] > 0.5
            num_gt = int(gt_mask.sum().item())
            preds = decoded[b]
            kp = preds['coord_xy'].shape[0]
            pred_count_total = pred_count_total + float(kp)
            if preds['score'].numel() > 0:
                score_sum = score_sum + preds['score'].sum()
                score_n = score_n + float(kp)
            if num_gt == 0:
                fp_total = fp_total + float(kp)
                continue
            if kp == 0:
                fn_total = fn_total + float(num_gt)
                continue
            gt_x = points_gt[b][gt_mask][..., 0]      # (Lg, N)
            gt_v = vis[b][gt_mask]                    # (Lg, N)
            pred_x = preds['coord_xy'][..., 0]        # (Kp, N)
            full_mask = torch.ones_like(pred_x)
            # Match each pred row to each GT row: shapes (Kp, Lg).
            ious = _line_iou_1d(
                pred_x[:, None, :], gt_x[None, :, :],
                gt_v[None, :, :].expand(kp, num_gt, -1) * full_mask[:, None, :],
                self.radius,
            )                                         # (Kp, Lg)
            # Greedy assignment: pick highest IoU pair, mark TP, exclude rows
            # and columns, repeat. Caps either kp or num_gt iterations.
            tp_local = 0
            assigned_pred = torch.zeros(kp, dtype=torch.bool, device=device)
            assigned_gt = torch.zeros(num_gt, dtype=torch.bool, device=device)
            for _ in range(min(kp, num_gt)):
                # Mask out already-used rows/cols by setting their IoU to -1.
                ious_view = ious.clone()
                ious_view[assigned_pred] = -1.0
                ious_view[:, assigned_gt] = -1.0
                best = ious_view.max()
                if best < float(self.line_iou_threshold):
                    break
                idx = ious_view.argmax()
                pi = int((idx // num_gt).item())
                gi = int((idx % num_gt).item())
                tp_local += 1
                assigned_pred[pi] = True
                assigned_gt[gi] = True
            tp_total = tp_total + float(tp_local)
            fp_total = fp_total + float(kp - tp_local)
            fn_total = fn_total + float(num_gt - tp_local)

        precision = tp_total / (tp_total + fp_total + 1e-6)
        recall = tp_total / (tp_total + fn_total + 1e-6)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-6)
        avg_score = score_sum / score_n.clamp(min=1.0)
        sfx = self.key_suffix
        return {
            f'lane/decoded{sfx}_f1': f1,
            f'lane/decoded{sfx}_precision': precision,
            f'lane/decoded{sfx}_recall': recall,
            f'lane/decoded{sfx}_tp': tp_total,
            f'lane/decoded{sfx}_fp': fp_total,
            f'lane/decoded{sfx}_fn': fn_total,
            f'lane/decoded{sfx}_pred_count': pred_count_total,
            f'lane/decoded{sfx}_avg_score': avg_score,
            f'lane/decoded{sfx}_top_k_used': torch.tensor(float(self.top_k), device=device),
        }
