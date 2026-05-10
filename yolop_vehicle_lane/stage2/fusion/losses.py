"""Stage 2 fusion loss module.

Lane-curve representation used here is the (existence, points, visibility,
lane_type) form produced by `stage2.fusion.lane_targets.frame_to_lane_targets`.
The CLRKDNet (max_lanes, 78) line-prior representation is handled by the
vendored CLRKDNet trainer; this module is the in-house lane head's loss.

Composition (Part E):

    L_lane = w_cls   * L_cls
           + w_reg   * L_point_reg
           + w_xytl  * L_start_theta_length
           + w_iou   * L_line_iou
           + w_mask  * L_mask_aux
           + w_smooth * L_smooth
           + w_distill * L_distill   (optional)

    L_total = uncertainty_weight(L_det, L_lane)   (optional)
            -- or --
    L_total = L_det + lambda_lane * L_lane        (default baseline)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class FusionLossConfig:
    # Top-level multi-task balance.
    lambda_lane: float = 1.0          # used when uncertainty=False
    use_uncertainty: bool = False     # learnable log-variance weights

    # Lane sub-loss weights.
    w_cls: float = 2.0
    w_reg: float = 0.5
    w_xytl: float = 0.2
    w_iou: float = 2.0
    w_mask: float = 1.0
    w_smooth: float = 0.05
    w_distill: float = 0.0

    use_lane_matching: bool = False
    match_cost_cls: float = 1.0
    match_cost_point: float = 5.0
    match_cost_iou: float = 2.0
    match_cost_xytl: float = 0.2
    lane_assigner: str = 'hungarian'  # 'hungarian' baseline; 'dynamic_k' is a CLRKD-style optional assigner.
    dynamic_k_topk: int = 8

    # Focal loss params for lane existence/classification.
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    cls_balance_mode: str = 'balanced'  # 'balanced' protects positives after matching; 'mean' keeps raw focal mean.

    # Asymmetric focal (ASL) controls. When cls_loss_type='asl', the binary
    # focal modulator uses gamma_pos for positive samples and gamma_neg for
    # negatives (Ben-Baruch et al. 2020). Gamma_pos=0 removes the flat-gradient
    # zone that traps the model at p=0.5 in the standard focal regime; raising
    # gamma_neg pushes harder on negatives. This is the recommended fix when
    # standard focal stalls at the uniform-predictor equilibrium (cls loss
    # ~0.08 with no score separation), as observed in Exp2G short10.
    cls_loss_type: str = 'focal'   # 'focal' | 'asl'
    asl_gamma_pos: float = 0.0
    asl_gamma_neg: float = 4.0
    asl_clip: float = 0.05         # probability margin shift on negatives

    # Lane existence cls target formulation.
    # 'matched_existence': binary {0, 1} from dynamic-k matching outcome (the
    # original CLRKDNet-style task). Observed across Exp2G/H/I/J to be unstable
    # because the same prior is positive in some batches and negative in others
    # (matching depends on which competing priors win the IoU contest), driving
    # the cls head to a degenerate "predict 0.5 everywhere" equilibrium.
    # 'lineiou_regression': continuous target = max LineIoU between this prior's
    # current predicted curve (detached) and any valid GT lane in this image.
    # Loss is BCE or QFL on the continuous target; see cls_loss_type. Removes
    # the matching-instability confound and aligns the cls head's output with
    # the natural ranking signal at inference time.
    cls_target_type: str = 'matched_existence'

    # When cls_target_type='lineiou_regression', cls_loss_type extends to:
    #   'bce' (default for regression; Exp2K, hit recall collapse because
    #          ~95% of priors have target ~ 0 so bulk gradient pushes all
    #          logits to 0).
    #   'qfl' (Exp2L, RTMDet/GFL-style Quality Focal Loss). Weights BCE by
    #          |target - pred|^qfl_gamma so easy negatives get near-zero
    #          gradient and the few high-target priors dominate.
    qfl_gamma: float = 2.0
    # When > 0 and != 1, raise the LineIoU target to this power before BCE/QFL.
    # 0.5 = sqrt rescaling: compresses high IoU, expands low IoU so even
    # near-prior misses carry a meaningful supervision signal. Used by Exp2M.
    lineiou_target_pow: float = 1.0
    # VarifocalLoss (Zhang 2021, used by RTMDet/VarifocalNet). Positives
    # weighted by target IoU (no `(1-alpha)` discount), negatives weighted
    # by `alpha * pred^gamma` (only confident-wrong negatives contribute).
    # Designed for moderate class imbalance (1:30 to 1:100) where standard
    # focal alpha=0.25 sets the symmetric equilibrium that has been
    # collapsing our cls. Used by Exp2QQ.
    vfl_alpha: float = 0.75
    vfl_gamma: float = 2.0

    # OHEM / hard-negative mining for lane existence classification.
    # When cls_ohem_topk_per_pos > 0, the negative cls loss is computed only
    # on the top-K hardest unmatched priors per image, where K = max(min_topk,
    # cls_ohem_topk_per_pos * num_pos). This addresses Exp2H's observation
    # that 1500 negatives' gradients dilute 40 positives' representational
    # pull through the shared per_lane feature, causing cls separation to
    # collapse over training. Setting <= 0 disables OHEM (back-compat).
    cls_ohem_topk_per_pos: int = 0
    cls_ohem_min_topk: int = 32

    # LineIoU sampling band (in normalized image-x units).
    line_iou_radius: float = 0.015

    # Smoothness penalty (2nd derivative of x along the curve).
    smooth_eps: float = 1e-6

    # Evaluation/classification threshold for lane-existence diagnostics.
    existence_threshold: float = 0.5

    # CLRKDLaneHead auxiliary supervision on intermediate refinement stages.
    # When the head emits aux_stage_outputs, each intermediate stage gets the
    # full lane loss weighted by 1 / (num_stages - i), matching the CLRNet
    # auxiliary-decoder convention. Set 0.0 to disable.
    aux_stage_loss_weight: float = 1.0

    # Exp2T: HybridPriorQueryHead emits stage1_cls_logits / stage1_coord_pred
    # / stage1_lane_param / stage1_lane_offsets corresponding to the 192
    # prior-based curves before the K=12 query refiner. When > 0, apply the
    # full lane loss on those stage-1 outputs as well, scaled by this weight.
    # This is the fix for Exp2Q's diagnosed bottleneck (stage 1 had no direct
    # geometry supervision so its curves stayed at matched_iou=0.14 instead
    # of recovering Exp2N's 0.42). Default 0 = back-compat with Exp2Q.
    stage1_aux_loss_weight: float = 0.0

    # Exp2MM: dual-scoring auxiliary head supervision. When CLRKDLaneHead is
    # built with dual_score=True it emits a parallel `iou_logits` (B, P) tensor
    # alongside `cls_logits`. When w_iou_aux > 0, this loss term applies BCE
    # (or QFL when iou_aux_loss_type='qfl') between sigmoid(iou_logits) and the
    # continuous LineIoU target produced by `_compute_lineiou_target`. The
    # binary `cls_logits` keeps its matched_existence supervision via the
    # existing focal/ASL path. At inference, the decoder ranks priors by
    # sigmoid(cls_logits) * sigmoid(iou_logits): cls answers "is this prior a
    # match?" and iou answers "if so, how good is the geometry?", which
    # decouples ranking from the matching-instability that has been
    # collapsing the binary cls on the 192-anchor head (Exp2II/KK).
    w_iou_aux: float = 0.0
    iou_aux_loss_type: str = 'bce'   # 'bce' | 'qfl'
    iou_aux_qfl_gamma: float = 2.0
    iou_aux_target_pow: float = 1.0


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------
def _binary_focal_loss(
    logit: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = 'mean',
) -> torch.Tensor:
    """Binary focal loss on logits."""
    bce = F.binary_cross_entropy_with_logits(logit, target.float(), reduction='none')
    p = torch.sigmoid(logit)
    p_t = p * target + (1 - p) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    loss = alpha_t * (1 - p_t).clamp(min=0).pow(gamma) * bce
    if reduction == 'mean':
        return loss.mean()
    if reduction == 'sum':
        return loss.sum()
    return loss


def _binary_asl_loss(
    logit: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma_pos: float = 0.0,
    gamma_neg: float = 4.0,
    clip: float = 0.05,
    reduction: str = 'none',
) -> torch.Tensor:
    """Asymmetric focal loss (Ben-Baruch 2020) on binary logits.

    gamma_pos = 0 removes the flat-gradient region for positives so the model
    keeps moving even when p_pos sits at 0.5. gamma_neg > gamma_pos amplifies
    suppression of confident negatives. clip > 0 shifts negative probabilities
    by a margin (probability shift, not logit shift) before applying the focal
    modulator, ignoring "easy" negatives below the clip.
    """
    target = target.float()
    p = torch.sigmoid(logit)
    if clip is not None and clip > 0:
        p_neg = (p - clip).clamp(min=0.0)
    else:
        p_neg = p
    pos_term = target * torch.log(p.clamp(min=1e-8))
    neg_term = (1.0 - target) * torch.log((1.0 - p_neg).clamp(min=1e-8))
    pos_focal = (1.0 - p).clamp(min=0.0).pow(gamma_pos) if gamma_pos > 0 else torch.ones_like(p)
    neg_focal = p_neg.clamp(min=0.0).pow(gamma_neg)
    loss = -(alpha * pos_focal * pos_term + (1.0 - alpha) * neg_focal * neg_term)
    if reduction == 'mean':
        return loss.mean()
    if reduction == 'sum':
        return loss.sum()
    return loss


def _binary_cls_raw(
    logit: torch.Tensor,
    target: torch.Tensor,
    cfg: 'FusionLossConfig',
) -> torch.Tensor:
    """Dispatch between standard focal, ASL, and VFL based on cfg.cls_loss_type."""
    loss_type = str(getattr(cfg, 'cls_loss_type', 'focal')).lower()
    if loss_type == 'asl':
        return _binary_asl_loss(
            logit, target,
            alpha=float(cfg.focal_alpha),
            gamma_pos=float(getattr(cfg, 'asl_gamma_pos', 0.0)),
            gamma_neg=float(getattr(cfg, 'asl_gamma_neg', 4.0)),
            clip=float(getattr(cfg, 'asl_clip', 0.05)),
            reduction='none',
        )
    if loss_type in {'vfl', 'varifocal'}:
        # VFL on binary {0, 1} matched_existence target. Same asymmetric
        # weighting as on the continuous IoU target -- positives carry full
        # weight (target=1), negatives only contribute when sigmoid is high.
        return _varifocal_loss(
            logit, target,
            alpha=float(getattr(cfg, 'vfl_alpha', 0.75)),
            gamma=float(getattr(cfg, 'vfl_gamma', 2.0)),
            reduction='none',
        )
    return _binary_focal_loss(
        logit, target,
        alpha=float(cfg.focal_alpha),
        gamma=float(cfg.focal_gamma),
        reduction='none',
    )


def _line_iou_1d(pred_x: torch.Tensor, gt_x: torch.Tensor,
                 mask: torch.Tensor, radius: float) -> torch.Tensor:
    """LineIoU between two per-row x sequences using a band of `radius`.

    pred_x / gt_x : (..., N) normalized x in [0, 1]
    mask          : (..., N) 1 for valid rows, 0 otherwise
    Returns: (...,) IoU in [0, 1]
    """
    pa_lo = pred_x - radius
    pa_hi = pred_x + radius
    pb_lo = gt_x - radius
    pb_hi = gt_x + radius
    inter = (torch.minimum(pa_hi, pb_hi) - torch.maximum(pa_lo, pb_lo)).clamp(min=0)
    union = (torch.maximum(pa_hi, pb_hi) - torch.minimum(pa_lo, pb_lo)).clamp(min=1e-6)
    iou = (inter * mask).sum(dim=-1) / ((union * mask).sum(dim=-1) + 1e-6)
    return iou


def _quality_focal_loss(
    logit: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    reduction: str = 'none',
) -> torch.Tensor:
    """RTMDet/GFL-style Quality Focal Loss on a continuous IoU-like target.

    Weight = |target - sigmoid(logit)|^gamma applied to BCE.
    - Easy correct cases (target ~ pred) get near-zero weight.
    - Hard mismatches (high target / low pred, or vice versa) dominate.

    This is the published fix for Exp2K's failure mode where ~95 % of priors
    have target ~ 0 so plain BCE collapses every logit to 0.
    """
    p = torch.sigmoid(logit)
    bce = F.binary_cross_entropy_with_logits(logit, target, reduction='none')
    weight = (target - p).abs().clamp(min=0.0).pow(gamma)
    loss = weight * bce
    if reduction == 'mean':
        return loss.mean()
    if reduction == 'sum':
        return loss.sum()
    return loss


def _varifocal_loss(
    logit: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
    reduction: str = 'none',
) -> torch.Tensor:
    """Varifocal Loss (Zhang et al. CVPR 2021), used by RTMDet/VarifocalNet.

    Weight is asymmetric in target:
      - For positives (target > 0): weight = target (the IoU itself, so
        good-IoU priors carry more loss; below-threshold ones get tiny weight).
      - For negatives (target == 0): weight = alpha * sigmoid(logit)^gamma
        (only confident-wrong negatives count; uniform-sigmoid negatives get
        tiny weight).

    Critically, there is NO `(1-alpha)` discount on positives. With our 5:187
    pos:neg ratio per image, plain ASL with alpha=0.25 set positive gradient
    weight to 0.25 / total -> degenerate equilibrium at sigmoid ~ 0.5
    everywhere across NB39/40/41/42/43/45. VFL flips that: positives get full
    `target` weight, negatives get only `alpha * pred^gamma`. The asymmetry
    breaks the symmetric local minimum.

    target can be {0, 1} (binary cls) or continuous IoU in [0, 1] (RTMDet
    style). Continuous target is the recommended combination.
    """
    target = target.float()
    p = torch.sigmoid(logit)
    bce = F.binary_cross_entropy_with_logits(logit, target, reduction='none')
    pos_mask = target > 0
    weight = torch.where(
        pos_mask,
        target,                              # positives weighted by target value
        alpha * p.pow(gamma) * (1.0 - target),  # negatives: confident-wrong only
    )
    loss = weight * bce
    if reduction == 'mean':
        return loss.mean()
    if reduction == 'sum':
        return loss.sum()
    return loss


@torch.no_grad()
def _compute_lineiou_target(
    coord_pred: torch.Tensor,
    points_gt: torch.Tensor,
    visibility: torch.Tensor,
    radius: float,
    target_pow: float = 1.0,
) -> torch.Tensor:
    """Per-prior continuous LineIoU target for cls regression.

    For each prior, compute max LineIoU between its predicted curve and any
    valid GT lane in the same image. Target is in [0, 1]. The caller is
    expected to detach `coord_pred` so this target does not become a moving
    learning target through a feedback loop.

    coord_pred : (B, Q, N, 2) normalized (x, y) per prior, per row
    points_gt  : (B, L, N, 2) normalized GT polylines
    visibility : (B, L, N)    1 for valid GT rows, 0 padded

    Returns: (B, Q) continuous LineIoU target, clamped to [0, 1].
    """
    B, Q, N, _ = coord_pred.shape
    L = points_gt.shape[1]
    target = coord_pred.new_zeros(B, Q)
    if L == 0:
        return target
    pred_x = coord_pred[..., 0]                                # (B, Q, N)
    gt_x = points_gt[..., 0]                                   # (B, L, N)
    for b in range(B):
        # A GT lane is valid if at least one row is visible.
        valid_gt = visibility[b].sum(dim=-1) > 0               # (L,)
        if not valid_gt.any():
            continue
        gt_x_b = gt_x[b][valid_gt]                             # (Lv, N)
        gt_v_b = visibility[b][valid_gt]                       # (Lv, N)
        # Broadcast pred (Q, 1, N) vs gt (1, Lv, N), mask (1, Lv, N).
        ious = _line_iou_1d(
            pred_x[b].unsqueeze(1),
            gt_x_b.unsqueeze(0),
            gt_v_b.unsqueeze(0),
            radius,
        )                                                       # (Q, Lv)
        target[b] = ious.max(dim=-1).values                     # (Q,)
    target = target.clamp(0.0, 1.0)
    if target_pow != 1.0 and target_pow > 0.0:
        target = target.clamp(min=0.0).pow(target_pow)
    return target


@torch.no_grad()
def _compute_mask_consistency_target(
    mask_logit: torch.Tensor,
    coord_pred: torch.Tensor,
) -> torch.Tensor:
    """Per-prior mean sigmoid(mask) sampled along the prior's predicted curve.

    Used by Exp2NN as either a self-distillation target for the cls head or
    as the ranking source at inference. Detached so this target does not
    drive gradients into the mask path; the mask is independently supervised
    by its own BCE+Dice loss.
    """
    if mask_logit is None or mask_logit.dim() != 4:
        return coord_pred.new_zeros(coord_pred.shape[:2])
    grid = (coord_pred * 2.0 - 1.0).clamp(-1.0, 1.0)
    sampled = F.grid_sample(
        mask_logit.detach().float(), grid.float(),
        mode='bilinear', padding_mode='border', align_corners=False,
    )
    return torch.sigmoid(sampled).squeeze(1).mean(dim=-1).clamp(0.0, 1.0)


def _dice_loss(
    logit: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    p = torch.sigmoid(logit)
    num = 2 * (p * target).sum() + eps
    den = (p + target).sum() + eps
    return 1 - num / den


def _smoothness_x(points: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Second-difference penalty on x-coordinate sequences.

    points : (..., N, 2) normalized
    mask   : (..., N) valid-row indicator
    """
    if points.shape[-2] < 3:
        return points.new_zeros(())
    x = points[..., 0]
    d2 = x[..., 2:] - 2 * x[..., 1:-1] + x[..., :-2]
    valid = mask[..., 2:] * mask[..., 1:-1] * mask[..., :-2]
    if valid.sum() < 1:
        return points.new_zeros(())
    return ((d2.abs() * valid).sum() / (valid.sum() + 1e-6))


def _target_lane_param(points: torch.Tensor, vis: torch.Tensor) -> torch.Tensor:
    valid = vis > 0.5
    b, l, n = vis.shape
    out = points.new_zeros((b, l, 4))
    for bi in range(b):
        for li in range(l):
            idx = torch.nonzero(valid[bi, li], as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            first = idx[0]
            last = idx[-1]
            p0 = points[bi, li, first]
            p1 = points[bi, li, last]
            dy = (p1[1] - p0[1]).clamp(min=-1.0, max=1.0)
            dx = (p1[0] - p0[0]).clamp(min=-1.0, max=1.0)
            theta = torch.atan2(dy.abs() + 1e-4, dx.abs() + 1e-4) / torch.pi
            length = idx.numel() / float(max(1, n))
            out[bi, li, 0] = p0[1].clamp(0.0, 1.0)
            out[bi, li, 1] = p0[0].clamp(0.0, 1.0)
            out[bi, li, 2] = theta.clamp(0.0, 1.0)
            out[bi, li, 3] = float(length)
    return out


def _hungarian_match(cost: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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


def _dynamic_k_match(cost: torch.Tensor, line_iou: torch.Tensor, topk: int = 8) -> Tuple[torch.Tensor, torch.Tensor]:
    """CLRKD-style dynamic assignment fallback for lane priors.

    For each GT lane, estimate the number of positive predictions from the sum
    of top-k LineIoU scores, then resolve conflicts by keeping the lowest-cost
    GT for each prediction slot. This is an optional ablation when existence
    recall is too low under strict Hungarian one-to-one matching.
    """
    if cost.numel() == 0:
        device = cost.device
        return torch.zeros((0,), dtype=torch.long, device=device), torch.zeros((0,), dtype=torch.long, device=device)
    q, n = cost.shape
    k = min(int(max(1, topk)), q)
    ious = line_iou.detach().clamp(min=0.0, max=1.0)
    matching = torch.zeros((q, n), dtype=torch.bool, device=cost.device)
    for gi in range(n):
        topk_ious, _ = torch.topk(ious[:, gi], k=k, largest=True)
        dynamic_k = int(torch.clamp(topk_ious.sum().round(), min=1, max=k).item())
        _, pred_idx = torch.topk(cost[:, gi], k=dynamic_k, largest=False)
        matching[pred_idx, gi] = True
    multi = matching.sum(dim=1) > 1
    if multi.any():
        conflicted = torch.nonzero(multi, as_tuple=False).flatten()
        best_gt = torch.argmin(cost[conflicted], dim=1)
        matching[conflicted] = False
        matching[conflicted, best_gt] = True
    pred_idx, gt_idx = torch.nonzero(matching, as_tuple=True)
    if pred_idx.numel() == 0:
        return _hungarian_match(cost)
    return pred_idx.long(), gt_idx.long()


# ---------------------------------------------------------------------------
# Lane fusion loss
# ---------------------------------------------------------------------------
class FusionLaneLoss(nn.Module):
    """Lane-side loss for the in-house curve lane head.

    Predictions (per-image, fixed max_lanes slots):
        cls_logits  : (B, max_lanes)               existence logit
        coord_pred  : (B, max_lanes, num_points, 2) normalized (x, y)
        mask_logit  : (B, 1, H, W) (optional)      auxiliary rendered mask
        cls_logits_dist (optional)                 teacher-style logits for KD
        coord_pred_dist (optional)                 teacher coord for KD

    Targets:
        existence   : (B, max_lanes) {0, 1}
        points      : (B, max_lanes, num_points, 2) normalized
        visibility  : (B, max_lanes, num_points)    {0, 1}
        mask_target : (B, 1, H, W) (optional)
    """

    def __init__(self, cfg: FusionLossConfig = None):
        super().__init__()
        self.cfg = cfg or FusionLossConfig()

    def match_targets(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        cfg = self.cfg
        if not bool(getattr(cfg, 'use_lane_matching', False)):
            return target
        cls_logit = pred['cls_logits']
        coord_pred = pred['coord_pred']
        existence = target['existence']
        points_gt = target['points']
        vis = target['visibility']
        bsz, pred_lanes, num_points, _ = coord_pred.shape
        matched = {}
        matched['existence'] = coord_pred.new_zeros((bsz, pred_lanes))
        matched['points'] = coord_pred.new_zeros((bsz, pred_lanes, num_points, 2))
        matched['visibility'] = coord_pred.new_zeros((bsz, pred_lanes, num_points))
        if 'lane_type' in target and torch.is_tensor(target['lane_type']):
            matched['lane_type'] = torch.full((bsz, pred_lanes), -1, dtype=target['lane_type'].dtype, device=target['lane_type'].device)
        for key, value in target.items():
            if key in matched:
                continue
            matched[key] = value
        if 'mask_target' in target and torch.is_tensor(target['mask_target']):
            matched['mask_target'] = target['mask_target']
        with torch.no_grad():
            param_gt_all = _target_lane_param(points_gt, vis)
            param_pred = pred.get('lane_param')
            if param_pred is None:
                param_pred = coord_pred.new_zeros((bsz, pred_lanes, 4))
            for b in range(bsz):
                gt_idx = torch.nonzero(existence[b] > 0.5, as_tuple=False).flatten()
                if gt_idx.numel() == 0:
                    continue
                gt_points = points_gt[b, gt_idx]
                gt_vis = vis[b, gt_idx]
                valid = gt_vis.unsqueeze(0)
                point_cost = ((coord_pred[b].unsqueeze(1) - gt_points.unsqueeze(0)).abs().sum(-1) * valid).sum(-1) / (valid.sum(-1) + 1e-6)
                line_iou = _line_iou_1d(coord_pred[b, :, None, :, 0], gt_points[None, :, :, 0], valid, cfg.line_iou_radius)
                iou_cost = 1.0 - line_iou
                xytl_cost = (param_pred[b].unsqueeze(1) - param_gt_all[b, gt_idx].unsqueeze(0)).abs().mean(-1)
                cls_cost = -torch.sigmoid(cls_logit[b]).unsqueeze(1).expand_as(point_cost)
                cost = (
                    float(cfg.match_cost_cls) * cls_cost
                    + float(cfg.match_cost_point) * point_cost
                    + float(cfg.match_cost_iou) * iou_cost
                    + float(cfg.match_cost_xytl) * xytl_cost
                )
                assigner = str(getattr(cfg, 'lane_assigner', 'hungarian')).lower()
                if assigner in {'dynamic_k', 'dynamic', 'clrkd_dynamic_k'}:
                    pred_idx, local_gt_idx = _dynamic_k_match(cost, line_iou, topk=int(getattr(cfg, 'dynamic_k_topk', 8)))
                else:
                    pred_idx, local_gt_idx = _hungarian_match(cost)
                if pred_idx.numel() == 0:
                    continue
                src_gt_idx = gt_idx[local_gt_idx]
                matched['existence'][b, pred_idx] = existence[b, src_gt_idx]
                matched['points'][b, pred_idx] = points_gt[b, src_gt_idx]
                matched['visibility'][b, pred_idx] = vis[b, src_gt_idx]
                if 'lane_type' in target and torch.is_tensor(target['lane_type']):
                    matched['lane_type'][b, pred_idx] = target['lane_type'][b, src_gt_idx]
        return matched

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        teacher: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        total, components = self._forward_single_stage(pred, target, teacher)

        # Exp2T: HybridPriorQueryHead path -- supervise stage 1's 192 prior
        # outputs with the same lane loss when stage1_aux_loss_weight > 0.
        # This fixes Exp2Q's diagnosed failure to inherit Exp2N geometry by
        # giving stage 1's coord_pred direct supervision instead of relying
        # on indirect gradient through stage 2's per-prior feature reads.
        stage1_weight = float(getattr(self.cfg, 'stage1_aux_loss_weight', 0.0))
        if stage1_weight > 0.0 and isinstance(pred, dict) and 'stage1_coord_pred' in pred:
            stage1_input = {
                'cls_logits': pred['stage1_cls_logits'],
                'coord_pred': pred['stage1_coord_pred'],
                'lane_param': pred.get('stage1_lane_param'),
                'lane_offsets': pred.get('stage1_lane_offsets'),
            }
            stage1_total, stage1_comp = self._forward_single_stage(stage1_input, target, None)
            total = total + stage1_weight * stage1_total
            components['lane/stage1_aux_total'] = stage1_total.detach()
            components['lane/stage1_aux_cls'] = stage1_comp.get('lane/cls', stage1_total.new_zeros(())).detach()
            components['lane/stage1_aux_iou'] = stage1_comp.get('lane/line_iou', stage1_total.new_zeros(())).detach()
            components['lane/stage1_aux_reg'] = stage1_comp.get('lane/reg', stage1_total.new_zeros(())).detach()

        # CLRKDLaneHead path: add auxiliary loss on each intermediate refinement
        # stage. The final stage is already covered above; we only iterate over
        # aux_stage_outputs (everything except the last). Auxiliary mask /
        # distillation are not duplicated -- they are tied to the final-stage
        # feature map only.
        aux_stages = pred.get('aux_stage_outputs') if isinstance(pred, dict) else None
        if aux_stages and self.cfg.aux_stage_loss_weight > 0.0:
            num_aux = len(aux_stages)
            num_total = num_aux + 1  # final stage + aux stages
            for i, stage_pred in enumerate(aux_stages):
                stage_input = {
                    'cls_logits': stage_pred['cls_logits'],
                    'coord_pred': stage_pred['coord_pred'],
                    'lane_param': stage_pred.get('lane_param'),
                    'lane_offsets': stage_pred.get('lane_offsets'),
                }
                # Aux stages skip mask/distill; weight grows as we approach final.
                stage_weight = self.cfg.aux_stage_loss_weight * (i + 1) / float(num_total)
                stage_total, stage_comp = self._forward_single_stage(stage_input, target, None)
                total = total + stage_weight * stage_total
                components[f'lane/aux{i}_total'] = stage_total.detach()
        return total, components

    def _forward_single_stage(
        self,
        pred: Dict[str, torch.Tensor],
        target: Dict[str, torch.Tensor],
        teacher: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        cfg = self.cfg
        target = self.match_targets(pred, target)

        cls_logit = pred['cls_logits']                  # (B, L)
        coord_pred = pred['coord_pred']                 # (B, L, N, 2)

        existence = target['existence']                 # (B, L)
        points_gt = target['points']                    # (B, L, N, 2)
        vis = target['visibility']                      # (B, L, N)

        # 1) Existence / classification.
        # Two formulations dispatched by cfg.cls_target_type:
        #
        # 'matched_existence' (default, CLRKDNet original): binary {0, 1} from
        #   the dynamic-k matching outcome. Uses focal or ASL via cls_loss_type.
        #   Optional OHEM over hardest negatives. Optional balanced averaging.
        #
        # 'lineiou_regression' (Exp2K): continuous target = max LineIoU between
        #   this prior's predicted curve (detached) and any valid GT lane in
        #   the same image. Plain BCE-with-logits, no OHEM/focal/ASL/balanced
        #   averaging because the continuous balanced target makes those moot.
        #   Removes the matching-instability confound that drove cls to a
        #   degenerate equilibrium across Exp2G/H/I/J.
        pos_mask = existence > 0.5
        neg_mask = ~pos_mask
        cls_target_type = str(getattr(cfg, 'cls_target_type', 'matched_existence')).lower()
        if cls_target_type in {'lineiou_regression', 'iou_regression', 'iou'}:
            iou_target = _compute_lineiou_target(
                coord_pred.detach(), points_gt, vis,
                radius=float(cfg.line_iou_radius),
                target_pow=float(getattr(cfg, 'lineiou_target_pow', 1.0)),
            )
            cls_loss_subtype = str(getattr(cfg, 'cls_loss_type', 'focal')).lower()
            if cls_loss_subtype == 'qfl':
                cls_raw = _quality_focal_loss(
                    cls_logit, iou_target,
                    gamma=float(getattr(cfg, 'qfl_gamma', 2.0)),
                    reduction='none',
                )
            elif cls_loss_subtype in {'vfl', 'varifocal'}:
                # Exp2QQ: VFL on continuous IoU target. Positives weighted by
                # target IoU; negatives only count when sigmoid is high (i.e.,
                # the model is confidently wrong). Breaks the alpha=0.25
                # degenerate equilibrium that has been collapsing cls across
                # NB39-NB45.
                cls_raw = _varifocal_loss(
                    cls_logit, iou_target,
                    alpha=float(getattr(cfg, 'vfl_alpha', 0.75)),
                    gamma=float(getattr(cfg, 'vfl_gamma', 2.0)),
                    reduction='none',
                )
            else:
                cls_raw = F.binary_cross_entropy_with_logits(
                    cls_logit, iou_target, reduction='none',
                )
            cls_loss = cls_raw.mean()
            cls_pos_loss = cls_raw[pos_mask].mean() if pos_mask.any() else cls_logit.new_zeros(())
            cls_neg_loss = cls_raw[neg_mask].mean() if neg_mask.any() else cls_logit.new_zeros(())
        elif cls_target_type in {'mask_consistency', 'mask', 'mask_aux'}:
            # Exp2NN: self-distillation. cls target = mean sigmoid(mask)
            # sampled along the prior's predicted curve (detached). The aux
            # mask is trained independently via BCE+Dice on the rendered GT
            # lanes -- it has no per-prior matching instability, so its
            # sigmoid is a clean per-prior geometric verifier. Forcing cls
            # to mimic the mask score gives the cls head a stable, deterministic
            # supervision target across batches and a discriminative signal
            # that actually correlates with "is this prior near a real lane?"
            mask_target = _compute_mask_consistency_target(
                pred.get('mask_logit'), coord_pred.detach(),
            )
            cls_loss_subtype = str(getattr(cfg, 'cls_loss_type', 'bce')).lower()
            if cls_loss_subtype == 'qfl':
                cls_raw = _quality_focal_loss(
                    cls_logit, mask_target,
                    gamma=float(getattr(cfg, 'qfl_gamma', 2.0)),
                    reduction='none',
                )
            else:
                cls_raw = F.binary_cross_entropy_with_logits(
                    cls_logit, mask_target, reduction='none',
                )
            cls_loss = cls_raw.mean()
            cls_pos_loss = cls_raw[pos_mask].mean() if pos_mask.any() else cls_logit.new_zeros(())
            cls_neg_loss = cls_raw[neg_mask].mean() if neg_mask.any() else cls_logit.new_zeros(())
        else:
            cls_raw = _binary_cls_raw(cls_logit, existence, cfg)
            cls_pos_loss = cls_raw[pos_mask].mean() if pos_mask.any() else cls_logit.new_zeros(())
            ohem_topk_per_pos = int(getattr(cfg, 'cls_ohem_topk_per_pos', 0) or 0)
            if ohem_topk_per_pos > 0 and neg_mask.any():
                ohem_min_topk = int(getattr(cfg, 'cls_ohem_min_topk', 32))
                losses_per_image = []
                num_pos_per_image = pos_mask.sum(dim=-1)
                for b in range(cls_raw.shape[0]):
                    neg_b = neg_mask[b]
                    if not neg_b.any():
                        continue
                    neg_losses_b = cls_raw[b][neg_b]
                    num_pos_b = int(num_pos_per_image[b].item())
                    k = max(ohem_min_topk, ohem_topk_per_pos * num_pos_b)
                    k = min(k, neg_losses_b.numel())
                    topk_losses, _ = torch.topk(neg_losses_b, k=k, largest=True)
                    losses_per_image.append(topk_losses)
                if losses_per_image:
                    cls_neg_loss = torch.cat(losses_per_image).mean()
                else:
                    cls_neg_loss = cls_logit.new_zeros(())
            else:
                cls_neg_loss = cls_raw[neg_mask].mean() if neg_mask.any() else cls_logit.new_zeros(())
            if str(getattr(cfg, 'cls_balance_mode', 'balanced')).lower() in {'balanced', 'pos_neg_balanced', 'matched_balanced'}:
                cls_loss = 0.5 * cls_pos_loss + 0.5 * cls_neg_loss
            else:
                cls_loss = cls_raw.mean()

        # 2) Coordinate regression — only on valid lanes/rows.
        valid_lane = existence.unsqueeze(-1)            # (B, L, 1)
        valid_row = valid_lane * vis                    # (B, L, N)
        diff = (coord_pred - points_gt).abs()           # (B, L, N, 2)
        # SmoothL1 manually, with per-row mask.
        beta = 0.05
        sl1 = torch.where(diff < beta, 0.5 * diff * diff / beta, diff - 0.5 * beta)
        reg_loss = (sl1.sum(-1) * valid_row).sum() / (valid_row.sum() * 2 + 1e-6)

        # 3) CLRKD-style [start_y, start_x, theta, length] regression.
        if 'lane_param' in pred and pred['lane_param'] is not None:
            param_gt = _target_lane_param(points_gt, vis)
            param_pred = pred['lane_param']
            param_diff = (param_pred - param_gt).abs()
            beta_param = 0.05
            param_sl1 = torch.where(param_diff < beta_param, 0.5 * param_diff * param_diff / beta_param, param_diff - 0.5 * beta_param)
            xytl_loss = (param_sl1.sum(-1) * existence).sum() / (existence.sum() * 4 + 1e-6)
        else:
            xytl_loss = cls_logit.new_zeros(())

        # 4) LineIoU on x-coordinates (per lane).
        pred_x = coord_pred[..., 0]
        gt_x = points_gt[..., 0]
        iou = _line_iou_1d(pred_x, gt_x, vis, radius=cfg.line_iou_radius)  # (B, L)
        iou_loss = ((1 - iou) * existence).sum() / (existence.sum() + 1e-6)

        # 5) Smoothness regularization on predicted curves.
        smooth_loss = _smoothness_x(coord_pred, valid_row)

        # 6) Auxiliary mask loss (Dice + BCE).
        if 'mask_logit' in pred and 'mask_target' in target and pred['mask_logit'] is not None:
            mlogit = pred['mask_logit']
            mtarget = target['mask_target']
            bce = F.binary_cross_entropy_with_logits(mlogit, mtarget.float())
            dice = _dice_loss(mlogit, mtarget.float())
            mask_loss = 0.5 * bce + 0.5 * dice
        else:
            mask_loss = cls_logit.new_zeros(())

        # 7) Optional distillation.
        distill_loss = cls_logit.new_zeros(())
        if cfg.w_distill > 0 and teacher is not None:
            t_cls = teacher.get('cls_logits')
            t_coord = teacher.get('coord_pred')
            if t_cls is not None and t_cls.shape == cls_logit.shape:
                distill_loss = distill_loss + F.mse_loss(
                    torch.sigmoid(cls_logit), torch.sigmoid(t_cls).detach()
                )
            if t_coord is not None and t_coord.shape == coord_pred.shape:
                distill_loss = distill_loss + (
                    (coord_pred - t_coord.detach()).abs() * valid_row.unsqueeze(-1)
                ).sum() / (valid_row.sum() * 2 + 1e-6)

        # 8) Exp2MM: dual-scoring auxiliary IoU regression head supervision.
        # When CLRKDLaneHead is built with dual_score=True, it emits a parallel
        # iou_logits tensor. We supervise it on the continuous LineIoU target
        # while the binary cls_logits keeps its matched_existence supervision.
        iou_aux_loss = cls_logit.new_zeros(())
        w_iou_aux = float(getattr(cfg, 'w_iou_aux', 0.0))
        if w_iou_aux > 0.0 and 'iou_logits' in pred and pred['iou_logits'] is not None:
            iou_logit = pred['iou_logits']
            if iou_logit.shape == cls_logit.shape:
                iou_aux_target = _compute_lineiou_target(
                    coord_pred.detach(), points_gt, vis,
                    radius=float(cfg.line_iou_radius),
                    target_pow=float(getattr(cfg, 'iou_aux_target_pow', 1.0)),
                )
                iou_aux_subtype = str(getattr(cfg, 'iou_aux_loss_type', 'bce')).lower()
                if iou_aux_subtype == 'qfl':
                    iou_aux_raw = _quality_focal_loss(
                        iou_logit, iou_aux_target,
                        gamma=float(getattr(cfg, 'iou_aux_qfl_gamma', 2.0)),
                        reduction='none',
                    )
                else:
                    iou_aux_raw = F.binary_cross_entropy_with_logits(
                        iou_logit, iou_aux_target, reduction='none',
                    )
                iou_aux_loss = iou_aux_raw.mean()

        total = (
            cfg.w_cls * cls_loss
            + cfg.w_reg * reg_loss
            + cfg.w_xytl * xytl_loss
            + cfg.w_iou * iou_loss
            + cfg.w_mask * mask_loss
            + cfg.w_smooth * smooth_loss
            + cfg.w_distill * distill_loss
            + w_iou_aux * iou_aux_loss
        )
        unweighted_total = cls_loss + reg_loss + xytl_loss + iou_loss + mask_loss + smooth_loss + distill_loss + iou_aux_loss
        geometry_raw = reg_loss + xytl_loss + iou_loss + smooth_loss
        geometry_weighted = cfg.w_reg * reg_loss + cfg.w_xytl * xytl_loss + cfg.w_iou * iou_loss + cfg.w_smooth * smooth_loss
        components = {
            'lane/cls': cls_loss.detach(),
            'lane/cls_pos': cls_pos_loss.detach(),
            'lane/cls_neg': cls_neg_loss.detach(),
            'lane/reg': reg_loss.detach(),
            'lane/xytl': xytl_loss.detach(),
            'lane/line_iou': iou_loss.detach(),
            'lane/mask_aux': mask_loss.detach(),
            'lane/smooth': smooth_loss.detach(),
            'lane/distill': distill_loss.detach(),
            'lane/iou_aux': iou_aux_loss.detach(),
            'lane/geometry_raw': geometry_raw.detach(),
            'lane/geometry_weighted': geometry_weighted.detach(),
            'lane/unweighted_total': unweighted_total.detach(),
            'lane/weighted_total': total.detach(),
            'lane/total': total.detach(),
        }
        return total, components


@torch.no_grad()
def compute_lane_eval_metrics(
    pred: Dict[str, torch.Tensor],
    target: Dict[str, torch.Tensor],
    lane_loss: Optional[FusionLaneLoss] = None,
    threshold: Optional[float] = None,
) -> Dict[str, torch.Tensor]:
    """Evaluate lane existence and geometry after optional Hungarian matching.

    The old single ``lane_exist_acc`` number was hard to interpret because it
    mixed background slots and real lane slots. This helper keeps that metric
    for continuity, but also reports precision/recall/F1, matched-lane MAE,
    LineIoU, and FP/FN slot counts.
    """
    cls_logit = pred['cls_logits']
    coord_pred = pred['coord_pred']
    device = cls_logit.device

    if lane_loss is not None and bool(getattr(lane_loss.cfg, 'use_lane_matching', False)):
        matched_target = lane_loss.match_targets(pred, target)
        radius = float(getattr(lane_loss.cfg, 'line_iou_radius', 0.015))
        default_threshold = float(getattr(lane_loss.cfg, 'existence_threshold', 0.5))
    else:
        matched_target = target
        radius = 0.015
        default_threshold = 0.5
    if threshold is None:
        threshold = default_threshold

    gt_exist_original = target['existence'] > 0.5
    gt_exist = matched_target['existence'] > 0.5
    pred_exist = torch.sigmoid(cls_logit) >= float(threshold)

    tp = (pred_exist & gt_exist).float().sum()
    fp = (pred_exist & ~gt_exist).float().sum()
    fn = (~pred_exist & gt_exist).float().sum()
    tn = (~pred_exist & ~gt_exist).float().sum()
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-6)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-6)

    visibility = matched_target['visibility']
    valid = gt_exist.float().unsqueeze(-1) * visibility
    if valid.sum() > 0:
        point_mae = ((coord_pred - matched_target['points']).abs().sum(-1) * valid).sum() / (valid.sum() + 1e-6)
        line_iou = _line_iou_1d(coord_pred[..., 0], matched_target['points'][..., 0], visibility, radius=radius)
        matched_line_iou = (line_iou * gt_exist.float()).sum() / (gt_exist.float().sum() + 1e-6)
    else:
        point_mae = cls_logit.new_zeros(())
        matched_line_iou = cls_logit.new_zeros(())

    score = torch.sigmoid(cls_logit)
    best_f1 = cls_logit.new_zeros(())
    best_thr = cls_logit.new_zeros(())
    for thr in torch.linspace(0.05, 0.95, steps=19, device=score.device):
        cand = score >= thr
        cand_tp = (cand & gt_exist).float().sum()
        cand_fp = (cand & ~gt_exist).float().sum()
        cand_fn = (~cand & gt_exist).float().sum()
        cand_prec = cand_tp / (cand_tp + cand_fp + 1e-6)
        cand_rec = cand_tp / (cand_tp + cand_fn + 1e-6)
        cand_f1 = 2.0 * cand_prec * cand_rec / (cand_prec + cand_rec + 1e-6)
        if cand_f1 > best_f1:
            best_f1 = cand_f1
            best_thr = thr
    pos_scores = score[gt_exist]
    neg_scores = score[~gt_exist]
    pos_score_mean = pos_scores.mean() if pos_scores.numel() else cls_logit.new_zeros(())
    neg_score_mean = neg_scores.mean() if neg_scores.numel() else cls_logit.new_zeros(())

    return {
        'lane_exist_acc': acc,
        'lane_exist_precision': precision,
        'lane_exist_recall': recall,
        'lane_exist_f1': f1,
        'lane_exist_best_f1': best_f1,
        'lane_exist_best_threshold': best_thr,
        'lane_exist_pos_score_mean': pos_score_mean,
        'lane_exist_neg_score_mean': neg_score_mean,
        'lane_point_mae': point_mae,
        'matched_lane_point_mae': point_mae,
        'matched_line_iou': matched_line_iou,
        'num_gt_lanes': gt_exist_original.float().sum(),
        'num_pred_lanes': pred_exist.float().sum(),
        'num_matched_lanes': gt_exist.float().sum(),
        'false_positive_lane_slots': fp,
        'false_negative_lane_slots': fn,
    }


# ---------------------------------------------------------------------------
# Multi-task uncertainty weighting (Kendall, Gal, Cipolla 2018)
# ---------------------------------------------------------------------------
class UncertaintyMultiTaskLoss(nn.Module):
    """Weighted sum L = sum_i (1 / (2*exp(s_i))) * L_i + s_i / 2 with
    learnable scalars s_i (s_i = log_variance). Uses 1/(2 sigma^2) for
    regression-like losses; ok for cross-entropy too in practice.
    """

    def __init__(self, n_tasks: int = 2, init: float = 0.0):
        super().__init__()
        self.log_var = nn.Parameter(torch.full((n_tasks,), float(init)))

    def forward(self, losses: Iterable[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        losses = list(losses)
        assert len(losses) == self.log_var.numel(), (
            f'UncertaintyMultiTaskLoss got {len(losses)} losses, expects {self.log_var.numel()}.'
        )
        total = losses[0].new_zeros(())
        comp: Dict[str, torch.Tensor] = {}
        for i, l in enumerate(losses):
            s = self.log_var[i]
            weighted = 0.5 * torch.exp(-s) * l + 0.5 * s
            total = total + weighted
            comp[f'mtl/log_var_{i}'] = s.detach()
            comp[f'mtl/weighted_{i}'] = weighted.detach()
        comp['mtl/total'] = total.detach()
        return total, comp


# ---------------------------------------------------------------------------
# Gradient cosine diagnostic for task-conflict measurement
# ---------------------------------------------------------------------------
def compute_grad_norm_ratio(
    loss_a: torch.Tensor,
    loss_b: torch.Tensor,
    shared_params: Iterable[torch.nn.Parameter],
    clamp_min: float = 0.05,
    clamp_max: float = 2.0,
) -> float:
    shared_params = [p for p in shared_params if p.requires_grad]
    if not shared_params:
        return 1.0
    g_a = torch.autograd.grad(loss_a, shared_params, retain_graph=True, create_graph=False, allow_unused=True)
    g_b = torch.autograd.grad(loss_b, shared_params, retain_graph=True, create_graph=False, allow_unused=True)
    norm_a = loss_a.new_zeros(())
    norm_b = loss_b.new_zeros(())
    for ga, gb in zip(g_a, g_b):
        if ga is not None:
            norm_a = norm_a + ga.detach().pow(2).sum()
        if gb is not None:
            norm_b = norm_b + gb.detach().pow(2).sum()
    norm_a = norm_a.sqrt()
    norm_b = norm_b.sqrt()
    ratio = (norm_a / (norm_b + 1e-12)).clamp(clamp_min, clamp_max)
    return float(ratio.item())


def compute_grad_cosine(
    loss_a: torch.Tensor,
    loss_b: torch.Tensor,
    shared_params: Iterable[torch.nn.Parameter],
) -> float:
    """Cosine similarity between gradients of two losses w.r.t. shared params.

    Negative values indicate task conflict on the shared backbone.
    Caller must NOT have called .backward() yet.
    """
    shared_params = [p for p in shared_params if p.requires_grad]
    if not shared_params:
        return float('nan')
    g_a = torch.autograd.grad(loss_a, shared_params, retain_graph=True, create_graph=False, allow_unused=True)
    g_b = torch.autograd.grad(loss_b, shared_params, retain_graph=True, create_graph=False, allow_unused=True)
    flat_a = []
    flat_b = []
    for ga, gb in zip(g_a, g_b):
        if ga is None or gb is None:
            continue
        flat_a.append(ga.flatten())
        flat_b.append(gb.flatten())
    if not flat_a:
        return float('nan')
    va = torch.cat(flat_a)
    vb = torch.cat(flat_b)
    denom = (va.norm() * vb.norm()).clamp(min=1e-12)
    return float((va @ vb / denom).item())
