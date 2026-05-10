"""Top-K lane decode + lane-NMS for the CLRKDLaneHead output.

This module implements the inference-time pipeline that turns the head's raw
192-prior output into a small set of ranked lane curves -- the missing piece
that has been making `val/lane/clrkd_style_f1` report 0 across every Exp2 run
even when the geometry is good.

Design follows external_repos/CLRKDNet-master (top-K by score, suppress
overlapping curves by LineIoU). Pure PyTorch, no MMCV / no GPU op kernels.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from .losses import _line_iou_1d


def decode_top_k_lanes(
    pred: Dict[str, torch.Tensor],
    top_k: int = 4,
    score_threshold: float = 0.0,
    scores_override: Optional[torch.Tensor] = None,
) -> List[Dict[str, torch.Tensor]]:
    """Return per-image top-K predicted lane curves ranked by sigmoid(cls_logit).

    Args
    ----
    pred:
        CLRKDLaneHead output dict. Required keys: 'cls_logits' (B, P),
        'coord_pred' (B, P, N, 2). Optional 'visibility' (B, P, N) is ignored
        here (the head's offsets path produces dense per-row coordinates so
        every row is "visible").
    top_k:
        Maximum lanes to keep per image, before NMS.
    score_threshold:
        If > 0, drop priors whose sigmoid score is below this floor before
        top-K selection. Default 0.0 means top-K by descending score with no
        floor -- the right behaviour when the model's score distribution is
        compressed (which has been the case across every Exp2 run).
    scores_override:
        If provided, a (B, P) tensor used as the per-prior ranking score in
        place of sigmoid(cls_logits). Set by `LaneF1DecodedMetric` when its
        score_source='oracle_iou', so the diagnostic upper-bound metric can
        rank priors by ground-truth LineIoU instead of the cls head's output.
        Should already be in [0, 1] range; not re-sigmoided.

    Returns
    -------
    List of length B. Each item is a dict:
        'coord_xy': (Kp, N, 2) tensor on the same device as pred. Kp <= top_k.
        'score'   : (Kp,) tensor of sigmoid scores.
        'prior_id': (Kp,) long tensor of prior indices.
    The list is unbatched because per-image Kp can differ once score_threshold
    > 0, and lane NMS in the next stage operates per image anyway.
    """
    cls_logits = pred['cls_logits']                # (B, P)
    coord_pred = pred['coord_pred']                # (B, P, N, 2)
    if cls_logits.dim() != 2 or coord_pred.dim() != 4:
        raise RuntimeError(
            'decode_top_k_lanes expects cls_logits (B, P) and coord_pred (B, P, N, 2); '
            f'got {tuple(cls_logits.shape)} and {tuple(coord_pred.shape)}'
        )
    if scores_override is not None:
        if scores_override.shape != cls_logits.shape:
            raise RuntimeError(
                'scores_override must have shape (B, P) matching cls_logits; '
                f'got {tuple(scores_override.shape)} vs {tuple(cls_logits.shape)}'
            )
        scores = scores_override.to(cls_logits.dtype)
    else:
        scores = cls_logits.sigmoid()              # (B, P)
    bsz, num_priors = scores.shape
    k = max(1, min(int(top_k), num_priors))
    out: List[Dict[str, torch.Tensor]] = []
    for b in range(bsz):
        s = scores[b]                              # (P,)
        if score_threshold > 0.0:
            keep_mask = s >= float(score_threshold)
            if keep_mask.any():
                s_keep = s[keep_mask]
                idx_full = torch.arange(num_priors, device=s.device)[keep_mask]
                kk = min(k, s_keep.numel())
                top_scores, top_local = torch.topk(s_keep, k=kk, largest=True)
                top_idx = idx_full[top_local]
            else:
                top_scores = s.new_zeros((0,))
                top_idx = torch.zeros((0,), dtype=torch.long, device=s.device)
        else:
            top_scores, top_idx = torch.topk(s, k=k, largest=True)
        out.append({
            'coord_xy': coord_pred[b, top_idx],     # (Kp, N, 2)
            'score': top_scores,                    # (Kp,)
            'prior_id': top_idx.long(),             # (Kp,)
        })
    return out


def lane_nms(
    decoded: List[Dict[str, torch.Tensor]],
    line_iou_threshold: float = 0.5,
    radius: float = 0.015,
) -> List[Dict[str, torch.Tensor]]:
    """Suppress lower-scored lanes whose LineIoU with a higher-scored kept
    lane exceeds `line_iou_threshold`.

    Greedy by descending score. Implementation uses the existing
    `_line_iou_1d` from `stage2.fusion.losses` so the IoU semantics match the
    training-time line_iou loss exactly (band radius, mask handling).
    """
    out: List[Dict[str, torch.Tensor]] = []
    for item in decoded:
        scores = item['score']
        coord = item['coord_xy']                  # (Kp, N, 2)
        priors = item['prior_id']
        if scores.numel() == 0:
            out.append(item)
            continue
        # Sort by descending score; we walk top-down and keep non-overlapping.
        order = torch.argsort(scores, descending=True)
        coord_sorted = coord[order]
        scores_sorted = scores[order]
        priors_sorted = priors[order]
        n = scores_sorted.numel()
        kept_mask = torch.ones(n, dtype=torch.bool, device=scores.device)
        # Pre-compute pairwise LineIoU on x-coordinates (lanes are 1-D in x
        # along fixed y rows). Mask of all-1 since the head emits dense rows.
        pred_x = coord_sorted[..., 0]              # (n, N)
        # Broadcast (n, 1, N) vs (1, n, N) for pairwise IoU.
        full_mask = torch.ones_like(pred_x)
        ious = _line_iou_1d(
            pred_x[:, None, :], pred_x[None, :, :],
            full_mask[None, :, :].expand(n, n, -1), radius,
        )                                          # (n, n)
        for i in range(n):
            if not kept_mask[i]:
                continue
            # Suppress all later j whose IoU with i exceeds threshold.
            for j in range(i + 1, n):
                if kept_mask[j] and ious[i, j] >= line_iou_threshold:
                    kept_mask[j] = False
        out.append({
            'coord_xy': coord_sorted[kept_mask],
            'score': scores_sorted[kept_mask],
            'prior_id': priors_sorted[kept_mask],
        })
    return out


if __name__ == '__main__':
    # Documentation-only smoke check; not executed during training.
    torch.manual_seed(0)
    cls = torch.randn(2, 16) * 0.5
    coord = torch.rand(2, 16, 8, 2)
    out = decode_top_k_lanes({'cls_logits': cls, 'coord_pred': coord}, top_k=4)
    assert len(out) == 2 and out[0]['coord_xy'].shape == (4, 8, 2)
    after = lane_nms(out, line_iou_threshold=0.5)
    assert len(after) == 2
    print('lane_decode self-check OK', [a['coord_xy'].shape for a in after])
