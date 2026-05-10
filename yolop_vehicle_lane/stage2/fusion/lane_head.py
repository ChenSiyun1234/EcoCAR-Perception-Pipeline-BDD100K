from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _conv_bn_act(c_in: int, c_out: int, k: int = 3, s: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=k, stride=s, padding=k // 2, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class CurveLaneHead(nn.Module):
    def __init__(
        self,
        in_channels: Sequence[int] = (128, 128, 128),
        embed_dim: int = 128,
        max_lanes: int = 10,
        num_points: int = 72,
        mask_size: Tuple[int, int] = (72, 128),
        mask_aux: bool = True,
        num_lane_classes: int = 7,
        num_priors: int = 192,
        output_all_priors: bool = False,
        use_roi_gather: bool = False,
        roi_refine_layers: int = 1,
    ):
        super().__init__()
        self.in_channels = list(in_channels)
        self.embed_dim = int(embed_dim)
        self.max_lanes = int(max_lanes)
        self.num_points = int(num_points)
        self.mask_size = (int(mask_size[0]), int(mask_size[1]))
        self.mask_aux = bool(mask_aux)
        self.num_lane_classes = int(num_lane_classes)
        self.num_priors = int(num_priors)
        self.output_all_priors = bool(output_all_priors)
        self.use_roi_gather = bool(use_roi_gather)
        self.roi_refine_layers = max(1, int(roi_refine_layers))
        self.lateral = nn.ModuleList([_conv_bn_act(c, embed_dim, k=1) for c in in_channels])
        self.refine = _conv_bn_act(embed_dim, embed_dim, k=3)
        self.prior_embeddings = nn.Embedding(self.num_priors, 3)
        self._init_prior_embeddings()
        self.query_proj = nn.Linear(3, embed_dim)
        self.cls_head = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(inplace=True), nn.Linear(embed_dim, 1))
        self.param_head = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(inplace=True), nn.Linear(embed_dim, 4))
        self.offset_head = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(inplace=True), nn.Linear(embed_dim, num_points))
        self.lane_class_head = nn.Linear(embed_dim, num_lane_classes)
        self.score_select = nn.Linear(embed_dim, 1)
        self.roi_context = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=max(1, embed_dim // 32), bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(embed_dim, embed_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.roi_proj = nn.Linear(embed_dim, embed_dim)
        self.roi_norm = nn.LayerNorm(embed_dim)
        if self.mask_aux:
            self.mask_decoder = nn.Sequential(_conv_bn_act(embed_dim, max(16, embed_dim // 2), k=3), nn.Conv2d(max(16, embed_dim // 2), 1, kernel_size=1))
        self.register_buffer('prior_ys', torch.linspace(1.0, 0.0, steps=self.num_points, dtype=torch.float32))

    def _init_prior_embeddings(self) -> None:
        bottom_priors_nums = self.num_priors * 3 // 4
        left_priors_nums = self.num_priors // 8
        right_start = left_priors_nums + bottom_priors_nums
        left_denom = max(1, left_priors_nums // 2 - 1)
        strip_size = 0.5 / left_denom
        bottom_strip_size = 1.0 / max(1, bottom_priors_nums // 4 + 1)
        with torch.no_grad():
            for i in range(left_priors_nums):
                self.prior_embeddings.weight[i, 0] = (i // 2) * strip_size
                self.prior_embeddings.weight[i, 1] = 0.0
                self.prior_embeddings.weight[i, 2] = 0.16 if i % 2 == 0 else 0.32
            for i in range(left_priors_nums, right_start):
                self.prior_embeddings.weight[i, 0] = 0.0
                self.prior_embeddings.weight[i, 1] = ((i - left_priors_nums) // 4 + 1) * bottom_strip_size
                self.prior_embeddings.weight[i, 2] = 0.2 * ((i - left_priors_nums) % 4 + 1)
            for i in range(right_start, self.num_priors):
                self.prior_embeddings.weight[i, 0] = ((i - right_start) // 2) * strip_size
                self.prior_embeddings.weight[i, 1] = 1.0
                self.prior_embeddings.weight[i, 2] = 0.68 if i % 2 == 0 else 0.84

    def _topdown(self, feats: List[torch.Tensor]) -> torch.Tensor:
        proj = [layer(f) for layer, f in zip(self.lateral, feats)]
        order = sorted(range(len(proj)), key=lambda i: proj[i].shape[-2] * proj[i].shape[-1])
        merged = proj[order[0]]
        for k in order[1:]:
            target = proj[k]
            merged = F.interpolate(merged, size=target.shape[-2:], mode='bilinear', align_corners=False) + target
        return self.refine(merged)

    def _prior_curves(self, params: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        start_y = params[..., 0].unsqueeze(-1)
        start_x = params[..., 1].unsqueeze(-1)
        theta = params[..., 2].unsqueeze(-1)
        ys = self.prior_ys.to(params.device, params.dtype).view(1, 1, self.num_points)
        base_x = start_x + ((ys - start_y) / torch.tan(theta * math.pi + 1e-4))
        x = (base_x + offsets).clamp(0.0, 1.0)
        y = ys.expand_as(x).clamp(0.0, 1.0)
        return torch.stack([x, y], dim=-1)

    def _gather_curve_features(self, feat: torch.Tensor, curves: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feat.shape
        lanes = curves.shape[1]
        x = curves[..., 0].clamp(0.0, 1.0)
        y = curves[..., 1].clamp(0.0, 1.0)
        ix = torch.round(x * float(max(1, w - 1))).long().clamp(0, w - 1)
        iy = torch.round(y * float(max(1, h - 1))).long().clamp(0, h - 1)
        flat_idx = (iy * w + ix).view(b, 1, lanes * self.num_points).expand(-1, c, -1)
        flat_feat = feat.flatten(2)
        samples = torch.gather(flat_feat, 2, flat_idx).view(b, c, lanes, self.num_points)
        samples = samples.permute(0, 2, 1, 3).contiguous().view(b * lanes, c, self.num_points)
        pooled = self.roi_context(samples).squeeze(-1).view(b, lanes, c)
        return pooled

    def _predict_from_lane_features(self, per_lane: torch.Tensor, selected_prior: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cls_logits = self.cls_head(per_lane).squeeze(-1)
        delta = self.param_head(per_lane)
        params = torch.sigmoid(torch.cat([delta[..., 0:2] + selected_prior[..., 0:2], delta[..., 2:3] + selected_prior[..., 2:3], delta[..., 3:4]], dim=-1))
        offsets = 0.05 * torch.tanh(self.offset_head(per_lane))
        coord_pred = self._prior_curves(params[..., :3], offsets)
        return cls_logits, params, offsets, coord_pred

    def forward(self, feats: List[torch.Tensor]) -> dict:
        if not isinstance(feats, (list, tuple)) or len(feats) == 0:
            raise RuntimeError('CurveLaneHead expects a non-empty list of feature maps')
        x = self._topdown(list(feats))
        b, _c, _h, _w = x.shape
        gap = F.adaptive_avg_pool2d(x, 1).flatten(1)
        prior = self.prior_embeddings.weight.clamp(0.0, 1.0)
        q = self.query_proj(prior).unsqueeze(0).expand(b, -1, -1)
        per_prior = q + gap.unsqueeze(1)
        select_logits = self.score_select(per_prior).squeeze(-1)
        if self.output_all_priors:
            per_lane = per_prior
            selected_prior = prior.unsqueeze(0).expand(b, -1, -1)
            selected_indices = torch.arange(self.num_priors, device=x.device).view(1, -1).expand(b, -1)
        else:
            topk = min(self.max_lanes, self.num_priors)
            _, idx = torch.topk(select_logits, k=topk, dim=1)
            gather_idx = idx.unsqueeze(-1).expand(-1, -1, per_prior.shape[-1])
            per_lane = torch.gather(per_prior, 1, gather_idx)
            prior_idx = idx.unsqueeze(-1).expand(-1, -1, 3)
            selected_prior = torch.gather(prior.unsqueeze(0).expand(b, -1, -1), 1, prior_idx)
            selected_indices = idx

        refine_debug = []
        cls_logits, params, offsets, coord_pred = self._predict_from_lane_features(per_lane, selected_prior)
        if self.use_roi_gather:
            for _ in range(self.roi_refine_layers):
                roi_feat = self._gather_curve_features(x, coord_pred.detach())
                per_lane = self.roi_norm(per_lane + self.roi_proj(roi_feat))
                cls_logits, params, offsets, coord_pred = self._predict_from_lane_features(per_lane, selected_prior)
                refine_debug.append(coord_pred.detach())

        lane_class_logits = self.lane_class_head(per_lane)
        out = {
            'cls_logits': cls_logits,
            'coord_pred': coord_pred,
            'lane_param': params,
            'lane_offsets': offsets,
            'lane_class_logits': lane_class_logits,
            'selected_prior': selected_prior,
            'selected_prior_indices': selected_indices,
            'prior_select_logits': select_logits,
        }
        if refine_debug:
            out['refine_stage_coords'] = refine_debug
        if self.mask_aux:
            mask = self.mask_decoder(x)
            out['mask_logit'] = F.interpolate(mask, size=self.mask_size, mode='bilinear', align_corners=False)
        return out


class _PerScaleROIBlock(nn.Module):
    """Per-scale 1-D conv along the sample-points axis of (B, C, P, S).

    Mirrors CLRNet ROIGather convs (kernel 9 along the per-prior sample axis)
    but laid out as Conv2d with kernel (1, 9), padding (0, 4) so the conv runs
    over the S=sample_points dimension while keeping the P=priors dimension
    unchanged.
    """

    def __init__(self, channels: int, mid_channels: int, sample_points: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, mid_channels, kernel_size=(1, 9), padding=(0, 4), bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.sample_points = int(sample_points)

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        return self.conv(samples)


class _CrossAttention(nn.Module):
    def __init__(self, channels: int, num_priors: int, resize: Tuple[int, int] = (10, 25)) -> None:
        super().__init__()
        self.channels = int(channels)
        self.num_priors = int(num_priors)
        self.resize = (int(resize[0]), int(resize[1]))
        self.f_key = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.f_value = nn.Conv2d(channels, channels, kernel_size=1)
        self.f_query = nn.Sequential(
            nn.Conv1d(num_priors, num_priors, kernel_size=1, groups=num_priors),
            nn.ReLU(inplace=True),
        )
        self.W = nn.Conv1d(num_priors, num_priors, kernel_size=1, groups=num_priors)
        nn.init.constant_(self.W.weight, 0)
        nn.init.constant_(self.W.bias, 0)

    def _flatten_resize(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, self.resize, mode='bilinear', align_corners=False).flatten(2)

    def forward(self, roi: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        bs = feat.size(0)
        query = self.f_query(roi)
        key = self._flatten_resize(self.f_key(feat))
        value = self._flatten_resize(self.f_value(feat)).permute(0, 2, 1)
        sim = torch.matmul(query, key) * (self.channels ** -0.5)
        sim = F.softmax(sim, dim=-1)
        ctx = torch.matmul(sim, value)
        ctx = self.W(ctx)
        return roi + F.dropout(ctx, p=0.1, training=self.training)


class CLRKDLaneHead(nn.Module):
    """CLRNet/CLRKDNet-style lane head with proper per-prior ROI gather.

    Why this exists separate from CurveLaneHead: in CurveLaneHead the per-prior
    feature is a shared global avg pool plus a 3-d prior embedding, so all 192
    priors see the same image-level vector and gradients differentiate priors
    only through the prior_embeddings. That is the cause of the lane_point_mae
    plateau seen in stage2_trend_summary.csv (Exp2E flat at 0.40 across 10
    epochs).

    This module follows external_repos/CLRNet/clrnet/models/utils/roi_gather.py
    line 33-136: each prior gets its own per-scale, per-point bilinear
    sampling along the prior's current curve, then per-scale 1-D conv, then
    multi-scale fusion, then cross-attention to the full feature map, then a
    refinement step that updates the curve and re-samples. Three refinement
    layers are returned so the loss can apply auxiliary supervision on each.
    """

    def __init__(
        self,
        in_channels: Sequence[int] = (128, 128, 128),
        embed_dim: int = 128,
        max_lanes: int = 10,
        num_points: int = 72,
        mask_size: Tuple[int, int] = (72, 128),
        mask_aux: bool = True,
        num_lane_classes: int = 7,
        num_priors: int = 192,
        sample_points: int = 36,
        roi_refine_layers: int = 3,
        roi_mid_channels: int = 48,
        cross_attn_resize: Tuple[int, int] = (10, 25),
        output_all_priors: bool = True,
        cls_uses_prior_embedding: bool = False,
        prior_embed_encoder_dim: int = 0,
        cls_separate_path: bool = False,
        dual_score: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = list(in_channels)
        self.embed_dim = int(embed_dim)
        self.max_lanes = int(max_lanes)
        self.num_points = int(num_points)
        self.mask_size = (int(mask_size[0]), int(mask_size[1]))
        self.mask_aux = bool(mask_aux)
        self.num_lane_classes = int(num_lane_classes)
        self.num_priors = int(num_priors)
        self.sample_points = int(sample_points)
        self.roi_refine_layers = max(1, int(roi_refine_layers))
        self.roi_mid_channels = int(roi_mid_channels)
        self.output_all_priors = True  # CLRKD path always trains all priors
        self.cls_uses_prior_embedding = bool(cls_uses_prior_embedding)
        self.prior_embed_encoder_dim = int(prior_embed_encoder_dim)
        self.cls_separate_path = bool(cls_separate_path)
        # Exp2MM: dual scoring head emits a parallel `iou_logits` (B, P) trained
        # on continuous LineIoU regression while `cls_logits` keeps the binary
        # matched_existence supervision. At inference, the decode score is
        # sigmoid(cls_logits) * sigmoid(iou_logits) -- cls says "is this prior a
        # match?" and iou says "if so, how good is the geometry?" -- which
        # decouples ranking from the matching-instability that has been
        # collapsing the binary cls across Exp2II/KK on the 192-anchor head.
        self.dual_score = bool(dual_score)

        self.lateral = nn.ModuleList([_conv_bn_act(c, embed_dim, k=1) for c in in_channels])
        self.refine = _conv_bn_act(embed_dim, embed_dim, k=3)

        self.prior_embeddings = nn.Embedding(self.num_priors, 3)
        self._init_prior_embeddings()

        self.scale_blocks = nn.ModuleList([
            _PerScaleROIBlock(embed_dim, roi_mid_channels, sample_points)
            for _ in range(len(in_channels))
        ])
        self.scale_fusion = nn.ModuleList()
        for _ in range(self.roi_refine_layers):
            self.scale_fusion.append(nn.Sequential(
                nn.Conv2d(roi_mid_channels * len(in_channels), embed_dim,
                          kernel_size=(1, 9), padding=(0, 4), bias=False),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU(inplace=True),
            ))

        self.fc = nn.Linear(embed_dim * sample_points, embed_dim)
        self.fc_norm = nn.LayerNorm(embed_dim)

        self.cross_attn = nn.ModuleList([
            _CrossAttention(embed_dim, num_priors, resize=cross_attn_resize)
            for _ in range(self.roi_refine_layers)
        ])

        # Optional parallel cls aggregator. Same module shapes as the geometry
        # aggregator above, but disjoint parameters: cls gradients update only
        # these modules + cls_head + prior_embed_encoder, not param/offset heads
        # or the geometry aggregator. The two pathways share per-scale ROI
        # samples (computed once per stage in forward) so each prior sees the
        # same image evidence along its curve from both branches.
        if self.cls_separate_path:
            self.scale_blocks_cls = nn.ModuleList([
                _PerScaleROIBlock(embed_dim, roi_mid_channels, sample_points)
                for _ in range(len(in_channels))
            ])
            self.scale_fusion_cls = nn.ModuleList()
            for _ in range(self.roi_refine_layers):
                self.scale_fusion_cls.append(nn.Sequential(
                    nn.Conv2d(roi_mid_channels * len(in_channels), embed_dim,
                              kernel_size=(1, 9), padding=(0, 4), bias=False),
                    nn.BatchNorm2d(embed_dim),
                    nn.ReLU(inplace=True),
                ))
            self.fc_cls = nn.Linear(embed_dim * sample_points, embed_dim)
            self.fc_norm_cls = nn.LayerNorm(embed_dim)
            self.cross_attn_cls = nn.ModuleList([
                _CrossAttention(embed_dim, num_priors, resize=cross_attn_resize)
                for _ in range(self.roi_refine_layers)
            ])
        else:
            self.scale_blocks_cls = None
            self.scale_fusion_cls = None
            self.fc_cls = None
            self.fc_norm_cls = None
            self.cross_attn_cls = None

        if self.cls_uses_prior_embedding:
            if self.prior_embed_encoder_dim > 0:
                self.prior_embed_encoder = nn.Sequential(
                    nn.Linear(3, self.prior_embed_encoder_dim),
                    nn.ReLU(inplace=True),
                )
                cls_extra_dim = self.prior_embed_encoder_dim
            else:
                self.prior_embed_encoder = nn.Identity()
                cls_extra_dim = 3
        else:
            self.prior_embed_encoder = None
            cls_extra_dim = 0
        cls_in_dim = embed_dim + cls_extra_dim
        self.cls_head = nn.Sequential(
            nn.Linear(cls_in_dim, embed_dim), nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 1),
        )
        # Exp2MM: parallel iou_head emits a continuous LineIoU score per prior.
        # Trained by FusionLaneLoss when w_iou_aux > 0 with the same
        # _compute_lineiou_target used by lineiou_regression cls_target_type.
        # Reads from the geometry feature `per_lane_geom` (not the cls feature)
        # because the IoU score is a property of the predicted curve geometry,
        # not the matching outcome.
        if self.dual_score:
            self.iou_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim), nn.ReLU(inplace=True),
                nn.Linear(embed_dim, 1),
            )
        else:
            self.iou_head = None
        self.param_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 4),
        )
        self.offset_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(inplace=True),
            nn.Linear(embed_dim, num_points),
        )
        self.lane_class_head = nn.Linear(embed_dim, num_lane_classes)

        if self.mask_aux:
            self.mask_decoder = nn.Sequential(
                _conv_bn_act(embed_dim, max(16, embed_dim // 2), k=3),
                nn.Conv2d(max(16, embed_dim // 2), 1, kernel_size=1),
            )

        self.register_buffer('prior_ys', torch.linspace(1.0, 0.0, steps=self.num_points, dtype=torch.float32))
        self.register_buffer('sample_ys', torch.linspace(1.0, 0.0, steps=self.sample_points, dtype=torch.float32))

    def _init_prior_embeddings(self) -> None:
        # Same left/bottom/right CLRKDNet-style initialization as CurveLaneHead.
        bottom_priors_nums = self.num_priors * 3 // 4
        left_priors_nums = self.num_priors // 8
        right_start = left_priors_nums + bottom_priors_nums
        left_denom = max(1, left_priors_nums // 2 - 1)
        strip_size = 0.5 / left_denom
        bottom_strip_size = 1.0 / max(1, bottom_priors_nums // 4 + 1)
        with torch.no_grad():
            for i in range(left_priors_nums):
                self.prior_embeddings.weight[i, 0] = (i // 2) * strip_size
                self.prior_embeddings.weight[i, 1] = 0.0
                self.prior_embeddings.weight[i, 2] = 0.16 if i % 2 == 0 else 0.32
            for i in range(left_priors_nums, right_start):
                self.prior_embeddings.weight[i, 0] = 0.0
                self.prior_embeddings.weight[i, 1] = ((i - left_priors_nums) // 4 + 1) * bottom_strip_size
                self.prior_embeddings.weight[i, 2] = 0.2 * ((i - left_priors_nums) % 4 + 1)
            for i in range(right_start, self.num_priors):
                self.prior_embeddings.weight[i, 0] = ((i - right_start) // 2) * strip_size
                self.prior_embeddings.weight[i, 1] = 1.0
                self.prior_embeddings.weight[i, 2] = 0.68 if i % 2 == 0 else 0.84

    def _project_features(self, feats: List[torch.Tensor]) -> List[torch.Tensor]:
        return [layer(f) for layer, f in zip(self.lateral, feats)]

    def _topdown(self, projected: List[torch.Tensor]) -> torch.Tensor:
        order = sorted(range(len(projected)), key=lambda i: projected[i].shape[-2] * projected[i].shape[-1])
        merged = projected[order[0]]
        for k in order[1:]:
            target = projected[k]
            merged = F.interpolate(merged, size=target.shape[-2:], mode='bilinear', align_corners=False) + target
        return self.refine(merged)

    def _prior_curves_full(self, params: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        start_y = params[..., 0].unsqueeze(-1)
        start_x = params[..., 1].unsqueeze(-1)
        theta = params[..., 2].unsqueeze(-1)
        ys = self.prior_ys.to(params.device, params.dtype).view(1, 1, self.num_points)
        base_x = start_x + ((ys - start_y) / torch.tan(theta * math.pi + 1e-4))
        x = (base_x + offsets).clamp(0.0, 1.0)
        y = ys.expand_as(x).clamp(0.0, 1.0)
        return torch.stack([x, y], dim=-1)

    def _sample_curves(self, params: torch.Tensor) -> torch.Tensor:
        # Sample sample_points along each prior's current curve; returns (B, P, S, 2).
        start_y = params[..., 0].unsqueeze(-1)
        start_x = params[..., 1].unsqueeze(-1)
        theta = params[..., 2].unsqueeze(-1)
        ys = self.sample_ys.to(params.device, params.dtype).view(1, 1, self.sample_points)
        base_x = start_x + ((ys - start_y) / torch.tan(theta * math.pi + 1e-4))
        x = base_x.clamp(0.0, 1.0)
        y = ys.expand_as(x).clamp(0.0, 1.0)
        return torch.stack([x, y], dim=-1)

    def _grid_sample(self, feat: torch.Tensor, sample_xy: torch.Tensor) -> torch.Tensor:
        # feat: (B, C, H, W). sample_xy: (B, P, S, 2) in [0, 1].
        b, c, _h, _w = feat.shape
        p, s = sample_xy.shape[1], sample_xy.shape[2]
        # grid_sample expects normalized coords in [-1, 1] with order (x, y).
        grid = sample_xy * 2.0 - 1.0  # (B, P, S, 2)
        # Reshape to (B, P, S, 2) -> H'=P, W'=S grid for grid_sample.
        sampled = F.grid_sample(feat, grid, mode='bilinear', padding_mode='border', align_corners=False)
        # sampled: (B, C, P, S)
        return sampled.view(b, c, p, s)

    def _predict(
        self,
        per_lane_geom: torch.Tensor,
        prior: torch.Tensor,
        per_lane_cls: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        cls_feat = per_lane_cls if per_lane_cls is not None else per_lane_geom
        if self.cls_uses_prior_embedding:
            prior_feat = self.prior_embed_encoder(prior) if self.prior_embed_encoder is not None else prior
            cls_input = torch.cat([cls_feat, prior_feat], dim=-1)
        else:
            cls_input = cls_feat
        cls_logits = self.cls_head(cls_input).squeeze(-1)
        if self.iou_head is not None:
            iou_logits = self.iou_head(per_lane_geom).squeeze(-1)
        else:
            iou_logits = None
        delta = self.param_head(per_lane_geom)
        params = torch.sigmoid(torch.cat([
            delta[..., 0:2] + prior[..., 0:2],
            delta[..., 2:3] + prior[..., 2:3],
            delta[..., 3:4],
        ], dim=-1))
        offsets = 0.05 * torch.tanh(self.offset_head(per_lane_geom))
        coord_pred = self._prior_curves_full(params[..., :3], offsets)
        return cls_logits, params, offsets, coord_pred, iou_logits

    def forward(self, feats: List[torch.Tensor]) -> Dict[str, object]:
        if not isinstance(feats, (list, tuple)) or len(feats) == 0:
            raise RuntimeError('CLRKDLaneHead expects a non-empty list of feature maps')
        projected = self._project_features(list(feats))
        x = self._topdown(projected)
        b = x.shape[0]

        prior = self.prior_embeddings.weight.clamp(0.0, 1.0)              # (P, 3)
        prior_b = prior.unsqueeze(0).expand(b, -1, -1).contiguous()       # (B, P, 3)
        # Stage 0 starts from each prior's own initial geometry, with zero offset.
        params = torch.cat([prior_b, prior_b.new_zeros(b, self.num_priors, 1)], dim=-1)

        stage_outputs: List[Dict[str, torch.Tensor]] = []
        per_lane: Optional[torch.Tensor] = None

        for stage in range(self.roi_refine_layers):
            sample_xy = self._sample_curves(params[..., :3])              # (B, P, S, 2)
            # Sample features once per scale; the geometry and (optional) cls
            # aggregator branches share these per-scale ROI tensors so each
            # prior sees the same image evidence along its curve, while the
            # branches' aggregator parameters stay disjoint.
            roi_per_scale: List[torch.Tensor] = []
            for proj_feat in projected:
                roi_per_scale.append(self._grid_sample(proj_feat, sample_xy))  # (B, C, P, S)

            scale_feats: List[torch.Tensor] = [
                block(roi) for block, roi in zip(self.scale_blocks, roi_per_scale)
            ]                                                              # (B, mid, P, S)
            cat_feat = torch.cat(scale_feats, dim=1)                       # (B, mid*K, P, S)
            fused = self.scale_fusion[stage](cat_feat)                     # (B, embed, P, S)
            roi_vec_geom = fused.permute(0, 2, 1, 3).contiguous().view(
                b, self.num_priors, self.embed_dim * self.sample_points)
            roi_vec_geom = F.relu(self.fc_norm(self.fc(roi_vec_geom)))     # (B, P, embed)
            roi_vec_geom = self.cross_attn[stage](roi_vec_geom, x)         # (B, P, embed)

            if self.cls_separate_path:
                scale_feats_cls: List[torch.Tensor] = [
                    block(roi) for block, roi in zip(self.scale_blocks_cls, roi_per_scale)
                ]
                cat_feat_cls = torch.cat(scale_feats_cls, dim=1)
                fused_cls = self.scale_fusion_cls[stage](cat_feat_cls)
                roi_vec_cls = fused_cls.permute(0, 2, 1, 3).contiguous().view(
                    b, self.num_priors, self.embed_dim * self.sample_points)
                roi_vec_cls = F.relu(self.fc_norm_cls(self.fc_cls(roi_vec_cls)))
                roi_vec_cls = self.cross_attn_cls[stage](roi_vec_cls, x)
            else:
                roi_vec_cls = None

            cls_logits, params, offsets, coord_pred, iou_logits = self._predict(
                roi_vec_geom, prior_b, per_lane_cls=roi_vec_cls,
            )
            # `per_lane` is exposed downstream (lane_class_head, mask) -- keep it
            # tied to the geometry pathway so existing consumers see the same
            # representation as before when cls_separate_path is False.
            per_lane = roi_vec_geom
            stage_dict: Dict[str, torch.Tensor] = {
                'cls_logits': cls_logits,
                'lane_param': params,
                'lane_offsets': offsets,
                'coord_pred': coord_pred,
            }
            if iou_logits is not None:
                stage_dict['iou_logits'] = iou_logits
            stage_outputs.append(stage_dict)

        assert per_lane is not None
        lane_class_logits = self.lane_class_head(per_lane)
        final = stage_outputs[-1]
        out: Dict[str, object] = {
            'cls_logits': final['cls_logits'],
            'coord_pred': final['coord_pred'],
            'lane_param': final['lane_param'],
            'lane_offsets': final['lane_offsets'],
            'lane_class_logits': lane_class_logits,
            'selected_prior': prior_b,
            'selected_prior_indices': torch.arange(self.num_priors, device=x.device).view(1, -1).expand(b, -1),
            'prior_select_logits': final['cls_logits'],
            'aux_stage_outputs': stage_outputs[:-1],
            # Per-prior feature embedding (B, P, embed_dim) and the merged
            # multi-scale spatial feature (B, embed_dim, H, W) -- exposed so
            # HybridPriorQueryHead (Exp2Q) can attend over them in stage 2.
            'per_prior_features': per_lane,
            'spatial_features': x,
        }
        if 'iou_logits' in final:
            # Exp2MM: parallel score for dual cls x IoU decoding.
            out['iou_logits'] = final['iou_logits']
        if self.mask_aux:
            mask = self.mask_decoder(x)
            out['mask_logit'] = F.interpolate(mask, size=self.mask_size, mode='bilinear', align_corners=False)
        return out


class _MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 3) -> None:
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        layers: List[nn.Module] = []
        for i in range(num_layers):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < num_layers - 1:
                layers.append(nn.ReLU(inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class LaneQueryHead(nn.Module):
    """DETR-style lane prediction head with K learned queries.

    Why this exists separate from CLRKDLaneHead: across Exp2G-Exp2O the
    prior-based head's cls collapsed to a degenerate equilibrium where all
    192 priors output the same sigmoid score (~0.13). The Exp2O oracle
    diagnostic showed `oracle_f1 = 0.30` (perfect-ranking ceiling) while
    `decoded_f1 = 0.01` (cls-ranked) -- a 30x gap purely from cls failure.
    The bottleneck is the prior-based design itself, not any specific loss.

    This head abandons the 192-prior + dynamic-k design entirely. K=12
    learned queries are passed through a 3-layer transformer decoder that
    cross-attends to flattened multi-scale features. Each query outputs
    cls + curve params + row offsets. Hungarian matching to GT (1-to-1)
    handles supervision -- no class imbalance, no per-prior ranking task,
    no batch-to-batch matching instability.

    Output dict keys are identical to CLRKDLaneHead with P=K queries, so
    FusionLaneLoss + LaneF1DecodedMetric work without modification.
    Reference: DETR (Carion 2020), RMT-PPAD lane formulation.
    """

    def __init__(
        self,
        in_channels: Sequence[int] = (128, 128, 128),
        embed_dim: int = 128,
        max_lanes: int = 12,
        num_points: int = 72,
        mask_size: Tuple[int, int] = (72, 128),
        mask_aux: bool = False,
        num_lane_classes: int = 7,
        num_queries: int = 12,
        num_decoder_layers: int = 3,
        num_heads: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_channels = list(in_channels)
        self.embed_dim = int(embed_dim)
        self.max_lanes = int(max_lanes)
        self.num_points = int(num_points)
        self.mask_size = (int(mask_size[0]), int(mask_size[1]))
        self.mask_aux = bool(mask_aux)
        self.num_lane_classes = int(num_lane_classes)
        self.num_queries = int(num_queries)
        # `num_priors` accessor for back-compat with FusionLaneLoss + checkpoint
        # save paths that read this attribute.
        self.num_priors = int(num_queries)

        self.lateral = nn.ModuleList([_conv_bn_act(c, embed_dim, k=1) for c in in_channels])

        # Learned scale embedding so the decoder can distinguish features that
        # came from different feature levels after flattening + concatenation.
        self.scale_embed = nn.Embedding(len(in_channels), embed_dim)

        # Learned query content + positional embeddings (DETR convention).
        self.query_embed = nn.Embedding(num_queries, embed_dim)
        self.query_pos = nn.Embedding(num_queries, embed_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.cls_head = nn.Linear(embed_dim, 1)
        self.param_head = _MLP(embed_dim, embed_dim, 4, num_layers=3)
        self.offset_head = _MLP(embed_dim, embed_dim, num_points, num_layers=3)
        self.lane_class_head = nn.Linear(embed_dim, num_lane_classes)

        self.register_buffer('prior_ys', torch.linspace(1.0, 0.0, steps=self.num_points, dtype=torch.float32))

        if self.mask_aux:
            self.mask_decoder = nn.Sequential(
                _conv_bn_act(embed_dim, max(16, embed_dim // 2), k=3),
                nn.Conv2d(max(16, embed_dim // 2), 1, kernel_size=1),
            )

    def _flatten_with_scale_emb(self, feats: List[torch.Tensor]) -> torch.Tensor:
        # Project each scale and flatten to (B, sum_HW, C). Add a per-scale
        # learned embedding so the decoder can attend by scale.
        chunks: List[torch.Tensor] = []
        for i, f in enumerate(feats):
            f_proj = self.lateral[i](f)
            f_flat = f_proj.flatten(2).transpose(1, 2)        # (B, H*W, C)
            scale_e = self.scale_embed.weight[i].view(1, 1, -1).expand_as(f_flat)
            chunks.append(f_flat + scale_e)
        return torch.cat(chunks, dim=1)                        # (B, sum_HW, C)

    def _curves_from_params(self, params: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        # params: (B, K, 4) -> start_y, start_x, theta, length (all in [0, 1]).
        # offsets: (B, K, num_points) row offsets in normalized image-x.
        start_y = params[..., 0:1]
        start_x = params[..., 1:2]
        theta = params[..., 2:3]
        ys = self.prior_ys.to(params.device, params.dtype).view(1, 1, self.num_points)
        base_x = start_x + ((ys - start_y) / torch.tan(theta * math.pi + 1e-4))
        x = (base_x + offsets).clamp(0.0, 1.0)
        y = ys.expand_as(x).clamp(0.0, 1.0)
        return torch.stack([x, y], dim=-1)

    def forward(self, feats: List[torch.Tensor]) -> Dict[str, object]:
        if not isinstance(feats, (list, tuple)) or len(feats) == 0:
            raise RuntimeError('LaneQueryHead expects a non-empty list of feature maps')
        memory = self._flatten_with_scale_emb(list(feats))     # (B, sum_HW, C)
        b = memory.shape[0]

        tgt = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1).contiguous()
        pos = self.query_pos.weight.unsqueeze(0).expand(b, -1, -1).contiguous()
        decoded = self.decoder(tgt + pos, memory)              # (B, K, C)

        cls_logits = self.cls_head(decoded).squeeze(-1)        # (B, K)
        params = self.param_head(decoded).sigmoid()            # (B, K, 4)
        offsets = 0.05 * torch.tanh(self.offset_head(decoded))  # (B, K, num_points)
        coord_pred = self._curves_from_params(params, offsets)  # (B, K, N, 2)
        lane_class_logits = self.lane_class_head(decoded)      # (B, K, num_lane_classes)

        out: Dict[str, object] = {
            'cls_logits': cls_logits,
            'coord_pred': coord_pred,
            'lane_param': params,
            'lane_offsets': offsets,
            'lane_class_logits': lane_class_logits,
            # Compatibility shims so downstream code (matched-target machinery,
            # eval) keeps working unchanged.
            'selected_prior': params[..., :3].contiguous(),
            'selected_prior_indices': torch.arange(self.num_queries, device=memory.device).view(1, -1).expand(b, -1),
            'prior_select_logits': cls_logits,
        }
        if self.mask_aux:
            largest = max(feats, key=lambda f: f.shape[-2] * f.shape[-1])
            mask = self.mask_decoder(self.lateral[0](largest))
            out['mask_logit'] = F.interpolate(mask, size=self.mask_size, mode='bilinear', align_corners=False)
        return out


# ---------------------------------------------------------------------------
# Exp2Q -- Hybrid prior-generator + query-refiner (Sparse R-CNN / Mask2Former
# style two-stage pattern adapted to lanes).
#
# Stage 1: CLRKDLaneHead produces 192 prior-based curves with proven matched
# IoU around 0.42 (Exp2N champion). We expose its per-prior features and
# spatial features.
# Stage 2: K=12 learned queries cross-attend over the 192 prior features
# (and optionally the spatial features) and output K refined predictions.
# Hungarian matches the K outputs to GT. The cls task is owned entirely by
# stage 2 -- stage 1's cls is unused at inference and only contributes to
# stage 1's geometry losses.
# ---------------------------------------------------------------------------
class HybridPriorQueryHead(nn.Module):
    """Stage-1 CLRKDLaneHead + stage-2 query refiner. Output is K queries.

    Why this exists: Exp2N (priors) had matched_iou=0.42 but cls collapsed
    to all-priors-cluster-at-0.13. Exp2P (queries) had val_lane_f1=0.65 but
    matched_iou=0.13 (geometry collapsed because queries learn slowly from
    random init). This head puts the proven prior generator under a query
    decoder so we get the geometry of the former and the ranking of the
    latter. The two-stage cascade pattern of Sparse R-CNN (Sun 2021) and
    Mask2Former (Cheng 2022).
    """

    def __init__(
        self,
        in_channels: Sequence[int] = (128, 128, 128),
        embed_dim: int = 128,
        max_lanes: int = 12,
        num_points: int = 72,
        mask_size: Tuple[int, int] = (72, 128),
        mask_aux: bool = False,
        num_lane_classes: int = 7,
        # Stage-1 CLRKDLaneHead parameters.
        num_priors: int = 192,
        sample_points: int = 36,
        roi_refine_layers: int = 3,
        roi_mid_channels: int = 48,
        cross_attn_resize: Tuple[int, int] = (10, 25),
        # Stage-2 query refiner parameters.
        num_queries: int = 12,
        refiner_layers: int = 2,
        refiner_heads: int = 8,
        refiner_ff_dim: int = 512,
        refiner_dropout: float = 0.1,
        # Whether to also let queries attend to spatial features (vs only
        # the per-prior pooled features). True is more expressive but slower.
        refiner_attn_to_spatial: bool = True,
    ) -> None:
        super().__init__()
        self.num_priors = int(num_priors)
        self.num_queries = int(num_queries)
        self.embed_dim = int(embed_dim)
        self.max_lanes = int(max_lanes)
        self.num_points = int(num_points)
        self.mask_size = (int(mask_size[0]), int(mask_size[1]))
        self.mask_aux = bool(mask_aux)
        self.num_lane_classes = int(num_lane_classes)
        self.refiner_attn_to_spatial = bool(refiner_attn_to_spatial)

        self.stage1 = CLRKDLaneHead(
            in_channels=in_channels,
            embed_dim=embed_dim,
            max_lanes=num_priors,           # stage1's max_lanes is the prior count
            num_points=num_points,
            mask_size=mask_size,
            mask_aux=False,                  # stage 1 does not produce the aux mask in hybrid
            num_lane_classes=num_lane_classes,
            num_priors=num_priors,
            sample_points=sample_points,
            roi_refine_layers=roi_refine_layers,
            roi_mid_channels=roi_mid_channels,
            cross_attn_resize=cross_attn_resize,
            output_all_priors=True,
            cls_uses_prior_embedding=False,
            prior_embed_encoder_dim=0,
            cls_separate_path=False,
        )

        # Stage 2: K learnable queries. Two transformer decoder layers cross-
        # attend over the 192 prior features.
        self.query_embed = nn.Embedding(num_queries, embed_dim)
        self.query_pos = nn.Embedding(num_queries, embed_dim)

        prior_decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=refiner_heads,
            dim_feedforward=refiner_ff_dim,
            dropout=refiner_dropout,
            batch_first=True,
        )
        self.prior_decoder = nn.TransformerDecoder(prior_decoder_layer, num_layers=refiner_layers)

        if refiner_attn_to_spatial:
            spatial_decoder_layer = nn.TransformerDecoderLayer(
                d_model=embed_dim,
                nhead=refiner_heads,
                dim_feedforward=refiner_ff_dim,
                dropout=refiner_dropout,
                batch_first=True,
            )
            self.spatial_decoder = nn.TransformerDecoder(spatial_decoder_layer, num_layers=1)
        else:
            self.spatial_decoder = None

        self.cls_head = nn.Linear(embed_dim, 1)
        self.param_head = _MLP(embed_dim, embed_dim, 4, num_layers=3)
        self.offset_head = _MLP(embed_dim, embed_dim, num_points, num_layers=3)
        self.lane_class_head = nn.Linear(embed_dim, num_lane_classes)

        self.register_buffer('prior_ys', torch.linspace(1.0, 0.0, steps=self.num_points, dtype=torch.float32))

        if self.mask_aux:
            self.mask_decoder = nn.Sequential(
                _conv_bn_act(embed_dim, max(16, embed_dim // 2), k=3),
                nn.Conv2d(max(16, embed_dim // 2), 1, kernel_size=1),
            )

    def _curves_from_params(self, params: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        start_y = params[..., 0:1]
        start_x = params[..., 1:2]
        theta = params[..., 2:3]
        ys = self.prior_ys.to(params.device, params.dtype).view(1, 1, self.num_points)
        base_x = start_x + ((ys - start_y) / torch.tan(theta * math.pi + 1e-4))
        x = (base_x + offsets).clamp(0.0, 1.0)
        y = ys.expand_as(x).clamp(0.0, 1.0)
        return torch.stack([x, y], dim=-1)

    def forward(self, feats: List[torch.Tensor]) -> Dict[str, object]:
        # Stage 1: prior-based head produces per-prior features + curves.
        s1 = self.stage1(feats)
        prior_features = s1['per_prior_features']      # (B, P, embed)
        spatial_features = s1['spatial_features']      # (B, embed, H, W)
        b = prior_features.shape[0]

        # Stage 2: K queries cross-attend over the 192 prior features.
        tgt = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1).contiguous()
        pos = self.query_pos.weight.unsqueeze(0).expand(b, -1, -1).contiguous()
        decoded = self.prior_decoder(tgt + pos, prior_features)         # (B, K, embed)

        if self.spatial_decoder is not None:
            sf = spatial_features.flatten(2).transpose(1, 2)            # (B, H*W, embed)
            decoded = self.spatial_decoder(decoded, sf)

        cls_logits = self.cls_head(decoded).squeeze(-1)                  # (B, K)
        params = self.param_head(decoded).sigmoid()                      # (B, K, 4)
        offsets = 0.05 * torch.tanh(self.offset_head(decoded))           # (B, K, num_points)
        coord_pred = self._curves_from_params(params, offsets)           # (B, K, N, 2)
        lane_class_logits = self.lane_class_head(decoded)                # (B, K, num_lane_classes)

        out: Dict[str, object] = {
            'cls_logits': cls_logits,
            'coord_pred': coord_pred,
            'lane_param': params,
            'lane_offsets': offsets,
            'lane_class_logits': lane_class_logits,
            'selected_prior': params[..., :3].contiguous(),
            'selected_prior_indices': torch.arange(self.num_queries, device=tgt.device).view(1, -1).expand(b, -1),
            'prior_select_logits': cls_logits,
            # Stage 1 outputs flow through under namespaced keys so the
            # geometry losses can still supervise stage 1's prior curves
            # for representation learning. The FusionLaneLoss reads
            # cls_logits / coord_pred / etc. for the K-output supervision;
            # these stage-1 keys are exposed for diagnostic eval (oracle on
            # the prior curves still works) and for an optional auxiliary
            # geometry loss applied to stage 1.
            'stage1_cls_logits': s1['cls_logits'],
            'stage1_coord_pred': s1['coord_pred'],
            'stage1_lane_param': s1['lane_param'],
            'stage1_lane_offsets': s1['lane_offsets'],
        }
        if self.mask_aux:
            largest = max(feats, key=lambda f: f.shape[-2] * f.shape[-1])
            mask = self.mask_decoder(spatial_features)
            out['mask_logit'] = F.interpolate(mask, size=self.mask_size, mode='bilinear', align_corners=False)
        return out


# ---------------------------------------------------------------------------
# Exp2R -- LaneQueryHead with anchor-conditioned queries (DAB-DETR) +
# denoising queries (DN-DETR). Addresses Exp2P's geometry collapse: queries
# from random init don't specialize in 10 epochs.
#
# Anchor conditioning: each query has a learnable (start_y, start_x, theta)
# anchor whose sinusoidal embedding is added to the query at every decoder
# layer. This gives queries explicit positional bias so they specialize
# spatially much faster.
#
# Denoising queries: during training, take GT lanes, perturb their (start_y,
# start_x, theta) by Gaussian noise, pass them as auxiliary queries that
# get supervised to recover the un-perturbed GT. Cuts DETR convergence
# from 50+ epochs to ~12 (DN-DETR; Li 2022).
# ---------------------------------------------------------------------------
def _sinusoidal_pos_embed(coords: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal position embedding for a (..., D) coordinate tensor.

    coords: (..., D) values in [0, 1].
    Returns: (..., D * dim) embedding tensor with sin/cos at log-spaced
    frequencies, suitable for use as positional encoding.
    """
    *prefix_shape, d = coords.shape
    n_freq = dim // 2
    device = coords.device
    freqs = torch.arange(n_freq, device=device, dtype=coords.dtype)
    freqs = (2.0 * math.pi) * (2.0 ** freqs)                            # (n_freq,)
    angles = coords.unsqueeze(-1) * freqs.view(1, n_freq)               # (..., D, n_freq)
    sin_part = torch.sin(angles)
    cos_part = torch.cos(angles)
    embed = torch.cat([sin_part, cos_part], dim=-1)                     # (..., D, dim)
    return embed.flatten(-2)                                             # (..., D*dim)


class LaneQueryHeadAnchorDN(nn.Module):
    """LaneQueryHead with DAB-style anchor positional encoding and DN-DETR-
    style denoising queries.

    Output shape and key set are identical to LaneQueryHead so downstream
    machinery (FusionLaneLoss, LaneF1DecodedMetric) is unchanged. Denoising
    queries are returned under 'dn_*' keys when training; the train script
    can apply an auxiliary loss using them.
    """

    def __init__(
        self,
        in_channels: Sequence[int] = (128, 128, 128),
        embed_dim: int = 128,
        max_lanes: int = 12,
        num_points: int = 72,
        mask_size: Tuple[int, int] = (72, 128),
        mask_aux: bool = False,
        num_lane_classes: int = 7,
        num_queries: int = 12,
        num_decoder_layers: int = 3,
        num_heads: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        # DN-DETR controls.
        dn_num_groups: int = 4,         # # noised groups per GT
        dn_noise_std_xy: float = 0.05,  # noise std on start_x / start_y
        dn_noise_std_theta: float = 0.04,  # noise std on theta
        dn_label_noise_prob: float = 0.0,  # not used (binary cls)
    ) -> None:
        super().__init__()
        self.num_queries = int(num_queries)
        self.num_priors = int(num_queries)
        self.embed_dim = int(embed_dim)
        self.max_lanes = int(max_lanes)
        self.num_points = int(num_points)
        self.mask_size = (int(mask_size[0]), int(mask_size[1]))
        self.mask_aux = bool(mask_aux)
        self.num_lane_classes = int(num_lane_classes)

        self.lateral = nn.ModuleList([_conv_bn_act(c, embed_dim, k=1) for c in in_channels])
        self.scale_embed = nn.Embedding(len(in_channels), embed_dim)

        # Content embedding (DAB-DETR's "content query") and learnable anchor
        # parameters (DAB-DETR's "positional anchor"). Anchors are 3-d
        # sigmoid'd (start_y, start_x, theta) to match curve params.
        self.content_embed = nn.Embedding(num_queries, embed_dim)
        self.anchor_params = nn.Parameter(torch.zeros(num_queries, 3))
        nn.init.uniform_(self.anchor_params, -0.5, 0.5)  # logit space; sigmoid will spread to (0.38, 0.62)
        # Spread anchors more widely after init to encourage diverse positions.
        with torch.no_grad():
            self.anchor_params[:, 0].uniform_(-2.0, 2.0)  # start_y
            self.anchor_params[:, 1].uniform_(-2.0, 2.0)  # start_x
            self.anchor_params[:, 2].uniform_(-1.5, 1.5)  # theta

        # Sinusoidal embedding dim per coordinate.
        self.pos_dim_per_coord = embed_dim // 3
        # Project the 3 * pos_dim_per_coord embedding to embed_dim.
        self.pos_proj = nn.Linear(self.pos_dim_per_coord * 3, embed_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.cls_head = nn.Linear(embed_dim, 1)
        self.param_head = _MLP(embed_dim, embed_dim, 4, num_layers=3)
        self.offset_head = _MLP(embed_dim, embed_dim, num_points, num_layers=3)
        self.lane_class_head = nn.Linear(embed_dim, num_lane_classes)

        self.register_buffer('prior_ys', torch.linspace(1.0, 0.0, steps=self.num_points, dtype=torch.float32))

        # Denoising config -- training-only.
        self.dn_num_groups = int(dn_num_groups)
        self.dn_noise_std_xy = float(dn_noise_std_xy)
        self.dn_noise_std_theta = float(dn_noise_std_theta)

        if self.mask_aux:
            self.mask_decoder = nn.Sequential(
                _conv_bn_act(embed_dim, max(16, embed_dim // 2), k=3),
                nn.Conv2d(max(16, embed_dim // 2), 1, kernel_size=1),
            )

    def _flatten_with_scale_emb(self, feats: List[torch.Tensor]) -> torch.Tensor:
        chunks: List[torch.Tensor] = []
        for i, f in enumerate(feats):
            f_proj = self.lateral[i](f)
            f_flat = f_proj.flatten(2).transpose(1, 2)
            scale_e = self.scale_embed.weight[i].view(1, 1, -1).expand_as(f_flat)
            chunks.append(f_flat + scale_e)
        return torch.cat(chunks, dim=1)

    def _anchor_pos_embed(self, anchors_sigmoid: torch.Tensor) -> torch.Tensor:
        # anchors_sigmoid: (B, K, 3) in [0, 1].
        emb = _sinusoidal_pos_embed(anchors_sigmoid, self.pos_dim_per_coord)  # (B, K, 3 * pos_dim_per_coord)
        return self.pos_proj(emb)                                              # (B, K, embed_dim)

    def _curves_from_params(self, params: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        start_y = params[..., 0:1]
        start_x = params[..., 1:2]
        theta = params[..., 2:3]
        ys = self.prior_ys.to(params.device, params.dtype).view(1, 1, self.num_points)
        base_x = start_x + ((ys - start_y) / torch.tan(theta * math.pi + 1e-4))
        x = (base_x + offsets).clamp(0.0, 1.0)
        y = ys.expand_as(x).clamp(0.0, 1.0)
        return torch.stack([x, y], dim=-1)

    def _build_dn_queries(
        self,
        memory: torch.Tensor,
        targets: Optional[Dict[str, torch.Tensor]],
    ) -> Optional[Dict[str, torch.Tensor]]:
        if targets is None or self.dn_num_groups <= 0 or not self.training:
            return None
        existence = targets.get('existence')
        points = targets.get('points')
        visibility = targets.get('visibility')
        if existence is None or points is None or visibility is None:
            return None
        b = memory.shape[0]
        device = memory.device

        # Collect a fixed-size batched-ragged buffer of noised GT queries.
        # We pad each image's noised queries to the max-noised length in the
        # batch so the decoder can run vectorized.
        max_per_image = 0
        per_image_anchors: List[torch.Tensor] = []
        per_image_targets: List[Dict[str, torch.Tensor]] = []
        for bi in range(b):
            gt_idx = torch.nonzero(existence[bi] > 0.5, as_tuple=False).flatten()
            if gt_idx.numel() == 0:
                per_image_anchors.append(memory.new_zeros((0, 3)))
                per_image_targets.append({'gt_anchor': memory.new_zeros((0, 3))})
                continue
            gt_pts = points[bi, gt_idx]                 # (Lg, N, 2)
            gt_vis = visibility[bi, gt_idx]              # (Lg, N)
            # Compute (start_y, start_x, theta) target for each GT lane.
            anchors_gt = self._gt_to_anchor(gt_pts, gt_vis)  # (Lg, 3) in [0, 1]
            # Replicate per group with noise.
            ng = self.dn_num_groups
            anchors_rep = anchors_gt.unsqueeze(0).expand(ng, -1, -1).reshape(-1, 3)  # (ng*Lg, 3)
            noise = anchors_rep.new_zeros(anchors_rep.shape)
            noise[:, 0] = torch.randn_like(noise[:, 0]) * self.dn_noise_std_xy
            noise[:, 1] = torch.randn_like(noise[:, 1]) * self.dn_noise_std_xy
            noise[:, 2] = torch.randn_like(noise[:, 2]) * self.dn_noise_std_theta
            anchors_noised = (anchors_rep + noise).clamp(0.0, 1.0)
            per_image_anchors.append(anchors_noised)
            per_image_targets.append({'gt_anchor': anchors_rep})        # (ng*Lg, 3)
            max_per_image = max(max_per_image, anchors_noised.shape[0])

        if max_per_image == 0:
            return None

        # Pad to a batched tensor.
        anchor_batch = memory.new_zeros((b, max_per_image, 3))
        anchor_target = memory.new_zeros((b, max_per_image, 3))
        valid_mask = torch.zeros((b, max_per_image), dtype=torch.bool, device=device)
        for bi in range(b):
            n = per_image_anchors[bi].shape[0]
            if n == 0:
                continue
            anchor_batch[bi, :n] = per_image_anchors[bi]
            anchor_target[bi, :n] = per_image_targets[bi]['gt_anchor']
            valid_mask[bi, :n] = True
        return {
            'anchors': anchor_batch,
            'targets': anchor_target,
            'valid_mask': valid_mask,
        }

    def _gt_to_anchor(self, points: torch.Tensor, visibility: torch.Tensor) -> torch.Tensor:
        # points: (Lg, N, 2), visibility: (Lg, N).
        # Anchor = (start_y, start_x, theta) of the GT lane in [0, 1].
        Lg = points.shape[0]
        out = points.new_zeros((Lg, 3))
        for li in range(Lg):
            idx = torch.nonzero(visibility[li] > 0.5, as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            first = idx[0]
            last = idx[-1]
            p0 = points[li, first]
            p1 = points[li, last]
            dy = (p1[1] - p0[1]).clamp(min=-1.0, max=1.0)
            dx = (p1[0] - p0[0]).clamp(min=-1.0, max=1.0)
            theta = torch.atan2(dy.abs() + 1e-4, dx.abs() + 1e-4) / math.pi
            out[li, 0] = p0[1].clamp(0.0, 1.0)        # start_y
            out[li, 1] = p0[0].clamp(0.0, 1.0)        # start_x
            out[li, 2] = theta.clamp(0.0, 1.0)
        return out

    def forward(self, feats: List[torch.Tensor], targets: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, object]:
        memory = self._flatten_with_scale_emb(list(feats))           # (B, sum_HW, C)
        b = memory.shape[0]

        # ---- Standard query path ----
        anchors_sigmoid = self.anchor_params.sigmoid().unsqueeze(0).expand(b, -1, -1)  # (B, K, 3)
        pos = self._anchor_pos_embed(anchors_sigmoid)                                   # (B, K, C)
        tgt = self.content_embed.weight.unsqueeze(0).expand(b, -1, -1).contiguous()
        decoded = self.decoder(tgt + pos, memory)                                       # (B, K, C)

        cls_logits = self.cls_head(decoded).squeeze(-1)                                  # (B, K)
        param_delta = self.param_head(decoded)                                           # (B, K, 4)
        # Combine anchor + delta for params, then sigmoid for [0, 1].
        params_3d = anchors_sigmoid + param_delta[..., :3] * 0.1                          # gentle refinement
        params = torch.cat([params_3d, param_delta[..., 3:4].sigmoid()], dim=-1).sigmoid()  # apply sigmoid for safety
        offsets = 0.05 * torch.tanh(self.offset_head(decoded))                            # (B, K, num_points)
        coord_pred = self._curves_from_params(params, offsets)
        lane_class_logits = self.lane_class_head(decoded)

        out: Dict[str, object] = {
            'cls_logits': cls_logits,
            'coord_pred': coord_pred,
            'lane_param': params,
            'lane_offsets': offsets,
            'lane_class_logits': lane_class_logits,
            'selected_prior': params[..., :3].contiguous(),
            'selected_prior_indices': torch.arange(self.num_queries, device=memory.device).view(1, -1).expand(b, -1),
            'prior_select_logits': cls_logits,
        }

        # ---- Denoising query path (training-only) ----
        dn_pkg = self._build_dn_queries(memory, targets)
        if dn_pkg is not None:
            anchors_noised = dn_pkg['anchors']                                            # (B, M, 3)
            anchors_target = dn_pkg['targets']                                            # (B, M, 3)
            valid_mask = dn_pkg['valid_mask']                                              # (B, M)
            pos_dn = self._anchor_pos_embed(anchors_noised)                                # (B, M, C)
            tgt_dn = pos_dn.new_zeros(pos_dn.shape) + self.content_embed.weight[0]         # share content
            decoded_dn = self.decoder(tgt_dn + pos_dn, memory)                              # (B, M, C)
            param_delta_dn = self.param_head(decoded_dn)
            params_dn_3d = anchors_noised + param_delta_dn[..., :3] * 0.1
            params_dn = torch.cat(
                [params_dn_3d, param_delta_dn[..., 3:4].sigmoid()], dim=-1
            ).sigmoid()
            offsets_dn = 0.05 * torch.tanh(self.offset_head(decoded_dn))
            cls_logits_dn = self.cls_head(decoded_dn).squeeze(-1)
            coord_pred_dn = self._curves_from_params(params_dn, offsets_dn)
            out['dn_cls_logits'] = cls_logits_dn
            out['dn_coord_pred'] = coord_pred_dn
            out['dn_lane_param'] = params_dn
            out['dn_anchor_target'] = anchors_target
            out['dn_valid_mask'] = valid_mask

        if self.mask_aux:
            largest = max(feats, key=lambda f: f.shape[-2] * f.shape[-1])
            mask = self.mask_decoder(self.lateral[0](largest))
            out['mask_logit'] = F.interpolate(mask, size=self.mask_size, mode='bilinear', align_corners=False)
        return out


# ---------------------------------------------------------------------------
# Exp2S -- Bezier-curve query head (BezierLaneNet-inspired).
#
# All previous query-style experiments converged val_lane_f1 ~ 0.65 in 10
# epochs but matched_iou stayed at ~ 0.13 because predicting 72 row offsets
# from random init takes far longer than 10 epochs to converge. Bezier
# curves are a much more compact lane representation: 4 control points (8
# DOF) replaces start_y/start_x/theta/length + 72 row offsets (76 DOF).
# An 8x reduction in output dimensionality -> faster geometry convergence.
#
# Reference: BezierLaneNet (CVPR 2022, Feng et al. "Rethinking Efficient
# Lane Detection via Curve Modeling").
# ---------------------------------------------------------------------------
def _bezier_curve_from_control_points(
    control_points: torch.Tensor,
    num_sample_points: int,
) -> torch.Tensor:
    """Sample a cubic Bezier curve at num_sample_points evenly spaced t.

    control_points: (..., 4, 2) where the last dim is (x, y).
    Returns: (..., num_sample_points, 2)
    """
    *prefix, num_ctrl, _ = control_points.shape
    if num_ctrl != 4:
        raise RuntimeError(f'Cubic Bezier expects 4 control points, got {num_ctrl}')
    device = control_points.device
    dtype = control_points.dtype
    t = torch.linspace(0.0, 1.0, steps=num_sample_points, device=device, dtype=dtype)  # (S,)
    # Bernstein basis for cubic: (1-t)^3, 3(1-t)^2 t, 3(1-t) t^2, t^3
    one_minus_t = 1.0 - t
    b0 = one_minus_t ** 3
    b1 = 3.0 * (one_minus_t ** 2) * t
    b2 = 3.0 * one_minus_t * (t ** 2)
    b3 = t ** 3
    basis = torch.stack([b0, b1, b2, b3], dim=-1)                                       # (S, 4)
    # control_points: (..., 4, 2). Multiply by basis (S, 4) -> (..., S, 2).
    # Expand control_points to (..., 1, 4, 2), basis to (1, S, 4, 1).
    expanded_cp = control_points.unsqueeze(-3)                                          # (..., 1, 4, 2)
    expanded_basis = basis.view((1,) * len(prefix) + (num_sample_points, 4, 1))         # (..., S, 4, 1)
    weighted = expanded_cp * expanded_basis                                             # (..., S, 4, 2)
    return weighted.sum(dim=-2)                                                          # (..., S, 2)


@torch.no_grad()
def _gt_lanes_to_bezier_targets(
    points: torch.Tensor,
    visibility: torch.Tensor,
    num_sample_points: int,
) -> torch.Tensor:
    """Approximate per-GT-lane Bezier control points from sampled GT points.

    Uses a least-squares fit of a cubic Bezier to the visible GT points.
    Returns: (B, L, 4, 2) control point tensor. For lanes with < 4 valid
    points, returns zeros (the caller should mask these out).
    Cached only at training time; not on the autograd path (uses lstsq).
    """
    b, L, n, _ = points.shape
    targets = points.new_zeros((b, L, 4, 2))
    # Build the Bernstein matrix once.
    t_full = torch.linspace(0.0, 1.0, steps=n, device=points.device, dtype=points.dtype)
    one_minus_t = 1.0 - t_full
    basis = torch.stack([
        one_minus_t ** 3,
        3.0 * (one_minus_t ** 2) * t_full,
        3.0 * one_minus_t * (t_full ** 2),
        t_full ** 3,
    ], dim=-1)                                                                           # (n, 4)
    for bi in range(b):
        for li in range(L):
            mask = visibility[bi, li] > 0.5
            if mask.sum() < 4:
                continue
            B_mat = basis[mask]                                                          # (k, 4)
            P_mat = points[bi, li, mask]                                                 # (k, 2)
            # Solve B_mat @ C = P_mat for C of shape (4, 2).
            try:
                solution = torch.linalg.lstsq(B_mat, P_mat).solution                     # (4, 2)
                targets[bi, li] = solution.clamp(0.0, 1.0)
            except Exception:
                continue
    return targets


class BezierLaneQueryHead(nn.Module):
    """Query head with Bezier-curve outputs.

    K=12 learned queries through a transformer decoder. Each query outputs
    cls (1d) + 4 cubic Bezier control points (8d) = 9 dof per lane vs the
    standard query head's 1 + 4 + 72 = 77 dof. The control points are in
    normalized image coordinates [0, 1]^2.

    For loss compatibility with FusionLaneLoss, the head also samples the
    Bezier curve at num_points (default 72) to produce the standard
    coord_pred (B, K, num_points, 2). This means dynamic_k / Hungarian and
    LineIoU all work unchanged.

    Output dict has the same keys as LaneQueryHead with one addition:
    'bezier_control_points': (B, K, 4, 2). The train script can optionally
    add a control-point regression loss if `loss.lane.w_bezier_ctrl > 0`,
    which is supervised by least-squares-fitted Bezier targets from the GT
    points (computed in losses.py if `cls_target_type='matched_existence'`).
    """

    def __init__(
        self,
        in_channels: Sequence[int] = (128, 128, 128),
        embed_dim: int = 128,
        max_lanes: int = 12,
        num_points: int = 72,
        mask_size: Tuple[int, int] = (72, 128),
        mask_aux: bool = False,
        num_lane_classes: int = 7,
        num_queries: int = 12,
        num_decoder_layers: int = 3,
        num_heads: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_queries = int(num_queries)
        self.num_priors = int(num_queries)
        self.embed_dim = int(embed_dim)
        self.max_lanes = int(max_lanes)
        self.num_points = int(num_points)
        self.mask_size = (int(mask_size[0]), int(mask_size[1]))
        self.mask_aux = bool(mask_aux)
        self.num_lane_classes = int(num_lane_classes)

        self.lateral = nn.ModuleList([_conv_bn_act(c, embed_dim, k=1) for c in in_channels])
        self.scale_embed = nn.Embedding(len(in_channels), embed_dim)

        self.query_embed = nn.Embedding(num_queries, embed_dim)
        self.query_pos = nn.Embedding(num_queries, embed_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.cls_head = nn.Linear(embed_dim, 1)
        # 4 control points * 2 coords = 8 outputs, sigmoid'd to [0, 1] image space.
        self.ctrl_head = _MLP(embed_dim, embed_dim, 8, num_layers=3)
        self.lane_class_head = nn.Linear(embed_dim, num_lane_classes)

        self.register_buffer('prior_ys', torch.linspace(1.0, 0.0, steps=self.num_points, dtype=torch.float32))

        if self.mask_aux:
            self.mask_decoder = nn.Sequential(
                _conv_bn_act(embed_dim, max(16, embed_dim // 2), k=3),
                nn.Conv2d(max(16, embed_dim // 2), 1, kernel_size=1),
            )

    def _flatten_with_scale_emb(self, feats: List[torch.Tensor]) -> torch.Tensor:
        chunks: List[torch.Tensor] = []
        for i, f in enumerate(feats):
            f_proj = self.lateral[i](f)
            f_flat = f_proj.flatten(2).transpose(1, 2)
            scale_e = self.scale_embed.weight[i].view(1, 1, -1).expand_as(f_flat)
            chunks.append(f_flat + scale_e)
        return torch.cat(chunks, dim=1)

    def forward(self, feats: List[torch.Tensor]) -> Dict[str, object]:
        memory = self._flatten_with_scale_emb(list(feats))
        b = memory.shape[0]

        tgt = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1).contiguous()
        pos = self.query_pos.weight.unsqueeze(0).expand(b, -1, -1).contiguous()
        decoded = self.decoder(tgt + pos, memory)                                        # (B, K, C)

        cls_logits = self.cls_head(decoded).squeeze(-1)                                  # (B, K)
        ctrl_raw = self.ctrl_head(decoded)                                               # (B, K, 8)
        ctrl_xy = ctrl_raw.view(b, self.num_queries, 4, 2).sigmoid()                     # (B, K, 4, 2) in [0,1]
        coord_pred = _bezier_curve_from_control_points(ctrl_xy, self.num_points)         # (B, K, N, 2)

        # For compat with FusionLaneLoss: derive (start_y, start_x, theta, length)
        # from the Bezier-sampled curve.
        first_pt = coord_pred[:, :, 0, :]                                                # (B, K, 2) at t=0 -> y=1
        last_pt = coord_pred[:, :, -1, :]                                                # (B, K, 2) at t=1 -> y=0
        start_x = first_pt[..., 0:1]
        start_y = first_pt[..., 1:2]
        dy = (last_pt[..., 1:2] - first_pt[..., 1:2]).abs()
        dx = (last_pt[..., 0:1] - first_pt[..., 0:1]).abs()
        theta = torch.atan2(dy + 1e-4, dx + 1e-4) / math.pi                              # in [0, 0.5]
        length = (dx ** 2 + dy ** 2 + 1e-8).sqrt().clamp(0.0, 1.0)
        params = torch.cat([start_y, start_x, theta, length], dim=-1)                    # (B, K, 4)

        # Compute "row offsets" relative to the linear-from-anchor baseline so
        # the existing offset loss term can supervise the Bezier curve. Each
        # offset[i] = predicted_x[i] - linear_baseline_x[i].
        ys = self.prior_ys.to(coord_pred.device, coord_pred.dtype).view(1, 1, self.num_points)
        base_x = start_x + ((ys - start_y) / torch.tan(theta * math.pi + 1e-4))
        offsets = (coord_pred[..., 0] - base_x.squeeze(-1)).clamp(-1.0, 1.0)             # (B, K, num_points)

        lane_class_logits = self.lane_class_head(decoded)

        out: Dict[str, object] = {
            'cls_logits': cls_logits,
            'coord_pred': coord_pred,
            'lane_param': params,
            'lane_offsets': offsets,
            'lane_class_logits': lane_class_logits,
            'bezier_control_points': ctrl_xy,
            'selected_prior': params[..., :3].contiguous(),
            'selected_prior_indices': torch.arange(self.num_queries, device=memory.device).view(1, -1).expand(b, -1),
            'prior_select_logits': cls_logits,
        }
        if self.mask_aux:
            largest = max(feats, key=lambda f: f.shape[-2] * f.shape[-1])
            mask = self.mask_decoder(self.lateral[0](largest))
            out['mask_logit'] = F.interpolate(mask, size=self.mask_size, mode='bilinear', align_corners=False)
        return out
