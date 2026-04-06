"""
DualPathNet — weakly-coupled dual-path perception model.

This revision adds two engineering mechanisms specifically aimed at reducing
multi-task interference between detection and lane prediction:

1) task-specific residual adapters on top of the shared FPN features
2) scheduled / gated cross-branch attention instead of unconditional fusion
"""

import torch
import torch.nn as nn
from typing import Dict, List

from .backbone import BackboneFPN
from .encoder import HybridEncoder
from .detection_head import DetectionHead, inverse_sigmoid
from .lane_head import LaneHead
from .config import Config, NUM_CLASSES, NUM_LANE_TYPES
from .segmentation_head import LightweightSegmentationHead


class TaskFeatureAdapter(nn.Module):
    def __init__(self, channels: int, hidden_ratio: float = 0.5):
        super().__init__()
        hidden = max(32, int(channels * hidden_ratio))
        self.block = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.scale = nn.Parameter(torch.tensor(0.10))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale * self.block(x)


class CrossBranchAttention(nn.Module):
    def __init__(self, d_model: int = 256, nhead: int = 8, num_layers: int = 1):
        super().__init__()
        self.det_to_lane = nn.ModuleList([nn.MultiheadAttention(d_model, nhead, batch_first=True) for _ in range(num_layers)])
        self.lane_to_det = nn.ModuleList([nn.MultiheadAttention(d_model, nhead, batch_first=True) for _ in range(num_layers)])
        self.norm_det = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.norm_lane = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.det_gate = nn.Parameter(torch.tensor(0.0))
        self.lane_gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, det_queries: torch.Tensor, lane_queries: torch.Tensor, gate_scale: float = 1.0) -> tuple:
        det_mix = torch.sigmoid(self.det_gate) * gate_scale
        lane_mix = torch.sigmoid(self.lane_gate) * gate_scale
        for attn_d2l, attn_l2d, norm_d, norm_l in zip(self.det_to_lane, self.lane_to_det, self.norm_det, self.norm_lane):
            lane_q = norm_l(lane_queries)
            lane_queries = lane_queries + lane_mix * attn_d2l(lane_q, det_queries, det_queries)[0]
            det_q = norm_d(det_queries)
            det_queries = det_queries + det_mix * attn_l2d(det_q, lane_queries, lane_queries)[0]
        return det_queries, lane_queries


class DualPathNet(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.current_epoch = 0
        nc = 7 if cfg.use_expanded_classes else NUM_CLASSES
        self.num_classes = nc
        self.backbone = BackboneFPN(name=cfg.backbone, pretrained=cfg.pretrained, fpn_channels=cfg.fpn_channels)

        self.use_task_adapters = bool(getattr(cfg, 'use_task_adapters', True))
        if self.use_task_adapters:
            ratio = float(getattr(cfg, 'adapter_hidden_ratio', 0.5))
            self.det_adapters = nn.ModuleList([TaskFeatureAdapter(cfg.fpn_channels, ratio) for _ in range(3)])
            self.lane_adapters = nn.ModuleList([TaskFeatureAdapter(cfg.fpn_channels, ratio) for _ in range(3)])
        else:
            self.det_adapters = None
            self.lane_adapters = None

        self.det_encoder = HybridEncoder(d_model=cfg.det_dim, nhead=cfg.det_nhead, ffn_dim=cfg.det_ffn_dim, num_layers=cfg.det_enc_layers, dropout=cfg.det_dropout)
        self.lane_encoder = HybridEncoder(d_model=cfg.lane_dim, nhead=cfg.lane_nhead, ffn_dim=cfg.lane_ffn_dim, num_layers=cfg.lane_enc_layers, dropout=cfg.lane_dropout)
        self.det_proj = nn.Conv2d(cfg.fpn_channels, cfg.det_dim, 1) if cfg.fpn_channels != cfg.det_dim else nn.Identity()
        self.lane_proj = nn.Conv2d(cfg.fpn_channels, cfg.lane_dim, 1) if cfg.fpn_channels != cfg.lane_dim else nn.Identity()
        self.det_head = DetectionHead(num_classes=nc, d_model=cfg.det_dim, nhead=cfg.det_nhead, ffn_dim=cfg.det_ffn_dim, num_layers=cfg.det_dec_layers, num_queries=cfg.det_num_queries, dropout=cfg.det_dropout)
        self.lane_head = LaneHead(num_lane_types=NUM_LANE_TYPES, num_points=cfg.lane_points, d_model=cfg.lane_dim, nhead=cfg.lane_nhead, ffn_dim=cfg.lane_ffn_dim, num_layers=cfg.lane_dec_layers, num_queries=cfg.lane_num_queries, dropout=cfg.lane_dropout)
        self.cross_attn = None
        self.seg_head = LightweightSegmentationHead(in_channels=cfg.fpn_channels, hidden_dim=cfg.seg_hidden_dim, num_prototypes=cfg.seg_num_prototypes, mask_dim=cfg.seg_mask_dim) if getattr(cfg, 'enable_segmentation', False) else None
        if cfg.cross_attn and cfg.det_dim == cfg.lane_dim:
            self.cross_attn = CrossBranchAttention(d_model=cfg.det_dim, nhead=cfg.det_nhead, num_layers=cfg.cross_attn_layers)
        self._arch_config = {
            "backbone": cfg.backbone,
            "num_classes": nc,
            "det_queries": cfg.det_num_queries,
            "lane_queries": cfg.lane_num_queries,
            "lane_points": cfg.lane_points,
            "det_dim": cfg.det_dim,
            "lane_dim": cfg.lane_dim,
            "cross_attn": cfg.cross_attn,
            "use_task_adapters": self.use_task_adapters,
            "cross_attn_start_epoch": int(getattr(cfg, 'cross_attn_start_epoch', 0)),
        }

    def set_epoch(self, epoch: int):
        self.current_epoch = int(epoch)

    def _task_adapt(self, fpn_features: List[torch.Tensor]) -> tuple[List[torch.Tensor], List[torch.Tensor]]:
        if not self.use_task_adapters:
            return fpn_features, fpn_features
        det_features = [adapter(feat) for adapter, feat in zip(self.det_adapters, fpn_features)]
        lane_features = [adapter(feat) for adapter, feat in zip(self.lane_adapters, fpn_features)]
        return det_features, lane_features

    def _cross_gate_scale(self) -> float:
        start_epoch = int(getattr(self.cfg, 'cross_attn_start_epoch', 0))
        if self.current_epoch < start_epoch:
            return 0.0
        ramp = max(1, int(getattr(self.cfg, 'task_warmup_epochs', 8)))
        progress = min(1.0, float(self.current_epoch - start_epoch + 1) / float(ramp))
        return progress

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        fpn_features = self.backbone(images)
        det_fpn, lane_fpn = self._task_adapt(fpn_features)
        det_features = self.det_encoder([self.det_proj(f) for f in det_fpn])
        lane_features = self.lane_encoder([self.lane_proj(f) for f in lane_fpn])
        det_out = self.det_head(det_features)
        lane_out = self.lane_head(lane_features)

        if self.cross_attn is not None:
            gate_scale = self._cross_gate_scale()
            if gate_scale > 0.0:
                det_q, lane_q = self.cross_attn(det_out["query_features"], lane_out["query_features"], gate_scale=gate_scale)
                det_out["query_features"] = det_q
                lane_out["query_features"] = lane_q
                det_logits = self.det_head.class_heads[-1](det_q)
                det_boxes = (self.det_head.box_heads[-1](det_q) + inverse_sigmoid(det_out["pred_boxes"].detach())).sigmoid()
                lane_exist = self.lane_head.exist_heads[-1](lane_q)
                lane_delta = self.lane_head.point_heads[-1](lane_q).view(images.shape[0], self.lane_head.num_queries, self.lane_head.num_points, 2)
                lane_points = (lane_out["pred_points"].detach() + 0.10 * torch.tanh(lane_delta)).clamp(0.0, 1.0)
                lane_vis = self.lane_head.vis_heads[-1](lane_q)
                lane_type = self.lane_head.type_heads[-1](lane_q)
                det_out["pred_logits"] = det_logits
                det_out["pred_boxes"] = det_boxes
                lane_out["exist_logits"] = lane_exist
                lane_out["pred_points"] = lane_points
                lane_out["vis_logits"] = lane_vis
                lane_out["type_logits"] = lane_type

        merged = {**{f"det_{k}": v for k, v in det_out.items()}, **{f"lane_{k}": v for k, v in lane_out.items()}}
        if self.seg_head is not None:
            seg_out = self.seg_head(fpn_features[0], det_out["query_features"])
            merged.update(seg_out)
        return merged

    def print_summary(self):
        bb = sum(p.numel() for p in self.backbone.parameters())
        da = sum(p.numel() for p in self.det_adapters.parameters()) if self.det_adapters is not None else 0
        la = sum(p.numel() for p in self.lane_adapters.parameters()) if self.lane_adapters is not None else 0
        de = sum(p.numel() for p in self.det_encoder.parameters())
        dh = sum(p.numel() for p in self.det_head.parameters())
        le = sum(p.numel() for p in self.lane_encoder.parameters())
        lh = sum(p.numel() for p in self.lane_head.parameters())
        ca = sum(p.numel() for p in self.cross_attn.parameters()) if self.cross_attn else 0
        sh = sum(p.numel() for p in self.seg_head.parameters()) if self.seg_head else 0
        total = sum(p.numel() for p in self.parameters())
        print("DualPathNet summary:")
        print(f"  Backbone+FPN : {bb:>12,}")
        if da:
            print(f"  Det adapters : {da:>12,}")
        if la:
            print(f"  Lane adapters: {la:>12,}")
        print(f"  Det encoder  : {de:>12,}")
        print(f"  Det decoder  : {dh:>12,}")
        print(f"  Lane encoder : {le:>12,}")
        print(f"  Lane decoder : {lh:>12,}")
        if ca:
            print(f"  Cross-attn   : {ca:>12,}")
        if sh:
            print(f"  Seg head     : {sh:>12,}")
        print(f"  Total        : {total:>12,}")
        print(f"  Det queries={self.cfg.det_num_queries}, Lane queries={self.cfg.lane_num_queries}, Lane points={self.cfg.lane_points}")
        print(f"  Task adapters={self.use_task_adapters}, cross-attn start epoch={getattr(self.cfg, 'cross_attn_start_epoch', 0)}")


def build_model(cfg: Config) -> DualPathNet:
    return DualPathNet(cfg)
