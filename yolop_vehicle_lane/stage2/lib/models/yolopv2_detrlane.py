"""
YOLOPv2-DETRLane — YOLOPv2 encoder/neck + IDetect head + DETR-style set
lane head with point prediction (no lane mask).

Design goals:
  1. Keep the YOLOPv2 ELAN backbone + SPPCSPC + PAN neck IDENTICAL to
     `yolopv2_baseline.py`. Zero-touch on the baseline.
  2. Keep the anchor-based IDetect vehicle head IDENTICAL.
  3. Replace the lane seg decoder (layers 29..37 in the baseline
     block_cfg) with a DETR-style `LaneSetHead` that consumes the same
     P3 / P4 / P5 PAN outputs the detector consumes.
  4. Add optional task-specific adapters + stage-based training gate
     (both are well-justified gradient-conflict mitigators in the 2024
     multi-task-learning literature: PCGrad, CAGrad, expert-squads,
     task adapters).

Nothing in this file is in the YOLOPv2 baseline file; picking the
`YOLOPv2-DETRLane` baseline in `configs/` leaves the reproduction path
unaffected.
"""

from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn as nn

from lib.models.common import (
    Conv, Concat, ELAN, MP, SPPCSPC, IDetect, Detect,
)
from lib.models.lane_set_head import LaneSetHead, TaskFeatureAdapter
from lib.utils import initialize_weights, check_anchor_order


# Reuse the YOLOPv2 block_cfg layout up to the PAN (layers 0..27) — we
# drop the 29..37 lane seg decoder and replace it with `LaneSetHead`.
# Detect head still takes features from layers [21, 24, 27].
YOLOPv2DETRCfg_Backbone = [
    [-1, Conv, [3,  32, 3, 1]],              # 0
    [-1, Conv, [32, 64, 3, 2]],              # 1
    [-1, Conv, [64, 64, 3, 1]],              # 2
    [-1, Conv, [64, 128, 3, 2]],             # 3
    [-1, ELAN, [128, 128]],                  # 4
    [-1, MP,   [128, 128]],                  # 5
    [-1, ELAN, [128, 256]],                  # 6  P3 (encoder out)
    [-1, MP,   [256, 256]],                  # 7
    [-1, ELAN, [256, 512]],                  # 8  P4 (encoder out)
    [-1, MP,   [512, 512]],                  # 9
    [-1, ELAN, [512, 1024]],                 # 10
    [-1, SPPCSPC, [1024, 512]],              # 11
    [-1, Conv,     [512, 256, 1, 1]],        # 12
    [-1, nn.Upsample, [None, 2, 'nearest']], # 13
    [8,  Conv,     [512, 256, 1, 1]],        # 14
    [[-1, 13], Concat, [1]],                 # 15
    [-1, ELAN,     [512, 256]],              # 16
    [-1, Conv,     [256, 128, 1, 1]],        # 17
    [-1, nn.Upsample, [None, 2, 'nearest']], # 18
    [6,  Conv,     [256, 128, 1, 1]],        # 19
    [[-1, 18], Concat, [1]],                 # 20
    [-1, ELAN,     [256, 128]],              # 21  P3 PAN  (→ det + lane)
    [-1, MP,       [128, 256]],              # 22
    [[-1, 16], Concat, [1]],                 # 23
    [-1, ELAN,     [512, 256]],              # 24  P4 PAN  (→ det + lane)
    [-1, MP,       [256, 512]],              # 25
    [[-1, 11], Concat, [1]],                 # 26
    [-1, ELAN,     [1024, 512]],             # 27  P5 PAN  (→ det + lane)
]

# Channels at the three PAN outputs consumed by both heads.
PAN_OUT_CHANNELS = (128, 256, 512)
# Indices into the sequential encoder that produce the PAN outputs.
PAN_OUT_INDICES = (21, 24, 27)


class YOLOPv2DETRLaneNet(nn.Module):
    def __init__(self,
                 nc: int = 5,
                 # Lane-head config
                 lane_num_queries: int = 10,
                 lane_num_points: int = 72,
                 lane_num_types: int = 7,
                 lane_d_model: int = 256,
                 lane_nhead: int = 8,
                 lane_ffn_dim: int = 1024,
                 lane_dec_layers: int = 3,
                 lane_dropout: float = 0.0,
                 use_task_adapters: bool = True,
                 adapter_hidden_ratio: float = 0.5,
                 # Anchor config — matches yolopv2_baseline.py
                 anchors=None):
        super().__init__()
        self.nc = nc
        self.current_epoch = 0
        self.det_only_epochs = 0   # set from trainer for staged training

        anchors = anchors or [[3, 9, 5, 11, 4, 20],
                              [7, 18, 6, 39, 12, 31],
                              [19, 50, 38, 81, 68, 157]]

        # ── Build the shared encoder (YOLOPv2 backbone + neck + PAN) ──
        layers, save = [], []
        for i, (from_, block, args) in enumerate(YOLOPv2DETRCfg_Backbone):
            block_ = block(*args)
            block_.index, block_.from_ = i, from_
            layers.append(block_)
            save.extend(x % i for x in ([from_] if isinstance(from_, int) else from_) if x != -1)
        # Detection head is separate from the sequential encoder — we pluck
        # the features from the encoder's PAN outputs and feed them to
        # IDetect and LaneSetHead directly.
        self.encoder = nn.Sequential(*layers)
        self.encoder_save = sorted(set(list(save) + list(PAN_OUT_INDICES)))

        # ── Task-specific residual adapters ──
        # Reduce gradient conflict by letting each head shape its features
        # slightly differently while sharing the trunk.
        self.use_task_adapters = use_task_adapters
        if use_task_adapters:
            self.det_adapters = nn.ModuleList(
                TaskFeatureAdapter(c, adapter_hidden_ratio) for c in PAN_OUT_CHANNELS)
            self.lane_adapters = nn.ModuleList(
                TaskFeatureAdapter(c, adapter_hidden_ratio) for c in PAN_OUT_CHANNELS)
        else:
            self.det_adapters = None
            self.lane_adapters = None

        # ── Detection head (anchor-based IDetect, unchanged from baseline) ──
        self.detect = IDetect(nc, anchors, list(PAN_OUT_CHANNELS))
        self.detect.from_ = list(PAN_OUT_INDICES)
        self.detect.index = -1

        # ── Lane set head (point-based, DETR-style) ──
        self.lane_head = LaneSetHead(
            in_channels=list(PAN_OUT_CHANNELS),
            num_lane_types=lane_num_types,
            num_points=lane_num_points,
            d_model=lane_d_model,
            nhead=lane_nhead,
            ffn_dim=lane_ffn_dim,
            num_layers=lane_dec_layers,
            num_queries=lane_num_queries,
            dropout=lane_dropout,
        )

        self.names = [str(i) for i in range(nc)]

        # ── Anchor stride calibration (YOLOP-style) ──
        s = 128
        with torch.no_grad():
            det_out, _ = self._forward_encoder_and_det(torch.zeros(1, 3, s, s))
        self.detect.stride = torch.tensor([s / x.shape[-2] for x in det_out])
        self.detect.anchors /= self.detect.stride.view(-1, 1, 1)
        check_anchor_order(self.detect)
        self.stride = self.detect.stride
        self._initialize_biases()

        initialize_weights(self)

    # ── Config setters for staged training / inference ────────────────
    def set_epoch(self, epoch: int):
        self.current_epoch = int(epoch)

    def set_det_only_epochs(self, n: int):
        """If current_epoch < det_only_epochs the lane head is frozen in
        forward (returns its pre-initialised prior points and zero logits
        with no grad). Trainer-side should also multiply lane loss by 0.
        """
        self.det_only_epochs = int(n)

    # ── Forward helpers ───────────────────────────────────────────────
    def _forward_encoder(self, x: torch.Tensor) -> Dict[int, torch.Tensor]:
        cache = {}
        for i, block in enumerate(self.encoder):
            if block.from_ != -1:
                x = cache[block.from_] if isinstance(block.from_, int) else \
                    [x if j == -1 else cache[j] for j in block.from_]
            x = block(x)
            if i in self.encoder_save:
                cache[i] = x
        return cache

    def _pan_features(self, cache) -> List[torch.Tensor]:
        return [cache[i] for i in PAN_OUT_INDICES]

    def _forward_encoder_and_det(self, x: torch.Tensor):
        cache = self._forward_encoder(x)
        pan = self._pan_features(cache)
        det_feats = [a(f) for a, f in zip(self.det_adapters, pan)] if self.det_adapters else pan
        det_out = self.detect(det_feats)
        return det_out, pan

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        det_out, pan = self._forward_encoder_and_det(x)
        if self.current_epoch < self.det_only_epochs:
            # Stage A: detection-only warmup.
            # Emit a frozen lane prediction with no gradient on lane params.
            with torch.no_grad():
                lane_out = self.lane_head([f.detach() for f in pan])
        else:
            lane_feats = [a(f) for a, f in zip(self.lane_adapters, pan)] if self.lane_adapters else pan
            lane_out = self.lane_head(lane_feats)

        return {
            'det_out': det_out,          # (infer, train) per IDetect
            'lane': lane_out,            # dict
        }

    def _initialize_biases(self, cf=None):
        m = self.detect
        for mi, s in zip(m.m, m.stride):
            b = mi.bias.view(m.na, -1)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())
            mi.bias = nn.Parameter(b.view(-1), requires_grad=True)


def get_net_yolopv2_detrlane(cfg=None, **kwargs) -> YOLOPv2DETRLaneNet:
    """Factory. Reads optional cfg.MODEL.LANE.* and MODEL.NC."""
    nc = 5
    lane_kw = {}
    if cfg is not None:
        nc = int(getattr(cfg.MODEL, 'NC', 5))
        lane = getattr(cfg.MODEL, 'LANE', None)
        if lane is not None:
            lane_kw = dict(
                lane_num_queries=int(getattr(lane, 'NUM_QUERIES', 10)),
                lane_num_points=int(getattr(lane, 'NUM_POINTS', 72)),
                lane_num_types=int(getattr(lane, 'NUM_TYPES', 7)),
                lane_d_model=int(getattr(lane, 'D_MODEL', 256)),
                lane_nhead=int(getattr(lane, 'NHEAD', 8)),
                lane_ffn_dim=int(getattr(lane, 'FFN_DIM', 1024)),
                lane_dec_layers=int(getattr(lane, 'DEC_LAYERS', 3)),
                lane_dropout=float(getattr(lane, 'DROPOUT', 0.0)),
                use_task_adapters=bool(getattr(lane, 'USE_TASK_ADAPTERS', True)),
                adapter_hidden_ratio=float(getattr(lane, 'ADAPTER_HIDDEN_RATIO', 0.5)),
            )
    return YOLOPv2DETRLaneNet(nc=nc, **lane_kw, **kwargs)


if __name__ == '__main__':
    m = get_net_yolopv2_detrlane()
    m.eval()
    with torch.no_grad():
        out = m(torch.zeros(1, 3, 640, 640))
    print('lane exist_logits:', out['lane']['exist_logits'].shape)
    print('lane pred_points:', out['lane']['pred_points'].shape)
    print('det train_out[0]:', out['det_out'][1][0].shape if isinstance(out['det_out'], tuple) else out['det_out'][0].shape)
    print(f'params: {sum(p.numel() for p in m.parameters()) / 1e6:.2f} M')
