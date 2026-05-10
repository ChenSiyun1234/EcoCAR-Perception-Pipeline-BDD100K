from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import ConvBNAct, DepthwiseConvBNAct, CSPRepBlock, AIFI, TaskAdapter, GateControlAdapter


class YOLO26Conv(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, groups: int = 1):
        super().__init__()
        self.block = ConvBNAct(c1, c2, k=k, s=s, groups=groups, act='silu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class YOLO26Bottleneck(nn.Module):
    def __init__(self, channels: int, expansion: float = 0.5, use_depthwise: bool = True, shortcut: bool = True):
        super().__init__()
        hidden = max(16, int(channels * expansion))
        self.shortcut = bool(shortcut)
        self.cv1 = YOLO26Conv(channels, hidden, k=1, s=1)
        if use_depthwise:
            self.cv2 = DepthwiseConvBNAct(hidden, hidden, k=3, s=1, act='silu')
        else:
            self.cv2 = YOLO26Conv(hidden, hidden, k=3, s=1)
        self.cv3 = YOLO26Conv(hidden, channels, k=1, s=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv3(self.cv2(self.cv1(x)))
        if self.shortcut:
            y = y + x
        return y


class YOLO26CSPBlock(nn.Module):
    """C2f/CSP-style aggregation block used by the custom YOLO26-inspired backbone.

    This is not a black-box import of a YOLO26 checkpoint. It implements the design
    idea we want for Stage 2: cheap residual feature aggregation with a split path,
    local texture preservation, and export-friendly conv/bn/activation operators.
    """

    def __init__(self, c1: int, c2: int, repeats: int = 2, expansion: float = 0.5, use_depthwise: bool = True):
        super().__init__()
        hidden = max(16, int(c2 * expansion))
        self.cv1 = YOLO26Conv(c1, hidden * 2, k=1, s=1)
        self.blocks = nn.ModuleList(
            [YOLO26Bottleneck(hidden, expansion=1.0, use_depthwise=use_depthwise, shortcut=True) for _ in range(max(1, repeats))]
        )
        self.cv2 = YOLO26Conv(hidden * (2 + max(1, repeats)), c2, k=1, s=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        left, right = self.cv1(x).chunk(2, dim=1)
        outputs = [left, right]
        y = right
        for block in self.blocks:
            y = block(y)
            outputs.append(y)
        return self.cv2(torch.cat(outputs, dim=1))


class YOLO26Stage(nn.Module):
    def __init__(self, c1: int, c2: int, repeats: int, use_depthwise: bool = True):
        super().__init__()
        self.down = YOLO26Conv(c1, c2, k=3, s=2)
        self.body = YOLO26CSPBlock(c2, c2, repeats=repeats, use_depthwise=use_depthwise)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(self.down(x))


class P5GlobalContextBlock(nn.Module):
    def __init__(self, channels: int, use_attention: bool = True):
        super().__init__()
        self.use_attention = bool(use_attention)
        self.pre = YOLO26Conv(channels, channels, k=1, s=1)
        self.global_block = AIFI(channels, num_heads=max(1, channels // 32)) if self.use_attention else nn.Identity()
        self.post = YOLO26Conv(channels, channels, k=1, s=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.pre(x)
        x = self.global_block(x)
        x = self.post(x)
        return x + residual


class LanePreservingP3Block(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.local_dw = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.dilated_dw = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.mix = YOLO26Conv(channels * 2, channels, k=1, s=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.local_dw(x)
        long_range = self.dilated_dw(x)
        return x + self.mix(torch.cat([local, long_range], dim=1))


class YOLO26PANNeck(nn.Module):
    def __init__(self, c3: int, c4: int, c5: int, out_channels: int = 128, use_p5_global: bool = True, use_lane_p3_refine: bool = True):
        super().__init__()
        out_c = int(out_channels)
        self.p3_in = YOLO26Conv(c3, out_c, k=1, s=1)
        self.p4_in = YOLO26Conv(c4, out_c, k=1, s=1)
        self.p5_in = YOLO26Conv(c5, out_c, k=1, s=1)
        self.p5_global = P5GlobalContextBlock(out_c, use_attention=use_p5_global)
        self.fpn4 = CSPRepBlock(out_c * 2, out_c, repeats=1)
        self.fpn3 = CSPRepBlock(out_c * 2, out_c, repeats=1)
        self.p3_lane_refine = LanePreservingP3Block(out_c) if use_lane_p3_refine else nn.Identity()
        self.down3 = YOLO26Conv(out_c, out_c, k=3, s=2)
        self.pan4 = CSPRepBlock(out_c * 2, out_c, repeats=1)
        self.down4 = YOLO26Conv(out_c, out_c, k=3, s=2)
        self.pan5 = CSPRepBlock(out_c * 2, out_c, repeats=1)

    def forward(self, c3: torch.Tensor, c4: torch.Tensor, c5: torch.Tensor) -> List[torch.Tensor]:
        p3 = self.p3_in(c3)
        p4 = self.p4_in(c4)
        p5 = self.p5_global(self.p5_in(c5))
        p4_td = self.fpn4(torch.cat([F.interpolate(p5, size=p4.shape[-2:], mode='nearest'), p4], dim=1))
        p3_td = self.fpn3(torch.cat([F.interpolate(p4_td, size=p3.shape[-2:], mode='nearest'), p3], dim=1))
        p3_td = self.p3_lane_refine(p3_td)
        p4_out = self.pan4(torch.cat([self.down3(p3_td), p4_td], dim=1))
        p5_out = self.pan5(torch.cat([self.down4(p4_out), p5], dim=1))
        return [p3_td, p4_out, p5_out]


class YOLO26InspiredJointBackboneNeck(nn.Module):
    """Official-YOLO26-inspired custom backbone/neck for Stage 2 joint fusion.

    Design target:
      image -> efficient CNN stages -> custom P3/P4/P5 neck -> optional GCA split
      P3/P4/P5 are our own neck outputs, not hooks into an external model.
    """

    def __init__(
        self,
        width: float = 0.5,
        depth: float = 0.33,
        out_channels: int = 128,
        use_p5_global: bool = True,
        use_lane_p3_refine: bool = True,
        use_gca: bool = False,
        use_depthwise: bool = True,
    ):
        super().__init__()
        width = float(width)
        depth = float(depth)
        out_c = int(out_channels)
        c2 = max(32, int(128 * width))
        c3 = max(64, int(256 * width))
        c4 = max(128, int(512 * width))
        c5 = max(256, int(1024 * width))
        r3 = max(1, round(3 * depth))
        r4 = max(2, round(6 * depth))
        r5 = max(1, round(3 * depth))
        self.feature_channels = [out_c, out_c, out_c]
        self.use_gca = bool(use_gca)
        self.stem = nn.Sequential(
            YOLO26Conv(3, max(16, c2 // 2), k=3, s=2),
            YOLO26Conv(max(16, c2 // 2), c2, k=3, s=2),
        )
        self.stage3 = YOLO26Stage(c2, c3, repeats=r3, use_depthwise=use_depthwise)
        self.stage4 = YOLO26Stage(c3, c4, repeats=r4, use_depthwise=use_depthwise)
        self.stage5 = YOLO26Stage(c4, c5, repeats=r5, use_depthwise=use_depthwise)
        self.neck = YOLO26PANNeck(c3, c4, c5, out_channels=out_c, use_p5_global=use_p5_global, use_lane_p3_refine=use_lane_p3_refine)
        self.det_adapters = nn.ModuleList([TaskAdapter(out_c) for _ in range(3)])
        self.lane_adapters = nn.ModuleList([TaskAdapter(out_c) for _ in range(3)])
        self.gate_det = nn.ModuleList([GateControlAdapter(out_c) for _ in range(3)])
        self.gate_lane = nn.ModuleList([GateControlAdapter(out_c) for _ in range(3)])

    def _shared_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        c3 = self.stage3(x)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)
        return self.neck(c3, c4, c5)

    def forward(self, x: torch.Tensor):
        shared = self._shared_features(x)
        if not self.use_gca:
            return shared
        det_feats: List[torch.Tensor] = []
        lane_feats: List[torch.Tensor] = []
        det_gates: List[torch.Tensor] = []
        lane_gates: List[torch.Tensor] = []
        for i, feat in enumerate(shared):
            det_task = self.det_adapters[i](feat)
            lane_task = self.lane_adapters[i](feat)
            det_feat, det_gate = self.gate_det[i](feat, det_task)
            lane_feat, lane_gate = self.gate_lane[i](feat, lane_task)
            det_feats.append(det_feat)
            lane_feats.append(lane_feat)
            det_gates.append(det_gate)
            lane_gates.append(lane_gate)
        gate_stats = {
            'gate/det_mean': torch.stack([g.mean() for g in det_gates]).mean(),
            'gate/lane_mean': torch.stack([g.mean() for g in lane_gates]).mean(),
            'gate/det_sat_low': torch.stack([(g <= 0.051).float().mean() for g in det_gates]).mean(),
            'gate/det_sat_high': torch.stack([(g >= 0.949).float().mean() for g in det_gates]).mean(),
            'gate/lane_sat_low': torch.stack([(g <= 0.051).float().mean() for g in lane_gates]).mean(),
            'gate/lane_sat_high': torch.stack([(g >= 0.949).float().mean() for g in lane_gates]).mean(),
        }
        return {'shared': shared, 'det': det_feats, 'lane': lane_feats, 'gates': {'det': det_gates, 'lane': lane_gates}, 'gate_stats': gate_stats}

    def describe_feature_shapes(self, image_size: Tuple[int, int] = (384, 640)) -> Dict[str, Tuple[int, int]]:
        h, w = int(image_size[0]), int(image_size[1])
        return {
            'P3_stride8': (h // 8, w // 8),
            'P4_stride16': (h // 16, w // 16),
            'P5_stride32': (h // 32, w // 32),
        }
