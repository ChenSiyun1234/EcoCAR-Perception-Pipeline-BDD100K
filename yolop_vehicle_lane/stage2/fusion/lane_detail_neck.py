from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import ConvBNAct


class DepthwiseDilatedRefine(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.local = ConvBNAct(channels, channels, 3, 1, groups=channels)
        self.dilated = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=2, dilation=2, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )
        self.mix = ConvBNAct(channels, channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mix(self.dilated(self.local(x))) + x


class LaneDetailNeck(nn.Module):
    def __init__(self, c2_channels: int, out_channels: int = 128):
        super().__init__()
        self.c2_align = ConvBNAct(c2_channels, out_channels, 1, 1)
        self.p3_fuse = ConvBNAct(out_channels * 2, out_channels, 3, 1)
        self.p3_refine = DepthwiseDilatedRefine(out_channels)
        self.p4_refine = DepthwiseDilatedRefine(out_channels)
        self.p5_refine = DepthwiseDilatedRefine(out_channels)
        self.global_p5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, c2: torch.Tensor, shared: List[torch.Tensor]) -> List[torch.Tensor]:
        if len(shared) < 3:
            raise RuntimeError(f'LaneDetailNeck expects P3/P4/P5, got {len(shared)} feature maps')
        p3, p4, p5 = shared[:3]
        c2_detail = self.c2_align(c2)
        c2_detail = F.interpolate(c2_detail, size=p3.shape[-2:], mode='bilinear', align_corners=False)
        p3_lane = self.p3_refine(self.p3_fuse(torch.cat([p3, c2_detail], dim=1)))
        p4_lane = self.p4_refine(p4 + F.interpolate(p3_lane, size=p4.shape[-2:], mode='bilinear', align_corners=False))
        p5_gate = self.global_p5(p5)
        p5_lane = self.p5_refine(p5 * p5_gate + F.interpolate(p4_lane, size=p5.shape[-2:], mode='bilinear', align_corners=False))
        return [p3_lane, p4_lane, p5_lane]
