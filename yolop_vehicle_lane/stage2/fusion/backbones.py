from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, groups: int = 1, act: str = 'silu'):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, padding=k // 2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'gelu':
            self.act = nn.GELU()
        else:
            self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DepthwiseConvBNAct(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 3, s: int = 1, act: str = 'silu'):
        super().__init__()
        self.dw = ConvBNAct(c1, c1, k, s, groups=c1, act=act)
        self.pw = ConvBNAct(c1, c2, 1, 1, act=act)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pw(self.dw(x))


class HGStem(nn.Module):
    def __init__(self, c1: int = 3, cm: int = 32, c2: int = 64):
        super().__init__()
        self.stem1 = ConvBNAct(c1, cm, 3, 2, act='relu')
        self.stem2 = ConvBNAct(cm, cm, 3, 1, act='relu')
        self.stem3 = ConvBNAct(cm, c2, 3, 2, act='relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem3(self.stem2(self.stem1(x)))


class HGBlock(nn.Module):
    def __init__(self, c1: int, cm: int, c2: int, repeats: int = 4, light: bool = True, shortcut: bool = False):
        super().__init__()
        block = DepthwiseConvBNAct if light else ConvBNAct
        self.blocks = nn.ModuleList()
        for i in range(repeats):
            self.blocks.append(block(c1 if i == 0 else cm, cm, 3, 1, act='relu'))
        self.squeeze = ConvBNAct(c1 + repeats * cm, max(8, c2 // 2), 1, 1, act='relu')
        self.expand = ConvBNAct(max(8, c2 // 2), c2, 1, 1, act='relu')
        self.shortcut = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [x]
        y = x
        for block in self.blocks:
            y = block(y)
            outs.append(y)
        y = self.expand(self.squeeze(torch.cat(outs, dim=1)))
        if self.shortcut:
            y = y + x
        return y


class AIFI(nn.Module):
    def __init__(self, channels: int = 256, num_heads: int = 8, ffn_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * ffn_ratio),
            nn.GELU(),
            nn.Linear(channels * ffn_ratio, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        z = self.norm1(tokens)
        attn, _ = self.attn(z, z, z, need_weights=False)
        tokens = tokens + attn
        tokens = tokens + self.ffn(self.norm2(tokens))
        return tokens.transpose(1, 2).reshape(b, c, h, w)


class CSPRepBlock(nn.Module):
    def __init__(self, c1: int, c2: int, repeats: int = 2):
        super().__init__()
        hidden = max(32, c2 // 2)
        self.cv1 = ConvBNAct(c1, hidden, 1, 1)
        self.cv2 = ConvBNAct(c1, hidden, 1, 1)
        blocks = []
        for _ in range(repeats):
            blocks.append(nn.Sequential(ConvBNAct(hidden, hidden, 3, 1), ConvBNAct(hidden, hidden, 3, 1)))
        self.blocks = nn.ModuleList(blocks)
        self.cv3 = ConvBNAct(hidden * 2, c2, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y1 = self.cv1(x)
        for block in self.blocks:
            y1 = y1 + block(y1)
        y2 = self.cv2(x)
        return self.cv3(torch.cat([y1, y2], dim=1))


class TaskAdapter(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.refine = nn.Sequential(
            ConvBNAct(channels, channels, 3, 1),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.refine(x)


class GateControlAdapter(nn.Module):
    def __init__(self, channels: int, reduction: int = 8, clamp_min: float = 0.05, clamp_max: float = 0.95):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )
        self.alpha_net = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1, groups=channels),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, shared: torch.Tensor, task: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([shared, task], dim=1)
        c_gate = self.channel_att(x)
        s_gate = self.spatial_att(x)
        alpha = self.alpha_net(x)
        gate = alpha * c_gate + (1.0 - alpha) * s_gate
        gate = gate.clamp(self.clamp_min, self.clamp_max)
        out = shared + gate * (task - shared)
        return out, gate


class RMTBackboneNeck(nn.Module):
    feature_channels = [128, 128, 128]

    def __init__(self, width: float = 0.5, use_aifi: bool = True, use_gca: bool = False):
        super().__init__()
        c2 = max(32, int(96 * width))
        c3 = max(64, int(192 * width))
        c4 = max(128, int(384 * width))
        c5 = max(256, int(768 * width))
        out_c = 128
        self.c2_channels = c2
        self.c3_channels = c3
        self.c4_channels = c4
        self.c5_channels = c5
        self.out_channels = out_c
        self.use_gca = bool(use_gca)
        self.stem = HGStem(3, max(24, c2 // 2), c2)
        self.stage2 = nn.Sequential(HGBlock(c2, max(24, c2 // 2), c2, repeats=3, light=False), ConvBNAct(c2, c3, 3, 2, act='relu'))
        self.stage3 = nn.Sequential(HGBlock(c3, max(48, c3 // 2), c4, repeats=4, light=True), ConvBNAct(c4, c4, 3, 2, act='relu'))
        self.stage4 = nn.Sequential(HGBlock(c4, max(96, c4 // 2), c5, repeats=4, light=True), ConvBNAct(c5, c5, 3, 2, act='relu'))
        self.p3_in = ConvBNAct(c3, out_c, 1, 1)
        self.p4_in = ConvBNAct(c4, out_c, 1, 1)
        self.p5_in = ConvBNAct(c5, out_c, 1, 1)
        self.aifi = AIFI(out_c, 4) if use_aifi else nn.Identity()
        self.fpn4 = CSPRepBlock(out_c * 2, out_c, repeats=1)
        self.fpn3 = CSPRepBlock(out_c * 2, out_c, repeats=1)
        self.down3 = ConvBNAct(out_c, out_c, 3, 2)
        self.pan4 = CSPRepBlock(out_c * 2, out_c, repeats=1)
        self.down4 = ConvBNAct(out_c, out_c, 3, 2)
        self.pan5 = CSPRepBlock(out_c * 2, out_c, repeats=1)
        self.det_adapters = nn.ModuleList([TaskAdapter(out_c) for _ in range(3)])
        self.lane_adapters = nn.ModuleList([TaskAdapter(out_c) for _ in range(3)])
        self.gate_det = nn.ModuleList([GateControlAdapter(out_c) for _ in range(3)])
        self.gate_lane = nn.ModuleList([GateControlAdapter(out_c) for _ in range(3)])

    def _shared_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        c3 = self.stage2(x)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)
        p3 = self.p3_in(c3)
        p4 = self.p4_in(c4)
        p5 = self.aifi(self.p5_in(c5))
        p4_td = self.fpn4(torch.cat([F.interpolate(p5, size=p4.shape[-2:], mode='nearest'), p4], dim=1))
        p3_td = self.fpn3(torch.cat([F.interpolate(p4_td, size=p3.shape[-2:], mode='nearest'), p3], dim=1))
        p4_out = self.pan4(torch.cat([self.down3(p3_td), p4_td], dim=1))
        p5_out = self.pan5(torch.cat([self.down4(p4_out), p5], dim=1))
        return [p3_td, p4_out, p5_out]

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


class RMTGCALaneDetailBackboneNeck(RMTBackboneNeck):
    feature_channels = [128, 128, 128]

    def __init__(self, width: float = 0.5, use_aifi: bool = True):
        super().__init__(width=width, use_aifi=use_aifi, use_gca=True)
        from .lane_detail_neck import LaneDetailNeck
        self.lane_detail_neck = LaneDetailNeck(self.c2_channels, self.out_channels)

    def _shared_features_with_c2(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        c2 = self.stem(x)
        c3 = self.stage2(c2)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)
        p3 = self.p3_in(c3)
        p4 = self.p4_in(c4)
        p5 = self.aifi(self.p5_in(c5))
        p4_td = self.fpn4(torch.cat([F.interpolate(p5, size=p4.shape[-2:], mode='nearest'), p4], dim=1))
        p3_td = self.fpn3(torch.cat([F.interpolate(p4_td, size=p3.shape[-2:], mode='nearest'), p3], dim=1))
        p4_out = self.pan4(torch.cat([self.down3(p3_td), p4_td], dim=1))
        p5_out = self.pan5(torch.cat([self.down4(p4_out), p5], dim=1))
        return c2, [p3_td, p4_out, p5_out]

    def forward(self, x: torch.Tensor):
        c2, shared = self._shared_features_with_c2(x)
        lane_detail = self.lane_detail_neck(c2, shared)
        det_feats: List[torch.Tensor] = []
        lane_feats: List[torch.Tensor] = []
        det_gates: List[torch.Tensor] = []
        lane_gates: List[torch.Tensor] = []
        for i, feat in enumerate(shared):
            det_task = self.det_adapters[i](feat)
            det_feat, det_gate = self.gate_det[i](feat, det_task)
            lane_feat, lane_gate = self.gate_lane[i](feat, lane_detail[i])
            det_feats.append(det_feat)
            lane_feats.append(lane_feat)
            det_gates.append(det_gate)
            lane_gates.append(lane_gate)
        names = ['p3', 'p4', 'p5']
        gate_stats = {
            'gate/det_mean': torch.stack([g.mean() for g in det_gates]).mean(),
            'gate/lane_mean': torch.stack([g.mean() for g in lane_gates]).mean(),
            'gate/det_sat_low': torch.stack([(g <= 0.051).float().mean() for g in det_gates]).mean(),
            'gate/det_sat_high': torch.stack([(g >= 0.949).float().mean() for g in det_gates]).mean(),
            'gate/lane_sat_low': torch.stack([(g <= 0.051).float().mean() for g in lane_gates]).mean(),
            'gate/lane_sat_high': torch.stack([(g >= 0.949).float().mean() for g in lane_gates]).mean(),
        }
        for name, det_gate, lane_gate in zip(names, det_gates, lane_gates):
            gate_stats[f'gate/{name}_det_mean'] = det_gate.mean()
            gate_stats[f'gate/{name}_lane_mean'] = lane_gate.mean()
            gate_stats[f'gate/{name}_det_sat_low'] = (det_gate <= 0.051).float().mean()
            gate_stats[f'gate/{name}_det_sat_high'] = (det_gate >= 0.949).float().mean()
            gate_stats[f'gate/{name}_lane_sat_low'] = (lane_gate <= 0.051).float().mean()
            gate_stats[f'gate/{name}_lane_sat_high'] = (lane_gate >= 0.949).float().mean()
        return {
            'shared': shared,
            'det': det_feats,
            'lane': lane_feats,
            'lane_detail': lane_detail,
            'gates': {'det': det_gates, 'lane': lane_gates},
            'gate_stats': gate_stats,
        }


class HybridRMTCLRKDBackboneNeck(RMTBackboneNeck):
    def __init__(self, width: float = 0.5, use_aifi: bool = True):
        super().__init__(width=width, use_aifi=use_aifi, use_gca=True)


class YOLO26BackboneNeck(nn.Module):
    feature_channels = [128, 128, 128]

    def __init__(self, width: float = 0.5, source_root: Optional[str] = None, model_path: Optional[str] = None, feature_indices: Optional[Sequence[int]] = None, strict: bool = True):
        super().__init__()
        self.strict = bool(strict)
        self.source_root = source_root or os.environ.get('YOLO26_ROOT', '')
        self.model_path = model_path or os.environ.get('YOLO26_MODEL', '')
        self.feature_indices = list(feature_indices or [])
        self.inner: Optional[nn.Module] = None
        self.hook_outputs: List[torch.Tensor] = []
        self.loaded_from = ''
        self.adapters = nn.ModuleList([nn.Sequential(nn.LazyConv2d(128, 1, bias=False), nn.BatchNorm2d(128), nn.SiLU(inplace=True)) for _ in range(3)])
        self._load_real_yolo26()

    def _candidate_roots(self) -> List[Path]:
        roots: List[Path] = []
        if self.source_root:
            roots.append(Path(self.source_root))
        for env in ['CLAUDE_WORKSPACE', 'PROJECT_ROOT', 'ECOCAR_WORKSPACE']:
            val = os.environ.get(env)
            if val:
                roots.append(Path(val))
        roots.extend([Path('/content/drive/MyDrive/EcoCAR'), Path('/content/drive/MyDrive'), Path.cwd()])
        unique: List[Path] = []
        for root in roots:
            if root not in unique:
                unique.append(root)
        return unique

    def _find_model_path(self) -> Optional[Path]:
        if self.model_path and Path(self.model_path).exists():
            return Path(self.model_path)
        patterns = ['**/yolo26*.pt', '**/yolo26*.yaml', '**/*yolo26*.pt', '**/*yolo26*.yaml']
        for root in self._candidate_roots():
            if not root.exists():
                continue
            for pattern in patterns:
                hits = sorted(root.glob(pattern))
                if hits:
                    return hits[0]
        return None

    def _load_real_yolo26(self) -> None:
        path = self._find_model_path()
        if path is None:
            msg = 'YOLO26 real model path not found. Set YOLO26_ROOT or YOLO26_MODEL to a top-level YOLO26 project/model file.'
            if self.strict:
                raise FileNotFoundError(msg)
            return
        try:
            from ultralytics import YOLO
            yolo = YOLO(str(path))
            self.inner = yolo.model
            self.loaded_from = str(path)
        except Exception as exc:
            if self.strict:
                raise RuntimeError(f'Failed to load real YOLO26 model from {path}: {exc}') from exc
            self.inner = None
            return
        modules = list(self.inner.modules())
        if not self.feature_indices:
            self.feature_indices = self._infer_feature_indices(modules)
        self._register_feature_hooks(modules)

    def _infer_feature_indices(self, modules: Sequence[nn.Module]) -> List[int]:
        conv_like = []
        for idx, module in enumerate(modules):
            if isinstance(module, (nn.Conv2d, nn.Sequential, nn.ModuleList)):
                conv_like.append(idx)
        if len(conv_like) < 3:
            return [len(modules) - 3, len(modules) - 2, len(modules) - 1]
        return conv_like[-3:]

    def _register_feature_hooks(self, modules: Sequence[nn.Module]) -> None:
        for idx in self.feature_indices:
            if idx < 0:
                idx = len(modules) + idx
            if idx < 0 or idx >= len(modules):
                raise IndexError(f'YOLO26 feature index {idx} is outside model module range 0..{len(modules)-1}')
            modules[idx].register_forward_hook(self._save_hook)

    def _save_hook(self, _module: nn.Module, _inp, output) -> None:
        if torch.is_tensor(output) and output.dim() == 4:
            self.hook_outputs.append(output)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        if self.inner is None:
            raise RuntimeError('YOLO26BackboneNeck is in strict real-import mode but no YOLO26 model was loaded.')
        self.hook_outputs = []
        _ = self.inner(x)
        feats = [f for f in self.hook_outputs if torch.is_tensor(f) and f.dim() == 4]
        if len(feats) < 3:
            raise RuntimeError(f'Expected at least 3 YOLO26 feature maps from hooks, got {len(feats)}. Configure model.feature_indices explicitly.')
        feats = sorted(feats[-3:], key=lambda t: t.shape[-2] * t.shape[-1], reverse=True)
        return [adapter(feat) for adapter, feat in zip(self.adapters, feats)]


class YOLO26GCABackboneNeck(YOLO26BackboneNeck):
    def __init__(self, width: float = 0.5, source_root: Optional[str] = None, model_path: Optional[str] = None, feature_indices: Optional[Sequence[int]] = None, strict: bool = True):
        super().__init__(width=width, source_root=source_root, model_path=model_path, feature_indices=feature_indices, strict=strict)
        self.det_adapters = nn.ModuleList([TaskAdapter(128) for _ in range(3)])
        self.lane_adapters = nn.ModuleList([TaskAdapter(128) for _ in range(3)])
        self.gate_det = nn.ModuleList([GateControlAdapter(128) for _ in range(3)])
        self.gate_lane = nn.ModuleList([GateControlAdapter(128) for _ in range(3)])

    def forward(self, x: torch.Tensor):
        shared = super().forward(x)
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
