"""
DETR-style lane query head.

[SOURCE] Vendored from DETR_GeoLane_pipeline/src/lane_head.py
(plus the `build_2d_sincos_pos_embed` helper from detection_head.py).

Interface:
    head = LaneSetHead(d_model=..., num_queries=..., num_points=...)
    out  = head([P3, P4, P5])     # list of feature maps (B, C, H, W)
    out = {
        'exist_logits': [B, Q, 1],
        'pred_points' : [B, Q, P, 2],
        'vis_logits'  : [B, Q, P],
        'type_logits' : [B, Q, C],
        'aux_outputs' : list of dicts with the same keys (per-layer auxiliaries)
    }

Design principle: the head consumes exactly the feature maps the
YOLOPv2 detection head consumes (P3 / P4 / P5 from the PAN), so
swapping lane heads does not require encoder / neck surgery.
"""

from typing import Dict, List

import torch
import torch.nn as nn


def build_2d_sincos_pos_embed(h: int, w: int, dim: int, device, dtype) -> torch.Tensor:
    if dim % 4 != 0:
        raise ValueError(f"d_model must be divisible by 4, got {dim}")
    y, x = torch.meshgrid(
        torch.linspace(0, 1, h, device=device, dtype=dtype),
        torch.linspace(0, 1, w, device=device, dtype=dtype),
        indexing="ij",
    )
    omega = torch.arange(dim // 4, device=device, dtype=dtype)
    omega = 1.0 / (10000 ** (omega / max(dim // 4 - 1, 1)))
    x = x.reshape(-1, 1) * omega.reshape(1, -1)
    y = y.reshape(-1, 1) * omega.reshape(1, -1)
    return torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], dim=1)


class LaneDecoderLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model), nn.Dropout(dropout),
        )

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        q = self.norm1(queries)
        queries = queries + self.self_attn(q, q, q)[0]
        q = self.norm2(queries)
        queries = queries + self.cross_attn(q, memory, memory)[0]
        queries = queries + self.ffn(self.norm3(queries))
        return queries


class LaneSetHead(nn.Module):
    def __init__(self,
                 in_channels: List[int],
                 num_lane_types: int = 7,
                 num_points: int = 72,
                 d_model: int = 256,
                 nhead: int = 8,
                 ffn_dim: int = 1024,
                 num_layers: int = 3,
                 num_queries: int = 10,
                 dropout: float = 0.0):
        super().__init__()
        self.num_queries = num_queries
        self.num_points = num_points
        self.d_model = d_model

        # Per-scale projection to a common dim (YOLOPv2's PAN outputs are
        # 128 / 256 / 512 ch for P3/P4/P5, so we 1x1 project them all to d_model).
        self.input_projs = nn.ModuleList(
            nn.Conv2d(c, d_model, kernel_size=1) if c != d_model else nn.Identity()
            for c in in_channels
        )

        self.query_embed = nn.Embedding(num_queries, d_model)
        self.query_pos = nn.Embedding(num_queries, d_model)
        self.layers = nn.ModuleList([
            LaneDecoderLayer(d_model, nhead, ffn_dim, dropout) for _ in range(num_layers)
        ])
        self.exist_heads = nn.ModuleList([nn.Linear(d_model, 1) for _ in range(num_layers)])
        self.point_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(),
                nn.Linear(d_model, d_model), nn.ReLU(),
                nn.Linear(d_model, num_points * 2),
            ) for _ in range(num_layers)
        ])
        self.vis_heads = nn.ModuleList([nn.Linear(d_model, num_points) for _ in range(num_layers)])
        self.type_heads = nn.ModuleList([nn.Linear(d_model, num_lane_types) for _ in range(num_layers)])

        # Lane priors — vertical center lines equally spaced in X.
        y = torch.linspace(0.15, 0.95, num_points)
        centers = torch.linspace(0.15, 0.85, num_queries)
        priors = []
        for cx in centers:
            x = torch.full_like(y, cx)
            priors.append(torch.stack([x, y], dim=-1))
        self.lane_priors = nn.Parameter(torch.stack(priors, dim=0))
        self._init_weights()

    def _init_weights(self):
        for head in self.exist_heads:
            nn.init.constant_(head.bias, -2.5)      # start with "not a lane"
        for mod in self.point_heads:
            for layer in mod:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0.0)

    def _flatten_features(self, features: List[torch.Tensor]) -> torch.Tensor:
        tokens = []
        for feat, proj in zip(features, self.input_projs):
            feat = proj(feat)
            b, c, h, w = feat.shape
            pos = build_2d_sincos_pos_embed(h, w, c, feat.device, feat.dtype)
            pos = pos.unsqueeze(0).expand(b, -1, -1)
            tok = feat.flatten(2).permute(0, 2, 1) + pos
            tokens.append(tok)
        return torch.cat(tokens, dim=1)

    def forward(self, features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        b = features[0].shape[0]
        memory = self._flatten_features(features)
        queries = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)
        queries = queries + self.query_pos.weight.unsqueeze(0).expand(b, -1, -1)
        priors = self.lane_priors.unsqueeze(0).expand(b, -1, -1, -1)

        aux_outputs = []
        exist_logits = pred_points = vis_logits = type_logits = None
        ref_points = priors
        for layer, eh, ph, vh, th in zip(
            self.layers, self.exist_heads, self.point_heads,
            self.vis_heads, self.type_heads
        ):
            queries = layer(queries, memory)
            exist_logits = eh(queries)
            point_delta = ph(queries).view(b, self.num_queries, self.num_points, 2)
            pred_points = (ref_points + 0.15 * torch.tanh(point_delta)).clamp(0.0, 1.0)
            vis_logits = vh(queries)
            type_logits = th(queries)
            ref_points = pred_points.detach()
            aux_outputs.append({
                'exist_logits': exist_logits,
                'pred_points': pred_points,
                'vis_logits': vis_logits,
                'type_logits': type_logits,
            })

        return {
            'exist_logits': exist_logits,
            'pred_points': pred_points,
            'vis_logits': vis_logits,
            'type_logits': type_logits,
            'query_features': queries,
            'aux_outputs': aux_outputs[:-1],   # exclude the final layer (already primary)
        }


class TaskFeatureAdapter(nn.Module):
    """Light residual adapter per feature scale, one stack for each task.
    [SOURCE] DETR_GeoLane_pipeline/src/model.py::TaskFeatureAdapter.
    Used to reduce gradient conflict between detection and lane heads
    (see the multi-task gradient-conflict literature; PCGrad, CAGrad).
    """

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
