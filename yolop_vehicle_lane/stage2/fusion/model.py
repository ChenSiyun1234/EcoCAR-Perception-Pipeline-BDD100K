from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .lane_head import CurveLaneHead


class FusionModel(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        feature_channels: Sequence[int],
        detection_head: Optional[nn.Module] = None,
        lane_head: Optional[nn.Module] = None,
        lane_in_indices: Optional[Sequence[int]] = None,
        max_lanes: int = 10,
        num_points: int = 72,
        mask_size: Tuple[int, int] = (72, 128),
    ):
        super().__init__()
        self.backbone = backbone
        self.feature_channels = list(feature_channels)
        self.detection_head = detection_head
        self.lane_in_indices = list(range(len(feature_channels))) if lane_in_indices is None else list(lane_in_indices)
        if lane_head is None:
            lane_channels = [self.feature_channels[i] for i in self.lane_in_indices]
            self.lane_head = CurveLaneHead(in_channels=lane_channels, max_lanes=max_lanes, num_points=num_points, mask_size=mask_size)
        else:
            self.lane_head = lane_head

    def forward(self, x: torch.Tensor) -> Dict[str, object]:
        raw_feats = self.backbone(x)
        aux: Dict[str, object] = {}
        if isinstance(raw_feats, dict):
            shared_feats = raw_feats.get('shared', raw_feats.get('features', raw_feats.get('det')))
            det_feats = raw_feats.get('det', shared_feats)
            lane_source_feats = raw_feats.get('lane', shared_feats)
            aux = {k: v for k, v in raw_feats.items() if k not in {'shared', 'features', 'det', 'lane'}}
        else:
            shared_feats = raw_feats
            det_feats = raw_feats
            lane_source_feats = raw_feats
        if not isinstance(shared_feats, (list, tuple)):
            shared_feats = [shared_feats]
        if not isinstance(det_feats, (list, tuple)):
            det_feats = [det_feats]
        if not isinstance(lane_source_feats, (list, tuple)):
            lane_source_feats = [lane_source_feats]
        shared_feats = list(shared_feats)
        det_feats = list(det_feats)
        lane_source_feats = list(lane_source_feats)
        lane_feats = [lane_source_feats[i] for i in self.lane_in_indices]
        lane_out = self.lane_head(lane_feats)
        det_out = self.detection_head(det_feats) if self.detection_head is not None else None
        return {
            'lane': lane_out,
            'det': det_out,
            'features': shared_feats,
            'det_features': det_feats,
            'lane_features': lane_source_feats,
            'aux': aux,
        }
