"""Stage 2 fusion package: detection + CLRKD-style lane.

Submodules:
    lane_targets : BDD100K JSON -> fixed-size per-image curve targets.
    losses       : multi-task loss for detection + curve lane.
    lane_head    : CLRKD-style curve lane head.
    model        : minimal detection + lane fusion model wrapper.
"""

from .lane_targets import (
    LANE_TRAIN_CATS,
    LANE_CAT_TO_ID,
    extract_lane_labels,
    frame_to_lane_targets,
    soft_polyline_mask_numpy,
    LaneLabelCache,
)
from .losses import (
    FusionLossConfig,
    FusionLaneLoss,
    UncertaintyMultiTaskLoss,
    compute_grad_cosine,
    compute_grad_norm_ratio,
)

__all__ = [
    'LANE_TRAIN_CATS',
    'LANE_CAT_TO_ID',
    'extract_lane_labels',
    'frame_to_lane_targets',
    'soft_polyline_mask_numpy',
    'LaneLabelCache',
    'FusionLossConfig',
    'FusionLaneLoss',
    'UncertaintyMultiTaskLoss',
    'compute_grad_cosine',
    'compute_grad_norm_ratio',
    'DetectionLossConfig',
    'SimpleVehicleDetectionHead',
    'SimpleVehicleDetectionLoss',
    'read_yolo_label',
]

from .detection import DetectionLossConfig, SimpleVehicleDetectionHead, SimpleVehicleDetectionLoss, read_yolo_label
from .experiment_factory import build_joint_model
from .detection import DETRVehicleDetectionHead, DETRVehicleDetectionLoss

from .yolo26_inspired import YOLO26InspiredJointBackboneNeck
