from __future__ import annotations

from typing import Dict

from .backbones import HybridRMTCLRKDBackboneNeck, RMTBackboneNeck, RMTGCALaneDetailBackboneNeck, YOLO26BackboneNeck, YOLO26GCABackboneNeck
from .yolo26_inspired import YOLO26InspiredJointBackboneNeck
from .detection import DETRVehicleDetectionHead, SimpleVehicleDetectionHead
from .lane_head import BezierLaneQueryHead, CLRKDLaneHead, CurveLaneHead, HybridPriorQueryHead, LaneQueryHead, LaneQueryHeadAnchorDN
from .model import FusionModel


def build_joint_model(cfg: Dict) -> FusionModel:
    model_cfg = cfg['model']
    backbone_name = str(model_cfg.get('backbone', 'rmt_backbone_neck')).lower()
    width = float(model_cfg.get('width', 0.5))
    if backbone_name in {'rmt_backbone_neck', 'rmt_hgnet_fpn_pan', 'rmt', 'exp2a_rmt_shared'}:
        backbone = RMTBackboneNeck(width=width, use_aifi=bool(model_cfg.get('use_aifi', True)), use_gca=False)
        feature_channels = list(backbone.feature_channels)
    elif backbone_name in {'hybrid_rmt_clrkd', 'hybrid_rmt_clrkd_adapters', 'rmt_clrkd_hybrid', 'exp2b_rmt_gca', 'exp2c_rmt_gca_kd'}:
        backbone = HybridRMTCLRKDBackboneNeck(width=width, use_aifi=bool(model_cfg.get('use_aifi', True)))
        feature_channels = list(backbone.feature_channels)
    elif backbone_name in {'exp2d_rmt_gca_lane_detail', 'exp2e_rmt_gca_lane_matching', 'exp2f_rmt_gca_lane_detail_matching', 'rmt_gca_lane_detail'}:
        backbone = RMTGCALaneDetailBackboneNeck(width=width, use_aifi=bool(model_cfg.get('use_aifi', True)))
        feature_channels = list(backbone.feature_channels)
    elif backbone_name in {'exp3a_yolo26_inspired', 'exp3b_yolo26_inspired_global', 'exp3c_yolo26_inspired_gca', 'yolo26_inspired', 'official_yolo26_inspired'}:
        yi_cfg = model_cfg.get('yolo26_inspired', {})
        use_gca = bool(yi_cfg.get('use_gca', backbone_name in {'exp3c_yolo26_inspired_gca'}))
        use_p5_global = bool(yi_cfg.get('use_p5_global', backbone_name in {'exp3b_yolo26_inspired_global', 'exp3c_yolo26_inspired_gca'}))
        backbone = YOLO26InspiredJointBackboneNeck(
            width=width,
            depth=float(yi_cfg.get('depth', model_cfg.get('depth', 0.33))),
            out_channels=int(yi_cfg.get('out_channels', 128)),
            use_p5_global=use_p5_global,
            use_lane_p3_refine=bool(yi_cfg.get('use_lane_p3_refine', True)),
            use_gca=use_gca,
            use_depthwise=bool(yi_cfg.get('use_depthwise', True)),
        )
        feature_channels = list(backbone.feature_channels)
    elif backbone_name in {'yolo26_backbone_neck', 'yolo26', 'yolo26_real', 'exp3a_yolo26_real'}:
        yolo_cfg = model_cfg.get('yolo26', {})
        backbone = YOLO26BackboneNeck(
            width=width,
            source_root=yolo_cfg.get('source_root'),
            model_path=yolo_cfg.get('model_path'),
            feature_indices=yolo_cfg.get('feature_indices'),
            strict=bool(yolo_cfg.get('strict', True)),
        )
        feature_channels = list(backbone.feature_channels)
    elif backbone_name in {'exp3c_yolo26_gca', 'yolo26_real_gca'}:
        yolo_cfg = model_cfg.get('yolo26', {})
        backbone = YOLO26GCABackboneNeck(
            width=width,
            source_root=yolo_cfg.get('source_root'),
            model_path=yolo_cfg.get('model_path'),
            feature_indices=yolo_cfg.get('feature_indices'),
            strict=bool(yolo_cfg.get('strict', True)),
        )
        feature_channels = list(backbone.feature_channels)
    else:
        raise ValueError(f'Unknown Stage 2 joint backbone: {backbone_name}')

    det_cfg = model_cfg.get('detection_head', {})
    det_type = str(det_cfg.get('type', 'detr')).lower()
    if det_type in {'detr', 'detr_lite', 'rt_detr_lite', 'set_prediction', 'transformer_detr'}:
        detection_head = DETRVehicleDetectionHead(
            in_channels=int(det_cfg.get('in_channels', feature_channels[0])),
            hidden_dim=int(det_cfg.get('embed_dim', 256)),
            num_queries=int(det_cfg.get('num_queries', 100)),
            num_classes=int(det_cfg.get('num_classes', 1)),
            num_decoder_layers=int(det_cfg.get('num_decoder_layers', 3)),
            num_heads=int(det_cfg.get('num_heads', 8)),
            dim_feedforward=int(det_cfg.get('dim_feedforward', 1024)),
        )
    elif det_type in {'grid', 'simple_grid'}:
        detection_head = SimpleVehicleDetectionHead(
            in_channels=int(det_cfg.get('in_channels', feature_channels[0])),
            hidden_dim=int(det_cfg.get('embed_dim', 128)),
            num_classes=int(det_cfg.get('num_classes', 1)),
        )
    else:
        raise ValueError(f'Unknown detection head type: {det_type}')

    lane_cfg = model_cfg.get('lane_head', {})
    lane_head_type = str(lane_cfg.get('type', 'curve')).lower()
    if lane_head_type in {'bezier', 'bezier_query', 'exp2s'}:
        lane_head = BezierLaneQueryHead(
            in_channels=feature_channels,
            embed_dim=int(lane_cfg.get('embed_dim', 128)),
            max_lanes=int(lane_cfg.get('max_lanes', cfg['dataset'].get('max_lanes', 12))),
            num_points=int(lane_cfg.get('num_points', cfg['dataset'].get('num_points', 72))),
            mask_size=tuple(lane_cfg.get('mask_size', cfg['dataset'].get('aux_mask_size', [72, 128]))),
            mask_aux=bool(lane_cfg.get('mask_aux', False)),
            num_lane_classes=int(lane_cfg.get('num_lane_classes', 7)),
            num_queries=int(lane_cfg.get('num_queries', 12)),
            num_decoder_layers=int(lane_cfg.get('num_decoder_layers', 3)),
            num_heads=int(lane_cfg.get('num_heads', 8)),
            dim_feedforward=int(lane_cfg.get('dim_feedforward', 512)),
            dropout=float(lane_cfg.get('dropout', 0.1)),
        )
    elif lane_head_type in {'hybrid', 'hybrid_prior_query', 'exp2q'}:
        lane_head = HybridPriorQueryHead(
            in_channels=feature_channels,
            embed_dim=int(lane_cfg.get('embed_dim', 128)),
            max_lanes=int(lane_cfg.get('max_lanes', cfg['dataset'].get('max_lanes', 12))),
            num_points=int(lane_cfg.get('num_points', cfg['dataset'].get('num_points', 72))),
            mask_size=tuple(lane_cfg.get('mask_size', cfg['dataset'].get('aux_mask_size', [72, 128]))),
            mask_aux=bool(lane_cfg.get('mask_aux', False)),
            num_lane_classes=int(lane_cfg.get('num_lane_classes', 7)),
            num_priors=int(lane_cfg.get('num_priors', 192)),
            sample_points=int(lane_cfg.get('sample_points', 36)),
            roi_refine_layers=int(lane_cfg.get('roi_refine_layers', 3)),
            roi_mid_channels=int(lane_cfg.get('roi_mid_channels', 48)),
            cross_attn_resize=tuple(lane_cfg.get('cross_attn_resize', [10, 25])),
            num_queries=int(lane_cfg.get('num_queries', 12)),
            refiner_layers=int(lane_cfg.get('refiner_layers', 2)),
            refiner_heads=int(lane_cfg.get('refiner_heads', 8)),
            refiner_ff_dim=int(lane_cfg.get('refiner_ff_dim', 512)),
            refiner_dropout=float(lane_cfg.get('refiner_dropout', 0.1)),
            refiner_attn_to_spatial=bool(lane_cfg.get('refiner_attn_to_spatial', True)),
        )
    elif lane_head_type in {'query_anchor', 'lane_query_anchor', 'dab_detr', 'exp2r'}:
        lane_head = LaneQueryHeadAnchorDN(
            in_channels=feature_channels,
            embed_dim=int(lane_cfg.get('embed_dim', 128)),
            max_lanes=int(lane_cfg.get('max_lanes', cfg['dataset'].get('max_lanes', 12))),
            num_points=int(lane_cfg.get('num_points', cfg['dataset'].get('num_points', 72))),
            mask_size=tuple(lane_cfg.get('mask_size', cfg['dataset'].get('aux_mask_size', [72, 128]))),
            mask_aux=bool(lane_cfg.get('mask_aux', False)),
            num_lane_classes=int(lane_cfg.get('num_lane_classes', 7)),
            num_queries=int(lane_cfg.get('num_queries', 12)),
            num_decoder_layers=int(lane_cfg.get('num_decoder_layers', 3)),
            num_heads=int(lane_cfg.get('num_heads', 8)),
            dim_feedforward=int(lane_cfg.get('dim_feedforward', 512)),
            dropout=float(lane_cfg.get('dropout', 0.1)),
            dn_num_groups=int(lane_cfg.get('dn_num_groups', 0)),
            dn_noise_std_xy=float(lane_cfg.get('dn_noise_std_xy', 0.05)),
            dn_noise_std_theta=float(lane_cfg.get('dn_noise_std_theta', 0.04)),
        )
    elif lane_head_type in {'query', 'lane_query', 'detr', 'exp2p'}:
        lane_head = LaneQueryHead(
            in_channels=feature_channels,
            embed_dim=int(lane_cfg.get('embed_dim', 128)),
            max_lanes=int(lane_cfg.get('max_lanes', cfg['dataset'].get('max_lanes', 12))),
            num_points=int(lane_cfg.get('num_points', cfg['dataset'].get('num_points', 72))),
            mask_size=tuple(lane_cfg.get('mask_size', cfg['dataset'].get('aux_mask_size', [72, 128]))),
            mask_aux=bool(lane_cfg.get('mask_aux', False)),
            num_lane_classes=int(lane_cfg.get('num_lane_classes', 7)),
            num_queries=int(lane_cfg.get('num_queries', 12)),
            num_decoder_layers=int(lane_cfg.get('num_decoder_layers', 3)),
            num_heads=int(lane_cfg.get('num_heads', 8)),
            dim_feedforward=int(lane_cfg.get('dim_feedforward', 512)),
            dropout=float(lane_cfg.get('dropout', 0.1)),
        )
    elif lane_head_type in {'clrkd', 'clrkd_lane_head', 'clrkd_roi_gather', 'exp2g', 'exp2h', 'exp2i', 'exp2j'}:
        lane_head = CLRKDLaneHead(
            in_channels=feature_channels,
            embed_dim=int(lane_cfg.get('embed_dim', 128)),
            max_lanes=int(lane_cfg.get('max_lanes', cfg['dataset'].get('max_lanes', 10))),
            num_points=int(lane_cfg.get('num_points', cfg['dataset'].get('num_points', 72))),
            mask_size=tuple(lane_cfg.get('mask_size', cfg['dataset'].get('aux_mask_size', [72, 128]))),
            mask_aux=bool(lane_cfg.get('mask_aux', True)),
            num_lane_classes=int(lane_cfg.get('num_lane_classes', 7)),
            num_priors=int(lane_cfg.get('num_priors', 192)),
            sample_points=int(lane_cfg.get('sample_points', 36)),
            roi_refine_layers=int(lane_cfg.get('roi_refine_layers', 3)),
            roi_mid_channels=int(lane_cfg.get('roi_mid_channels', 48)),
            cross_attn_resize=tuple(lane_cfg.get('cross_attn_resize', [10, 25])),
            cls_uses_prior_embedding=bool(lane_cfg.get('cls_uses_prior_embedding', False)),
            prior_embed_encoder_dim=int(lane_cfg.get('prior_embed_encoder_dim', 0)),
            cls_separate_path=bool(lane_cfg.get('cls_separate_path', False)),
            dual_score=bool(lane_cfg.get('dual_score', False)),
        )
    else:
        lane_head = CurveLaneHead(
            in_channels=feature_channels,
            embed_dim=int(lane_cfg.get('embed_dim', 128)),
            max_lanes=int(lane_cfg.get('max_lanes', cfg['dataset'].get('max_lanes', 10))),
            num_points=int(lane_cfg.get('num_points', cfg['dataset'].get('num_points', 72))),
            mask_size=tuple(lane_cfg.get('mask_size', cfg['dataset'].get('aux_mask_size', [72, 128]))),
            mask_aux=bool(lane_cfg.get('mask_aux', True)),
            num_lane_classes=int(lane_cfg.get('num_lane_classes', 7)),
            num_priors=int(lane_cfg.get('num_priors', 192)),
            output_all_priors=bool(lane_cfg.get('output_all_priors', False)),
            use_roi_gather=bool(lane_cfg.get('use_roi_gather', False)),
            roi_refine_layers=int(lane_cfg.get('roi_refine_layers', 1)),
        )
    return FusionModel(
        backbone=backbone,
        feature_channels=feature_channels,
        detection_head=detection_head,
        lane_head=lane_head,
        lane_in_indices=list(range(len(feature_channels))),
        max_lanes=int(lane_cfg.get('max_lanes', cfg['dataset'].get('max_lanes', 10))),
        num_points=int(lane_cfg.get('num_points', cfg['dataset'].get('num_points', 72))),
        mask_size=tuple(lane_cfg.get('mask_size', cfg['dataset'].get('aux_mask_size', [72, 128]))),
    )
