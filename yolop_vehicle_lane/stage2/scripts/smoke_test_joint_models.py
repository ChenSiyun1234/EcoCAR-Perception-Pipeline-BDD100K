from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage2.fusion.detection import DETRVehicleDetectionLoss, DetectionLossConfig
from stage2.fusion.experiment_factory import build_joint_model
from stage2.fusion.losses import FusionLaneLoss, FusionLossConfig, compute_grad_cosine, compute_grad_norm_ratio


def make_small_cfg(path: Path) -> Dict:
    cfg = yaml.safe_load(path.read_text(encoding='utf-8'))
    cfg['model']['width'] = 0.25
    cfg['model']['detection_head']['num_queries'] = 4
    cfg['model']['detection_head']['embed_dim'] = 32
    cfg['model']['detection_head']['num_heads'] = 4
    cfg['model']['detection_head']['dim_feedforward'] = 64
    cfg['model']['detection_head']['num_decoder_layers'] = 1
    cfg['model']['lane_head']['embed_dim'] = 32
    cfg['model']['lane_head']['num_priors'] = 16
    cfg['dataset']['max_lanes'] = 2
    cfg['model']['lane_head']['max_lanes'] = 2
    return cfg


def run_one(path: Path) -> None:
    torch.set_num_threads(2)
    cfg = make_small_cfg(path)
    model = build_joint_model(cfg)
    image = torch.randn(1, 3, 64, 96)
    out = model(image)
    # Read mask_size from config so the smoke target matches the configured
    # auxiliary mask resolution. Bug fix for Exp2AA (NB32) which used
    # mask_size=(90,160) but the smoke target was hardcoded (72,128) and
    # produced "Target size must be the same as input size" at the BCE call.
    mask_h, mask_w = cfg['dataset'].get('aux_mask_size', [72, 128])
    lane_target = {
        'existence': torch.zeros(1, 2),
        'points': torch.zeros(1, 2, 72, 2),
        'visibility': torch.zeros(1, 2, 72),
        'mask_target': torch.zeros(1, 1, int(mask_h), int(mask_w)),
    }
    lane_target['existence'][:, 0] = 1.0
    lane_target['points'][:, 0, :, 0] = 0.5
    lane_target['points'][:, 0, :, 1] = torch.linspace(1.0, 0.0, 72)
    lane_target['visibility'][:, 0, :] = 1.0
    lane_loss_fn = FusionLaneLoss(FusionLossConfig(**cfg['loss']['lane']))
    det_loss_fn = DETRVehicleDetectionLoss(DetectionLossConfig(**cfg['loss']['det']))
    lane_loss, lane_comp = lane_loss_fn(out['lane'], lane_target)
    det_loss, det_comp = det_loss_fn(out['det'], [torch.tensor([[0, 0.5, 0.5, 0.2, 0.2]], dtype=torch.float32)])
    grad_cos = compute_grad_cosine(det_loss, lane_loss, model.backbone.parameters())
    grad_ratio = compute_grad_norm_ratio(det_loss, lane_loss, model.backbone.parameters())
    total = det_loss + grad_ratio * lane_loss
    total.backward()
    print(f'OK {path.name}')
    print(f"  lane_shape={tuple(out['lane']['coord_pred'].shape)} det_shape={tuple(out['det']['box_pred'].shape)}")
    print(f"  lane_loss={float(lane_loss.detach()):.4f} det_loss={float(det_loss.detach()):.4f} grad_cos={grad_cos:.4f} lambda_lane={grad_ratio:.4f}")
    if 'aux' in out and isinstance(out['aux'], dict) and 'gate_stats' in out['aux']:
        stats = {k: float(v.detach()) for k, v in out['aux']['gate_stats'].items()}
        print(f'  gate_stats={stats}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('configs', nargs='+')
    args = parser.parse_args()
    for item in args.configs:
        run_one(Path(item))


if __name__ == '__main__':
    main()
