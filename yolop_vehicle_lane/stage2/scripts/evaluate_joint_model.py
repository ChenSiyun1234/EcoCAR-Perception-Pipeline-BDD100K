from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage2.scripts.train_joint_model_experiment import (
    BDDJointCurveDataset,
    build_det_loss,
    collate_fn,
    evaluate,
    extract_tar_once,
    log,
)
from stage2.fusion.experiment_factory import build_joint_model
from stage2.fusion.losses import FusionLaneLoss, FusionLossConfig


def find_checkpoint(run_dir: Path) -> Path:
    names = ['best.pt', 'last.pt', 'checkpoint_best.pt', 'checkpoint_last.pt']
    for name in names:
        hits = list(run_dir.rglob(name))
        if hits:
            return hits[0]
    hits = sorted(run_dir.rglob('*.pt'), key=lambda p: p.stat().st_mtime, reverse=True)
    if not hits:
        raise FileNotFoundError(f'No .pt checkpoint found under {run_dir}')
    return hits[0]


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate a saved Stage 2 joint run on the curve validation split.')
    parser.add_argument('--run-tar', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--curve-tar', required=True)
    parser.add_argument('--curve-root', required=True)
    parser.add_argument('--eval-root', default='/content/stage2_eval_runs')
    parser.add_argument('--limit-val', type=int, default=1000)
    parser.add_argument('--val-batches', type=int, default=None)
    parser.add_argument('--force-extract', action='store_true')
    parser.add_argument('--list-only', action='store_true', help='Resolve and print paths but do not run evaluation. Useful for diagnosing CalledProcessError from a notebook.')
    args = parser.parse_args()

    run_tar = Path(args.run_tar)
    if args.list_only:
        log(f'[list_only] run_tar={run_tar} exists={run_tar.exists()}')
        log(f'[list_only] config={args.config} exists={Path(args.config).exists()}')
        log(f'[list_only] curve_tar={args.curve_tar} exists={Path(args.curve_tar).exists()}')
        log(f'[list_only] curve_root={args.curve_root}')
        log(f'[list_only] eval_root={args.eval_root}')
        return
    if not run_tar.exists():
        raise FileNotFoundError(f'Missing run tar: {run_tar}')
    with open(args.config, 'r', encoding='utf-8') as fh:
        cfg = yaml.safe_load(fh)

    eval_root = Path(args.eval_root)
    eval_root.mkdir(parents=True, exist_ok=True)
    run_dir = eval_root / run_tar.stem
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log(f'[eval] extracting run tar: {run_tar} -> {run_dir}')
    subprocess.check_call(['tar', '-xf', str(run_tar), '-C', str(run_dir)])
    ckpt_path = find_checkpoint(run_dir)
    log(f'[eval] checkpoint={ckpt_path}')

    curve_root = Path(args.curve_root)
    extract_tar_once(Path(args.curve_tar), curve_root, force=args.force_extract)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log(f'[eval] device={device}')
    model = build_joint_model(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get('model', ckpt.get('state_dict', ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    log(f'[eval] loaded missing={len(missing)} unexpected={len(unexpected)} epoch={ckpt.get("epoch", "unknown")}')

    lane_cfg = cfg['loss']['lane']
    lane_loss = FusionLaneLoss(FusionLossConfig(lambda_lane=float(cfg['loss'].get('lambda_lane', 1.0)), use_uncertainty=bool(cfg['loss'].get('use_uncertainty', False)), **lane_cfg)).to(device)
    if 'lane_exist_threshold' in cfg.get('eval', {}):
        lane_loss.cfg.existence_threshold = float(cfg['eval']['lane_exist_threshold'])
    det_loss = build_det_loss(cfg).to(device)

    val_set = BDDJointCurveDataset(
        curve_root,
        'val',
        image_size=tuple(cfg['dataset']['image_size']),
        aux_mask_size=tuple(cfg['dataset']['aux_mask_size']),
        max_lanes=int(cfg['dataset']['max_lanes']),
        num_points=int(cfg['dataset']['num_points']),
        limit=int(args.limit_val),
        require_det_labels=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(cfg['train'].get('batch_size', 8)),
        shuffle=False,
        num_workers=int(cfg['train'].get('workers', 2)),
        pin_memory=True,
        collate_fn=collate_fn,
    )
    max_batches = int(args.val_batches or cfg.get('eval', {}).get('val_batches', 40))
    metrics = evaluate(
        model, lane_loss, det_loss, val_loader, device,
        max_batches=max_batches,
        eval_cfg=cfg.get('eval', {}),
    )
    out_path = run_dir / 'eval_metrics.json'
    out_path.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    log('[eval_summary] ' + ' '.join(f'{k}={v:.5f}' for k, v in sorted(metrics.items()) if any(t in k for t in ['val/det_loss','val/lane_loss','map50','lane_exist','lane_point','matched_line','clrkd_style'])))
    log(f'[eval] metrics_json={out_path}')


if __name__ == '__main__':
    main()
