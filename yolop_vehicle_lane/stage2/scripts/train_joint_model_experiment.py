from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader, Dataset

torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

def log(message: object) -> None:
    print(message, flush=True)


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage2.fusion.detection import DETRVehicleDetectionLoss, DetectionLossConfig, SimpleVehicleDetectionLoss, read_yolo_label
from stage2.fusion.experiment_factory import build_joint_model
from stage2.fusion.lane_targets import soft_polyline_mask_numpy
from stage2.fusion.losses import FusionLaneLoss, FusionLossConfig, UncertaintyMultiTaskLoss, compute_grad_cosine, compute_grad_norm_ratio, compute_lane_eval_metrics
from stage2.metrics.original_metric_adapters import CLRKDStyleLaneMetric, VehicleDetectionMetric, metric_source_report
from stage2.metrics.lane_f1_decoded import LaneF1DecodedMetric



def set_requires_grad(module: torch.nn.Module, value: bool) -> None:
    for param in module.parameters():
        param.requires_grad = value


def apply_training_phase(model: torch.nn.Module, phase: str) -> None:
    phase = str(phase).lower()
    if phase in {'head_warmup', 'warmup_heads', 'phase1'}:
        set_requires_grad(model.backbone, False)
        if hasattr(model, 'detection_head') and model.detection_head is not None:
            set_requires_grad(model.detection_head, True)
        if hasattr(model, 'lane_head') and model.lane_head is not None:
            set_requires_grad(model.lane_head, True)
    elif phase in {'adapter_warmup', 'neck_adapter', 'phase2'}:
        set_requires_grad(model, True)
        for name, param in model.backbone.named_parameters():
            if not any(token in name for token in ['fpn', 'pan', 'down', 'adapter', 'gate', 'p3_in', 'p4_in', 'p5_in', 'aifi']):
                param.requires_grad = False
    else:
        set_requires_grad(model, True)


def phase_for_epoch(cfg: Dict, epoch: int) -> str:
    phases = cfg.get('train', {}).get('phases', [])
    if not phases:
        return 'full_finetune'
    selected = phases[-1].get('name', 'full_finetune')
    for item in phases:
        if epoch <= int(item.get('until_epoch', epoch)):
            selected = item.get('name', selected)
            break
    return str(selected)


def build_optimizer(model: torch.nn.Module, lane_loss: torch.nn.Module, mtl: Optional[torch.nn.Module], cfg: Dict):
    lr0 = float(cfg['train']['lr0'])
    weight_decay = float(cfg['train']['weight_decay'])
    groups = []
    groups.append({'params': list(model.backbone.parameters()), 'lr': lr0 * float(cfg['train'].get('backbone_lr_mult', 0.1)), 'weight_decay': weight_decay})
    head_params = []
    if model.detection_head is not None:
        head_params += list(model.detection_head.parameters())
    head_params += list(model.lane_head.parameters())
    head_params += list(lane_loss.parameters())
    if mtl is not None:
        head_params += list(mtl.parameters())
    groups.append({'params': head_params, 'lr': lr0 * float(cfg['train'].get('head_lr_mult', 2.0)), 'weight_decay': weight_decay})
    return optim.AdamW(groups)


def load_lane_teacher(model: torch.nn.Module, checkpoint_path: str, device: torch.device):
    if not checkpoint_path:
        return None
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f'Lane teacher checkpoint not found: {ckpt_path}')
    # Build a teacher with the same lane_head class as the student. Use the
    # head's __init__ signature to filter only kwargs the head accepts -- so
    # this works for CurveLaneHead, CLRKDLaneHead, LaneQueryHead,
    # BezierLaneQueryHead, HybridPriorQueryHead, etc.
    import inspect as _inspect
    head_cls = type(model.lane_head)
    sig = _inspect.signature(head_cls.__init__)
    candidate_kwargs = {
        'in_channels': [int(c) for c in model.feature_channels],
        'embed_dim': int(getattr(model.lane_head, 'embed_dim', 128)),
        'max_lanes': int(getattr(model.lane_head, 'max_lanes', 10)),
        'num_points': int(getattr(model.lane_head, 'num_points', 72)),
        'mask_size': tuple(getattr(model.lane_head, 'mask_size', (72, 128))),
        'mask_aux': bool(getattr(model.lane_head, 'mask_aux', True)),
        'num_lane_classes': int(getattr(model.lane_head, 'num_lane_classes', 7)),
        'num_priors': int(getattr(model.lane_head, 'num_priors', 192)),
        'num_queries': int(getattr(model.lane_head, 'num_queries', 12)),
        'num_decoder_layers': int(getattr(model.lane_head, 'num_decoder_layers', 3)) if hasattr(model.lane_head, 'num_decoder_layers') else 3,
        'num_heads': int(getattr(model.lane_head, 'num_heads', 8)) if hasattr(model.lane_head, 'num_heads') else 8,
        'dim_feedforward': int(getattr(model.lane_head, 'dim_feedforward', 512)) if hasattr(model.lane_head, 'dim_feedforward') else 512,
        'dropout': float(getattr(model.lane_head, 'dropout', 0.1)) if hasattr(model.lane_head, 'dropout') else 0.1,
        'output_all_priors': bool(getattr(model.lane_head, 'output_all_priors', False)),
        'use_roi_gather': bool(getattr(model.lane_head, 'use_roi_gather', False)),
        'roi_refine_layers': int(getattr(model.lane_head, 'roi_refine_layers', 1)) if hasattr(model.lane_head, 'roi_refine_layers') else 1,
        'sample_points': int(getattr(model.lane_head, 'sample_points', 36)) if hasattr(model.lane_head, 'sample_points') else 36,
        'roi_mid_channels': int(getattr(model.lane_head, 'roi_mid_channels', 48)) if hasattr(model.lane_head, 'roi_mid_channels') else 48,
        'cross_attn_resize': tuple(getattr(model.lane_head, 'cross_attn_resize', (10, 25))) if hasattr(model.lane_head, 'cross_attn_resize') else (10, 25),
        'cls_uses_prior_embedding': bool(getattr(model.lane_head, 'cls_uses_prior_embedding', False)),
        'prior_embed_encoder_dim': int(getattr(model.lane_head, 'prior_embed_encoder_dim', 0)),
        'cls_separate_path': bool(getattr(model.lane_head, 'cls_separate_path', False)),
    }
    accepted = {k: v for k, v in candidate_kwargs.items() if k in sig.parameters}
    teacher = head_cls(**accepted).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get('state_dict', ckpt.get('model', ckpt))
    lane_state = {}
    for key, value in state.items():
        if key.startswith('lane_head.'):
            lane_state[key[len('lane_head.'):]] = value
        elif key.startswith('module.lane_head.'):
            lane_state[key[len('module.lane_head.'):]] = value
    if not lane_state:
        lane_state = state
    missing, unexpected = teacher.load_state_dict(lane_state, strict=False)
    log(f'loaded lane teacher={ckpt_path} missing={len(missing)} unexpected={len(unexpected)}')
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    return teacher


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _tar_signature(tar_path: Path) -> Dict[str, object]:
    stat = tar_path.stat()
    return {
        'path': str(tar_path),
        'size': int(stat.st_size),
        'mtime_ns': int(stat.st_mtime_ns),
    }


def _dataset_has_required_files(dest: Path) -> bool:
    required = [
        dest / 'list' / 'train_gt.txt',
        dest / 'list' / 'val.txt',
        dest / 'labels' / 'train',
        dest / 'labels' / 'val',
        dest / 'images' / 'train',
        dest / 'images' / 'val',
    ]
    return all(path.exists() for path in required)


def extract_tar_once(tar_path: Path, dest: Path, force: bool = False) -> None:
    marker = dest / '.extract.ok'
    if not tar_path.exists():
        raise FileNotFoundError(f'Missing tar archive: {tar_path}')
    current_sig = _tar_signature(tar_path)
    can_reuse = False
    if marker.exists() and not force:
        try:
            old_sig = json.loads(marker.read_text(encoding='utf-8'))
            can_reuse = old_sig == current_sig and _dataset_has_required_files(dest)
        except Exception:
            can_reuse = False
    if can_reuse:
        log(f'Already extracted and validated: {tar_path} -> {dest}')
        return
    if dest.exists():
        shutil.rmtree(dest)
    ensure_dir(dest)
    log(f'Extracting dataset tar: {tar_path} -> {dest}')
    subprocess.check_call(['tar', '-xf', str(tar_path), '-C', str(dest)])
    if not _dataset_has_required_files(dest):
        raise RuntimeError(
            f'Extracted {tar_path}, but required dataset files are missing under {dest}. '
            'Re-run Stage 2 notebook 00 to regenerate bdd100k_clrkd_curve.tar.'
        )
    marker.write_text(json.dumps(current_sig, indent=2), encoding='utf-8')


def resample_polyline(points: np.ndarray, n: int, img_w: int, img_h: int) -> Tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] < 2:
        return np.zeros((n, 2), dtype=np.float32), np.zeros((n,), dtype=np.float32)
    diffs = points[1:] - points[:-1]
    seg_len = np.sqrt((diffs * diffs).sum(axis=1))
    arc = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(arc[-1])
    if total <= 1e-6:
        return np.zeros((n, 2), dtype=np.float32), np.zeros((n,), dtype=np.float32)
    target = np.linspace(0.0, total, n)
    xs = np.interp(target, arc, points[:, 0]) / float(img_w)
    ys = np.interp(target, arc, points[:, 1]) / float(img_h)
    out = np.stack([xs, ys], axis=1).astype(np.float32)
    vis = ((out[:, 0] >= 0) & (out[:, 0] <= 1) & (out[:, 1] >= 0) & (out[:, 1] <= 1)).astype(np.float32)
    return np.clip(out, 0.0, 1.0), vis


def read_lines_targets(path: Path, max_lanes: int, num_points: int, img_w: int, img_h: int) -> Dict[str, np.ndarray]:
    existence = np.zeros((max_lanes,), dtype=np.float32)
    points = np.zeros((max_lanes, num_points, 2), dtype=np.float32)
    visibility = np.zeros((max_lanes, num_points), dtype=np.float32)
    lane_type = np.full((max_lanes,), -1, dtype=np.int64)
    rows = []
    if path.exists():
        rows = [line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    parsed = []
    for row in rows:
        vals = [float(x) for x in row.split()]
        if len(vals) < 4 or len(vals) % 2 != 0:
            continue
        arr = np.asarray(vals, dtype=np.float32).reshape(-1, 2)
        yspan = float(arr[:, 1].max() - arr[:, 1].min())
        length = float(np.sqrt(((arr[1:] - arr[:-1]) ** 2).sum(axis=1)).sum())
        if length >= 2.0 and yspan >= 1.0:
            parsed.append((yspan, length, arr))
    parsed.sort(key=lambda x: (-x[0], -x[1]))
    for i, (_ys, _ln, arr) in enumerate(parsed[:max_lanes]):
        pts, vis = resample_polyline(arr, num_points, img_w, img_h)
        existence[i] = 1.0
        points[i] = pts
        visibility[i] = vis
        lane_type[i] = 0
    return {'existence': existence, 'points': points, 'visibility': visibility, 'lane_type': lane_type}


class BDDJointCurveDataset(Dataset):
    def __init__(self, root: Path, split: str, image_size: Tuple[int, int], aux_mask_size: Tuple[int, int], max_lanes: int, num_points: int, limit: int = 0, require_det_labels: bool = True):
        self.root = Path(root)
        self.split = split
        self.image_size = tuple(image_size)
        self.aux_mask_size = tuple(aux_mask_size)
        self.max_lanes = int(max_lanes)
        self.num_points = int(num_points)
        self.label_dir = self.root / 'labels' / split
        list_file = self.root / 'list' / ('train_gt.txt' if split == 'train' else 'val.txt')
        if not list_file.exists():
            raise FileNotFoundError(f'Missing split list: {list_file}')
        items = []
        for row in list_file.read_text(encoding='utf-8').splitlines():
            if not row.strip():
                continue
            rel = row.split()[0].lstrip('/')
            items.append(self.root / rel)
        if limit and limit > 0:
            items = items[:limit]
        self.items = items
        if not self.items:
            raise RuntimeError(f'No samples found in {list_file}')
        self.det_label_files = [self.label_dir / f'{p.stem}.txt' for p in self.items]
        self.det_nonempty = sum(1 for p in self.det_label_files if p.exists() and p.read_text(encoding='utf-8').strip())
        if require_det_labels and self.det_nonempty == 0:
            preview = []
            for path in self.det_label_files[:10]:
                preview.append(f'{path.name}: exists={path.exists()} size={path.stat().st_size if path.exists() else -1}')
            summary_path = self.root / 'prepare_summary.json'
            summary_text = summary_path.read_text(encoding='utf-8')[:1200] if summary_path.exists() else 'missing prepare_summary.json'
            raise RuntimeError(
                f'No non-empty detection labels found for split={split} under {self.label_dir}. '
                'Stage 2 joint experiments must train detector and lane branch together. '
                'Most likely the local /content dataset is stale or the Drive tar was generated before detection labels were packed. '
                'Run notebook 00 again, then rerun this notebook with --force-extract.\n'
                f'Label preview: {preview}\n'
                f'prepare_summary preview: {summary_text}'
            )
        log(f'[dataset] split={split} samples={len(self.items)} nonempty_det_labels={self.det_nonempty} label_dir={self.label_dir}')

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        img_path = self.items[idx]
        image = cv2.imread(str(img_path))
        if image is None:
            raise FileNotFoundError(f'Failed to read image: {img_path}')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        org_h, org_w = image.shape[:2]
        resized = cv2.resize(image, (self.image_size[1], self.image_size[0]), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        line_path = img_path.with_suffix('.lines.txt')
        lane = read_lines_targets(line_path, self.max_lanes, self.num_points, org_w, org_h)
        mask = soft_polyline_mask_numpy(lane['points'], lane['visibility'], height=self.aux_mask_size[0], width=self.aux_mask_size[1])
        target = {k: torch.from_numpy(v) for k, v in lane.items()}
        target['mask_target'] = torch.from_numpy(mask[None, ...].astype(np.float32))
        det_labels = read_yolo_label(self.label_dir / f'{img_path.stem}.txt')
        return tensor, target, det_labels, img_path.name


def collate_fn(batch):
    images, targets, det_labels, names = zip(*batch)
    images = torch.stack(images, 0)
    merged = {}
    for key in targets[0].keys():
        merged[key] = torch.stack([t[key] for t in targets], 0)
    return images, merged, list(det_labels), list(names)


def tensor_dict_to_device(data: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in data.items()}


def build_det_loss(cfg: Dict):
    det_cfg = DetectionLossConfig(**cfg['loss']['det'])
    det_type = str(cfg['model'].get('detection_head', {}).get('type', 'detr_lite')).lower()
    if det_type in {'grid', 'simple_grid'}:
        return SimpleVehicleDetectionLoss(det_cfg)
    return DETRVehicleDetectionLoss(det_cfg, no_object_weight=float(cfg['loss'].get('no_object_weight', 0.25)))


@torch.no_grad()
def evaluate(
    model, lane_loss, det_loss, loader, device,
    max_batches: int = 20,
    eval_cfg: Optional[Dict] = None,
) -> Dict[str, float]:
    model.eval()
    eval_cfg = eval_cfg or {}
    det_metric = VehicleDetectionMetric()
    lane_metric = CLRKDStyleLaneMetric(
        score_threshold=float(getattr(lane_loss.cfg, 'existence_threshold', 0.5)),
        radius=float(getattr(lane_loss.cfg, 'line_iou_radius', 0.015)),
    )
    # Top-K decode + lane-NMS lane-F1 (Exp2N) -- bypasses the hardcoded-0.5
    # score-threshold bug in CLRKDStyleLaneMetric so we get a real lane-line
    # F1 number that compares to CLRKDNet's published values.
    # The score_source defaults to 'cls' but accepts 'cls_x_iou' (Exp2MM) when
    # the lane head emits iou_logits, so the decoder ranks by
    # sigmoid(cls) * sigmoid(iou) instead of plain sigmoid(cls).
    primary_score_source = str(eval_cfg.get('decoded_score_source', 'cls'))
    decoded_metric = LaneF1DecodedMetric(
        top_k=int(eval_cfg.get('decoded_top_k', 4)),
        line_iou_threshold=float(eval_cfg.get('decoded_lane_iou_threshold', 0.5)),
        lane_nms_iou=float(eval_cfg.get('decoded_lane_nms_iou', 0.5)),
        radius=float(getattr(lane_loss.cfg, 'line_iou_radius', 0.015)),
        score_threshold=float(eval_cfg.get('decoded_score_threshold', 0.0)),
        score_source=primary_score_source,
        key_suffix='',
    )
    # Optional Exp2O oracle diagnostic: same decode pipeline but rank priors
    # by ground-truth LineIoU instead of cls sigmoid. Reveals the F1 ceiling
    # the geometry can support independent of cls quality. Gated by
    # eval.decoded_oracle_enabled (default: false for back-compat).
    decoded_oracle_metric = None
    if bool(eval_cfg.get('decoded_oracle_enabled', False)):
        decoded_oracle_metric = LaneF1DecodedMetric(
            top_k=int(eval_cfg.get('decoded_top_k', 4)),
            line_iou_threshold=float(eval_cfg.get('decoded_lane_iou_threshold', 0.5)),
            lane_nms_iou=float(eval_cfg.get('decoded_lane_nms_iou', 0.5)),
            radius=float(getattr(lane_loss.cfg, 'line_iou_radius', 0.015)),
            score_threshold=0.0,           # oracle never thresholds
            score_source='oracle_iou',
            key_suffix='_oracle',
        )
    # Companion diagnostic metrics. When the primary metric ranks by a hybrid
    # score (cls_x_iou or cls_x_mask), also report the constituent rankings so
    # we can attribute the gain. When the primary metric ranks by mask alone,
    # also report cls so we can verify the cls head's behaviour.
    decoded_cls_only_metric = None
    decoded_mask_only_metric = None
    if primary_score_source in ('cls_x_iou', 'cls_x_mask'):
        decoded_cls_only_metric = LaneF1DecodedMetric(
            top_k=int(eval_cfg.get('decoded_top_k', 4)),
            line_iou_threshold=float(eval_cfg.get('decoded_lane_iou_threshold', 0.5)),
            lane_nms_iou=float(eval_cfg.get('decoded_lane_nms_iou', 0.5)),
            radius=float(getattr(lane_loss.cfg, 'line_iou_radius', 0.015)),
            score_threshold=float(eval_cfg.get('decoded_score_threshold', 0.0)),
            score_source='cls',
            key_suffix='_cls_only',
        )
    if primary_score_source in ('cls_x_mask', 'mask_consistency'):
        decoded_mask_only_metric = LaneF1DecodedMetric(
            top_k=int(eval_cfg.get('decoded_top_k', 4)),
            line_iou_threshold=float(eval_cfg.get('decoded_lane_iou_threshold', 0.5)),
            lane_nms_iou=float(eval_cfg.get('decoded_lane_nms_iou', 0.5)),
            radius=float(getattr(lane_loss.cfg, 'line_iou_radius', 0.015)),
            score_threshold=float(eval_cfg.get('decoded_score_threshold', 0.0)),
            score_source='mask_consistency',
            key_suffix='_mask_only',
        )
    sums: Dict[str, float] = {}
    count = 0
    for images, lane_targets, det_targets, _names in loader:
        images = images.to(device, non_blocking=True)
        lane_targets = tensor_dict_to_device(lane_targets, device)
        out = model(images)
        l_lane, lane_comp = lane_loss(out['lane'], lane_targets)
        l_det, det_comp = det_loss(out['det'], det_targets)
        lane_metrics = compute_lane_eval_metrics(
            out['lane'],
            lane_targets,
            lane_loss=lane_loss,
            threshold=float(getattr(lane_loss.cfg, 'existence_threshold', 0.5)),
        )
        matched_lane_targets = lane_loss.match_targets(out['lane'], lane_targets)
        lane_external_metrics = lane_metric(out['lane'], matched_lane_targets)
        # Decoded metric reads the ORIGINAL lane_targets, not matched_targets.
        decoded_external_metrics = decoded_metric(out['lane'], lane_targets)
        if decoded_oracle_metric is not None:
            decoded_external_metrics.update(
                decoded_oracle_metric(out['lane'], lane_targets)
            )
        if decoded_cls_only_metric is not None:
            decoded_external_metrics.update(
                decoded_cls_only_metric(out['lane'], lane_targets)
            )
        if decoded_mask_only_metric is not None:
            decoded_external_metrics.update(
                decoded_mask_only_metric(out['lane'], lane_targets)
            )
        det_external_metrics = det_metric(out['det'], det_targets)
        comp = {
            'val/lane_loss': l_lane,
            'val/lane_loss_scaled': l_lane,
            'val/det_loss': l_det,
        }
        comp.update({f'val/{k}': v for k, v in lane_metrics.items()})
        comp.update({f'val/{k}': v for k, v in lane_external_metrics.items()})
        comp.update({f'val/{k}': v for k, v in decoded_external_metrics.items()})
        comp.update({f'val/{k}': v for k, v in det_external_metrics.items()})
        comp.update({f'val/{k}': v for k, v in lane_comp.items()})
        comp.update({f'val/{k}': v for k, v in det_comp.items()})
        for key, value in comp.items():
            val = float(value.detach().cpu().item()) if torch.is_tensor(value) else float(value)
            if not np.isnan(val):
                sums[key] = sums.get(key, 0.0) + val
        count += 1
        if count >= max_batches:
            break
    return {k: v / max(1, count) for k, v in sums.items()}




def _format_metric(metrics: Dict[str, float], key: str, digits: int = 4, missing: str = 'n/a') -> str:
    value = metrics.get(key)
    if value is None:
        return missing
    try:
        return f'{float(value):.{digits}f}'
    except Exception:
        return missing


def log_epoch_summary(epoch: int, phase: str, metrics: Dict[str, float], best_score: float, elapsed_s: float, device: torch.device) -> None:
    parts = [
        '[epoch_summary]',
        f'epoch={epoch}',
        f'phase={phase}',
        f'train_total={_format_metric(metrics, "train/total")}',
        f'train_det={_format_metric(metrics, "train/det/total")}',
        f'train_lane={_format_metric(metrics, "train/lane/total")}',
        f'val_det={_format_metric(metrics, "val/det_loss")}',
        f'val_lane_scaled={_format_metric(metrics, "val/lane_loss_scaled")}',
        f'val_lane_unweighted={_format_metric(metrics, "val/lane/unweighted_total")}',
        f'val_lane_cls={_format_metric(metrics, "val/lane/cls")}',
        f'val_lane_reg={_format_metric(metrics, "val/lane/reg")}',
        f'val_lane_line_iou_loss={_format_metric(metrics, "val/lane/line_iou")}',
        f'val_lane_mask={_format_metric(metrics, "val/lane/mask_aux")}',
        f'val_lane_exist_acc={_format_metric(metrics, "val/lane_exist_acc")}',
        f'val_lane_f1={_format_metric(metrics, "val/lane_exist_f1")}',
        f'val_lane_best_f1={_format_metric(metrics, "val/lane_exist_best_f1")}',
        f'best_thr={_format_metric(metrics, "val/lane_exist_best_threshold", digits=2)}',
        f'val_lane_precision={_format_metric(metrics, "val/lane_exist_precision")}',
        f'val_lane_recall={_format_metric(metrics, "val/lane_exist_recall")}',
        f'val_lane_point_mae={_format_metric(metrics, "val/lane_point_mae")}',
        f'val_matched_line_iou={_format_metric(metrics, "val/matched_line_iou")}',
        f'pred_lanes={_format_metric(metrics, "val/num_pred_lanes", digits=1)}',
        f'gt_lanes={_format_metric(metrics, "val/num_gt_lanes", digits=1)}',
        f'matched_lanes={_format_metric(metrics, "val/num_matched_lanes", digits=1)}',
        f'fp_slots={_format_metric(metrics, "val/false_positive_lane_slots", digits=1)}',
        f'fn_slots={_format_metric(metrics, "val/false_negative_lane_slots", digits=1)}',
        f'pos_score={_format_metric(metrics, "val/lane_exist_pos_score_mean")}',
        f'neg_score={_format_metric(metrics, "val/lane_exist_neg_score_mean")}',
        f'val_clrkd_f1={_format_metric(metrics, "val/lane/clrkd_style_f1")}',
        f'val_decoded_f1={_format_metric(metrics, "val/lane/decoded_f1")}',
        f'val_decoded_p={_format_metric(metrics, "val/lane/decoded_precision")}',
        f'val_decoded_r={_format_metric(metrics, "val/lane/decoded_recall")}',
        f'decoded_pred={_format_metric(metrics, "val/lane/decoded_pred_count", digits=1)}',
        f'val_oracle_f1={_format_metric(metrics, "val/lane/decoded_oracle_f1")}',
        f'val_oracle_p={_format_metric(metrics, "val/lane/decoded_oracle_precision")}',
        f'val_oracle_r={_format_metric(metrics, "val/lane/decoded_oracle_recall")}',
        f'val_map50={_format_metric(metrics, "val/det/metric_map50")}',
        f'best_score={best_score:.4f}',
        f'elapsed_s={elapsed_s:.1f}',
    ]
    if device.type == 'cuda':
        mem_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
        parts.append(f'peak_mem_mb={mem_mb:.0f}')
    log(' '.join(parts))


def main() -> None:
    parser = argparse.ArgumentParser(description='Train one complete Stage 2 joint detection + CLRKD-curve lane experiment.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--curve-tar', default=None)
    parser.add_argument('--curve-root', default=None)
    parser.add_argument('--work-dir', default=None)
    parser.add_argument('--output-tar', default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--print-every', type=int, default=None, help='Override training print interval. The first step is always printed.')
    parser.add_argument('--limit-train', type=int, default=0)
    parser.add_argument('--limit-val', type=int, default=512)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--no-grad-cosine', action='store_true')
    parser.add_argument('--allow-empty-det-labels', action='store_true')
    parser.add_argument('--force-extract', action='store_true', help='Delete the local curve dataset and re-extract the tar archive before training.')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as fh:
        cfg = yaml.safe_load(fh)
    seed = args.seed if args.seed is not None else int(cfg['run'].get('seed', 0))
    seed_everything(seed)
    torch.set_num_threads(min(8, os.cpu_count() or 2))
    cfg['run']['seed'] = seed

    curve_tar = Path(args.curve_tar or cfg['dataset']['bdd_archive'])
    curve_root = Path(args.curve_root or cfg['dataset']['local_dir'])
    work_dir = ensure_dir(Path(args.work_dir or cfg['run']['work_dir']))
    output_tar = Path(args.output_tar or cfg['run']['output_tar'])
    if args.print_every is not None:
        cfg['train']['print_every'] = int(args.print_every)
    log('==== Stage 2 Joint Training Preflight ====')
    log(f"run_name={cfg['run'].get('name')}")
    log(f'config={args.config}')
    log(f'curve_tar={curve_tar}')
    log(f'curve_root={curve_root}')
    log(f'work_dir={work_dir}')
    log(f'output_tar={output_tar}')
    log(f"backbone={cfg['model'].get('backbone')} detection_head={cfg['model'].get('detection_head', {}).get('type')} lane_head=CLRKD-style curve head")
    log(f"epochs={args.epochs or cfg['train']['end_epoch']} batch_size={args.batch_size or cfg['train']['batch_size']} print_every={cfg['train'].get('print_every')} limit_train={args.limit_train} limit_val={args.limit_val}")
    log(f"loss=lambda_mode:{cfg['loss'].get('lambda_mode')} lambda_lane:{cfg['loss'].get('lambda_lane')} use_uncertainty:{cfg['loss'].get('use_uncertainty')}")
    extract_tar_once(curve_tar, curve_root, force=args.force_extract)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    log(f'device={device}')
    if device.type == 'cuda':
        log(f'gpu={torch.cuda.get_device_name(0)}')

    image_size = tuple(cfg['dataset']['image_size'])
    aux_mask_size = tuple(cfg['dataset']['aux_mask_size'])
    train_set = BDDJointCurveDataset(
        curve_root,
        'train',
        image_size=image_size,
        aux_mask_size=aux_mask_size,
        max_lanes=int(cfg['dataset']['max_lanes']),
        num_points=int(cfg['dataset']['num_points']),
        limit=args.limit_train,
        require_det_labels=not args.allow_empty_det_labels,
    )
    val_set = BDDJointCurveDataset(
        curve_root,
        'val',
        image_size=image_size,
        aux_mask_size=aux_mask_size,
        max_lanes=int(cfg['dataset']['max_lanes']),
        num_points=int(cfg['dataset']['num_points']),
        limit=args.limit_val,
        require_det_labels=not args.allow_empty_det_labels,
    )
    train_loader = DataLoader(train_set, batch_size=args.batch_size or int(cfg['train']['batch_size']), shuffle=True, num_workers=int(cfg['train'].get('workers', 2)), pin_memory=True, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size or int(cfg['train']['batch_size']), shuffle=False, num_workers=int(cfg['train'].get('workers', 2)), pin_memory=True, collate_fn=collate_fn)
    log(f'[loader] train_batches={len(train_loader)} val_batches={len(val_loader)} workers={int(cfg["train"].get("workers", 2))}')

    model = build_joint_model(cfg).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f'[model] class={model.__class__.__name__} params={total_params:,} trainable_initial={trainable_params:,}')
    if hasattr(model, 'feature_channels'):
        log(f'[model] feature_channels={getattr(model, "feature_channels")}')
    if hasattr(model, 'lane_head'):
        lh = model.lane_head
        log(f'[lane_head] max_lanes={getattr(lh, "max_lanes", None)} num_priors={getattr(lh, "num_priors", None)} output_all_priors={getattr(lh, "output_all_priors", None)} use_roi_gather={getattr(lh, "use_roi_gather", None)} roi_refine_layers={getattr(lh, "roi_refine_layers", None)}')
    log('[metric_sources] ' + json.dumps(metric_source_report(), indent=2))
    teacher_cfg = cfg.get('teacher', {}) or {}
    lane_teacher = load_lane_teacher(model, str(teacher_cfg.get('lane_head_checkpoint') or ''), device)
    lane_cfg = cfg['loss']['lane']
    lane_loss = FusionLaneLoss(FusionLossConfig(lambda_lane=float(cfg['loss'].get('lambda_lane', 1.0)), use_uncertainty=bool(cfg['loss'].get('use_uncertainty', False)), **lane_cfg)).to(device)
    if 'lane_exist_threshold' in cfg.get('eval', {}):
        lane_loss.cfg.existence_threshold = float(cfg['eval']['lane_exist_threshold'])
    base_lane_weights = {
        'w_reg': float(lane_loss.cfg.w_reg),
        'w_xytl': float(lane_loss.cfg.w_xytl),
        'w_iou': float(lane_loss.cfg.w_iou),
        'w_smooth': float(lane_loss.cfg.w_smooth),
    }
    det_loss = build_det_loss(cfg).to(device)
    mtl = UncertaintyMultiTaskLoss(2).to(device) if bool(cfg['loss'].get('use_uncertainty', False)) else None
    opt = build_optimizer(model, lane_loss, mtl, cfg)
    epochs = args.epochs or int(cfg['train']['end_epoch'])

    # Cosine LR scheduler with linear warmup. Standard published recipe for
    # lane detectors (CLRKDNet, CLRNet, RMT-PPAD all use cosine). Critical
    # for stable training past convergence: NB34 (Exp2CC) saw constant-LR
    # divergence at epoch 22+ where matched_iou collapsed 0.157 -> 0.050
    # because Adam kept stepping at full magnitude near the minimum and
    # eventually stepped out of it. Cosine decay shrinks step size toward
    # zero, eliminating this failure mode.
    sched_cfg = cfg['train'].get('lr_scheduler', {}) or {}
    sched_kind = str(sched_cfg.get('kind', 'none')).lower()
    if sched_kind == 'cosine':
        warmup_epochs = int(sched_cfg.get('warmup_epochs', 1))
        warmup_start_lr_factor = float(sched_cfg.get('warmup_start_lr_factor', 0.01))
        min_lr_factor = float(sched_cfg.get('min_lr_factor', 0.01))
        def lr_lambda(epoch_idx: int) -> float:
            # epoch_idx is 0-indexed (0..epochs-1).
            if epoch_idx < warmup_epochs:
                return warmup_start_lr_factor + (1.0 - warmup_start_lr_factor) * (epoch_idx + 1) / max(1, warmup_epochs)
            progress = (epoch_idx - warmup_epochs) / max(1, epochs - warmup_epochs)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_factor + (1.0 - min_lr_factor) * cosine
        scheduler = optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        log(f'[lr_scheduler] cosine kind={sched_kind} warmup={warmup_epochs} epochs={epochs} min_lr_factor={min_lr_factor}')
    else:
        scheduler = None
        log('[lr_scheduler] disabled (constant LR)')

    # AMP (mixed precision) support. Reads cfg['train']['amp']:
    #   kind: 'bfloat16' (recommended on RTX 6000 / A100 / H100), 'float16',
    #         or 'none' (default).
    # bfloat16 needs no GradScaler and is numerically stable for lane/det
    # losses. Gives ~2x speedup on supported GPUs without changing LR/batch
    # so the user's stage1 stability concerns are not affected.
    amp_cfg = cfg['train'].get('amp', {}) or {}
    amp_kind = str(amp_cfg.get('kind', 'none')).lower()
    if amp_kind == 'bfloat16':
        amp_dtype = torch.bfloat16
        amp_enabled = True
    elif amp_kind == 'float16':
        amp_dtype = torch.float16
        amp_enabled = True
    else:
        amp_dtype = torch.float32
        amp_enabled = False
    log(f'[amp] kind={amp_kind} enabled={amp_enabled} dtype={amp_dtype}')

    metrics: List[Dict[str, float]] = []
    best_score = float('-inf')

    for epoch in range(1, epochs + 1):
        epoch_t0 = time.perf_counter()
        phase = phase_for_epoch(cfg, epoch)
        apply_training_phase(model, phase)
        warmup_epochs = int(cfg['loss'].get('geometry_warmup_epochs', 0) or 0)
        if warmup_epochs > 0:
            geom_scale = min(1.0, max(0.25, epoch / float(warmup_epochs)))
            for key, base_value in base_lane_weights.items():
                setattr(lane_loss.cfg, key, base_value * geom_scale)
        else:
            geom_scale = 1.0
        epoch_lambda_lane = float(cfg['loss'].get('lambda_lane', 1.0))
        log(f'epoch={epoch} phase={phase} batches={len(train_loader)} lane_matching={bool(getattr(lane_loss.cfg, "use_lane_matching", False))} geom_scale={geom_scale:.3f}')
        log(f'epoch={epoch} waiting_for_first_batch...')
        model.train()
        step_t0 = time.perf_counter()
        sums: Dict[str, float] = {}
        count = 0
        # Track running grad_cos for joint-conflict diagnostics (responding to
        # user observation that NB38 showed lane and detection trading off).
        grad_cos_running_sum = 0.0
        grad_cos_running_count = 0
        for step, (images, lane_targets, det_targets, _names) in enumerate(train_loader, 1):
            images = images.to(device, non_blocking=True)
            lane_targets = tensor_dict_to_device(lane_targets, device)
            with torch.amp.autocast(device_type=('cuda' if device.type == 'cuda' else 'cpu'), dtype=amp_dtype, enabled=amp_enabled):
                out = model(images)
                teacher_out = None
                if lane_teacher is not None:
                    with torch.no_grad():
                        teacher_out = lane_teacher([f.detach() for f in out['lane_features']])
                l_lane, lane_comp = lane_loss(out['lane'], lane_targets, teacher=teacher_out)
                l_det, det_comp = det_loss(out['det'], det_targets)
            lambda_lane_runtime = epoch_lambda_lane
            grad_cos = float('nan')
            grad_ratio = float('nan')
            # Compute running grad_cos every 50 steps for visibility into
            # task conflict trends within an epoch (more frequent than the
            # original step==1 only). bfloat16 forward + float32 backward
            # for grad_cos to keep numerical stability.
            grad_cos_probe_interval = int(cfg.get('eval', {}).get('grad_cos_probe_interval', 50))
            should_probe = (step == 1) or (grad_cos_probe_interval > 0 and step % grad_cos_probe_interval == 0)
            if should_probe and not args.no_grad_cosine and bool(cfg.get('eval', {}).get('log_grad_cosine', True)):
                try:
                    grad_cos = compute_grad_cosine(l_det, l_lane, model.backbone.parameters())
                    if str(cfg['loss'].get('lambda_mode', 'fixed')).lower() in {'grad_norm', 'grad_norm_calibration'}:
                        grad_ratio = compute_grad_norm_ratio(
                            l_det,
                            l_lane,
                            model.backbone.parameters(),
                            clamp_min=float(cfg['loss'].get('lambda_min', 0.05)),
                            clamp_max=float(cfg['loss'].get('lambda_max', 2.0)),
                        )
                        lambda_lane_runtime = grad_ratio
                        epoch_lambda_lane = grad_ratio
                except RuntimeError:
                    grad_cos = float('nan')
                    grad_ratio = float('nan')
            # Update running grad_cos average for the epoch.
            if not math.isnan(grad_cos):
                grad_cos_running_sum += grad_cos
                grad_cos_running_count += 1
            if mtl is None:
                # Exp2BB diagnostic: lambda_det scales the detection loss
                # before joint combination. Default 1.0 = full det weight (back
                # compat). Setting lambda_det < 1 lets us test the 'is the
                # non-functional detection head corrupting backbone gradients
                # for lane?' hypothesis without changing the det head's own
                # internal loss weights.
                lambda_det = float(cfg['loss'].get('lambda_det', 1.0))
                total = lambda_det * l_det + lambda_lane_runtime * l_lane
                mtl_comp = {
                    'mtl/lambda_lane_runtime': torch.tensor(lambda_lane_runtime, device=device),
                    'mtl/grad_norm_ratio': torch.tensor(grad_ratio, device=device),
                    'mtl/lambda_det': torch.tensor(lambda_det, device=device),
                }
            else:
                total, mtl_comp = mtl([l_det, l_lane])
            gate_reg_weight = float(cfg['loss'].get('lambda_gate_reg', 0.0))
            if gate_reg_weight > 0 and 'aux' in out and isinstance(out['aux'], dict) and 'gates' in out['aux']:
                gate_terms = []
                gates = out['aux']['gates']
                for gate_list in gates.values():
                    for gate in gate_list:
                        gate_terms.append(((gate - 0.5) ** 2).mean())
                if gate_terms:
                    gate_reg = torch.stack(gate_terms).mean()
                    total = total + gate_reg_weight * gate_reg
                    mtl_comp['mtl/gate_reg'] = gate_reg.detach()
            opt.zero_grad(set_to_none=True)
            total.backward()
            clip = float(cfg['train'].get('grad_clip_norm', 0.0))
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            opt.step()
            comp = {'train/total': total.detach(), 'train/grad_cosine': torch.tensor(grad_cos), 'train/phase_id': torch.tensor(float(epoch)), 'train/mtl/lambda_lane_epoch': torch.tensor(epoch_lambda_lane), 'train/lane/geometry_scale': torch.tensor(geom_scale)}
            comp.update({f'train/{k}': v for k, v in lane_comp.items()})
            comp.update({f'train/{k}': v for k, v in det_comp.items()})
            comp.update({f'train/{k}': v for k, v in mtl_comp.items()})
            if 'aux' in out and isinstance(out['aux'], dict) and 'gate_stats' in out['aux']:
                comp.update({f'train/{k}': v for k, v in out['aux']['gate_stats'].items()})
            for key, value in comp.items():
                val = float(value.detach().cpu().item()) if torch.is_tensor(value) else float(value)
                if not np.isnan(val):
                    sums[key] = sums.get(key, 0.0) + val
            count += 1
            if step == 1 or step % int(cfg['train'].get('print_every', 50)) == 0:
                elapsed = max(time.perf_counter() - step_t0, 1e-6)
                steps_per_s = step / elapsed
                remaining = max(len(train_loader) - step, 0) / max(steps_per_s, 1e-6)
                log(
                    f'epoch={epoch} step={step}/{len(train_loader)} '
                    f'total={float(total.detach().cpu()):.4f} det={float(l_det.detach().cpu()):.4f} '
                    f'lane={float(l_lane.detach().cpu()):.4f} lambda_lane={lambda_lane_runtime:.4f} '
                    f'cls={float(lane_comp.get("lane/cls", torch.tensor(float("nan"), device=device)).detach().cpu()):.4f} '
                    f'reg={float(lane_comp.get("lane/reg", torch.tensor(float("nan"), device=device)).detach().cpu()):.4f} '
                    f'iou={float(lane_comp.get("lane/line_iou", torch.tensor(float("nan"), device=device)).detach().cpu()):.4f} '
                    f'grad_cos={grad_cos:.4f} speed={steps_per_s:.2f}step/s eta_min={remaining/60.0:.1f}'
                )
        epoch_metrics = {k: v / max(1, count) for k, v in sums.items()}
        epoch_metrics['epoch'] = float(epoch)
        # Joint conflict diagnostic: report mean grad_cos across all probes
        # this epoch. Negative = lane and det are pulling in opposite
        # directions on the shared backbone; positive = aligned. Running
        # mean is more reliable than the step==1 single-probe value.
        if grad_cos_running_count > 0:
            epoch_metrics['train/grad_cosine_epoch_mean'] = grad_cos_running_sum / grad_cos_running_count
            epoch_metrics['train/grad_cosine_probe_count'] = float(grad_cos_running_count)
            log(f'epoch={epoch} grad_cosine_epoch_mean={epoch_metrics["train/grad_cosine_epoch_mean"]:.4f} probes={grad_cos_running_count}')
        if epoch % int(cfg['train'].get('val_every', 1)) == 0:
            log(f'epoch={epoch} validation_start batches={min(len(val_loader), int(cfg["eval"].get("val_batches", 20)))}')
            epoch_metrics.update(evaluate(
                model, lane_loss, det_loss, val_loader, device,
                max_batches=int(cfg['eval'].get('val_batches', 20)),
                eval_cfg=cfg.get('eval', {}),
            ))
            log(f'epoch={epoch} validation_done')
        metrics.append(epoch_metrics)
        log(json.dumps(epoch_metrics, indent=2))
        ckpt = {'model': model.state_dict(), 'optimizer': opt.state_dict(), 'epoch': epoch, 'config': cfg, 'metrics': epoch_metrics}
        torch.save(ckpt, work_dir / 'last.pt')
        score = -epoch_metrics.get('val/det_loss', 0.0) - float(cfg['loss'].get('lambda_lane', 1.0)) * epoch_metrics.get('val/lane_loss', 0.0)
        if score > best_score:
            best_score = score
            torch.save(ckpt, work_dir / 'best.pt')
        log_epoch_summary(epoch, phase, epoch_metrics, best_score, time.perf_counter() - epoch_t0, device)
        # Step the LR scheduler at the END of each epoch so the warmup window
        # is respected as configured (warmup uses the first N epochs at full
        # base LR, then cosine decay applies from epoch N+1 onward).
        if scheduler is not None:
            scheduler.step()
            try:
                cur_lrs = [g['lr'] for g in opt.param_groups]
                log(f'[lr_scheduler] after_epoch={epoch} param_group_lrs={cur_lrs}')
            except Exception:
                pass
        if epoch % int(cfg['train'].get('save_every', 5)) == 0:
            torch.save(ckpt, work_dir / f'epoch_{epoch}.pt')
        (work_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        (work_dir / 'config_snapshot.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')

    ensure_dir(output_tar.parent)
    if output_tar.exists():
        output_tar.unlink()
    subprocess.check_call(['tar', '-cf', str(output_tar), '-C', str(work_dir), '.'])
    log(f'Wrote {output_tar}')
    log('Training artifacts inside tar: best.pt, last.pt, metrics.json, config_snapshot.yaml')


if __name__ == '__main__':
    main()
