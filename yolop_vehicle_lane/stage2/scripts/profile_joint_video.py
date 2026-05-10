from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from stage2.fusion.experiment_factory import build_joint_model


def load_checkpoint(path: Path, device: torch.device) -> Dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location=device)


def cxcywh_to_xyxy(box: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = box.unbind(-1)
    return torch.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1).clamp(0.0, 1.0)


def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.numel() == 0 or b.numel() == 0:
        return a.new_zeros((a.shape[0], b.shape[0]))
    lt = torch.maximum(a[:, None, :2], b[None, :, :2])
    rb = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = ((a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0))[:, None]
    area_b = ((b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0))[None, :]
    return inter / (area_a + area_b - inter + 1e-6)


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thr: float, max_det: int) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device)
    order = torch.argsort(scores, descending=True)
    keep = []
    while order.numel() > 0 and len(keep) < max_det:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break
        ious = box_iou(boxes[i].view(1, 4), boxes[order[1:]]).view(-1)
        order = order[1:][ious <= iou_thr]
    return torch.stack(keep) if keep else torch.zeros((0,), dtype=torch.long, device=boxes.device)


def decode_detections(det: Dict[str, torch.Tensor], conf_thr: float, iou_thr: float, max_det: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if det is None or 'obj_logits' not in det:
        device = next(iter(det.values())).device if det else torch.device('cpu')
        return torch.zeros((0, 4), device=device), torch.zeros((0,), device=device)
    if 'box_pred' in det:
        obj = det['obj_logits'][0].sigmoid()
        cls = det['cls_logits'][0].sigmoid().max(dim=-1).values
        scores = obj * cls
        boxes = cxcywh_to_xyxy(det['box_pred'][0])
    elif 'box_raw' in det:
        obj = det['obj_logits'][0, 0].sigmoid().flatten()
        cls = det['cls_logits'][0].sigmoid().amax(dim=0).flatten()
        scores = obj * cls
        raw = det['box_raw'][0].sigmoid().permute(1, 2, 0).reshape(-1, 4)
        boxes = cxcywh_to_xyxy(raw)
    else:
        device = det['obj_logits'].device
        return torch.zeros((0, 4), device=device), torch.zeros((0,), device=device)
    keep = torch.nonzero(scores >= conf_thr, as_tuple=False).flatten()
    if keep.numel() == 0:
        return boxes.new_zeros((0, 4)), scores.new_zeros((0,))
    boxes = boxes[keep]
    scores = scores[keep]
    keep2 = nms(boxes, scores, iou_thr=iou_thr, max_det=max_det)
    return boxes[keep2], scores[keep2]


def draw_predictions(frame: np.ndarray, out: Dict[str, Dict[str, torch.Tensor]], det_conf: float, det_iou: float, lane_conf: float) -> Tuple[np.ndarray, int, int]:
    orig_h, orig_w = frame.shape[:2]
    canvas = frame.copy()
    boxes, scores = decode_detections(out.get('det'), det_conf, det_iou, max_det=80)
    boxes_np = boxes.detach().cpu().numpy()
    scores_np = scores.detach().cpu().numpy()
    for box, score in zip(boxes_np, scores_np):
        x1 = int(box[0] * orig_w)
        y1 = int(box[1] * orig_h)
        x2 = int(box[2] * orig_w)
        y2 = int(box[3] * orig_h)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(canvas, f'veh {score:.2f}', (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    lane_scores = out['lane']['cls_logits'][0].sigmoid()
    lanes = out['lane']['coord_pred'][0]
    lane_keep = torch.nonzero(lane_scores >= lane_conf, as_tuple=False).flatten()
    if lane_keep.numel() > 24:
        lane_keep = lane_keep[torch.argsort(lane_scores[lane_keep], descending=True)[:24]]
    for idx in lane_keep.tolist():
        pts = lanes[idx].detach().cpu().numpy()
        pts_xy = np.stack([pts[:, 0] * orig_w, pts[:, 1] * orig_h], axis=1).astype(np.int32)
        valid = (pts_xy[:, 0] >= 0) & (pts_xy[:, 0] < orig_w) & (pts_xy[:, 1] >= 0) & (pts_xy[:, 1] < orig_h)
        pts_xy = pts_xy[valid]
        if len(pts_xy) >= 2:
            cv2.polylines(canvas, [pts_xy.reshape(-1, 1, 2)], False, (0, 0, 255), 2, cv2.LINE_AA)
    return canvas, int(len(boxes_np)), int(lane_keep.numel())


def main() -> None:
    parser = argparse.ArgumentParser(description='Profile a complete Stage 2 joint model on a video and optionally write visualized predictions.')
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--run-tar', default=None)
    parser.add_argument('--video', required=True)
    parser.add_argument('--output-dir', default='/content/stage2_video_profile')
    parser.add_argument('--max-frames', type=int, default=600)
    parser.add_argument('--write-video', action='store_true')
    parser.add_argument('--det-conf', type=float, default=0.05)
    parser.add_argument('--det-iou', type=float, default=0.60)
    parser.add_argument('--lane-conf', type=float, default=0.30)
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as fh:
        cfg = yaml.safe_load(fh)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = Path(args.checkpoint) if args.checkpoint else None
    if args.run_tar:
        run_tar = Path(args.run_tar)
        if not run_tar.exists():
            raise FileNotFoundError(f'Missing run tar: {run_tar}')
        run_dir = Path(args.output_dir) / f'extracted_{run_tar.stem}'
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        print(f'[profile] extracting run tar: {run_tar} -> {run_dir}', flush=True)
        subprocess.check_call(['tar', '-xf', str(run_tar), '-C', str(run_dir)])
        candidates = []
        for name in ['best.pt', 'last.pt', 'checkpoint_best.pt', 'checkpoint_last.pt']:
            candidates.extend(run_dir.rglob(name))
        if not candidates:
            candidates = sorted(run_dir.rglob('*.pt'), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise FileNotFoundError(f'No checkpoint found inside {run_tar}')
        ckpt_path = candidates[0]
    if ckpt_path is None:
        raise ValueError('Provide either --checkpoint or --run-tar')
    model = build_joint_model(cfg).to(device)
    ckpt = load_checkpoint(ckpt_path, device)
    state = ckpt.get('model', ckpt.get('state_dict', ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f'[profile] loaded checkpoint={ckpt_path} missing={len(missing)} unexpected={len(unexpected)}', flush=True)
    model.eval()

    image_h, image_w = cfg['dataset']['image_size']
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise FileNotFoundError(f'Cannot open video: {args.video}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = None
    if args.write_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        out_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(out_dir / 'profile_preview.mp4'), fourcc, fps, (out_w, out_h))

    rows = []
    frame_id = 0
    while frame_id < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (image_w, image_h), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
        if device.type == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(tensor)
        if device.type == 'cuda':
            torch.cuda.synchronize()
            peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        else:
            peak_mb = 0.0
        dt = time.perf_counter() - t0
        draw_frame, det_count, lane_count = draw_predictions(frame, out, args.det_conf, args.det_iou, args.lane_conf)
        rows.append({'frame': frame_id, 'latency_ms': dt * 1000.0, 'fps': 1.0 / max(dt, 1e-9), 'peak_memory_mb': peak_mb, 'det_count': det_count, 'lane_count': lane_count})
        if frame_id < 10 or frame_id % 30 == 0:
            print(f'[profile_frame] frame={frame_id} latency_ms={dt*1000.0:.2f} det_count={det_count} lane_count={lane_count}', flush=True)
        if writer is not None:
            writer.write(draw_frame)
        frame_id += 1
    cap.release()
    if writer is not None:
        writer.release()

    csv_path = out_dir / 'profile.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as fh:
        writer_csv = csv.DictWriter(fh, fieldnames=['frame', 'latency_ms', 'fps', 'peak_memory_mb', 'det_count', 'lane_count'])
        writer_csv.writeheader()
        writer_csv.writerows(rows)
    summary = {
        'frames': len(rows),
        'mean_latency_ms': float(np.mean([r['latency_ms'] for r in rows])) if rows else 0.0,
        'p95_latency_ms': float(np.percentile([r['latency_ms'] for r in rows], 95)) if rows else 0.0,
        'mean_fps': float(np.mean([r['fps'] for r in rows])) if rows else 0.0,
        'peak_memory_mb': float(np.max([r['peak_memory_mb'] for r in rows])) if rows else 0.0,
        'mean_det_count': float(np.mean([r['det_count'] for r in rows])) if rows else 0.0,
        'mean_lane_count': float(np.mean([r['lane_count'] for r in rows])) if rows else 0.0,
        'preview_video': str(out_dir / 'profile_preview.mp4') if args.write_video else None,
        'device': str(device),
    }
    (out_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('[profile_summary] ' + json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
