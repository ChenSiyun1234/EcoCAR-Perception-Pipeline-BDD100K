"""
BDD100K dataset variant emitting **point-based** lane targets for the
YOLOPv2-DETRLane head, alongside the same box labels the YOLOPv2
anchor head consumes.

Differences vs `BddDataset`:
  * lane ground truth is (existence, points, visibility, lane_type),
    not a rendered binary mask.
  * __getitem__ returns target = [det_labels, lane_point_targets_dict].
  * collate_fn pads lane tensors to (B, M, P, 2) with M = cfg.LANE.MAX_GT.

This class deliberately reuses the path resolver, augmentation, and
detection-label pipeline from `BddDataset`. The only thing it replaces
is the lane side.
"""

from __future__ import annotations

import os
from typing import Dict, List

import cv2
import numpy as np
import torch

from .bdd import BddDataset
from ..utils.lane_targets import LaneLabelCache, frame_to_lane_targets, extract_lane_labels_any
from ..utils import letterbox, augment_hsv, random_perspective, xyxy2xywh


class BddDatasetPoints(BddDataset):
    """BDD100K with point lane targets. Detection path is inherited."""

    def __init__(self, cfg, is_train, inputsize, transform=None):
        # We bypass BddDataset.__init__ at the point where it builds
        # the mask-based `db`; we still want its `_resolve_split_paths`
        # infra from AutoDriveDataset.
        from .AutoDriveDataset import AutoDriveDataset
        AutoDriveDataset.__init__(self, cfg, is_train, inputsize, transform)

        self.cfg = cfg
        self.max_gt_lanes = int(getattr(cfg, 'LANE', object()).__dict__.get('MAX_GT', 10)
                                if hasattr(cfg, 'LANE') else 10)
        self.num_points = int(getattr(cfg, 'LANE', object()).__dict__.get('NUM_POINTS', 72)
                              if hasattr(cfg, 'LANE') else 72)

        # Locate lane JSON source for this split.
        lane_json = self._resolve_lane_json_path(is_train)
        self.lane_cache = LaneLabelCache(
            source_path=lane_json,
            max_lanes=self.max_gt_lanes,
            num_points=self.num_points,
        )
        print(f'[BddDatasetPoints] lane JSON: {lane_json} | cached frames: {len(self.lane_cache)}')

        self.db = self._get_db_points()

    # ──────────────────────────────────────────────────────────────
    def _resolve_lane_json_path(self, is_train: bool) -> str:
        cfg = self.cfg
        if is_train:
            p = getattr(cfg.DATASET, 'LANE_JSON_TRAIN', '')
        else:
            p = getattr(cfg.DATASET, 'LANE_JSON_VAL', '')
        return str(p) if p else ''

    def _get_db_points(self):
        # Like BddDataset._get_db but indexes by *images that also have
        # a lane-JSON record*. Images without lane labels become
        # detection-only samples (lane existence all zero).
        print('[BddDatasetPoints] building database...')
        gt_db = []
        height, width = self.shapes
        if not os.path.isdir(self.img_root):
            print(f'ERROR: image root not found: {self.img_root}')
            return gt_db
        for name in sorted(os.listdir(self.img_root)):
            if not name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            stem, _ = os.path.splitext(name)
            img_path = os.path.join(self.img_root, name)
            gt = self._load_detection_labels(stem, width, height)
            lane_targets = self.lane_cache.get(name) if self.lane_cache else None
            gt_db.append({
                'image': img_path,
                'label': gt,
                'lane_targets': lane_targets,   # may be None (= no lanes)
            })
        print(f'[BddDatasetPoints] db size: {len(gt_db)}')
        return gt_db

    # ──────────────────────────────────────────────────────────────
    def __getitem__(self, idx: int):
        data = self.db[idx]
        img = cv2.imread(data['image'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if img is None:
            raise FileNotFoundError(f'Failed to read image: {data["image"]}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h0, w0 = img.shape[:2]

        resized_shape = self.inputsize
        if isinstance(resized_shape, list):
            resized_shape = max(resized_shape)
        r = resized_shape / max(h0, w0)
        if r != 1:
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
        h, w = img.shape[:2]

        # Letterbox — lanes are (N_lanes, N_points, 2) in normalized [0,1] coords
        # at ORIGINAL resolution, so we transform them analytically here (no
        # interpolation of a mask needed).
        letterbox_out = letterbox((img, np.zeros((h, w), dtype=np.uint8)),
                                  resized_shape, auto=True, scaleup=self.is_train)
        (img, _dummy_lane), ratio, pad = letterbox_out
        shapes = (h0, w0), ((h / h0, w / w0), pad)

        # Detection labels (normalized xywh in original → pixel xyxy in letterboxed).
        det = data['label']
        labels = []
        if det.size > 0:
            labels = det.copy()
            labels[:, 1] = ratio[0] * w * (det[:, 1] - det[:, 3] / 2) + pad[0]
            labels[:, 2] = ratio[1] * h * (det[:, 2] - det[:, 4] / 2) + pad[1]
            labels[:, 3] = ratio[0] * w * (det[:, 1] + det[:, 3] / 2) + pad[0]
            labels[:, 4] = ratio[1] * h * (det[:, 2] + det[:, 4] / 2) + pad[1]

        # Lane point targets — convert normalized (0,1) coords to the
        # letterboxed frame coords (still normalized 0..1 of letterbox size).
        existence, lane_points, visibility, lane_type = self._prepare_lane_targets(
            data['lane_targets'], ratio, pad, h, w)

        # ── Augmentation: perspective + HSV + flip (no mosaic for now) ──
        if self.is_train:
            # For simplicity, we apply perspective only to img; lane points
            # and boxes are augmented consistently via the same matrix.
            # Using BddDataset's existing `random_perspective` helper would
            # require lane to be a mask, so here we only do HSV + flip on
            # the point-lane track to keep geometry alignment exact.
            augment_hsv(img,
                        hgain=self.cfg.DATASET.HSV_H,
                        sgain=self.cfg.DATASET.HSV_S,
                        vgain=self.cfg.DATASET.HSV_V)
            if len(labels):
                labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])
                labels[:, [2, 4]] /= img.shape[0]
                labels[:, [1, 3]] /= img.shape[1]
            # Horizontal flip (lanes: mirror x, boxes: mirror cx)
            import random
            if random.random() < 0.5:
                img = np.fliplr(img)
                if len(labels):
                    labels[:, 1] = 1 - labels[:, 1]
                lane_points[..., 0] = 1.0 - lane_points[..., 0]
        else:
            if len(labels):
                labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])
                labels[:, [2, 4]] /= img.shape[0]
                labels[:, [1, 3]] /= img.shape[1]

        labels_out = torch.zeros((len(labels), 6), dtype=torch.float32)
        if len(labels):
            labels_out[:, 1:] = torch.from_numpy(np.asarray(labels, dtype=np.float32))
        img = np.ascontiguousarray(img)

        target = {
            'det': labels_out,
            'lane_existence': torch.from_numpy(existence),
            'lane_points': torch.from_numpy(lane_points),
            'lane_visibility': torch.from_numpy(visibility),
            'lane_type': torch.from_numpy(lane_type),
            'has_lanes': torch.tensor(float(existence.sum() > 0)),
        }
        img = self.transform(img)
        return img, target, data['image'], shapes

    # ──────────────────────────────────────────────────────────────
    def _prepare_lane_targets(self, lane_targets, ratio, pad, h, w):
        M = self.max_gt_lanes
        P = self.num_points
        existence = np.zeros(M, dtype=np.float32)
        points = np.zeros((M, P, 2), dtype=np.float32)
        visibility = np.zeros((M, P), dtype=np.float32)
        lane_type = np.zeros(M, dtype=np.int64)

        if lane_targets is None:
            return existence, points, visibility, lane_type

        raw_existence = lane_targets['existence']
        raw_points = lane_targets['points']       # (M, P, 2) normalized to ORIGINAL image
        raw_visibility = lane_targets['visibility']
        raw_type = lane_targets['lane_type']

        # Letterbox coords: normalized (0,1) of LETTERBOXED (H_lb, W_lb).
        # raw_points are normalized (0,1) of ORIGINAL (h0, w0) — we need to
        # map them into the letterboxed frame: x_lb = (x_orig * w * ratio + pad_x) / W_lb
        W_lb = img_w = w + pad[0] * 2 if False else None  # avoid unused
        # After letterbox, final tensor has shape (H_lb, W_lb) where
        #   W_lb = w + 2*pad_w, H_lb = h + 2*pad_h  (approximately)
        # but pad is (dw, dh) already accounting for asymmetric 0.1 rounding.
        W_lb = int(round(w + pad[0] * 2))
        H_lb = int(round(h + pad[1] * 2))

        M_raw = raw_existence.shape[0]
        K = min(M, M_raw)
        existence[:K] = raw_existence[:K]
        visibility[:K] = raw_visibility[:K]
        lane_type[:K] = raw_type[:K]

        # raw_points were normalized by (w0, h0) of ORIGINAL image. Our
        # resized image has shape (h, w) = (h0*r, w0*r). After letterbox,
        # final shape is (H_lb, W_lb) with top-left padding (pad[1], pad[0]).
        # So x_lb_norm = (x_orig_norm * w + pad[0]) / W_lb.
        raw = raw_points[:K].copy()
        raw[..., 0] = (raw[..., 0] * w + pad[0]) / max(W_lb, 1e-6)
        raw[..., 1] = (raw[..., 1] * h + pad[1]) / max(H_lb, 1e-6)
        points[:K] = np.clip(raw, 0.0, 1.0).astype(np.float32)
        return existence, points, visibility, lane_type

    # ──────────────────────────────────────────────────────────────
    @staticmethod
    def collate_fn(batch):
        imgs, targets, paths, shapes = zip(*batch)
        det_labels = []
        for i, t in enumerate(targets):
            lbl = t['det']
            if lbl.numel() > 0:
                lbl[:, 0] = i
                det_labels.append(lbl)
        det_labels = (torch.cat(det_labels, 0) if det_labels
                      else torch.zeros((0, 6), dtype=torch.float32))
        return (
            torch.stack(imgs, 0),
            {
                'det': det_labels,
                'lane_existence': torch.stack([t['lane_existence'] for t in targets], 0),
                'lane_points': torch.stack([t['lane_points'] for t in targets], 0),
                'lane_visibility': torch.stack([t['lane_visibility'] for t in targets], 0),
                'lane_type': torch.stack([t['lane_type'] for t in targets], 0),
                'has_lanes': torch.stack([t['has_lanes'] for t in targets], 0),
            },
            paths,
            shapes,
        )
