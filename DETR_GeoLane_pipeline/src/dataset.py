
"""
BDD100K dataset for dual-path training.

Supports both original BDD class ids and already-remapped YOLO vehicle ids.
This is important because many previous YOLO26 pipelines already rewrote labels
to vehicle-only ids [0..4], and interpreting those as original BDD ids silently
drops valid motorcycle/bicycle labels.
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple

from .config import Config, BDD_TO_VEHICLE, BDD_TO_EXPANDED
from .lane_targets import LaneLabelCache

_VEHICLE_IDENTITY = {i: i for i in range(5)}
_EXPANDED_IDENTITY = {i: i for i in range(7)}

def infer_label_schema(labels_dir: str, use_expanded_classes: bool = False, max_files: int = 256) -> str:
    txt_files = sorted([f for f in os.listdir(labels_dir) if f.endswith('.txt')])[:max_files]
    if not txt_files:
        return 'bdd_original'
    observed = set()
    for fname in txt_files:
        path = os.path.join(labels_dir, fname)
        try:
            with open(path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    try:
                        observed.add(int(float(parts[0])))
                    except Exception:
                        continue
                    if len(observed) >= 16:
                        break
        except Exception:
            continue
        if len(observed) >= 16:
            break
    if not observed:
        return 'bdd_original'
    if use_expanded_classes and observed.issubset(set(_EXPANDED_IDENTITY.keys())):
        return 'expanded_remapped'
    if (not use_expanded_classes) and observed.issubset(set(_VEHICLE_IDENTITY.keys())):
        return 'vehicle_remapped'
    if observed.issubset(set(BDD_TO_VEHICLE.keys())) or observed.issubset(set(BDD_TO_EXPANDED.keys())):
        return 'bdd_original'
    if max(observed) <= 4 and not use_expanded_classes:
        return 'vehicle_remapped'
    if max(observed) <= 6 and use_expanded_classes:
        return 'expanded_remapped'
    return 'bdd_original'

class BDD100KDualDataset(Dataset):
    def __init__(self, images_dir: str, labels_dir: str,
                 lane_cache: Optional[LaneLabelCache] = None,
                 img_size: int = 640,
                 max_lanes: int = 10, lane_points: int = 72,
                 use_expanded_classes: bool = False,
                 label_schema: str = 'auto',
                 augment: bool = False):
        super().__init__()
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.lane_cache = lane_cache
        self.img_size = img_size
        self.max_lanes = max_lanes
        self.lane_points = lane_points
        self.augment = augment
        self.use_expanded_classes = use_expanded_classes

        self.image_files = sorted([f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        print(f"  Dataset: {len(self.image_files)} images from {images_dir}")

        if label_schema == 'auto':
            label_schema = infer_label_schema(labels_dir, use_expanded_classes=use_expanded_classes)
        self.label_schema = label_schema
        if use_expanded_classes:
            self.class_map = BDD_TO_EXPANDED if label_schema == 'bdd_original' else _EXPANDED_IDENTITY
        else:
            self.class_map = BDD_TO_VEHICLE if label_schema == 'bdd_original' else _VEHICLE_IDENTITY
        print(f"  Detection label schema: {self.label_schema}")
        self._print_label_histogram()

    def _print_label_histogram(self, max_files: int = 512):
        counts = {int(k): 0 for k in sorted(set(self.class_map.values()))}
        txt_files = sorted([f for f in os.listdir(self.labels_dir) if f.endswith('.txt')])[:max_files]
        for fname in txt_files:
            path = os.path.join(self.labels_dir, fname)
            try:
                with open(path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            continue
                        try:
                            raw_cls = int(float(parts[0]))
                        except Exception:
                            continue
                        if raw_cls in self.class_map:
                            counts[int(self.class_map[raw_cls])] = counts.get(int(self.class_map[raw_cls]), 0) + 1
            except Exception:
                continue
        if counts:
            human = [f"{k}:{v}" for k, v in sorted(counts.items())]
            print("  Sampled detection class counts -> " + ", ".join(human))

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        fname = self.image_files[idx]
        img_path = os.path.join(self.images_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))

        do_flip = bool(self.augment and (np.random.rand() < 0.5))
        if do_flip:
            img = img[:, ::-1, :].copy()
        img_tensor = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)

        stem = os.path.splitext(fname)[0]
        label_path = os.path.join(self.labels_dir, stem + '.txt')
        det_targets = self._load_det_labels(label_path)

        lane_targets = self.lane_cache.get(fname) if self.lane_cache is not None else None
        if lane_targets is None:
            lane_targets = {
                'existence': np.zeros(self.max_lanes, dtype=np.float32),
                'points': np.zeros((self.max_lanes, self.lane_points, 2), dtype=np.float32),
                'visibility': np.zeros((self.max_lanes, self.lane_points), dtype=np.float32),
                'lane_type': np.zeros(self.max_lanes, dtype=np.int64),
            }
            has_lanes = False
        else:
            has_lanes = True

        if do_flip:
            if det_targets.shape[0] > 0:
                det_targets[:, 1] = 1.0 - det_targets[:, 1]
            if has_lanes:
                lane_targets['points'][:, :, 0] = 1.0 - lane_targets['points'][:, :, 0]

        return {
            'image': img_tensor,
            'det_targets': torch.from_numpy(det_targets),
            'lane_existence': torch.from_numpy(lane_targets['existence']),
            'lane_points': torch.from_numpy(lane_targets['points']),
            'lane_visibility': torch.from_numpy(lane_targets['visibility']),
            'lane_type': torch.from_numpy(lane_targets['lane_type']),
            'has_lanes': torch.tensor(1.0 if has_lanes else 0.0),
            'image_name': fname,
        }

    def _load_det_labels(self, path: str) -> np.ndarray:
        if not os.path.isfile(path):
            return np.zeros((0, 5), dtype=np.float32)
        labels = []
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                try:
                    raw_cls = int(float(parts[0]))
                except Exception:
                    continue
                if raw_cls not in self.class_map:
                    continue
                mapped_cls = self.class_map[raw_cls]
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                if w <= 0.0 or h <= 0.0:
                    continue
                labels.append([mapped_cls, cx, cy, w, h])
        if not labels:
            return np.zeros((0, 5), dtype=np.float32)
        return np.asarray(labels, dtype=np.float32)

def dual_collate_fn(batch: List[dict]) -> dict:
    images = torch.stack([b['image'] for b in batch])
    det_list = []
    for bi, b in enumerate(batch):
        tgt = b['det_targets']
        if tgt.shape[0] > 0:
            batch_idx = torch.full((tgt.shape[0], 1), bi, dtype=tgt.dtype)
            det_list.append(torch.cat([batch_idx, tgt], dim=1))
    det_targets = torch.cat(det_list) if det_list else torch.zeros((0, 6))
    return {
        'images': images,
        'det_targets': det_targets,
        'lane_existence': torch.stack([b['lane_existence'] for b in batch]),
        'lane_points': torch.stack([b['lane_points'] for b in batch]),
        'lane_visibility': torch.stack([b['lane_visibility'] for b in batch]),
        'lane_type': torch.stack([b['lane_type'] for b in batch]),
        'has_lanes': torch.stack([b['has_lanes'] for b in batch]),
    }

def build_dataloaders(cfg: Config, train_lane_json: Optional[str] = None, val_lane_json: Optional[str] = None) -> Tuple[DataLoader, DataLoader]:
    root = cfg.dataset_root
    train_lane_cache = LaneLabelCache(train_lane_json, cfg.max_lanes, cfg.lane_points) if train_lane_json else None
    val_lane_cache = LaneLabelCache(val_lane_json, cfg.max_lanes, cfg.lane_points) if val_lane_json else None

    label_schema = getattr(cfg, 'label_schema', 'auto')
    train_ds = BDD100KDualDataset(
        images_dir=os.path.join(root, 'images', 'train'),
        labels_dir=os.path.join(root, 'labels', 'train'),
        lane_cache=train_lane_cache,
        img_size=cfg.img_size,
        max_lanes=cfg.max_lanes,
        lane_points=cfg.lane_points,
        use_expanded_classes=cfg.use_expanded_classes,
        label_schema=label_schema,
        augment=True,
    )
    val_schema = train_ds.label_schema if label_schema == 'auto' else label_schema
    val_ds = BDD100KDualDataset(
        images_dir=os.path.join(root, 'images', 'val'),
        labels_dir=os.path.join(root, 'labels', 'val'),
        lane_cache=val_lane_cache,
        img_size=cfg.img_size,
        max_lanes=cfg.max_lanes,
        lane_points=cfg.lane_points,
        use_expanded_classes=cfg.use_expanded_classes,
        label_schema=val_schema,
        augment=False,
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, collate_fn=dual_collate_fn, pin_memory=True, drop_last=True, persistent_workers=cfg.num_workers > 0)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, collate_fn=dual_collate_fn, pin_memory=True, persistent_workers=cfg.num_workers > 0)
    return train_loader, val_loader
