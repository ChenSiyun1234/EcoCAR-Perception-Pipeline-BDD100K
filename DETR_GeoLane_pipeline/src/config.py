"""
Centralized path conventions and default configuration.

All Google Drive paths are defined here so the rest of the codebase
never hardcodes a Drive path. On Colab the Drive is mounted at
/content/drive/MyDrive; locally the paths are unused.
"""

import os
import yaml
import zipfile
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict

# ── Drive path constants (match the existing EcoCAR layout) ──────────
ECOCAR_ROOT      = "/content/drive/MyDrive/EcoCAR"
DATASET_ROOT     = os.path.join(ECOCAR_ROOT, "datasets", "bdd100k_yolo")
WEIGHTS_DIR      = os.path.join(ECOCAR_ROOT, "weights")
TRAINING_RUNS    = os.path.join(ECOCAR_ROOT, "training_runs")
OUTPUTS_DIR      = os.path.join(ECOCAR_ROOT, "outputs")
VIDEO_DIR        = os.path.join(ECOCAR_ROOT, "video")
DOWNLOADS_DIR    = os.path.join(ECOCAR_ROOT, "downloads")
PATHS_CONFIG_YAML = os.path.join(ECOCAR_ROOT, "paths_config.yaml")

# Local fast‑IO mirror (Colab local SSD — extracted from Drive tars)
LOCAL_DATASET     = "/content/bdd100k_yolo"
RAW_BDD_ROOT      = "/content/bdd100k_raw"

# BDD100K original image size
BDD_IMG_W, BDD_IMG_H = 1280, 720

# ── Vehicle detection classes ────────────────────────────────────────
VEHICLE_CLASSES = ["car", "truck", "bus", "motorcycle", "bicycle"]
NUM_CLASSES = len(VEHICLE_CLASSES)
BDD_FULL_CLASSES = [
    "person", "rider", "car", "truck", "bus",
    "train", "motorcycle", "bicycle", "traffic light", "traffic sign",
]
BDD_TO_VEHICLE = {2: 0, 3: 1, 4: 2, 6: 3, 7: 4}
EXPANDED_CLASSES = ["car", "truck", "bus", "train", "motorcycle", "bicycle", "rider"]
BDD_TO_EXPANDED = {2: 0, 3: 1, 4: 2, 5: 3, 6: 4, 7: 5, 1: 6}

# ── Lane categories from BDD100K ────────────────────────────────────
LANE_CATEGORIES = [
    "lane/single white",
    "lane/single yellow",
    "lane/single other",
    "lane/double white",
    "lane/double yellow",
    "lane/double other",
    "lane/road curb",
    "lane/crosswalk",
]
LANE_TRAIN_CATS = [c for c in LANE_CATEGORIES if "crosswalk" not in c]
NUM_LANE_TYPES = len(LANE_TRAIN_CATS)
LANE_CAT_TO_ID = {c: i for i, c in enumerate(LANE_TRAIN_CATS)}


def _safe_load_paths_yaml() -> Dict:
    if os.path.isfile(PATHS_CONFIG_YAML):
        try:
            with open(PATHS_CONFIG_YAML, 'r') as f:
                data = yaml.safe_load(f) or {}
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}


def _raw_roots() -> List[str]:
    """Candidate extraction roots for raw BDD labels/images."""
    y = _safe_load_paths_yaml()
    roots = []
    # notebook02/07 style custom raw roots, if present
    for k in [
        'bdd_raw_dir', 'bdd_root', 'bdd100k_root', 'bdd_dataset_root',
        'raw_bdd_dir', 'raw_bdd_root'
    ]:
        v = y.get(k)
        if isinstance(v, str) and v.strip():
            roots.append(v)
    roots += [
        RAW_BDD_ROOT,
        os.path.join(ECOCAR_ROOT, 'datasets', 'bdd100k_raw'),
        os.path.join(ECOCAR_ROOT, 'datasets', 'bdd100k'),
        os.path.join(DOWNLOADS_DIR, 'bdd100k'),
        DOWNLOADS_DIR,
    ]
    out=[]
    seen=set()
    for r in roots:
        r=os.path.normpath(r)
        if r not in seen:
            seen.add(r)
            out.append(r)
    return out


def _labels_zip_candidates() -> List[str]:
    return [
        os.path.join(DOWNLOADS_DIR, 'bdd100k_labels.zip'),
        os.path.join(ECOCAR_ROOT, 'datasets', 'bdd100k_labels.zip'),
        os.path.join(ECOCAR_ROOT, 'downloads', 'labels.zip'),
    ]


def ensure_bdd_labels_extracted(verbose: bool = False) -> str:
    """Ensure official BDD labels zip is extracted to a usable raw root.

    Official old-format labels zip usually expands to:
      bdd100k/labels/bdd100k_labels_images_train.json
      bdd100k/labels/bdd100k_labels_images_val.json
    as used by the official bdd2coco.py script.
    """
    # If already extracted anywhere, use it.
    for root in _raw_roots():
        for rel in [
            os.path.join('bdd100k', 'labels', 'bdd100k_labels_images_train.json'),
            os.path.join('labels', 'bdd100k_labels_images_train.json'),
            os.path.join('bdd100k', 'labels', 'lane', 'polygons', 'lane_train.json'),
            os.path.join('labels', 'lane', 'polygons', 'lane_train.json'),
        ]:
            if os.path.isfile(os.path.join(root, rel)):
                return root

    zip_path = next((p for p in _labels_zip_candidates() if os.path.isfile(p)), None)
    target_root = RAW_BDD_ROOT
    if zip_path is None:
        return target_root

    os.makedirs(target_root, exist_ok=True)
    marker = os.path.join(target_root, '.bdd_labels_extracted')
    if not os.path.isfile(marker):
        if verbose:
            print(f'Extracting BDD labels zip: {zip_path} -> {target_root}')
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(target_root)
        with open(marker, 'w') as f:
            f.write(zip_path)
    return target_root


def lane_json_candidates(split: str = 'train') -> List[str]:
    """Return candidates in the same spirit as official BDD releases.

    Priority order:
    1) old official consolidated labels JSON used by bdd2coco.py
    2) new task-specific lane polygons JSON
    """
    roots = [ensure_bdd_labels_extracted(False)] + _raw_roots()
    cands=[]
    for root in roots:
        cands.extend([
            os.path.join(root, 'bdd100k', 'labels', f'bdd100k_labels_images_{split}.json'),
            os.path.join(root, 'labels', f'bdd100k_labels_images_{split}.json'),
            os.path.join(root, 'bdd100k', 'labels', 'lane', 'polygons', f'lane_{split}.json'),
            os.path.join(root, 'labels', 'lane', 'polygons', f'lane_{split}.json'),
        ])
    # unique preserve order
    out=[]
    seen=set()
    for p in cands:
        p=os.path.normpath(p)
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def find_lane_labels(split: str = 'train') -> Optional[str]:
    for path in lane_json_candidates(split):
        if os.path.isfile(path):
            return path
    return None


def lane_search_debug(split: str = 'train') -> Dict[str, object]:
    raw_root = ensure_bdd_labels_extracted(verbose=False)
    return {
        'paths_config_yaml': PATHS_CONFIG_YAML,
        'zip_candidates': _labels_zip_candidates(),
        'raw_root': raw_root,
        'candidates': lane_json_candidates(split),
        'chosen': find_lane_labels(split),
    }

# ── Default training config ─────────────────────────────────────────
@dataclass
class Config:
    run_name: str = 'dualpath_v1'
    device: str = 'cuda'
    amp: bool = True
    seed: int = 42
    dataset_root: str = LOCAL_DATASET
    img_size: int = 640
    batch_size: int = 8
    num_workers: int = 4
    max_lanes: int = 10
    lane_points: int = 72
    use_expanded_classes: bool = False
    backbone: str = 'resnet50'
    pretrained: bool = True
    fpn_channels: int = 256
    det_num_queries: int = 100
    det_enc_layers: int = 1
    det_dec_layers: int = 3
    det_dim: int = 256
    det_nhead: int = 8
    det_ffn_dim: int = 1024
    det_dropout: float = 0.0
    lane_num_queries: int = 10
    lane_enc_layers: int = 1
    lane_dec_layers: int = 3
    lane_dim: int = 256
    lane_nhead: int = 8
    lane_ffn_dim: int = 1024
    lane_dropout: float = 0.0
    cross_attn: bool = True
    cross_attn_layers: int = 1
    lr: float = 1e-4
    backbone_lr_scale: float = 0.1
    weight_decay: float = 1e-4
    epochs: int = 50
    warmup_epochs: int = 5
    min_lr_ratio: float = 0.01
    det_cls_weight: float = 2.0
    det_l1_weight: float = 5.0
    det_giou_weight: float = 2.0
    lane_exist_weight: float = 2.0
    lane_pts_weight: float = 5.0
    lane_type_weight: float = 1.0
    det_task_weight: float = 1.0
    lane_task_weight: float = 1.0
    conf_thresh: float = 0.3
    nms_iou: float = 0.5
    lane_match_thresh: float = 15.0
    save_dir: str = ''
    patience: int = 15

    def __post_init__(self):
        if not self.save_dir:
            self.save_dir = os.path.join(TRAINING_RUNS, self.run_name)

    def to_dict(self):
        return asdict(self)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def from_yaml(cls, path: str) -> 'Config':
        with open(path) as f:
            d = yaml.safe_load(f)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_dict(cls, d: dict) -> 'Config':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def ensure_dirs(cfg: Config):
    for d in [cfg.save_dir, os.path.join(cfg.save_dir, 'weights'), WEIGHTS_DIR, OUTPUTS_DIR]:
        os.makedirs(d, exist_ok=True)
