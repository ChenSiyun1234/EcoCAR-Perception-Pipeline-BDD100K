"""
One-shot updater. Rewrites stage1 post-02 notebooks so they default to
YOLOPX, fail loudly on missing checkpoint/dataset, and never silently
fall back to the older YOLOP family.

Run from the project repo root:
    python yolop_vehicle_lane/stage1/notebooks/_update_to_yolopx.py
"""
from __future__ import annotations

import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent

# -------------------------------------------------------------------------
# Standard YOLOPX-aware config block
# -------------------------------------------------------------------------
YOLOPX_YAML_MAP = """yaml_map = {
    'YOLOPX':             os.path.join(REPO_ROOT, 'stage1', 'configs', 'yolopx_vehicle_lane_baseline.yaml'),
    'YOLOP':              os.path.join(REPO_ROOT, 'stage1', 'configs', 'yolop_vehicle_lane_baseline.yaml'),
    'YOLOPv2-paper-no-da': os.path.join(REPO_ROOT, 'stage1', 'configs', 'yolopv2_paper_no_da.yaml'),
    'YOLOPv2-best-row':   os.path.join(REPO_ROOT, 'stage1', 'configs', 'yolopv2_best_row.yaml'),
    'YOLOPv2-focal-only': os.path.join(REPO_ROOT, 'stage1', 'configs', 'yolopv2_focal_only_ablation.yaml'),
}"""

YOLOPX_RUN_NAME_MAP = """run_name_map = {
    'YOLOPX':             'yolopx',
    'YOLOP':              'yolop',
    'YOLOPv2-paper-no-da': 'yolopv2_paper_no_da',
    'YOLOPv2-best-row':   'yolopv2_best_row',
    'YOLOPv2-focal-only': 'yolopv2_focal_only',
}"""


# -------------------------------------------------------------------------
# Cell content per notebook
# -------------------------------------------------------------------------
NB_02B_CELL2 = f"""import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as T
from torch.utils.data import DataLoader

from lib.config import cfg
from lib.models import get_net
from lib.core import get_loss
from lib.dataset import BddDataset
from lib.utils.drive_dataset import (
    ensure_local_dataset_from_drive, find_raw_bdd_root,
    resolve_bdd_images_100k_dir, resolve_bdd_labels_100k_dir,
)

# Stage1 default = YOLOPX. Flip CONFIG manually only when comparing against
# an older baseline. The notebook will fail loudly if the matching YOLOPX
# checkpoint or config is missing — never silently falls back to YOLOP.
CONFIG = 'YOLOPX'
CKPT_NAME = 'best_joint.pth'    # 'best_joint.pth' | 'best_det.pth' | 'best_lane.pth' | 'latest.pth' | None

{YOLOPX_YAML_MAP}
{YOLOPX_RUN_NAME_MAP}

assert CONFIG in yaml_map, f'Unknown CONFIG={{CONFIG}}. Valid: {{list(yaml_map)}}'
yaml_path = yaml_map[CONFIG]
assert os.path.exists(yaml_path), f'Missing config YAML: {{yaml_path}}'

cfg.defrost()
cfg.merge_from_file(yaml_path)

ECOCAR_ROOT = '/content/drive/MyDrive/EcoCAR'
DATASET_ROOT = ensure_local_dataset_from_drive('bdd100k_vehicle5', ECOCAR_ROOT)
RAW_BDD_ROOT = find_raw_bdd_root(ECOCAR_ROOT)
BDD_IMAGES = resolve_bdd_images_100k_dir(RAW_BDD_ROOT, ECOCAR_ROOT)
BDD_LABELS = resolve_bdd_labels_100k_dir(RAW_BDD_ROOT)
cfg.DATASET.ROOT = DATASET_ROOT
cfg.DATASET.DATAROOT = BDD_IMAGES
cfg.DATASET.LABELROOT = BDD_LABELS
cfg.DATASET.LANEROOT = os.path.join(DATASET_ROOT, 'masks')

run_name = run_name_map[CONFIG]
cfg.DRIVE.ROOT = ECOCAR_ROOT
cfg.DRIVE.CHECKPOINT_DIR = os.path.join(ECOCAR_ROOT, 'yolop_vehicle_lane', 'stage1', 'checkpoints', run_name)
cfg.DRIVE.METRICS_DIR    = os.path.join(ECOCAR_ROOT, 'yolop_vehicle_lane', 'stage1', 'metrics',     run_name)

cfg.TEST.PLOTS = False
cfg.freeze()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Loud preflight: expose every path the rest of the notebook depends on.
print('=' * 64)
print('[Stage1 / 02b diagnostics preflight]')
print(f'  Model family    : {{CONFIG}}')
print(f'  Model.NAME      : {{cfg.MODEL.NAME}}')
print(f'  Config YAML     : {{yaml_path}}')
print(f'  Checkpoint dir  : {{cfg.DRIVE.CHECKPOINT_DIR}}')
print(f'  Checkpoint file : {{CKPT_NAME}}')
print(f'  Dataset root    : {{DATASET_ROOT}}')
print(f'  BDD images      : {{BDD_IMAGES}}')
print(f'  Metrics dir     : {{cfg.DRIVE.METRICS_DIR}}')
print(f'  NC / classes    : {{cfg.MODEL.NC}}  {{cfg.MODEL.VEHICLE_CLASSES}}')
print(f'  Device          : {{device}}')
print('=' * 64)

# Fail loudly if the YOLOPX checkpoint is missing — never fall back to YOLOP.
if CKPT_NAME is not None:
    expected_ckpt = os.path.join(cfg.DRIVE.CHECKPOINT_DIR, CKPT_NAME)
    if not os.path.exists(expected_ckpt):
        raise FileNotFoundError(
            f'Expected {{CONFIG}} checkpoint not found: {{expected_ckpt}}\\n'
            f'Run notebook 02 (YOLOPX baseline) first to produce {{CKPT_NAME}}, '
            f'or change CKPT_NAME to None to run with random init.'
        )
"""


NB_03_CELL2 = f"""import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as T

from lib.config import cfg
from lib.models import get_net
from lib.core import get_loss, validate
from lib.dataset import BddDataset
from lib.utils.drive_dataset import (
    ensure_local_dataset_from_drive,
    find_raw_bdd_root,
    resolve_bdd_images_100k_dir,
    resolve_bdd_labels_100k_dir,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Stage1 default eval target = YOLOPX. Fail loud if the YOLOPX checkpoint
# is missing rather than silently evaluating the older YOLOP baseline.
CONFIG = 'YOLOPX'

{YOLOPX_YAML_MAP}
{YOLOPX_RUN_NAME_MAP}

assert CONFIG in yaml_map, f'Unknown CONFIG={{CONFIG}}. Valid: {{list(yaml_map)}}'
yaml_path = yaml_map[CONFIG]
assert os.path.exists(yaml_path), f'Missing config YAML: {{yaml_path}}'

cfg.defrost()
cfg.merge_from_file(yaml_path)

ECOCAR_ROOT = '/content/drive/MyDrive/EcoCAR'
DATASET_ROOT = ensure_local_dataset_from_drive('bdd100k_vehicle5', ECOCAR_ROOT)
RAW_BDD_ROOT = find_raw_bdd_root(ECOCAR_ROOT)
BDD_IMAGES = resolve_bdd_images_100k_dir(RAW_BDD_ROOT, ECOCAR_ROOT)
BDD_LABELS = resolve_bdd_labels_100k_dir(RAW_BDD_ROOT)
cfg.DATASET.ROOT = DATASET_ROOT
cfg.DATASET.DATAROOT = BDD_IMAGES
cfg.DATASET.LABELROOT = BDD_LABELS
cfg.DATASET.LANEROOT = os.path.join(DATASET_ROOT, 'masks')

run_name = run_name_map[CONFIG]
cfg.DRIVE.ROOT = ECOCAR_ROOT
cfg.DRIVE.CHECKPOINT_DIR = os.path.join(ECOCAR_ROOT, 'yolop_vehicle_lane', 'stage1', 'checkpoints', run_name)
cfg.DRIVE.METRICS_DIR    = os.path.join(ECOCAR_ROOT, 'yolop_vehicle_lane', 'stage1', 'metrics',     run_name)

cfg.TEST.PLOTS = True
cfg.freeze()

print('=' * 64)
print('[Stage1 / 03 eval preflight]')
print(f'  Model family    : {{CONFIG}}')
print(f'  Model.NAME      : {{cfg.MODEL.NAME}}')
print(f'  Config YAML     : {{yaml_path}}')
print(f'  Checkpoint dir  : {{cfg.DRIVE.CHECKPOINT_DIR}}')
print(f'  Dataset root    : {{DATASET_ROOT}}')
print(f'  BDD images      : {{BDD_IMAGES}}')
print(f'  Metrics dir     : {{cfg.DRIVE.METRICS_DIR}}')
print(f'  NC / classes    : {{cfg.MODEL.NC}}  {{cfg.MODEL.VEHICLE_CLASSES}}')
print(f'  Val image size  : {{tuple(cfg.TEST.IMAGE_SIZE)}}')
print(f'  Dice gain       : {{cfg.LOSS.LL_DICE_GAIN}}')
print('=' * 64)
"""


# Cell 3 of nb 03 — load best.pth and fail loudly with a YOLOPX-specific message.
NB_03_CELL3 = """# ── Load model + best checkpoint ──
# get_net reads cfg.MODEL.NC and names — do NOT override them.
model = get_net(cfg).to(device)
model.gr = 1.0

ckpt_path = os.path.join(cfg.DRIVE.CHECKPOINT_DIR, 'best.pth')
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(
        f'Expected {CONFIG} checkpoint not found: {ckpt_path}\\n'
        f'Run notebook 02 first to produce best.pth for the {CONFIG} run.'
    )
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt['state_dict'])
print(f'[{CONFIG}] Loaded checkpoint  : {ckpt_path}')
print(f'  epoch={ckpt.get("epoch", "?")}  nc={model.nc}  names={model.names}')
print(f'  best_map={ckpt.get("best_map", "N/A")}, best_ll_iou={ckpt.get("best_ll_iou", "N/A")}')
"""


NB_06_CELL2 = f"""import os
import sys
REPO_ROOT = '/content/drive/MyDrive/EcoCAR/yolop_vehicle_lane'
os.chdir(REPO_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import json
import logging
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader

from lib.config import cfg
from lib.models import get_net
from lib.core import get_loss, validate
from lib.dataset import BddDataset
from lib.utils.drive_dataset import (
    ensure_local_dataset_from_drive,
    find_raw_bdd_root,
    resolve_bdd_images_100k_dir,
    resolve_bdd_labels_100k_dir,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Stage1 default final-eval / export target = YOLOPX.
CONFIG = 'YOLOPX'
CHECKPOINT_NAME = 'best_joint.pth'
RUN_EVAL = True
EXPORT_ONNX = True
EXPORT_TORCHSCRIPT = True

{YOLOPX_YAML_MAP}
{YOLOPX_RUN_NAME_MAP}

assert CONFIG in yaml_map, f'Unknown CONFIG={{CONFIG}}. Valid: {{list(yaml_map)}}'
yaml_path = yaml_map[CONFIG]
assert os.path.exists(yaml_path), f'Missing config YAML: {{yaml_path}}'

cfg.defrost()
cfg.merge_from_file(yaml_path)

ECOCAR_ROOT = '/content/drive/MyDrive/EcoCAR'
DATASET_ROOT = ensure_local_dataset_from_drive('bdd100k_vehicle5', ECOCAR_ROOT)
RAW_BDD_ROOT = find_raw_bdd_root(ECOCAR_ROOT)
BDD_IMAGES = resolve_bdd_images_100k_dir(RAW_BDD_ROOT, ECOCAR_ROOT)
BDD_LABELS = resolve_bdd_labels_100k_dir(RAW_BDD_ROOT)

cfg.DATASET.ROOT = DATASET_ROOT
cfg.DATASET.DATAROOT = BDD_IMAGES
cfg.DATASET.LABELROOT = BDD_LABELS
cfg.DATASET.LANEROOT = os.path.join(DATASET_ROOT, 'masks')

run_name = run_name_map[CONFIG]
cfg.DRIVE.ROOT = ECOCAR_ROOT
cfg.DRIVE.CHECKPOINT_DIR = os.path.join(ECOCAR_ROOT, 'yolop_vehicle_lane', 'stage1', 'checkpoints', run_name)
cfg.DRIVE.METRICS_DIR    = os.path.join(ECOCAR_ROOT, 'yolop_vehicle_lane', 'stage1', 'metrics',     run_name)
cfg.freeze()

# Fail loudly if expected YOLOPX checkpoints are missing — never fall back.
checkpoint_candidates = [
    os.path.join(cfg.DRIVE.CHECKPOINT_DIR, CHECKPOINT_NAME),
    os.path.join(cfg.DRIVE.CHECKPOINT_DIR, 'best_joint.pth'),
    os.path.join(cfg.DRIVE.CHECKPOINT_DIR, 'best.pth'),
    os.path.join(cfg.DRIVE.CHECKPOINT_DIR, 'latest.pth'),
]
ckpt_path = next((p for p in checkpoint_candidates if os.path.exists(p)), None)
if ckpt_path is None:
    raise FileNotFoundError(
        f'No {{CONFIG}} checkpoint found in {{cfg.DRIVE.CHECKPOINT_DIR}}.\\n'
        f'Tried: {{checkpoint_candidates}}\\n'
        f'Run notebook 02 first to produce a YOLOPX checkpoint.'
    )

model = get_net(cfg).to(device)
model.gr = 1.0
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt['state_dict'])
model.eval()

print('=' * 64)
print('[Stage1 / 06 final eval+export preflight]')
print(f'  Model family    : {{CONFIG}}')
print(f'  Model.NAME      : {{cfg.MODEL.NAME}}')
print(f'  Config YAML     : {{yaml_path}}')
print(f'  Checkpoint dir  : {{cfg.DRIVE.CHECKPOINT_DIR}}')
print(f'  Checkpoint file : {{ckpt_path}}')
print(f'  Dataset root    : {{DATASET_ROOT}}')
print(f'  BDD images      : {{BDD_IMAGES}}')
print(f'  Metrics dir     : {{cfg.DRIVE.METRICS_DIR}}')
print(f'  Val image size  : {{tuple(cfg.TEST.IMAGE_SIZE)}}')
print(f'  epoch={{ckpt.get("epoch", "?")}}  nc={{model.nc}}  names={{model.names}}')
print('=' * 64)
"""


NB_07_CELL3 = f"""# ── Load model ──
# Stage1 default profiling target = YOLOPX. Fail loudly if the YOLOPX
# checkpoint is missing — never silently profile the older YOLOP model.
CONFIG = 'YOLOPX'
CHECKPOINT_NAME = 'best_joint.pth'

{YOLOPX_YAML_MAP}
{YOLOPX_RUN_NAME_MAP}

assert CONFIG in yaml_map, f'Unknown CONFIG={{CONFIG}}. Valid: {{list(yaml_map)}}'
yaml_path = yaml_map[CONFIG]
assert os.path.exists(yaml_path), f'Missing config YAML: {{yaml_path}}'

cfg.defrost()
cfg.merge_from_file(yaml_path)
run_name = run_name_map[CONFIG]
cfg.DRIVE.CHECKPOINT_DIR = os.path.join(REPO_ROOT, 'stage1', 'checkpoints', run_name)
cfg.DRIVE.METRICS_DIR = os.path.join(REPO_ROOT, 'stage1', 'metrics', run_name)
cfg.freeze()

checkpoint_candidates = [
    os.path.join(cfg.DRIVE.CHECKPOINT_DIR, CHECKPOINT_NAME),
    os.path.join(cfg.DRIVE.CHECKPOINT_DIR, 'best_joint.pth'),
    os.path.join(cfg.DRIVE.CHECKPOINT_DIR, 'best.pth'),
    os.path.join(cfg.DRIVE.CHECKPOINT_DIR, 'latest.pth'),
]
ckpt_path = next((p for p in checkpoint_candidates if os.path.exists(p)), None)
if ckpt_path is None:
    raise FileNotFoundError(
        f'No {{CONFIG}} checkpoint found in {{cfg.DRIVE.CHECKPOINT_DIR}}.\\n'
        f'Tried: {{checkpoint_candidates}}\\n'
        f'Run notebook 02 first to produce a YOLOPX checkpoint.'
    )

model = get_net(cfg).to(device)
model.gr = 1.0
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt['state_dict'])
model.eval()
use_fp16 = torch.cuda.is_available()
if use_fp16:
    model.half()
num_params = sum(p.numel() for p in model.parameters())

print('=' * 64)
print('[Stage1 / 07 video-profile preflight]')
print(f'  Model family    : {{CONFIG}}')
print(f'  Model.NAME      : {{cfg.MODEL.NAME}}')
print(f'  Config YAML     : {{yaml_path}}')
print(f'  Checkpoint dir  : {{cfg.DRIVE.CHECKPOINT_DIR}}')
print(f'  Checkpoint file : {{ckpt_path}}')
print(f'  Metrics dir     : {{cfg.DRIVE.METRICS_DIR}}')
print(f'  epoch={{ckpt.get("epoch", "?")}} params={{num_params/1e6:.2f}} M  nc={{model.nc}}')
print(f'  FP16            : {{use_fp16}}')
print('=' * 64)

with torch.no_grad():
    z = torch.zeros(1, 3, 640, 640, device=device, dtype=torch.float16 if use_fp16 else torch.float32)
    _det, lane_logits = model(z)
    _det_p, lane_prob = model.predict(z)
print(f'forward() lane shape : {{tuple(lane_logits.shape)}}')
print(f'predict() lane shape : {{tuple(lane_prob.shape)}}')
"""


# -------------------------------------------------------------------------
# Apply
# -------------------------------------------------------------------------
def _set_cell_source(nb, idx, src):
    nb['cells'][idx]['source'] = src.splitlines(keepends=True)
    # Drop outputs and execution count for code cells we replaced.
    if nb['cells'][idx]['cell_type'] == 'code':
        nb['cells'][idx]['outputs'] = []
        nb['cells'][idx]['execution_count'] = None


def _update(nb_path: Path, edits):
    nb = json.loads(nb_path.read_text(encoding='utf-8'))
    for idx, src in edits:
        _set_cell_source(nb, idx, src)
    nb_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f'updated: {nb_path.name}')


def main():
    _update(NB_DIR / '02b_stage1_diagnostics.ipynb', [(2, NB_02B_CELL2)])
    _update(NB_DIR / '03_eval_and_backbone_ablation.ipynb', [(2, NB_03_CELL2), (3, NB_03_CELL3)])
    _update(NB_DIR / '06_final_train_eval_export.ipynb', [(2, NB_06_CELL2)])
    _update(NB_DIR / '07_a5000_video_profile.ipynb', [(3, NB_07_CELL3)])


if __name__ == '__main__':
    main()
