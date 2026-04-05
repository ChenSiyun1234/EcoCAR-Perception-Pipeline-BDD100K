
from __future__ import annotations
import json
import os
import shutil
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional

VEHICLE_CATEGORIES = ['car', 'truck', 'bus', 'motorcycle', 'bicycle']
VEHICLE_TO_ID = {name: i for i, name in enumerate(VEHICLE_CATEGORIES)}


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _zip_members(zip_path: str | Path) -> List[str]:
    with zipfile.ZipFile(zip_path, 'r') as zf:
        return zf.namelist()


def unzip_if_needed(zip_path: str | Path, dest_root: str | Path) -> Path:
    zip_path = Path(zip_path)
    dest_root = ensure_dir(dest_root)
    marker = dest_root / f'.extracted_{zip_path.stem}'
    if marker.exists():
        return dest_root
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(dest_root)
    marker.write_text('ok')
    return dest_root


def _find_by_suffix(root: str | Path, suffixes: List[str]) -> Optional[Path]:
    root = Path(root)
    suffixes = [s.lower() for s in suffixes]
    for p in root.rglob('*'):
        if p.is_file() and any(str(p).lower().endswith(s) for s in suffixes):
            return p
    return None


def _find_all_jsons(root: str | Path) -> List[Path]:
    return sorted([p for p in Path(root).rglob('*.json') if p.is_file()])


def _score_json_candidate(path: Path) -> int:
    s = str(path).lower()
    score = 0
    if 'train' in s:
        score += 4
    if 'val' in s or 'valid' in s:
        score += 4
    if 'detect' in s or 'det_' in s or 'labels_images' in s:
        score += 3
    if 'lane' in s or 'seg' in s or 'drivable' in s:
        score -= 4
    return score


def _classify_json_by_content(path: Path) -> Optional[str]:
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except Exception:
        return None
    if not isinstance(data, list) or len(data) == 0:
        return None
    sample = data[0]
    if isinstance(sample, dict):
        labels = sample.get('labels')
        name = sample.get('name')
        if isinstance(labels, list) and isinstance(name, str):
            cats = []
            for lab in labels[:20]:
                if isinstance(lab, dict):
                    c = lab.get('category')
                    if isinstance(c, str):
                        cats.append(c)
            if cats:
                return 'detection'
    return None


def locate_detection_jsons(raw_root: str | Path) -> Tuple[Path, Path]:
    raw_root = Path(raw_root)
    jsons = _find_all_jsons(raw_root)
    det_jsons = []
    for p in jsons:
        kind = _classify_json_by_content(p)
        if kind == 'detection':
            det_jsons.append(p)
    if not det_jsons:
        raise FileNotFoundError(f'No detection-style JSONs found under {raw_root}')

    train_candidates = [p for p in det_jsons if 'train' in str(p).lower()]
    val_candidates = [p for p in det_jsons if 'val' in str(p).lower() or 'valid' in str(p).lower()]

    if not train_candidates or not val_candidates:
        # fallback: pick best-scored pair by name and size
        det_jsons = sorted(det_jsons, key=lambda p: (-_score_json_candidate(p), str(p)))
        train_candidates = det_jsons
        val_candidates = det_jsons

    train_json = sorted(train_candidates, key=lambda p: (-_score_json_candidate(p), str(p)))[0]
    val_json = sorted(val_candidates, key=lambda p: (-_score_json_candidate(p), str(p)))[0]

    if train_json == val_json:
        # choose another val candidate if possible
        for p in det_jsons:
            if p != train_json and ('val' in str(p).lower() or 'valid' in str(p).lower()):
                val_json = p
                break

    if train_json == val_json:
        raise FileNotFoundError(
            f'Found detection JSONs under {raw_root}, but could not distinguish train/val. Candidates: {[str(p) for p in det_jsons[:10]]}'
        )
    return train_json, val_json


def locate_image_dirs(raw_root: str | Path) -> Tuple[Path, Path]:
    raw_root = Path(raw_root)
    train_dir = None
    val_dir = None
    for p in raw_root.rglob('*'):
        if p.is_dir():
            s = str(p).lower().replace('\','/')
            if s.endswith('/train') and '/images/' in s and ('100k' in s or '10k' in s):
                train_dir = p if train_dir is None else train_dir
            if s.endswith('/val') and '/images/' in s and ('100k' in s or '10k' in s):
                val_dir = p if val_dir is None else val_dir
    if train_dir is None or val_dir is None:
        # fallback broad search
        dirs = [p for p in raw_root.rglob('*') if p.is_dir()]
        for p in dirs:
            s = str(p).lower().replace('\','/')
            if train_dir is None and s.endswith('/train') and '/images/' in s:
                train_dir = p
            if val_dir is None and s.endswith('/val') and '/images/' in s:
                val_dir = p
    if train_dir is None or val_dir is None:
        raise FileNotFoundError(f'Could not locate images train/val dirs under {raw_root}')
    return train_dir, val_dir


def locate_lane_json(raw_root: str | Path) -> Optional[Path]:
    raw_root = Path(raw_root)
    candidates = []
    for p in raw_root.rglob('*.json'):
        s = str(p).lower()
        if 'lane' in s and ('train' in s or 'val' in s or 'labels' in s or 'poly' in s):
            candidates.append(p)
    if candidates:
        return sorted(candidates, key=lambda p: (0 if 'lane' in str(p).lower() else 1, str(p)))[0]
    return None


def locate_seg_maps_root(raw_root: str | Path) -> Optional[Path]:
    raw_root = Path(raw_root)
    for p in raw_root.rglob('*'):
        if p.is_dir():
            s = str(p).lower().replace('\','/')
            if '/seg/' in s or 'seg_maps' in s or s.endswith('/train') and 'labels' in s:
                return p
    return None


def _extract_xywh(obj: dict) -> Optional[Tuple[float, float, float, float]]:
    box2d = obj.get('box2d')
    if not isinstance(box2d, dict):
        return None
    try:
        x1 = float(box2d['x1']); y1 = float(box2d['y1']); x2 = float(box2d['x2']); y2 = float(box2d['y2'])
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    xc = (x1 + x2) / 2.0
    yc = (y1 + y2) / 2.0
    w = x2 - x1
    h = y2 - y1
    return xc, yc, w, h


def convert_detection_json_to_vehicle_yolo(json_path: str | Path, labels_out: str | Path) -> Dict[str, int]:
    labels_out = ensure_dir(labels_out)
    with open(json_path, 'r') as f:
        data = json.load(f)
    counts = Counter()
    written = 0
    skipped_no_box = 0
    for item in data:
        name = item.get('name')
        attrs = item.get('attributes', {}) or {}
        img_w = float(item.get('width', 1280) or 1280)
        img_h = float(item.get('height', 720) or 720)
        labels = item.get('labels', []) or []
        rows = []
        for lab in labels:
            cat = lab.get('category')
            if cat not in VEHICLE_TO_ID:
                continue
            box = _extract_xywh(lab)
            if box is None:
                skipped_no_box += 1
                continue
            xc, yc, w, h = box
            rows.append(f"{VEHICLE_TO_ID[cat]} {xc/img_w:.6f} {yc/img_h:.6f} {w/img_w:.6f} {h/img_h:.6f}")
            counts[cat] += 1
        stem = Path(name).stem if name else f'item_{written:06d}'
        (labels_out / f'{stem}.txt').write_text('
'.join(rows))
        written += 1
    return {
        'files_written': written,
        'skipped_no_box': skipped_no_box,
        **{k: int(counts[k]) for k in VEHICLE_CATEGORIES},
    }


def _link_or_copy_images(src_dir: str | Path, dst_dir: str | Path):
    src_dir = Path(src_dir); dst_dir = ensure_dir(dst_dir)
    count = 0
    for p in src_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
            continue
        dst = dst_dir / p.name
        if dst.exists():
            count += 1
            continue
        try:
            os.symlink(p, dst)
        except Exception:
            shutil.copy2(p, dst)
        count += 1
    return count


def write_vehicle_yaml(dataset_root: str | Path) -> Path:
    dataset_root = Path(dataset_root)
    yaml_path = dataset_root / 'bdd100k_vehicle5.yaml'
    content = (
        f"path: {dataset_root}
"
        "train: images/train
"
        "val: images/val
"
        "nc: 5
"
        "names:
"
        "  0: car
"
        "  1: truck
"
        "  2: bus
"
        "  3: motorcycle
"
        "  4: bicycle
"
    )
    yaml_path.write_text(content)
    return yaml_path


def write_paths_config(dataset_root: str | Path, raw_root: str | Path, lane_json: Optional[str | Path], seg_root: Optional[str | Path]) -> Path:
    dataset_root = Path(dataset_root)
    p = dataset_root / 'paths_config.yaml'
    lines = [f"dataset_root: {dataset_root}", f"raw_root: {Path(raw_root)}"]
    lines.append(f"lane_json: {Path(lane_json) if lane_json else ''}")
    lines.append(f"seg_maps_root: {Path(seg_root) if seg_root else ''}")
    p.write_text('
'.join(lines) + '
')
    return p


def inspect_download_archives(downloads_dir: str | Path) -> Dict[str, List[str]]:
    downloads_dir = Path(downloads_dir)
    out = {}
    for name in ['bdd100k_labels.zip', 'bdd100k_images_100k.zip', 'bdd100k_seg_maps.zip']:
        zp = downloads_dir / name
        if zp.exists():
            members = _zip_members(zp)
            out[name] = members[:30]
        else:
            out[name] = []
    return out


def rebuild_dualpath_dataset(downloads_dir: str | Path, raw_root: str | Path, output_root: str | Path, force_reextract: bool = False) -> Dict[str, object]:
    downloads_dir = Path(downloads_dir)
    raw_root = ensure_dir(raw_root)
    output_root = ensure_dir(output_root)

    required = {
        'labels': downloads_dir / 'bdd100k_labels.zip',
        'images': downloads_dir / 'bdd100k_images_100k.zip',
        'seg': downloads_dir / 'bdd100k_seg_maps.zip',
    }
    missing = [str(p) for p in required.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f'Missing required zip files: {missing}')

    for key, zp in required.items():
        marker = raw_root / f'.extracted_{zp.stem}'
        if force_reextract and marker.exists():
            marker.unlink()
        unzip_if_needed(zp, raw_root)

    train_json, val_json = locate_detection_jsons(raw_root)
    train_img_dir, val_img_dir = locate_image_dirs(raw_root)
    lane_json = locate_lane_json(raw_root)
    seg_root = locate_seg_maps_root(raw_root)

    img_train_out = ensure_dir(output_root / 'images' / 'train')
    img_val_out = ensure_dir(output_root / 'images' / 'val')
    lbl_train_out = ensure_dir(output_root / 'labels' / 'train')
    lbl_val_out = ensure_dir(output_root / 'labels' / 'val')

    train_img_count = _link_or_copy_images(train_img_dir, img_train_out)
    val_img_count = _link_or_copy_images(val_img_dir, img_val_out)
    train_counts = convert_detection_json_to_vehicle_yolo(train_json, lbl_train_out)
    val_counts = convert_detection_json_to_vehicle_yolo(val_json, lbl_val_out)
    yaml_path = write_vehicle_yaml(output_root)
    paths_cfg = write_paths_config(output_root, raw_root, lane_json, seg_root)

    return {
        'train_json': str(train_json),
        'val_json': str(val_json),
        'train_image_dir': str(train_img_dir),
        'val_image_dir': str(val_img_dir),
        'lane_json': str(lane_json) if lane_json else None,
        'seg_maps_root': str(seg_root) if seg_root else None,
        'train_image_count': int(train_img_count),
        'val_image_count': int(val_img_count),
        'train_counts': train_counts,
        'val_counts': val_counts,
        'yaml_path': str(yaml_path),
        'paths_config': str(paths_cfg),
    }
