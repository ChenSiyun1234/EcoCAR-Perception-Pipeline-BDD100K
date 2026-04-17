"""
Lane mask rendering utilities.
Migrated from yolo26_pipeline/src/lane_utils.py.
Renders BDD100K poly2d lane annotations into binary segmentation masks.
Supports both consolidated JSON files and the older per-image JSON directory
layout used by the working DETR_GeoLane pipeline.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from tqdm import tqdm

from .lane_targets import extract_lane_labels_any, parse_poly2d


BDD_LANE_CATEGORIES = [
    "lane/crosswalk",
    "lane/double other",
    "lane/double white",
    "lane/double yellow",
    "lane/road curb",
    "lane/single other",
    "lane/single white",
    "lane/single yellow",
]


def _record_image_name(record, fallback_json_path=None):
    if not isinstance(record, dict):
        if fallback_json_path:
            return Path(fallback_json_path).stem + '.jpg'
        return ''
    for key in ['name', 'image', 'imageName', 'filename', 'id']:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            base = os.path.basename(value)
            if '.' not in base:
                base += '.jpg'
            return base
    if fallback_json_path:
        return Path(fallback_json_path).stem + '.jpg'
    return ''


def _iter_records_from_source(json_path: str):
    if os.path.isdir(json_path):
        json_files = sorted(str(p) for p in Path(json_path).glob('*.json'))
        for jpath in json_files:
            try:
                with open(jpath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as exc:
                yield ('__error__', jpath, exc)
                continue
            if isinstance(data, list):
                for rec in data:
                    if isinstance(rec, dict):
                        yield (rec, jpath, None)
            elif isinstance(data, dict):
                yield (data, jpath, None)
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, list):
        for rec in data:
            if isinstance(rec, dict):
                yield (rec, json_path, None)
    elif isinstance(data, dict):
        yield (data, json_path, None)


def render_lane_mask(
    labels: List[Dict],
    mask_width: int = 640,
    mask_height: int = 640,
    img_width: int = 1280,
    img_height: int = 720,
    line_thickness: int = 3,
) -> np.ndarray:
    """Render lane polylines as a binary mask.

    [INFERRED] YOLOPv2 paper §3 says masks are drawn on the centerline
    between the two annotated lines of each lane. BDD100K stores each
    lane edge as a separate `poly2d` with no explicit "which two belong
    together" link, so computing the centerline requires a pairing
    heuristic. We currently draw each annotated poly2d directly at
    `line_thickness`. This over-represents lane pixels by ~2× at small
    widths but converges to a similar supervision signal at the paper's
    width 8 (train) because the two edges fuse. At width 2 (test) this
    matters more — documented as a known limitation.
    """
    mask = np.zeros((mask_height, mask_width), dtype=np.uint8)
    sx = mask_width / float(img_width)
    sy = mask_height / float(img_height)

    for label in labels:
        geom_field = label.get('poly2d')
        if geom_field is None and label.get('seg2d') is not None:
            geom_field = label.get('seg2d')
        for dense_pts in parse_poly2d(geom_field):
            pts = []
            for x0, y0 in dense_pts:
                x = int(round(float(x0) * sx))
                y = int(round(float(y0) * sy))
                x = min(max(x, 0), mask_width - 1)
                y = min(max(y, 0), mask_height - 1)
                pts.append((x, y))
            if len(pts) >= 2:
                cv2.polylines(mask, [np.asarray(pts, dtype=np.int32)], False, 255, line_thickness)

    return mask


def convert_bdd_lanes_to_masks(
    json_path: str,
    output_mask_dir: Optional[str] = None,
    mask_width: int = 640,
    mask_height: int = 640,
    img_width: int = 1280,
    img_height: int = 720,
    line_thickness: int = 3,
    debug_limit: Optional[int] = None,
    overwrite: bool = True,
    **legacy_kwargs,
) -> Dict[str, int]:
    """Convert BDD100K lane labels to binary mask PNGs.

    Accepts either:
    - a consolidated JSON file (lane_train.json / lane_val.json)
    - an old-style per-image JSON directory (100k/train / 100k/val)

    Backward-compatible aliases are accepted so older notebooks copied from
    earlier experiments still run:
      output_dir   -> output_mask_dir
      img_w / img_h -> img_width / img_height
      mask_w / mask_h -> mask_width / mask_height
    """
    if output_mask_dir is None:
        output_mask_dir = legacy_kwargs.pop('output_dir', None)
    if output_mask_dir is None:
        raise TypeError('convert_bdd_lanes_to_masks() missing required argument: output_mask_dir')

    if 'img_w' in legacy_kwargs:
        img_width = legacy_kwargs.pop('img_w')
    if 'img_h' in legacy_kwargs:
        img_height = legacy_kwargs.pop('img_h')
    if 'mask_w' in legacy_kwargs:
        mask_width = legacy_kwargs.pop('mask_w')
    if 'mask_h' in legacy_kwargs:
        mask_height = legacy_kwargs.pop('mask_h')
    if legacy_kwargs:
        raise TypeError(f'Unexpected keyword arguments: {sorted(legacy_kwargs.keys())}')

    os.makedirs(output_mask_dir, exist_ok=True)

    stats = {
        'total_records_seen': 0,
        'total_images': 0,
        'images_with_lanes': 0,
        'total_lane_annotations': 0,
        'json_errors': 0,
        'written_masks': 0,
        'skipped_existing': 0,
    }

    debug_examples = []
    iterator = _iter_records_from_source(json_path)
    desc = 'Rendering lane masks from directory' if os.path.isdir(json_path) else 'Rendering lane masks from file'

    for item, source_path, error in tqdm(iterator, desc=desc):
        if item == '__error__':
            stats['json_errors'] += 1
            if len(debug_examples) < 3:
                debug_examples.append({'source': source_path, 'error': str(error)})
            continue

        stats['total_records_seen'] += 1
        if debug_limit is not None and stats['total_records_seen'] > int(debug_limit):
            break

        record = item
        image_name = _record_image_name(record, source_path)
        if not image_name:
            continue

        lane_labels = extract_lane_labels_any(record)
        mask_name = Path(image_name).stem + '.png'
        mask_path = os.path.join(output_mask_dir, mask_name)

        stats['total_images'] += 1
        if lane_labels:
            stats['images_with_lanes'] += 1
            stats['total_lane_annotations'] += len(lane_labels)

        if (not overwrite) and os.path.isfile(mask_path):
            stats['skipped_existing'] += 1
            continue

        mask = render_lane_mask(
            lane_labels,
            mask_width=mask_width,
            mask_height=mask_height,
            img_width=img_width,
            img_height=img_height,
            line_thickness=line_thickness,
        )
        cv2.imwrite(mask_path, mask)
        stats['written_masks'] += 1

        if len(debug_examples) < 3:
            debug_examples.append({
                'image_name': image_name,
                'source': source_path,
                'n_lane_labels': len(lane_labels),
                'mask_path': mask_path,
                'mask_has_pixels': int(mask.sum() > 0),
            })

    if debug_examples:
        print('Lane render debug examples:')
        for ex in debug_examples:
            print(' ', ex)
    return stats


def print_lane_stats(stats: Dict[str, int]) -> None:
    print(f"\n{'='*40}")
    print(' Lane Mask Statistics')
    print(f"{'='*40}")
    print(f" Total records seen:     {stats.get('total_records_seen', 0):,}")
    print(f" Total images:          {stats['total_images']:,}")
    print(f" Images with lanes:     {stats['images_with_lanes']:,}")
    pct = (stats['images_with_lanes'] / max(1, stats['total_images'])) * 100
    print(f" Lane coverage:         {pct:.1f}%")
    print(f" Total lane annotations:{stats['total_lane_annotations']:,}")
    avg = stats['total_lane_annotations'] / max(1, stats['images_with_lanes'])
    print(f" Avg lanes per image:   {avg:.1f}")
    print(f" Written masks:         {stats.get('written_masks', 0):,}")
    print(f" Skipped existing:      {stats.get('skipped_existing', 0):,}")
    print(f" JSON errors:           {stats.get('json_errors', 0):,}")
    print(f"{'='*40}")
