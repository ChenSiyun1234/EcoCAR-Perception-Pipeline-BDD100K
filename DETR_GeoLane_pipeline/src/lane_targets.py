"""
BDD100K poly2d → structured lane targets.

Supports both:
1) consolidated JSON files
2) per-image JSON directories like /content/bdd100k_labels_unzipped/100k/train/*.json

Also supports both newer "labels" format and older "frames/objects" format.
"""
import json
import os
import numpy as np
from typing import Dict, List, Optional, Tuple

from .config import (
    BDD_IMG_W, BDD_IMG_H, LANE_TRAIN_CATS, LANE_CAT_TO_ID,
)

def parse_poly2d(poly2d_field) -> List[np.ndarray]:
    if poly2d_field is None:
        return []
    polylines = []
    if isinstance(poly2d_field, dict):
        poly2d_field = [poly2d_field]
    for item in poly2d_field:
        if isinstance(item, dict):
            verts = item.get("vertices", [])
        elif isinstance(item, (list, tuple)):
            if len(item) > 0 and isinstance(item[0], (list, tuple)):
                verts = item
            else:
                continue
        else:
            continue
        if len(verts) >= 2:
            polylines.append(np.array(verts, dtype=np.float64))
    return polylines

def _maybe_prefix_lane_category(cat: str) -> str:
    if not cat:
        return ""
    cat = str(cat)
    if cat.startswith("lane/"):
        return cat
    if cat in [c.split("/", 1)[1] for c in LANE_TRAIN_CATS]:
        return "lane/" + cat
    return cat

def _extract_lane_labels_from_record(record: dict) -> Tuple[str, List[dict]]:
    """
    Return (image_name, lane_labels) from either:
      - {'name': ..., 'labels': [...]}
      - {'name': ..., 'frames': [{'objects': [...]}]}
      - per-image dict with top-level labels/objects
    """
    if not isinstance(record, dict):
        return "", []

    image_name = record.get("name", "") or ""
    lane_labels: List[dict] = []

    # Newer format: top-level labels
    labels = record.get("labels")
    if isinstance(labels, list):
        for lab in labels:
            if not isinstance(lab, dict):
                continue
            cat = _maybe_prefix_lane_category(lab.get("category", ""))
            poly2d = lab.get("poly2d")
            if poly2d and cat.startswith("lane/"):
                lane_labels.append({
                    "category": cat,
                    "poly2d": poly2d,
                })

    # Older format: frames -> objects
    frames = record.get("frames")
    if isinstance(frames, list):
        for fr in frames:
            if not isinstance(fr, dict):
                continue
            if not image_name:
                image_name = fr.get("name", "") or image_name
            objs = fr.get("objects") or fr.get("labels") or []
            if not isinstance(objs, list):
                continue
            for obj in objs:
                if not isinstance(obj, dict):
                    continue
                cat = _maybe_prefix_lane_category(obj.get("category", ""))
                # old format may store category='lane' with attributes
                attrs = obj.get("attributes") or {}
                if cat == "lane":
                    lane_type = attrs.get("laneType")
                    if isinstance(lane_type, str) and lane_type:
                        cat = f"lane/{lane_type}"
                poly2d = obj.get("poly2d")
                if poly2d and cat.startswith("lane/"):
                    lane_labels.append({
                        "category": cat,
                        "poly2d": poly2d,
                    })

    # Some per-image JSONs may directly store objects
    objs = record.get("objects")
    if isinstance(objs, list):
        for obj in objs:
            if not isinstance(obj, dict):
                continue
            cat = _maybe_prefix_lane_category(obj.get("category", ""))
            attrs = obj.get("attributes") or {}
            if cat == "lane":
                lane_type = attrs.get("laneType")
                if isinstance(lane_type, str) and lane_type:
                    cat = f"lane/{lane_type}"
            poly2d = obj.get("poly2d")
            if poly2d and cat.startswith("lane/"):
                lane_labels.append({
                    "category": cat,
                    "poly2d": poly2d,
                })

    return image_name, lane_labels

def resample_polyline(pts: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    if pts[-1, 1] < pts[0, 1]:
        pts = pts[::-1].copy()
    diffs = np.diff(pts, axis=0)
    seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
    cum_len = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = cum_len[-1]
    if total < 1e-6:
        out = np.tile(pts[0], (n, 1))
        return out, np.ones(n, dtype=bool)
    sample_dists = np.linspace(0.0, total, n)
    resampled = np.zeros((n, 2), dtype=np.float64)
    for i, d in enumerate(sample_dists):
        idx = np.searchsorted(cum_len, d, side="right") - 1
        idx = np.clip(idx, 0, len(pts) - 2)
        seg_start = cum_len[idx]
        seg_len = seg_lens[idx]
        if seg_len < 1e-9:
            resampled[i] = pts[idx]
        else:
            t = (d - seg_start) / seg_len
            resampled[i] = pts[idx] * (1 - t) + pts[idx + 1] * t
    visibility = (
        (resampled[:, 0] >= 0) & (resampled[:, 0] <= BDD_IMG_W - 1) &
        (resampled[:, 1] >= 0) & (resampled[:, 1] <= BDD_IMG_H - 1)
    )
    return resampled, visibility

def frame_to_lane_targets(labels: List[dict], max_lanes: int = 10,
                          num_points: int = 72,
                          img_w: int = BDD_IMG_W,
                          img_h: int = BDD_IMG_H) -> Dict[str, np.ndarray]:
    existence = np.zeros(max_lanes, dtype=np.float32)
    points = np.zeros((max_lanes, num_points, 2), dtype=np.float32)
    visibility = np.zeros((max_lanes, num_points), dtype=np.float32)
    lane_type = np.zeros(max_lanes, dtype=np.int64)

    # sort longer vertical lanes first
    expanded = []
    for label in labels:
        cat = _maybe_prefix_lane_category(label.get("category", ""))
        if cat not in LANE_CAT_TO_ID:
            continue
        for pl in parse_poly2d(label.get("poly2d")):
            if len(pl) >= 2:
                yspan = float(np.max(pl[:,1]) - np.min(pl[:,1]))
                expanded.append((yspan, cat, pl))
    expanded.sort(key=lambda x: x[0], reverse=True)

    lane_idx = 0
    for _, cat, pl in expanded:
        if lane_idx >= max_lanes:
            break
        pl = pl.copy()
        pl[:, 0] = np.clip(pl[:, 0], 0, img_w - 1)
        pl[:, 1] = np.clip(pl[:, 1], 0, img_h - 1)
        resampled, vis = resample_polyline(pl, num_points)
        resampled[:, 0] /= img_w
        resampled[:, 1] /= img_h
        resampled = np.clip(resampled, 0.0, 1.0)
        existence[lane_idx] = 1.0
        points[lane_idx] = resampled.astype(np.float32)
        visibility[lane_idx] = vis.astype(np.float32)
        lane_type[lane_idx] = LANE_CAT_TO_ID[cat]
        lane_idx += 1

    return {
        "existence": existence,
        "points": points,
        "visibility": visibility,
        "lane_type": lane_type,
    }

class LaneLabelCache:
    """Load BDD100K lane annotations from either a JSON file or a directory of per-image JSONs."""
    def __init__(self, source_path: str, max_lanes: int = 10, num_points: int = 72):
        self.max_lanes = max_lanes
        self.num_points = num_points
        self._cache: Dict[str, List[dict]] = {}

        if not source_path:
            print("  No lane labels source provided")
            return

        if os.path.isdir(source_path):
            print(f"Loading lane labels from directory {source_path} ...")
            count = 0
            for name in sorted(os.listdir(source_path)):
                if not name.lower().endswith(".json"):
                    continue
                fpath = os.path.join(source_path, name)
                try:
                    with open(fpath, "r") as f:
                        data = json.load(f)
                except Exception:
                    continue

                if isinstance(data, list):
                    # treat as mini-consolidated list
                    records = data
                else:
                    records = [data]

                file_cached = False
                for rec in records:
                    image_name, lane_labels = _extract_lane_labels_from_record(rec)
                    if not image_name:
                        image_name = os.path.splitext(name)[0] + ".jpg"
                    if lane_labels:
                        self._cache[image_name] = lane_labels
                        file_cached = True
                if file_cached:
                    count += 1
            print(f"  Cached lane labels for {len(self._cache)} frames")
            return

        if os.path.isfile(source_path):
            print(f"Loading lane labels from {source_path} ...")
            with open(source_path, "r") as f:
                data = json.load(f)
            records = data if isinstance(data, list) else [data]
            for rec in records:
                image_name, lane_labels = _extract_lane_labels_from_record(rec)
                if image_name and lane_labels:
                    self._cache[image_name] = lane_labels
            print(f"  Cached lane labels for {len(self._cache)} frames")
        else:
            print(f"  No lane labels found at: {source_path}")

    def get(self, image_name: str) -> Optional[Dict[str, np.ndarray]]:
        labels = self._cache.get(image_name)
        if labels is None:
            return None
        return frame_to_lane_targets(labels, self.max_lanes, self.num_points)

    def __len__(self):
        return len(self._cache)

    def has_lanes(self, image_name: str) -> bool:
        return image_name in self._cache
