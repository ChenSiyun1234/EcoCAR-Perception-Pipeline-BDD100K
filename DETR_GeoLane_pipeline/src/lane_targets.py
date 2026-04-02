"""
BDD100K poly2d → structured lane targets.
"""

import json
import os
import numpy as np
from typing import Dict, List, Optional, Tuple

from .config import BDD_IMG_W, BDD_IMG_H, LANE_CAT_TO_ID


def _bezier_curve(p0, p1, p2, p3, num_points=30):
    points = []
    for i in range(num_points + 1):
        t = i / num_points
        mt = 1.0 - t
        x = mt**3 * p0[0] + 3 * mt**2 * t * p1[0] + 3 * mt * t**2 * p2[0] + t**3 * p3[0]
        y = mt**3 * p0[1] + 3 * mt**2 * t * p1[1] + 3 * mt * t**2 * p2[1] + t**3 * p3[1]
        points.append((x, y))
    return points


def _poly2d_to_dense_points(vertices, types):
    points = []
    i = 0
    n = len(vertices)
    while i < n:
        vx, vy = float(vertices[i][0]), float(vertices[i][1])
        if i + 3 < n and i + 1 < len(types) and str(types[i + 1]).upper().startswith("C"):
            p0 = (vx, vy)
            p1 = (float(vertices[i + 1][0]), float(vertices[i + 1][1]))
            p2 = (float(vertices[i + 2][0]), float(vertices[i + 2][1]))
            p3 = (float(vertices[i + 3][0]), float(vertices[i + 3][1]))
            bez = _bezier_curve(p0, p1, p2, p3)
            if points and bez:
                bez = bez[1:]
            points.extend(bez)
            i += 4
        else:
            points.append((vx, vy))
            i += 1
    return np.asarray(points, dtype=np.float64) if len(points) >= 2 else np.empty((0, 2), dtype=np.float64)


def parse_poly2d(poly2d_field) -> List[np.ndarray]:
    if poly2d_field is None:
        return []
    if isinstance(poly2d_field, dict):
        poly2d_field = [poly2d_field]
    polylines = []
    if not isinstance(poly2d_field, list):
        return polylines
    for item in poly2d_field:
        if isinstance(item, dict):
            verts = item.get("vertices", []) or []
            types = item.get("types", "") or ("L" * len(verts))
            if len(verts) >= 2:
                dense = _poly2d_to_dense_points(verts, types)
                if len(dense) >= 2:
                    polylines.append(dense)
        elif isinstance(item, (list, tuple)) and len(item) >= 2 and isinstance(item[0], (list, tuple)):
            arr = np.asarray(item, dtype=np.float64)
            if len(arr) >= 2:
                polylines.append(arr)
    return polylines


def resample_polyline(pts: np.ndarray, n: int) -> Tuple[np.ndarray, np.ndarray]:
    if pts[-1, 1] < pts[0, 1]:
        pts = pts[::-1].copy()
    diffs = np.diff(pts, axis=0)
    seg_lens = np.sqrt((diffs ** 2).sum(axis=1))
    cum_len = np.concatenate([[0.0], np.cumsum(seg_lens)])
    total = cum_len[-1]
    if total < 1e-6:
        out = np.tile(pts[0], (n, 1))
        vis = np.zeros(n, dtype=bool)
        vis[:1] = True
        return out, vis
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
        (resampled[:, 0] >= 0.0) & (resampled[:, 0] <= BDD_IMG_W - 1) &
        (resampled[:, 1] >= 0.0) & (resampled[:, 1] <= BDD_IMG_H - 1)
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

    candidates = []
    for label in labels:
        cat = label.get("category", "")
        if not isinstance(cat, str) or not cat.startswith("lane/"):
            continue
        for pl in parse_poly2d(label.get("poly2d")):
            if len(pl) < 2:
                continue
            y_span = float(np.max(pl[:, 1]) - np.min(pl[:, 1]))
            candidates.append((y_span, cat, pl))

    candidates.sort(key=lambda t: t[0], reverse=True)
    candidates = candidates[:max_lanes]
    for lane_idx, (_span, cat, pl) in enumerate(candidates):
        resampled, vis = resample_polyline(pl, num_points)
        clipped = resampled.copy()
        clipped[:, 0] = np.clip(clipped[:, 0], 0.0, img_w - 1)
        clipped[:, 1] = np.clip(clipped[:, 1], 0.0, img_h - 1)
        clipped[:, 0] /= img_w
        clipped[:, 1] /= img_h
        existence[lane_idx] = 1.0
        points[lane_idx] = clipped.astype(np.float32)
        visibility[lane_idx] = vis.astype(np.float32)
        lane_type[lane_idx] = LANE_CAT_TO_ID.get(cat, 0)

    return {"existence": existence, "points": points, "visibility": visibility, "lane_type": lane_type}


class LaneLabelCache:
    def __init__(self, json_path: Optional[str], max_lanes: int = 10, num_points: int = 72):
        self.json_path = json_path
        self.max_lanes = max_lanes
        self.num_points = num_points
        self._cache = {}
        if json_path is not None and os.path.isfile(json_path):
            self._load()

    def _load(self):
        with open(self.json_path, "r") as f:
            data = json.load(f)
        for frame in data:
            name = frame.get("name", None)
            labels = frame.get("labels", [])
            if name is None:
                continue
            self._cache[name] = frame_to_lane_targets(labels, max_lanes=self.max_lanes, num_points=self.num_points)

    def get(self, image_name: str) -> Optional[Dict[str, np.ndarray]]:
        return self._cache.get(image_name)

    def __len__(self):
        return len(self._cache)
