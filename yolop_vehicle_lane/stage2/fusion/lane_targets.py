"""BDD100K JSON -> fixed-size per-image curve lane targets.

Ported and trimmed from DETR_GeoLane_pipeline/src/lane_targets.py and
DETR_GeoLane_pipeline/src/data_prep.py. The math, schema handling, and
Bezier sampling are unchanged. This file is the canonical Stage 2
in-process API; the existing `stage2/scripts/04_prepare_bdd_curve_labels.py`
emits the CLRKDNet `.lines.txt` form for the vendored CLRKDNet trainer.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# BDD100K image / category constants
# ---------------------------------------------------------------------------
BDD_IMG_W = 1280
BDD_IMG_H = 720

KNOWN_LANE_SUBTYPES = {
    'crosswalk',
    'double other',
    'double white',
    'double yellow',
    'road curb',
    'single other',
    'single white',
    'single yellow',
}

# Categories used at training time. Crosswalk is excluded because it is
# not a lane line. Same convention as DETR_GeoLane.
LANE_TRAIN_CATS = [
    'lane/double other',
    'lane/double white',
    'lane/double yellow',
    'lane/road curb',
    'lane/single other',
    'lane/single white',
    'lane/single yellow',
]
LANE_CAT_TO_ID = {name: idx for idx, name in enumerate(LANE_TRAIN_CATS)}


# ---------------------------------------------------------------------------
# Schema-tolerant readers for both old BDD per-image and new Scalabel records
# ---------------------------------------------------------------------------
def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value if str(x).strip()]
    return [str(value)]


def _record_image_name(record: dict, fallback_json_path: Optional[str] = None) -> str:
    for key in ('name', 'image', 'imageName', 'filename', 'id'):
        v = record.get(key)
        if isinstance(v, str) and v.strip():
            base = os.path.basename(v)
            if '.' not in base:
                base += '.jpg'
            return base
    if fallback_json_path:
        return Path(fallback_json_path).stem + '.jpg'
    return ''


def _normalize_lane_category(label: dict) -> Optional[str]:
    category = str(label.get('category', '') or '').strip().lower()
    attrs = label.get('attributes') or {}
    lane_types: List[str] = []
    for key in ('laneTypes', 'laneType', 'lane_type', 'type', 'types', 'subtype', 'subtypes'):
        lane_types.extend(_as_list(attrs.get(key)))
        if key in label:
            lane_types.extend(_as_list(label.get(key)))

    if category.startswith('lane/'):
        normalized: Optional[str] = category
    elif category == 'lane':
        normalized = None
        for lane_type in lane_types:
            item = lane_type.strip().lower()
            if item in KNOWN_LANE_SUBTYPES:
                normalized = f'lane/{item}'
                break
    elif category in KNOWN_LANE_SUBTYPES:
        normalized = f'lane/{category}'
    else:
        normalized = None

    if normalized is None:
        text_bits = [category]
        for value in attrs.values():
            text_bits.extend(_as_list(value))
        joined = ' '.join(x.lower() for x in text_bits)
        for subtype in KNOWN_LANE_SUBTYPES:
            if subtype in joined:
                normalized = f'lane/{subtype}'
                break
    if normalized not in LANE_CAT_TO_ID:
        return None
    return normalized


def _iter_candidate_objects(record: Any) -> Iterable[dict]:
    if not isinstance(record, dict):
        return
    labels = record.get('labels')
    if isinstance(labels, list):
        for label in labels:
            if isinstance(label, dict):
                yield label
    frames = record.get('frames')
    if isinstance(frames, list):
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            objects = frame.get('objects')
            if isinstance(objects, list):
                for obj in objects:
                    if isinstance(obj, dict):
                        yield obj
    objects = record.get('objects')
    if isinstance(objects, list):
        for obj in objects:
            if isinstance(obj, dict):
                yield obj


# ---------------------------------------------------------------------------
# poly2d -> dense (x, y) points
# ---------------------------------------------------------------------------
def _normalize_poly2d_items(poly2d_field: Any) -> List[Any]:
    if poly2d_field is None:
        return []
    if isinstance(poly2d_field, dict):
        return [poly2d_field]
    if not isinstance(poly2d_field, list) or len(poly2d_field) == 0:
        return []
    first = poly2d_field[0]
    if isinstance(first, dict):
        return [x for x in poly2d_field if isinstance(x, dict)]
    if isinstance(first, (list, tuple)):
        if len(first) >= 2 and isinstance(first[0], (int, float)):
            return [poly2d_field]
        if len(first) > 0 and isinstance(first[0], (list, tuple, dict)):
            return [seg for seg in poly2d_field if isinstance(seg, (list, tuple, dict))]
    return []


def _coerce_point(value: Any):
    if isinstance(value, dict):
        if 'x' in value and 'y' in value:
            try:
                return float(value['x']), float(value['y']), str(value.get('type', value.get('types', 'L')) or 'L')[0]
            except Exception:
                return None
        if 'point' in value and isinstance(value['point'], (list, tuple)) and len(value['point']) >= 2:
            try:
                return float(value['point'][0]), float(value['point'][1]), str(value.get('type', 'L') or 'L')[0]
            except Exception:
                return None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            x = float(value[0])
            y = float(value[1])
        except Exception:
            return None
        ptype = 'L'
        if len(value) >= 3 and isinstance(value[2], str) and value[2]:
            ptype = value[2][0]
        return x, y, ptype
    return None


def _coerce_vertices_and_types(poly: Any) -> Tuple[np.ndarray, str, bool]:
    if isinstance(poly, dict):
        vertices = poly.get('vertices', []) or []
        types = poly.get('types', '') or ''
        closed = bool(poly.get('closed', False))
    else:
        vertices = poly
        types = ''
        closed = False

    out_v: List[List[float]] = []
    out_t: List[str] = []
    for i, vertex in enumerate(vertices):
        point = _coerce_point(vertex)
        if point is None:
            continue
        x, y, embedded = point
        out_v.append([x, y])
        ptype = embedded
        if embedded == 'L' and i < len(types):
            if isinstance(types[i], str) and types[i]:
                ptype = types[i][0]
        if ptype not in ('L', 'C'):
            ptype = 'L'
        out_t.append(ptype)
    return np.asarray(out_v, dtype=np.float64), ''.join(out_t), closed


def _sample_quad(p0, p1, p2, count=12):
    ts = np.linspace(0.0, 1.0, int(max(2, count)))
    return np.stack([((1 - t) ** 2) * p0 + 2 * (1 - t) * t * p1 + (t ** 2) * p2 for t in ts], axis=0)


def _sample_cubic(p0, p1, p2, p3, count=20):
    ts = np.linspace(0.0, 1.0, int(max(2, count)))
    return np.stack([
        ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * (t ** 2) * p2 + (t ** 3) * p3
        for t in ts
    ], axis=0)


def _segment_to_dense_points(poly: Any, bezier_samples: int = 20) -> np.ndarray:
    vertices, types, _closed = _coerce_vertices_and_types(poly)
    if len(vertices) < 2:
        return np.zeros((0, 2), dtype=np.float64)

    dense: List[np.ndarray] = [vertices[0].copy()]
    i = 1
    while i < len(vertices):
        current = dense[-1]
        ptype = types[i] if i < len(types) else 'L'
        if ptype == 'L':
            dense.append(vertices[i].copy())
            i += 1
            continue
        if i + 2 < len(vertices):
            t0 = types[i] if i < len(types) else 'L'
            t1 = types[i + 1] if i + 1 < len(types) else 'L'
            t2 = types[i + 2] if i + 2 < len(types) else 'L'
            if t0 == 'C' and t1 == 'C' and t2 == 'L':
                pts = _sample_cubic(current, vertices[i], vertices[i + 1], vertices[i + 2], bezier_samples)
                dense.extend(pts[1:])
                i += 3
                continue
        if i + 1 < len(vertices):
            pts = _sample_quad(current, vertices[i], vertices[i + 1], bezier_samples)
            dense.extend(pts[1:])
            i += 2
            continue
        # fallback: treat as linear
        dense.append(vertices[i].copy())
        i += 1

    arr = np.asarray(dense, dtype=np.float64)
    if arr.shape[0] < 2:
        return arr
    keep = [0]
    for k in range(1, arr.shape[0]):
        if not np.allclose(arr[k], arr[keep[-1]], atol=1e-3):
            keep.append(k)
    return arr[keep]


# ---------------------------------------------------------------------------
# Public extraction API
# ---------------------------------------------------------------------------
def extract_lane_labels(record: dict) -> List[dict]:
    """Return the list of lane-label dicts from one BDD record, augmented
    with `_dense_points` (np.ndarray, shape (N, 2), pixel coords) and
    `_normalized_category` (str like 'lane/single white')."""
    out: List[dict] = []
    for label in _iter_candidate_objects(record):
        cat = _normalize_lane_category(label)
        if cat is None:
            continue
        poly_field = label.get('poly2d') or label.get('seg2d')
        items = _normalize_poly2d_items(poly_field)
        if not items:
            continue
        for poly in items:
            dense = _segment_to_dense_points(poly)
            if dense.shape[0] < 2:
                continue
            entry = dict(label)
            entry['_normalized_category'] = cat
            entry['_dense_points'] = dense
            out.append(entry)
    return out


def _resample_polyline(pts: np.ndarray, n: int,
                       img_w: int = BDD_IMG_W, img_h: int = BDD_IMG_H,
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Resample a polyline to n points along arc length, normalized to [0,1]."""
    pts = np.asarray(pts, dtype=np.float64)
    seg = pts[1:] - pts[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = float(arc[-1]) if arc[-1] > 0 else 0.0
    if total <= 1e-6:
        return np.zeros((n, 2), dtype=np.float32), np.zeros((n,), dtype=np.float32)
    targets = np.linspace(0.0, total, n)
    xs = np.interp(targets, arc, pts[:, 0])
    ys = np.interp(targets, arc, pts[:, 1])
    out = np.stack([xs / float(img_w), ys / float(img_h)], axis=1).astype(np.float32)
    vis = ((out[:, 0] >= 0) & (out[:, 0] <= 1) & (out[:, 1] >= 0) & (out[:, 1] <= 1)).astype(np.float32)
    return out, vis


def frame_to_lane_targets(labels: List[dict],
                          max_lanes: int = 10,
                          num_points: int = 72,
                          img_w: int = BDD_IMG_W,
                          img_h: int = BDD_IMG_H,
                          min_arc: float = 2.0,
                          min_yspan: float = 1.0,
                          ) -> Dict[str, np.ndarray]:
    """Convert a list of lane labels (already augmented by extract_lane_labels)
    into fixed-size per-image arrays.

    Returns a dict with:
        existence  : (max_lanes,)   float32  in {0., 1.}
        points     : (max_lanes, num_points, 2) float32, normalized [0, 1]
        visibility : (max_lanes, num_points)    float32, in {0., 1.}
        lane_type  : (max_lanes,)   int64     ; -1 for empty slots
    """
    candidates: List[Tuple[float, float, np.ndarray, str]] = []
    for label in labels:
        dense = label.get('_dense_points')
        cat = label.get('_normalized_category')
        if dense is None or cat is None or dense.shape[0] < 2:
            continue
        seg_len = float(np.linalg.norm(dense[1:] - dense[:-1], axis=1).sum())
        yspan = float(dense[:, 1].max() - dense[:, 1].min())
        if seg_len < min_arc or yspan < min_yspan:
            continue
        candidates.append((yspan, seg_len, dense, cat))

    candidates.sort(key=lambda t: (-t[0], -t[1]))
    candidates = candidates[:max_lanes]

    points = np.zeros((max_lanes, num_points, 2), dtype=np.float32)
    visibility = np.zeros((max_lanes, num_points), dtype=np.float32)
    existence = np.zeros((max_lanes,), dtype=np.float32)
    lane_type = np.full((max_lanes,), -1, dtype=np.int64)

    for slot, (_y, _l, dense, cat) in enumerate(candidates):
        pts, vis = _resample_polyline(dense, num_points, img_w=img_w, img_h=img_h)
        points[slot] = pts
        visibility[slot] = vis
        existence[slot] = 1.0
        lane_type[slot] = LANE_CAT_TO_ID.get(cat, -1)

    return {
        'existence': existence,
        'points': points,
        'visibility': visibility,
        'lane_type': lane_type,
    }


def soft_polyline_mask_numpy(points: np.ndarray,
                             visibility: Optional[np.ndarray] = None,
                             height: int = 72,
                             width: int = 128,
                             thickness: float = 0.03,
                             sharpness: float = 80.0,
                             ) -> np.ndarray:
    """Render a per-image soft mask from one set of normalized points.

    points     : (max_lanes, num_points, 2) in [0, 1]
    visibility : (max_lanes, num_points)    in {0, 1} (optional)
    Returns float32 mask in [0, 1] shaped (height, width).
    """
    if visibility is None:
        visibility = np.ones(points.shape[:2], dtype=np.float32)

    ys = (np.arange(height) + 0.5) / float(height)
    xs = (np.arange(width) + 0.5) / float(width)
    grid_x, grid_y = np.meshgrid(xs, ys)  # (H, W)
    mask = np.zeros((height, width), dtype=np.float32)

    for lane_idx in range(points.shape[0]):
        pts = points[lane_idx]
        vis = visibility[lane_idx]
        if pts.sum() == 0 or vis.sum() < 2:
            continue
        valid = vis > 0.5
        if valid.sum() < 2:
            continue
        valid_pts = pts[valid]
        # Distance from every grid cell to the nearest segment of the polyline.
        gx = grid_x[..., None]    # (H, W, 1)
        gy = grid_y[..., None]
        a = valid_pts[:-1]        # (S, 2)
        b = valid_pts[1:]         # (S, 2)
        ax, ay = a[:, 0], a[:, 1]
        bx, by = b[:, 0], b[:, 1]
        dx = bx - ax
        dy = by - ay
        denom = dx * dx + dy * dy + 1e-9
        t = ((gx - ax) * dx + (gy - ay) * dy) / denom
        t = np.clip(t, 0.0, 1.0)
        px = ax + t * dx
        py = ay + t * dy
        d = np.sqrt((gx - px) ** 2 + (gy - py) ** 2)  # (H, W, S)
        d_min = d.min(axis=-1)                        # (H, W)
        line = 1.0 / (1.0 + np.exp(sharpness * (d_min - thickness)))
        mask = np.maximum(mask, line.astype(np.float32))
    return mask


# ---------------------------------------------------------------------------
# Index of label files for fast image-name -> targets lookup
# ---------------------------------------------------------------------------
class LaneLabelCache:
    """In-memory map: image basename -> per-image targets dict.

    Loads from a directory of per-image JSONs OR from a consolidated JSON
    file. JSON parsing is lazy: targets are computed once per image when
    `get` is first called.
    """

    def __init__(self, source: str | Path,
                 max_lanes: int = 10,
                 num_points: int = 72,
                 img_w: int = BDD_IMG_W,
                 img_h: int = BDD_IMG_H):
        self.source = Path(source)
        self.max_lanes = int(max_lanes)
        self.num_points = int(num_points)
        self.img_w = int(img_w)
        self.img_h = int(img_h)

        self._records: Dict[str, dict] = {}
        if self.source.is_dir():
            self._load_directory(self.source)
        elif self.source.is_file():
            self._load_file(self.source)
        else:
            raise FileNotFoundError(f'Lane label source not found: {self.source}')

        self._cache: Dict[str, Optional[Dict[str, np.ndarray]]] = {}

    def _load_directory(self, path: Path) -> None:
        for json_path in sorted(path.rglob('*.json')):
            try:
                with json_path.open('r', encoding='utf-8') as fh:
                    record = json.load(fh)
            except Exception:
                continue
            if isinstance(record, list):
                for item in record:
                    if isinstance(item, dict):
                        name = _record_image_name(item, str(json_path))
                        if name:
                            self._records[name] = item
            elif isinstance(record, dict):
                name = _record_image_name(record, str(json_path))
                if name:
                    self._records[name] = record

    def _load_file(self, path: Path) -> None:
        with path.open('r', encoding='utf-8') as fh:
            data = json.load(fh)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    name = _record_image_name(item, str(path))
                    if name:
                        self._records[name] = item
        elif isinstance(data, dict):
            for value in data.values():
                if isinstance(value, dict):
                    name = _record_image_name(value, str(path))
                    if name:
                        self._records[name] = value
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            name = _record_image_name(item, str(path))
                            if name:
                                self._records[name] = item

    def __len__(self) -> int:
        return len(self._records)

    def keys(self) -> Iterable[str]:
        return self._records.keys()

    def has_lanes(self, image_name: str) -> bool:
        targets = self.get(image_name)
        return bool(targets is not None and targets['existence'].sum() > 0)

    def get(self, image_name: str) -> Optional[Dict[str, np.ndarray]]:
        if image_name in self._cache:
            return self._cache[image_name]
        record = self._records.get(image_name)
        if record is None:
            self._cache[image_name] = None
            return None
        labels = extract_lane_labels(record)
        if not labels:
            self._cache[image_name] = None
            return None
        targets = frame_to_lane_targets(
            labels,
            max_lanes=self.max_lanes,
            num_points=self.num_points,
            img_w=self.img_w,
            img_h=self.img_h,
        )
        self._cache[image_name] = targets
        return targets

    def diagnostics(self, sample_limit: int = 5) -> Dict[str, Any]:
        """Return aggregate counts and a few example targets for debugging."""
        total = len(self._records)
        with_lanes = 0
        empty = 0
        per_image_counts: List[int] = []
        category_counter: Dict[str, int] = {}
        example_names: List[str] = []
        examples: List[Dict[str, Any]] = []

        for name in list(self._records.keys()):
            targets = self.get(name)
            if targets is None:
                empty += 1
                continue
            n_valid = int(targets['existence'].sum())
            if n_valid == 0:
                empty += 1
                continue
            with_lanes += 1
            per_image_counts.append(n_valid)
            for lane_idx in range(self.max_lanes):
                if targets['existence'][lane_idx] > 0:
                    cat_id = int(targets['lane_type'][lane_idx])
                    if 0 <= cat_id < len(LANE_TRAIN_CATS):
                        cat = LANE_TRAIN_CATS[cat_id]
                        category_counter[cat] = category_counter.get(cat, 0) + 1
            if len(examples) < sample_limit:
                example_names.append(name)
                examples.append({
                    'image': name,
                    'n_lanes': n_valid,
                    'lane_types': [int(x) for x in targets['lane_type'].tolist() if x >= 0],
                    'first_lane_pts_head': targets['points'][0, :5].tolist() if n_valid > 0 else [],
                })

        return {
            'total_images_indexed': total,
            'images_with_valid_lanes': with_lanes,
            'images_with_no_lanes': empty,
            'lanes_per_image_min': int(min(per_image_counts)) if per_image_counts else 0,
            'lanes_per_image_max': int(max(per_image_counts)) if per_image_counts else 0,
            'lanes_per_image_mean': float(np.mean(per_image_counts)) if per_image_counts else 0.0,
            'category_counts': category_counter,
            'examples': examples,
        }
