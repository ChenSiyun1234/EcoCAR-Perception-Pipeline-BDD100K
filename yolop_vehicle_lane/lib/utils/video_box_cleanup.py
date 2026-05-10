"""Video bounding-box cleanup helpers.

The video profiler uses a lower confidence threshold than validation so that an
unfinished checkpoint still draws boxes. That makes close vehicles prone to
fragment boxes, especially when several class-specific predictions survive NMS.
These helpers run a conservative second-stage cleanup after NMS and before
tracking.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple


Detection = Dict[str, object]


def _area(box: Sequence[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))


def _intersection(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(float(ax1), float(bx1))
    iy1 = max(float(ay1), float(by1))
    ix2 = min(float(ax2), float(bx2))
    iy2 = min(float(ay2), float(by2))
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    inter = _intersection(a, b)
    denom = _area(a) + _area(b) - inter
    if denom <= 1e-6:
        return 0.0
    return inter / denom


def _inside_center(inner: Sequence[float], outer: Sequence[float]) -> bool:
    x1, y1, x2, y2 = inner
    ox1, oy1, ox2, oy2 = outer
    cx = 0.5 * (float(x1) + float(x2))
    cy = 0.5 * (float(y1) + float(y2))
    return float(ox1) <= cx <= float(ox2) and float(oy1) <= cy <= float(oy2)


def _clip_box(box: Sequence[float], width: int, height: int) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = min(max(x1, 0.0), float(width - 1))
    x2 = min(max(x2, 0.0), float(width - 1))
    y1 = min(max(y1, 0.0), float(height - 1))
    y2 = min(max(y2, 0.0), float(height - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def _should_suppress_fragment(
    small: Detection,
    large: Detection,
    frame_area: float,
    close_object_area_ratio: float,
    containment_threshold: float,
    center_inside_inter_threshold: float,
    confidence_margin: float,
    same_object_iou_threshold: float,
) -> bool:
    small_box = small["bbox"]
    large_box = large["bbox"]
    small_area = _area(small_box)
    large_area = _area(large_box)
    if small_area <= 1e-6 or large_area <= 1e-6:
        return False
    if large_area < small_area:
        return False

    inter = _intersection(small_box, large_box)
    inter_over_small = inter / max(small_area, 1e-6)
    area_ratio = small_area / max(large_area, 1e-6)
    large_is_close = large_area >= frame_area * close_object_area_ratio
    large_score_ok = float(large["conf"]) >= float(small["conf"]) - confidence_margin

    if _iou(small_box, large_box) >= same_object_iou_threshold and large_score_ok:
        return True

    if large_is_close and inter_over_small >= containment_threshold and large_score_ok:
        return True

    if large_is_close and _inside_center(small_box, large_box):
        if area_ratio <= 0.72 and inter_over_small >= center_inside_inter_threshold and large_score_ok:
            return True

    return False


def cleanup_vehicle_detections(
    detections: Iterable[Detection],
    image_shape: Tuple[int, int],
    close_object_area_ratio: float = 0.025,
    containment_threshold: float = 0.58,
    center_inside_inter_threshold: float = 0.20,
    confidence_margin: float = 0.20,
    same_object_iou_threshold: float = 0.72,
    min_box_area: float = 0.0,
) -> Tuple[List[Detection], Dict[str, int]]:
    """Suppress duplicate fragment boxes caused by close vehicles.

    Args:
        detections: List of dicts with keys ``bbox``, ``conf``, and ``cls``.
        image_shape: Original frame shape as ``(height, width)``.
        close_object_area_ratio: Area ratio used to identify close foreground
            vehicles. Fragment suppression is conservative for small/far boxes.
        containment_threshold: Suppress a smaller box when this fraction of it
            is covered by a larger close vehicle box.
        center_inside_inter_threshold: Secondary rule for small boxes whose
            centers lie inside a larger close vehicle box.
        confidence_margin: Allows a large whole-vehicle box to suppress a
            slightly higher-confidence fragment.
        same_object_iou_threshold: Class-agnostic duplicate threshold after the
            model's original NMS.
        min_box_area: Drop invalid or tiny boxes before cleanup.

    Returns:
        Cleaned detections and integer diagnostic counters.
    """
    height, width = int(image_shape[0]), int(image_shape[1])
    frame_area = float(max(height * width, 1))

    clipped: List[Detection] = []
    dropped_tiny = 0
    for det in detections:
        box = _clip_box(det["bbox"], width, height)
        if _area(box) < min_box_area:
            dropped_tiny += 1
            continue
        clipped.append({"bbox": box, "conf": float(det["conf"]), "cls": int(det["cls"])})

    suppress = set()
    n = len(clipped)
    for i in range(n):
        if i in suppress:
            continue
        for j in range(n):
            if i == j or j in suppress:
                continue
            area_i = _area(clipped[i]["bbox"])
            area_j = _area(clipped[j]["bbox"])
            if area_i <= area_j:
                continue
            if _should_suppress_fragment(
                small=clipped[j],
                large=clipped[i],
                frame_area=frame_area,
                close_object_area_ratio=close_object_area_ratio,
                containment_threshold=containment_threshold,
                center_inside_inter_threshold=center_inside_inter_threshold,
                confidence_margin=confidence_margin,
                same_object_iou_threshold=same_object_iou_threshold,
            ):
                suppress.add(j)

    cleaned = [det for idx, det in enumerate(clipped) if idx not in suppress]
    cleaned.sort(key=lambda d: float(d["conf"]), reverse=True)
    stats = {
        "input": n,
        "dropped_tiny": dropped_tiny,
        "suppressed_fragments": len(suppress),
        "output": len(cleaned),
    }
    return cleaned, stats
