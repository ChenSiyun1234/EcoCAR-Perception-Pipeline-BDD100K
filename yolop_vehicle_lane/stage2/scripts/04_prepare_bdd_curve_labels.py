from __future__ import annotations

import argparse
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any, Iterable, Optional

import cv2
import numpy as np

BDD_IMG_W = 1280
BDD_IMG_H = 720
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
KNOWN_LANE_SUBTYPES = {
    "crosswalk",
    "double other",
    "double white",
    "double yellow",
    "road curb",
    "single other",
    "single white",
    "single yellow",
}


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def norm_path(path: str | Path) -> str:
    return str(path).lower().replace("\\", "/")


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    ensure_dir(dst.parent)
    try:
        os.symlink(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value if str(x).strip()]
    return [str(value)]


def record_image_name(record: dict, fallback_json_path: Optional[str] = None) -> str:
    for key in ["name", "image", "imageName", "filename", "id"]:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            base = os.path.basename(value)
            if "." not in base:
                base += ".jpg"
            return base
    if fallback_json_path:
        return Path(fallback_json_path).stem + ".jpg"
    return ""


def normalize_lane_category(label: dict) -> Optional[str]:
    category = str(label.get("category", "") or "").strip().lower()
    attrs = label.get("attributes") or {}
    lane_types: list[str] = []
    for key in ["laneTypes", "laneType", "lane_type", "type", "types", "subtype", "subtypes"]:
        lane_types.extend(as_list(attrs.get(key)))
        if key in label:
            lane_types.extend(as_list(label.get(key)))

    if category.startswith("lane/"):
        normalized = category
    elif category == "lane":
        normalized = None
        for lane_type in lane_types:
            item = lane_type.strip().lower()
            if item in KNOWN_LANE_SUBTYPES:
                normalized = f"lane/{item}"
                break
    elif category in KNOWN_LANE_SUBTYPES:
        normalized = f"lane/{category}"
    else:
        normalized = None

    if normalized is None:
        text_bits = [category]
        for value in attrs.values():
            text_bits.extend(as_list(value))
        joined = " ".join(x.lower() for x in text_bits)
        for subtype in KNOWN_LANE_SUBTYPES:
            if subtype in joined:
                normalized = f"lane/{subtype}"
                break
    return normalized


def iter_candidate_objects(record: Any) -> Iterable[dict]:
    if not isinstance(record, dict):
        return
    labels = record.get("labels")
    if isinstance(labels, list):
        for label in labels:
            if isinstance(label, dict):
                yield label
    frames = record.get("frames")
    if isinstance(frames, list):
        for frame in frames:
            if not isinstance(frame, dict):
                continue
            objects = frame.get("objects")
            if isinstance(objects, list):
                for obj in objects:
                    if isinstance(obj, dict):
                        yield obj
    objects = record.get("objects")
    if isinstance(objects, list):
        for obj in objects:
            if isinstance(obj, dict):
                yield obj


def normalize_poly2d_items(poly2d_field: Any) -> list[Any]:
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


def coerce_point(value: Any):
    if isinstance(value, dict):
        if "x" in value and "y" in value:
            try:
                return float(value["x"]), float(value["y"]), str(value.get("type", value.get("types", "L")) or "L")[0]
            except Exception:
                return None
        if "point" in value and isinstance(value["point"], (list, tuple)) and len(value["point"]) >= 2:
            try:
                return float(value["point"][0]), float(value["point"][1]), str(value.get("type", "L") or "L")[0]
            except Exception:
                return None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            x_value = float(value[0])
            y_value = float(value[1])
        except Exception:
            return None
        point_type = "L"
        if len(value) >= 3 and isinstance(value[2], str) and value[2]:
            point_type = value[2][0]
        return x_value, y_value, point_type
    return None


def coerce_vertices_and_types(poly: Any) -> tuple[np.ndarray, str, bool]:
    if isinstance(poly, dict):
        vertices = poly.get("vertices", []) or []
        types = poly.get("types", "") or ""
        closed = bool(poly.get("closed", False))
    else:
        vertices = poly
        types = ""
        closed = False

    out_vertices = []
    out_types = []
    for i, vertex in enumerate(vertices):
        point = coerce_point(vertex)
        if point is None:
            continue
        x_value, y_value, embedded_type = point
        out_vertices.append([x_value, y_value])
        point_type = embedded_type
        if embedded_type == "L" and i < len(types):
            if isinstance(types[i], str) and types[i]:
                point_type = types[i][0]
        if point_type not in ("L", "C"):
            point_type = "L"
        out_types.append(point_type)
    return np.asarray(out_vertices, dtype=np.float64), "".join(out_types), closed


def sample_quadratic(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, count: int = 12) -> np.ndarray:
    ts = np.linspace(0.0, 1.0, int(max(2, count)))
    return np.stack([((1 - t) ** 2) * p0 + 2 * (1 - t) * t * p1 + (t ** 2) * p2 for t in ts], axis=0)


def sample_cubic(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, count: int = 20) -> np.ndarray:
    ts = np.linspace(0.0, 1.0, int(max(2, count)))
    return np.stack([
        ((1 - t) ** 3) * p0 + 3 * ((1 - t) ** 2) * t * p1 + 3 * (1 - t) * (t ** 2) * p2 + (t ** 3) * p3
        for t in ts
    ], axis=0)


def segment_to_dense_points(poly: Any, bezier_samples: int = 20) -> np.ndarray:
    vertices, types, closed = coerce_vertices_and_types(poly)
    if len(vertices) < 2:
        return np.zeros((0, 2), dtype=np.float64)

    dense = [vertices[0].copy()]
    i = 1
    while i < len(vertices):
        current = dense[-1]
        point_type = types[i] if i < len(types) else "L"
        if point_type == "L":
            dense.append(vertices[i].copy())
            i += 1
            continue
        if i + 2 < len(vertices):
            t0 = types[i] if i < len(types) else "L"
            t1 = types[i + 1] if i + 1 < len(types) else "L"
            t2 = types[i + 2] if i + 2 < len(types) else "L"
            if t0 == "C" and t1 == "C" and t2 == "L":
                points = sample_cubic(current, vertices[i], vertices[i + 1], vertices[i + 2], bezier_samples)
                dense.extend(points[1:])
                i += 3
                continue
        if i + 1 < len(vertices):
            t0 = types[i] if i < len(types) else "L"
            t1 = types[i + 1] if i + 1 < len(types) else "L"
            if t0 == "C" and t1 == "L":
                points = sample_quadratic(current, vertices[i], vertices[i + 1], max(8, bezier_samples // 2))
                dense.extend(points[1:])
                i += 2
                continue
        dense.append(vertices[i].copy())
        i += 1

    if closed and len(dense) > 2:
        if np.linalg.norm(np.asarray(dense[0]) - np.asarray(dense[-1])) > 1e-6:
            dense.append(np.asarray(dense[0]).copy())

    points = np.asarray(dense, dtype=np.float64)
    keep = [0]
    for idx in range(1, len(points)):
        if np.linalg.norm(points[idx] - points[keep[-1]]) > 1e-6:
            keep.append(idx)
    return points[keep]


def parse_poly2d(poly2d_field: Any, bezier_samples: int = 20) -> list[np.ndarray]:
    output = []
    for segment in normalize_poly2d_items(poly2d_field):
        points = segment_to_dense_points(segment, bezier_samples=bezier_samples)
        if len(points) >= 2:
            output.append(points)
    return output


def extract_lanes(record: dict) -> list[np.ndarray]:
    lanes: list[np.ndarray] = []
    for label in iter_candidate_objects(record):
        if normalize_lane_category(label) is None:
            continue
        geom_field = label.get("poly2d")
        if geom_field is None and label.get("seg2d") is not None:
            geom_field = label.get("seg2d")
        for points in parse_poly2d(geom_field):
            if len(points) < 2:
                continue
            arc = float(np.sqrt(((np.diff(points, axis=0)) ** 2).sum(axis=1)).sum())
            span_y = float(points[:, 1].max() - points[:, 1].min())
            if arc >= 2.0 and span_y >= 1.0:
                lanes.append(points)
    lanes.sort(key=lambda pts: (float(pts[:, 1].max() - pts[:, 1].min()), float(np.sqrt(((np.diff(pts, axis=0)) ** 2).sum(axis=1)).sum())), reverse=True)
    return lanes


def load_lane_records(source: Path) -> dict[str, list[np.ndarray]]:
    records: dict[str, list[np.ndarray]] = {}
    json_paths = sorted(source.glob("*.json")) if source.is_dir() else [source]
    for json_path in json_paths:
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        items = data if isinstance(data, list) else [data]
        for record in items:
            if not isinstance(record, dict):
                continue
            name = record_image_name(record, str(json_path))
            lanes = extract_lanes(record)
            if name and lanes:
                records[name] = lanes
    return records


def find_zip(downloads_root: Path, patterns: list[str]) -> Optional[Path]:
    if not downloads_root.exists():
        return None
    for pattern in patterns:
        hits = sorted(downloads_root.rglob(pattern))
        if hits:
            return hits[0]
    return None


def safe_extract_zip(zip_path: Path, dest: Path) -> None:
    ensure_dir(dest)
    marker = dest / f".extracted_{zip_path.stem}.ok"
    if marker.exists():
        print(f"Already extracted: {zip_path} -> {dest}")
        return
    print(f"Extracting {zip_path} -> {dest}")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dest)
    marker.write_text(str(zip_path), encoding="utf-8")


def locate_lane_source(raw_root: Path, split: str) -> Path:
    file_candidates = [
        raw_root / "labels" / "lane" / "polygons" / f"lane_{split}.json",
        raw_root / "bdd100k" / "labels" / "lane" / "polygons" / f"lane_{split}.json",
        raw_root / "bdd100k_labels_release" / "bdd100k" / "labels" / "lane" / "polygons" / f"lane_{split}.json",
        raw_root / f"lane_{split}.json",
    ]
    for candidate in file_candidates:
        if candidate.is_file():
            return candidate

    dir_candidates = [
        raw_root / "100k" / split,
        raw_root / "bdd100k" / "100k" / split,
        raw_root / "labels" / "100k" / split,
        raw_root / "bdd100k" / "labels" / "100k" / split,
    ]
    for candidate in dir_candidates:
        if candidate.is_dir() and len(list(candidate.glob("*.json"))) > 0:
            return candidate

    rglob_files = sorted(raw_root.rglob(f"lane_{split}.json"))
    if rglob_files:
        return rglob_files[0]

    scored = []
    for path in raw_root.rglob("*"):
        if not path.is_dir() or path.name.lower() != split:
            continue
        json_count = len(list(path.glob("*.json")))
        if json_count < 1:
            continue
        score = json_count
        text = norm_path(path)
        if "100k" in text:
            score += 100000
        if "lane" in text:
            score += 50000
        scored.append((score, path))
    if not scored:
        raise FileNotFoundError(f"Could not locate BDD100K lane JSON directory/file for split={split} under {raw_root}")
    return sorted(scored, key=lambda item: (-item[0], str(item[1])))[0][1]


def locate_image_dir(dataset_root: Path, raw_root: Path, split: str) -> Path:
    candidates = [
        dataset_root / "images" / split,
        dataset_root / "100k" / split,
        dataset_root / "images" / "100k" / split,
        raw_root / "images" / "100k" / split,
        raw_root / "bdd100k" / "images" / "100k" / split,
        raw_root / "bdd100k_images_100k" / "bdd100k" / "images" / "100k" / split,
        raw_root / "100k" / split,
    ]
    for candidate in candidates:
        if candidate.is_dir() and any(p.suffix.lower() in IMAGE_SUFFIXES for p in candidate.iterdir() if p.is_file()):
            return candidate

    scored = []
    for path in raw_root.rglob("*"):
        if not path.is_dir() or path.name.lower() != split:
            continue
        img_count = sum(1 for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        if img_count < 1:
            continue
        score = img_count
        text = norm_path(path)
        if "images" in text:
            score += 100000
        if "100k" in text:
            score += 50000
        scored.append((score, path))
    if not scored:
        raise FileNotFoundError(
            f"Could not locate image directory for split={split}. "
            f"Checked dataset_root={dataset_root} and raw_root={raw_root}."
        )
    return sorted(scored, key=lambda item: (-item[0], str(item[1])))[0][1]


def maybe_extract_official_zips(raw_root: Path, downloads_root: Path, need_images: bool) -> None:
    ensure_dir(raw_root)
    label_zip = find_zip(downloads_root, ["bdd100k_labels.zip", "*labels*.zip"])
    if label_zip is not None:
        try:
            locate_lane_source(raw_root, "train")
            locate_lane_source(raw_root, "val")
        except FileNotFoundError:
            safe_extract_zip(label_zip, raw_root)
    else:
        print(f"No labels zip found under {downloads_root}; assuming labels are already extracted.")

    if need_images:
        image_zip = find_zip(downloads_root, ["bdd100k_images_100k.zip", "*images*100k*.zip"])
        if image_zip is not None:
            try:
                locate_image_dir(Path("/nonexistent_dataset_root"), raw_root, "train")
                locate_image_dir(Path("/nonexistent_dataset_root"), raw_root, "val")
            except FileNotFoundError:
                safe_extract_zip(image_zip, raw_root)
        else:
            print(f"No image zip found under {downloads_root}; assuming images are already available.")


def write_lines_file(path: Path, lanes: list[np.ndarray]) -> None:
    rows = []
    for lane in lanes:
        flat = []
        for x_value, y_value in lane:
            flat.append(f"{float(x_value):.3f}")
            flat.append(f"{float(y_value):.3f}")
        if len(flat) >= 4:
            rows.append(" ".join(flat))
    path.write_text("\n".join(rows), encoding="utf-8")


def write_mask(path: Path, lanes: list[np.ndarray], width: int, height: int, thickness: int) -> None:
    ensure_dir(path.parent)
    mask = np.zeros((height, width), dtype=np.uint8)
    for lane in lanes:
        pts = lane.copy()
        pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
        pts_i = np.round(pts).astype(np.int32).reshape(-1, 1, 2)
        if len(pts_i) >= 2:
            cv2.polylines(mask, [pts_i], isClosed=False, color=1, thickness=thickness)
    cv2.imwrite(str(path), mask)


def image_shape(path: Path) -> tuple[int, int]:
    image = cv2.imread(str(path))
    if image is None:
        return BDD_IMG_W, BDD_IMG_H
    height, width = image.shape[:2]
    return int(width), int(height)


def prepare_split(dataset_root: Path, raw_root: Path, output_root: Path, split: str, lane_source_arg: Optional[str], mask_thickness: int) -> dict[str, int | str]:
    image_dir = locate_image_dir(dataset_root, raw_root, split)
    lane_source = Path(lane_source_arg) if lane_source_arg else locate_lane_source(raw_root, split)
    lane_records = load_lane_records(lane_source)

    out_image_dir = ensure_dir(output_root / "images" / split)
    out_mask_dir = ensure_dir(output_root / "masks" / split)
    out_list_dir = ensure_dir(output_root / "list")

    image_paths = sorted([p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES])
    list_rows = []
    with_lanes = 0
    total_lanes = 0

    for image_path in image_paths:
        lanes = lane_records.get(image_path.name, [])
        if lanes:
            with_lanes += 1
            total_lanes += len(lanes)
        out_image = out_image_dir / image_path.name
        link_or_copy(image_path, out_image)
        line_path = out_image.with_suffix(".lines.txt")
        write_lines_file(line_path, lanes)

        width, height = image_shape(image_path)
        mask_path = out_mask_dir / f"{image_path.stem}.png"
        write_mask(mask_path, lanes, width, height, mask_thickness)

        rel_image = "/" + str(out_image.relative_to(output_root)).replace("\\", "/")
        rel_mask = "/" + str(mask_path.relative_to(output_root)).replace("\\", "/")
        if split == "train":
            list_rows.append(f"{rel_image} {rel_mask} {1 if lanes else 0}")
        else:
            list_rows.append(rel_image)

    if split == "train":
        (out_list_dir / "train_gt.txt").write_text("\n".join(list_rows) + "\n", encoding="utf-8")
    else:
        (out_list_dir / "val.txt").write_text("\n".join(list_rows) + "\n", encoding="utf-8")
        (out_list_dir / "test.txt").write_text("\n".join(list_rows) + "\n", encoding="utf-8")

    return {
        "images": len(image_paths),
        "frames_with_lanes": with_lanes,
        "lane_instances": total_lanes,
        "lane_json_records": len(lane_records),
        "lane_source": str(lane_source),
        "image_source": str(image_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare BDD100K lane curve labels for CLRKDNet-style training.")
    parser.add_argument("--dataset-root", default="/content/bdd100k_vehicle5", help="Stage-1 dataset root if available.")
    parser.add_argument("--raw-root", default="/content/bdd100k_raw", help="Extracted BDD100K labels/images root.")
    parser.add_argument("--downloads-root", default="/content/drive/MyDrive/EcoCAR/downloads", help="Official BDD100K zip archive directory.")
    parser.add_argument("--auto-extract", action="store_true", help="Extract official labels/images zips from downloads-root if needed.")
    parser.add_argument("--output-root", default="/content/bdd100k_clrkd_curve", help="Output root for CLRKDNet-compatible curve dataset.")
    parser.add_argument("--train-lane-json", default=None, help="Optional train lane JSON directory or consolidated file.")
    parser.add_argument("--val-lane-json", default=None, help="Optional val lane JSON directory or consolidated file.")
    parser.add_argument("--mask-thickness", type=int, default=5, help="Auxiliary binary lane mask thickness in original image pixels.")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    raw_root = Path(args.raw_root)
    downloads_root = Path(args.downloads_root)
    output_root = ensure_dir(args.output_root)

    if args.auto_extract:
        need_images = False
        for split in ("train", "val"):
            try:
                locate_image_dir(dataset_root, raw_root, split)
            except FileNotFoundError:
                need_images = True
        maybe_extract_official_zips(raw_root, downloads_root, need_images=need_images)

    train_stats = prepare_split(dataset_root, raw_root, output_root, "train", args.train_lane_json, args.mask_thickness)
    val_stats = prepare_split(dataset_root, raw_root, output_root, "val", args.val_lane_json, args.mask_thickness)

    summary = {
        "output_root": str(output_root),
        "train": train_stats,
        "val": val_stats,
        "format": "CLRKDNet/CULane-style: image symlinks, .lines.txt curve files, list/train_gt.txt, list/val.txt, and auxiliary masks generated from curves.",
    }
    (output_root / "prepare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
