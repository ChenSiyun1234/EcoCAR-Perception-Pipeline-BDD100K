from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def count_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.iterdir() if item.is_file())


def ensure_lane_link(dataset_root: Path, split: str, allow_copy: bool) -> bool:
    source = dataset_root / "masks" / split
    target = dataset_root / "mask" / "lane" / split

    if not source.exists():
        print(f"[info] Stage-1 lane mask source is missing: {source}")
        print("[info] This is acceptable for the CLRKD curve/prior route; it does not train from masks.")
        return False

    target.parent.mkdir(parents=True, exist_ok=True)

    if target.is_symlink():
        current = Path(os.readlink(target))
        if current == source:
            return True
        target.unlink()

    if target.exists():
        if count_files(target) > 0:
            return True
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()

    try:
        target.symlink_to(source, target_is_directory=True)
    except OSError:
        if not allow_copy:
            raise
        shutil.copytree(source, target)
    return True


def run_curve_prepare(args: argparse.Namespace) -> None:
    script = Path(__file__).resolve().parent / "04_prepare_bdd_curve_labels.py"
    command = [
        sys.executable,
        str(script),
        "--dataset-root",
        str(args.dataset_root),
        "--raw-root",
        str(args.raw_root),
        "--downloads-root",
        str(args.downloads_root),
        "--output-root",
        str(args.output_root),
        "--mask-thickness",
        str(args.mask_thickness),
    ]
    if args.auto_extract:
        command.append("--auto-extract")
    if args.train_lane_json:
        command += ["--train-lane-json", args.train_lane_json]
    if args.val_lane_json:
        command += ["--val-lane-json", args.val_lane_json]

    print("[info] Preparing CLRKDNet curve/prior labels:")
    print(" ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare Stage-2 dataset assets. If old mask folders exist, create RMT mask links. "
            "If masks are missing, continue with the CLRKD curve/prior dataset instead."
        )
    )
    parser.add_argument("--dataset-root", default="/content/bdd100k_vehicle5")
    parser.add_argument("--raw-root", default="/content/bdd100k_raw")
    parser.add_argument("--downloads-root", default="/content/drive/MyDrive/EcoCAR/downloads")
    parser.add_argument("--output-root", default="/content/bdd100k_clrkd_curve")
    parser.add_argument("--allow-copy", action="store_true")
    parser.add_argument("--auto-extract", action="store_true")
    parser.add_argument("--skip-curve", action="store_true", help="Only try old RMT mask links; do not prepare CLRKD curve labels.")
    parser.add_argument("--train-lane-json", default=None)
    parser.add_argument("--val-lane-json", default=None)
    parser.add_argument("--mask-thickness", type=int, default=5)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    linked = []
    for split in ("train", "val"):
        linked.append(ensure_lane_link(dataset_root, split, args.allow_copy))

    checks = {
        "train_images": dataset_root / "images" / "train",
        "val_images": dataset_root / "images" / "val",
        "train_labels": dataset_root / "labels" / "train",
        "val_labels": dataset_root / "labels" / "val",
        "old_train_lane_masks": dataset_root / "mask" / "lane" / "train",
        "old_val_lane_masks": dataset_root / "mask" / "lane" / "val",
    }

    for name, path in checks.items():
        print(f"{name}: {count_files(path)} files -> {path}")

    if args.skip_curve:
        print("[done] Mask-link-only preparation finished.")
        return

    run_curve_prepare(args)

    output_root = Path(args.output_root)
    print("[done] Stage-2 curve dataset is ready.")
    print(f"curve_train_list: {output_root / 'list' / 'train_gt.txt'}")
    print(f"curve_val_list:   {output_root / 'list' / 'val.txt'}")
    print(f"summary:          {output_root / 'prepare_summary.json'}")


if __name__ == "__main__":
    main()
