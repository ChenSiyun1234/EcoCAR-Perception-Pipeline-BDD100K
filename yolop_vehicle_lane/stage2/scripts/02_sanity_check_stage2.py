from __future__ import annotations

import argparse
import sys
from pathlib import Path


def count_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.iterdir() if item.is_file())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default="/content/drive/MyDrive/EcoCAR/yolop_vehicle_lane")
    parser.add_argument("--dataset-root", default="/content/bdd100k_vehicle5")
    parser.add_argument("--model", default="rmt_ppad_lane_only.yaml")
    args = parser.parse_args()

    project_root = Path(args.project_root).expanduser().resolve()
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    sys.path.insert(0, str(project_root / "stage2" / "vendor" / "RMT-PPAD"))

    from ultralytics import MTDETR

    model_yaml = project_root / "stage2" / "configs" / args.model
    model = MTDETR(str(model_yaml))

    print(f"model_yaml: {model_yaml}")
    print(f"model_class: {type(model.model).__name__}")
    print(f"decoder: {type(model.model.model[-1]).__name__}")
    print(f"seg_decoder: {getattr(model.model.model[-1], 'seg_decoder_name', 'unknown')}")
    print(f"seg_head: {type(model.model.model[-1].seg_head).__name__}")

    for split in ("train", "val"):
        print(f"{split}_images: {count_files(dataset_root / 'images' / split)}")
        print(f"{split}_labels: {count_files(dataset_root / 'labels' / split)}")
        print(f"{split}_lane_masks: {count_files(dataset_root / 'mask' / 'lane' / split)}")


if __name__ == "__main__":
    main()
