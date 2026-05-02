from __future__ import annotations

import argparse
import sys
from pathlib import Path


def add_vendor_path(project_root: Path) -> None:
    vendor_root = project_root / "stage2" / "vendor" / "RMT-PPAD"
    sys.path.insert(0, str(vendor_root))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default="/content/drive/MyDrive/EcoCAR/yolop_vehicle_lane")
    parser.add_argument("--data", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--name", default="rmt_ppad_clrkd_fused_lane")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr0", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="0")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--val-period", type=int, default=1)
    parser.add_argument("--save-period", type=int, default=1)
    parser.add_argument("--sfl", type=float, default=1.0)
    parser.add_argument("--tl", type=float, default=1.0)
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    add_vendor_path(project_root)

    from ultralytics import MTDETR

    data_yaml = Path(args.data) if args.data else project_root / "stage2" / "configs" / "bdd100k_vehicle_lane_rmt.yaml"
    model_yaml = Path(args.model) if args.model else project_root / "stage2" / "configs" / "rmt_ppad_clrkd_fused_lane.yaml"
    run_project = project_root / "stage2" / "runs"

    model = MTDETR(str(model_yaml))

    train_kwargs = dict(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        lr0=args.lr0,
        optimizer="AdamW",
        cos_lr=True,
        workers=args.workers,
        device=args.device,
        amp=args.amp,
        project=str(run_project),
        name=args.name,
        exist_ok=True,
        save=True,
        save_period=args.save_period,
        val=True,
        val_period=args.val_period,
        mask_threshold=[0.5],
        sfl=args.sfl,
        tl=args.tl,
    )

    if args.resume:
        train_kwargs["resume"] = args.resume

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
