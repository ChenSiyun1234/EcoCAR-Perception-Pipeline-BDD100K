# Stage-2 CLRKD curve-prior lane pipeline

This update adds a true CLRKDNet-style curve/prior lane-training path for BDD100K. The previous Stage-2 CLRKD fusion was only a CLRKDNet-inspired dense mask decoder inside RMT-PPAD. It did not use CLRKDNet's `CLRHead`, line priors, dynamic assignment, `lane_line` targets, or LineIoU loss. This new path creates BDD100K curve labels and trains the original CLRKDNet lane formulation instead of treating lane detection as only binary mask segmentation.

## What changed

1. `stage2/scripts/04_prepare_bdd_curve_labels.py`
   - Reads BDD100K per-image JSON labels using the same robust schema handling used in the DETR_GeoLane pipeline.
   - Supports old BDD `frames -> objects` records and newer Scalabel-style `labels` records.
   - Parses `poly2d` / `seg2d`, including line and Bezier control points.
   - Writes CLRKDNet/CULane-style `.lines.txt` files beside each image.
   - Writes `list/train_gt.txt`, `list/val.txt`, and `list/test.txt`.
   - Writes auxiliary masks generated from the same curve labels only for CLRKDNet's auxiliary segmentation loss. The primary training target is still `lane_line`, not the mask.

2. `stage2/vendor/CLRKDNet/clrkd/datasets/bdd100k_curve.py`
   - Adds a new `BDDCurve` dataset class registered with CLRKDNet.
   - Loads BDD curve `.lines.txt` files and passes lane points into CLRKDNet's `GenerateLaneLine` processor.
   - Uses CLRKDNet's native `lane_line` tensor target: class scores, start position, angle, length, and sampled x-offsets.
   - Adds a lightweight validation metric based on greedy lane matching by average point distance so the training loop can validate without the official CULane evaluation binary.

3. `stage2/configs/BDD100K_CLRKD_Curve.py`
   - Uses CLRKDNet's original `Detector + ResNetWrapper + Aggregator + CLRHead` structure.
   - Keeps CLRKDNet losses: focal classification, x/y/theta/length regression, LineIoU, and a small auxiliary segmentation loss.
   - Uses `batch_size = 24` as a stable RTX PRO 6000 starting point.

4. `stage2/scripts/05_train_bdd_clrkd_curve.py`
   - Wrapper that launches CLRKDNet training from the project root.

## How to run

From the project root:

```bash
python stage2/scripts/00_prepare_rmt_dataset_links.py \
  --dataset-root /content/bdd100k_vehicle5 \
  --raw-root /content/bdd100k_raw \
  --downloads-root /content/drive/MyDrive/EcoCAR/downloads \
  --output-root /content/bdd100k_clrkd_curve \
  --auto-extract
```

Then train:

```bash
python stage2/scripts/05_train_bdd_clrkd_curve.py \
  --project-root /content/drive/MyDrive/EcoCAR/yolop_vehicle_lane \
  --gpus 0 \
  --work-dirs /content/drive/MyDrive/EcoCAR/training_runs/stage2_clrkd_curve
```

## Important distinction

This is the strict CLRKDNet lane-branch reproduction/adaptation path. It does not yet replace RMT-PPAD's MTDETR detection decoder with a unified joint curve head. That joint integration needs a separate, riskier change: RMT's Ultralytics dataloader must emit CLRKDNet `lane_line` tensors, and `MTDETRDLoss` must accept both object targets and curve-prior lane targets. This package prepares the correct dataset format and a runnable CLRKDNet curve-training baseline first, so the lane method is no longer limited to mask segmentation.
