# Stage 2: RMT-PPAD lane-only baseline and CLRKDNet fusion

This Stage 2 replaces the previous Stage-2 flow. The old files were moved to `stage2_legacy_removed/` and the active Stage 2 now starts from a vendored copy of RMT-PPAD.

## What is included

1. `stage2/vendor/RMT-PPAD/`
   - Vendored RMT-PPAD code.
   - Patched only where required for lane-only training and the controlled CLRKDNet fusion experiment.

2. `stage2/vendor/CLRKDNet/`
   - Vendored CLRKDNet code for reference and later deeper fusion work.

3. `stage2/configs/rmt_ppad_lane_only.yaml`
   - Strict RMT-PPAD RT-DETR-L multi-task config.
   - Drivable-area segmentation is removed.
   - Detection class count is 1.
   - Segmentation class count is 1.

4. `stage2/configs/rmt_ppad_clrkd_fused_lane.yaml`
   - Same detection path as RMT-PPAD.
   - Lane decoder is changed to a CLRKDNet-inspired feature aggregation decoder.

5. `stage2/configs/bdd100k_vehicle_lane_rmt.yaml`
   - Dataset YAML for Stage-1 BDD100K vehicle+lane output.

## Recommended order

Run these notebooks in order:

1. `stage2/notebooks/00_prepare_rmt_dataset_links.ipynb`
2. `stage2/notebooks/01_sanity_check_rmt_ppad_lane_only.ipynb`
3. `stage2/notebooks/02_train_rmt_ppad_lane_only.ipynb`
4. `stage2/notebooks/03_train_rmt_ppad_clrkd_fused_lane.ipynb`

## RTX Pro 6000 default training settings

The default Stage-2 scripts use:

- image size: `640`
- batch size: `16`
- learning rate `lr0`: `0.0001`
- optimizer: `AdamW`
- cosine LR: enabled
- validation: every epoch
- checkpoint saving: every epoch
- AMP: disabled by default unless `--amp` is passed

This is intentionally conservative for the first full Stage-2 run. The RMT-PPAD detector is RT-DETR-L style and is heavier than the Stage-1 YOLOPX baseline, so a stable batch is more important than fully filling the 96 GB VRAM at the beginning.

## Dataset layout

Stage 1 normally creates:

```text
/content/bdd100k_vehicle5/images/train
/content/bdd100k_vehicle5/images/val
/content/bdd100k_vehicle5/labels/train
/content/bdd100k_vehicle5/labels/val
/content/bdd100k_vehicle5/masks/train
/content/bdd100k_vehicle5/masks/val
```

RMT-PPAD expects segmentation masks under:

```text
/content/bdd100k_vehicle5/mask/lane/train
/content/bdd100k_vehicle5/mask/lane/val
```

The preparation script creates symlinks from the Stage-1 mask folders to the RMT-PPAD layout.

## Commands

```bash
cd /content/drive/MyDrive/EcoCAR/yolop_vehicle_lane

python stage2/scripts/00_prepare_rmt_dataset_links.py --dataset-root /content/bdd100k_vehicle5

python stage2/scripts/02_sanity_check_stage2.py \
  --project-root /content/drive/MyDrive/EcoCAR/yolop_vehicle_lane \
  --dataset-root /content/bdd100k_vehicle5 \
  --model rmt_ppad_lane_only.yaml

python stage2/scripts/01_train_rmt_ppad_lane_only.py \
  --project-root /content/drive/MyDrive/EcoCAR/yolop_vehicle_lane \
  --epochs 250 --batch 16 --lr0 0.0001 --workers 8 --device 0

python stage2/scripts/03_train_rmt_ppad_clrkd_fused_lane.py \
  --project-root /content/drive/MyDrive/EcoCAR/yolop_vehicle_lane \
  --epochs 250 --batch 16 --lr0 0.0001 --workers 8 --device 0
```

## Important limitation

The CLRKDNet fusion in this package is a dense-lane-mask adaptation, not a full anchor-line CLRKDNet target-format conversion. This is deliberate: the current Stage-1 dataset produces dense lane masks, while CLRKDNet's original head predicts lane priors/curves. The fusion therefore imports the CLRKDNet architectural idea that is compatible with the current data pipeline: a lighter feature aggregation decoder and single lane prediction path.

## CLRKD curve-prior lane pipeline update

See `stage2/CLRKD_CURVE_PIPELINE.md` for the new BDD100K curve-label preparation and CLRKDNet lane-prior training flow. This is the strict CLRKDNet-style lane path and replaces the earlier mask-only compromise for the CLRKD stage.
