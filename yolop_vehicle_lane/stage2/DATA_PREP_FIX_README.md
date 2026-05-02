Fix for missing /content/bdd100k_vehicle5/masks/train:
1. `stage2/scripts/00_prepare_rmt_dataset_links.py` no longer fails when Stage-1 masks are missing.
2. If masks are missing, it proceeds to CLRKDNet curve/prior label generation.
3. `stage2/scripts/04_prepare_bdd_curve_labels.py` can now locate images from either the Stage-1 dataset root or extracted official BDD100K image folders.
4. It can locate lane labels from both per-image JSON folders and official consolidated `lane_train.json` / `lane_val.json` files.
5. With `--auto-extract`, it extracts official archives from `/content/drive/MyDrive/EcoCAR/downloads` into `/content/bdd100k_raw` only when needed.
