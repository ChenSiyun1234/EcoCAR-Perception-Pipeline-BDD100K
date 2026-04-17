# YOLOPv2 Alignment Repair — Summary

See `AUDIT_YOLOPV2_ALIGNMENT.md` for the dimension-by-dimension table
this summary operationalises.

## What changed

### Phase-1 baseline is now a true YOLOP → YOLOPv2 edit, not a freehand rewrite

- `lib/models/yolopv2_baseline.py` replaced. The new file is YOLOP's
  `MCnet_0` block_cfg with the drivable-area decoder removed and the
  following paper-justified deltas:
  - backbone `BottleneckCSP` → `ELAN` (with `groups=2` at stride ≥ 16)
  - neck keeps `SPP` (not `SPPCSPC` — that was a wrong inference)
  - lane decoder's `Upsample + Conv` stages → `ConvTranspose2d`
    (deconvolution, per paper §3)
  - lane decoder still taps from layer 16 (same as YOLOP MCnet_0)
  - detection head: `Detect(nc=5, …)` with the YOLOP BDD-kmeans anchors
- `lib/models/common.py::ELAN` gained a `groups` argument so the
  `E-ELAN with group convolution` claim from the paper can be set in
  the block_cfg.

### Loss stack aligned to YOLOPv2 paper

- `lib/core/loss.py::get_loss` now supports `LOSS.LL_FL_GAMMA`. When
  set > 0, `BCEseg` is focal-wrapped (paper §3: "focal loss for lane
  seg"). YOLOP's existing `FL_GAMMA` still controls det cls/obj focal.
- YOLOPv2 YAML sets `FL_GAMMA=1.5` and `LL_FL_GAMMA=1.5`. YOLOP YAML
  keeps both at 0.

### Training protocol aligned to YOLOPv2 paper §3

`configs/yolopv2_vehicle_lane_baseline.yaml`:

| param        | before | now (paper) |
|--------------|-------:|------------:|
| OPTIMIZER    | adam   | sgd         |
| LR0          | 0.001  | 0.01        |
| WD           | 0.0005 | 0.005       |
| END_EPOCH    | 100    | 300         |
| WARMUP_EPOCHS| 3.0    | 3.0         |
| MOMENTUM     | 0.937  | 0.937       |
| Train size   | 640×640| 640×640     |
| **Test size**| 640×640| **640×384** |
| FL_GAMMA     | 0      | 1.5         |
| LL_FL_GAMMA  | 0      | 1.5         |

Mosaic and MixUp are wired in `lib/dataset/AutoDriveDataset.py`
already; the YAML toggles them on. `notebook 02` now uses
`cfg.TEST.IMAGE_SIZE` to build the validation dataloader at 384 short
side.

### Lane preprocessing aligned

- `notebook 00` now renders **train masks at width 8** and **val masks
  at width 2** — matches paper §3. Each split uses `overwrite=True` so
  re-running picks up the width change.
- `lib/utils/lane_render.py::render_lane_mask` docstring now records
  that BDD centerline pairing is [INFERRED] — we render each annotated
  poly2d directly. The width-8 train masks obscure this divergence in
  practice; the width-2 test masks are stricter, so metrics on the
  val set may be slightly pessimistic vs the paper's tooling.

### Phase 1 no longer polluted by stage-2 experiments

Quarantined under `stage2/` (nothing phase-1 imports these):
- `stage2/lib/models/yolopv2_detrlane.py`
- `stage2/lib/models/lane_set_head.py`
- `stage2/lib/core/lane_set_loss.py`
- `stage2/lib/core/loss_detrlane.py`
- `stage2/lib/dataset/bdd_points.py`
- `stage2/configs/yolopv2_detrlane_vehicle_lane.yaml`
- `stage2/notebooks/05_lane_head_and_loss_upgrade.ipynb`

`lib/models/__init__.py`, `lib/core/__init__.py`, `lib/dataset/__init__.py`
no longer import any stage-2 symbols, so the phase-1 factory surface
is clean.

### Notebooks

Active phase-1 notebooks:
- `00_rebuild_dataset_and_lane_cache.ipynb` — masks-only archive,
  width 8/2 split, single `.tar.gz` upload to
  `EcoCAR/datasets/bdd100k_vehicle5.tar.gz`.
- `01_augmentation_lab.ipynb` — visualizes what the training
  dataloader emits including Mosaic/MixUp path.
- `02_train_yolopv2_vehicle_lane_baseline.ipynb` — BASELINE switch
  between `YOLOP` and `YOLOPv2`; reads the matching YAML; uses
  `cfg.TEST.IMAGE_SIZE` for rectangular val letterbox.
- `03_eval_and_backbone_ablation.ipynb` — evaluation entry point.
- `07_a5000_video_profile.ipynb` — A5000 MFU / latency profile.

Stage-2 notebook (optional, not part of the faithful baseline):
- `stage2/notebooks/05_lane_head_and_loss_upgrade.ipynb` — DETR-style
  point-lane experiment. This is *not* a YOLOPv2 deviation — it's a
  deliberate Stage-2 research direction.

## What now matches YOLOPv2

- Backbone is ELAN (paper text), not the earlier SPPCSPC-style rewrite.
- Detection branch keeps SPP in the neck (paper text).
- Lane branch taps from the encoder like YOLOP and uses deconvolution
  (paper text).
- Training protocol (SGD / 0.01 / 0.937 / 0.005 / 300 epochs / warmup 3).
- Mosaic + MixUp active in the real dataloader.
- Focal loss on both detection and lane seg (paper text).
- Train 640×640 / test 640×384.
- Lane masks drawn at width 8 train / width 2 test.

## What is still `[INFERRED]`

Each of these is tagged in code; swap in upstream values if YOLOPv2
source is ever released:

1. **E-ELAN `groups` value.** Paper names "group conv" but doesn't
   publish the number. We use `groups=2` at stride ≥ 16. Parameter
   count lands in the paper's 38-40 M band. Site:
   `lib/models/yolopv2_baseline.py::YOLOPv2Cfg`.
2. **Exact E-ELAN depth / channel schedule.** Mirrors YOLOP's MCnet_0
   channels; paper doesn't publish YOLOPv2's internal channels.
3. **Focal loss γ.** Paper says "focal" but not γ. We default to 1.5
   (YOLOv5 default, also YOLOP's intended setting). Sites:
   `lib/core/loss.py::get_loss` + `configs/yolopv2_vehicle_lane_baseline.yaml`.
4. **Hybrid focal + dice variant.** Paper mentions this as an
   ablation. We don't add it to the baseline; it would be a second
   config. Deferred.
5. **BDD lane centerline pairing.** Paper renders centerlines between
   two annotated edges. BDD100K doesn't provide the pairing. We
   render each `poly2d` directly; width 8 masks largely fuse the
   double-edge visually but strict paper equivalence would need a
   pairing pass. Site: `lib/utils/lane_render.py::render_lane_mask`.
6. **Lane output contract.** Paper's demo ships `ll` as 1-ch sigmoid
   at H/2×W/2. Our model outputs 2-ch sigmoid at full resolution so
   the existing `MultiHeadLoss` and `validate()` work unchanged.
   Deployment code can compress to the paper contract by taking
   `out[..., 1]` and 2× downsample. Site:
   `lib/models/yolopv2_baseline.py::YOLOPv2Cfg` (final layer) + module
   docstring.

## Acceptance check

- [x] Phase-1 baseline is a YOLOP edit, not a renamed rewrite.
- [x] Lane branch is deconvolution, not upsample-only.
- [x] Augmentation (Mosaic + MixUp) is in the real dataloader path.
- [x] Training protocol matches paper (SGD / 0.01 / 0.937 / 0.005 /
      300 epochs / warmup 3).
- [x] Lane preprocessing uses paper widths (8 train / 2 test).
- [x] Docs distinguish MATCH / PARTIAL / MISMATCH / INFERRED.
- [x] Phase-1 factory (`lib/models/__init__.py`) does not import
      stage-2.
- [x] Drive persistence: single `.tar.gz` to `EcoCAR/datasets/`.
- [x] Only drivable-area is intentionally deleted; every other
      YOLOPv2 component is present.
