# Stage 1 YOLOPX consistency + Stage 2 fusion plan — implementation report

Date: 2026-05-04

## Files inspected

### Stage 1 baseline (canonical YOLOPX paths)
- `stage1/notebooks/02_train_yolopx_vehicle_lane_baseline.ipynb`
  → confirmed `CONFIG = 'YOLOPX'`, run_name `yolopx`, checkpoint dir
  `.../stage1/checkpoints/yolopx/`, best file `best.pth` (mirrored from
  `best_joint.pth`).
- `stage1/configs/yolopx_vehicle_lane_baseline.yaml` — model `YOLOPX`,
  NC=1, train img 640x640, val img 640x384, AdamW LR0=3e-4.

### Stage 1 post-02 notebooks (needed updates)
- `stage1/notebooks/02b_stage1_diagnostics.ipynb` — `yaml_map` was missing
  `'YOLOPX'` entry, would have raised `KeyError` at runtime.
- `stage1/notebooks/03_eval_and_backbone_ablation.ipynb` — defaulted to
  `CONFIG = 'YOLOP'`, no YOLOPX entry.
- `stage1/notebooks/06_final_train_eval_export.ipynb` — defaulted to
  `CONFIG = 'YOLOP'`.
- `stage1/notebooks/07_a5000_video_profile.ipynb` — defaulted to
  `CONFIG = 'YOLOP'`, run_name `yolop_old`.

### Stage 2 sources surveyed
- Sister project `DETR_GeoLane_pipeline/src/lane_targets.py` — full lane
  parser with Bezier handling, polyline resampling, and per-image target
  assembly.
- `external_repos/CLRKDNet-master/clrkd/...` — lane prior representation
  `(max_lanes, 78)` = `[neg, pos, start_y, start_x, theta, length, x0..x71]`,
  CLRHead architecture, LineIoU/Focal/SmoothL1 losses.
- `external_repos/RMT-PPAD-main/ultralytics/...` — `ResNetLayer` backbone
  block, `AIFI` / `STB` neck blocks, `MTDETRDecoder` detection head,
  `MTDETRDLoss`, `MSDeformAttn`. Vendored copy at
  `stage2/vendor/RMT-PPAD/` is identical except for two small fixes in
  `mtdetr/predict.py` and `mtdetr/val.py`.
- `yolo26_pipeline/src/multitask_model.py` — backbone+neck extracted from
  Ultralytics YOLO26 model; `build_multitask_model(cfg, weights=...)`
  factory; `warm_start_from_checkpoint` for backbone-only weight loading.

## Files changed / added

### Stage 1 (Part A)
- **EDITED** `stage1/notebooks/02b_stage1_diagnostics.ipynb`
  cell 2: rewrote with `CONFIG = 'YOLOPX'` default, added YOLOPX to
  yaml_map and run_name_map, added loud preflight printout, fail-loud
  on missing checkpoint.
- **EDITED** `stage1/notebooks/03_eval_and_backbone_ablation.ipynb`
  cells 2 + 3: defaulted to YOLOPX, added YOLOPX entries, added
  preflight, fail-loud `FileNotFoundError` if `best.pth` is missing.
- **EDITED** `stage1/notebooks/06_final_train_eval_export.ipynb`
  cell 2: defaulted to YOLOPX, fixed run_name_map to
  `'YOLOPX': 'yolopx'`, added preflight + fail-loud on missing checkpoint.
- **EDITED** `stage1/notebooks/07_a5000_video_profile.ipynb`
  cell 3: defaulted to YOLOPX, replaced the `yolop_old` run_name (which
  no longer matches the active YOLOP run), added preflight + fail-loud.
- **ADDED** `stage1/notebooks/_update_to_yolopx.py`
  one-shot script that produced the four edits above. Re-runnable.

### Stage 2 — fusion module (Parts B, C, D, E)
- **ADDED** `stage2/fusion/__init__.py`
- **ADDED** `stage2/fusion/lane_targets.py`
  Ports DETR_GeoLane logic: BDD JSON parsing, Bezier handling, polyline
  resampling, fixed-size `(existence, points, visibility, lane_type)`
  targets, soft polyline mask renderer, `LaneLabelCache` class with
  diagnostics method.
- **ADDED** `stage2/fusion/losses.py`
  `FusionLossConfig` + `FusionLaneLoss` (focal BCE existence, masked
  SmoothL1 coord regression, 1D LineIoU, Dice+BCE aux mask, second-
  difference smoothness, optional distillation),
  `UncertaintyMultiTaskLoss`, `compute_grad_cosine`.
- **ADDED** `stage2/fusion/lane_head.py`
  `CurveLaneHead` — light CLRKD-style head: top-down feature merge,
  learnable lane-prior queries, separate cls / coord / lane-class /
  aux-mask MLPs.
- **ADDED** `stage2/fusion/model.py`
  `FusionModel` wrapper that combines a backbone, optional detection
  head, and the lane head. Lane-input scales are pluggable via
  `lane_in_indices`.

### Stage 2 — configs
- **ADDED** `stage2/configs/clrkd_curve_baseline.yaml` (Experiment 2 light)
- **ADDED** `stage2/configs/rmt_clrkd_basic_fusion.yaml` (Experiment 3)
- **ADDED** `stage2/configs/yolo26_clrkd_experiment.yaml` (Experiment 6)

### Stage 2 — notebooks
- **ADDED** `stage2/notebooks/04_prepare_clrkd_curve_dataset_colab.ipynb`
- **ADDED** `stage2/notebooks/05_train_clrkd_curve_baseline_colab.ipynb`
- **ADDED** `stage2/notebooks/06_train_rmt_clrkd_basic_fusion_colab.ipynb`
- **ADDED** `stage2/notebooks/07_train_rmt_backbone_clrkd_curve_head_colab.ipynb`
- **ADDED** `stage2/notebooks/08_yolo26_backbone_neck_experiment_colab.ipynb`
- **ADDED** `stage2/notebooks/09_stage2_eval_video_profile_colab.ipynb`
- **ADDED** `stage2/notebooks/_build_stage2_notebooks.py`
  generator script (re-runnable).
- **ADDED** `stage2/notebooks/_fix_escapes.py`
  post-process to remove stray `\\'` escapes from generated source.

### Stage 2 — documentation
- **ADDED** `stage2/STAGE2_PLAN.md` — full plan, experiment sequence,
  loss design, eval requirements, Colab discipline.
- **ADDED** `STAGE2_IMPLEMENTATION_REPORT.md` (this file).

### Files left untouched (intentional)
- `stage2/scripts/04_prepare_bdd_curve_labels.py` — already produces the
  CULane-style `.lines.txt` outputs the vendored CLRKDNet trainer expects.
  Reused from notebook 04.
- `stage2/scripts/05_train_bdd_clrkd_curve.py` — wrapper around vendored
  CLRKDNet `main.py`. Reused from notebook 05.
- `stage2/configs/BDD100K_CLRKD_Curve.py` — Python-style config used by
  the vendored CLRKDNet trainer.
- `stage2/vendor/CLRKDNet/`, `stage2/vendor/RMT-PPAD/` — vendored, not
  modified.
- All legacy `stage2/notebooks/00..03_*.ipynb`, `stage2/configs/rmt_ppad_*.yaml`,
  `stage2/configs/bdd100k_vehicle_lane_rmt.yaml` — kept as historical
  comparison points per Part H rule 6.

## What still needs to be run (in Colab)

1. **Stage 1 verification**
   - Open `stage1/notebooks/02b_stage1_diagnostics.ipynb`, run preflight
     cell (cell 2). Should print model family `YOLOPX`, no errors.
   - Open `stage1/notebooks/03_eval_and_backbone_ablation.ipynb`,
     `06_final_train_eval_export.ipynb`, `07_a5000_video_profile.ipynb`.
     Each should fail loudly if the YOLOPX checkpoint at
     `/content/drive/MyDrive/EcoCAR/yolop_vehicle_lane/stage1/checkpoints/yolopx/best.pth`
     is missing — that is the intended behavior.

2. **Stage 2 — Experiment 1: prepare CLRKD curve dataset**
   - Run `stage2/notebooks/04_prepare_clrkd_curve_dataset_colab.ipynb`.
   - Validates output tar at
     `/content/drive/MyDrive/EcoCAR/datasets/bdd100k_clrkd_curve.tar`.
   - First run will take roughly 20-30 minutes.

3. **Stage 2 — Experiment 2 (strict): vendored CLRKDNet lane-only baseline**
   - Run `stage2/notebooks/05_train_clrkd_curve_baseline_colab.ipynb`.
   - Output tar:
     `/content/drive/MyDrive/EcoCAR/training_runs/stage2_clrkd_lane_only.tar`.

4. **Stage 2 — Experiment 3: basic in-house lane head training**
   - Run `stage2/notebooks/06_train_rmt_clrkd_basic_fusion_colab.ipynb`.
   - Currently trains lane-only on the CSP backbone (detection head wiring
     left as a clearly marked TODO inside the notebook). Run this once to
     confirm `stage2.fusion` modules behave correctly under real data
     before fusing in detection.

5. **Stage 2 — evaluation**
   - Run `stage2/notebooks/09_stage2_eval_video_profile_colab.ipynb`
     against the checkpoint produced by notebook 06.
   - Outputs `stage2_eval_summary.json` to
     `/content/drive/MyDrive/EcoCAR/training_runs/`.

6. **Stage 2 — heavier experiments (skeletons)**
   - Notebooks 07 (RMT backbone) and 08 (YOLO26 backbone) are runnable
     scaffolds with explicit TODOs. They need:
     - 07: detection-head wiring (RT-DETR `MTDETRDecoder` + `MTDETRDLoss`).
     - 08: confirmation of YOLO26 path from Drive vs `/content`.

## Open follow-ups not implemented

- **CLRKDNet distillation losses (attention map / prior embedding /
  logit)** — the loss module exposes a `w_distill` weight and a teacher
  dict but only logit-style and coord-style KD are wired. Attention-map
  distillation requires hooking specific intermediate features, which is
  experiment-5 scope.
- **PCGrad** — `compute_grad_cosine` is provided as a diagnostic so the
  decision to apply PCGrad can be data-driven; the surgery itself is not
  in `losses.py` yet.
- **F1 / official lane evaluation** — `09_*` reports a LineIoU proxy
  and existence accuracy. CULane-style F1 requires the official
  evaluation binary, which is the same approach the vendored CLRKDNet
  trainer takes.

## Sanity checks performed

- All four edited Stage 1 notebooks: JSON re-loads cleanly; cell counts
  preserved; `'YOLOPX'` literal present in each.
- All six new Stage 2 notebooks: JSON re-loads cleanly; cell counts:
  04=13, 05=11, 06=11, 07=9, 08=7, 09=11.
- `stage2/fusion/{__init__, lane_targets, losses, lane_head, model}.py`
  pass `python -m py_compile`.
- Notebook code cells (excluding pure-shell `!...` cells and IPython
  magic-mixed Drive-mount cells) parse cleanly via `ast.parse`.

## Git hygiene reminder

Many `__pycache__/` directories appear in `git status`. They are not part
of this change. Add to `.gitignore` if not already covered, but do not
include in the same commit as the Stage 1 / Stage 2 edits.
