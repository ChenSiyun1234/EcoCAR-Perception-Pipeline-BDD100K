# Joint refactor changelog 2026-05-05

## Fixed stage organization

- Removed old Stage 2 notebook ordering and replaced it with active notebooks in execution order.
- Stage 2 notebooks now run complete joint models only. They do not train a detached lane-only head as the main experiment.
- Fixed Stage 1 notebook 03 result labeling so the saved JSON uses the selected `CONFIG` instead of the old hard-coded YOLOP/CSP baseline label.

## Exp2 implementation

- `exp01_rmt_shared_joint.yaml`: RMT-style shared backbone/neck with joint DETR vehicle detection and CLRKD-style lane curve head.
- `exp02_rmt_gca_joint.yaml`: same model with detection/lane task adapters and GCA gates.
- `exp03_rmt_gca_clrkd_kd_joint.yaml`: same as Exp2B with lane-teacher checkpoint support for CLRKD-style distillation.

## Exp3 implementation

- Replaced the previous YOLO26 black-box hook path with a custom official-YOLO26-inspired backbone/neck.
- `stage2/fusion/yolo26_inspired.py` adds:
  - `YOLO26Conv`
  - `YOLO26Bottleneck`
  - `YOLO26CSPBlock`
  - `YOLO26Stage`
  - `P5GlobalContextBlock`
  - `LanePreservingP3Block`
  - `YOLO26PANNeck`
  - `YOLO26InspiredJointBackboneNeck`
- `exp04_yolo26_inspired_joint.yaml`: Exp3A custom YOLO26-inspired joint CNN baseline.
- `exp05_yolo26_inspired_global_joint.yaml`: Exp3B adds P5 global context.
- `exp06_yolo26_inspired_gca_joint.yaml`: Exp3C adds RMT-style GCA task split.
- The old real-YOLO26 wrapper files were moved to `stage2/legacy/` and disabled so they are not mistaken for the active path.

## Loss implementation

- DETR branch uses Hungarian matching over query predictions.
- Detection loss: objectness/classification + L1 box + GIoU.
- Lane loss: focal existence + point SmoothL1 + xytl SmoothL1 + LineIoU + auxiliary mask + smoothness + optional teacher distillation.
- Joint loss supports gradient-norm lane-weight calibration and optional GCA gate regularization.

## Validation done here

- Python syntax compile passed for all Stage 2 scripts and fusion modules.
- Smoke forward/loss/backward passed for Exp1 through Exp6 on synthetic data.
- The validation log is stored at `stage2/validation/smoke_test_exp01_to_exp06_yolo26_inspired_20260505.txt`.

## Validation not done here

- Full Colab training was not run because the uploaded project does not include the full BDD100K dataset.
- Exp2C still requires a lane teacher checkpoint before the distillation term becomes a real teacher-supervised experiment.
