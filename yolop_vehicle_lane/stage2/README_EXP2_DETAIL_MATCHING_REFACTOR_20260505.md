# Exp2 lane-geometry refactor, 2026-05-05

The latest Exp2A/Exp2B outputs show that GCA helps detection but does not solve lane curve geometry. Exp2B has lower validation detection loss than Exp2A, while `val/lane_point_mae` stays roughly flat around 0.52-0.54. This means the joint model can learn lane existence, but the predicted row-wise lane coordinates are still weak.

## Added experiments

### Exp2D: RMT-GCA + Lane Detail Neck

Files:
- `stage2/fusion/lane_detail_neck.py`
- `stage2/fusion/backbones.py`
- `stage2/configs/exp03_rmt_gca_lane_detail_joint.yaml`
- `stage2/notebooks/stage2_notebook_03_exp2d_rmt_gca_lane_detail_joint.ipynb`

Change:
- Detection keeps the RMT-style P3/P4/P5 path.
- Lane branch receives an extra C2/stride-4 detail path before the CLRKD-style curve head.
- The C2 detail feature is fused into P3, then propagated through refined P4/P5 lane features.

Purpose:
- Test whether lane geometry is weak because P3/P4/P5 alone lost too much high-resolution lane edge detail.

### Exp2E: RMT-GCA + Hungarian-style lane matching

Files:
- `stage2/fusion/losses.py`
- `stage2/configs/exp04_rmt_gca_lane_matching_joint.yaml`
- `stage2/notebooks/stage2_notebook_04_exp2e_rmt_gca_lane_matching_joint.ipynb`

Change:
- Lane loss now optionally matches predicted lane slots to GT lanes using a cost based on lane existence score, point L1, LineIoU, and XYTL parameters.
- This replaces fixed slot-order supervision when `use_lane_matching: true`.

Purpose:
- Test whether lane geometry is weak because predicted lane slots and GT lane order are not aligned.

### Exp2F: Lane Detail Neck + Hungarian-style lane matching

Files:
- `stage2/configs/exp05_rmt_gca_lane_detail_matching_joint.yaml`
- `stage2/notebooks/stage2_notebook_05_exp2f_rmt_gca_lane_detail_matching_joint.ipynb`

Change:
- Combines Exp2D and Exp2E.

Purpose:
- Main next candidate. If this improves `val/lane_point_mae`, the problem was likely both feature resolution and lane assignment.

## Training/logging fixes

Files:
- `stage2/scripts/notebook_utils.py`
- `stage2/scripts/train_joint_model_experiment.py`

Changes:
- Notebook command output now runs through direct `tee`, so training logs should appear in both the notebook output cell and Drive log files.
- The first training step and every `print_every` steps are printed.
- The train script logs whether lane matching is enabled and the geometry warmup scale for each epoch.
- Gradient-calibrated lane weight is now stored as an epoch-level runtime lambda instead of being used only on the first batch.
- Evaluation uses matched lane targets when lane matching is enabled, so `val/lane_point_mae` matches the training objective.

## Run order

1. `stage2_notebook_00_prepare_joint_dataset.ipynb` only if the dataset tar is missing or stale.
2. Skip rerunning Exp2A/Exp2B unless you want a new baseline.
3. Run `stage2_notebook_03_exp2d_rmt_gca_lane_detail_joint.ipynb`.
4. Run `stage2_notebook_04_exp2e_rmt_gca_lane_matching_joint.ipynb`.
5. Run `stage2_notebook_05_exp2f_rmt_gca_lane_detail_matching_joint.ipynb`.
6. Run `stage2_notebook_08_joint_eval_visualization_and_profile.ipynb` after at least one new tar exists.

Exp3 is intentionally lower priority until Exp2F shows whether lane geometry can be fixed inside the RMT-GCA + CLRKD fusion route.
