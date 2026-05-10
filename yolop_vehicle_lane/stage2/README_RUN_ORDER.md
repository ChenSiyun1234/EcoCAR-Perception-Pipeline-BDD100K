# Stage 2 run order and debug workflow

Do not start with Exp3. The current evidence says Exp2 lane matching is the highest-value fix.

## Recommended order

1. `stage2_notebook_00_prepare_joint_dataset.ipynb`
   - Builds `/content/drive/MyDrive/EcoCAR/datasets/bdd100k_clrkd_curve.tar`.
   - Must be rerun if detection labels or lane `.lines.txt` files are missing.

2. `stage2_notebook_04_exp2e_rmt_gca_lane_matching_joint.ipynb`
   - Runs the fixed Exp2E baseline: RMT-GCA + CLRKD lane head + Hungarian lane matching.
   - First keep `DEBUG_MODE = True`: 2 epochs, 512 train samples, 256 val samples.
   - Then set `DEBUG_MODE = False`: 10 epochs, 5000 train samples, 1000 val samples.

3. `stage2_notebook_05_exp2f_rmt_gca_lane_detail_matching_joint.ipynb`
   - Runs the fixed Exp2F model: Exp2E plus a lane-only C2/P3 detail neck.
   - Use the same debug-first workflow.

4. `stage2_notebook_08_joint_eval_visualization_and_profile.ipynb`
   - Compares result tars and visualization outputs.

5. Stage 1 `07_a5000_video_profile.ipynb`
   - Re-run after the Stage1 video preflight/adaptive-NMS fix if the output video has lane only or duplicate close-vehicle boxes.

## Logging

Every notebook call uses `stage2.scripts.notebook_utils.run_streaming()`.
It mirrors subprocess output to three places:

1. the notebook cell output,
2. the Colab runtime log,
3. `/content/drive/MyDrive/EcoCAR/yolop_vehicle_lane/stage2/logs/*.log`.

## Metrics to watch

- `val/det_loss`: lower means the vehicle detector is improving.
- `val/lane_point_mae`: lower means predicted lane points are closer to matched GT points.
- `val/lane_exist_f1`: better than raw existence accuracy for checking lane-slot calibration.
- `val/lane_exist_precision`: low value means too many false-positive lane slots.
- `val/lane_exist_recall`: low value means the model is missing true lane slots.
- `val/matched_line_iou`: higher means the predicted lane curve shape overlaps the GT curve better.
- `train/gate/p3_lane_mean`, `train/gate/p4_lane_mean`, `train/gate/p5_lane_mean`: show whether GCA is actually using scale-specific lane features.
