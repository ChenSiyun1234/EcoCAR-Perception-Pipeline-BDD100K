# Stage 2 current run order and Exp2 fix notes

## Why the run order changed

Current trends show that Exp2A/Exp2B reduce validation loss but do not reduce lane point MAE, Exp2D gives only a small geometry gain, and Exp2E gives the largest geometry improvement because Hungarian lane matching fixes unstable lane-slot order. Therefore, the next runs should focus on Exp2E/Exp2F, not Exp3.

## Recommended run order

1. `stage2_notebook_00_prepare_joint_dataset.ipynb`
2. `stage2_notebook_04_exp2e_rmt_gca_lane_matching_joint.ipynb` in debug mode
3. Same notebook with `DEBUG_MODE = False` for the 10-epoch short run
4. `stage2_notebook_05_exp2f_rmt_gca_lane_detail_matching_joint.ipynb` in debug mode
5. Same notebook with `DEBUG_MODE = False` for the 10-epoch short run
6. `stage2_notebook_08_joint_eval_visualization_and_profile.ipynb`
7. Stage1 video profiling notebook after the detection preflight confirms boxes are produced

Do not run Exp3 yet. Exp3 should wait until lane matching, existence calibration, and comparable metrics are stable.

## What was fixed in this update

1. Notebook output logging now uses `run_streaming()` with unbuffered subprocess output and heartbeat messages, so long silent first batches are visible in the Jupyter output cell and in the Drive log file.
2. Exp2E/Exp2F lane classification now uses balanced positive/negative focal loss to reduce the existence-calibration failure seen after Hungarian matching.
3. Exp2E/Exp2F now log scaled lane loss, unweighted lane loss, lane components, best existence F1 threshold, CLRKD-style lane F1, and DETR-style AP50 proxy.
4. Original-metric source adapters were added in `stage2/metrics/original_metric_adapters.py`. They point to the vendored RMT and CLRKDNet metric source files and provide runnable fallback metrics for this project format.
5. `stage2/scripts/analyze_stage2_trends.py` can regenerate trend plots from metric JSON files or experiment tar files.

## What to watch during training

- `val/lane_point_mae`: lower means better curve geometry.
- `val/lane_exist_best_f1` and `val/lane_exist_best_threshold`: show whether the existence classifier is calibrated or only using the wrong threshold.
- `val/lane/cls_pos` vs `val/lane/cls_neg`: tells whether positives or negatives dominate the lane existence loss.
- `val/lane/unweighted_total` vs `val/lane_loss_scaled`: separates real component trend from geometry/lambda scaling.
- `val/det/metric_map50`: approximate vehicle AP50; use this along with `val/det_loss`, not loss alone.
