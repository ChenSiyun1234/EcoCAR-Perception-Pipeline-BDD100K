# Stage2 Exp2 fixes after Exp2A/B/D/E/F results

## What the results showed

- Exp2B GCA improved vehicle detection loss compared with Exp2A, so task-specific gating is useful for detection/lane conflict.
- Exp2D lane-detail neck slightly improved lane point MAE, but detection loss degraded, so detail features should remain lane-only and should not pollute the detection branch.
- Exp2E Hungarian lane matching produced the largest lane point MAE improvement, so unstable lane slot ordering was the main cause of poor lane geometry.
- Exp2E also reduced lane existence accuracy, so the next fix is not more backbone complexity; it is better lane-slot classification diagnostics and background supervision.

## Code changes

1. `stage2/scripts/notebook_utils.py`
   - Replaced shell-pipe logging with `subprocess.Popen` line-by-line streaming.
   - Output is now mirrored to the notebook cell, Colab runtime log, and a Drive log file.

2. `stage2/fusion/losses.py`
   - Kept Hungarian matching for geometry.
   - Made focal lane-existence classification operate over all matched priors: matched lanes are positive, unmatched priors are background.
   - Added positive and negative classification loss diagnostics.
   - Added `compute_lane_eval_metrics()` with precision, recall, F1, matched point MAE, matched LineIoU, GT/pred/matched lane counts, FP lane slots, and FN lane slots.

3. `stage2/scripts/train_joint_model_experiment.py`
   - Uses the new lane metrics during validation.
   - Prints F1, precision, recall, point MAE, LineIoU, and gate statistics in epoch summaries.

4. `stage2/configs/exp04_rmt_gca_lane_matching_joint.yaml`
   - Keeps Exp2E as the core baseline.
   - Adds `eval.lane_exist_threshold: 0.45` and slightly increases classification weight to improve lane-slot calibration.

5. `stage2/configs/exp05_rmt_gca_lane_detail_matching_joint.yaml`
   - Keeps Exp2F as detail-neck + matching.
   - Uses the same fixed matching/classification diagnostics.

6. `stage2/notebooks/stage2_notebook_04_exp2e_rmt_gca_lane_matching_joint.ipynb`
   - Defaults to `DEBUG_MODE = True`: 2 epochs, 512 train samples, 256 val samples.
   - Set `DEBUG_MODE = False` for the 10-epoch short run.

7. `stage2/notebooks/stage2_notebook_05_exp2f_rmt_gca_lane_detail_matching_joint.ipynb`
   - Same debug-first workflow.

8. `stage2/scripts/compare_experiment_metrics.py`
   - Adds the new lane F1 / precision / recall / matched LineIoU metrics.
   - Can save trend plots with `--plot-dir`.

9. `stage1/notebooks/07_a5000_video_profile.ipynb`
   - Adds detection preflight over the first 10 video frames.
   - Prints raw prediction score statistics and NMS detection counts at confidence thresholds 0.25, 0.10, 0.05, and 0.01.
   - Uses adaptive NMS in the video loop so a partially trained checkpoint is not silently rendered as lane-only.
   - Keeps close-vehicle fragment cleanup for duplicate boxes on nearby vehicles.

## Validation performed here

- `python -m py_compile` passed for modified Stage2 scripts.
- `smoke_test_joint_models.py` passed for both Exp2E and Exp2F configs with random input and synthetic targets.

## Next run order

1. `stage2_notebook_00_prepare_joint_dataset.ipynb`
2. `stage2_notebook_04_exp2e_rmt_gca_lane_matching_joint.ipynb` with `DEBUG_MODE=True`
3. Same notebook with `DEBUG_MODE=False`
4. `stage2_notebook_05_exp2f_rmt_gca_lane_detail_matching_joint.ipynb` with `DEBUG_MODE=True`
5. Same notebook with `DEBUG_MODE=False`
6. `stage2_notebook_08_joint_eval_visualization_and_profile.ipynb`
7. Stage1 `07_a5000_video_profile.ipynb` for video output debug/fix
