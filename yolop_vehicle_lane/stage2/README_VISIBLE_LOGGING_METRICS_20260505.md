# Stage 2 visible logging and current Exp01/Exp02 notes

## What changed

1. Added `stage2/scripts/notebook_utils.py`.
   - `run_streaming(...)` captures subprocess stdout/stderr and immediately re-prints it from the notebook kernel.
   - This makes the same training lines visible in the Jupyter/Colab output cell and in the runtime log.
   - It also writes a copy under `/content/drive/MyDrive/EcoCAR/training_runs/notebook_logs/`.

2. Updated every Stage 2 notebook to call `run_streaming(...)` instead of plain `subprocess.check_call(...)` for smoke tests, dataset preparation, and training.

3. Updated `stage2/scripts/train_joint_model_experiment.py`.
   - Forces line-buffered stdout/stderr.
   - Prints preflight information before training.
   - Prints first batch and periodic batch progress with loss, speed, and ETA.
   - Prints `[epoch_summary]` after each epoch with the main train/val metrics and peak GPU memory.

4. Updated `stage1/notebooks/07_a5000_video_profile.ipynb`.
   - Uses the YOLOPX `predict(...)` inference wrapper when available.
   - Robustly unpacks YOLOPX detection output before NMS.
   - Lowers the video confidence threshold to 0.20.
   - Draws raw detections when the tracker has no confirmed boxes yet.
   - Writes detection counters into the video profile JSON.

## Where to find outputs

Stage 2 training artifacts are saved as tar archives in:

`/content/drive/MyDrive/EcoCAR/training_runs/`

The log copies are saved in:

`/content/drive/MyDrive/EcoCAR/training_runs/notebook_logs/`

Each experiment tar should contain:

- `best.pt`
- `last.pt`
- `metrics.json`
- `config_snapshot.yaml`

## Metric meaning

- `train/total`: full joint loss actually backpropagated.
- `train/det/total`: vehicle detection branch loss.
- `train/lane/total`: lane curve branch loss.
- `val/det_loss`: detection loss on validation batches.
- `val/lane_loss`: lane curve loss on validation batches.
- `val/lane_exist_acc`: slot-level lane existence accuracy. It can improve even if lane geometry is still poor.
- `val/lane_point_mae`: average normalized point error for visible GT lane points. Lower is better.
- `lane/line_iou`: logged value is the LineIoU loss component, so lower is better.
- `train/gate/*`: only appears for GCA experiments. Gate means near 0.5 and low saturation mean the gates are not collapsed.
- `grad_cos`: gradient cosine between detection loss and lane loss on shared features. Negative values suggest task conflict.

## Current Exp01/Exp02 interpretation

Exp02 with GCA is better than Exp01 on detection loss and slightly better on lane loss, but lane point geometry has not clearly improved yet. That means the GCA branch is helping conflict control, but the lane curve target/head still needs deeper validation before moving to heavier Exp3 runs.
