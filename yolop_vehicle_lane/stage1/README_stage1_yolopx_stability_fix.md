# Stage1 YOLOPX Stability Fix

This package keeps the Stage1 YOLOPX vehicle + lane baseline, but changes the RTX PRO 6000 profile from a speed/aggressive setting to a safer training setting.

## Why this was changed

The uploaded training log shows a late-run instability pattern: epoch 68 starts with stable loss around 0.12, then the loss rises sharply around iteration 1520 and validation after epoch 69 collapses to almost zero detection and lane metrics. This is not a data-loading speed issue, because the logged data time is essentially 0.000 s after warmup.

## Main changes

1. `stage1/configs/yolopx_vehicle_lane_baseline.yaml`
   - `LR0: 0.001` -> `0.0003`
   - `LRF: 0.1` -> `0.05`
   - `WARMUP_EPOCHS: 5.0` -> `8.0`
   - `GRAD_CLIP_NORM: 0.0` -> `5.0`
   - `MIXUP: true` -> `false`
   - milder geometry/color augmentation
   - `WORKERS: 8` -> `4`

2. `lib/core/function.py`
   - skips non-finite loss batches instead of stepping the optimizer
   - clips gradients when `TRAIN.GRAD_CLIP_NORM > 0`
   - logs the current LR in each training line

3. `stage1/notebooks/02_train_yolopx_vehicle_lane_baseline.ipynb`
   - turns off persistent DataLoader workers for Colab stability
   - restores optimizer/scheduler/scaler state when resuming
   - keeps `best_joint` when resuming
   - saves `latest_train.pth` before validation
   - only keeps `latest.pth` as a safe post-validation checkpoint
   - detects validation collapse, saves `unstable_epoch_XXXX.pth`, restores `best.pth`, and stops training

## Recommended clean restart

Run these before starting the notebook again if you want a true fresh run:

```bash
rm -rf /content/drive/MyDrive/EcoCAR/yolop_vehicle_lane/stage1/checkpoints/yolopx
rm -rf /content/drive/MyDrive/EcoCAR/yolop_vehicle_lane/stage1/metrics/yolopx
rm -rf /content/drive/MyDrive/EcoCAR/yolop_vehicle_lane/stage1/tb_logs/yolopx
```

Then run:

```bash
cd /content/drive/MyDrive/EcoCAR/yolop_vehicle_lane
jupyter notebook stage1/notebooks/02_train_yolopx_vehicle_lane_baseline.ipynb
```

Keep `BATCH_OVERRIDE = None` first. The YAML batch is 32. Increase batch only after 5-10 stable epochs.
