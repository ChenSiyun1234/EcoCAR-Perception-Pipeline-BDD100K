# Stage 2 Training Logging Fix - 2026-05-05

This patch fixes the issue where Stage 2 notebook 01 appeared to train without printing useful progress.

## What changed

1. `stage2/scripts/train_joint_model_experiment.py`
   - Added unbuffered-style logging through a `log()` helper using `flush=True`.
   - Added a preflight summary before extraction/training:
     - run name
     - config path
     - dataset tar path
     - local dataset path
     - work/output paths
     - backbone type
     - detection/lane head type
     - epoch, batch size, print interval, train/val limits
     - loss mode and lambda settings
   - Added dataset and dataloader summaries.
   - Added model parameter count and feature-channel summary.
   - Added first-step logging for every epoch.
   - Added batch progress logging every `print_every` steps.
   - Added `--print-every` CLI override.
   - Batch log now includes:
     - epoch
     - step / total steps
     - total loss
     - detection loss
     - lane loss
     - runtime lambda_lane
     - gradient cosine

2. Stage 2 notebooks 01-07
   - Training subprocesses now use `python -u` to avoid buffered notebook output.
   - Training commands pass `--print-every 10`.
   - Printed command lines are flushed immediately.

3. Stage 2 configs
   - `print_every` is set to `10` for the active Exp01-Exp06 configs.

## Expected notebook behavior

Notebook 01 should now print output similar to:

```text
==== Stage 2 Joint Training Preflight ====
run_name=exp01_rmt_shared_joint
config=stage2/configs/exp01_rmt_shared_joint.yaml
curve_tar=/content/drive/MyDrive/EcoCAR/datasets/bdd100k_clrkd_curve.tar
curve_root=/content/bdd100k_clrkd_curve
backbone=exp2a_rmt_shared detection_head=detr lane_head=CLRKD-style curve head
epochs=10 batch_size=8 print_every=10 limit_train=0 limit_val=512
Extracting dataset tar: ...
device=cuda
gpu=...
[dataset] split=train samples=... nonempty_det_labels=...
[loader] train_batches=... val_batches=...
[model] class=JointPerceptionModel params=... trainable_initial=...
epoch=1 phase=head_warmup batches=...
epoch=1 step=1/... total=... det=... lane=... lambda_lane=... grad_cos=...
epoch=1 step=10/... total=... det=... lane=... lambda_lane=... grad_cos=...
```

If the notebook still shows no output, it is likely spending time inside tar extraction or first DataLoader batch creation. The preflight line before extraction should still appear immediately.
