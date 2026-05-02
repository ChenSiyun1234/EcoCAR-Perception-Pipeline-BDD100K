Do not delete old checkpoints after a late-stage collapse.

Recommended recovery:
1. Back up stage1/checkpoints/yolopx.
2. Use best.pth or best_joint.pth, not latest.pth.
3. In stage1/notebooks/02_train_yolopx_vehicle_lane_baseline.ipynb set:
   RESUME_CKPT_OVERRIDE = '/content/drive/MyDrive/EcoCAR/yolop_vehicle_lane/stage1/checkpoints/yolopx/best.pth'
   RESUME_WEIGHTS_ONLY = True
4. Continue training with the stable LR/augmentation settings.
