# Stage 2 Joint Fusion Experiments

This version removes the old Stage 2 lane-only notebook path from the main notebook sequence. The current Stage 2 notebooks are ordered as:

1. `00_prepare_joint_curve_detection_dataset_colab.ipynb`
   - Builds `bdd100k_clrkd_curve.tar` from Drive resources.
   - Saves images, CLRKD-style lane curve `.lines.txt`, auxiliary lane masks, and vehicle detection labels under the same tar.
   - Fails if the packed dataset has no images or no vehicle boxes.

2. `01_joint_exp01_rmt_backbone_neck_joint_colab.ipynb`
   - Complete joint model.
   - RMT-style backbone/neck shared by detection and lane branches.
   - DETR-lite vehicle set-prediction head + CLRKD-style curve lane head.

3. `02_joint_exp02_hybrid_rmt_clrkd_joint_colab.ipynb`
   - Complete joint model.
   - Shared RMT-style backbone/neck plus task adapters for detection and lane features.
   - Uses uncertainty weighting by default for multi-task loss balancing.

4. `03_joint_exp03_yolo26_backbone_neck_joint_colab.ipynb`
   - Complete joint model.
   - YOLO26-style ELAN/PAN backbone-neck with the same DETR-lite vehicle head and CLRKD-style lane head.

5. `04_joint_stability_sweep_colab.ipynb`
   - Repeats the three complete joint models over multiple seeds.
   - Intended to check stability before committing to full training.

6. `05_joint_eval_video_profile_colab.ipynb`
   - Loads a trained complete joint model and profiles video inference.
   - Saves frame-level latency, FPS, memory, predicted detection count, and predicted lane count.

## Loss design

Each Stage 2 joint experiment optimizes one forward pass with two task branches:

`L_total = L_det + lambda_lane * L_lane`

or, when enabled:

`L_total = uncertainty_weight(L_det, L_lane)`

Detection loss uses a DETR-style set prediction head:

`L_det = w_obj * BCE(objectness) + w_cls * BCE(class) + w_box * SmoothL1(box) + w_giou * (1 - GIoU)`

Lane loss uses the CLRKD-style fixed lane-slot curve representation:

`L_lane = w_cls * focal(existence) + w_reg * point regression + w_iou * line-IoU + w_mask * auxiliary mask BCE + w_smooth * curve smoothness`

The training script logs gradient cosine between detection and lane losses on the shared backbone. Negative cosine means the two tasks are fighting each other, which is why the hybrid adapter and uncertainty-weighted experiment exist.

## Colab storage rule

Every notebook assumes `/content` is temporary. The dataset and training outputs are always packed back to Drive:

- Dataset: `/content/drive/MyDrive/EcoCAR/datasets/bdd100k_clrkd_curve.tar`
- Runs: `/content/drive/MyDrive/EcoCAR/training_runs/*.tar`

No notebook assumes another notebook's local `/content` files still exist.
