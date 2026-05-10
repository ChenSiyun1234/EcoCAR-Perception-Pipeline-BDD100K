# Stage 2 joint vehicle + lane curve fusion experiments

This Stage 2 refactor removes the earlier lane-only/head-only prototype logic. Every active experiment now follows the same complete joint-model path:

`image -> shared backbone/neck -> P3/P4/P5 -> DETR-style vehicle detector + CLRKD-style lane curve head -> joint loss -> shared-feature backpropagation`

## Notebook order

1. `stage2_notebook_00_prepare_joint_dataset.ipynb`  
   Builds the Drive-persisted tar archive used by all later Colab notebooks.

2. `stage2_notebook_01_exp2a_rmt_shared_joint.ipynb`  
   Exp2A: RMT-style HGNet/AIFI/FPN-PAN shared backbone/neck, no GCA, DETR vehicle detector, CLRKD-style lane prior head.

3. `stage2_notebook_02_exp2b_rmt_gca_joint.ipynb`  
   Exp2B: same shared encoder, but detection and lane features go through task adapters plus GCA gates before their heads.

4. `stage2_notebook_03_exp2c_rmt_gca_clrkd_kd_joint.ipynb`  
   Exp2C: same as Exp2B with CLRKD distillation support. Teacher-loading is activated by pointing the config to a frozen CLRKD checkpoint.

5. `stage2_notebook_04_exp3a_yolo26_inspired_joint.ipynb`  
   Exp3A: custom official-YOLO26-inspired CNN backbone/neck baseline. This is not a YOLO26 black-box hook.

6. `stage2_notebook_05_exp3b_yolo26_inspired_global_joint.ipynb`  
   Exp3B: Exp3A plus an AIFI-like P5 global context block.

7. `stage2_notebook_06_exp3c_yolo26_inspired_gca_joint.ipynb`  
   Exp3C: Exp3B plus RMT-style GCA task separation for detection and lane features.

8. `stage2_notebook_07_loss_weight_sweep_and_stability.ipynb`  
   Seed and lambda sweep for stability testing.

9. `stage2_notebook_08_joint_eval_visualization_and_profile.ipynb`  
   Unified result review and video/GPU profiling.

## Main code files changed

- `stage2/fusion/backbones.py`: RMT-style backbone/neck, AIFI, FPN/PAN, GCA adapters. The old real-YOLO26 wrapper remains only as legacy support and is not used by the active Exp3 configs.
- `stage2/fusion/yolo26_inspired.py`: new custom official-YOLO26-inspired joint backbone/neck: efficient CNN stages, CSP/C2f-style aggregation, PAN/FPN neck, P5 global context, lane-preserving P3 refinement, optional GCA.
- `stage2/fusion/lane_head.py`: CLRKD-style fixed-prior lane head with prior selection, start-y/start-x/theta/length regression, row-wise lane curves, and auxiliary mask output.
- `stage2/fusion/detection.py`: DETR-style query detector and Hungarian matching detection loss.
- `stage2/fusion/losses.py`: CLRKD-style lane loss terms: focal existence, point regression, xytl regression, LineIoU, mask auxiliary loss, smoothness, optional distillation.
- `stage2/scripts/train_joint_model_experiment.py`: staged joint training, gradient-norm lane loss calibration, gate regularization, phase freezing, Colab tar output.
- `stage2/scripts/smoke_test_joint_models.py`: fast shape/loss/gradient sanity check for Exp2 and Exp3.

## Colab rule

Every notebook extracts Drive tar/zip data into `/content`, runs training/evaluation from local SSD, and saves results back to Drive as tar files. No notebook assumes `/content` survives across notebook sessions.
