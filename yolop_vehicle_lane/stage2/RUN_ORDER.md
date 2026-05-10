# Stage 2 run order after Exp2 refactor

Do not assume `/content` persists between notebooks. Each training notebook extracts `bdd100k_clrkd_curve.tar` from Drive into `/content` and writes its outputs back to Drive as a `.tar`.

## Required first step

0. `stage2_notebook_00_prepare_joint_dataset.ipynb`
   - Creates `/content/drive/MyDrive/EcoCAR/datasets/bdd100k_clrkd_curve.tar`.
   - Re-run this only if dataset labels/images/tar content changed.

## Already-run baselines

1. `stage2_notebook_01_exp2a_rmt_shared_joint.ipynb`
   - Exp2A: RMT shared P3/P4/P5 without GCA.

2. `stage2_notebook_02_exp2b_rmt_gca_joint.ipynb`
   - Exp2B: RMT shared P3/P4/P5 with detection/lane GCA.

## New focused Exp2 runs

3. `stage2_notebook_03_exp2d_rmt_gca_lane_detail_joint.ipynb`
   - Exp2D: adds a C2/stride-4 lane detail path before the CLRKD-style lane head.

4. `stage2_notebook_04_exp2e_rmt_gca_lane_matching_joint.ipynb`
   - Exp2E: keeps the Exp2B architecture but changes lane supervision from slot-wise loss to Hungarian-style lane matching.

5. `stage2_notebook_05_exp2f_rmt_gca_lane_detail_matching_joint.ipynb`
   - Exp2F: combines C2 lane detail path and Hungarian-style lane matching.

## Optional later runs

6. `stage2_notebook_06_exp2c_rmt_gca_clrkd_kd_joint.ipynb`
   - Optional KD run. Only use this when the CLRKD lane teacher checkpoint exists.

7. `stage2_notebook_07_loss_weight_sweep_and_stability.ipynb`
   - Run only after Exp2D/E/F identify the best architecture.

8. `stage2_notebook_08_joint_eval_visualization_and_profile.ipynb`
   - Unified evaluation/visualization/profile notebook.

Exp3 notebooks are now lower priority. Run them only after Exp2F gives a meaningful lane geometry improvement.
