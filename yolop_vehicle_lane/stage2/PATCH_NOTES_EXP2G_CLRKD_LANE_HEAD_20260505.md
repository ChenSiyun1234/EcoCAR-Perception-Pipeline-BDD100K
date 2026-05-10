# Stage 2 Exp2G — CLRKD-style lane head patch notes

## Why this patch exists

`stage2_trend_summary.csv` shows that Exp2E (`exp04_rmt_gca_lane_matching_joint`) has `val_lane_point_mae` flat across all 10 epochs (epoch 1: 0.4060 → epoch 10: 0.4011, change 0.5 %). The agent A audit of saved notebook outputs showed `val/matched_line_iou` at epoch 10 was ~0.08, and `val/lane_exist_*` had precision ≈ recall ≈ accuracy ≈ 0.55 — i.e. no real positive/negative score separation. The lane head was not actually learning beyond what target-slot matching (added in the previous patch) already gave for free.

Reading `stage2/fusion/lane_head.py` showed the cause: in `CurveLaneHead.forward` (around line 138-141) every one of the 192 priors received the same global average pool of the merged feature map. Differentiation came only through a 3-d `(start_y, start_x, theta)` prior embedding projected to `embed_dim`. The optional `use_roi_gather` flag fired *after* an initial prediction made from this shared context, used nearest-neighbor index sampling on a single collapsed feature map (no gradient through positions, no multi-scale view), and did one refinement pass. None of the components that make CLRNet's lane head work (per-prior multi-scale bilinear sampling, cross-attention to the feature map, multi-stage refinement with auxiliary supervision) were present.

This patch keeps `CurveLaneHead` untouched so Exp2A/B/D/E remain reproducible and adds a new `CLRKDLaneHead` plus an `Exp2G` experiment. The reference design is `external_repos/CLRNet/clrnet/models/utils/roi_gather.py` lines 33-136 and `external_repos/CLRKDNet-master/clrkd/models/heads/clr_head.py`.

## What changed

1. `stage2/fusion/lane_head.py` — added `CLRKDLaneHead` at the end of the file. Same `_init_prior_embeddings` left/bottom/right CLRKDNet-style scheme as `CurveLaneHead`. New components:
   - `_grid_sample` uses `F.grid_sample(mode='bilinear', padding_mode='border', align_corners=False)` so positions get gradients.
   - `_PerScaleROIBlock` wraps a Conv2d with kernel `(1, 9)` padding `(0, 4)` so the conv runs over the sample-points axis while leaving the priors axis unchanged. (Matches CLRNet's intent; the layout is `(B, C, P, S)` instead of CLRNet's `(B*P, C, S, 1)`.)
   - `_CrossAttention` uses grouped Conv1d on `(B, P, embed_dim)` with `groups=num_priors` so each prior queries the feature map independently. Key/value are interpolated to a `(10, 25)` grid before matmul.
   - Three refinement layers by default. Each layer outputs `cls_logits`, `(start_x, start_y, theta, length)`, row-wise offsets, and a curve. Stage 0 starts from each prior's own initial geometry (zero offset), so each prior has a distinct local feature from epoch 0.
   - Intermediate stages are returned as `aux_stage_outputs`. The final stage feeds the existing keys (`cls_logits`, `coord_pred`, `lane_param`, `mask_logit`, etc.) so downstream code is unchanged.

2. `stage2/fusion/losses.py` — `FusionLaneLoss.forward` now detects `aux_stage_outputs` and adds the same lane loss on each intermediate stage with weight `aux_stage_loss_weight * (i + 1) / (num_stages)`. Mask and distillation losses are not duplicated. The internals of the original loss were extracted into `_forward_single_stage`, so behaviour for `CurveLaneHead` (which does not emit `aux_stage_outputs`) is identical to before. Also fixed a latent name bug in `match_targets` (`coord_pred.new_zeros((bsz, lanes, 4))` referenced an undefined `lanes`; corrected to `pred_lanes`).

3. `stage2/fusion/experiment_factory.py` — read `model.lane_head.type` from the YAML (defaults to `'curve'` for back-compat). When set to `'clrkd'`, `'clrkd_lane_head'`, `'clrkd_roi_gather'`, or `'exp2g'`, the factory builds `CLRKDLaneHead` with its own kwargs (`sample_points`, `roi_refine_layers`, `roi_mid_channels`, `cross_attn_resize`). All other code paths continue to build `CurveLaneHead` unchanged.

4. `stage2/configs/exp07_rmt_gca_clrkd_lane_head_joint.yaml` — new config. Backbone `exp2b_rmt_gca` (RMT + GCA, same as Exp2B/E so the comparison isolates the head). DETR detection head, dynamic-k matching, focal cls, geometry warmup 3 epochs (down from Exp2E's 5; the new head is meant to converge faster), `aux_stage_loss_weight: 1.0`, `roi_refine_layers: 3`, `sample_points: 36`, `roi_mid_channels: 48`.

5. `stage2/notebooks/stage2_notebook_12_exp2g_clrkd_lane_head_joint.ipynb` — new notebook mirroring NB04. Cells: drive mount + repo setup, smoke test against the new config, training cell (`DEBUG_MODE=True` by default), markdown describing what to watch.

6. `stage2/notebooks/stage2_notebook_08_joint_eval_visualization_and_profile.ipynb`:
   - Cell 3 (eval loop) now wraps each per-experiment evaluation in `try/except` so a single failure (missing tar, mismatched checkpoint) no longer blocks the trend plot or video cell. The historical `CalledProcessError` was caused by a missing `evaluate_joint_model.py` at the time NB08 first ran (added later in the May 5 patch); the file exists now and the wrap prevents future single-point failures.
   - Cell 3 also includes Exp2G (exp07) candidates (short10, debug, base).
   - Cell 5 (plot trend cell) appends the Exp2G short10 metrics path.
   - Cell 7 (video profile) prefers the Exp2G run tar if present and falls back to Exp2E.

7. `stage2/scripts/evaluate_joint_model.py` — added `--list-only` flag. Resolves and prints `run_tar`, `config`, `curve_tar`, `curve_root`, and `eval_root` paths without running the eval. Useful to debug NB08 path issues without needing a full extraction cycle.

8. `stage1/notebooks/02_train_yolopx_vehicle_lane_baseline.ipynb` — added `_persist_per_epoch_metrics` helper to cell 10. After every validation, appends one JSON row (`completed_epoch`, `mAP50`, `mAP`, `ll_iou`, `ll_acc`, `joint_score`, `val_loss`) to `stage1/metrics/yolopx/per_epoch.json`. Existing rows for the same completed epoch are overwritten on resume so the file stays clean. This exposes Stage 1 lane metrics per epoch, which were previously only saved at the final-eval JSON.

## What this is meant to fix

- The flat `val/lane_point_mae` plateau at ~0.40 across Exp2E/Exp2F. Each of the 192 priors now sees a feature gathered from its own predicted curve, at multiple scales, with bilinear sampling that backpropagates into the curve geometry.
- The poorly-calibrated lane existence score (`precision ≈ recall ≈ acc ≈ 0.55`). The cross-attention block lets each prior's score depend on its own curve evidence rather than only on the shared global pool.
- The opaque `val/lane/line_iou` ~0.89 metric that hid the real geometry signal. Watch `val/matched_line_iou` (only positive priors) instead.

## Run order after this patch

1. Do not rerun Notebook 00. The dataset tar `/content/drive/MyDrive/EcoCAR/datasets/bdd100k_clrkd_curve.tar` is unchanged.
2. Open Stage 2 Notebook 12 with `DEBUG_MODE = True`. Run the smoke cell first; it must print `OK exp07_*.yaml` with shapes before training. If it fails, do not proceed.
3. Run the debug training cell (2 epochs, 512 train / 256 val).
4. If debug succeeds, set `DEBUG_MODE = False` and run the short10 cell (10 epochs).
5. Run Notebook 08 to plot per-epoch trends and produce the box+lane preview video.
6. If Exp2G clearly improves `val/lane_point_mae`, `val/matched_line_iou`, and `val/lane_exist_best_f1`, retire Exp2E/Exp2F and continue toward Priority 5 (DETR decoder upgrades) and Priority 6 (Exp3 YOLO26-inspired).

## Local smoke test

`python -m compileall stage2/fusion stage2/scripts/{smoke_test_joint_models,train_joint_model_experiment,evaluate_joint_model}.py` passes with no errors after this patch. The actual torch forward+backward smoke test runs from inside Notebook 12 cell 4 (`stage2/scripts/smoke_test_joint_models.py exp07_*.yaml`) on Colab; no torch is installed in the local Windows environment.

## What proves the patch worked

After 10 short10 epochs of Exp2G:

- `val/lane_point_mae` strictly decreases by ≥ 15 % from epoch 1 to epoch 10 and ends below 0.34 (current Exp2E/F plateau is ≈ 0.40).
- `val/matched_line_iou` rises from < 0.10 to ≥ 0.20.
- `val/lane_exist_best_f1` ≥ 0.75 with both precision and recall ≥ 0.65.
- `val/lane_exist_pos_score_mean − val/lane_exist_neg_score_mean ≥ 0.20`.
- `val/det/metric_map50` does not regress more than 0.02 from Exp2B.
- `train/lane/aux0_total` and `train/lane/aux1_total` (intermediate refinement stages) decrease together with `train/lane/total` rather than plateauing.

## What would mean another approach is needed

- `val/lane_point_mae` still plateaus and `val/matched_line_iou < 0.12` after 10 epochs → the issue is upstream of the head (geometry parameterization or anchor coverage). Try direct row-anchor regression, or revisit the prior init scheme (denser left/right priors, more theta variants).
- Existence F1 stays below 0.65 despite point_mae improving → focal-loss imbalance. Try asymmetric focal (ASL) or hard-negative mining over unmatched priors.
- Detection mAP50 regresses sharply → GCA gates saturate under the heavier lane branch. Add gate-saturation regularization (`loss.lambda_gate_reg`) or reduce lane branch LR.
