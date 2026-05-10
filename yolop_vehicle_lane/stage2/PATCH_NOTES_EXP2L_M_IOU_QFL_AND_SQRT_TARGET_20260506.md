# Stage 2 Exp2L (QFL) & Exp2M (sqrt target) — IoU-regression cls rescue, two independent ablations

## Why this patch exists

Exp2K short10 (NB16) replaced the binary "is matched" cls target with continuous LineIoU regression and confirmed that approach removes the matching-instability that broke Exp2G/H/I/J. But Exp2K's plain BCE on the IoU target had its own pathology: with ~95 % of priors having target ≈ 0, the loss is dominated by "predict 0" pressure and every logit collapses toward 0. End of Exp2K (epoch 10):

| metric                 | value     | observation                                                       |
|------------------------|-----------|-------------------------------------------------------------------|
| pos_score_mean         | 0.042     | both pos and neg sigmoids stuck at the all-zeros attractor        |
| neg_score_mean         | 0.042     |                                                                   |
| pred_lanes / batch     | **0.0**   | model predicts 'no lane' everywhere at threshold 0.3              |
| fn_slots               | 42.0      | every GT lane is missed                                           |
| val/lane_exist_best_f1 | 0.032     |                                                                   |
| val/matched_line_iou   | **0.410** | geometry champion territory (Exp2G was 0.428)                     |
| val/lane_point_mae     | **0.325** | matches Exp2G's 0.324                                              |

The good news: Exp2K's clean recipe (no separate path / no OHEM / no ASL) recovered Exp2G-quality geometry. The bad news: scoring is unusable for top-K decode because all logits are equal.

Two independent surgical fixes, run as ablations of Exp2K (no changes to architecture or matching):

## Exp2L — Quality Focal Loss (RTMDet/GFL)

Replace BCE-on-IoU-target with QFL: `loss = |target − sigmoid(logit)|^γ · BCE`. The published fix for exactly this scenario (continuous quality target, sparse non-zero values).

- Easy correct cases (target ≈ 0, pred ≈ 0) get weight ≈ 0 → near-zero gradient. Bulk-pull-to-zero pressure is removed.
- Hard mismatches get the full BCE gradient. The few high-IoU priors finally pull their logits up.

Single config knob change vs Exp2K: `cls_loss_type: focal → qfl`, plus `qfl_gamma: 2.0`. Reference: Li et al. 2020 *Generalized Focal Loss* (NeurIPS).

## Exp2M — sqrt rescaling on the IoU target

Set `lineiou_target_pow: 0.5`. Target becomes `√(LineIoU)` instead of `LineIoU`:

| LineIoU | target (Exp2K) | target (Exp2M) |
|--------:|---------------:|---------------:|
| 0.04    | 0.04           | 0.20           |
| 0.16    | 0.16           | 0.40           |
| 0.36    | 0.36           | 0.60           |
| 0.81    | 0.81           | 0.90           |

Effect: priors with small but non-zero IoU now carry a meaningful supervision signal (0.20 instead of 0.04). The "predict 0 for everyone" attractor is broken because more priors have non-trivial targets. Heuristic but cheap, clean fallback if QFL fails.

Single config knob change vs Exp2K: `lineiou_target_pow: 0.5`. cls_loss_type stays `bce`.

Both ablations are **independent**: they touch different knobs (loss function vs target distribution). Run NB17 (Exp2L) and NB18 (Exp2M) in any order or both in parallel; nothing else has to change between them.

## Files changed

1. [stage2/fusion/losses.py](stage2/fusion/losses.py):
   - Added `_quality_focal_loss(logit, target, gamma, reduction)` helper.
   - Added `qfl_gamma: float = 2.0` and `lineiou_target_pow: float = 1.0` fields to `FusionLossConfig`.
   - `_compute_lineiou_target` now accepts `target_pow` and applies `target.clamp(0).pow(target_pow)` after the max-IoU computation when `target_pow != 1.0`.
   - In `_forward_single_stage`, when `cls_target_type='lineiou_regression'`, dispatch on `cls_loss_type`:
     - `'qfl'` → `_quality_focal_loss(logit, target, gamma=qfl_gamma)`.
     - anything else (including `'bce'` and the back-compat `'focal'` / `'asl'`) → plain BCE-with-logits on the continuous target. ASL/focal modulators are never applied to a continuous target — the dispatch is intentionally narrow.

2. [stage2/configs/exp12_rmt_gca_clrkd_iou_qfl_joint.yaml](stage2/configs/exp12_rmt_gca_clrkd_iou_qfl_joint.yaml) — Exp2L config.

3. [stage2/configs/exp13_rmt_gca_clrkd_iou_sqrt_target_joint.yaml](stage2/configs/exp13_rmt_gca_clrkd_iou_sqrt_target_joint.yaml) — Exp2M config.

4. [stage2/notebooks/stage2_notebook_17_exp2l_clrkd_iou_qfl_joint.ipynb](stage2/notebooks/stage2_notebook_17_exp2l_clrkd_iou_qfl_joint.ipynb) — NB17 (Exp2L).

5. [stage2/notebooks/stage2_notebook_18_exp2m_clrkd_iou_sqrt_target_joint.ipynb](stage2/notebooks/stage2_notebook_18_exp2m_clrkd_iou_sqrt_target_joint.ipynb) — NB18 (Exp2M).

6. [stage2/notebooks/stage2_notebook_08_joint_eval_visualization_and_profile.ipynb](stage2/notebooks/stage2_notebook_08_joint_eval_visualization_and_profile.ipynb) — exp12 + exp13 entries added to cells 3, 5, 7.

## Local smoke

`python -m compileall yolop_vehicle_lane/stage2/fusion` passes. The torch forward+backward smokes run from inside NB17/NB18 cell 3 on Colab.

## Pass criteria (same for both)

After 10 short10 epochs:

- `pred_lanes / batch` > 100 (Exp2K was 0). The model is producing *some* predictions.
- `pos_score_mean − neg_score_mean ≥ 0.10`. The score distributions separate.
- `val/lane_exist_best_f1 ≥ 0.30` at epoch 10.
- **Geometry holds**: `val/lane_point_mae ≤ 0.34` and `val/matched_line_iou ≥ 0.40`. Neither fix should disrupt geometry; if it does, w_cls is too high.
- `val/lane/clrkd_style_f1` rises noticeably above the ~0.02 floor — this is the metric that compares directly to CLRKDNet on lane-line F1.

## Failure criteria → next ablations

- **Exp2L fails (pred_lanes still ≈ 0)**: QFL gamma=2 not aggressive enough. Try gamma=4 or 6. Or fall back to Exp2M.
- **Exp2M fails (pred_lanes still ≈ 0)**: sqrt rescaling not aggressive enough. Try `lineiou_target_pow: 0.25` (4-th root). Or fall back to Exp2L.
- **Both fail**: the ROI sample positions themselves are insufficient for cls discrimination. Need stop-gradient from cls into curve params and/or higher prior_embed_encoder_dim.
- **One passes**: that becomes the parent for **Exp2N — lane decode + NMS + proper lane-line F1**. With a working scoring head, we can finally produce ranked top-K lane outputs and compute the metric that compares to CLRKDNet's published F1 on CULane / TuSimple.

## Run order

These are independent ablations. Either or both can be run.

1. Open NB17 (Exp2L), keep `DEBUG_MODE = True`, run smoke + debug.
2. If debug succeeds, set `DEBUG_MODE = False` and run short10.
3. Independently: open NB18 (Exp2M), same pattern.
4. After short10s finish, run NB08 to plot Exp2K vs Exp2L vs Exp2M side-by-side.
5. The winner becomes the parent for Exp2N (proper lane decode + NMS + lane-F1).
