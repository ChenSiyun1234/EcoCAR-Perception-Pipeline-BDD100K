# Stage 2 Exp2K — LineIoU regression cls target

## Why this patch exists

Exp2J short10 (NB15) was a **negative result**, finishing the cls-rescue cycle G→H→I→J. Per-epoch from the saved NB15 output:

| epoch | phase            | val/lane_exist_best_f1 | pos − neg | val/matched_line_iou |
|------:|------------------|-----------------------:|----------:|---------------------:|
| 1     | head_warmup      | 0.186                  | +0.015    | 0.343                |
| 2     | adapter_warmup   | 0.091                  | +0.003    | 0.381                |
| 4     | full_finetune    | 0.056                  | +0.001    | 0.336                |
| 10    | full_finetune    | **0.053**              | **+0.000**| **0.362**            |

Versus Exp2I e10 (best_f1=0.057, pos−neg=+0.001, matched_iou=0.391), Exp2J was **worse on every metric**. The separate cls pathway not only failed to fix cls (best_f1 essentially identical), it hurt geometry (matched_iou 0.391 → 0.362). Critically, it never reproduced the Exp2I-epoch-2 spike of `best_f1 = 0.535` — Exp2J's epoch 1 peaked at only 0.186.

`train/lane/cls_pos` was frozen at ~0.140 across all 10 epochs, which works out to `p_pos = exp(-0.140/0.25) = 0.572` — exactly the `pos_score_mean` we observe. **The cls head literally outputs the same value for everything**. This pattern repeats across G/H/I/J: focal vs ASL, OHEM vs all-priors, single path vs separate path — none of it broke the equilibrium.

### The real bottleneck is the task formulation

Across four architectures the cls head consistently converges to "predict ~0.5 for everyone." The math is consistent with the model genuinely seeing contradictory supervision: with 192 priors and ~5 GT lanes per image, dynamic-k matching makes the same prior positive in some batches and negative in others depending on which competing priors win the IoU contest. The cls head's stable solution under contradictory binary signals is the uniform predictor.

**No feature/loss/optimizer fix can stabilize a target that flips on its own.** Exp2J's Exp2I-e2 spike was a one-shot optimizer artifact, not a fundamental signal — and even that artifact was below Exp2J's epoch 1 (0.186 < Exp2I e2 0.535) so it isn't reproducible from an architectural fix.

## What changed

Exp2K removes the matching dependency from the cls supervision entirely.

1. [stage2/fusion/losses.py](stage2/fusion/losses.py):
   - New `FusionLossConfig.cls_target_type: str = 'matched_existence'` (default keeps Exp2G/H/I/J behavior).
   - Setting `cls_target_type: 'lineiou_regression'` switches to a continuous target. New helper `_compute_lineiou_target(coord_pred.detach(), points_gt, vis, radius)` returns `(B, Q)` per-prior target, where each prior's target is the max LineIoU between its current predicted curve (detached, no feedback through the target) and any valid GT lane in the same image. Loss becomes plain `BCE_with_logits(cls_logit, iou_target)` over all priors, no OHEM, no focal/ASL, no balanced averaging — the continuous balanced target makes those moot.

2. [stage2/configs/exp11_rmt_gca_clrkd_iou_regression_joint.yaml](stage2/configs/exp11_rmt_gca_clrkd_iou_regression_joint.yaml) — new config. Diff vs `exp10` (Exp2J):
   - `cls_target_type: lineiou_regression`.
   - `cls_separate_path: false` (Exp2J showed it hurts; confirmed lateral-or-worse on all metrics).
   - `cls_ohem_topk_per_pos: 0` (continuous target — class imbalance is gone).
   - `cls_balance_mode: mean` (regression branch ignores this anyway).
   - All other settings (RMT+GCA backbone, DETR det head, ROI gather + 3 stages + dynamic-k matching for *geometry only*, prior_embed_encoder dim 64, λ_min=0.5, w_iou=2.0) identical to Exp2J's recipe minus the separate-path and OHEM bits.

3. [stage2/notebooks/stage2_notebook_16_exp2k_clrkd_iou_regression_joint.ipynb](stage2/notebooks/stage2_notebook_16_exp2k_clrkd_iou_regression_joint.ipynb) — new notebook mirroring NB15. Smoke first, debug-mode default.

4. [stage2/notebooks/stage2_notebook_08_joint_eval_visualization_and_profile.ipynb](stage2/notebooks/stage2_notebook_08_joint_eval_visualization_and_profile.ipynb) — exp11 entries added to cells 3 (eval items), 5 (metrics list), 7 (video profile candidates, Exp2K preferred over Exp2J/I/H/G).

## What this experiment isolates

Following the user's "target one goal, remove confounds" directive: the goal is **lane scoring/ranking that produces a meaningful val/clrkd_style_f1**. Confounds being removed in this iteration:

- Drop OHEM (the matched-existence helper is gone with the binary target).
- Drop ASL (the matched-existence loss type is gone).
- Drop separate cls path (Exp2J disproved its value).
- Drop the binary matching-dependent target itself (the actual root cause).

What remains (everything else identical to Exp2I/J recipe so the comparison is controlled):
- ROI gather + multi-scale + 3-stage refinement (geometry, healthy from Exp2G).
- Dynamic-k matching for geometry losses only (point_mae, reg, line_iou, smooth, mask).
- Prior embed encoder Linear(3, 64) into cls input.
- RMT + GCA backbone, DETR det head, λ_min=0.5, w_iou=2.0.

## Pass criteria

After 10 short10 epochs:

- **The persistence test**: `val/lane_exist_best_f1 ≥ 0.30 by epoch 5 and ≥ 0.50 by epoch 10`. The four-experiment cluster G/H/I/J topped out at 0.083; we need at least a 5× jump to declare the matching-instability hypothesis confirmed.
- `val/lane_exist_pos_score_mean − val/lane_exist_neg_score_mean ≥ 0.20` at epoch 10. The score now means "predicted IoU"; matched priors carry high IoU targets so their predictions should sit far above unmatched priors.
- `val/lane/clrkd_style_f1 ≥ 0.05` (vs G/H/I/J cluster at ~0.018-0.022). This is the metric closest to CLRKDNet's reported numbers, and it's the project's actual end-task target.
- `pred_lanes / batch < 500` (was stuck at ~1535 across G/H/I/J because both pos and neg sat at sigmoid(logit)=0.57, all above eval threshold 0.3).
- **Geometry holds**: `val/lane_point_mae ≤ 0.34` and `val/matched_line_iou ≥ 0.40` at epoch 10 (close to or matching Exp2G's 0.428).

## Failure criteria → next ablation

- best_f1 < 0.20 at epoch 10 → the IoU target is too sparse (most priors have IoU ~0). Fix: rescale target via `clamp(2*iou - 0.5, 0, 1)` to spread the distribution. Or compute target only on each prior's nearest GT (rather than max over all GTs).
- pred_lanes drops to ≈ 0 with recall collapse → IoU target too small for most priors → cls head learns to predict near-zero everywhere. Same rescaling fix as above.
- Geometry regresses (point_mae > 0.36) → loss balance changed because cls_loss now sums over *all priors* not just balanced 0.5+0.5. Pull `w_cls` from 4.0 down to 2.0.

## What comes after Exp2K

If the cls scoring works, the next step is **lane decode + NMS** (Priority 1 from the original architectural plan). With per-prior IoU scores from a working cls head, top-K decode becomes a simple sort + threshold + per-image NMS based on LineIoU between predicted curves. Then `val/clrkd_style_f1` measures lane-level F1 against GT lanes the way CLRKDNet does — the proper apples-to-apples comparison.

Other follow-ups still on the plate (separate from cls fix):

- **GCA gates frozen at 0.500 across all Exp2 runs** — `lambda_gate_reg=0.001` may be too small; gate inputs may be too similar at our scale.
- **Detection mAP50 ≈ 0.003** — independent issue (DETR head: 3 decoder layers, no denoising).
- **CLRKD knowledge distillation** — `loss.w_distill: 0.0` everywhere; the project's name promises this and it's still unfulfilled.

## Local smoke

`python -m compileall yolop_vehicle_lane/stage2/fusion` passes. The torch forward+backward smoke runs from inside NB16 cell 4 on Colab.

## Run order

1. Do not rerun NB00.
2. Open NB16, keep `DEBUG_MODE = True`, run smoke + debug.
3. If debug succeeds, set `DEBUG_MODE = False` and run short10.
4. Run NB08 to plot Exp2G / H / I / J / K side-by-side per epoch.
