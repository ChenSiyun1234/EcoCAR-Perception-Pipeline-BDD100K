# Stage 2 Exp2H — ASL classification rescue patch notes

## Why this patch exists

Exp2G short10 (NB12, run 2026-05-05) verified the geometry hypothesis from the previous patch. Per-epoch trends from the saved notebook output:

| epoch | val/lane_point_mae | val/matched_line_iou | val/lane_exist_best_f1 | pos_score − neg_score | val/det/metric_map50 |
|------:|-------------------:|---------------------:|-----------------------:|----------------------:|---------------------:|
| 1     | 0.3389             | 0.3836               | 0.168                  | +0.042                | 0.0018               |
| 5     | 0.3313             | 0.3660               | 0.120                  | +0.013                | 0.0028               |
| 10    | **0.3244**         | **0.4285**           | **0.075**              | **+0.004**            | 0.0028               |

Compared to the Exp2E plateau (epoch 10: point_mae 0.401, matched_line_iou ~0.08), Exp2G is a clear win on geometry: point_mae −19 % with strict monotonic decrease, matched_line_iou ~5 ×. The CLRKDLaneHead architectural fix is confirmed.

But Exp2G also exposed three previously-hidden problems:

1. **Lane existence classification is stuck at the focal-uniform-predictor equilibrium.** `val/lane/cls_pos` and `val/lane/cls_neg` froze at ~0.07-0.09 (the math: focal at p=0.5, α=0.35, γ=2 gives ≈ 0.06 for negatives and ≈ 0.08 for positives — exactly what was observed). `pos_score − neg_score` collapsed from +0.042 at epoch 1 to +0.004 at epoch 10. `pred_lanes / batch ≈ 1535 ≈ 192 × 8` — every prior is predicted positive. The standard focal modulator has near-flat gradient at p=0.5 so the dominant geometry loss carried the per-prior representation while cls flatlined.
2. **Detection mAP50 stuck at ~0.003** across all 10 epochs. Pre-existing issue, not an Exp2G regression. Tracked as a separate follow-up.
3. **GCA gates frozen at 0.5** (gate/det_mean=0.499, gate/lane_mean=0.500 across all epochs) and **lambda_lane pinned at the floor 0.2** (grad_norm_calibration says lane gradient norm is ≥ 5 × det gradient norm). Combined with `backbone_lr_mult=0.1` the backbone barely updated.

This patch targets problems 1 and 3.

## What changed

1. [stage2/fusion/losses.py](stage2/fusion/losses.py) — added `_binary_asl_loss` (Asymmetric Focal, Ben-Baruch et al. 2020) and a `_binary_cls_raw` dispatch helper. The new fields on `FusionLossConfig`:
   - `cls_loss_type: 'focal' | 'asl'` (defaults to `'focal'` for back-compat).
   - `asl_gamma_pos`, `asl_gamma_neg`, `asl_clip` (defaults: 0.0, 4.0, 0.05).
   - `gamma_pos = 0` removes the flat-gradient zone for positives so the loss keeps moving when `p_pos` sits at 0.5. `gamma_neg = 4` amplifies suppression of confident negatives. `clip = 0.05` shifts negative probabilities by a small probability margin (Ben-Baruch's "probability shifting"), which lets the model ignore extremely-easy negatives.
2. [stage2/fusion/lane_head.py](stage2/fusion/lane_head.py) — added `cls_uses_prior_embedding` kwarg to `CLRKDLaneHead`. When True, the 3-d clamped prior embedding `(start_y, start_x, theta)` is concatenated with `per_lane` before going into `cls_head`. This breaks the symmetry that geometry-only losses imposed on `per_lane`: every matched prior learned a representation of "what lane is here", which is similar across priors that happen to find the same nearby lane, so the cls head had no per-prior discriminator. Concatenating the prior embedding gives cls a non-learned positional feature it can use.
3. [stage2/fusion/experiment_factory.py](stage2/fusion/experiment_factory.py) — pass the new kwarg through and accept `'exp2h'` as a `lane_head.type` alias.
4. [stage2/configs/exp08_rmt_gca_clrkd_asl_cls_rescue_joint.yaml](stage2/configs/exp08_rmt_gca_clrkd_asl_cls_rescue_joint.yaml) — new config. Diff vs `exp07`:
   - `cls_loss_type: asl`, `asl_gamma_pos: 0.0`, `asl_gamma_neg: 4.0`, `asl_clip: 0.05`.
   - `cls_uses_prior_embedding: true`.
   - `w_cls: 2.0 → 6.0` and `w_iou: 2.0 → 1.0` (geometry already healthy at iou=2.0; give cls more relative weight).
   - `lambda_min: 0.2 → 0.5` so `λ_lane` does not collapse to the floor.
   - Everything else (3 refinement layers, 36 sample points, dynamic-k matching, geometry warmup 3 epochs, RMT+GCA backbone) is identical to Exp2G.
5. [stage2/notebooks/stage2_notebook_13_exp2h_clrkd_asl_cls_rescue_joint.ipynb](stage2/notebooks/stage2_notebook_13_exp2h_clrkd_asl_cls_rescue_joint.ipynb) — new notebook mirroring NB12. Smoke cell first, debug-mode default. Markdown documents what to watch and explicit pass / fail signals.
6. [stage2/notebooks/stage2_notebook_08_joint_eval_visualization_and_profile.ipynb](stage2/notebooks/stage2_notebook_08_joint_eval_visualization_and_profile.ipynb) — Exp2H entries added to:
   - Cell 3 `EVAL_ITEMS` (eval loop): exp08 short10 / debug / base candidates.
   - Cell 5 metrics list: `exp08_*_short10_metrics.json`.
   - Cell 7 video profile candidates: Exp2H tar preferred over Exp2G when present.

## Local smoke

`python -m compileall stage2/fusion` passes. The torch forward+backward smoke runs from inside NB13 cell 4 on Colab (`smoke_test_joint_models.py exp08_*.yaml`).

## Pass criteria

After 10 short10 epochs of Exp2H:

- `val/lane_exist_best_f1 ≥ 0.65` at epoch 10 (Exp2G e10 was 0.075).
- `val/lane_exist_pos_score_mean − val/lane_exist_neg_score_mean ≥ 0.15` at epoch 10 (Exp2G e10 was 0.004 — at least 30 × increase).
- `val/lane/cls_pos` strictly below `val/lane/cls_neg` and both decreasing over epochs (Exp2G had them flat at ~0.07-0.09).
- `pred_lanes / batch` drops from ≈ 192 (Exp2G all-positive) toward ≈ 5-10 (real predictions).
- **No geometry regression**: `val/lane_point_mae` ≤ 0.34 at epoch 10, `val/matched_line_iou` ≥ 0.30. (Exp2G hit 0.3244 / 0.4285. Exp2H must hold these.)
- `train/mtl/lambda_lane_runtime` no longer pinned at 0.2 (lambda_min raised to 0.5).

## Failure criteria → next ablation

- `val/lane_exist_best_f1 < 0.20` at epoch 10 → ASL is not enough; add hard-negative mining (top-k hardest unmatched priors per batch) or supervise existence through the auxiliary mask logits. Try `gamma_neg = 6` or add `OHEMSampler`.
- Geometry regresses (`point_mae > 0.36`) → `w_cls=6 / w_iou=1` weight shift was too aggressive; pull back to `w_cls=4 / w_iou=1.5`.
- `pred_lanes / batch ≈ 0` with recall collapse → ASL `gamma_neg = 4` too aggressive; reduce to 2 or 3.

## Independent follow-up: detection mAP50 ≈ 0.003

This is **not addressed by Exp2H** but flagged here for visibility. NB12 short10 shows `val/det/metric_map50` in the 0.0018 → 0.0045 band across all 10 epochs, with `val/det_loss` only mildly decreasing (2.21 → 2.09). The DETR head has 100 queries, 3 decoder layers, no denoising, and the metric adapter computes mAP50 only (not mAP50-95). Hypotheses to investigate independently:

- Decoder layers are too few (3) for set-prediction matching to converge in 10 epochs at this batch size. RT-DETR uses 6 + denoising queries.
- The adapter at `stage2/metrics/original_metric_adapters.py` may be IoU-thresholding incorrectly against the dataset's box format.
- `val_batches: 40` × `batch_size: 8` = 320 images is small but should still resolve mAP50 above noise; if it doesn't, the problem is in the head or the adapter, not sample size.

Prefix for the follow-up: `EXP_DET_MAP50_DEBUG`. Touchpoints: `stage2/fusion/detection.py`, `stage2/metrics/original_metric_adapters.py`. Run plan: small ablation enabling `dn_loss_weight` > 0 + 6 decoder layers + 8.5K val examples.

## Run order

1. Do not rerun NB00. The dataset tar is unchanged.
2. Open NB13, keep `DEBUG_MODE = True`, run smoke + debug.
3. If debug succeeds, set `DEBUG_MODE = False` and run short10.
4. Run NB08 to plot Exp2G vs Exp2H side-by-side and (if a video is present) generate the box+lane preview.
5. If Exp2H meets the pass criteria, retire Exp2E/Exp2F as the lane-head reference and continue toward Lane Decode + NMS (Priority 1 from the previous plan), then KD (Priority 2), then DETR upgrades (Priority 3 — which now matters more given the mAP50 ≈ 0.003 finding).
