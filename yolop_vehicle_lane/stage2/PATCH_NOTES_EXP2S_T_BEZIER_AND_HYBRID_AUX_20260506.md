# Stage 2 Exp2S + Exp2T — Bezier-curve queries and Hybrid + stage 1 supervision

## Why this patch exists

Per-epoch reads from NB22 (Exp2Q hybrid) and NB23 (Exp2R DAB-DETR) showed:

| metric @ ep10 | Exp2N (priors) | Exp2P (queries) | Exp2Q (hybrid) | Exp2R (DAB) |
|---|---:|---:|---:|---:|
| val_lane_f1 | 0.05 | 0.65 | 0.65 | 0.65 |
| matched_iou | **0.42** | 0.13 | 0.14 | **0.078** ⚠️ |
| decoded_f1 | 0.011 | 0.026 | 0.039 | 0.018 |
| oracle_f1 | **0.27** | 0.07 | 0.07 | 0.035 |
| val_det | 2.07 | 2.05 | 2.02 | **3.36** ⚠️ |

**Two unmistakable diagnoses:**

1. **Exp2Q's hybrid did NOT inherit Exp2N geometry.** Stage 1 (the 192-prior generator that hit matched_iou=0.42 in Exp2N) stayed at 0.14 inside the hybrid wrapper. Code review of the hybrid head and `FusionLaneLoss.forward`: stage 1's `coord_pred` is exposed in the output dict but is **not directly supervised**. The 192 prior curves only get gradient indirectly through stage 2's reading of `per_prior_features` — which optimizes those features for stage 2's K=12 ranking task, not for producing high-quality prior curves.

2. **Exp2R's DAB head broke joint training.** matched_iou collapsed monotonically (0.106 → 0.078), point_mae rose to 0.52, AND detection mAP simultaneously broke (val_det 2.21 → 3.36). Both train_lane and train_det rose together — a real joint-training divergence, not just lane-side trouble. **Cause: a real bug in my code** — the param head applies sigmoid twice (once on the anchor, once on the combined params), compressing the param space to ~[0.5, 0.73] so geometry literally cannot reach proper start positions.

Both diagnoses are concrete. This patch addresses (1) directly with **Exp2T**, and tests an **architectural alternative for the geometry-in-10-epochs problem** with **Exp2S**.

---

## Exp2T — Hybrid Q + auxiliary geometry supervision on stage 1

The diagnosed fix for Exp2Q. Add a `stage1_aux_loss_weight` field to `FusionLossConfig`. When > 0, `FusionLaneLoss.forward` calls `_forward_single_stage` *recursively* on the hybrid head's `stage1_*` keys (`stage1_cls_logits`, `stage1_coord_pred`, `stage1_lane_param`, `stage1_lane_offsets`) and adds the result to total loss with the configured weight.

This forces stage 1 to actually train as the 192-prior geometry champion (the same architecture Exp2N produced matched_iou=0.42 with). Stage 2 (K=12 query refiner) still owns the ranking task end-to-end.

### Files

- [stage2/fusion/losses.py](yolop_vehicle_lane/stage2/fusion/losses.py) — `stage1_aux_loss_weight` field + recursive call into `_forward_single_stage`. New components: `lane/stage1_aux_total`, `lane/stage1_aux_cls`, `lane/stage1_aux_iou`, `lane/stage1_aux_reg`.
- [stage2/configs/exp20_rmt_gca_hybrid_with_stage1_aux_joint.yaml](yolop_vehicle_lane/stage2/configs/exp20_rmt_gca_hybrid_with_stage1_aux_joint.yaml) — clone of Exp2Q's exp17 + `loss.lane.stage1_aux_loss_weight: 0.5`.
- [stage2/notebooks/stage2_notebook_25_exp2t_hybrid_stage1_aux_joint.ipynb](yolop_vehicle_lane/stage2/notebooks/stage2_notebook_25_exp2t_hybrid_stage1_aux_joint.ipynb).

### Hypothesis & predicted result

Stage 1's matched_iou should recover toward Exp2N's 0.42 (probably 0.30 under joint training). Stage 2's val_lane_f1 stays near 0.65. **decoded_f1 ≈ 0.30 × 0.65 ≈ 0.20** if the compositions are roughly multiplicative — 5× Exp2Q's 0.039 and an order of magnitude better than every prior-based attempt.

### Pass criteria

- `val/matched_line_iou ≥ 0.30`
- `val/lane_exist_best_f1 ≥ 0.55`
- `val/lane/decoded_f1 ≥ 0.15`
- `val/lane/decoded_oracle_f1 ≥ 0.18`

---

## Exp2S — Bezier-curve query head (BezierLaneNet-inspired)

A different angle on the geometry-in-10-epochs problem. All previous query experiments converged val_lane_f1 ~0.65 in 10 epochs but matched_iou stayed at ~0.13 because predicting `start_y/start_x/theta/length + 72 row offsets` (76 dof per lane) is too much for K=12 random-init queries to learn that fast.

**Cubic Bezier with 4 control points = 8 dof per lane**. An 8× reduction in output dimensionality. Cubic Bezier exactly represents straight and gentle-curve lanes (which dominate BDD).

### Files

- [stage2/fusion/lane_head.py](yolop_vehicle_lane/stage2/fusion/lane_head.py) — added `BezierLaneQueryHead`, `_bezier_curve_from_control_points` (forward sampler), and `_gt_lanes_to_bezier_targets` (exposed for an optional control-point regression loss; not used by default in Exp2S).
- [stage2/fusion/experiment_factory.py](yolop_vehicle_lane/stage2/fusion/experiment_factory.py) — `lane_head.type: 'bezier'` alias.
- [stage2/configs/exp19_rmt_gca_bezier_query_joint.yaml](yolop_vehicle_lane/stage2/configs/exp19_rmt_gca_bezier_query_joint.yaml).
- [stage2/notebooks/stage2_notebook_24_exp2s_bezier_query_joint.ipynb](yolop_vehicle_lane/stage2/notebooks/stage2_notebook_24_exp2s_bezier_query_joint.ipynb).

### Implementation

- K=12 queries through the same 3-layer transformer decoder used in Exp2P.
- Output per query: 1 cls logit + 4 control points (8d, sigmoid'd to [0,1]^2). Total 9 dof vs the standard 77.
- 72-point `coord_pred` is sampled deterministically from the Bezier curve at evenly-spaced t in [0, 1] using the cubic Bernstein basis. This keeps `FusionLaneLoss` and `LaneF1DecodedMetric` working unchanged.
- `lane_param` (start_y, start_x, theta, length) is derived analytically from the sampled curve so the existing reg/xytl/smooth losses still apply.
- `lane_offsets` is computed as `predicted_x − linear_baseline_x` so the offset loss term still makes sense.

### Pass criteria

- `val/matched_line_iou ≥ 0.30` (vs Exp2P's 0.13 — 2-3× better expected from the 8× compactness ratio).
- `val/lane_exist_best_f1 ≥ 0.55` (cls task unchanged).
- `val/lane/decoded_f1 ≥ 0.10`, ideally 0.15+.
- `val/lane/decoded_oracle_f1 ≥ 0.18`.

### Failure modes

- Geometry doesn't recover: BDD has lanes with multiple inflection points that need 4+ control points. Try quintic Bezier (6 control points) or revert.
- Cls regresses: parameterization shift broke the cls/geometry balance; raise `w_cls`.

---

## Independence

- **Exp2T** modifies `FusionLaneLoss` (adds aux supervision pathway) and uses the existing `HybridPriorQueryHead`. No architecture change.
- **Exp2S** adds a brand-new `BezierLaneQueryHead`. Doesn't touch the loss path.

Either can succeed independently. They test fundamentally different hypotheses about where the impasse comes from: T says "the hybrid was right, just incompletely supervised"; S says "the parameterization is too high-dimensional for fast convergence".

## Local smoke

`python -m compileall yolop_vehicle_lane/stage2/{fusion,metrics,scripts}` passes. Forward+backward smokes run from inside NB24 / NB25 cell 3 on Colab.

## Other follow-ups (not bundled)

- **Exp2R bug fix**: the double-sigmoid issue in `LaneQueryHeadAnchorDN._predict` should be fixed (`params = torch.cat([..., delta[..., 3:4].sigmoid()], dim=-1)` instead of an outer sigmoid). Trivial follow-up.
- **GCA gates frozen at 0.500** (`EXP_GCA_DIAGNOSTIC`).
- **Detection mAP50 ≈ 0.003** (`EXP_DET_MAP50_DEBUG`).
- **Real CLRKDNet KD**: still the project's promised feature, still unimplemented.

## Run order

Independent experiments. Run either or both:

1. **NB25 (Exp2T)**: smoke + debug + short10. Higher-confidence move; addresses the diagnosed Exp2Q bottleneck directly.
2. **NB24 (Exp2S)**: smoke + debug + short10. Speculative move; tests parameterization compactness as a lever.

After short10 runs, NB08 plots Exp2K through Exp2T side-by-side. The winner (or the best combination of both signals) becomes the parent for Stage 3 (extended training, video profiling, KD from teacher).
