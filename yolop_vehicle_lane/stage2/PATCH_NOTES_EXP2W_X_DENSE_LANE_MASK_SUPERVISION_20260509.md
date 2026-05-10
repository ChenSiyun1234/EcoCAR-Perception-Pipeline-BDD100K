# Stage 2 Exp2W + Exp2X — Restore dense lane mask supervision

## Why this patch exists

Exp2U (NB26 lane-only diagnostic) decisively **falsified** the detection-as-saboteur hypothesis. With detection completely zeroed, lane metrics moved by < 0.005 across every key indicator:

| | Exp2P (joint) | Exp2U (lane-only) | Δ |
|---|---:|---:|---:|
| val_lane_f1 | 0.652 | 0.642 | −0.010 |
| matched_iou | 0.133 | 0.139 | +0.006 |
| decoded_f1 | 0.026 | 0.029 | +0.003 |
| oracle_f1 | 0.071 | 0.076 | +0.005 |

**Detection has not been sabotaging the lane head.** The lane impasse is intrinsic to the lane task itself.

Re-examining 22 prior experiments through this lens reveals a striking pattern:

| Family | mask_aux | w_mask | matched_iou (e10) |
|---|:---:|:---:|---:|
| Exp2N (priors) | **true** | **1.0** | **0.42** |
| Exp2P–U (queries) | false | 0.0 | 0.13 |

**Every modern lane detector** — CLRNet (CVPR 2022), CLRKDNet (TIP 2024), CondLaneNet (ICCV 2021), BezierLaneNet (CVPR 2022) — uses a **per-pixel lane segmentation auxiliary head** for backbone supervision. We've been training the backbone *without any pixel-level lane signal* across every query-style experiment. **That's the missing piece.**

The infrastructure is already in place:
- The dataset produces `mask_target` from GT polylines via `soft_polyline_mask_numpy` (verified at [train_joint_model_experiment.py:295](yolop_vehicle_lane/stage2/scripts/train_joint_model_experiment.py)).
- `LaneQueryHead` and `HybridPriorQueryHead` already have a `mask_decoder` branch that activates when `mask_aux=True`.
- `FusionLaneLoss.forward` already computes BCE + Dice on `mask_logit + mask_target` when `w_mask > 0`.

We just have to turn it on.

---

## Exp2W — Single-knob test of the mask-supervision hypothesis

Same architecture as Exp2P (proven query-style cls=0.65). Two single-knob changes:
- `lane_head.mask_aux: false → true`
- `loss.lane.w_mask: 0.0 → 2.0` (CLRNet uses 1.0; we go higher to make the missing piece dominate gradient magnitude initially).

Nothing else changes. This is the clean ablation that tells us how much mask supervision matters in isolation.

### File

- [stage2/configs/exp23_rmt_gca_query_with_mask_aux_joint.yaml](yolop_vehicle_lane/stage2/configs/exp23_rmt_gca_query_with_mask_aux_joint.yaml)
- [stage2/notebooks/stage2_notebook_28_exp2w_query_with_mask_aux_joint.ipynb](yolop_vehicle_lane/stage2/notebooks/stage2_notebook_28_exp2w_query_with_mask_aux_joint.ipynb)

### Pass criteria (epoch 10)

- `val/lane/decoded_f1 ≥ 0.10` (~4× current best of 0.040)
- `val/matched_line_iou ≥ 0.30` (closer to Exp2N's 0.42)
- `val/lane/decoded_oracle_f1 ≥ 0.20`
- `val/lane/mask_aux` strictly decreases (training signal is real)

---

## Exp2X — Combine ALL working signals on the hybrid architecture

Same architecture as Exp2Q (the project's best decoded_f1 = 0.039). Three additions, layering every fix from prior experiments that should have worked:

1. **`mask_aux: true, w_mask: 2.0`** — the dense supervision (Exp2W's hypothesis).
2. **`stage1_aux_loss_weight: 0.5`** — Exp2T's diagnosed fix for the hybrid head's missing stage 1 supervision (gives stage 1's 192-prior generator direct geometry losses, like Exp2N had).
3. **`match_cost_iou: 2.0 → 4.0`** — when stage 1's matched_iou improves, the K=12 query selection cost should weight LineIoU more heavily so queries pick the best-IoU priors.

This is the kitchen-sink combination of every working ingredient identified across 22 prior experiments.

### File

- [stage2/configs/exp24_rmt_gca_hybrid_combined_signals_joint.yaml](yolop_vehicle_lane/stage2/configs/exp24_rmt_gca_hybrid_combined_signals_joint.yaml)
- [stage2/notebooks/stage2_notebook_29_exp2x_hybrid_combined_signals_joint.ipynb](yolop_vehicle_lane/stage2/notebooks/stage2_notebook_29_exp2x_hybrid_combined_signals_joint.ipynb)

### Pass criteria (epoch 10)

- `val/lane/decoded_f1 ≥ 0.15` (4-6× Exp2Q)
- `val/matched_line_iou ≥ 0.35` (closer to Exp2N's 0.42)
- `val/lane/decoded_oracle_f1 ≥ 0.20`
- `lane/stage1_aux_total` decreases (stage 1 supervision active)
- `val/lane/mask_aux` decreases (mask supervision active)

---

## Independence + comparison logic

| Exp2W decoded_f1 | Exp2X decoded_f1 | Interpretation |
|---|---|---|
| **high** (≥ 0.10) | **high** (≥ 0.15) | Mask supervision is the missing piece. Hybrid + everything wins. |
| high | not higher | Mask alone explains the impasse; stage 1 aux + matching cost don't add. |
| low | high | Hybrid combination matters; mask alone isn't enough. |
| low | low | Mask supervision isn't the answer. Pivot to extended training, KD from teacher, or higher resolution. |

This is a 2×2 design that resolves the mask-supervision question in one round.

## Local smoke

`python -m compileall yolop_vehicle_lane/stage2/{fusion,metrics,scripts}` passes. Forward+backward smokes run from inside NB28 / NB29 cell 3 on Colab.

## Why these are big moves

- **Exp2W**: enabling mask supervision is a *dense per-pixel auxiliary task* that we've explicitly turned off across every query experiment. It's not a knob — it's an entire supervision channel that's been silent.
- **Exp2X**: the kitchen-sink combination of every architectural+supervision insight learned across 22 experiments. If this fails, we know the bottleneck isn't anything we can solve on the lane-head side alone.

## Open follow-ups (not bundled)

- **NB27 (Exp2V grid det) smoke bug**: `smoke_test_joint_models.py` uses DETR's output format; needs to handle SimpleVehicleDetectionHead's tensor shape. Not on critical path now that Exp2U ruled out detection sabotage. Can be fixed if we revisit grid det later.
- **CLRKDNet teacher KD**: still the project's namesake feature, still unimplemented. Becomes the natural next big move if Exp2W/X both fail to break the impasse.
- **Higher resolution + extended training**: the obvious next move if mask supervision alone isn't enough.

## Run order

Independent. Run either or both:

1. **NB28 (Exp2W)**: smoke + debug + short10. The decisive single-knob test.
2. **NB29 (Exp2X)**: smoke + debug + short10. The kitchen-sink combination.

After short10 runs, NB08 plots Exp2P / Exp2U / Exp2W / Exp2X side-by-side. The decoded_f1 deltas tell us exactly how much dense lane supervision was costing us.
