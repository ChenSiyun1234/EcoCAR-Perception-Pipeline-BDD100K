# Stage 2 Exp2U + Exp2V — Is detection sabotaging the lane head?

## Why this patch exists

After fourteen attempts to fix the lane head architecture (Exp2G–Exp2T spanning prior-based, query-based, hybrid, anchor-conditioned, and Bezier curves), `decoded_f1` has been frozen in **[0.026, 0.040]** for all six query-style experiments. Meanwhile across **every** Exp2 run:

| signal | value | implication |
|---|---|---|
| `mAP50` | 0.003–0.005 | detection has **never** converged |
| `lambda_lane` (grad_norm) | 0.05–0.17 | det's gradient is **6–20× the lane gradient** in the joint loss |
| `train_det` | 2.0–3.4 (rises in some runs) | det loss is large and noisy throughout training |

**Hypothesis** (the user's hint): detection has been **sabotaging the lane head** the entire time. DETR with 100 queries needs 50+ epochs to converge — we train for 10 — so the detection task produces large but non-converging gradient flow into the **shared backbone** for the entire run. The shared backbone gets pulled toward features that satisfy a non-converging set-prediction task at 6–20× the magnitude of lane gradients. Lane features are reset by detection's backbone updates every batch. This is the textbook multi-task interference pattern that GradNorm / PCGrad were designed for.

Two parallel diagnostics, both single-knob changes vs Exp2P (the cleanest query-style baseline). They are independent.

---

## Exp2U — Lane-only training (decisive diagnostic)

Same architecture and lane head as Exp2P. **Zero all detection loss weights** (`bbox_loss_weight=0`, `cls_loss_weight=0`, `obj_loss_weight=0`, `giou_loss_weight=0`, `dn_loss_weight=0`). **Fix `lambda_lane=1.0`, `lambda_mode: fixed`** so grad_norm calibration doesn't compensate for the missing det signal.

The detection head still runs forward (model architecture is identical) but contributes nothing to the joint loss. **No gradient flows from detection into the shared backbone.** Lane is now training in isolation while keeping every other variable identical.

If lane metrics jump dramatically vs Exp2P, detection has been the saboteur. If they don't move, the lane impasse is something deeper.

### Files

- [stage2/configs/exp21_rmt_gca_lane_only_diagnostic_joint.yaml](yolop_vehicle_lane/stage2/configs/exp21_rmt_gca_lane_only_diagnostic_joint.yaml) — clone of Exp2P (`exp16`) with `loss.det.* = 0`, `lambda_lane = 1.0`, `lambda_mode: fixed`, `lambda_gate_reg: 0.0`, `no_object_weight: 0.0`.
- [stage2/notebooks/stage2_notebook_26_exp2u_lane_only_diagnostic_joint.ipynb](yolop_vehicle_lane/stage2/notebooks/stage2_notebook_26_exp2u_lane_only_diagnostic_joint.ipynb).

### Pass criteria (decision tree at epoch 10)

- **`decoded_f1 ≥ 0.15` (4× current best)** → detection IS the saboteur. Stage 3 path: fix detection (Exp2V's grid head, or freeze backbone after lane converges, or curriculum), then re-enable joint training.
- **`decoded_f1 ≥ 0.10` AND `matched_iou ≥ 0.30`** → strong but partial confirmation; detection is hurting but not the only issue.
- **`decoded_f1 ~ 0.04`** (no change from Exp2P) → detection is NOT the saboteur. The lane impasse is something else: training duration (10 vs CLRKDNet's 70 epochs), image resolution (384×640 vs 590×1640), data scale (3K vs 88K images per epoch), or KD from a real teacher.

This single experiment gives definitive signal on the core question.

---

## Exp2V — Grid detection head + query lane head

The codebase already has `SimpleVehicleDetectionHead` (anchor-free single-shot grid head) and `SimpleVehicleDetectionLoss`. Same lane head as Exp2P, single-knob change: `model.detection_head.type: detr → grid`.

A grid head can converge in our 10-epoch budget where DETR cannot. If the grid head reaches `mAP50 ≥ 0.10` AND lane metrics improve, then **a working detection task is compatible with a working lane head**, and the path forward is grid-det + query-lane jointly. If the grid head converges but lane stays stuck, the multi-task interference is structural rather than convergence-driven.

### Files

- [stage2/configs/exp22_rmt_gca_grid_det_with_query_lane_joint.yaml](yolop_vehicle_lane/stage2/configs/exp22_rmt_gca_grid_det_with_query_lane_joint.yaml) — clone of Exp2P with `model.detection_head.type: grid` plus the simpler grid head config (`embed_dim: 128`, `num_classes: 1`).
- [stage2/notebooks/stage2_notebook_27_exp2v_grid_det_with_query_lane_joint.ipynb](yolop_vehicle_lane/stage2/notebooks/stage2_notebook_27_exp2v_grid_det_with_query_lane_joint.ipynb).

### Pass criteria (epoch 10)

- **`val/det/metric_map50 ≥ 0.10`** — grid head converges where DETR couldn't.
- **`val/lane/decoded_f1 ≥ 0.10`** — lane benefits from a converging detection signal.
- **`val/matched_line_iou ≥ 0.25`**.
- `val_lane_f1 ≥ 0.55`.

---

## Independence + comparison logic

Both experiments are **independent** (no order, no dependency). Together they form a decisive matrix:

| Exp2U decoded_f1 | Exp2V decoded_f1 | Interpretation |
|---|---|---|
| **high** | **high** | Detection convergence is what matters; grid head fixes it |
| high | low | Detection is poison regardless of head choice; need to remove or freeze it |
| low | high | Lane benefits from a working detection signal (positive feature sharing) |
| low | low | Detection isn't the issue; pivot to training duration / resolution / KD |

This is a 2×2 design that resolves the user's question in one round.

## Local smoke

`python -m compileall yolop_vehicle_lane/stage2/{fusion,metrics,scripts}` passes. Forward+backward smokes run from inside NB26 / NB27 cell 3 on Colab.

## Why this is "big move", not tuning

The user asked specifically about whether detection affects lane. These experiments don't tweak hyperparameters of the existing setup — they explicitly test the multi-task interference hypothesis with two architecturally distinct treatments:

- **Exp2U** removes the entire detection signal from backprop (a structural change to the training objective).
- **Exp2V** swaps the entire detection head architecture from DETR (set-prediction transformer) to anchor-free grid (single-shot conv). These are fundamentally different paradigms — not a knob.

Together they isolate the hypothesis cleanly.

## Open follow-ups (not bundled)

- **GCA gates frozen at 0.500** — separate `EXP_GCA_DIAGNOSTIC` follow-up.
- **CLRKDNet teacher KD** — still the project's namesake feature, deferred until detection question is resolved.
- **Extended training (30+ epochs)** — if Exp2U/V both fail, this is the obvious next test.

## Run order

Independent. Run either or both:

1. **NB26 (Exp2U)**: smoke + debug + short10. The diagnostic.
2. **NB27 (Exp2V)**: smoke + debug + short10. The "make detection work" alternative.

After short10 runs, NB08 plots Exp2P vs Exp2U vs Exp2V side-by-side. The deltas tell us exactly how much detection has been costing us, and whether the fix is removal or a working head.
