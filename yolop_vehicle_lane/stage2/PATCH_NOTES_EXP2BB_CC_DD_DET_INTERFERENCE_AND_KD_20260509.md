# Stage 2 Exp2BB + Exp2CC + Exp2DD — Det interference test, extended training, project's namesake KD

## Why this patch exists

NB30/31/32 results:

| @ ep15 | val_lane_f1 | matched_iou | **decoded_f1** | **oracle_f1** | val_det_loss |
|---|---:|---:|---:|---:|---:|
| Exp2W (no fix) | 0.644 | 0.143 | 0.0427 | 0.074 | 2.04 |
| Exp2Y (fixed λ) | 0.689 | 0.138 | 0.0388 | 0.079 | 2.01 |
| **Exp2Z (uncertainty)** | 0.681 | **0.162** | 0.0442 | **0.101** | **1.93** |
| Exp2AA (480×800) | — | — | — | — | smoke FAILED |

**Three signals:**

1. **Exp2Z (Kendall uncertainty weighting) is the new project best**: matched_iou=0.162, oracle_f1=0.101 (first time crossing 0.10), val_det_loss=1.93, all simultaneously. The `train_total: 2.84 → 1.90` is monotonic with no sawtooth — uncertainty weighting cleanly fixed the lambda oscillation.
2. **decoded_f1 still stuck at ~0.044**: the cls head throws away 56% of the geometric quality (decoded/oracle = 0.44). Lambda stability lifted the ceiling but didn't break the decode bottleneck.
3. **NB32 (Exp2AA 480×800) failed at smoke**: `smoke_test_joint_models.py` hardcoded `mask_target=(1,1,72,128)` instead of reading the config's `aux_mask_size`. Bug in my code. Fixed in this patch.

## The user's hypothesis: detection head interferes with lane

Quoted insight: **"is the detection head affecting the lane head?"**

The data backs this up:
- Exp2Z's `val_det_loss` decreases monotonically (2.20 → 1.93 across 15 epochs)
- But `val/det/metric_map50` stays at ~0.005 (det isn't actually getting better at detection)
- Det's gradients flow into the shared backbone with no functional payoff
- **Periods where lane improvements stall coincide with det loss decreasing**

This patch tests three hypotheses about why decoded_f1 plateaus, in three independent experiments.

## What changed

### Infrastructure

1. [`stage2/scripts/smoke_test_joint_models.py`](yolop_vehicle_lane/stage2/scripts/smoke_test_joint_models.py) — read `cfg['dataset']['aux_mask_size']` for the dummy `mask_target` shape instead of hardcoding `(1,1,72,128)`. **Unblocks NB32 (Exp2AA 480×800)**: the user can re-run NB32 now and it will pass smoke.

2. [`stage2/scripts/train_joint_model_experiment.py`](yolop_vehicle_lane/stage2/scripts/train_joint_model_experiment.py) — added `loss.lambda_det` config knob (default 1.0). When set, scales `l_det` before joint combination: `total = lambda_det * l_det + lambda_lane * l_lane`. Det still gets full supervision through its own loss components; only its weight in the joint backbone-gradient path is scaled.

### Three independent experiments

#### Exp2BB (NB33, exp28) — Lane-dominant joint (det interference test)

[exp28_*.yaml](yolop_vehicle_lane/stage2/configs/exp28_rmt_gca_mask_lane_dominant_joint.yaml): identical to Exp2Z + `loss.lambda_det: 0.1`. Det's influence on the shared backbone is 10× weaker.

**The user's hypothesis test.** If decoded_f1 jumps from 0.044 → 0.07+, det's gradients were actively corrupting backbone for lane.

#### Exp2CC (NB34, exp29) — Exp2Z + 30 epochs (convergence test)

[exp29_*.yaml](yolop_vehicle_lane/stage2/configs/exp29_rmt_gca_mask_uncertainty_long30_joint.yaml): identical to Exp2Z + `end_epoch: 15 → 30`. matched_iou was *still climbing* at epoch 15 (0.114 → 0.162 trajectory had no plateau).

**The convergence-budget test.** If oracle_f1 keeps climbing past 0.15 at epoch 30, the impasse was just training duration. CLRKDNet trains 70+ epochs.

#### Exp2DD (NB35, exp30) — Self-distillation from Exp2Z teacher (PROJECT NAMESAKE)

[exp30_*.yaml](yolop_vehicle_lane/stage2/configs/exp30_rmt_gca_mask_self_distillation_joint.yaml): identical to Exp2Z + `loss.lane.w_distill: 1.0` + `teacher.lane_head_checkpoint: <Exp2Z best.pt path>`.

**Knowledge distillation (CLRKD = CLRKDNet KD) is the project's literal name and has never been turned on.** The plumbing already exists in `train_joint_model_experiment.py:95` (`load_lane_teacher`) and `losses.py:w_distill`. We just turn it on.

NB35 includes a one-time setup cell that extracts `best.pt` from the Exp2Z tar before training. The student then trains with both:
- Hard targets from BDD GT (the standard supervision)
- Soft targets from frozen Exp2Z teacher (matches teacher's `cls_logits` and `coord_pred`)

## Independence + comparison logic

| Exp2BB | Exp2CC | Exp2DD | Interpretation |
|---|---|---|---|
| **decoded_f1 jumps** | similar | similar | Det interference was the issue. Reduce det loss weight or freeze det. |
| similar | **decoded_f1 jumps** | similar | Convergence-limited. CLRKDNet's 70-epoch budget is real. |
| similar | similar | **decoded_f1 jumps** | KD is the unlock. Project-name feature finally pays off. |
| all jump | | | Multiple bottlenecks in parallel; combine fixes (Exp2EE = lane-dom + KD + 30 ep). |
| none jump | | | Architecture's true capacity reached. Pivot to bigger backbone or higher resolution. |

Three experiments × independent hypotheses = decisive resolution of the next-step question.

## Files

- New configs: [exp28](yolop_vehicle_lane/stage2/configs/exp28_rmt_gca_mask_lane_dominant_joint.yaml), [exp29](yolop_vehicle_lane/stage2/configs/exp29_rmt_gca_mask_uncertainty_long30_joint.yaml), [exp30](yolop_vehicle_lane/stage2/configs/exp30_rmt_gca_mask_self_distillation_joint.yaml)
- New notebooks: [NB33 (Exp2BB)](yolop_vehicle_lane/stage2/notebooks/stage2_notebook_33_exp2bb_lane_dominant_joint.ipynb), [NB34 (Exp2CC)](yolop_vehicle_lane/stage2/notebooks/stage2_notebook_34_exp2cc_uncertainty_long30_joint.ipynb), [NB35 (Exp2DD)](yolop_vehicle_lane/stage2/notebooks/stage2_notebook_35_exp2dd_self_distillation_joint.ipynb)
- NB08 extended with all three
- Smoke harness fix: NB32 (Exp2AA 480×800) is now re-runnable. Just re-run cell 3 then cell 4.

## Run order

Independent — any combination, any order:
1. **NB33 (Exp2BB)**: smoke + debug + short15. Tests user's det-interference hypothesis.
2. **NB34 (Exp2CC)**: smoke + debug + long30. Tests convergence-budget hypothesis.
3. **NB35 (Exp2DD)**: requires Exp2Z (NB31) finished first to provide teacher checkpoint. Then extract teacher .pt + smoke + debug + short15.
4. **NB32 (Exp2AA)** retry: re-run cell 3 (smoke now works) then cell 4.

## Pass criteria

Any of: `decoded_f1 ≥ 0.07` (1.6× Exp2Z's 0.044). The first to break this becomes the parent for Stage 3.

If all four plateau at the Exp2Z level (~0.044), the bottleneck is genuinely outside what these treatments reach, and the next move is real CLRKDNet teacher KD (loading external pre-trained CLRKDNet weights) — significantly more work but addresses the architecture's true capacity ceiling.
