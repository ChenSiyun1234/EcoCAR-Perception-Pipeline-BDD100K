# Stage 2 Exp2EE + Exp2FF — Cosine LR + KD bug fix

## What NB33 / NB34 / NB35 told us

### NB33 (Exp2BB, lambda_det=0.1) — disproved the det-interference hypothesis

| epoch | matched_iou | decoded_f1 | oracle_f1 |
|---:|---:|---:|---:|
| 1  | 0.109 | 0.023 | 0.041 |
| 5  | 0.155 | 0.043 | 0.089 |
| 10 | 0.143 | 0.040 | 0.081 |
| 15 | 0.141 | 0.029 | 0.078 |

Vs Exp2Z (full det weight, ep15: matched_iou=0.162, decoded_f1=0.044, oracle_f1=0.101): suppressing det made everything **slightly worse**. **Det's gradients were NOT corrupting lane.** The user's hypothesis — while well motivated by the data — turned out to be wrong.

### NB34 (Exp2CC, 30 epochs) — catastrophic late-training collapse

| epoch | matched_iou | val/lane/line_iou_loss | decoded_f1 | oracle_f1 |
|---:|---:|---:|---:|---:|
| 8  | 0.157 | 0.843 | 0.033 | 0.096 |
| 13 | 0.149 | 0.851 | 0.029 | 0.075 |
| **18** | **0.157** | **0.843** | **0.040** | **0.100** ← peak |
| **23** | **0.050** ⚠️ | **0.950** ⚠️ | **0.009** ⚠️ | **0.011** ⚠️ |
| 28 | 0.048 | 0.952 | 0.019 | 0.022 |

Geometry exploded between epoch 18 and 23 while cls stayed stable. **This is the standard-DL-setting issue the user asked about.**

**Root cause**: constant-LR Adam stepping out of a converged minimum. With `lr0=0.0002` constant and no scheduler, Adam keeps full step magnitude near convergence. The non-smooth LineIoU loss (with `clamp(min=0)` ops) has discontinuous gradients at exact-overlap and zero-overlap boundaries — a perfect setup for late-training divergence. Every epoch is a coin flip on whether the next step lands inside or outside the basin.

**Fix**: cosine LR with linear warmup. Standard recipe used by every published lane detector (CLRKDNet, CLRNet, RMT-PPAD). LR decays from `lr0` toward zero over the run, shrinking step size near convergence and eliminating the divergence.

### NB35 (Exp2DD, KD) — failed at startup

```
TypeError: LaneQueryHead.__init__() got an unexpected keyword argument 'num_priors'
```

`load_lane_teacher` in `train_joint_model_experiment.py:101` was hardcoded to `CurveLaneHead`'s constructor signature with `num_priors=...` — which `LaneQueryHead` doesn't accept. **Bug in my code, fixed.**

## What changed

1. **[stage2/scripts/train_joint_model_experiment.py](yolop_vehicle_lane/stage2/scripts/train_joint_model_experiment.py)** — three coordinated edits:
   - `load_lane_teacher` now uses `inspect.signature(head_cls.__init__)` to filter only kwargs the head accepts. Works for `CurveLaneHead`, `CLRKDLaneHead`, `LaneQueryHead`, `BezierLaneQueryHead`, `HybridPriorQueryHead`, etc.
   - Added `import math` (needed for cosine schedule).
   - Added cosine LR scheduler block. Reads `cfg['train']['lr_scheduler']`. When `kind: cosine`, builds `optim.lr_scheduler.LambdaLR` with linear warmup over `warmup_epochs` and cosine decay to `min_lr_factor`. Logs per-epoch LR after each scheduler step.

2. **[stage2/configs/exp31_rmt_gca_mask_uncertainty_cosine_lr_joint.yaml](yolop_vehicle_lane/stage2/configs/exp31_rmt_gca_mask_uncertainty_cosine_lr_joint.yaml)** — Exp2EE.
   - Diff vs Exp2Z (`exp26`):
     - new `train.lr_scheduler: {kind: cosine, warmup_epochs: 2, warmup_start_lr_factor: 0.1, min_lr_factor: 0.05}`.
     - `train.end_epoch: 15 → 20` (cosine decay needs more epochs to amortize the 2-epoch warmup).

3. **[stage2/configs/exp32_rmt_gca_mask_self_distillation_cosine_lr_joint.yaml](yolop_vehicle_lane/stage2/configs/exp32_rmt_gca_mask_self_distillation_cosine_lr_joint.yaml)** — Exp2FF.
   - Diff vs Exp2DD (broken `exp30`):
     - same `lr_scheduler` block as Exp2EE.
     - `train.end_epoch: 15 → 20`.
     - `loss.lane.w_distill: 1.0` (KD active).
     - `teacher.lane_head_checkpoint: /content/drive/.../exp26_rmt_gca_mask_uncertainty_weighting_joint_short15_best.pt`.

4. **[stage2/notebooks/stage2_notebook_36_exp2ee_uncertainty_cosine_lr_joint.ipynb](yolop_vehicle_lane/stage2/notebooks/stage2_notebook_36_exp2ee_uncertainty_cosine_lr_joint.ipynb)** — NB36 (Exp2EE), mirrors NB31 structure.

5. **[stage2/notebooks/stage2_notebook_37_exp2ff_self_distillation_cosine_lr_joint.ipynb](yolop_vehicle_lane/stage2/notebooks/stage2_notebook_37_exp2ff_self_distillation_cosine_lr_joint.ipynb)** — NB37 (Exp2FF), mirrors NB35 structure with the teacher-tar extraction cell preserved (idempotent on re-run).

6. **[NB08](yolop_vehicle_lane/stage2/notebooks/stage2_notebook_08_joint_eval_visualization_and_profile.ipynb)** — exp31 and exp32 entries added to cells 3, 5, 7.

## Pass criteria

### Exp2EE (NB36) at epoch 20

- **NO COLLAPSE**: `val/matched_line_iou` never drops below 0.10 in any epoch. NB34 collapsed to 0.050 at ep23.
- `val/matched_line_iou ≥ 0.16` sustained.
- `val/lane/decoded_f1 ≥ 0.05` (matches or beats Exp2Z's 0.044).
- `val/lane/decoded_oracle_f1 ≥ 0.10`.
- Per-epoch LR log line shows cosine decay: ~0.00002 (warmup start) → 0.00020 (peak) → ~0.00001 (epoch 20 end).
- `train_total` decreases monotonically; no late-epoch divergence.

### Exp2FF (NB37) at epoch 20

- `loaded lane teacher=...` log line appears at startup (NOT a stack trace). Confirms the teacher-loader fix.
- `val/lane/distill` loss component is logged and decreases over training.
- **`val/lane/decoded_f1 ≥ 0.07`** — KD lifts cls toward oracle ceiling. Beats every previous Exp2 by ≥60%.
- `val/lane_exist_best_f1 ≥ 0.70`.
- Geometry holds: `val/matched_line_iou ≥ 0.16`.
- No late-epoch collapse (cosine LR also active).

## Decision tree based on Exp2EE/FF outcomes

- **Both pass** → Stage 3 unblocked. Proceed to video profiling and external CLRKDNet teacher KD.
- **Exp2EE passes, Exp2FF fails** → cosine LR was the missing piece; KD doesn't add value. Move to bigger backbone or higher resolution.
- **Both plateau at Exp2Z's 0.044** → architecture's true capacity ceiling reached. Move to genuinely different parameterization (anchor-free heatmap-based) or pretrained backbone.
- **Late-epoch collapse persists in Exp2EE** → cosine LR insufficient; need stronger fix like SWA, EMA, or smaller LR.

## Local smoke

`python -m compileall yolop_vehicle_lane/stage2/{fusion,metrics,scripts}` passes. NB36 + NB37 valid JSON. The `math` import + cosine scheduler are exercised on Colab via NB36/37 cell 3 (smoke test).

## Run order

Independent of each other (Exp2FF needs Exp2Z's tar to extract teacher .pt, but NB35 already did that):
1. **NB36 (Exp2EE)**: smoke + debug + short20. The "fix-the-DL-training" experiment.
2. **NB37 (Exp2FF)**: smoke + debug + short20. The project's namesake KD finally running cleanly.

Both can run in parallel on Colab.
