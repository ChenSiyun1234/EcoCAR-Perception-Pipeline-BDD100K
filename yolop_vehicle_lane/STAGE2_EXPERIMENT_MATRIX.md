# Stage2 Experiment Matrix

Every stage2 row is either enabled directly by a YAML knob or by a
branch-specific YAML file. Reference line is Stage1 best-row.

## Ablation matrix

| id | config YAML | taxonomy | warm-start | distill | grad-balance | notebook | measures |
|---|---|---|---|---|---|---|---|
| **S1-best** | stage1/configs/yolopv2_best_row.yaml | 1c (stage1_vehicle_merged) | n/a | n/a | n/a | stage1/02 | fair-comparison reference |
| **S2-raw-1c** | stage2/configs/detrlane_vehicle1c.yaml | 1c (stage2_1c_vehicle_merged) | ❌ | ❌ | none | stage2/02 | architectural gain of stage2 lane head alone |
| **S2-warm-1c** | stage2/configs/detrlane_vehicle1c_warmstart.yaml | 1c | ✅ from S1-best | ❌ | none | stage2/03 | + training-stabilization gain from warm-start alone |
| **S2-warm-3c** | stage2/configs/detrlane_vehicle3c_warmstart.yaml | **3c (vehicle + motorcycle + bicycle)** | ✅ class-remapped | ❌ | none | stage2/03 | taxonomy expansion gain on top of warm-start |
| **S2-distill-3c** | stage2/configs/detrlane_vehicle3c_distill.yaml | 3c | ✅ | ✅ (box + obj + vehicle-cls KD from S1) | none | stage2/04 | + detection protection from distillation |
| **S2-distill-gradnorm-3c** | stage2/configs/detrlane_vehicle3c_distill_gradnorm.yaml | 3c | ✅ | ✅ | **uncertainty** (swap to `pcgrad` in YAML) | stage2/05 | + multi-task gradient-conflict mitigation |

## Gain decomposition

These A/B pairs isolate individual effects cleanly:

| gain measured | A | B | delta interpretation |
|---|---|---|---|
| architectural gain | S1-best | S2-raw-1c | effect of the stage2 lane head alone |
| warm-start gain | S2-raw-1c | S2-warm-1c | effect of warm-starting from stage1 |
| taxonomy expansion | S2-warm-1c | S2-warm-3c | cost/benefit of adding motorcycle + bicycle |
| distillation gain | S2-warm-3c | S2-distill-3c | KD signal's contribution to detection stability |
| gradient-balancing gain | S2-distill-3c | S2-distill-gradnorm-3c | MTL conflict mitigation contribution |

## Stage2 class taxonomies

### `stage2_1c_vehicle_merged` (same as stage1)

```
class 0 = vehicle   (car + bus + truck + train merged)
```

Directly comparable to the Stage1 baseline.

### `stage2_3c_extended` (added VRU classes)

```
class 0 = vehicle   (car + bus + truck + train merged)
class 1 = motorcycle
class 2 = bicycle
```

Not a drop-in comparison against stage1. Use for "does adding VRU
detection degrade vehicle mAP by more than it's worth?" ablations.

## Class-remap handling during warm-start

When a 3c student warm-starts from a 1c teacher:

- `box_{tx,ty,tw,th}` rows copied verbatim.
- `obj` row copied verbatim.
- teacher `vehicle` class row (row 5 per anchor) → student `vehicle`
  row 5 per anchor.
- student rows 6 (motorcycle) and 7 (bicycle) per anchor stay at
  student init (Kaiming for conv weights, YOLOP-style bias prior).

Implementation: `stage2/lib/utils/warm_start.py::warm_start_from_stage1`.

## Gradient balancing options

- `GRAD_BALANCE = 'none'` (default on the raw + warm-start + distill
  branches): plain summed loss.
- `GRAD_BALANCE = 'uncertainty'` (default on the full-stack branch):
  Kendall log-variance weighting. Cheap, robust.
- `GRAD_BALANCE = 'pcgrad'`: project conflicting gradients per pair
  before optimizer step. Expensive (one backward per task) but can
  help when preflight gradient-cosine diagnostics show negative
  correlation between det and lane gradients.

Implementation: `stage2/lib/core/grad_balance.py`.

## What stage2 does **not** change vs stage1

- Backbone (YOLOPv2 ELAN + SPP + FPN + PAN).
- Augmentation recipe is task-dependent: stage2 DETR-lane branches
  default `MOSAIC/MIXUP=False` because mosaic corrupts lane
  polylines; stage1's mask-based lane branch keeps them on.
- Focal γ + dice gain inherited from stage1 best-row.
- SGDR scheduler (with `T_0 = END_EPOCH // 2` for the shorter stage2
  runs).
- 640×384 rectangular val.

## Success criteria for stage2 → stage3 promotion

A stage2 branch is promotable to stage3 only if:

1. Lane IoU ≥ stage1 best-row lane IoU (no regression on the lane
   task).
2. Vehicle mAP@0.5 ≥ 0.95 × stage1 best-row mAP@0.5 (detection loses
   no more than 5 %).
3. A5000 FPS ≥ 30 at batch=1, FP16 (real-time deployment gate).

If no branch satisfies all three, stage3 inherits stage1 instead —
stage2 is declared an exploration-only stage for that run.
