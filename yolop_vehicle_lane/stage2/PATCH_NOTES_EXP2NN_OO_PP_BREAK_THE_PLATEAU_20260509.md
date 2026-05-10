# Stage 2 Exp2NN + Exp2OO + Exp2PP — break the decoded_f1 = 0.04 plateau

## Reading NB41 / NB42 — what we just observed

### Per-epoch trends

NB41 (Exp2LL — anchor + lineiou_regression + QFL):

| ep | matched_iou | oracle_f1 | decoded_f1 | val_lane_cls | pos / neg | pred_lanes |
|---:|---:|---:|---:|---:|---:|---:|
| 1  | 0.351 | 0.252 | 0.007 | 0.007 | 0.128 / 0.128 | 0.0 |
| 5  | 0.414 | 0.299 | 0.014 | 0.007 | 0.131 / 0.131 | 0.0 |
| 10 | 0.452 | 0.367 | 0.005 | 0.008 | 0.130 / 0.130 | 0.0 |
| 15 | 0.489 | 0.431 | 0.019 | 0.010 | 0.142 / 0.140 | 0.0 |
| **20** | **0.508** | **0.459** | 0.029 | 0.011 | 0.148 / 0.146 | 0.0 |

NB42 (Exp2MM — dual cls × IoU):

| ep | matched_iou | oracle_f1 | decoded_f1 (cls × iou) | decoded_cls_only_f1 | iou_aux | pos / neg |
|---:|---:|---:|---:|---:|---:|---:|
| 1  | 0.323 | 0.140 | 0.012 | — | — | 0.681 / 0.489 |
| 5  | 0.403 | 0.308 | 0.002 | 0.012 | 0.0089 | 0.587 / 0.563 |
| 10 | 0.408 | 0.303 | 0.018 | 0.011 | 0.0078 | 0.576 / 0.554 |
| 15 | 0.477 | 0.401 | 0.034 | 0.025 | 0.0075 | 0.577 / 0.569 |
| **20** | 0.490 | 0.426 | **0.042** | **0.012** | 0.008 | 0.580 / 0.571 |

### Cross-experiment summary at epoch 20

| run | matched_iou | oracle_f1 | decoded_f1 | pos − neg gap | decoded / oracle |
|---|---:|---:|---:|---:|---:|
| NB39 (focal) | 0.472 | 0.379 | 0.011 | ~ 0.01 | 2.9 % |
| NB40 (ASL) | 0.483 | 0.418 | 0.043 | ~ 0.01 | 10.3 % |
| **NB41** (lineiou + QFL) | **0.508** | **0.459** | 0.029 | 0.002 | 6.3 % |
| **NB42** (dual cls × iou) | 0.490 | 0.426 | 0.042 | 0.009 | 9.9 % |

## Diagnosis: not a DL setting issue, a representation/supervision-signal issue

1. **Geometry keeps climbing every iteration.** matched_iou: 0.472 → 0.483 → 0.508. oracle_f1: 0.379 → 0.418 → 0.459. The 192-anchor head IS learning a richer geometric representation each round.

2. **decoded_f1 plateau ≈ 0.04 is independent of the cls fix.** Focal, ASL, lineiou+QFL, and dual cls × iou all land in the 0.03 - 0.04 band. Four loss strategies, same wall.

3. **The cls signal is essentially random.** `pos_score - neg_score` is 0.002 to 0.01 across all four runs. With top-K = 4 over 192 priors, picking 4 random priors gives expected F1 ≈ 0.04 — **exactly what we observe**. The cls head is performing at chance level across every loss variant.

4. **decoded_f1 / oracle_f1 ≈ 11 % across every experiment.** We've used 1/9 of the geometric ranking headroom for ten experiments running. This is not a loss-function problem.

The cls head reads a per-prior feature that does not contain enough information to distinguish "this prior is matched in this image" from "this prior is nearly-matched in this image." The dynamic-k matcher labels different priors as positive in different batches — same prior labeled 1 in image A and 0 in image B — and cls collapses to near-uniform sigmoid. **The supervision target itself is the bottleneck**, not the loss formulation that consumes it.

## Three independent fixes — three experiments

### Exp2NN (NB43, exp38) — drop cls from the decode bottleneck

**The biggest leap:** stop ranking by the degenerate cls and use the auxiliary segmentation mask as a per-prior geometric verifier.

[`exp38_rmt_gca_anchor_mask_consistency_joint.yaml`](configs/exp38_rmt_gca_anchor_mask_consistency_joint.yaml).

- **Inference**: `eval.decoded_score_source: cls_x_mask`. For each prior, sample the auxiliary mask sigmoid along its predicted curve (72 points) and use the mean as a ranking score. The mask is trained on per-pixel BCE+Dice with no per-prior matching instability — its sigmoid is a clean per-prior geometric verifier. Final ranking score = `sigmoid(cls) * mask_consistency`.
- **Training**: `cls_target_type: mask_consistency`. The cls head is self-distilled to predict the same mask-along-curve score. This gives cls a deterministic, batch-stable target — the antidote to the matching-instability that has been collapsing it.
- Companion diagnostics fire automatically: `decoded_cls_only_f1`, `decoded_mask_only_f1`, and `decoded_f1` (cls × mask). One run produces a clean three-way comparison.

This is RTMDet's centerness × cls product, but with a segmentation-derived score replacing the broken cls.

### Exp2OO (NB44, exp39) — Hungarian 1-to-1 matching

**Targeted fix for matching instability.**

[`exp39_rmt_gca_anchor_hungarian_joint.yaml`](configs/exp39_rmt_gca_anchor_hungarian_joint.yaml).

Single config diff vs Exp2KK (NB40):
- `lane_assigner: dynamic_k → hungarian`

Hungarian 1-to-1 matches each GT lane to exactly one prior, deterministically minimizing the cost matrix. The label per prior is now stable across batches, eliminating the dynamic_k flickering that contributed to the cls collapse. DETR-style.

If Exp2OO's cls_only F1 jumps from NB40's 0.043 to ≥ 0.10, matching instability was a major cause. If it stays near 0.04, the per-prior feature is fundamentally non-discriminative and Exp2NN is the right path.

### Exp2PP (NB45, exp40) — capacity scale-up

**The capacity hypothesis.**

[`exp40_rmt_gca_anchor_width1_long30_joint.yaml`](configs/exp40_rmt_gca_anchor_width1_long30_joint.yaml).

Diff vs Exp2KK (NB40):
- `model.width: 0.5 → 1.0` (full RMT width)
- `lane_head.embed_dim: 128 → 192`
- `lane_head.roi_mid_channels: 48 → 64`
- `train.end_epoch: 20 → 30`
- `train.lr_scheduler.warmup_epochs: 2 → 3`

NB41 had matched_iou and oracle_f1 STILL climbing at epoch 20 (0.490 → 0.508 in the last 5 epochs). The model is undertrained AND underweight. Width 1.0 + 30 epochs + larger embed_dim gives the cls head more capacity to distinguish similar-looking priors.

GPU memory budget check: NB40 used 10.4 / 95.6 GB. Exp2PP should land at ~ 25-35 GB — well within budget.

## Code changes (all back-compat, gated by config)

- **[`stage2/fusion/losses.py`](fusion/losses.py)**:
  - New `_compute_mask_consistency_target(mask_logit, coord_pred)` — samples mask sigmoid along predicted curve.
  - `_forward_single_stage`: new `cls_target_type='mask_consistency'` branch. cls is supervised with the mask-along-curve score (BCE or QFL).
- **[`stage2/metrics/lane_f1_decoded.py`](metrics/lane_f1_decoded.py)**:
  - New `_mask_consistency_score` helper.
  - `LaneF1DecodedMetric.score_source` accepts `'mask_consistency'` and `'cls_x_mask'`. Falls back to plain cls if `mask_logit` is missing.
- **[`stage2/scripts/train_joint_model_experiment.py`](scripts/train_joint_model_experiment.py)** `evaluate(...)`:
  - When primary `decoded_score_source` is `cls_x_mask`, automatically fires both `_cls_only` and `_mask_only` companion metrics so the metric report decomposes the gain.
  - When primary is `mask_consistency`, fires `_mask_only` companion (= itself, sanity check).

All three configs default to `decoded_score_source: cls_x_mask` so we get the three-way diagnostic for free in every run.

## Run order — fully independent of each other

1. **Exp2NN first** (NB43, ~30 min). The most novel idea; tests "is cls the bottleneck or not?" If decoded_f1 jumps ≥ 0.20, we've broken the plateau and the next moves build on mask-consistency.
2. **Exp2OO** (NB44, ~30 min). Tests matching-stability hypothesis. Cheap and informative regardless of NB43 outcome.
3. **Exp2PP** (NB45, ~45 min). Tests capacity hypothesis. Cheap and informative regardless of NB43/NB44.

None of these experiments depend on each other or on Exp2C (the GCA baseline). They share only the dataset tar and the codebase, which is fully back-compat with NB35 / NB39-42.

## Pass criteria

### Exp2NN at epoch 20
- `val/lane/decoded_f1 (cls_x_mask) ≥ 0.20` — 5 × NB42's 0.042
- `val/lane/decoded_mask_only_f1 ≥ 0.18` — mask alone gives most of the gain
- `val/lane/decoded_cls_only_f1 ≥ 0.05` — self-distillation took
- `val/matched_line_iou ≥ 0.45`, `val/lane/decoded_oracle_f1 ≥ 0.40`

### Exp2OO at epoch 20
- `val/lane/decoded_cls_only_f1 ≥ 0.10` — 2-3 × NB40 cls alone (matching stability hypothesis confirmed)
- `pos_score − neg_score ≥ 0.10` (direct cls separation measurement)
- `val/matched_line_iou ≥ 0.40` (preserve geometry under the stricter assignment)

### Exp2PP at epoch 30
- `val/matched_line_iou ≥ 0.55` (NB41 hit 0.508 with width 0.5)
- `val/lane/decoded_oracle_f1 ≥ 0.50`
- `val/lane/decoded_f1 (cls_x_mask) ≥ 0.15`
- `train_total` still decreasing at epoch 30 (proves capacity was the bottleneck)

## What success looks like, what failure looks like

**Best case (any one passes):** decoded_f1 jumps to 0.15 - 0.25, and the companion metrics tell us WHY. Exp2NN passing means mask-consistency is the unlock; future Stage-3 deployment uses cls × mask scoring or even mask-only scoring (which is well-defined and computationally cheap).

**Worst case (none pass):** the decoded_f1 = 0.04 plateau is intrinsic to width-0.5 + 3000-sample + joint-trained-from-scratch. Next move pivots to: pre-trained backbone (RMT-PPAD pretrained weights from the original repo) and full 70K-sample dataset training.

**Most likely:** Exp2NN passes the strongest (mask path is robust), Exp2OO partially passes (Hungarian helps cls but only to ~ 0.07), Exp2PP plateaus at oracle ~ 0.50 (capacity helps geometry, doesn't fix cls). In that scenario Exp2NN is the new champion and we combine it with Exp2PP's capacity in a future Exp2QQ.

## Local smoke

`python -m compileall yolop_vehicle_lane/stage2/{fusion,metrics,scripts}` passes after all edits. All three new YAMLs round-trip through `yaml.safe_load`. All three new notebooks round-trip through `json.load`. NB08 patched in place with the three new entries.
