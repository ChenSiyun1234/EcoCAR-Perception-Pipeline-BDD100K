# Stage 2 Exp2LL + Exp2MM — fix the cls collapse on the anchor head

## Reading NB40 (Exp2KK, anchor head + ASL + AMP) — what we just observed

Per-epoch trends from `stage2_notebook_40_exp2kk_anchor_asl_amp_joint.ipynb`:

| ep | matched_iou | oracle_f1 | decoded_f1 | val_lane_cls | grad_cos_mean | pred_lanes |
|---:|---:|---:|---:|---:|---:|---:|
| 1  | 0.371 | 0.232 | 0.014 | 0.090 | +0.05 | 1532 |
| 5  | 0.421 | 0.318 | 0.024 | 0.089 | +0.02 | 1535 |
| 10 | 0.466 | 0.396 | 0.039 | 0.088 | +0.01 | 1536 |
| 15 | 0.479 | 0.412 | 0.041 | 0.088 | -0.02 | 1536 |
| **20** | **0.483** ✓ | **0.418** ✓ | **0.043** | **0.087** ⚠️ | **-0.06** ⚠️ | **1536** ⚠️ |

Three things are simultaneously true:
1. **Geometry is the project's all-time best.** matched_iou=0.483 and oracle_f1=0.418 are CLRKDNet-CULane territory. The 192-anchor head + ROI gather + dynamic-k matching is doing exactly what it should geometrically.
2. **Binary cls is FLAT.** val_lane_cls hugged 0.087-0.090 across all 20 epochs. ASL γ_neg=4 (the NB40 fix) did not separate the 5/192 matched priors from the 187/192 unmatched ones. Decoded threshold 0.3 → ALL 192 priors above threshold every batch → pred_lanes = 192 × 8 = 1536.
3. **Late-training task conflict.** grad_cosine flipped from +0.05 (ep1) to -0.06 (ep20). Lane and det gradients are pulling the shared backbone in opposite directions in the back half of training, which explains why decoded_f1 stalled even as oracle_f1 kept climbing.

**Decoded-to-oracle ratio:** 0.043 / 0.418 = 10.3 %. Geometry has 9.7 × the ranking headroom we are using. The cls task is the bottleneck and the binary {0, 1} formulation is structurally incompatible with the 192-anchor design.

## Why binary matched_existence keeps failing on this head

The matching outcome itself is unstable. With dynamic-k top-k=4 and 5 GT lanes per image (mean 5.81 from the dataset visualization), the matcher labels ~20 priors as positive in any single batch — but **which** 20 priors gets re-decided every batch depending on which competing priors win the IoU contest. So the same prior swings positive ↔ negative across batches, and the cls head learns a degenerate "predict ~ 0.5 for everyone" equilibrium because that minimizes expected loss across the unstable label.

ASL γ_neg=4 amplified the gradient on confident negatives but the underlying instability remained — sigmoid still settled near 0.5 because the matched_existence target itself flickers.

## Two fixes, two experiments

### Exp2LL (NB41, exp36) — replace binary cls with continuous LineIoU regression

[`exp36_rmt_gca_anchor_iou_regression_joint.yaml`](configs/exp36_rmt_gca_anchor_iou_regression_joint.yaml).

**Single-knob diff vs Exp2KK (exp35):**
- `cls_target_type: matched_existence → lineiou_regression`
- `cls_loss_type: asl → qfl`
- `w_cls: 4.0 → 5.0` (QFL has ~ 5 × smaller scale than ASL on this dataset)

The cls target is now `_compute_lineiou_target(coord_pred.detach(), points_gt, vis)` — a deterministic continuous value in [0, 1] that does not depend on which competing prior wins the dynamic-k contest. QFL weights BCE by `|target − sigmoid(logit)|^γ` so the 95 % of priors with target ~ 0 get near-zero gradient and cannot collapse all logits to 0 the way Exp2K (plain BCE) did.

This is the **loss-side** attack on the cls collapse. Zero code changes required — `lineiou_regression` + `qfl` is already supported in `FusionLaneLoss._forward_single_stage`.

### Exp2MM (NB42, exp37) — keep binary cls AND add a parallel iou regression head

[`exp37_rmt_gca_anchor_dual_score_joint.yaml`](configs/exp37_rmt_gca_anchor_dual_score_joint.yaml).

**Architectural diff vs Exp2KK (exp35):**
- `model.lane_head.dual_score: true` — new flag in `CLRKDLaneHead` that adds a parallel `iou_head` MLP and emits `iou_logits` (B, P) alongside `cls_logits`.
- `loss.lane.w_iou_aux: 2.0` (+ `iou_aux_loss_type: qfl`, `iou_aux_qfl_gamma: 2.0`, `iou_aux_target_pow: 1.0`) — new `FusionLossConfig` fields. Continuous LineIoU regression supervises the new iou head.
- `eval.decoded_score_source: cls_x_iou` — new in `LaneF1DecodedMetric`. Ranks priors by `sigmoid(cls_logits) * sigmoid(iou_logits)` instead of plain `sigmoid(cls_logits)`.
- Training script also reports a `_cls_only` companion metric so we can attribute the gain (cls alone vs cls × iou).

The cls head keeps binary matched_existence + ASL (NB40 setting). The new iou head provides the deterministic continuous signal that NB40 was missing. At inference, cls answers "is this prior a match?" while iou answers "if so, how good is the geometry?". The matching-instability that has been collapsing the binary cls task no longer determines the decode rank because iou regression has a per-prior target that does not depend on dynamic-k's batch-level decisions.

This is the **architecture-side** attack on the cls collapse.

## Code changes (NEW since NB40)

Files modified:

- **[`stage2/fusion/lane_head.py`](fusion/lane_head.py) `CLRKDLaneHead`**:
  - New `dual_score: bool = False` constructor flag. When `True`, builds an `iou_head` MLP (Linear → ReLU → Linear) operating on the geometry feature pathway.
  - `_predict()` now returns 5-tuple `(cls_logits, params, offsets, coord_pred, iou_logits)`; `iou_logits` is `None` when `dual_score=False` (back-compat).
  - Stage outputs and the final out dict propagate `iou_logits` when present.
- **[`stage2/fusion/experiment_factory.py`](fusion/experiment_factory.py)**: passes `dual_score` from `lane_cfg` into `CLRKDLaneHead`.
- **[`stage2/fusion/losses.py`](fusion/losses.py) `FusionLossConfig`**:
  - New fields: `w_iou_aux: float = 0.0`, `iou_aux_loss_type: str = 'bce'`, `iou_aux_qfl_gamma: float = 2.0`, `iou_aux_target_pow: float = 1.0`. Defaults preserve back-compat.
- **[`stage2/fusion/losses.py`](fusion/losses.py) `FusionLaneLoss._forward_single_stage`**:
  - Section 8 (new): when `w_iou_aux > 0` and `pred['iou_logits']` exists, compute `_compute_lineiou_target` once (with `iou_aux_target_pow`) and apply BCE or QFL on it. Adds `w_iou_aux * iou_aux_loss` to total. Reports `lane/iou_aux` in the components dict.
- **[`stage2/metrics/lane_f1_decoded.py`](metrics/lane_f1_decoded.py)**:
  - New `score_source='cls_x_iou'`. Reads `pred['iou_logits']` and uses `sigmoid(cls) * sigmoid(iou)` as the per-prior ranking score. Falls back to plain cls scoring if `iou_logits` is missing.
- **[`stage2/scripts/train_joint_model_experiment.py`](scripts/train_joint_model_experiment.py) `evaluate(...)`**:
  - Reads `eval.decoded_score_source` (default `'cls'`) and passes it to the primary `LaneF1DecodedMetric`.
  - When primary source is `cls_x_iou`, also instantiates a `_cls_only` companion `LaneF1DecodedMetric` so we can attribute the gain in the metric report.

All changes are gated by config flags and back-compat with every prior experiment (NB35 default `cls_separate_path=False`, NB40 default `dual_score=False`, etc.).

## Pass criteria

### Exp2LL (NB41) at epoch 20

- **`val/lane/decoded_f1 ≥ 0.15`** — 3 × NB40's 0.043. The continuous QFL target is the standard published recipe (RTMDet/GFL) for fixing the imbalance Exp2K could not.
- **`val/num_pred_lanes < 400`** — pred_lanes drop from 1536 to ~ 50/image because the QFL weighting forces a clear high-IoU vs low-IoU separation.
- **`val/matched_line_iou ≥ 0.40`** — preserves NB40's geometric champion.
- **`val/lane/decoded_oracle_f1 ≥ 0.35`** — oracle ceiling stays high (proves the regression target did not distort geometry training).
- `train/grad_cosine_epoch_mean` logged each epoch.

### Exp2MM (NB42) at epoch 20

- **`val/lane/decoded_f1 ≥ 0.20`** — 4-5 × NB40. The dual-score head should close most of the gap to oracle.
- **`val/lane/decoded_f1 − val/lane/decoded_cls_only_f1 ≥ 0.10`** — iou re-ranking is doing real work; if this gap is < 0.05, the cls head dominates and dual scoring is wasted parameters.
- **`val/num_pred_lanes < 400`**.
- **`val/matched_line_iou ≥ 0.40`**.
- **`val/lane/iou_aux`** decreases monotonically over training (training-side health check on the new head).

If both pass, the next move is Exp2NN: combine Exp2LL's continuous regression cls target WITH Exp2MM's dual head (cls_target_type='lineiou_regression' + dual_score=True) — the cls head emits geometry-aware ranks AND the iou head supervises with a deterministic target, so any remaining noise in either signal gets canceled in the product.

If only Exp2LL passes: continue with the lineiou_regression family; the dual head was unnecessary capacity.

If only Exp2MM passes: keep binary cls but the dual scoring is the unlock; build Exp2NN around `dual_score=True` + lighter cls supervision.

If neither passes: the impasse really is the joint-training conflict (grad_cosine going negative in late epochs of NB40). The next move is detection-side: lower `lambda_det` to 0.5 OR add gradient surgery (PCGrad / GradVac) to disentangle the shared-backbone gradients.

## Run order

1. **Exp2LL first** (NB41, `exp36_*.yaml`). ~ 30 min wall-clock. Pure config change; if it works, it is the simplest fix.
2. **Exp2MM second** (NB42, `exp37_*.yaml`). ~ 30 min wall-clock with new dual-score MLP. Architectural change; runs only if Exp2LL didn't fully solve it (or to confirm the gain is from the dual scoring not just the regression target).

## Independence

Exp2LL is a pure config change, no risk to any prior experiment. Exp2MM adds an optional `dual_score=True` branch in `CLRKDLaneHead` that is back-compat by default. The new `w_iou_aux`, `iou_aux_loss_type` etc. default to off in `FusionLossConfig`. The new `decoded_score_source` defaults to `'cls'`. Re-running NB35 / NB40 / any earlier notebook will produce identical numbers to before.

## Local smoke

`python -m compileall yolop_vehicle_lane/stage2/{fusion,metrics,scripts}` passes after all edits. The two new YAMLs round-trip through `yaml.safe_load`. The two new notebooks round-trip through `json.load`.
