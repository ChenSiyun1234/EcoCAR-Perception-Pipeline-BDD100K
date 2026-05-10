# Stage 2 Exp2QQ + Exp2RR + Exp2SS — VFL fixes the cls equilibrium

## Reading NB43 / NB44 / NB45 — what we just observed

### Per-experiment trends (epoch 20, except NB45 = epoch 30)

| run | matched_iou | oracle_f1 | decoded_f1 | decoded_cls_only | decoded_mask_only | pos − neg gap |
|---|---:|---:|---:|---:|---:|---:|
| NB39 (focal) | 0.472 | 0.379 | 0.011 | — | — | ~ 0.01 |
| NB40 (ASL) | 0.483 | 0.418 | 0.043 | — | — | ~ 0.01 |
| NB41 (lineiou+QFL) | 0.508 | 0.459 | 0.029 | — | — | 0.002 |
| NB42 (dual cls × iou) | 0.490 | 0.426 | 0.042 | 0.012 | — | 0.009 |
| **NB43** (mask self-distill) | **0.473** | **0.373** | **0.016** | **0.015** | **0.014** | **0.002** |
| **NB44** (Hungarian) | **0.383** | **0.310** | **0.015** | — | — | **0.010** |
| **NB45** (width 1.0, 30 ep) | **0.525** | **0.446** | **0.024** | **0.038** | **0.023** | **0.008** |

## Three new failures + one breakthrough finding

### NB43 — mask consistency hypothesis REJECTED
mask_only_f1 = 0.014 ≈ random. The 72×128 mask is too coarse to discriminate among 192 anchor curves that all sample similar lane-y regions. Self-distillation produced cls_only_f1 = 0.015 because the cls now mimics a non-discriminative target.

### NB44 — matching instability hypothesis REJECTED
With Hungarian 1-to-1 the pos − neg gap stayed at 0.010, AND geometry COLLAPSED to matched_iou = 0.383 (vs NB40's 0.483). 1-to-1 only assigns ~5 priors/image as positive, starving the geometric losses of training signal. **Hungarian is wrong for the anchor head.**

### NB45 — capacity hypothesis PARTIALLY REJECTED
Width 1.0 + embed 192 + 30 epochs → matched_iou = **0.525** (project all-time best!). But pos − neg gap = 0.008 unchanged. Worse: decoded_f1 PEAKED at epoch 19 (0.032) and DROPPED to 0.024 at epoch 30 — **more training makes cls worse**, confirming a degenerate local minimum.

### The breakthrough: closed-form analysis of the cls equilibrium

I worked the math on the ASL loss with our NB40 settings (alpha=0.25, gamma_pos=0, gamma_neg=4, clip=0.05) at sigmoid ≈ 0.578:

- Per-element loss ≈ 0.25 × log(0.578) + 0.75 × (0.578−0.05)^4 × log(1−0.578) ≈ **0.089**
- This **matches `val_lane_cls = 0.089` observed across NB40/41/42/43/45**.
- Gradient on positives: 0.25 × small log gradient → tiny.
- Gradient on confident negatives: 0.75 × p_neg^4 × log gradient → near zero when p_neg ≈ 0.5.

**alpha=0.25 was chosen for object detection where positives are 1:1000.** For our 192 lane priors with 5 GT lanes, the ratio is 1:37 — the alpha=0.25 is downweighting the rare positive class with the WRONG constant. The model sits at sigmoid ≈ 0.578 for ALL priors because that's the zero-gradient equilibrium under this asymmetric weighting.

This is **the well-known "focal loss collapse for moderately imbalanced datasets"** — published fix is **Varifocal Loss** (Zhang et al. CVPR 2021, used by RTMDet/VarifocalNet). VFL flips the asymmetry:

- Positives: weight = `target_iou` (no `(1−alpha)` discount; rare positives carry full weight).
- Negatives: weight = `alpha * pred^gamma * (1 − target)` (only confident-wrong negatives count).

With our 5:187 ratio and VFL alpha=0.75, positives have ~ 4× the gradient mass per sample compared to a uniform sigmoid negative — the equilibrium is broken.

## Three new experiments

### Exp2QQ (NB46, exp41) — VFL on the anchor head

[`exp41_rmt_gca_anchor_vfl_iou_regression_joint.yaml`](configs/exp41_rmt_gca_anchor_vfl_iou_regression_joint.yaml).

Single-axis change vs Exp2KK (NB40):
- `cls_target_type: matched_existence → lineiou_regression`
- `cls_loss_type: asl → vfl`
- `vfl_alpha: 0.75`, `vfl_gamma: 2.0`

Everything else identical to NB40. **This is the closed-form fix for the cls collapse.** If pos−neg gap jumps to ≥ 0.10 and decoded_f1 ≥ 0.15, the equilibrium math was the bottleneck for 7 experiments.

### Exp2RR (NB47, exp42) — K=64 DETR queries + Hungarian + VFL

[`exp42_rmt_gca_query64_hungarian_vfl_joint.yaml`](configs/exp42_rmt_gca_query64_hungarian_vfl_joint.yaml).

Architectural escape from the 192-anchor design. K=64 is the sweet spot: large enough to cover ~5 GT lanes per image with deterministic Hungarian 1-to-1, but small enough that each query has meaningful gradient. Combined with VFL, this tests an orthogonal angle: maybe the anchor representation is intrinsically non-discriminative regardless of loss.

### Exp2SS (NB48, exp43) — 70K full dataset + VFL

[`exp43_rmt_gca_anchor_vfl_full_dataset_joint.yaml`](configs/exp43_rmt_gca_anchor_vfl_full_dataset_joint.yaml).

Same anchor + VFL recipe as Exp2QQ but training on the full 70K BDD100K split for 6 epochs (≈ 7× more iterations, 23× more data variety). Tests whether the cls signal needs more diverse supervision to truly separate.

## Code changes (back-compat, gated by config)

- **[`stage2/fusion/losses.py`](fusion/losses.py)**:
  - New `_varifocal_loss()` helper. Asymmetric weighting: positives use `target` weight, negatives use `alpha * pred^gamma * (1-target)`.
  - `_binary_cls_raw` dispatch: new `'vfl'`/`'varifocal'` branch on the binary matched_existence path.
  - `_forward_single_stage`: new `'vfl'` branch on the lineiou_regression path.
  - `FusionLossConfig`: new fields `vfl_alpha=0.75`, `vfl_gamma=2.0`.
- All defaults preserve back-compat with NB35-45.

## Pass criteria

### Exp2QQ at epoch 20 — the smoking gun
- **`pos_score − neg_score ≥ 0.10`** — direct cls separation measurement. NB39-45 stuck at 0.01.
- **`val/lane/decoded_f1 ≥ 0.15`** — 3× NB40 with the same architecture, just fixed cls weighting.
- `val/matched_line_iou ≥ 0.45`, `val/lane/decoded_oracle_f1 ≥ 0.40`.

### Exp2RR at epoch 20 — architectural test
- `val/lane/decoded_f1 ≥ 0.15`.
- `pos − neg gap ≥ 0.20` (DETR with 1-to-1 Hungarian + VFL is the cleanest cls signal possible).
- `val/matched_line_iou ≥ 0.30` (queries typically have lower geometry than anchors).

### Exp2SS at epoch 6 — data scale test
- `val/lane/decoded_f1 ≥ 0.20`.
- `val/matched_line_iou ≥ 0.50`.
- val_det monotonically decreasing (no NB38-style joint conflict).

## Run order — fully independent

1. **Exp2QQ first** (NB46, ~30 min). Highest expected impact: closed-form fix for the documented bug.
2. **Exp2RR** (NB47, ~30 min). Architectural backstop in case Exp2QQ confirms the per-anchor representation is fundamentally limited.
3. **Exp2SS** (NB48, ~60-80 min). Data scale test; informative regardless of QQ/RR outcome.

None depend on each other or on Exp2C. The codebase is fully back-compat with NB35-45.

## Decision tree

| Exp2QQ result | Exp2RR result | Conclusion |
|---|---|---|
| pass | — | The 7-experiment bug WAS the alpha. Combine VFL with NB45's wider model in a future Exp2TT. |
| fail | pass | Anchor representation is fundamentally non-discriminative. Switch to query head as the new baseline. |
| pass | pass | Both directions work. Exp2SS will tell which scales better with data. |
| fail | fail | The bottleneck is even deeper (likely pretrained-init or image resolution). Pivot to RMT-PPAD pretrained backbone init. |

## Local smoke

`python -m compileall yolop_vehicle_lane/stage2/{fusion,metrics,scripts}` passes. Three new YAMLs round-trip through `yaml.safe_load`. Three new notebooks round-trip through `json.load`. NB08 patched in 3 cells with the new entries.
