# Stage 2 Exp2Y + Exp2Z + Exp2AA — Fix the lambda oscillation + scale up

## What the user observed and the explanation

While inspecting NB28 (Exp2W) the user noticed lane loss appearing to oscillate epoch-by-epoch. After tracing the step-level logs, the cause is clear and the fix is direct.

### The data

For Exp2W, lambda_lane at step 1 of each epoch (where grad_norm calibration runs):

| epoch | lambda_lane | total | train_lane |
|---:|---:|---:|---:|
| 1 | 1.000 | 7.77 | 3.10 (warmup, geom_scale=0.5) |
| 2 | 1.000 | 5.93 | 3.71 (warmup, geom_scale=1.0 -- ramp jump) |
| 3 | **0.500** | 3.95 | 3.47 |
| 4 | 0.500 | 3.83 | 3.35 |
| 5 | 0.500 | 3.83 | 3.33 |
| 6 | **1.851** | **7.97** | 3.22 |
| 7 | **2.000** | **8.63** | 3.18 |
| 8 | 1.261 | 6.12 | 3.25 |
| 9 | 1.516 | 6.52 | 3.08 |
| 10 | 1.332 | 6.59 | 3.32 |

`train_lane` itself decreases monotonically (3.71 → 3.18 from epoch 2 to 9). It's `lambda_lane` swinging **4× between epochs** that creates the visible "down/up" sawtooth in `total = det + lambda * lane`.

### Root cause

[`losses.py compute_grad_norm_ratio`](yolop_vehicle_lane/stage2/fusion/losses.py): once per epoch, at step 1, we compute

```python
lambda = (||grad_det|| / ||grad_lane||).clamp(lambda_min=0.5, lambda_max=2.0)
```

then *lock that value for the entire epoch*. As training progresses, the relative gradient magnitudes change, so lambda changes per epoch. Two failure modes:

1. **Runaway feedback loop**: when lambda grows, lane gets weighted more, lane gradients get larger, lane converges faster, grad_lane shrinks, **next epoch's ratio shoots up further**. This is GradNorm's classic instability mode (the original paper acknowledges it and prescribes EMA-smoothing).
2. **Per-epoch quantization**: the calibration runs once per 375 steps. Between recalibrations, lambda is fixed. So you see step changes at every epoch boundary regardless of underlying gradient dynamics.

### Secondary contribution

`geometry_warmup_epochs: 2` doubles the geometry loss weights between epoch 1 and epoch 2 (geom_scale 0.5 → 1.0). This is intentional warmup but contributes a ~1.0 jump to `train_lane` at the boundary. Reducing to 1 epoch shrinks this.

---

## The fix — three independent treatments

### Exp2Y (NB30, exp25) — lock lambda + extended training

Direct fix:
- `lambda_mode: fixed`, `lambda_lane: 1.0`, `lambda_min/max: 1.0` — no recalibration, no oscillation.
- `geometry_warmup_epochs: 2 → 1` — shrink the warmup ramp jump.
- `end_epoch: 10 → 15` — with stable lambda, more time should reward.

All other Exp2W settings (mask_aux=true, w_mask=2.0, query head, RMT+GCA backbone) preserved.

### Exp2Z (NB31, exp26) — Kendall uncertainty weighting

Theoretically grounded fix: `loss.use_uncertainty: true` enables `UncertaintyMultiTaskLoss` (already in losses.py, never used). The model learns log-variance parameters σ_lane and σ_det. Joint loss becomes:

$$L_{\text{total}} = \frac{1}{2\sigma_{\text{lane}}^2} L_{\text{lane}} + \log\sigma_{\text{lane}} + \frac{1}{2\sigma_{\text{det}}^2} L_{\text{det}} + \log\sigma_{\text{det}}$$

Sigmas are learnable parameters; `log(σ)` term prevents trivial σ → ∞ solution. **Monotonic by construction; oscillation is structurally impossible.**

Reference: Kendall, Gal, Cipolla 2018 *Multi-Task Learning Using Uncertainty to Weigh Losses* (CVPR).

### Exp2AA (NB32, exp27) — mask + higher resolution + extended training

Tests whether resolution itself is the bottleneck (parallel to the loss-balancing fix):
- `image_size: 384×640 → 480×800` (1.5× more pixels for thin lane lines)
- `aux_mask_size: 72×128 → 90×160` (matches new aspect)
- `batch_size: 8 → 4` (memory budget)
- `lambda_mode: fixed` (Exp2Y's stability fix bundled in)
- `end_epoch: 10 → 15`

CLRKDNet trains at 590×1640 on CULane; we've been at 384×640. At base resolution, lane lines are 1-2 pixels wide; at 480×800 they're 1.5× wider, giving the backbone more signal to localize.

---

## Independence + comparison logic

| Exp2Y (fixed λ) | Exp2Z (uncertainty) | Exp2AA (high res) | Interpretation |
|---|---|---|---|
| **>0.07** | similar | similar | λ stability was the issue; either weighting scheme fine |
| **>0.07** | fails | similar | uncertainty weighting has its own pathology in this regime |
| similar to W | similar | **>0.10** | resolution was the real bottleneck |
| all fail | | | impasse is something deeper (KD from teacher, or fundamental data limit) |

Three experiments × independent hypotheses = decisive resolution of the next-step question.

## Files

- New configs: [exp25](yolop_vehicle_lane/stage2/configs/exp25_rmt_gca_mask_fixed_lambda_long_joint.yaml), [exp26](yolop_vehicle_lane/stage2/configs/exp26_rmt_gca_mask_uncertainty_weighting_joint.yaml), [exp27](yolop_vehicle_lane/stage2/configs/exp27_rmt_gca_mask_high_resolution_joint.yaml).
- New notebooks: [NB30](yolop_vehicle_lane/stage2/notebooks/stage2_notebook_30_exp2y_mask_fixed_lambda_long_joint.ipynb), [NB31](yolop_vehicle_lane/stage2/notebooks/stage2_notebook_31_exp2z_mask_uncertainty_weighting_joint.ipynb), [NB32](yolop_vehicle_lane/stage2/notebooks/stage2_notebook_32_exp2aa_mask_high_resolution_joint.ipynb).
- NB08 extended with all three.
- `python -m compileall yolop_vehicle_lane/stage2/{fusion,metrics,scripts}` passes.

## Run order

Independent. Run any combination:

1. **NB30 (Exp2Y)**: smoke + debug + short15. The minimal-change stability fix.
2. **NB31 (Exp2Z)**: smoke + debug + short15. The principled stability fix.
3. **NB32 (Exp2AA)**: smoke + debug + short15. The resolution test.

Pass criteria across the three: **decoded_f1 ≥ 0.07 in any of them** would beat Exp2W's 0.043 substantially. If at least one breaks 0.10, that's the path to Stage 3 (video profiling, KD from teacher, etc).

If all three plateau at the Exp2W level (~0.04), the bottleneck is genuinely outside what these treatments can reach, and the next move is **CLRKDNet teacher KD** — the project's namesake feature, never implemented.
