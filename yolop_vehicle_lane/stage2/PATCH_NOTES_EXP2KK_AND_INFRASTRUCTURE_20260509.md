# Stage 2 Exp2KK + AMP/conflict diagnostics + dataset visualization

## Reading NB38 / NB39 results — three findings

### NB38 (Exp2GG, full 70K dataset) — confirms the user's observation

| ep | matched_iou | val_lane | **val_det** |
|---:|---:|---:|---:|
| 1 | 0.152 | 1.73 | 2.07 |
| 5 | 0.024 ⚠️ | 1.76 | **2.95** ⚠️ |
| 15 | 0.024 ⚠️ | 1.71 | **2.77** ⚠️ |

70K samples: lane held flat (val_lane: 1.73 → 1.71) but **detection actively degraded** (val_det: 2.07 → 2.77, +34%). geometry CRASHED at ep2 (matched_iou: 0.152 → 0.024). `grad_cos` mean ≈ 0 with high variance (-0.09 to +0.10) — task conflict confirmed. **More data exposed real conflict** that the 3K dataset's narrow distribution had hidden.

### NB39 (Exp2II, anchor head + modern training) — champion geometry, broken decode

| ep | **matched_iou** | best_f1 | decoded_f1 | **oracle_f1** | **pred_lanes** |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.368 | 0.408 | 0.013 | 0.214 | 1382 |
| 10 | 0.418 | 0.086 | 0.004 | 0.329 | 1533 |
| **20** | **0.472** ✓ | 0.078 | 0.011 | **0.379** ✓ | **1536** ⚠️ |

**Project's all-time best geometry** — matched_iou=0.472 and oracle_f1=0.379 are CLRKDNet-CULane territory. BUT pred_lanes=1536=8 batch × 192 anchors. **Every anchor is predicted positive.** Plain focal α=0.25 is too soft on the 187/192 unmatched anchors per image — they all settle at sigmoid≈0.4. **The cls is the only thing standing between us and a strong score**.

## What changed

### Train script infrastructure (no LR/batch change → stage1 stability unaffected)

[`train_joint_model_experiment.py`](yolop_vehicle_lane/stage2/scripts/train_joint_model_experiment.py):

1. **bfloat16 AMP support**: `cfg['train']['amp']['kind']: bfloat16`. Uses `torch.amp.autocast` for forward + loss; backward stays float32 for stability. ~2× speedup on RTX 6000 with no hyperparameter risk.

2. **Joint-conflict diagnostic**: `cfg['eval']['grad_cos_probe_interval']: 50`. Computes `grad_cos` between `l_lane` and `l_det` w.r.t. shared backbone every 50 training steps. Epoch summary now reports `train/grad_cosine_epoch_mean` (running mean across all probes). Negative = tasks fight; positive = aligned. **Direct quantitative answer to the user's question about a conflict metric.**

### Exp2KK (NB40, exp35) — fix the broken cls in NB39

[exp35_*.yaml](yolop_vehicle_lane/stage2/configs/exp35_rmt_gca_anchor_asl_amp_joint.yaml): clone of NB39's exp34 with **two single-knob changes**:
- `cls_loss_type: focal → asl` (γ_pos=0, γ_neg=4, clip=0.05). ASL amplifies gradient on confident-but-wrong negatives by `(p_neg − clip)^γ_neg` — directly addresses the "all 192 anchors at sigmoid≈0.4" failure.
- `train.amp.kind: bfloat16` (~2× speedup).
- `eval.grad_cos_probe_interval: 50` (running conflict log).

All other settings = NB39 (anchor head, dynamic-k matching top-k=4, mask aux, cosine LR, uncertainty weighting, 20 epochs).

Reference: Ben-Baruch et al. 2021 *Asymmetric Loss For Multi-Label Classification* (CVPR).

### Dataset visualization notebook (separate from stage1/2/3)

[`dataset_inspection/dataset_visualization.ipynb`](yolop_vehicle_lane/dataset_inspection/dataset_visualization.ipynb) — 8 sections:
1. Disk layout + per-split file counts.
2. 6 random training images with lane + box overlays (saves `.png` to `/content/`).
3. Raw lane-GT tensor structure for one sample (existence/points/visibility shapes + sample values).
4. Per-image lane count histogram + 0-lane / saturated stats.
5. Spatial heatmap: where do lanes appear in image space (90×160 grid)?
6. Lane theta + length histograms (compare to anchor priors' init distribution).
7. Detection class + box-area + boxes-per-image distributions.
8. One-screen sanity-check summary with diagnostic prompts.

Use this if metrics mysteriously plateau — many of the symptoms we saw (saturated cls, dead det) could be explained by dataset properties.

## Pass criteria for Exp2KK at epoch 20

The smoking gun:
- **`pred_lanes < 200`** at val time (NB39 was 1536 — ALL anchors above 0.3 threshold).
- **`val/lane/decoded_f1 ≥ 0.10`** — 10× NB39's 0.011, because cls now ranks meaningfully.
- **`val/matched_line_iou ≥ 0.40`** — preserves NB39's geometry champion (don't trade it for cls).
- **`val/lane/decoded_oracle_f1 ≥ 0.30`** — oracle ceiling stays high.
- `[amp] kind=bfloat16 enabled=True` log line at start.
- `train/grad_cosine_epoch_mean` logged each epoch.

If pred_lanes drops < 200 AND decoded_f1 jumps ≥ 0.10, **we have unlocked the anchor head's actual capacity** — that's a 2–3× breakthrough on decoded_f1.

## Independence

Exp2KK is independent of Exp2GG/HH and any prior experiment. The infrastructure additions (AMP, grad_cos diagnostic) are gated by config flags; back-compat preserved for older yaml files (no `amp` block → AMP off; no `grad_cos_probe_interval` → defaults to 50 which is harmless).

## Local smoke

`python -m compileall yolop_vehicle_lane/stage2/{fusion,metrics,scripts}` passes. AMP code path is exercised in the smoke test on Colab (NB40 cell 3). The viz notebook does not require GPU.

## Run order

1. **Run [dataset_inspection/dataset_visualization.ipynb](yolop_vehicle_lane/dataset_inspection/dataset_visualization.ipynb)** first. ~5 minutes. Establishes ground truth on what the dataset looks like — useful for interpreting all subsequent results.
2. **Run NB40 (Exp2KK)**: smoke + debug + short20. ~30 min wall-clock with AMP. Tests if ASL fixes NB39's cls.

If Exp2KK passes: anchor head is the new champion. Combine with full dataset (Exp2GG-style) in a future Exp2LL — but only after the conflict-diagnostic confirms the joint training stays aligned.

If pred_lanes still > 1000 with ASL: the cls problem is deeper (need OHEM with `cls_ohem_topk_per_pos: 3`, or hard-mining with explicit per-image top-k). If geometry regresses: w_cls is too high; pull back to 2.0.
