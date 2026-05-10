# Stage 2 Exp2GG + Exp2II — Full dataset + revisit the anchor architecture

## What NB36 / NB37 told us — the plateau is structural

NB36 (Exp2EE cosine LR, 20 epochs):
- **Cosine LR fix worked**: matched_iou stayed in 0.124-0.162 across all 20 epochs, no late-epoch collapse like NB34's. The `train_total: 2.90 → 2.04` decrease was monotonic.
- **decoded_f1 peak: 0.047 at epoch 19** — only +7% above Exp2Z's 0.044.

NB37 (Exp2FF KD + cosine LR, 20 epochs):
- **KD bug fix worked**: teacher loaded successfully, `val/lane/distill` was logged.
- **Geometry: matched_iou peaked at 0.166** (project's all-time best).
- **decoded_f1: 0.042 at epoch 20** — KD did NOT lift the decode metric. Self-distillation from Exp2Z provides no information beyond what the student would learn anyway (teacher and student saturate at the same architectural ceiling).

Trends across the entire Exp2 series on LaneQueryHead:

| metric | Exp2W | Exp2Z | Exp2EE | Exp2FF |
|---|---:|---:|---:|---:|
| matched_iou peak | 0.143 | 0.162 | 0.162 | **0.166** |
| oracle_f1 peak | 0.074 | 0.101 | 0.107 | 0.101 |
| decoded_f1 peak | 0.043 | 0.044 | **0.047** | 0.044 |

Six iterations on top of LaneQueryHead — each fixing a real DL pathology — moved decoded_f1 from 0.027 to 0.047 (+74%). **The plateau isn't a deep-learning settings problem anymore. It's structural.**

## The elephant in the room

I just confirmed `BDDJointCurveDataset.__init__` (line 262): when `limit > 0`, it slices `items[:limit]`. **Every Exp2 notebook has used `LIMIT_TRAIN = 3000`.** We've been training on the FIRST 3000 samples per epoch.

Compute budget comparison:
- CLRKDNet on CULane: 88K samples × 70 epochs = **6.2M sample-passes**
- Our Exp2 series: 3K samples × 20 epochs = **60K sample-passes**
- **We've been training with 100× less compute.**

Plus: we abandoned Exp2N's anchor-based architecture (CLRKDLaneHead, oracle_f1=0.27) because cls broke. But Exp2N also had no LR scheduler, no uncertainty weighting, no mask aux — all the things that made Exp2Z/EE/FF stable. **We never tried the anchor head WITH modern training tricks.**

## Two big moves

### Exp2GG (NB38, exp33) — Full BDD curve dataset

[exp33_*.yaml](yolop_vehicle_lane/stage2/configs/exp33_rmt_gca_mask_uncertainty_full_dataset_joint.yaml) — clone of Exp2EE config. Notebook command sets `LIMIT_TRAIN = 0` (the dataset's "use all items" sentinel), `LIMIT_VAL = 0`, `PRINT_EVERY = 50` (less verbose with ~10× more steps per epoch), `val_batches: 80`. Compute budget: ~10× slower epochs, ~2.5h for 15 epochs on Colab Pro.

**Tests data scarcity directly.** If decoded_f1 jumps to ≥0.10, training compute was the limit. If it plateaus at 0.05, architecture is the limit.

### Exp2II (NB39, exp34) — Anchor head + Exp2EE training tricks

[exp34_*.yaml](yolop_vehicle_lane/stage2/configs/exp34_rmt_gca_anchor_clrkd_modern_training_joint.yaml) — switches `lane_head.type: query → clrkd` (back to 192 anchors with ROI gather). Combines with all the Exp2EE/Z/W training fixes:
- Cosine LR + linear warmup (NB34's late-collapse fix)
- Uncertainty multi-task weighting (Exp2Z's stable joint balance)
- Mask auxiliary supervision (Exp2W's +65% lift)
- Dynamic-k matching with top-k=4 (CLRKDNet's published recipe)
- 20 epochs

**Tests whether anchor head's higher geometry ceiling (oracle_f1=0.27 in Exp2N) translates to decoded_f1 once cls supervision has the same training fixes that worked for query head.**

## Independence

Both run on the same dataset and architecture base. Independent of each other:
- Exp2GG: scales data
- Exp2II: scales architecture's geometric ceiling

## Pass criteria

### Exp2GG (NB38) at epoch 15
- `[dataset] split=train samples=N` reports N >> 3000 (likely 25K-35K).
- `val/lane/decoded_f1 ≥ 0.10` — 2× Exp2EE plateau.
- `val/matched_line_iou ≥ 0.20`.
- `val/lane/decoded_oracle_f1 ≥ 0.15`.
- No late-epoch collapse.

### Exp2II (NB39) at epoch 20
- `val/lane/decoded_oracle_f1 ≥ 0.20` — anchor architecture's high ceiling reappears.
- `val/lane_exist_best_f1 ≥ 0.65` — cls separation across 192 anchors works under modern training.
- `val/lane/decoded_f1 ≥ 0.07` — beats Exp2EE by ≥1.5× because anchor head's ceiling is higher.
- No late-epoch collapse.

## Decision tree

| Exp2GG | Exp2II | Interpretation |
|---|---|---|
| pass | pass | Data + architecture both contribute. Combine in Exp2JJ (anchor + full dataset). |
| pass | fail | Data scarcity was THE bottleneck. Train Exp2GG longer. |
| fail | pass | Anchor head + modern training is the right architecture. Scale data on top. |
| both fail | | True architectural ceiling. Move to bigger backbone (ResNet-50 ImageNet pretrained), higher resolution (480×800), or external CLRKDNet teacher KD. |

## Files

- New configs: [exp33](yolop_vehicle_lane/stage2/configs/exp33_rmt_gca_mask_uncertainty_full_dataset_joint.yaml), [exp34](yolop_vehicle_lane/stage2/configs/exp34_rmt_gca_anchor_clrkd_modern_training_joint.yaml).
- New notebooks: [NB38](yolop_vehicle_lane/stage2/notebooks/stage2_notebook_38_exp2gg_full_dataset_joint.ipynb), [NB39](yolop_vehicle_lane/stage2/notebooks/stage2_notebook_39_exp2ii_anchor_clrkd_modern_training_joint.ipynb).
- NB08 extended with both.
- `python -m compileall yolop_vehicle_lane/stage2/{fusion,metrics,scripts}` passes.

## Run order

Independent — run either or both:
1. **NB38 (Exp2GG)**: smoke + debug + full15. Tests the data hypothesis. **Highest expected impact** because we've been compute-starved by 100×.
2. **NB39 (Exp2II)**: smoke + debug + short20. Tests the architecture hypothesis. Cheap to run (3000 samples, 20 epochs).
