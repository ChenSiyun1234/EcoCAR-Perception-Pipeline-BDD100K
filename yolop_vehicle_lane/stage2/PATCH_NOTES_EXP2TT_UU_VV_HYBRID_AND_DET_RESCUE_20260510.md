# Stage 2 Exp2TT + Exp2UU + Exp2VV — combine the best of two architectures, rescue detection

## Reading NB46 / NB47 / NB48 — the breakthrough I missed

### NB46 (Exp2QQ — VFL on anchor head): VFL alone did NOT fix the cls collapse

| ep | matched_iou | oracle_f1 | decoded_f1 | pos / neg | gap |
|---:|---:|---:|---:|---:|---:|
| 1  | 0.333 | 0.198 | 0.011 | 0.160 / 0.160 | 0.000 |
| 10 | 0.391 | 0.259 | 0.012 | 0.170 / 0.167 | 0.003 |
| 20 | 0.495 | 0.397 | 0.036 | **0.197 / 0.197** | **0.000** |

**The closed-form math was correct but the VFL equilibrium just shifted to a different uniform value.** With dynamic-k matching, ANY loss formulation lands in a uniform-sigmoid equilibrium because the per-prior labels flicker batch-to-batch. The bug was deeper than the alpha constant — it's the matching scheme itself on the 192-anchor head.

### NB47 (Exp2RR — K=64 query head + Hungarian + VFL): the real breakthrough

| ep | matched_iou | oracle_f1 | decoded_f1 | pos / neg | **gap** | **val_lane_f1** | **val_lane_best_f1** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1  | 0.189 | 0.106 | 0.002 | 0.365 / 0.314 | **0.051** | **0.225** | **0.245** |
| 9  | 0.219 | 0.149 | 0.012 | 0.391 / 0.291 | **0.099** | **0.291** | **0.318** |
| 18 | 0.255 | 0.191 | 0.024 | 0.378 / 0.290 | 0.088 | 0.261 | **0.344** |
| 20 | 0.274 | 0.206 | 0.019 | 0.353 / 0.290 | 0.063 | **0.246** | 0.283 |

**The K=64 query head with Hungarian 1-to-1 matching is the FIRST experiment in 11 attempts to break the cls collapse.** pos-neg gap reached 0.099 (10× the historical 0.01); val_lane_f1 hit 0.246 (vs 0.0 across NB39-46); val_lane_best_f1 hit 0.344 (vs 0.05 across NB39-46). But **geometry is weak** (matched_iou 0.27 vs anchor's 0.54) because K=64 free-form queries can't densely cover lane space without the geometric prior init that the 192-anchor design provides.

### NB48 (Exp2SS — anchor + VFL + 70K full data, 6 epochs): geometry record + det broke

| ep | matched_iou | oracle_f1 | decoded_f1 | val_lane_f1 | gap | **val_det** | **val_map50** |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1  | 0.374 | 0.242 | 0.012 | 0.000 | 0.012 | **3.36** | 0.0009 |
| 4  | 0.526 | 0.426 | 0.048 | 0.038 | 0.008 | 3.26 | 0.0006 |
| 5  | 0.542 | 0.451 | 0.051 | 0.046 | 0.013 | 3.16 | 0.0004 |
| 6  | **0.544** | **0.467** | **0.050** | 0.070 | 0.015 | **3.16** | **0.000** |

**Geometry hit project records** (matched_iou=0.544, oracle_f1=0.467, decoded_f1=0.050 all best ever on the anchor head). But **detection completely failed**: val_det stalled at 3.16 (vs 1.97 in NB47 limit=3000), val_map50 ≈ 0. The full-dataset lane gradient is overwhelming the shared backbone and starving the det branch. Joint conflict, exposed.

## The unifying picture

| head | matching | cls discriminates? | geometry quality |
|---|---|---|---|
| 192 anchors (dynamic-k) | ~4 priors/GT, batch-flickering | **NO** (gap ≤ 0.015 across 11 experiments) | **Excellent** (matched_iou up to 0.544) |
| K=64 queries (Hungarian 1-to-1) | 1 prior/GT, deterministic | **YES** (gap up to 0.099) | Weak (matched_iou 0.27) |

The two architectures have **orthogonal strengths**. The cls collapse on the anchor head wasn't a loss-function bug — it's the matching scheme. dynamic-k labels the same prior as positive in some batches and negative in others, so the cls converges to a uniform sigmoid as the only stable equilibrium. Hungarian 1-to-1 gives every query a deterministic label, so the cls actually trains.

## Three new experiments

### Exp2TT (NB49, exp44) — HybridPriorQueryHead

Combine the two architectures in one model.
- **Stage 1**: 192 anchor priors with dynamic-k (NB48 geometry champion). Supervised by `stage1_aux_loss_weight=1.0`.
- **Stage 2**: K=12 queries that cross-attend to stage 1's per-prior features AND the spatial map. Hungarian 1-to-1 + VFL (NB47 cls champion).
- Inference: rank by stage 2's cls scores; curves come from stage 2 (which started from stage 1's geometry-rich pool).

The `HybridPriorQueryHead` module already exists from NB22 (Exp2Q, which failed because stage 1 had no direct supervision). The `stage1_aux_loss_weight` infrastructure was added later for Exp2T and is exactly what we need here.

### Exp2UU (NB50, exp45) — K=64 queries + full 70K dataset

Validate NB47 at scale. Single change vs NB47: 6 epochs at full 70K data instead of 20 epochs at limit=3000. NB48 showed full-data scaling lifts the anchor head's matched_iou by 0.02 (0.525 → 0.544); a similar lift on the K=64 query head from 0.27 should land it near 0.40 — close enough that combined with the working cls (val_lane_f1 0.25+) decoded_f1 should hit 0.10+.

### Exp2VV (NB51, exp46) — anchor + VFL + full data + det rescue

Fix the joint conflict NB48 exposed. Single set of changes vs NB48:
- `lambda_det: 1.0 → 2.0`
- `lambda_lane: 1.0 → 0.7`
- `use_uncertainty: true → false`

Kendall uncertainty learns weights from loss magnitudes; at full data scale lane loss decreases faster than det, so the learned weight migrates AWAY from det and accelerates the imbalance. Fixed weighting with explicit det boost is the standard fallback (Sener & Koltun 2018).

## Code changes — none needed

All three configs use existing infrastructure. `HybridPriorQueryHead` is dispatchable via `lane_head.type: hybrid` in experiment_factory; `stage1_aux_loss_weight` is in FusionLossConfig; VFL was just added in NB46-48; full-dataset support is via the train script's existing `--limit-train` (omitted = full).

`python -m compileall yolop_vehicle_lane/stage2/{fusion,metrics,scripts}` passes. Three new YAMLs round-trip; three new notebooks round-trip; NB08 patched in 3 cells.

## Pass criteria

### Exp2TT at epoch 20
- `val/matched_line_iou ≥ 0.45` (inherit NB48 stage 1 geometry, slight regression OK at limit=3000)
- `pos_score - neg_score ≥ 0.10` on stage 2 (inherit NB47 cls)
- `val/lane_f1 ≥ 0.20` and `val/lane_best_f1 ≥ 0.30`
- `val/lane/decoded_f1 ≥ 0.10` (2× NB48)

### Exp2UU at epoch 6
- `val/matched_line_iou ≥ 0.40` (1.5× NB47)
- `val/lane_f1 ≥ 0.30` and `val/lane_best_f1 ≥ 0.40`
- `val/lane/decoded_f1 ≥ 0.10` (5× NB47)

### Exp2VV at epoch 6
- `val_det ≤ 2.2` and `val/det/map50 ≥ 0.005` — detection actually trains
- `val/matched_line_iou ≥ 0.50` (preserve geometry; small regression from NB48's 0.544 is acceptable)
- `train/grad_cosine_epoch_mean` stays ≥ 0 across most epochs

## Run order — fully independent

1. **Exp2TT first** (NB49, ~30 min). Highest expected impact: architectural combination of two known wins.
2. **Exp2UU** (NB50, ~60-80 min). Validates NB47's discovery at scale.
3. **Exp2VV** (NB51, ~60-80 min). Fixes a known regression (det broke in NB48).

None depend on each other or on Exp2C. The decision tree:

| Exp2TT result | What it means | Next move |
|---|---|---|
| pass (decoded_f1 ≥ 0.10, gap ≥ 0.10) | Hybrid is the new champion | Combine with NB48's full-data scale in Exp2WW |
| partial (gap ≥ 0.10 but matched_iou ≤ 0.30) | Stage 2 distorts geometry | Tune `stage1_aux_loss_weight=2.0`, `aux_stage_loss_weight=0.0` |
| fail (gap ≤ 0.05) | K=12 too few queries on this dataset | K=24 + same hybrid |

## Bigger picture

We now have three independent paths that have each shown promise:
- **Geometry path**: NB48 (anchor + VFL + 70K data) gives matched_iou = 0.544
- **Cls path**: NB47 (K=64 query + Hungarian + VFL) gives val_lane_f1 = 0.246
- **Hybrid path**: Exp2TT (this) — should give both

If Exp2TT pass criteria are all met, this is the first genuinely deployable joint model. If detection rescue (Exp2VV) also works, the project is on track for Stage 3 deployment within a week.

## Why the user's depth/PGI/AdaBins suggestions are good but premature

The user mentioned GELAN-PGI backbone, AdaBins depth bins, NaviBridger spatial conditioning, STAL small-target awareness, PCGrad gradient surgery, and DDAD as a depth dataset. These are all sound suggestions for the depth head once it's added. But:

1. **The lane and det heads are not stable yet.** Adding a third task to an unstable two-task system would amplify all pathologies (the user explicitly said "stabilize lane and det first").
2. **PCGrad is a logical Exp2WW** if Exp2VV's static rebalancing fails — but Exp2VV is the simpler fix to try first.
3. **The architectural breakthrough we just discovered (hybrid = anchor geometry + query cls) is independent of backbone choice** and applies to GELAN/HGNetv2/RMT/anything else.
4. **Pretrained backbone init** would help, but matching the YOLO26 weights to our compressed RMT-GCA wrapper is a 1-2 day engineering job and should come AFTER the loss/architecture choices stabilize.

The depth head should integrate with whichever lane head wins (anchor / query / hybrid). Once Exp2TT/UU/VV give a clear answer, depth becomes a clean extension.
