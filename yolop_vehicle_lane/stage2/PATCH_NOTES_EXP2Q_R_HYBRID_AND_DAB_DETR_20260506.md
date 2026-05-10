# Stage 2 Exp2Q + Exp2R — Hybrid two-stage and DAB-DETR anchor queries

## Why this patch exists

Across NB17–NB21 the project hit a paradigm tradeoff that no per-loss tuning fixed:

| | Exp2N (priors) | Exp2P (queries) |
|---|---:|---:|
| matched_iou | **0.42** | 0.13 |
| val_lane_f1 | 0.05 | **0.65** |
| decoded_f1 | 0.011 | 0.026 |
| oracle_f1 (geometry ceiling) | **0.27** | 0.07 |

Prior-based design has good geometry, broken cls. Query-based design has good cls (val_lane_f1=0.65 — first time the ranking task ever worked), broken geometry. Decoded F1 is the product of both, so neither paradigm alone gets close to CLRKDNet's 0.80 territory.

The "big move" the user asked for is **two parallel architectural experiments that combine the two wins** — neither is a parameter tweak; both are published recipes with high-leverage architectural changes. They are independent and can run in any order.

---

## Exp2Q — Hybrid prior-generator + query-refiner

Two-stage cascade pattern from **Sparse R-CNN (Sun 2021)** and **Mask2Former (Cheng 2022)**:

- **Stage 1**: the proven `CLRKDLaneHead` produces 192 prior-based curves with good geometry. Per-prior 128-d features and the merged spatial feature map are exposed in its output dict (new keys `per_prior_features`, `spatial_features`).
- **Stage 2**: K=12 learned queries pass through a 2-layer transformer decoder that cross-attends to the 192 per-prior features (memory). An optional second 1-layer decoder cross-attends to the spatial feature map for fine-grained context. Each query outputs `cls_logit + curve params + row offsets`.
- **Hungarian matches the K=12 outputs to GT**. Stage 1's cls is unused at inference; geometry losses still flow through stage 1's prior outputs end-to-end.

### Files

- [stage2/fusion/lane_head.py](yolop_vehicle_lane/stage2/fusion/lane_head.py) — added `HybridPriorQueryHead`. CLRKDLaneHead now also exposes `per_prior_features` and `spatial_features` so the wrapper can attend over them.
- [stage2/configs/exp17_rmt_gca_hybrid_prior_query_joint.yaml](yolop_vehicle_lane/stage2/configs/exp17_rmt_gca_hybrid_prior_query_joint.yaml) — new config (`lane_head.type: hybrid`).
- [stage2/notebooks/stage2_notebook_22_exp2q_hybrid_prior_query_joint.ipynb](yolop_vehicle_lane/stage2/notebooks/stage2_notebook_22_exp2q_hybrid_prior_query_joint.ipynb).

### Hypothesis

Stage 1 owns geometry (proven matched_iou=0.42 in Exp2N). Stage 2 owns the K=12 ranking task (proven val_lane_f1=0.65 in Exp2P). Combined → both 0.4+ matched_iou and 0.5+ best_f1, decoded_f1 ≥ 0.20.

### Pass criteria (epoch 10)

- `val/matched_line_iou ≥ 0.30` (looser than Exp2N's 0.42 since training is now joint).
- `val/lane_exist_best_f1 ≥ 0.40`.
- **`val/lane/decoded_f1 ≥ 0.10`**, ideally ≥ 0.20.
- `val/lane/decoded_oracle_f1 ≥ 0.20`.

### Failure modes

- decoded_f1 < 0.05 with geometry held → queries not picking from priors. Add ROI-gather conditioning so queries see prior curves directly.
- Geometry collapsed (matched_iou < 0.20) → stage 1 supervision insufficient. Add explicit auxiliary geometry loss on `stage1_*` keys exposed in the head output.

---

## Exp2R — Anchor-conditioned queries (DAB-DETR)

The DETR-convergence-acceleration recipe from **DAB-DETR (Liu 2022)**: each query has a learnable `(start_y, start_x, theta)` anchor whose sinusoidal embedding becomes the query's positional encoding. Queries get explicit positional bias from initialization, so they specialize spatially much faster — the standard fix for DETR's slow convergence in <50 epochs.

The `LaneQueryHeadAnchorDN` module also includes scaffolding for **DN-DETR (Li 2022)** denoising queries (perturb GT lanes, supervise to recover) but DN is **disabled by default** in this experiment (`dn_num_groups: 0`). Plumbing target tensors through `FusionModel.forward` is a separate change; we test pure DAB first to isolate the anchor-conditioning effect.

Param head outputs a delta added to the anchor (gentle refinement, scaled by 0.1) — iterative refinement pattern from DAB-DETR.

### Files

- [stage2/fusion/lane_head.py](yolop_vehicle_lane/stage2/fusion/lane_head.py) — added `LaneQueryHeadAnchorDN` + `_sinusoidal_pos_embed` helper + `_MLP` shared with `LaneQueryHead`/`HybridPriorQueryHead`.
- [stage2/configs/exp18_rmt_gca_query_anchor_dab_joint.yaml](yolop_vehicle_lane/stage2/configs/exp18_rmt_gca_query_anchor_dab_joint.yaml) — new config (`lane_head.type: query_anchor`).
- [stage2/notebooks/stage2_notebook_23_exp2r_query_anchor_dab_joint.ipynb](yolop_vehicle_lane/stage2/notebooks/stage2_notebook_23_exp2r_query_anchor_dab_joint.ipynb).

### Hypothesis

Anchors give queries the spatial inductive bias they were missing. Geometry should converge much faster than Exp2P (matched_iou 0.13 → ≥ 0.25); cls should match or exceed Exp2P's 0.65.

### Pass criteria (epoch 10)

- `val/matched_line_iou ≥ 0.25` (Exp2P baseline 0.13).
- `val/lane_exist_best_f1 ≥ 0.55` (Exp2P baseline 0.67).
- **`val/lane/decoded_f1 ≥ 0.08`** (3× Exp2P's 0.026).
- `val/lane/decoded_oracle_f1 ≥ 0.15`.

### Failure modes

- Geometry still collapses (matched_iou < 0.20) → anchors not enough; need real DN. Plumb targets through `FusionModel.forward(images, lane_targets)` and turn on `dn_num_groups: 4`.
- Cls regresses → anchor delta + sigmoid scheme over-constrains predictions. Loosen by removing the 0.1 multiplier on `param_delta`.

---

## Independence

- **Exp2Q** wraps the existing `CLRKDLaneHead` in a query refiner. Untouched: existing prior-based code path.
- **Exp2R** modifies `LaneQueryHead` (adds anchors). Untouched: existing query-only code path.

Either can succeed independently. Both share the same backbone, detection head, and eval infrastructure (decoded_f1 + oracle_f1).

## Local smoke

`python -m compileall yolop_vehicle_lane/stage2/{fusion,metrics,scripts}` passes. Forward+backward smokes run from inside NB22 / NB23 cell 3 on Colab.

## Other follow-ups (intentionally not bundled)

- **GCA gates frozen at 0.500** across every Exp2 run (`EXP_GCA_DIAGNOSTIC`).
- **Detection mAP50 ≈ 0.003** (`EXP_DET_MAP50_DEBUG`).
- **Real DN-DETR**: needs `FusionModel.forward(images, lane_targets)` plumbing. Trivial change, deferred until Exp2R confirms anchor conditioning works.
- **CLRKDNet KD**: still unimplemented. The next big move *after* Q/R.

## Run order

Independent experiments. Run either or both:

1. NB22 (Exp2Q): smoke + debug + short10. The hybrid is the bigger architectural move; if it works, it likely takes the project the furthest.
2. NB23 (Exp2R): smoke + debug + short10. Anchor conditioning is a smaller change but has stronger theoretical backing for DETR-style convergence.

Then run NB08 to plot Exp2K through Exp2R side-by-side. The winner becomes the parent for Stage 3 (extended training, video profiling, KD from CLRKDNet teacher).
