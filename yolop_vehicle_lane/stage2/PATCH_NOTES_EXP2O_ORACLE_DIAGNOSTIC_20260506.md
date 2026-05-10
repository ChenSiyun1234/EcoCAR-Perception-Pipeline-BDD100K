# Stage 2 Exp2O — Oracle-IoU diagnostic for the lane-F1 ceiling

## Why this patch exists

Exp2N (NB19) confirmed the proper top-K decode + lane-NMS pipeline is wired correctly: `val/lane/decoded_pred_count = 31.1` per batch (out of 32 max for top_k=4 × batch_size=8), both precision and recall non-zero, decoder produces real lane outputs. **But `decoded_f1` is genuinely tiny** at 1-2 % across all 10 epochs. Geometry is the project's all-time best (`matched_iou = 0.424`, `point_mae = 0.320` at epoch 10), yet lane-F1 is far below CLRKDNet's published ~80 % on CULane.

The interpretation is straightforward: `matched_iou = 0.42` means the priors *the matcher picks* (highest LineIoU with each GT) average 0.42 IoU. That's the geometry capacity. `decoded_f1 = 0.01` means the priors *the cls head picks* (top-K by sigmoid score) almost never have IoU > 0.5 with any GT. The geometry is good for *some* priors; the cls head doesn't know which.

Before throwing more compute at fixing cls (extended training, hybrid head, etc.), Exp2O answers a single decisive question: **what's the lane-F1 ceiling our geometry can support if cls were perfect?**

The answer determines the next move:
- ceiling ≥ 0.40 → cls is the only blocker; CLRKDNet-comparable F1 is reachable.
- ceiling 0.10-0.40 → geometry is partial; either lower the F1@IoU threshold (0.3 is also a standard) or invest in tighter geometry.
- ceiling < 0.10 → geometry doesn't generalize; the matched_iou number is over-optimized in-loop and doesn't translate to GT-aligned predictions.

## What changed

1. **[stage2/fusion/lane_decode.py](yolop_vehicle_lane/stage2/fusion/lane_decode.py)** — `decode_top_k_lanes` now accepts `scores_override: Optional[torch.Tensor]`. When provided, the (B, P) tensor is used as the per-prior ranking score in place of `sigmoid(cls_logits)`. Range-checked, not re-sigmoided. Default behaviour unchanged.

2. **[stage2/metrics/lane_f1_decoded.py](yolop_vehicle_lane/stage2/metrics/lane_f1_decoded.py)** — `LaneF1DecodedMetric` adds two parameters:
   - `score_source: 'cls' | 'oracle_iou'` (default `'cls'`).
   - `key_suffix: str = ''` (e.g. `'_oracle'`) appended to all output keys so two instances can run side by side without colliding.
   - When `'oracle_iou'`, the metric calls `_compute_lineiou_target(coord_pred, points_gt, vis, radius)` from `losses.py` and passes the result as `scores_override` into the decoder.

3. **[stage2/scripts/train_joint_model_experiment.py](yolop_vehicle_lane/stage2/scripts/train_joint_model_experiment.py)** — `evaluate(...)` gates a second `LaneF1DecodedMetric` instance (`score_source='oracle_iou'`, `key_suffix='_oracle'`) on `eval.decoded_oracle_enabled`. Default: false (back-compat). When true, both metrics run on the same val batch; the new keys (`lane/decoded_oracle_f1`, `_precision`, `_recall`, etc.) merge into the per-epoch metrics dict.

4. **`epoch_summary` line** now includes `val_oracle_f1`, `val_oracle_p`, `val_oracle_r` alongside the existing `val_decoded_*` fields. Both report 0.0 when oracle is disabled (clean back-compat).

5. **[stage2/scripts/analyze_stage2_trends.py](yolop_vehicle_lane/stage2/scripts/analyze_stage2_trends.py)** and **[plot_stage2_metrics.py](yolop_vehicle_lane/stage2/scripts/plot_stage2_metrics.py)** — `lane/decoded_oracle_f1`, `_precision`, `_recall` added to the `KEYS` list so the new metric appears in trend CSVs and per-epoch plots.

6. **[stage2/configs/exp15_rmt_gca_clrkd_oracle_diagnostic_joint.yaml](yolop_vehicle_lane/stage2/configs/exp15_rmt_gca_clrkd_oracle_diagnostic_joint.yaml)** — clone of Exp2N's `exp14` plus `eval.decoded_oracle_enabled: true`. Different `output_tar` path. Model + losses identical to Exp2M/N.

7. **[stage2/notebooks/stage2_notebook_20_exp2o_oracle_iou_diagnostic_joint.ipynb](yolop_vehicle_lane/stage2/notebooks/stage2_notebook_20_exp2o_oracle_iou_diagnostic_joint.ipynb)** — new notebook mirroring NB19. Smoke first, debug-mode default. Markdown spells out the three outcomes (A/B/C) and the next experiment for each.

8. **[stage2/notebooks/stage2_notebook_08_*.ipynb](yolop_vehicle_lane/stage2/notebooks/stage2_notebook_08_joint_eval_visualization_and_profile.ipynb)** — `exp15` entries added to cells 3, 5, 7.

## Local smoke

`python -m compileall yolop_vehicle_lane/stage2/fusion yolop_vehicle_lane/stage2/metrics yolop_vehicle_lane/stage2/scripts` passes. Forward+backward smoke runs from inside NB20 cell 3 on Colab.

## Pass criteria

After 10 short10 epochs:

- **`val/lane/decoded_oracle_pred_count` > 0** for every epoch (sanity: oracle decoder works).
- **`val/lane/decoded_oracle_f1`** is reported and is *meaningfully larger* than `val/lane/decoded_f1`. The gap between them is the cls bottleneck.
- Geometry holds (Exp2M baseline): `point_mae ≤ 0.34`, `matched_line_iou ≥ 0.40`. Training is unchanged from Exp2N so this should be free.

## Decision tree based on `decoded_oracle_f1` at epoch 10

- **≥ 0.40** → **Exp2P = extended training** of Exp2M's recipe to 30+ epochs (CLRKDNet trains for 100s, our 10 epochs are an order of magnitude short). Or **Exp2Q = hybrid head** with a separate binary existence score and an IoU regression score, supervised independently (binary on matched assignment, regression on continuous IoU target). Either path is now justified by the oracle ceiling.
- **0.10 ≤ x < 0.40** → tighten the geometry. Try `decoded_lane_iou_threshold: 0.30` for the practical metric (still standard in some lane benchmarks like LLAMAS at IoU=0.3). Also consider denser priors (`num_priors: 256`) or tighter `line_iou_radius`.
- **< 0.10** → geometry doesn't generalize. The matched_iou is in-loop over-optimization; even oracle ranking can't pick GT-aligned predictions. Need to rethink prior parameterization (e.g., direct row-anchor regression) or loss formulation (LineIoU with stricter radius).

## Open follow-ups (intentionally not bundled)

- **GCA gates frozen at 0.500** across every Exp2 run (`EXP_GCA_DIAGNOSTIC`).
- **Detection mAP50 ≈ 0.003** (`EXP_DET_MAP50_DEBUG`).
- **Multi-IoU-threshold lane-F1** (F1@30 / F1@50 / F1@75) is what most lane-detection papers report. Adding the alt-threshold loop is a small follow-up if Outcome B applies.

## Run order

1. Do not rerun NB00.
2. Open NB20, keep `DEBUG_MODE = True`, run smoke + debug.
3. If debug succeeds, set `DEBUG_MODE = False` and run short10.
4. Run NB08 to plot Exp2K / Exp2L / Exp2M / Exp2N / Exp2O side-by-side, with both `lane/decoded_f1` and `lane/decoded_oracle_f1` curves.
5. The oracle-vs-cls gap at epoch 10 dictates the next experiment per the decision tree above.
