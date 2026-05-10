# Stage 2 Exp2N — Lane decode + lane-NMS + threshold-adaptive lane-line F1

## Why this patch exists

Across seven cls-supervision ablations (Exp2G/H/I/J/K/L/M), `val/lane/clrkd_style_f1` reported **0** in every single experiment. While auditing the metric I found the cause:

- [`metrics/original_metric_adapters.py:138`](yolop_vehicle_lane/stage2/metrics/original_metric_adapters.py): `CLRKDStyleLaneMetric` uses a hardcoded `score_threshold = 0.5`.
- Across every Exp2 run, the model's max sigmoid score has been ~ 0.15 — well below 0.5. The metric's `pred_exist = score >= self.score_threshold` is therefore all-False, so `tp = 0` and `f1 = 0` *by construction*.
- The metric also reads `matched_target` (post-matching alignment) instead of the original `target['existence']` / `target['points']`, so even with a fixed threshold it measures per-slot quality, not lane-line F1 the way CLRKDNet's published numbers measure.

**Our headline lane-F1 number has been an evaluator bug, not a training failure.** Exp2N stops chasing the cls-formulation rabbit hole and builds the proper evaluation pipeline so we can finally see real numbers.

The model is unchanged from Exp2M (sqrt rescaling on the LineIoU regression target — geometry champion at matched_iou ≈ 0.42, point_mae ≈ 0.32). What changes is the val-time decode + metric.

## What changed

1. **New: [`stage2/fusion/lane_decode.py`](yolop_vehicle_lane/stage2/fusion/lane_decode.py)** — pure-PyTorch top-K + lane-NMS:
   - `decode_top_k_lanes(pred, top_k, score_threshold=0.0)`: per-image top-K by descending sigmoid score. **No threshold floor by default** — ranking is what matters when the score distribution is compressed (which has been the case all along).
   - `lane_nms(decoded, line_iou_threshold)`: greedy by descending score; suppress later items whose pairwise LineIoU exceeds the threshold. Reuses the existing `_line_iou_1d` from `stage2/fusion/losses.py` so semantics match training-time loss exactly.

2. **New: [`stage2/metrics/lane_f1_decoded.py`](yolop_vehicle_lane/stage2/metrics/lane_f1_decoded.py)** — `LaneF1DecodedMetric`:
   - Calls `decode_top_k_lanes` then `lane_nms`.
   - Greedy assigns surviving predictions to GT lanes by descending pairwise LineIoU (avoiding one prediction satisfying multiple GTs).
   - Reads the **original** `target['existence']` / `target['points']` — *not* `matched_target` — so it measures prediction-vs-GT the way CLRKDNet's published F1 does.
   - Returns `lane/decoded_f1`, `lane/decoded_precision`, `lane/decoded_recall`, `lane/decoded_tp`, `lane/decoded_fp`, `lane/decoded_fn`, `lane/decoded_pred_count`, `lane/decoded_avg_score`, `lane/decoded_top_k_used`.

3. **Modified: [`stage2/scripts/train_joint_model_experiment.py`](yolop_vehicle_lane/stage2/scripts/train_joint_model_experiment.py)**:
   - `evaluate(...)` now accepts `eval_cfg` and instantiates `LaneF1DecodedMetric` with `decoded_top_k`, `decoded_lane_iou_threshold`, `decoded_lane_nms_iou`, `decoded_score_threshold` from the config.
   - Both metrics run in the val loop; the new metric reads `lane_targets` (original), the old metric reads `matched_lane_targets` (back-compat).
   - `epoch_summary` line now includes `val_decoded_f1`, `val_decoded_p`, `val_decoded_r`, `decoded_pred`.

4. **Modified: [`stage2/scripts/evaluate_joint_model.py`](yolop_vehicle_lane/stage2/scripts/evaluate_joint_model.py)** — pass `eval_cfg=cfg.get('eval', {})` into the shared `evaluate()` function.

5. **Modified: [`stage2/scripts/analyze_stage2_trends.py`](yolop_vehicle_lane/stage2/scripts/analyze_stage2_trends.py)** and [`plot_stage2_metrics.py`](yolop_vehicle_lane/stage2/scripts/plot_stage2_metrics.py) — added `lane/decoded_f1`, `decoded_precision`, `decoded_recall`, `decoded_pred_count`, `decoded_avg_score` to the KEYS list so per-experiment trend CSVs and plots include the new metric.

6. **New: [`stage2/configs/exp14_rmt_gca_clrkd_decoded_eval_joint.yaml`](yolop_vehicle_lane/stage2/configs/exp14_rmt_gca_clrkd_decoded_eval_joint.yaml)** — identical model + losses to Exp2M (`exp13`). Adds `eval` block with `decoded_top_k: 4`, `decoded_lane_iou_threshold: 0.5`, `decoded_lane_nms_iou: 0.5`, `decoded_score_threshold: 0.0`. Different `output_tar` so it doesn't collide with Exp2M's tar.

7. **New: [`stage2/notebooks/stage2_notebook_19_exp2n_decoded_lane_f1_joint.ipynb`](yolop_vehicle_lane/stage2/notebooks/stage2_notebook_19_exp2n_decoded_lane_f1_joint.ipynb)** — mirrors NB17/NB18 structure. Smoke first, debug-mode default, markdown explains it's an *evaluator change*, not a training-recipe change.

8. **Modified: [`stage2/notebooks/stage2_notebook_08_*.ipynb`](yolop_vehicle_lane/stage2/notebooks/stage2_notebook_08_joint_eval_visualization_and_profile.ipynb)** — exp14 entries added to cells 3, 5, 7. The trend plot will now include `lane/decoded_f1` for every experiment whose metrics.json was produced *after* this patch.

## Why a new metric, not just lowering the existing threshold

Two reasons:

- The existing `CLRKDStyleLaneMetric` reads `matched_target`, not the original `target`. Even with a fixed threshold it would still measure per-slot quality, which is not the same as lane-line F1.
- Keeping the historical metric unchanged means past run JSONs remain comparable across experiments. The new metric is additive, not destructive.

Both metrics now run in val. The historical one is the back-compat number; the new one is the metric that actually compares to CLRKDNet.

## Local smoke

`python -m compileall yolop_vehicle_lane/stage2/fusion yolop_vehicle_lane/stage2/metrics yolop_vehicle_lane/stage2/scripts` passes.

`stage2/fusion/lane_decode.py` includes a self-check (`if __name__ == '__main__'`) that builds random tensors and asserts output shapes. Skipped during training; useful for documentation.

The torch forward+backward smoke runs from inside NB19 cell 3 on Colab.

## Pass criteria

After 10 short10 epochs of Exp2N:

- `val/lane/decoded_pred_count > 0` for every epoch (sanity — proves decode produces output).
- **`val/lane/decoded_f1 ≥ 0.10` at epoch 10.** First non-zero lane-F1 the project will ever have measured. ≥ 0.20 is a strong result; ≥ 0.40 would be remarkable.
- `val/lane/decoded_precision` and `val/lane/decoded_recall` both > 0 (one-sided result means top_k or threshold mis-configured).
- Geometry holds (Exp2M baseline): `val/matched_line_iou ≥ 0.40`, `val/lane_point_mae ≤ 0.34`. The new evaluator must not perturb training.

## Failure criteria → next ablation

- `decoded_f1 < 0.05` → cls scoring genuinely cannot rank priors. Move to: (a) extended training with same config (30+ epochs since CLRKDNet trains for 100s), or (b) hybrid head outputting both a binary existence score and an IoU-regression score, with the binary head trained on matched assignment and the regression head on the continuous IoU target.
- `decoded_pred_count = 0` consistently → bug in `lane_decode.py` (off-by-one in top-K or NMS suppressing everything). Re-derive against `external_repos/CLRKDNet-master/clrkd/utils/lane.py`.
- Geometry regresses sharply (`matched_iou < 0.30`) → instrumentation introduced a side effect (val OOM or batch-skipping). Inspect `train_joint_model_experiment.py:evaluate`.
- `decoded_f1` plateau-equal to `clrkd_style_f1`'s (basically-zero) value → the new metric is computing the same thing the old one does. Re-check: it must read the original `target`, not `matched_target`.

## Open follow-ups (intentionally not bundled into Exp2N)

- **GCA gates frozen at 0.500** across every Exp2 run. `EXP_GCA_DIAGNOSTIC` — investigate after Exp2N.
- **Detection mAP50 ≈ 0.003** across every Exp2 run. DETR head with 3 decoder layers + no denoising too shallow. `EXP_DET_MAP50_DEBUG`.
- **Cls supervision still weak** even with proper decoding. Recorded in failure-criteria above.

## Run order

1. Do not rerun NB00.
2. Open NB19, keep `DEBUG_MODE = True`, run smoke + debug.
3. If debug succeeds, set `DEBUG_MODE = False` and run short10.
4. Run NB08 to plot Exp2K / Exp2L / Exp2M / Exp2N side-by-side, with the new `lane/decoded_f1` curve included.
5. If `decoded_f1 ≥ 0.10` at epoch 10 with healthy precision/recall, this is the project's first comparable lane-F1 number against CLRKDNet. Continue toward Stage 3 (KD with a CLRKDNet teacher, video profiling, GCA diagnostic, detection-map debug).
