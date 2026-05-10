# Stage 2 Exp2J — Separate cls feature pathway

## Why this patch exists

Exp2I short10 (NB14) confirmed that the cls head has the **capacity** to separate matched from unmatched priors, but only when backbone features are stable. Per-epoch from the saved NB14 output:

| epoch | phase            | val/lane_exist_best_f1 | pos − neg | val/matched_line_iou |
|------:|------------------|-----------------------:|----------:|---------------------:|
| 1     | head_warmup      | 0.050                  | +0.001    | 0.340                |
| **2** | **adapter_warmup** | **0.5355**           | **+0.163**| 0.279                |
| 3     | adapter_warmup   | 0.219                  | +0.034    | 0.356                |
| 4     | full_finetune    | 0.115                  | +0.012    | 0.387                |
| 5     | full_finetune    | 0.071                  | +0.002    | 0.388                |
| 10    | full_finetune    | **0.057**              | +0.001    | 0.391                |

Epoch 2 (first epoch of `adapter_warmup`) hit `best_f1 = 0.535` and `pos − neg = +0.163` — *above* the project's pass target on separation. The signal collapsed the moment `full_finetune` opened the full backbone, and by epoch 10 it was back to Exp2G/H levels.

`train/lane/cls_pos` rose 0.083 → 0.13 across the run, mirroring the val regression — the model is *genuinely losing* cls capability as training progresses, not just generalizing poorly.

### Mechanism

The cls head reads `per_lane`, the same 128-d feature also consumed by `param_head` and `offset_head` for curve geometry. Geometry losses (LineIoU, reg, xytl, smooth) dominate the gradient flow into `per_lane`. As training progresses, `per_lane` becomes specialized for "what curve goes through here" — a representation that is *similar* across geometrically-similar priors near the same GT lane. The cls task asks "is *this specific* prior matched?", which cuts a *different* boundary: dynamic-k matching picks one or two priors as positive among many similar ones. Once `per_lane` is curve-specialized, that boundary is unrepresentable, and OHEM + ASL + prior encoder cannot recover it because the bottleneck is *the feature gradient flow*, not the loss formulation.

The Exp2I patch notes named exactly this as the next step: *"if best_f1 < 0.20 at epoch 10, the per_lane representation itself is the bottleneck, not the loss; move to a separate cls feature pathway."*

## What changed

1. [stage2/fusion/lane_head.py](stage2/fusion/lane_head.py) — extended `CLRKDLaneHead` with a parallel cls aggregator. New kwarg `cls_separate_path: bool = False`. When True, the head builds disjoint `scale_blocks_cls`, `scale_fusion_cls`, `fc_cls`, `fc_norm_cls`, `cross_attn_cls` modules of identical shape to the geometry counterparts. In `forward`, per-scale ROI samples are computed once per stage via `_grid_sample` and *shared* between branches (so each prior sees the same image evidence along its curve from both branches; no extra sampling cost). The two branches' aggregator parameters are disjoint, so cls gradients update only `scale_blocks_cls + scale_fusion_cls + fc_cls + fc_norm_cls + cross_attn_cls + cls_head + prior_embed_encoder`, and geometry gradients update only the original geometry counterparts + `param_head + offset_head`. Both branches still backprop into the shared backbone via the bilinear sample positions, but each task only feels its own loss when shaping its branch's interpretation of those samples. `lane_class_head` and the optional `mask_decoder` continue to read the geometry pathway's `per_lane` so existing consumers see the same representation as before.

2. [stage2/fusion/experiment_factory.py](stage2/fusion/experiment_factory.py) — pass `cls_separate_path` through; accept `'exp2j'` as a `lane_head.type` alias.

3. [stage2/configs/exp10_rmt_gca_clrkd_separate_cls_path_joint.yaml](stage2/configs/exp10_rmt_gca_clrkd_separate_cls_path_joint.yaml) — new config. Diff vs `exp09`:
   - `cls_separate_path: true`.
   - `w_iou: 1.5 → 2.0` (recover Exp2G's geometry weight; with cls now isolated, no need to soften geometry).
   - All other settings (OHEM topk_per_pos=4 + min_topk=32, prior_embed_encoder_dim=64, ASL gamma_pos=0/gamma_neg=4, dynamic-k, 3 stages, RMT+GCA, λ_min=0.5) identical to Exp2I.

4. [stage2/notebooks/stage2_notebook_15_exp2j_clrkd_separate_cls_path_joint.ipynb](stage2/notebooks/stage2_notebook_15_exp2j_clrkd_separate_cls_path_joint.ipynb) — new notebook mirroring NB14. Smoke first, debug-mode default, markdown documenting the persistence test.

5. [stage2/notebooks/stage2_notebook_08_joint_eval_visualization_and_profile.ipynb](stage2/notebooks/stage2_notebook_08_joint_eval_visualization_and_profile.ipynb) — exp10 entries added to cells 3 (eval items), 5 (metrics list), 7 (video profile candidates, Exp2J preferred over Exp2I/H/G).

## Local smoke

`python -m compileall yolop_vehicle_lane/stage2/fusion` passes. The torch forward+backward smoke runs from inside NB15 cell 4 on Colab (`smoke_test_joint_models.py exp10_*.yaml`). Expected smoke output: `OK exp10_*.yaml`, `lane_shape=(1, 16, 72, 2) det_shape=(1, 4, 4)`, healthy gate stats around 0.50.

## Cost

The doubled aggregator only spans the post-sample portion of the head: `scale_blocks` (per-scale Conv2d), `scale_fusion` (multi-scale Conv2d per stage), `fc` (Linear), `cross_attn`. Per-scale `_grid_sample` is computed once and shared. End-to-end overhead estimate: lane head forward + backward roughly +50 %; total step roughly +10 % to +15 % since the shared backbone, AIFI, and DETR head dominate compute.

## Pass criteria for Exp2J

After 10 short10 epochs:

- **The persistence test (the entire point)**: `val/lane_exist_best_f1` ≥ 0.40 by epoch 5 and ≥ 0.65 by epoch 10. Most importantly: it does NOT collapse from epoch 3 onward as Exp2I did (0.535 → 0.219 → 0.115 → 0.071 → ... → 0.057).
- `val/lane_exist_pos_score_mean − val/lane_exist_neg_score_mean ≥ 0.15` at epoch 10 with no decay over training (Exp2I went +0.163 → +0.001).
- `val/lane/cls_pos` strictly *decreases* over training (Exp2I rose 0.083 → 0.13 — the regression we are fixing).
- `pred_lanes / batch` drops below 500 (Exp2G/H/I were stuck at ~1500 = all 192·8 priors above thr=0.3).
- **Geometry holds or improves**: `val/lane_point_mae ≤ 0.34` and `val/matched_line_iou ≥ 0.40` at epoch 10. With `w_iou` raised back to 2.0 we expect matched_iou close to or exceeding Exp2G's 0.428.

## Failure criteria → next ablation

- best_f1 still below 0.20 at epoch 10 → the bilinear sample positions themselves are the bottleneck (cls aggregator's input ROI sample locations are pulled by geometry through curve params). Next: **stop-gradient from cls into curve params** so the cls aggregator reads features at geometry-driven positions only. Or replace cls with the thresholded-LineIoU-regression alternative.
- Geometry regresses sharply (`point_mae > 0.36` or `matched_iou < 0.30`) → the cls aggregator is stealing capacity from the backbone via its grad path. Reduce parameter sharing: keep `scale_blocks` shared (early per-scale conv) and only diverge at `scale_fusion + cross_attn`.
- Forward pass cost or memory increases ≥ 30 % vs Exp2I → the doubled aggregator is too expensive for the budget. Reduce `roi_mid_channels` from 48 to 32 in the cls path only (would require an extra config knob).

## Run order

1. Do not rerun NB00.
2. Open NB15, keep `DEBUG_MODE = True`, run smoke + debug.
3. If debug succeeds, set `DEBUG_MODE = False` and run short10.
4. Run NB08 to plot Exp2G / Exp2H / Exp2I / Exp2J side-by-side per epoch.
5. If pass criteria are met, retire Exp2H/I as cls-rescue baselines and continue toward:
   - **Lane Decode + NMS** (still missing inference pipeline; CLRKDNet vendor has reference code).
   - **CLRKD KD distillation** (project name's core promise still unfulfilled; `loss.w_distill: 0.0` everywhere).
   - **GCA gate diagnostic** (gates stuck at 0.500 across all Exp2 runs; suspect `lambda_gate_reg=0.001` too small or task adapter inputs too similar).
   - **Detection mAP50 ≈ 0.003 investigation** (separate from lane head; DETR head needs more decoder layers / denoising / proper IoU-format adapter).
