# Stage 2 Exp2I — OHEM hard-negative mining + stronger prior encoder

## Why this patch exists

Exp2H short10 (NB13) was a lateral move on classification — it was the failure mode the previous patch explicitly named. Per-epoch from the saved NB13 output:

| epoch | val/lane_point_mae | val/matched_line_iou | val/lane_exist_best_f1 | pos − neg | val/lane/cls_pos | val/lane/cls_neg | λ_runtime |
|------:|-------------------:|---------------------:|-----------------------:|----------:|-----------------:|-----------------:|----------:|
| 1     | 0.3371             | 0.3576               | **0.245**              | +0.074    | 0.083            | 0.037            | 1.00      |
| 2     | 0.3359             | 0.3327               | 0.245                  | **+0.122**| 0.077            | 0.042            | 0.68      |
| 5     | 0.3272             | 0.3894               | 0.094                  | +0.027    | 0.088            | 0.042            | 0.50      |
| 10    | 0.3285             | 0.4018               | **0.083**              | +0.016    | **0.133**        | 0.042            | 0.50      |

Versus Exp2G epoch 10 (point_mae 0.3244, matched_iou 0.4285, best_f1 0.075, pos−neg +0.004): tiny cls gain (+0.008 best_f1), tiny geometry regression (−0.027 matched_iou). The ASL + prior-embedding fix produced a real *transient* signal at epoch 1-2 (best_f1=0.245, far above Exp2G's 0.168), then collapsed by epoch 10. The previous patch's failure-mode list explicitly said: "if best_f1 < 0.20, ASL is not enough; add OHEM hard-negative mining." That is the case.

### Mechanism behind the cls collapse

`val/lane/cls_pos` rose 0.083 → 0.133 while `pos_score_mean` fell 0.633 → 0.580 — positive predictions actively *degrade* over training. Score separation collapsed 7× from +0.122 (epoch 2) to +0.016 (epoch 10).

Dynamic-k matching produces a soft positive/negative boundary on geometrically-similar priors (multiple priors near the same GT lane have similar curves and are competed for matching, with the losers becoming negatives). Geometry losses train `per_lane` to encode "what curve goes through here" — similar across priors near the same GT — so the cls task ("is *this specific* prior matched") gets harder over time. The 1500 negatives' gradients dilute the 40 positives' representational pull through the shared `per_lane`. ASL alone cannot fix this because dilution happens through gradient flow into `per_lane`, not through loss weighting.

## What changed

1. [stage2/fusion/losses.py](stage2/fusion/losses.py) — OHEM in the cls path. New `FusionLossConfig` fields:
   - `cls_ohem_topk_per_pos: int = 0` (defaults off for back-compat).
   - `cls_ohem_min_topk: int = 32` (floor).
   - When `cls_ohem_topk_per_pos > 0`, the per-image negative cls loss is computed only on the top-K hardest unmatched priors, where `K = max(cls_ohem_min_topk, cls_ohem_topk_per_pos × num_pos_in_image)`. With ~5 GT lanes per image and topk_per_pos=4, that's ~32 hardest negatives per image instead of all ~187. Total negative gradient count drops from ~1500 to ~160 per batch — the positives' representational pull is no longer diluted.

2. [stage2/fusion/lane_head.py](stage2/fusion/lane_head.py) — stronger prior embedder. New `CLRKDLaneHead` kwarg `prior_embed_encoder_dim`. When `> 0`, the 3-d clamped prior embedding `(start_y, start_x, theta)` passes through `Linear(3, prior_embed_encoder_dim) + ReLU` before concatenation with `per_lane`. At init, the prior contributes `prior_embed_encoder_dim / (embed_dim + prior_embed_encoder_dim)` of the cls input — with `embed_dim=128, prior_embed_encoder_dim=64` that's ~33%, vs the ~2% ratio from raw 3-d concat in Exp2H. The cls head can rely much more heavily on the prior position as a hard discriminator.

3. [stage2/fusion/experiment_factory.py](stage2/fusion/experiment_factory.py) — pass the new kwarg through; accept `'exp2i'` as a `lane_head.type` alias.

4. [stage2/configs/exp09_rmt_gca_clrkd_ohem_prior_encoder_joint.yaml](stage2/configs/exp09_rmt_gca_clrkd_ohem_prior_encoder_joint.yaml) — new config. Diff vs `exp08`:
   - `cls_ohem_topk_per_pos: 4`, `cls_ohem_min_topk: 32`.
   - `prior_embed_encoder_dim: 64`.
   - `w_cls: 6.0 → 4.0` (Exp2H's bump didn't help and made geometry slightly worse; OHEM is the better lever).
   - `w_iou: 1.0 → 1.5` (recover some of the geometry weight; Exp2G had 2.0, Exp2H had 1.0, midpoint here).
   - All other settings identical to Exp2H.

5. [stage2/notebooks/stage2_notebook_14_exp2i_clrkd_ohem_prior_encoder_joint.ipynb](stage2/notebooks/stage2_notebook_14_exp2i_clrkd_ohem_prior_encoder_joint.ipynb) — new notebook mirroring NB13.

6. [stage2/notebooks/stage2_notebook_08_joint_eval_visualization_and_profile.ipynb](stage2/notebooks/stage2_notebook_08_joint_eval_visualization_and_profile.ipynb) — exp09 entries added to cells 3 (eval items), 5 (metrics list), 7 (video profile candidates, Exp2I preferred over Exp2H/G).

## Local smoke

`python -m compileall stage2/fusion` passes. The torch forward+backward smoke runs from inside NB14 cell 4 on Colab (`smoke_test_joint_models.py exp09_*.yaml`).

## Pass criteria for Exp2I

After 10 short10 epochs:

- **Persistence test (the key one)**: `val/lane_exist_best_f1` stays above 0.20 from epoch 1 onward and ends ≥ 0.65 at epoch 10. Exp2H peaked at 0.245 at epoch 1-2 then collapsed to 0.083 — Exp2I must hold the early signal.
- `val/lane_exist_pos_score_mean − val/lane_exist_neg_score_mean ≥ 0.15` at epoch 10 with no decay over training (Exp2H went +0.122 → +0.016).
- `val/lane/cls_pos` strictly **decreases** over training (Exp2H regressed 0.083 → 0.133). With OHEM concentrating gradient on hard cases, positives should improve, not degrade.
- `pred_lanes / batch` drops below 500 (Exp2G/H were stuck at ~1500 = nearly all 192·8 priors above thr=0.3).
- **Geometry holds**: `val/lane_point_mae ≤ 0.34` and `val/matched_line_iou ≥ 0.30` at epoch 10. With `w_iou` raised back to 1.5 we expect matched_iou closer to Exp2G's 0.43 than Exp2H's 0.40.

## Failure criteria → next ablation

- `val/lane_exist_best_f1 < 0.20` at epoch 10 → the per_lane representation itself is the bottleneck, not the loss. Move to a **separate cls feature pathway** (Option B): a parallel branch that consumes only ROI samples + prior embedding and is trained only by cls loss. Or replace the binary existence task with a direct LineIoU-against-nearest-GT regression head whose target is robust to matching reshuffles.
- Geometry regresses sharply (`point_mae > 0.36`) → `w_cls=4` still too high relative to `w_iou=1.5`. Pull `w_cls` to 2.5.
- `pred_lanes / batch ≈ 0` with recall collapse → OHEM too aggressive. Raise `cls_ohem_topk_per_pos` from 4 to 8 or `cls_ohem_min_topk` from 32 to 64.

## Run order

1. Do not rerun NB00.
2. Open NB14, keep `DEBUG_MODE = True`, run smoke + debug.
3. If debug succeeds, set `DEBUG_MODE = False` and run short10.
4. Run NB08 to plot Exp2G / Exp2H / Exp2I side-by-side.
5. If pass criteria are met, retire Exp2E/F/H as cls-rescue baselines and continue toward Lane Decode + NMS, KD (project name promise still unfulfilled), and the separate det/mAP50 ≈ 0.003 investigation.
