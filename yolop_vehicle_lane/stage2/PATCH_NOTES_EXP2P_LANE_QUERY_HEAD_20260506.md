# Stage 2 Exp2P — DETR-style lane queries (paradigm change)

## Why this patch exists

The Exp2O oracle diagnostic (NB20, exp15) was the smoking gun. With exactly the same model and training as Exp2N, val ran two metrics side by side:

| metric                          | epoch 1 | epoch 5 | epoch 10 |
|---------------------------------|--------:|--------:|---------:|
| `lane/decoded_f1` (cls-rank)    | 0.0147  | 0.0107  | 0.0082   |
| **`lane/decoded_oracle_f1`** (perfect rank) | **0.291**   | **0.305**   | **0.273**    |
| gap                             | 20×     | 29×     | 33×      |

**Oracle ranking lifts F1 by 30×.** The geometry already supports `decoded_f1 ~ 0.30` — the cls head is throwing away virtually all of that quality. Across **eight** prior-based cls-rescue attempts (Exp2G focal, H ASL, I OHEM, J separate path, K BCE-on-IoU, L QFL, M sqrt, N decoded eval) the cls always collapsed into the same all-priors-cluster-at-one-value attractor.

The 30× gap is not a tunable. The bottleneck is **the prior-based design itself**: 192 priors × 5 GT lanes per image = 2.6 % positive rate, dynamic-k matching that reshuffles which priors are positive each batch, and a shared per-prior feature pulled toward "what's a good curve" rather than "is *this* prior special". No focal/ASL/OHEM/cascade-pathway loss tweak fixed it.

Exp2P abandons the prior-based head entirely. K=12 learned lane queries through a 3-layer transformer decoder, Hungarian-matched 1-to-1 to GT, binary focal cls. The DETR / RMT-PPAD / GANet pattern.

## What changed

1. **[stage2/fusion/lane_head.py](yolop_vehicle_lane/stage2/fusion/lane_head.py)** — added `LaneQueryHead` module:
   - K=12 learnable query content + positional embeddings (`nn.Embedding(num_queries, embed_dim)` × 2).
   - Per-scale `Conv2d(c_i, embed_dim, 1)` + a learned scale embedding so flattened multi-scale features can be distinguished by feature level after concatenation.
   - `nn.TransformerDecoder` with 3 `TransformerDecoderLayer`s (`d_model=128`, `nhead=8`, `dim_feedforward=512`, `dropout=0.1`, `batch_first=True`).
   - Per-query output heads: `cls_head` (Linear → 1d), `param_head` (3-layer MLP → 4d for start_y/start_x/theta/length, sigmoid'd), `offset_head` (3-layer MLP → 72d row offsets, scaled by 0.05·tanh).
   - Curve coordinates computed deterministically from params + offsets (same parameterization as `CLRKDLaneHead._prior_curves_full`).
   - Output dict has the same keys as `CLRKDLaneHead` so `FusionLaneLoss`, `LaneF1DecodedMetric`, and the rest of the pipeline work without modification (substituting K=12 for P=192).

2. **[stage2/fusion/experiment_factory.py](yolop_vehicle_lane/stage2/fusion/experiment_factory.py)** — recognises `lane_head.type` aliases `'query'`, `'lane_query'`, `'detr'`, `'exp2p'` and constructs `LaneQueryHead` from `num_queries`, `num_decoder_layers`, `num_heads`, `dim_feedforward`, `dropout` config knobs.

3. **[stage2/configs/exp16_rmt_gca_lane_query_head_joint.yaml](yolop_vehicle_lane/stage2/configs/exp16_rmt_gca_lane_query_head_joint.yaml)** — new config:
   - `lane_head.type: query`, `num_queries: 12`, `num_decoder_layers: 3`, `num_heads: 8`, `dim_feedforward: 512`, `dropout: 0.1`.
   - `dataset.max_lanes: 12` (matches num_queries; dataset re-pads on the fly).
   - `loss.lane.cls_target_type: matched_existence` (back to binary; the IoU-regression formulation was a workaround for the prior-based ranking failure that no longer applies).
   - `loss.lane.cls_loss_type: focal` (standard binary focal, not ASL/QFL — class imbalance is now mild).
   - `loss.lane.lane_assigner: hungarian` (one-to-one, deterministic).
   - `loss.lane.match_cost_cls: 2.0` (raised from 1.0 — with only 12 queries, cls signal in the matching cost matters more).
   - `loss.lane.line_iou_radius: 0.02` (slightly relaxed from 0.015 — queries at random init can be far from GT, and a wider band gives non-zero LineIoU early).
   - `loss.lane.aux_stage_loss_weight: 0` (no per-stage refinement — single decoder).
   - `loss.lane.w_mask: 0` (no mask aux — query head doesn't naturally produce a per-pixel lane mask).
   - `eval.decoded_top_k: 6` (with K=12 queries, top-6 is a reasonable ceiling, comparable to typical lane count + slack).
   - `eval.decoded_oracle_enabled: true` (keep the diagnostic running).
   - Phases: `head_warmup: 2` epochs, then `full_finetune`. Skipping `adapter_warmup` since GCA gates have been frozen at 0.500 across every Exp2 run anyway (separate `EXP_GCA_DIAGNOSTIC` follow-up).

4. **[stage2/notebooks/stage2_notebook_21_exp2p_lane_query_head_joint.ipynb](yolop_vehicle_lane/stage2/notebooks/stage2_notebook_21_exp2p_lane_query_head_joint.ipynb)** — new notebook mirroring NB19/NB20 structure. Markdown spells out the diagnosis, the architectural reset, and the pass/fail signals. Smoke first, debug-mode default.

5. **[stage2/notebooks/stage2_notebook_08_*.ipynb](yolop_vehicle_lane/stage2/notebooks/stage2_notebook_08_joint_eval_visualization_and_profile.ipynb)** — exp16 entries added to cells 3, 5, 7.

## Local smoke

`python -m compileall yolop_vehicle_lane/stage2/fusion yolop_vehicle_lane/stage2/metrics yolop_vehicle_lane/stage2/scripts` passes. Forward+backward smoke runs from inside NB21 cell 3 on Colab.

## Pass criteria for Exp2P

After 10 short10 epochs:

- **`val/lane/decoded_f1 ≥ 0.10` at epoch 10** — a meaningful lane-line F1, breaking the 0.01 floor. Anything ≥ 0.20 is strong; ≥ 0.40 would mean the architectural reset alone closed most of the gap to CLRKDNet.
- **`pos_score_mean − neg_score_mean ≥ 0.30`** at epoch 10. With 5/12 queries positive and 7/12 negative, score separation should be clear; in prior-based runs this stayed at +0.001.
- `val/lane_exist_best_f1 ≥ 0.50` (same metric, 12 outputs makes the cls task substantially easier).
- `val/lane/decoded_oracle_f1` is reported alongside `decoded_f1`. If both rise together, query design works AND geometry holds.
- Geometry holds (looser than Exp2M since queries need more epochs to converge curves from scratch): `val/matched_line_iou ≥ 0.30`, `val/lane_point_mae ≤ 0.40`.
- `train/lane/cls` strictly decreases over training (in prior-based runs cls was flat from the start).

## Failure criteria → next ablation

- `decoded_f1 < 0.05` at epoch 10 → queries also can't learn in 10 epochs. Two plausible causes:
  - Need denoising queries / auxiliary decoder losses (RT-DETR convention) for faster convergence. Add per-decoder-layer cls + reg supervision.
  - Need 30+ epochs. Run **Exp2Q = Exp2P + extended training** to test the duration hypothesis.
- Geometry collapses (`matched_iou < 0.20`) → the (start_y, start_x, theta, length) + row offsets parameterization is too restrictive for queries trying to learn arbitrary BDD lane shapes from scratch. Switch to Bezier curve params or learned anchor offsets.
- Score separation tiny (< 0.10) and best_f1 < 0.20 → queries are failing the same way priors did. The bottleneck is something deeper (data quality, geometry resolution, training duration). Move toward higher input resolution (720×1280) and 30+ epoch training.

## Open follow-ups (intentionally not bundled)

- **GCA gates frozen at 0.500** (`EXP_GCA_DIAGNOSTIC`) — investigated separately.
- **Detection mAP50 ≈ 0.003** (`EXP_DET_MAP50_DEBUG`) — DETR head needs more decoder layers + denoising.
- **CLRKDNet KD** — the project's namesake feature, still unimplemented. If Exp2P succeeds, KD distillation from a CLRKDNet teacher becomes the natural next big move.

## Run order

1. Do not rerun NB00.
2. Open NB21, keep `DEBUG_MODE = True`, run smoke + debug.
3. If debug succeeds, set `DEBUG_MODE = False` and run short10.
4. Run NB08 to plot Exp2K through Exp2P side-by-side, with both `lane/decoded_f1` and `lane/decoded_oracle_f1` curves.
5. If Exp2P breaks the 0.01 cls floor, the prior-vs-query paradigm question is settled and we proceed to Stage 3 (extended training, video profiling, KD from teacher). If not, move on to Exp2Q (Exp2P + 30+ epochs / denoising / aux losses).
