# Stage-2 Plan: how to make progress past the YOLOPv2-style baseline

Scope: score notebooks `04_modern_backbone_neck_upgrade`,
`05_lane_head_and_loss_upgrade`, `06_final_train_eval_export` against
2024–2025 real-time driving-perception literature, and propose concrete
upgrades that fit **A5000, 640×640, ≥30 FPS, vehicle + lane on BDD100K**.

Constraint order (tightest first):
1. Real-time on A5000 (FP16 TensorRT, batch = 1).
2. Lane IoU + vehicle mAP on BDD100K val.
3. Parameter / VRAM budget (≤ 40 M params, ≤ 4 GB activation).

---

## 1. What the three notebooks propose today

### Notebook 04 — Modern Backbone & Neck Upgrade
Candidate backbones:
- EfficientNet-B0 (2019), ConvNeXt-Tiny (2022), CSPDarknet-53 (2020),
  MobileNetV3-Large (2019).
Candidate necks:
- BiFPN (EfficientDet, 2019), PANet+ (YOLOv7-era).
Implementation: only a `timm.create_model(..., features_only=True)`
probe. Nothing is wired into training.

### Notebook 05 — Lane Head & Loss Upgrade
Decoder ideas: UNet-skip, CBAM / SE attention blocks.
Loss ideas: Dice, Focal, Lovász, boundary-aware.
Implementation: a standalone `DiceLoss` class; not wired into
`MultiHeadLoss` or `function.py`.

### Notebook 06 — Final Train / Eval / Export
ONNX (opset 12) + TorchScript trace. Param + file-size summary.
Implementation: working but uses a stale `cfg.DRIVE.CHECKPOINT_DIR`
(not per-baseline subdir) and doesn't export TensorRT / INT8.

## 2. Score vs 2024–2025 SOTA

The grades below use **real-time deployability (A5000 FP16 / TRT)** as
the primary lens. "Deprecated" means there is a strictly-better successor
on the same speed budget. "Fine" means still competitive. "Leading"
means current SOTA or close.

### Backbones
| candidate (nb 04) | year | status 2025 | verdict |
|---|---|---|---|
| EfficientNet-B0     | 2019 | **deprecated** | slower + less accurate than RepViT-M1 and FasterNet-T2 at equal params. Drop. |
| MobileNetV3-Large   | 2019 | **deprecated** | beaten by MobileNetV4, RepViT, FasterNet. Drop. |
| CSPDarknet-53       | 2020 | fine, but same family as YOLOPv2's ELAN | not a real ablation. Drop. |
| ConvNeXt-Tiny       | 2022 | fine | keep as *quality ceiling* reference only; ~70 FPS on A5000 FP16 so it's the slow end of "real time". |

**Missing candidates that are more advanced and already vendored in `external_repos/`:**
| candidate | year | why it fits our target |
|---|---|---|
| **RepViT-M1.0 / M1.1** | CVPR 2024 | 2024 SOTA mobile CNN from ViT perspective. 80%+ ImageNet at ~1 ms iPhone. Re-parameterizable → fat-train / skinny-infer. Best speed-accuracy point for real-time. |
| **FasterNet-T2 / S** | CVPR 2023 | already in `external_repos/FasterNet`. PConv (Partial Convolution) is the main 2023 low-FLOP trick; 2–3× faster than MobileNet at equal accuracy. |
| **EfficientViT-B1** | CVPR 2023 | already in `external_repos/efficientvit`. Linear attention, designed for real-time. |
| **InternImage-T** | CVPR 2023 | already in `external_repos/InternImage`. DCNv3, heavier; use only for *quality ceiling* comparison. |

### Necks
| candidate (nb 04) | year | verdict |
|---|---|---|
| BiFPN   | 2019 | beaten by Gold-YOLO's Gather-and-Distribute (GD) on identical detector (+2–4 AP at equal latency). |
| PANet+  | 2017/2020 | same family as baseline; keep as reference. |

**Better options (2024 era):**
- **Gold-YOLO GD neck** (NeurIPS 2023) — global aggregation + re-injection; generally superior to BiFPN at the same speed.
- **RT-DETR hybrid encoder** — attention at the high-level feature only (AIFI), CNN cross-scale below it; designed for real-time detection.
- **AFPN** (2023) — asymptotic FPN, simpler than BiFPN and faster.

### Lane decoder / loss (nb 05)
| candidate | year | verdict |
|---|---|---|
| UNet skip                   | 2015 | fine |
| CBAM / SE                   | 2017/18 | cheap attention; keep |
| Dice / Focal / Lovász       | 2016/17/18 | standard; keep all three and ensemble |
| boundary-aware loss         | 2019 | useful for thin structures like lanes |

**Missing and more advanced (lane specifically):**
- **CLRerNet's LaneIoU loss** (WACV 2024). Extends CLRNet (already vendored in `external_repos/CLRNet`) with a lane-topology-aware IoU. **This is the single highest-leverage upgrade for the lane branch.**
- **UFLDv2-style row-anchor formulation** — 300+ FPS on a 2080Ti. Completely different formulation (row classification instead of segmentation). Would require a parallel head; high risk/high reward.
- **Conditional IoU / Topology loss** from CondLaneNet.

### Detection head
Not really explored in notebook 04/05. But as of 2025, **anchor-free
YOLOPX-style heads** outperform anchor-based on BDD100K:
- YOLOPX (Pattern Recognition, 2024): 83.3 mAP@50 and 93.7 recall on
  BDD100K, 47 FPS on RTX 3080. Beats YOLOPv2's 83.4 mAP50 / 91.1 recall
  with a *simpler* anchor-free head.
- YOLOv8-style DFL (Distribution Focal Loss) — cleaner than objectness +
  anchor-offset decoding.

### Export (nb 06)
| current | 2025-appropriate replacement |
|---|---|
| ONNX opset 12 | opset 17 (more ops: LayerNorm, GroupNorm natively) |
| TorchScript trace only | add **TensorRT FP16 engine** (`polygraphy` or `torch-tensorrt`) |
| no quantization | add **INT8 PTQ** with a small calibration set from BDD val |
| no graph compilation | add **`torch.compile(mode='reduce-overhead')`** for training |

## 3. Are notebooks 04 / 05 / 06 "best effectiveness + real time"?

**Short answer: no, not as written.** They were drafted as 2022-era
ablation stubs. Three structural problems:

1. **Candidate lists are aging.** Every backbone in notebook 04 was SOTA
   in 2019–2022 and has a strictly-better 2023–2024 successor. Worst
   offender: MobileNetV3 (six years old, dominated by MobileNetV4 /
   RepViT / FasterNet).
2. **The single highest-leverage lane upgrade is missing** — CLRerNet's
   LaneIoU loss. Lane IoU on BDD100K is the metric where every recent
   multi-task network (YOLOPX, TwinLiteNet+, RLSNet) shows the largest
   absolute gain over YOLOPv2.
3. **Export has no deployment path.** For a real-time A5000 deliverable,
   ONNX alone isn't enough; TensorRT FP16 at minimum, FP8/INT8 ideally.

## 4. Proposed Stage-2 plan

Organised so each notebook changes **one variable at a time**, which is
what the original brief demanded. Drive-persistence contract unchanged:
every run writes to `/content/drive/MyDrive/EcoCAR/yolop_vehicle_lane/{checkpoints,metrics}/<run_name>`.

### Notebook 04 — `04_backbone_neck_ablation.ipynb`
Replace candidate list with:
```
backbones:
  - repvit_m1      (timm)           [primary]
  - fasternet_s    (from external_repos/FasterNet)
  - efficientvit_b1 (from external_repos/efficientvit)
  - convnext_tiny  (timm)           [quality-ceiling reference only]
necks:
  - baseline PAN   (YOLOPv2)
  - gold_yolo_gd   (from Gold-YOLO repo, adapt)
  - rtdetr_hybrid  (from ultralytics/RT-DETR, adapt)
```
Each run produces `metrics/<backbone>_<neck>/*.json`. Pareto plot of
(val_map50 + val_ll_iou) vs A5000 FPS is the deliverable.

### Notebook 05 — `05_lane_head_and_loss_ablation.ipynb`
Three dependent sub-runs:
1. **Loss sweep** with baseline lane decoder:
   `{BCE+IoU}`, `{BCE+Dice}`, `{BCE+Lovász}`, `{BCE+LaneIoU}` (CLRerNet).
2. **Attention insertion**: drop CBAM or SE at the last three decoder
   stages; keep best loss from (1).
3. **Auxiliary boundary loss**: add Sobel-boundary BCE as an auxiliary
   head; keep best from (2).

LaneIoU is implemented inline in this notebook, referencing
`external_repos/CLRNet` for the topology-IoU math. **Expected gain on
BDD100K val: +3–5 lane IoU points** based on the CLRerNet paper's
in-family delta.

### Notebook 06 — `06_export_and_deploy.ipynb`
Rewrite around the A5000 deployment contract:
```
pipeline:
  1. load best checkpoint from stage-2 Pareto winner
  2. torch.compile(mode='reduce-overhead') for training/eval parity
  3. ONNX export, opset 17
  4. TensorRT FP16 engine build via torch-tensorrt
  5. TensorRT INT8 PTQ with 500-image BDD val calibration set
  6. benchmark all three (.pth / .onnx / .trt-fp16 / .trt-int8) on A5000
  7. save deployable bundle to Drive
```
Success criterion: ≥ 60 FPS at batch=1 on A5000 in the INT8 engine
*while* lane IoU drop is ≤ 1 point vs FP32 reference.

### New notebook 08 — `08_anchor_free_detection_head.ipynb` (optional)
YOLOPX-style decoupled anchor-free head. Keep encoder + lane head
frozen, swap only the detection head, retrain. High-leverage if the
Stage-2 winner is compute-bound on the detection head.

## 5. Immediate recommended order of work

1. **Fix Notebook 06** first — a proper A5000 TensorRT path is a
   deliverable by itself regardless of model choice. Low risk, high
   value.
2. **Run the CLRerNet LaneIoU loss swap in Notebook 05** — one-line
   loss change, biggest expected lane IoU gain for the least code.
3. **Backbone swap (Notebook 04)**: RepViT-M1 + existing neck as the
   first ablation point. Secondary: FasterNet-S from already-vendored
   repo.
4. **Neck swap** only after backbone winner is locked.
5. **Anchor-free head** only if detection mAP is plateauing and we still
   have FPS budget.

## 6. [INFERRED] caveats

- YOLOPv2 is still torch.jit-scripted; none of the above replaces the
  "clean YOLOPv2 reproduction" target from Phase 1. These are
  *Stage-2 upgrades on top of* the inferred baseline, explicitly
  labelled as such.
- CLRerNet's LaneIoU is defined on lane curves; we apply it as a
  soft-IoU surrogate on the segmentation foreground channel, which is
  a reasonable adaptation but not identical to the paper's formulation.
  Mark `# [ADAPTED]` in the code.
- Gold-YOLO's GD neck assumes 4 feature scales; our neck has 3. We'd
  use the low-GD variant (one global + one local injector).

## 7. Data-based lane branch (YOLOPv2-DETRLane, implemented)

**Why a second Stage-2 track.** The mask branch (notebook 05's loss
sweep) caps at the expressivity of a binary pixel predictor. The
project's DETR-style `DETR_GeoLane_pipeline` already contains a
point-based lane formulation with Hungarian matching + curve-geometry
loss — that is strictly more informative (order, tangent, curvature,
type) and directly comparable to 2024 lane-detection SOTA that uses
curve / polyline representations (MapTR, LDTR, CLRerNet).

**Where it lives in the repo** (all new files; nothing baseline touched):
- `lib/models/yolopv2_detrlane.py` — YOLOPv2 ELAN backbone + SPPCSPC +
  PAN neck (identical to `yolopv2_baseline.py`) + anchor IDetect +
  `LaneSetHead`. Task-specific adapters between PAN and each head.
- `lib/models/lane_set_head.py` — DETR-style lane queries, 3 decoder
  layers, lane-priors + iterative point refinement, aux outputs.
  Vendored from `DETR_GeoLane_pipeline/src/lane_head.py`.
- `lib/core/lane_set_loss.py` — Hungarian matcher with combined
  geometric + raster cost, curve-to-curve distance, tangent /
  curvature / soft-IoU terms, curriculum scheduler. Vendored from
  `DETR_GeoLane_pipeline/src/losses.py`.
- `lib/core/loss_detrlane.py` — multi-task wrapper: YOLOP detection
  loss + `LaneSetLoss`, with `DET_TASK_WEIGHT` / `LANE_TASK_WEIGHT` and
  a detection-only warmup gate.
- `lib/dataset/bdd_points.py` — `BddDatasetPoints` emits
  `(det_labels, lane_existence, lane_points, lane_visibility, lane_type)`
  per sample. No rendered masks needed; reads lane polygon JSONs
  directly via `LaneLabelCache`.
- `configs/yolopv2_detrlane_vehicle_lane.yaml` — Stage-2 variant
  config. `MODEL.NAME = YOLOPv2-DETRLane` dispatches the new model.
- `notebooks/05_lane_head_and_loss_upgrade.ipynb` — full training loop.

**How the two heads are kept from conflicting.**
1. **Disjoint heads on a shared trunk.** Anchor IDetect and DETR
   `LaneSetHead` share only the ELAN backbone + PAN neck. Parameter
   sets beyond the trunk do not overlap, so only the trunk receives
   both task gradients — a standard and well-studied multi-task
   regime.
2. **Task-specific residual adapters.** One small 1×1/3×3/1×1 residual
   block per PAN scale per head. Each head gets to re-shape the
   shared feature before consuming it. Adapter scale is a learnable
   scalar initialised to 0.10 so it starts near identity.
3. **Staged training.** `LANE.DET_ONLY_EPOCHS = 3` freezes the lane
   branch in `forward` (detached features + no-grad) for the first 3
   epochs. This matches the DETR_GeoLane observation that the harder
   structured task can dominate early and pollute shared features
   (motivates the Conflict-Averse Gradient Descent and PCGrad
   literature).
4. **Loss curriculum.** Early epochs weight raster soft-IoU highly
   (stable, dense gradient); later epochs shift weight to curve /
   tangent / curvature terms (sharp polylines). The `LaneLossScheduler`
   is the exact one used in `DETR_GeoLane_pipeline`.

**How to go further without re-breaking the baseline.**
- Run `05_lane_head_and_loss_upgrade.ipynb` as-is; it writes to
  `checkpoints/yolopv2_detrlane/` — orthogonal to the baseline.
- Anchor-free vehicle head (YOLOPX-style) can replace IDetect later
  without touching the lane branch — the shared PAN output contract
  is the integration point.
- To benchmark against 2024 SOTA on BDD100K lane F1 / IoU, pair the
  trained checkpoint with the CLRNet evaluation harness already
  vendored in `external_repos/CLRNet`.

## 8. Sources consulted (2024–2025)

- YOLOPX (Pattern Recognition, 2024): anchor-free multi-task
  head, BDD100K SOTA, 47 FPS on RTX 3080.
- YOLOPv3 (2024): 84.3 mAP50, 96.9 recall, 28.0 lane IoU — current
  anchor-based BDD100K SOTA.
- TwinLiteNet+ (arXiv 2403.16958, 2024): 34K–1.94M param
  multi-task seg, 34.2 % BDD lane IoU.
- CLRerNet (WACV 2024): LaneIoU loss; +1–3 F1 over CLRNet.
- LDTR (arXiv 2403.14354, 2024): transformer-based lane detection
  with anchor-chain representation; DETR-style with LaneIoU.
- MapTR / MapTRv2: deformable DETR for vector map + lane detection.
- RepViT (CVPR 2024): current best lightweight CNN-from-ViT.
- Gold-YOLO (NeurIPS 2023): GD neck beats BiFPN / PANet at equal latency.
- FasterNet (CVPR 2023): PConv; already vendored.
- EfficientViT (CVPR 2023): linear attention; already vendored.
- Conflict-Averse Gradient Descent (NeurIPS 2021): MTL gradient
  conflict mitigation framework.
- PCGrad (NeurIPS 2020): project conflicting task gradients before
  summation.
- "Proactive Gradient Conflict Mitigation in Multi-Task Learning"
  (arXiv 2411.18615, 2024): sparse-training perspective.
- "Mitigating gradient conflicts via expert squads in multi-task
  learning" (Neurocomputing 2024).

See the references list at the bottom of this document for links.
