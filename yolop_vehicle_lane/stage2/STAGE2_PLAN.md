# Stage 2 — Vehicle Detection + Curve Lane Fusion Plan

## Why this exists

Stage 1 produced a YOLOPX vehicle detection + dense lane segmentation baseline.
Stage 2 moves the lane branch from dense pixel masks to a **structured curve
representation** — the same kind of lane targets DETR_GeoLane uses, and the
input format CLRKDNet expects. Detection moves from anchor-based YOLOX to
RT-DETR-style queries.

The path is **incremental**. We do not jump into a single huge fused
architecture. Each experiment isolates one variable at a time.

We also explicitly **stop reproducing RMT-PPAD end-to-end**. RMT-PPAD is used
as an architectural source (RT-DETR head, RMT-style backbone, GCA blocks)
rather than a paper to chase.

## Directory layout

```
yolop_vehicle_lane/stage2/
  configs/
    BDD100K_CLRKD_Curve.py            # Experiment 2 — vendored CLRKDNet path
    clrkd_curve_baseline.yaml         # Experiment 2 — in-house light path
    rmt_clrkd_basic_fusion.yaml       # Experiment 3
    yolo26_clrkd_experiment.yaml      # Experiment 6
    rmt_ppad_lane_only.yaml           # legacy, kept for comparison
    rmt_ppad_clrkd_fused_lane.yaml    # legacy, kept for comparison
    bdd100k_vehicle_lane_rmt.yaml     # legacy, kept for comparison
  fusion/                              # NEW — in-house fusion code
    __init__.py
    lane_targets.py                    # BDD100K JSON -> curve targets (ports DETR_GeoLane)
    losses.py                          # Multi-task lane + det loss assembly
    lane_head.py                       # CLRKD-style curve lane head
    model.py                           # FusionModel wrapper
  notebooks/
    00_prepare_rmt_dataset_links.ipynb        # legacy; kept
    01_sanity_check_rmt_ppad_lane_only.ipynb  # legacy; kept
    02_train_rmt_ppad_lane_only.ipynb         # legacy; kept
    03_train_rmt_ppad_clrkd_fused_lane.ipynb  # legacy; kept
    04_prepare_clrkd_curve_dataset_colab.ipynb        # NEW
    05_train_clrkd_curve_baseline_colab.ipynb         # NEW
    06_train_rmt_clrkd_basic_fusion_colab.ipynb       # NEW
    07_train_rmt_backbone_clrkd_curve_head_colab.ipynb # NEW (skeleton)
    08_yolo26_backbone_neck_experiment_colab.ipynb     # NEW (skeleton)
    09_stage2_eval_video_profile_colab.ipynb           # NEW
  scripts/
    00_prepare_rmt_dataset_links.py
    01_train_rmt_ppad_lane_only.py
    02_sanity_check_stage2.py
    03_train_rmt_ppad_clrkd_fused_lane.py
    04_prepare_bdd_curve_labels.py    # KEEP — produces CULane-style files for vendored CLRKDNet
    05_train_bdd_clrkd_curve.py       # KEEP — wrapper for vendored CLRKDNet trainer
  vendor/
    CLRKDNet/        # vendored — used by Experiment 2 strict path
    RMT-PPAD/        # vendored — used by Experiments 3, 4, 5
```

## Experiment sequence

### Experiment 0 — Stage 1 YOLOPX baseline verification
**Purpose:** confirm Stage 1 YOLOPX is what is being evaluated, after the
Stage 1 post-02 notebook fixes.

**How to run:** `stage1/notebooks/03_eval_and_backbone_ablation.ipynb`,
`06_final_train_eval_export.ipynb`, `07_a5000_video_profile.ipynb`.

**Required outputs:** mAP50, lane IoU / mIoU, FPS profile, qualitative images.

### Experiment 1 — Build the CLRKD curve dataset
**Notebook:** `notebooks/04_prepare_clrkd_curve_dataset_colab.ipynb`

**What it does:**
1. Extracts BDD100K from `/content/drive/MyDrive/EcoCAR/downloads/...` into
   `/content/bdd100k_raw/100k`.
2. Runs `stage2/scripts/04_prepare_bdd_curve_labels.py` to write CULane-style
   `.lines.txt` files plus train/val/test list files plus auxiliary masks.
3. Runs an in-process diagnostic with `stage2.fusion.lane_targets.LaneLabelCache`
   to verify the parser independently (different code path, same DETR_GeoLane
   logic). Counts must agree with the script's outputs.
4. Visualizes 4 sample images with extracted points and rendered soft masks.
5. Tars the result and pushes it to Drive at:
   `/content/drive/MyDrive/EcoCAR/datasets/bdd100k_clrkd_curve.tar`

### Experiment 2 — CLRKD curve lane-only baseline
Two sub-paths exist; both are valid first passes.

**Strict (vendored CLRKDNet):**
- Notebook: `05_train_clrkd_curve_baseline_colab.ipynb`
- Config:   `configs/BDD100K_CLRKD_Curve.py`
- Trainer:  vendored CLRKDNet at `stage2/vendor/CLRKDNet/main.py` via
  `stage2/scripts/05_train_bdd_clrkd_curve.py`
- Outputs CLRKDNet's native `lane_line` tensor with class scores + LineIoU
  loss. Mirrors the upstream paper.

**Light (in-house fusion module):**
- Built into `notebooks/06_*` rather than a separate script — the in-house
  `CurveLaneHead` is exercised lane-only on the same BDD curve dataset.
- Config: `configs/clrkd_curve_baseline.yaml`
- Useful as a sanity check for `stage2/fusion/{losses, lane_head}.py`
  before plugging into fusion.

**Output:** `/content/drive/MyDrive/EcoCAR/training_runs/stage2_clrkd_lane_only.tar`

### Experiment 3 — Basic RMT detection + CLRKD lane fusion
**Notebook:** `notebooks/06_train_rmt_clrkd_basic_fusion_colab.ipynb`
**Config:**  `configs/rmt_clrkd_basic_fusion.yaml`

Conservative first fusion: keep the existing CSP backbone, swap only the lane
branch to the in-house `CurveLaneHead`. Detection branch is wired in as a
TODO so the lane branch can be validated alone first. Loss:

```
L_total = L_det + lambda_lane * L_lane
L_lane  = w_cls L_cls + w_reg L_reg + w_iou L_LineIoU
        + w_mask L_mask_aux + w_smooth L_smooth + w_distill L_distill
```

**Output:** `/content/drive/MyDrive/EcoCAR/training_runs/stage2_rmt_clrkd_basic_fusion.tar`

### Experiment 4 — RMT backbone + neck + CLRKD curve head
**Notebook:** `notebooks/07_train_rmt_backbone_clrkd_curve_head_colab.ipynb`

Replaces the CSP backbone with the vendored RMT-PPAD ResNet/RMT layers and the
hybrid encoder (AIFI / STB). Lane head consumes the higher-resolution P2/P3
features because lanes are thin and detail-sensitive. Detection consumes the
fuller P3/P4/P5 stack via `MTDETRDecoder`.

### Experiment 5 — Hybrid backbone/head fusion
This is research-mode. Try:

1. RMT detection head + CLRKD feature aggregation for lane.
2. CLRKD lane head + task-specific adapters.
3. GCA-style shared/task-specific feature splits between detection and lane.
4. Cosine-similarity gradient diagnostic (`stage2.fusion.losses.compute_grad_cosine`)
   to measure task conflict on shared backbone params.

This experiment doesn't have a dedicated notebook yet — it forks `notebook 07`
and changes the model wiring.

### Experiment 6 — YOLO26 backbone/neck experiment
**Notebook:** `notebooks/08_yolo26_backbone_neck_experiment_colab.ipynb`
**Config:**  `configs/yolo26_clrkd_experiment.yaml`

Plugs `yolo26_pipeline` (top-level sister project) into `FusionModel`. Status:
skeleton — see notebook for what's wired and what's TODO.

## Loss design (Part E)

Implemented in `stage2/fusion/losses.py`:

| Component | Class / function | Purpose |
|---|---|---|
| Existence (focal BCE) | `_binary_focal_loss` | per-lane existence, focal_alpha=0.25, focal_gamma=2 |
| Coord regression | `FusionLaneLoss.forward` | SmoothL1 over (x, y) per row, masked by visibility |
| LineIoU | `_line_iou_1d` | IoU between per-row x-bands of pred and gt |
| Aux mask | Dice + BCE | optional rendered mask supervision |
| Smoothness | `_smoothness_x` | second-difference penalty on x along the curve |
| Distillation | optional teacher dict | sigmoid-MSE on cls + masked L1 on coords |
| Multi-task balance | `UncertaintyMultiTaskLoss` | learnable log-variance weighting |
| Gradient diagnostic | `compute_grad_cosine` | cosine of det/lane gradients on shared params |

Default scheme: `L_total = L_det + lambda_lane * L_lane`. Switch to uncertainty
weighting by setting `loss.use_uncertainty = true` in the YAML.

## Evaluation requirements (Part F)

For detection: mAP50, mAP50-95 if available, recall, average preds per image,
qualitative detection images.

For lane: lane existence accuracy, soft-mask IoU, LineIoU proxy, lane F1 if
implemented, qualitative visualizations.

For runtime: FPS, latency per frame, GPU peak memory, per-stage timing
(preprocessing, backbone+neck, detection head, lane head, postprocess).

Notebook 09 implements the lane + runtime metrics; detection metrics are
wired into the same notebook once the detection head is in place.

## Colab discipline (Part G)

Every Stage 2 notebook follows the same shape:

1. Mount Drive.
2. Define Drive paths.
3. Extract required tar archives from Drive into `/content`.
4. Train / process locally on `/content`.
5. Save outputs to a local dir.
6. `tar -cf` that dir into a single file under
   `/content/drive/MyDrive/EcoCAR/training_runs/...`.
7. Print the Drive output path.

Drive paths follow the convention agreed for this project:
```
/content/drive/MyDrive/EcoCAR/yolop_vehicle_lane     # repo root
/content/drive/MyDrive/EcoCAR/datasets               # tar archives
/content/drive/MyDrive/EcoCAR/downloads              # raw zips
/content/drive/MyDrive/EcoCAR/training_runs          # checkpoints + metric tars
```

## What was deliberately not done

- **Full RMT-PPAD reproduction.** The RMT detection components (decoder,
  hybrid encoder, MSDeformAttn) are reused; the rest of RMT-PPAD's training
  framework is not.
- **PCGrad as a default.** Logged as a possible Experiment 5 add-on; not
  applied to the basic baselines because gradient surgery before the simple
  baseline is stable wastes time.
- **Direct CLRHead reuse inside the fusion model.** The vendored CLRHead is
  available but heavy. The lighter in-house `CurveLaneHead` is what the
  fusion experiments train. The vendored CLRHead remains as the strict
  reproduction path under Experiment 2.

## How to extend

- **Add a new backbone**: implement an `nn.Module` whose forward returns
  `[P3, P4, P5]` (or any list of feature maps); pass it to `FusionModel`.
- **Add a new detection head**: same; pass via `detection_head=`.
- **Add a new lane loss term**: extend `FusionLaneLoss.forward` and add a
  weight to `FusionLossConfig`.
- **Add a new experiment config**: create
  `stage2/configs/<experiment>.yaml`, mirror an existing one, and reference
  it from the experiment's notebook.
