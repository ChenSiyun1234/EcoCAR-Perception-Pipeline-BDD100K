# Stage 2 Exp3 YOLO26-Inspired Custom Backbone/Neck Refactor

This patch replaces the previous Exp3 black-box YOLO26 wrapper path with a custom official-YOLO26-inspired joint backbone/neck.

## Motivation

The previous Exp3 implementation searched for a YOLO26 model file and used forward hooks to grab the last three conv-like feature maps. That made the experiment fragile because the hooked layers might not be the true P3/P4/P5 neck outputs. It also behaved more like model reuse than a new joint-model design.

The new Exp3 implementation uses YOLO26 as a design reference instead of as a black box. It builds an in-project CNN backbone/neck with efficient CSP/C2f-style stages, PAN/FPN feature fusion, an optional P5 global context block, an optional lane-preserving P3 refinement path, and optional RMT-style GCA task split.

## New code

Main implementation:

- `stage2/fusion/yolo26_inspired.py`

Factory integration:

- `stage2/fusion/experiment_factory.py`
- `stage2/fusion/__init__.py`

New configs:

- `stage2/configs/exp04_yolo26_inspired_joint.yaml`
- `stage2/configs/exp05_yolo26_inspired_global_joint.yaml`
- `stage2/configs/exp06_yolo26_inspired_gca_joint.yaml`

New notebooks:

- `stage2/notebooks/stage2_notebook_04_exp3a_yolo26_inspired_joint.ipynb`
- `stage2/notebooks/stage2_notebook_05_exp3b_yolo26_inspired_global_joint.ipynb`
- `stage2/notebooks/stage2_notebook_06_exp3c_yolo26_inspired_gca_joint.ipynb`

The old real-YOLO26 hook-based files are kept as `.legacy.*` so they are not accidentally used as the main experiment path.

## Architecture

Exp3A:

```text
image
  -> YOLO26Conv stem, stride 2 + stride 2
  -> YOLO26Stage C3, stride 8
  -> YOLO26Stage C4, stride 16
  -> YOLO26Stage C5, stride 32
  -> YOLO26PANNeck
  -> P3/P4/P5
  -> DETR-style vehicle detector + CLRKD-style lane curve head
```

Exp3B adds:

```text
P5 -> P5GlobalContextBlock -> top-down FPN -> bottom-up PAN
```

Exp3C adds:

```text
shared P3/P4/P5
  -> GCA-det P3/P4/P5
  -> GCA-lane P3/P4/P5
```

## Layer intent

- `YOLO26Conv`: export-friendly Conv-BN-SiLU block.
- `YOLO26Bottleneck`: local residual block with optional depthwise conv for edge-oriented efficiency.
- `YOLO26CSPBlock`: C2f/CSP-style split-and-aggregate block for efficient feature reuse.
- `YOLO26Stage`: stride-2 downsampling plus CSP feature extraction.
- `YOLO26PANNeck`: custom P3/P4/P5 neck output definition.
- `P5GlobalContextBlock`: AIFI-like global feature interaction on low-resolution P5 only.
- `LanePreservingP3Block`: depthwise local + dilated depthwise long-range refinement for thin lane evidence.
- `YOLO26InspiredJointBackboneNeck`: complete joint feature extractor with optional GCA split.

## Validation

Validation command:

```bash
python -m py_compile stage2/fusion/*.py stage2/scripts/*.py
python stage2/scripts/smoke_test_joint_models.py \
  stage2/configs/exp01_rmt_shared_joint.yaml \
  stage2/configs/exp02_rmt_gca_joint.yaml \
  stage2/configs/exp03_rmt_gca_clrkd_kd_joint.yaml \
  stage2/configs/exp04_yolo26_inspired_joint.yaml \
  stage2/configs/exp05_yolo26_inspired_global_joint.yaml \
  stage2/configs/exp06_yolo26_inspired_gca_joint.yaml
```

Saved log:

- `stage2/validation/smoke_test_exp01_to_exp06_yolo26_inspired_20260505.txt`

The smoke test verifies forward pass, lane curve output, detection output, lane loss, detection loss, joint backward, gradient-ratio computation, and GCA gate statistics for GCA experiments.
