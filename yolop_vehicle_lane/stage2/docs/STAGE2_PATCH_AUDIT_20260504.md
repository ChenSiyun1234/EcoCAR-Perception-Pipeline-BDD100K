# Stage 2 Patch Audit — 2026-05-04

## Problems fixed

1. `stage2/scripts/04_prepare_bdd_curve_labels.py` accepted `--bdd-images` and `--bdd-labels` in notebooks, but the script did not define those arguments. The script now supports those aliases, keeps the original `--dataset-root` and `--raw-root` API, and can write a Drive-ready tar through `--pack-to`.

2. The CLRKD curve dataset tar previously risked storing symlinks to local `/content` images instead of image files. `pack_output_root()` now creates the tar with dereferencing enabled, so the archive is reusable in a later notebook/runtime.

3. Crosswalk labels were normalized as lane-like labels in the preparation path. Both `stage2/scripts/04_prepare_bdd_curve_labels.py` and `stage2/fusion/lane_targets.py` now filter training labels to actual lane-line categories only.

4. The vendored RMT-PPAD `MTDETRDecoder` parser treated the `clrkd_fpn` string as the hidden dimension positional argument. `ultralytics/nn/tasks.py` now expands default decoder arguments and passes `clrkd_fpn` to the `seg_decoder` position.

5. The vendored `CLRKDFusedSegmentationDecoder.forward()` did not match the caller signature `seg_head(x_proj, self.imgsz)` and returned only a tensor while RMT expects `(seg_mask, aux_list)`. It now accepts `imgsz`, resizes when requested, and returns `(out, [out])`.

6. Stage 2 notebook 06 previously trained only the lane branch even though it was described as fusion. It now calls `stage2/scripts/06_train_rmt_clrkd_basic_fusion.py`, which computes both detection and lane losses when Stage-1 YOLO labels are provided.

7. Stage 2 notebook 07 was a TODO skeleton. It now runs the patched vendored RMT-PPAD sanity check and the available RMT dense-CLRKD FPN training path.

8. Stage 2 notebook 08 used a hard-coded YOLO26 path. It now searches likely top-level workspace folders for YOLO26/Yolo26 projects.

9. Stage 2 notebook 09 had stale assumptions about the checkpoint layout. It now extracts the training tar, loads the notebook 06 checkpoint, and reports latency, FPS, and peak GPU memory.

10. Stage 1 evaluation/profiling notebooks were checked for stale YOLOP fallback behavior. The relevant notebooks default to YOLOPX and fail loudly if the matching YOLOPX checkpoint is missing.

## Remaining limitation

The true RMT backbone/neck plus in-house CLRKD curve-slot head is still not a fully upstream-native RMT training path. Notebook 07 now gives a working vendored RMT path with the patched dense CLRKD FPN decoder. Notebook 06 provides the curve-slot fusion loss path with a lightweight detection head. Merging those two into a full RMT feature wrapper is the next engineering step.
