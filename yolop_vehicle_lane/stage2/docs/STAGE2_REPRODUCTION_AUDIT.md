# Stage-2 reproduction audit

## RMT-PPAD replica

The active exact baseline is:

```text
stage2/configs/rmt_ppad_lane_only.yaml
stage2/scripts/01_train_rmt_ppad_lane_only.py
```

The code is vendored from RMT-PPAD:

```text
stage2/vendor/RMT-PPAD/
```

The original Stage-2 code was removed from the package. The active Stage 2 now contains only the RMT-PPAD baseline path, the CLRKDNet reference vendor copy, and the controlled CLRKDNet fusion path.

## Intentional differences from upstream RMT-PPAD

1. Drivable-area segmentation is removed.
   - Upstream RMT-PPAD uses two segmentation masks.
   - This project uses only lane segmentation.

2. Dataset path is adapted to Stage-1 output.
   - Stage 1 writes masks to `masks/{train,val}`.
   - RMT-PPAD expects `mask/lane/{train,val}`.
   - The Stage-2 preparation script creates the required links.

3. Single-GPU RTX Pro 6000 defaults are used.
   - Batch size is set to 16.
   - `lr0` is set to 0.0001.
   - Validation and checkpoint saving run every epoch.

4. Segmentation loss is generalized from 2 channels to 1 channel.
   - Upstream loss indexes channel 0 as drivable area and channel 1 as lane.
   - This project uses channel 0 as lane and sets drivable losses to zero.

5. The original RMT segmentation decoder hard-coded two task weights.
   - It was patched so the task dimension follows the configured segmentation class count.

## CLRKDNet fusion path

The fusion experiment is:

```text
stage2/configs/rmt_ppad_clrkd_fused_lane.yaml
stage2/scripts/03_train_rmt_ppad_clrkd_fused_lane.py
```

The detector, transformer decoder, matching loss, and RMT-PPAD training logic are unchanged. The lane segmentation branch uses a CLRKDNet-inspired feature aggregation decoder:

```text
CLRKDFusedSegmentationDecoder
```

This is a controlled first fusion. It does not yet convert BDD100K dense lane masks into CLRKDNet's original lane-prior/curve supervision.
