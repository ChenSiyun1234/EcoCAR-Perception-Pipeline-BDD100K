# RMT-PPAD patch summary

Patched files under `stage2/vendor/RMT-PPAD/`:

1. `ultralytics/nn/modules/transformer.py`
   - Made `TransformerSegmentationDecoder` support an arbitrary segmentation class count instead of hard-coding two segmentation tasks.
   - Added `CLRKDFusedSegmentationDecoder` for the CLRKDNet-inspired fusion experiment.

2. `ultralytics/nn/modules/head.py`
   - Imported `CLRKDFusedSegmentationDecoder`.
   - Added optional `seg_decoder` argument to `MTDETRDecoder`.
   - `seg_decoder="rmt"` keeps the original RMT-PPAD segmentation decoder.
   - `seg_decoder="clrkd_fpn"` switches only the lane branch to the CLRKDNet-inspired decoder.

3. `ultralytics/nn/tasks.py`
   - Updated `parse_model()` so `MTDETRDecoder` can receive optional decoder-selection arguments from YAML.

4. `ultralytics/models/utils/loss.py`
   - Generalized segmentation loss from two masks to one lane mask.
   - Drivable-area losses are set to zero in lane-only mode.

5. `ultralytics/models/mtdetr/val.py`
   - Made mask threshold broadcasting safe for one-channel segmentation.

6. `ultralytics/models/mtdetr/predict.py`
   - Made mask threshold broadcasting safe for one-channel segmentation.
