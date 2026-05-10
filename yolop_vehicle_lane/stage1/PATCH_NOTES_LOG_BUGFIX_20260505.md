# Stage 1 log bugfix patch — 2026-05-05

Fixed issues reported in `log.txt`:

1. `from lib.config import cfg` could indirectly import model/loss code through `lib/__init__.py`, so a stale/missing `lib/core/postprocess.py` on Drive crashed notebook 06 before preflight. `lib/__init__.py` is now lightweight and `lib/core/loss.py` has an internal fallback `build_targets()`.
2. Stage1 notebook 06 now creates metrics/visualization folders before validation and defines `RUN_EVAL`, `EXPORT_ONNX`, and `EXPORT_TORCHSCRIPT` defensively, so re-running later cells after a failed import no longer causes `NameError`.
3. `YOLOPXVehicleLaneNet` now implements `predict()`, returning detection output plus a stable one-channel lane foreground probability. Notebook 07 also has a local fallback helper for older checkpoints/code.
