# Stage-1 hotfix notes (2026-04-19)

1. Notebook 02b tiny-subset lane diagnostic:
   - disabled train-time augmentation for the 16-sample overfit check;
   - switched the lane IoU probe from sigmoid(fg_logit) to softmax(fg channel) for 2-class lane logits;
   - switched the tiny-debug optimizer to Adam lr=1e-3.

2. Notebook 06 final eval/export:
   - defaults now target the stage-1 YOLOP baseline;
   - checkpoint selection prefers best_joint.pth and falls back safely;
   - runs full validation first and saves a JSON summary before export;
   - keeps export as a second step.

3. Notebook 07 A5000 video profile:
   - defaults now target the stage-1 YOLOP baseline;
   - checkpoint selection prefers best_joint.pth;
   - safer FP16 handling and saved JSON reports for MFU/video profiling.
