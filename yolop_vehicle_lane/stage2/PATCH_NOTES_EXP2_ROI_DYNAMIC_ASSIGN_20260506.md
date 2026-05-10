# Stage 2 Exp2 ROI + dynamic matching refactor notes

This patch is based on the observed notebook 04/05/08 results:

1. Exp2A and Exp2B had decreasing raw validation lane loss, but lane point MAE stayed around 0.52-0.53, so the curve geometry was not improving.
2. Exp2D added a lane detail neck and slightly reduced lane point MAE, but the gain was small and validation lane loss trended upward under geometry warmup.
3. Exp2E added Hungarian lane matching and reduced lane point MAE to about 0.40, which showed that target-slot ordering was a major problem.
4. Exp2E also had weak lane existence calibration, so the next target is not just lower point MAE; it is stable lane precision/recall/F1 and better positive/negative slot separation.

Main changes in this patch:

1. `stage2/fusion/lane_head.py`
   - Added `output_all_priors` so Exp2E/Exp2F can train over all CLRKD-style lane priors instead of selecting only 10 priors before loss computation.
   - Added lightweight ROI gather along predicted lane curves.
   - Added iterative ROI refinement through `roi_refine_layers`.
   - This is closer to the important CLRKD idea: each lane prior should collect feature evidence along its own predicted curve.

2. `stage2/fusion/losses.py`
   - Updated lane matching so prediction slot count can differ from GT lane count.
   - This allows 192 predicted priors to match 10 padded BDD100K GT lanes.

3. `stage2/configs/exp04_rmt_gca_lane_matching_joint.yaml`
   - Enabled all-prior lane training.
   - Enabled ROI gather refinement.
   - Changed lane assignment from Hungarian to dynamic-k matching.
   - Kept all-prior classification loss so unmatched priors are explicit background.

4. `stage2/configs/exp05_rmt_gca_lane_detail_matching_joint.yaml`
   - Same lane-prior and ROI-gather changes as Exp2E.
   - Keeps the lane-detail branch for comparison.

5. `stage2/scripts/train_joint_model_experiment.py`
   - Prints lane-head architecture settings.
   - Prints lane loss components, lane existence statistics, predicted/GT/matched lane counts, FP/FN lane slots, and positive/negative lane scores in every epoch summary.

6. `stage2/scripts/evaluate_joint_model.py`
   - Added missing evaluation script used by notebook 08.
   - Evaluates saved run tars without re-training.

7. `stage2/scripts/profile_joint_video.py`
   - Now draws vehicle bounding boxes and lane curves in the preview video.
   - Supports `--run-tar` directly and prints per-frame detection/lane counts.

8. `stage2/scripts/plot_stage2_metrics.py`
   - Added a reusable script to plot per-epoch trend curves from metrics JSON files.

Run order after this patch:

1. Do not rerun notebook 00 unless the curve dataset tar is missing or corrupted.
2. Run notebook 04 in debug mode first.
3. Run notebook 04 short10 after debug succeeds.
4. Run notebook 05 in debug mode first.
5. Run notebook 05 short10 only after notebook 04 is stable.
6. Run notebook 08 to evaluate saved tars, plot trends, and optionally profile video.
