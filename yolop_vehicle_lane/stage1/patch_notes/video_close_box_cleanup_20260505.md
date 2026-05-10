# Stage1 Notebook 07 Close-Vehicle Box Cleanup

## Problem
In the stage1 video profiler, nearby vehicles could be drawn with several boxes on the same vehicle body. The issue is mostly a visualization/post-processing issue rather than a lane issue:

- the profiler uses a permissive confidence threshold so unfinished checkpoints can still draw boxes;
- regular NMS is class-aware by default, so class-confused duplicates can survive;
- close vehicles occupy a large image area, so part-level boxes may not overlap enough to be removed by normal IoU NMS.

## Changes

1. Added `lib/utils/video_box_cleanup.py`.
   - clips boxes to the original frame;
   - removes tiny invalid boxes;
   - suppresses fragment boxes contained in a larger close-vehicle box;
   - keeps separate small/far vehicles when they are not contained in a larger close object.

2. Updated `stage1/notebooks/07_a5000_video_profile.ipynb`.
   - changed video NMS to `agnostic=True`;
   - changed the video NMS threshold from `0.40` to `0.35`;
   - changed video confidence threshold from `0.20` to `0.25`;
   - added close-vehicle cleanup before tracking;
   - added cleanup counters to the output JSON summary.

## Output fields to check

The notebook summary now includes:

- `nms_agnostic`
- `close_vehicle_cleanup`
- `cleanup_input_boxes`
- `cleanup_output_boxes`
- `cleanup_suppressed_fragments`

If close-vehicle splitting is fixed, `cleanup_suppressed_fragments` should be greater than zero on videos where the bug previously appeared.
