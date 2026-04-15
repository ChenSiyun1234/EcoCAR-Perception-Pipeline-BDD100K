
What was fixed:
1. 00_rebuild_dataset_and_lane_cache.ipynb now uses DETR-style Drive/local-SSD dataset resolution.
2. lib/utils/lane_render.py now accepts legacy argument aliases used by older notebooks:
   output_dir, img_w, img_h, mask_w, mask_h.
3. Training/eval notebooks now resolve bdd100k_vehicle5 from Drive instead of assuming /content already exists.
