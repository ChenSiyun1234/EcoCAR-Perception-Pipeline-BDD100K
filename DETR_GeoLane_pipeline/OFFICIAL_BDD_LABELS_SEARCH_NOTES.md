This patch changes lane-label discovery to follow the official BDD100K old labels package first.

Priority order now:
1. bdd100k/labels/bdd100k_labels_images_train.json / val.json
2. bdd100k/labels/lane/polygons/lane_train.json / val.json

Why:
- The official ucbdrive/bdd100k bdd2coco.py reads bdd100k_labels_images_train.json and val.json.
- Lane poly2d is stored inside labels[*].poly2d in the official label format.

The notebook debug cell now prints zip candidates, extraction root, chosen JSON, and the actual JSON structure.
