
v2 dataset fix changes:
- Read EcoCAR/paths_config.yaml for the raw BDD root exactly like the DETR line.
- If raw BDD is missing locally, auto-extract Drive downloads/bdd100k_labels.zip and
  bdd100k_images_100k.zip back into /content/bdd100k_raw.
- Notebook00 no longer requires masks to exist before resolving bdd100k_vehicle5.
