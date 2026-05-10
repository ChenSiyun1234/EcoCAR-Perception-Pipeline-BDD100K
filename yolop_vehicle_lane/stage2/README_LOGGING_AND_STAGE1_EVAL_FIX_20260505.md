# 2026-05-05 logging and Stage 1 eval directory fix

## Stage 2 training logs

All active Stage 2 training notebooks now execute Python in unbuffered mode and pass `--print-every 10` to `stage2/scripts/train_joint_model_experiment.py`.

The training script now prints:

- preflight config, dataset tar, local extraction root, work dir, output tar
- device and GPU name
- dataset sample counts and non-empty detection-label counts
- DataLoader batch counts
- model class, parameter count, and feature channels
- epoch phase and `waiting_for_first_batch`
- step logs at step 1 and every `print_every` steps
- validation start and validation done markers
- epoch metrics JSON

Notebook 00 also uses unbuffered Python for dataset preparation, and the dataset preparation scripts flush their status prints.

## Stage 1 notebook 03 fix

`lib/core/function.py` now uses `os.makedirs(save_dir, exist_ok=True)` for the validation visualization directory. This fixes notebook 03 when notebook 02 was stopped before the full 200 epochs and the metrics directory was never created.

`stage1/notebooks/03_eval_and_backbone_ablation.ipynb` also creates:

- `cfg.DRIVE.CHECKPOINT_DIR`
- `cfg.DRIVE.METRICS_DIR`
- `cfg.DRIVE.METRICS_DIR/visualization`

before calling `validate()`.
