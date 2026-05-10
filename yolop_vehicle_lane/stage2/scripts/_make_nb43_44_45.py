"""Generate NB43 (Exp2NN), NB44 (Exp2OO), NB45 (Exp2PP) by cloning the NB40 structure.

Run from repo root:
    python yolop_vehicle_lane/stage2/scripts/_make_nb43_44_45.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
NOTEBOOKS = REPO / 'yolop_vehicle_lane' / 'stage2' / 'notebooks'
TEMPLATE = NOTEBOOKS / 'stage2_notebook_40_exp2kk_anchor_asl_amp_joint.ipynb'


def _new_nb_from_template(template_path: Path):
    nb = json.loads(template_path.read_text(encoding='utf-8'))
    for cell in nb['cells']:
        if cell.get('cell_type') == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
    return nb


_TRAIN_TEMPLATE = (
    "from pathlib import Path\n"
    "import os, sys\n"
    "\n"
    "CONFIG = '{config}'\n"
    "CURVE_TAR = '/content/drive/MyDrive/EcoCAR/datasets/bdd100k_clrkd_curve.tar'\n"
    "CURVE_ROOT = '/content/bdd100k_clrkd_curve'\n"
    "\n"
    "DEBUG_MODE = False\n"
    "\n"
    "if DEBUG_MODE:\n"
    "    RUN_TAG = 'debug'\n"
    "    EPOCHS = 2\n"
    "    BATCH_SIZE = 4\n"
    "    LIMIT_TRAIN = 512\n"
    "    LIMIT_VAL = 256\n"
    "    PRINT_EVERY = 5\n"
    "else:\n"
    "    RUN_TAG = '{run_tag}'\n"
    "    EPOCHS = {epochs}\n"
    "    BATCH_SIZE = 8\n"
    "    LIMIT_TRAIN = 3000\n"
    "    LIMIT_VAL = 1000\n"
    "    PRINT_EVERY = 5\n"
    "\n"
    "run_stem = Path(CONFIG).stem + '_' + RUN_TAG\n"
    "WORK_DIR = f'/content/{{run_stem}}'\n"
    "OUTPUT_TAR = f'/content/drive/MyDrive/EcoCAR/training_runs/{{run_stem}}.tar'\n"
    "LOG_FILE = os.path.join(LOG_DIR, f'{{run_stem}}_train.log')\n"
    "\n"
    "cmd = [\n"
    "    sys.executable, '-u', 'stage2/scripts/train_joint_model_experiment.py',\n"
    "    '--config', CONFIG,\n"
    "    '--curve-tar', CURVE_TAR,\n"
    "    '--curve-root', CURVE_ROOT,\n"
    "    '--work-dir', WORK_DIR,\n"
    "    '--output-tar', OUTPUT_TAR,\n"
    "    '--epochs', str(EPOCHS),\n"
    "    '--batch-size', str(BATCH_SIZE),\n"
    "    '--limit-train', str(LIMIT_TRAIN),\n"
    "    '--limit-val', str(LIMIT_VAL),\n"
    "    '--force-extract',\n"
    "    '--print-every', str(PRINT_EVERY),\n"
    "]\n"
    "\n"
    "print('DEBUG_MODE:', DEBUG_MODE, flush=True)\n"
    "print('About to run:', ' '.join(cmd), flush=True)\n"
    "print('Output tar:', OUTPUT_TAR, flush=True)\n"
    "print('Visible log file:', LOG_FILE, flush=True)\n"
    "run_streaming(cmd, log_path=LOG_FILE)"
)


_SMOKE_TEMPLATE = (
    "from pathlib import Path\n"
    "import os, sys\n"
    "\n"
    "CONFIG = '{config}'\n"
    "LOG_FILE = os.path.join(LOG_DIR, f'{{Path(CONFIG).stem}}_smoke.log')\n"
    "run_streaming([sys.executable, '-u', 'stage2/scripts/smoke_test_joint_models.py', CONFIG], log_path=LOG_FILE)"
)


def write_nb43() -> Path:
    nb = _new_nb_from_template(TEMPLATE)
    nb['cells'][0]['source'] = (
        "# Stage 2 Notebook 43 - Exp2NN Anchor + mask-consistency decode + cls self-distill\n"
        "\n"
        "**The biggest leap so far: drop the cls task from the decode bottleneck.** Across NB39 (focal), NB40 (ASL), "
        "NB41 (lineiou regression+QFL) and NB42 (dual cls+iou), the cls head's pos-vs-neg score gap stayed within "
        "0.002 to 0.01 -- the cls is performing at chance level on the 192-anchor head. Geometry is excellent "
        "(matched_iou=0.508, oracle_f1=0.459 in NB41) but decoded_f1 plateaued at 0.04 because we cannot rank the "
        "192 priors by the cls head's output.\n"
        "\n"
        "Exp2NN replaces cls-based ranking entirely:\n"
        "\n"
        "1. **Inference-side**: `eval.decoded_score_source = cls_x_mask`. For each prior, sample the auxiliary mask "
        "sigmoid along the predicted curve and use the mean as a ranking score. The aux mask is trained on per-pixel "
        "BCE+Dice with no per-prior matching instability, so its sigmoid is a clean per-prior geometric verifier. "
        "Hybrid score = sigmoid(cls) * mask_consistency keeps any cls signal that does separate priors.\n"
        "2. **Training-side**: `cls_target_type = mask_consistency`. cls is self-distilled to predict the same "
        "mask-along-curve score. This gives cls a stable, deterministic, batch-stable supervision target -- the "
        "antidote to the matching instability that has been collapsing it.\n"
        "3. **Companion diagnostics**: the train script automatically reports `decoded_cls_only_f1` and "
        "`decoded_mask_only_f1` alongside the primary `decoded_f1`. So one notebook produces a clean three-way "
        "comparison of cls / mask / cls*mask rankings.\n"
        "\n"
        "Code changes (new since NB42):\n"
        "- `losses.py`: new `_compute_mask_consistency_target` plus a `cls_target_type='mask_consistency'` branch in "
        "`_forward_single_stage` that supervises cls with the mask-along-curve score (BCE or QFL).\n"
        "- `metrics/lane_f1_decoded.py`: new `score_source='mask_consistency'` and `'cls_x_mask'`. "
        "`_mask_consistency_score` samples the head's `mask_logit` along each prior's `coord_pred`.\n"
        "- `train_joint_model_experiment.py`: `_cls_only` and `_mask_only` companion metrics auto-fire when the "
        "primary score source is a hybrid.\n"
        "\n"
        "Reference: this is conceptually similar to RTMDet's centerness * cls product, except we replace the cls "
        "with a segmentation-derived score that is robust to per-prior label noise."
    )

    nb['cells'][1]['source'] = (
        "### Run mode\n"
        "\n"
        "1. Keep `DEBUG_MODE = True` for the first run -- the new mask-consistency code path needs a smoke check.\n"
        "2. After smoke + debug pass, change to `False` for the 20-epoch short run.\n"
        "3. AMP keeps wall-clock ~ 30 minutes for 20 epochs at 3000 samples.\n"
        "4. Output mirrored to notebook cell, Colab runtime log, Drive log file.\n"
        "5. Do not rerun NB00. Independent of NB35 / NB39-42; only depends on the dataset tar."
    )

    nb['cells'][3]['source'] = _SMOKE_TEMPLATE.format(
        config='stage2/configs/exp38_rmt_gca_anchor_mask_consistency_joint.yaml'
    )
    nb['cells'][4]['source'] = _TRAIN_TEMPLATE.format(
        config='stage2/configs/exp38_rmt_gca_anchor_mask_consistency_joint.yaml',
        run_tag='short20',
        epochs=20,
    )

    nb['cells'][5]['source'] = (
        "## What to watch in Exp2NN training\n"
        "\n"
        "Reference NB41 (anchor + lineiou regression + QFL): matched_iou=0.508, oracle_f1=0.459, decoded_f1=0.029, "
        "pos-neg gap = 0.002.\n"
        "Reference NB42 (anchor + dual cls*iou): matched_iou=0.490, oracle_f1=0.426, decoded_f1=0.042 (cls*iou), "
        "decoded_cls_only_f1=0.012.\n"
        "\n"
        "Pass criteria at epoch 20:\n"
        "- **`val/lane/decoded_f1 >= 0.20`** (cls * mask). 5x NB42, 7x NB41. The mask-consistency score should "
        "actually rank priors meaningfully.\n"
        "- **`val/lane/decoded_mask_only_f1 >= 0.18`**. The mask alone (no cls) should already give most of the gain.\n"
        "- **`val/lane/decoded_cls_only_f1 >= 0.05`**. cls is now self-distilled to mimic the mask, so it should "
        "ALSO rank meaningfully (3x its NB42 value), proving the supervision target was the bottleneck.\n"
        "- **`val/matched_line_iou >= 0.45`** (preserve geometric champion).\n"
        "- **`val/lane/decoded_oracle_f1 >= 0.40`** (oracle ceiling stays high).\n"
        "\n"
        "Failure signals:\n"
        "- mask_only_f1 ~ cls_only_f1 ~ 0.05: the mask is also not discriminative on this dataset. Inspect "
        "val_lane_mask trend (should fall from 0.55 to ~ 0.40 over training).\n"
        "- decoded_cls_only_f1 << decoded_mask_only_f1: self-distillation didn't take. Bump `w_cls` to 7.0 or "
        "switch `cls_loss_type: qfl`.\n"
        "- matched_iou drops below 0.40: the mask supervision is somehow distorting geometry. Lower `w_mask` to 1.0.\n"
        "\n"
        "If decoded_f1 jumps to >= 0.20, the cls collapse on the anchor head was the structural bottleneck and "
        "Exp2NN is the new champion. The mask is geometry-aware and free of the matching-instability."
    )

    out = NOTEBOOKS / 'stage2_notebook_43_exp2nn_anchor_mask_consistency_joint.ipynb'
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    return out


def write_nb44() -> Path:
    nb = _new_nb_from_template(TEMPLATE)
    nb['cells'][0]['source'] = (
        "# Stage 2 Notebook 44 - Exp2OO Anchor + Hungarian 1-to-1 matching\n"
        "\n"
        "**Targeted fix for matching instability.** Across NB39/40/41/42 the dynamic_k matcher labels K=4 priors "
        "as positive per GT lane in each batch -- but WHICH K=4 priors gets re-decided every batch based on the "
        "current IoU contest. The same prior is positive in batch A, negative in batch B. cls cannot learn "
        "discriminative scores from contradictory labels and collapses to near-uniform sigmoid (pos-neg gap < 0.01 "
        "across all four runs).\n"
        "\n"
        "Exp2OO replaces dynamic_k with strict Hungarian 1-to-1: each GT lane is matched to exactly ONE prior, "
        "deterministically minimizing the cost matrix. The label per prior is now stable across batches "
        "(approximately -- still depends on which prior wins the cost contest, but with 1-to-1 there's a unique "
        "winner). DETR-style.\n"
        "\n"
        "Single config diff vs Exp2KK (NB40):\n"
        "- `lane_assigner: dynamic_k -> hungarian`\n"
        "\n"
        "Plus the new `decoded_score_source = cls_x_mask` so we get the same three-way diagnostic as NB43 "
        "(`decoded_f1`, `decoded_cls_only_f1`, `decoded_mask_only_f1`) -- this lets us isolate matching-stability "
        "effects from the mask-consistency effects of NB43."
    )

    nb['cells'][1]['source'] = (
        "### Run mode\n"
        "\n"
        "1. Keep `DEBUG_MODE = True` for the first run.\n"
        "2. After smoke + debug pass, change to `False` for the 20-epoch short run.\n"
        "3. AMP keeps wall-clock ~ 30 minutes for 20 epochs at 3000 samples.\n"
        "4. Output mirrored to notebook cell, Colab runtime log, Drive log file.\n"
        "5. Do not rerun NB00. Independent of any prior NB; only depends on the dataset tar."
    )

    nb['cells'][3]['source'] = _SMOKE_TEMPLATE.format(
        config='stage2/configs/exp39_rmt_gca_anchor_hungarian_joint.yaml'
    )
    nb['cells'][4]['source'] = _TRAIN_TEMPLATE.format(
        config='stage2/configs/exp39_rmt_gca_anchor_hungarian_joint.yaml',
        run_tag='short20',
        epochs=20,
    )

    nb['cells'][5]['source'] = (
        "## What to watch in Exp2OO training\n"
        "\n"
        "Pass criteria at epoch 20:\n"
        "- **`val/lane/decoded_cls_only_f1 >= 0.10`**. THIS is the hypothesis test for Exp2OO: with stable Hungarian "
        "labels, cls should now rank priors at least 2-3x better than NB40's 0.043 cls-only ranking. If it does, "
        "matching instability was the cls collapse cause.\n"
        "- **`pos_score - neg_score >= 0.10`**. Direct measurement of cls separation.\n"
        "- **`val/matched_line_iou >= 0.40`** (preserve geometry; Hungarian only matches num_GT priors per image so "
        "the rest get pure-negative gradient -- if this hurts geometry too much, dynamic_k was helping after all).\n"
        "- **`val/lane/decoded_f1 >= 0.10`** with `cls_x_mask` ranking.\n"
        "\n"
        "Failure signals:\n"
        "- pos-neg gap still < 0.02: matching instability was not the root cause; the per-prior feature is "
        "fundamentally not discriminative. (Confirms that Exp2NN's mask-consistency angle is the right one.)\n"
        "- matched_iou drops below 0.30: 1-to-1 matching is starving the geometry; only 5 priors per image get "
        "gradient on point regression. Switch to `match_cost_iou: 4.0` to bias matching toward best-fit priors."
    )

    out = NOTEBOOKS / 'stage2_notebook_44_exp2oo_anchor_hungarian_joint.ipynb'
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    return out


def write_nb45() -> Path:
    nb = _new_nb_from_template(TEMPLATE)
    nb['cells'][0]['source'] = (
        "# Stage 2 Notebook 45 - Exp2PP Anchor width 1.0 + 30 epochs scale-up\n"
        "\n"
        "**The capacity hypothesis.** Every loss tweak (NB39 focal, NB40 ASL, NB41 lineiou+QFL, NB42 dual head) "
        "plateaued at oracle_f1 ~ 0.45 and decoded_f1 ~ 0.04. The model has been width=0.5, embed_dim=128, 20 "
        "epochs across all four runs. With matched_iou and oracle_f1 still climbing at epoch 20 in NB41 (0.490 -> "
        "0.508 in the last 5 epochs) and NB40 (0.479 -> 0.483), the model is undertrained AND underweight.\n"
        "\n"
        "Exp2PP scales up two axes simultaneously while keeping NB40's stable loss recipe:\n"
        "\n"
        "- `model.width: 0.5 -> 1.0` (full RMT-PPAD width)\n"
        "- `lane_head.embed_dim: 128 -> 192` (larger per-prior feature so cls has more discriminative capacity)\n"
        "- `lane_head.roi_mid_channels: 48 -> 64`\n"
        "- `train.end_epoch: 20 -> 30` (cosine LR amortizes a longer decay window)\n"
        "- `train.lr_scheduler.warmup_epochs: 2 -> 3`\n"
        "\n"
        "Loss / matching / dataset are unchanged from NB40 (Exp2KK). This isolates the capacity question.\n"
        "\n"
        "GPU mem: NB40 used 10.4 GB / 95.6 GB peak. Width 1.0 + embed 192 should land ~ 25-35 GB, well within budget."
    )

    nb['cells'][1]['source'] = (
        "### Run mode\n"
        "\n"
        "1. Keep `DEBUG_MODE = True` for the first run -- the wider model needs a smoke check.\n"
        "2. After smoke + debug pass, change to `False` for the 30-epoch short run.\n"
        "3. AMP keeps wall-clock ~ 45 minutes for 30 epochs at 3000 samples (estimated; ~ 50% slower than NB40).\n"
        "4. Output mirrored to notebook cell, Colab runtime log, Drive log file.\n"
        "5. Do not rerun NB00. Independent of any prior NB; only depends on the dataset tar."
    )

    nb['cells'][3]['source'] = _SMOKE_TEMPLATE.format(
        config='stage2/configs/exp40_rmt_gca_anchor_width1_long30_joint.yaml'
    )
    nb['cells'][4]['source'] = _TRAIN_TEMPLATE.format(
        config='stage2/configs/exp40_rmt_gca_anchor_width1_long30_joint.yaml',
        run_tag='short30',
        epochs=30,
    )

    nb['cells'][5]['source'] = (
        "## What to watch in Exp2PP training\n"
        "\n"
        "Pass criteria at epoch 30:\n"
        "- **`val/matched_line_iou >= 0.55`** (NB41 hit 0.508 with width 0.5; doubling capacity + 50% more epochs "
        "should add 0.05).\n"
        "- **`val/lane/decoded_oracle_f1 >= 0.50`** (NB41 oracle was 0.459 still climbing).\n"
        "- **`val/lane/decoded_f1 (cls_x_mask) >= 0.15`**.\n"
        "- `train_total` curve should still be decreasing at epoch 30 (no plateau) -- if it plateaus by epoch 20, "
        "capacity was the bottleneck and we should retrain with width 1.5.\n"
        "\n"
        "Failure signals:\n"
        "- oracle_f1 plateau at 0.46 by epoch 30: capacity was NOT the bottleneck; the architecture itself can't "
        "improve geometry past this point on width-0.5-equivalent representations.\n"
        "- pos-neg gap still < 0.02: even at width 1.0, the cls task is fundamentally unsolvable by per-prior ROI "
        "features. (Confirms Exp2NN's mask-consistency direction.)\n"
        "- GPU OOM: drop batch_size to 4 or set width back to 0.75."
    )

    out = NOTEBOOKS / 'stage2_notebook_45_exp2pp_anchor_width1_long30_joint.ipynb'
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    return out


if __name__ == '__main__':
    for fn in (write_nb43, write_nb44, write_nb45):
        out = fn()
        print('wrote:', out)
