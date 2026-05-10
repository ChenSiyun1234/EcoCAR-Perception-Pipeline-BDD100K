"""Generate NB46 (Exp2QQ), NB47 (Exp2RR), NB48 (Exp2SS) by cloning NB40.

Run from repo root:
    python yolop_vehicle_lane/stage2/scripts/_make_nb46_47_48.py
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


_SMOKE_TEMPLATE = (
    "from pathlib import Path\n"
    "import os, sys\n"
    "\n"
    "CONFIG = '{config}'\n"
    "LOG_FILE = os.path.join(LOG_DIR, f'{{Path(CONFIG).stem}}_smoke.log')\n"
    "run_streaming([sys.executable, '-u', 'stage2/scripts/smoke_test_joint_models.py', CONFIG], log_path=LOG_FILE)"
)


def _train_cell(config: str, run_tag: str, epochs: int, limit_train, limit_val: int = 1000) -> str:
    if limit_train is None:
        train_arg = "    # full dataset (no --limit-train)\n"
        limit_block = "LIMIT_TRAIN = None\n"
    else:
        train_arg = "    '--limit-train', str(LIMIT_TRAIN),\n"
        limit_block = f"LIMIT_TRAIN = {limit_train}\n"
    return (
        "from pathlib import Path\n"
        "import os, sys\n"
        "\n"
        f"CONFIG = '{config}'\n"
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
        f"    RUN_TAG = '{run_tag}'\n"
        f"    EPOCHS = {epochs}\n"
        "    BATCH_SIZE = 8\n"
        f"    {limit_block}"
        f"    LIMIT_VAL = {limit_val}\n"
        "    PRINT_EVERY = 50\n"
        "\n"
        "run_stem = Path(CONFIG).stem + '_' + RUN_TAG\n"
        "WORK_DIR = f'/content/{run_stem}'\n"
        "OUTPUT_TAR = f'/content/drive/MyDrive/EcoCAR/training_runs/{run_stem}.tar'\n"
        "LOG_FILE = os.path.join(LOG_DIR, f'{run_stem}_train.log')\n"
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
        + (f"{train_arg}" if limit_train is not None else "")
        + "    '--limit-val', str(LIMIT_VAL),\n"
        "    '--force-extract',\n"
        "    '--print-every', str(PRINT_EVERY),\n"
        "]\n"
        "if LIMIT_TRAIN is not None:\n"
        "    cmd.extend(['--limit-train', str(LIMIT_TRAIN)])\n"
        "\n"
        "print('DEBUG_MODE:', DEBUG_MODE, flush=True)\n"
        "print('LIMIT_TRAIN:', LIMIT_TRAIN, flush=True)\n"
        "print('About to run:', ' '.join(cmd), flush=True)\n"
        "print('Output tar:', OUTPUT_TAR, flush=True)\n"
        "print('Visible log file:', LOG_FILE, flush=True)\n"
        "run_streaming(cmd, log_path=LOG_FILE)"
    )


def write_nb46() -> Path:
    nb = _new_nb_from_template(TEMPLATE)
    nb['cells'][0]['source'] = (
        "# Stage 2 Notebook 46 - Exp2QQ Anchor + Varifocal Loss + IoU regression\n"
        "\n"
        "**The closed-form fix for the cls collapse.** After 7 experiments (NB39-45) trying every loss tweak, "
        "matching change, mask scoring, and capacity scale-up, the cls head was still stuck at pos-neg gap = 0.01 "
        "and decoded_f1 = 0.04. I worked out the math on the ASL equilibrium with our actual settings and found "
        "the bug: `focal_alpha = 0.25`.\n"
        "\n"
        "alpha=0.25 is the RetinaNet recommendation for object detection where positives are 1 in 1000. For our "
        "192-anchor lane head, the ratio is only 1:37 -- much less imbalanced. With alpha=0.25, ASL gamma_neg=4, "
        "and clip=0.05, the loss has zero gradient at sigmoid ~ 0.578 for ALL priors, which matches exactly what "
        "we observed across NB39-45 (val_lane_cls = 0.089, pos_score = 0.578, neg_score = 0.571).\n"
        "\n"
        "Exp2QQ uses Varifocal Loss (Zhang et al. CVPR 2021, used by RTMDet/VarifocalNet) on the continuous "
        "LineIoU regression target:\n"
        "- Positives: weight = target_iou (no `(1-alpha)` discount; rare positives carry full weight)\n"
        "- Negatives: weight = alpha * sigmoid^gamma (only confident-wrong negatives count; uniform sigmoid ~ 0.5 "
        "gets weight `alpha * 0.25 ~ 0.19`, much smaller than positive's `0.7+`)\n"
        "\n"
        "This breaks the symmetric equilibrium quantitatively: positives now have ~ 4x the gradient of negatives, "
        "so sigmoid is pulled UP for matched priors and DOWN for unmatched ones until they actually separate.\n"
        "\n"
        "Reference: Zhang et al. 'VarifocalNet: An IoU-aware Dense Object Detector' CVPR 2021. The combination of "
        "VFL + continuous IoU target is the published recipe behind RTMDet's COCO-2022 dominance.\n"
        "\n"
        "Single config diff vs Exp2KK (NB40):\n"
        "- `cls_target_type: matched_existence -> lineiou_regression`\n"
        "- `cls_loss_type: asl -> vfl`\n"
        "- `vfl_alpha: 0.75`, `vfl_gamma: 2.0`\n"
        "All other settings = NB40 (anchor head, dynamic-k, mask aux, cosine LR, AMP, 20 epochs)."
    )
    nb['cells'][1]['source'] = (
        "### Run mode\n"
        "\n"
        "1. `DEBUG_MODE = True` for the first run -- VFL is a new code path; smoke first.\n"
        "2. After smoke + debug pass, change to `False` for the 20-epoch short run.\n"
        "3. AMP keeps wall-clock ~ 30 minutes for 20 epochs at 3000 samples.\n"
        "4. Output mirrored to notebook cell, Colab runtime log, Drive log file.\n"
        "5. Independent of NB35-45; only depends on the dataset tar."
    )
    nb['cells'][3]['source'] = _SMOKE_TEMPLATE.format(
        config='stage2/configs/exp41_rmt_gca_anchor_vfl_iou_regression_joint.yaml'
    )
    nb['cells'][4]['source'] = _train_cell(
        config='stage2/configs/exp41_rmt_gca_anchor_vfl_iou_regression_joint.yaml',
        run_tag='short20', epochs=20, limit_train=3000,
    )
    nb['cells'][5]['source'] = (
        "## What to watch in Exp2QQ training\n"
        "\n"
        "Reference NB45 (anchor head + ASL + width 1.0 + 30 epochs): pos-neg gap 0.008, cls_only_f1 0.038.\n"
        "Reference NB40 (anchor head + ASL + width 0.5 + 20 epochs): pos-neg gap 0.010, decoded_f1 0.043.\n"
        "\n"
        "Pass criteria at epoch 20:\n"
        "- **`pos_score - neg_score >= 0.10`** -- this is the smoking gun. If VFL fixes the equilibrium, the cls "
        "scores should separate by at least 10x the historical gap. If gap >= 0.20, even better.\n"
        "- **`val/lane/decoded_f1 >= 0.15`** (cls ranking) -- 3x NB40, because cls is now actually discriminative.\n"
        "- **`val/matched_line_iou >= 0.45`** -- preserves geometry; VFL should not interfere with the geometric "
        "losses since it only modifies the cls weighting.\n"
        "- **`val/lane/decoded_oracle_f1 >= 0.40`** -- oracle ceiling stays high.\n"
        "- `val_lane_cls` should be HIGHER than NB40's 0.089 in early epochs (because positives now carry full "
        "weight and pull harder on the matched priors), then DECREASE as cls actually learns.\n"
        "\n"
        "Failure signals:\n"
        "- pos-neg gap still < 0.05: VFL alpha=0.75 too gentle. Try alpha=0.95.\n"
        "- decoded_f1 < 0.05 with high pos-neg gap: cls is now discriminative but pointing at the wrong priors. "
        "Reduce `match_cost_cls` to 0.5 so matching is driven by IoU not cls.\n"
        "- matched_iou drops below 0.40: VFL is somehow distorting geometry training. Lower `w_cls` to 3.0.\n"
        "\n"
        "If decoded_f1 jumps to >= 0.15 AND pos-neg gap >= 0.10, **Exp2QQ is the new champion** and we have a "
        "working cls signal for the first time."
    )
    out = NOTEBOOKS / 'stage2_notebook_46_exp2qq_anchor_vfl_iou_regression_joint.ipynb'
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    return out


def write_nb47() -> Path:
    nb = _new_nb_from_template(TEMPLATE)
    nb['cells'][0]['source'] = (
        "# Stage 2 Notebook 47 - Exp2RR DETR K=64 queries + Hungarian + VFL\n"
        "\n"
        "**Architectural escape from the 192-anchor design.** Across NB39-45 the per-prior ROI feature could not "
        "discriminate among 192 anchors that all sample similar lane-y regions. The cls collapse was not a loss "
        "issue (NB39-42), not a matching issue (NB44), not a capacity issue (NB45), not a representation issue "
        "(NB43). It might be intrinsic to the geometric-prior design itself.\n"
        "\n"
        "Exp2RR abandons geometric priors and uses DETR-style learned queries instead. K=64 (vs LaneQueryHead "
        "default K=12 used in NB22-26) is large enough to cover the dataset's max ~10 lanes per image with "
        "comfortable slack for Hungarian 1-to-1 matching. Each query is a learned vector with no preset "
        "geometric bias -- queries differentiate via cross-attention to the multi-scale feature map. With "
        "Hungarian 1-to-1, each query gets at most ONE GT lane in any image, so the cls labels are stable per "
        "query across batches.\n"
        "\n"
        "Combined with the new VFL recipe from Exp2QQ, this tests the orthogonal architectural angle.\n"
        "\n"
        "Config:\n"
        "- `lane_head.type: clrkd -> query`\n"
        "- `lane_head.num_queries: 64`\n"
        "- `lane_assigner: dynamic_k -> hungarian`\n"
        "- `cls_loss_type: asl -> vfl` (same recipe as Exp2QQ)\n"
        "- All else = NB40."
    )
    nb['cells'][1]['source'] = (
        "### Run mode\n"
        "\n"
        "1. `DEBUG_MODE = True` smoke first -- larger query head + new loss path.\n"
        "2. `DEBUG_MODE = False` for 20 epochs short run.\n"
        "3. Wall-clock ~ 30 min on Colab Pro+.\n"
        "4. Independent of NB35-46."
    )
    nb['cells'][3]['source'] = _SMOKE_TEMPLATE.format(
        config='stage2/configs/exp42_rmt_gca_query64_hungarian_vfl_joint.yaml'
    )
    nb['cells'][4]['source'] = _train_cell(
        config='stage2/configs/exp42_rmt_gca_query64_hungarian_vfl_joint.yaml',
        run_tag='short20', epochs=20, limit_train=3000,
    )
    nb['cells'][5]['source'] = (
        "## What to watch in Exp2RR training\n"
        "\n"
        "Pass criteria at epoch 20:\n"
        "- **`val/lane/decoded_f1 >= 0.15`**.\n"
        "- **`pos_score - neg_score >= 0.20`** -- DETR queries with 1-to-1 Hungarian and VFL should produce a "
        "very clean cls separation since each query has a deterministic single-GT label or a stable no-object "
        "label.\n"
        "- **`val/matched_line_iou >= 0.30`** -- query head's geometry is typically lower than anchor head, but "
        "we want at least the level of NB22 (matched_iou = 0.42 with K=12 queries) -- ideally higher with K=64.\n"
        "- **`val/lane/decoded_oracle_f1 >= 0.30`**.\n"
        "\n"
        "Failure signals:\n"
        "- matched_iou < 0.20: K=64 queries can't ground themselves on the dataset (too many no-object slots, "
        "weak signal). Try `num_queries=32`.\n"
        "- decoded_f1 ~ pos-neg gap small: cls couldn't separate even with the cleanest matching. Confirms the "
        "192-anchor cls collapse is per-prior-feature-bound and we need richer features (Exp2QQ + bigger "
        "backbone OR pre-trained init)."
    )
    out = NOTEBOOKS / 'stage2_notebook_47_exp2rr_query64_hungarian_vfl_joint.ipynb'
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    return out


def write_nb48() -> Path:
    nb = _new_nb_from_template(TEMPLATE)
    nb['cells'][0]['source'] = (
        "# Stage 2 Notebook 48 - Exp2SS Full 70K dataset + VFL on anchor head\n"
        "\n"
        "**Test the data-scale hypothesis with the FIXED loss recipe.** The dataset visualization confirmed 70K "
        "training samples are healthy (mean 5.81 lanes/image, max_lanes=10 truncates 7.8%). Earlier full-dataset "
        "runs (NB28 / NB38) used the broken alpha=0.25 ASL recipe and showed lane plateau + det degradation.\n"
        "\n"
        "Exp2SS retries the data scale with the Exp2QQ VFL+IoU regression recipe so we can compare:\n"
        "- 3000 samples + VFL: Exp2QQ (NB46)\n"
        "- 70K samples + VFL: Exp2SS (this notebook)\n"
        "\n"
        "If VFL + 70K data unlocks the cls signal, decoded_f1 should reach >= 0.30 (CLRKDNet small-CULane "
        "territory). If 70K + VFL doesn't help past Exp2QQ, the bottleneck is architecture, not data.\n"
        "\n"
        "70K images / 8 batch = 8750 iter/epoch. 6 epochs * 8750 = 52500 iters total (vs 7500 in 20-epoch 3K runs). "
        "That's 7x more iter and each iter sees 23x more variety. Wall-clock ~ 60-80 minutes on RTX Pro 6000.\n"
        "\n"
        "Config:\n"
        "- Same as Exp2QQ (anchor + VFL + IoU regression).\n"
        "- `train.end_epoch: 20 -> 6` (we get the same total loss steps from full data).\n"
        "- LIMIT_TRAIN flag REMOVED in the train cell -> uses full 70K split."
    )
    nb['cells'][1]['source'] = (
        "### Run mode\n"
        "\n"
        "1. `DEBUG_MODE = True` smoke first.\n"
        "2. `DEBUG_MODE = False` for the full-dataset run. Note `LIMIT_TRAIN = None` is the toggle.\n"
        "3. Wall-clock ~ 60-80 min on RTX Pro 6000.\n"
        "4. Independent of all prior NBs except for shared codebase.\n"
        "5. If you hit OOM, drop batch_size to 6."
    )
    nb['cells'][3]['source'] = _SMOKE_TEMPLATE.format(
        config='stage2/configs/exp43_rmt_gca_anchor_vfl_full_dataset_joint.yaml'
    )
    nb['cells'][4]['source'] = _train_cell(
        config='stage2/configs/exp43_rmt_gca_anchor_vfl_full_dataset_joint.yaml',
        run_tag='full6', epochs=6, limit_train=None, limit_val=2000,
    )
    nb['cells'][5]['source'] = (
        "## What to watch in Exp2SS training\n"
        "\n"
        "Pass criteria at epoch 6:\n"
        "- **`val/lane/decoded_f1 >= 0.20`** -- 5x NB40, beating Exp2QQ if data scale matters.\n"
        "- **`val/matched_line_iou >= 0.50`** -- with 23x more data and proper VFL, geometry should be near "
        "the project ceiling (NB45 hit 0.525 with width 1.0; Exp2SS with width 0.5 + 70K data should match).\n"
        "- **`val/lane/decoded_oracle_f1 >= 0.45`**.\n"
        "- `val_det` should DECREASE monotonically (in NB38 it INCREASED from 2.07 to 2.77 -- a sign of joint "
        "conflict that should NOT recur with VFL since the lane gradient is now in a non-degenerate regime).\n"
        "\n"
        "Failure signals:\n"
        "- decoded_f1 < 0.10 even with 70K data: the cls representation bottleneck is intrinsic to the 192-anchor "
        "design. Exp2RR's K=64 query head is the right path.\n"
        "- val_det INCREASES like NB38 did: joint conflict is unrelated to cls. Lower lambda_det to 0.5 in a "
        "follow-up experiment.\n"
        "- OOM at full dataset: drop batch_size to 6 (`--batch-size 6`)."
    )
    out = NOTEBOOKS / 'stage2_notebook_48_exp2ss_anchor_vfl_full_dataset_joint.ipynb'
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    return out


if __name__ == '__main__':
    for fn in (write_nb46, write_nb47, write_nb48):
        out = fn()
        print('wrote:', out)
