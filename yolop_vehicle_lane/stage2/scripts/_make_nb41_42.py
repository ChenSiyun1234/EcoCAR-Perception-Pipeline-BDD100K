"""Generate NB41 (Exp2LL) and NB42 (Exp2MM) by cloning the NB40 structure.

Run from repo root:
    python yolop_vehicle_lane/stage2/scripts/_make_nb41_42.py
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


def _set_source(cell, text: str) -> None:
    cell['source'] = text


def write_nb41() -> Path:
    nb = _new_nb_from_template(TEMPLATE)

    nb['cells'][0]['source'] = (
        "# Stage 2 Notebook 41 - Exp2LL Anchor + LineIoU regression + QFL\n"
        "\n"
        "**The cls-collapse fix.** NB40 (Exp2KK) confirmed the anchor head holds the project all-time best geometry: "
        "matched_iou=0.483, oracle_f1=0.418. But pred_lanes still 1536 (= 192 priors x 8 batch) and decoded_f1=0.043 "
        "-- the binary cls task is degenerate. ASL gamma_neg=4 (NB40) and plain focal (NB39) both failed to separate "
        "the 5 matched priors from the 187 unmatched ones because the matching outcome is itself unstable across "
        "batches: the same prior is positive in some batches and negative in others depending on which competing "
        "priors win the dynamic-k IoU contest.\n"
        "\n"
        "Exp2LL replaces binary {0, 1} matched_existence with continuous LineIoU regression: target = max LineIoU "
        "between this prior's CURRENT predicted curve (detached) and any valid GT lane in the same image. The cls "
        "head's output IS the ranking score at inference -- no more matching-instability confound.\n"
        "\n"
        "`cls_loss_type: qfl` (Quality Focal Loss) weights BCE by `|target - sigmoid(logit)|^gamma` so the 95 percent "
        "of priors with target ~ 0 get near-zero gradient and cannot collapse all logits to 0 the way Exp2K (plain BCE) did.\n"
        "\n"
        "All other settings = NB40 (anchor head, dynamic-k matching top-k=4, mask aux, cosine LR, uncertainty weighting, AMP, 20 epochs).\n"
        "\n"
        "Reference: Li et al. 'Generalized Focal Loss V2' (2021). The published-recipe combination of continuous IoU "
        "target + QFL is what gives RTMDet/GFL their dense-prediction headroom."
    )

    nb['cells'][1]['source'] = (
        "### Run mode\n"
        "\n"
        "1. Keep `DEBUG_MODE = True` for the first run.\n"
        "2. After smoke + debug pass, change to `False` for the 20-epoch short run.\n"
        "3. With AMP enabled, expect ~30 minutes wall-clock for 20 epochs at 3000 samples (matches NB40).\n"
        "4. Output mirrored to notebook cell, Colab runtime log, Drive log file.\n"
        "5. Do not rerun NB00."
    )

    # cell 2 (mount/install) is reused as-is.

    nb['cells'][3]['source'] = (
        "from pathlib import Path\n"
        "import os, sys\n"
        "\n"
        "CONFIG = 'stage2/configs/exp36_rmt_gca_anchor_iou_regression_joint.yaml'\n"
        "LOG_FILE = os.path.join(LOG_DIR, f'{Path(CONFIG).stem}_smoke.log')\n"
        "run_streaming([sys.executable, '-u', 'stage2/scripts/smoke_test_joint_models.py', CONFIG], log_path=LOG_FILE)"
    )

    nb['cells'][4]['source'] = (
        "from pathlib import Path\n"
        "import os, sys\n"
        "\n"
        "CONFIG = 'stage2/configs/exp36_rmt_gca_anchor_iou_regression_joint.yaml'\n"
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
        "    RUN_TAG = 'short20'\n"
        "    EPOCHS = 20\n"
        "    BATCH_SIZE = 8\n"
        "    LIMIT_TRAIN = 3000\n"
        "    LIMIT_VAL = 1000\n"
        "    PRINT_EVERY = 5\n"
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

    nb['cells'][5]['source'] = (
        "## What to watch in Exp2LL training\n"
        "\n"
        "Reference NB40 (anchor head, ASL cls, matched_existence): matched_iou=0.483, oracle_f1=0.418, decoded_f1=0.043, "
        "**pred_lanes=1536** (broken binary cls).\n"
        "\n"
        "Pass criteria at epoch 20:\n"
        "- **`pred_lanes < 400`** -- continuous QFL target should naturally produce a large gap between high-IoU and "
        "low-IoU priors so only ~50 priors per image cross the 0.3 decode threshold.\n"
        "- **`val/lane/decoded_f1 >= 0.15`** -- 3x NB40, because the cls output now IS the ranking score we want.\n"
        "- **`val/matched_line_iou >= 0.40`** -- preserves NB40's geometry champion.\n"
        "- **`val/lane/decoded_oracle_f1 >= 0.35`** -- oracle ceiling stays high; the regression target should not "
        "distort geometry training.\n"
        "- `train/grad_cosine_epoch_mean` logged each epoch (joint-conflict diagnostic).\n"
        "\n"
        "Failure signals:\n"
        "- decoded_f1 < 0.05 with low pred_lanes: QFL collapsed to all-zero (Exp2K failure mode). Drop `lineiou_target_pow` to 0.5.\n"
        "- matched_iou drops below 0.30: cls regression is dragging geometry. Lower `w_cls` to 3.0.\n"
        "- pred_lanes still > 1000: QFL didn't bite hard enough. Bump `qfl_gamma` to 3.0.\n"
        "- decoded_f1 ~ oracle_f1: cls is now the bottleneck-free path; Exp2LL is the new champion."
    )

    out = NOTEBOOKS / 'stage2_notebook_41_exp2ll_anchor_iou_regression_joint.ipynb'
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    return out


def write_nb42() -> Path:
    nb = _new_nb_from_template(TEMPLATE)

    nb['cells'][0]['source'] = (
        "# Stage 2 Notebook 42 - Exp2MM Anchor + dual cls x IoU score\n"
        "\n"
        "**Architectural fix for the cls collapse.** NB39/NB40 stuck at pred_lanes=1536 because the binary "
        "matched_existence task is structurally unstable on the 192-anchor head: cls output sigmoid is roughly "
        "uniform across all priors so the top-K decoder always picks ALL of them. Exp2LL (NB41) attacks the loss "
        "side; Exp2MM attacks the architecture.\n"
        "\n"
        "Exp2MM keeps binary cls AND adds a parallel `iou_logits` head trained on continuous LineIoU regression. "
        "At decode time, the score becomes:\n"
        "\n"
        "    score = sigmoid(cls_logits) * sigmoid(iou_logits)\n"
        "\n"
        "cls answers 'is this prior a match?' (binary, with the existing ASL supervision); iou answers 'if so, how "
        "good is the geometry?' (continuous, supervised by QFL on the LineIoU target). The matching-instability "
        "that has been collapsing the binary cls task no longer determines the decode rank, because iou regression "
        "has a deterministic per-prior target that does not depend on which competing prior wins the dynamic-k contest.\n"
        "\n"
        "Code changes (new since NB40):\n"
        "- `lane_head.py CLRKDLaneHead`: `dual_score: bool` flag adds a parallel `iou_head` MLP and emits `iou_logits` (B, P).\n"
        "- `losses.py FusionLossConfig`: `w_iou_aux`, `iou_aux_loss_type='qfl'`, `iou_aux_qfl_gamma=2.0`, `iou_aux_target_pow=1.0`.\n"
        "- `metrics/lane_f1_decoded.py`: `score_source='cls_x_iou'` ranks priors by sigmoid(cls)*sigmoid(iou).\n"
        "- `train_joint_model_experiment.py`: `eval.decoded_score_source` chooses the primary metric; when `cls_x_iou`, "
        "a `_cls_only` companion metric is also reported so we can attribute the gain.\n"
        "\n"
        "Reference: GFLv2 (Li et al. 2021) -- predicted IoU as ranking score is the published recipe behind RTMDet's "
        "post-COCO-2022 dominance."
    )

    nb['cells'][1]['source'] = (
        "### Run mode\n"
        "\n"
        "1. Keep `DEBUG_MODE = True` for the first run -- the dual_score path is the first time the model emits "
        "`iou_logits`, so confirm the smoke test prints non-NaN losses before running 20 epochs.\n"
        "2. After smoke + debug pass, change to `False` for the 20-epoch short run.\n"
        "3. With AMP enabled, expect ~30 minutes wall-clock for 20 epochs at 3000 samples.\n"
        "4. Output mirrored to notebook cell, Colab runtime log, Drive log file.\n"
        "5. Do not rerun NB00."
    )

    nb['cells'][3]['source'] = (
        "from pathlib import Path\n"
        "import os, sys\n"
        "\n"
        "CONFIG = 'stage2/configs/exp37_rmt_gca_anchor_dual_score_joint.yaml'\n"
        "LOG_FILE = os.path.join(LOG_DIR, f'{Path(CONFIG).stem}_smoke.log')\n"
        "run_streaming([sys.executable, '-u', 'stage2/scripts/smoke_test_joint_models.py', CONFIG], log_path=LOG_FILE)"
    )

    nb['cells'][4]['source'] = (
        "from pathlib import Path\n"
        "import os, sys\n"
        "\n"
        "CONFIG = 'stage2/configs/exp37_rmt_gca_anchor_dual_score_joint.yaml'\n"
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
        "    RUN_TAG = 'short20'\n"
        "    EPOCHS = 20\n"
        "    BATCH_SIZE = 8\n"
        "    LIMIT_TRAIN = 3000\n"
        "    LIMIT_VAL = 1000\n"
        "    PRINT_EVERY = 5\n"
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

    nb['cells'][5]['source'] = (
        "## What to watch in Exp2MM training\n"
        "\n"
        "Reference NB40 (anchor head, ASL cls, no iou_logits): matched_iou=0.483, oracle_f1=0.418, decoded_f1=0.043, "
        "**pred_lanes=1536**.\n"
        "\n"
        "New metrics this notebook produces:\n"
        "- `val/lane/iou_aux` -- training loss of the new iou regression head (smaller is better; expect 0.4 -> 0.05 over 20 epochs).\n"
        "- `val/lane/decoded_f1` -- ranks priors by sigmoid(cls) * sigmoid(iou). The primary number.\n"
        "- `val/lane/decoded_cls_only_f1` -- companion metric ranking by sigmoid(cls) alone, so we can attribute the gain.\n"
        "- `val/lane/decoded_oracle_f1` -- unchanged; F1 ceiling under perfect ranking.\n"
        "\n"
        "Pass criteria at epoch 20:\n"
        "- **`val/lane/decoded_f1 >= 0.20`** -- 4-5x NB40. The dual-score head should close most of the gap to oracle.\n"
        "- **`val/lane/decoded_f1 - val/lane/decoded_cls_only_f1 >= 0.10`** -- the iou re-ranking is doing real work; "
        "if this gap is < 0.05, the cls head dominates and dual scoring is wasted parameters.\n"
        "- **`pred_lanes < 400`** -- the iou_logits naturally have a wider score distribution so multiplying with cls "
        "produces a clear top-K cutoff.\n"
        "- **`val/matched_line_iou >= 0.40`** -- preserves NB40 geometry champion.\n"
        "- **`val/lane/decoded_oracle_f1 >= 0.35`** -- oracle ceiling stays high.\n"
        "- `[amp] kind=bfloat16 enabled=True` log line at start.\n"
        "\n"
        "Failure signals:\n"
        "- iou_aux loss does not decrease: iou head is decoupled from the geometry it should describe. Bump `w_iou_aux` to 3.0.\n"
        "- decoded_f1 ~ decoded_cls_only_f1: the iou head learns to mirror cls. Try `iou_aux_target_pow: 0.5` to expand low-IoU range.\n"
        "- matched_iou drops below 0.30: the dual_score MLP is competing with offset_head/param_head for embed_dim capacity. "
        "Increase embed_dim 128 -> 192 or set `cls_separate_path: true` to give cls + iou their own ROI features."
    )

    out = NOTEBOOKS / 'stage2_notebook_42_exp2mm_anchor_dual_score_joint.ipynb'
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    return out


if __name__ == '__main__':
    p41 = write_nb41()
    p42 = write_nb42()
    print('wrote:', p41)
    print('wrote:', p42)
