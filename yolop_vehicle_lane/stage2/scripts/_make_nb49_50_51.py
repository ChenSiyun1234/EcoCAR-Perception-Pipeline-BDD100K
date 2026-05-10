"""Generate NB49 (Exp2TT), NB50 (Exp2UU), NB51 (Exp2VV) by cloning NB40.

Run from repo root:
    python yolop_vehicle_lane/stage2/scripts/_make_nb49_50_51.py
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
        limit_block = "LIMIT_TRAIN = None\n"
    else:
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
        "    '--limit-val', str(LIMIT_VAL),\n"
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


def write_nb49() -> Path:
    nb = _new_nb_from_template(TEMPLATE)
    nb['cells'][0]['source'] = (
        "# Stage 2 Notebook 49 - Exp2TT HybridPriorQueryHead (anchor stage 1 + K=12 query stage 2 with VFL)\n"
        "\n"
        "**The architectural breakthrough.** NB47 (K=64 query head + Hungarian + VFL) was the FIRST experiment "
        "in 10 attempts to break the cls collapse: pos-neg gap reached 0.099 (vs 0.01 historical), val_lane_f1 "
        "hit 0.246 (vs 0.0 prior), val_lane_best_f1 hit 0.344 (vs 0.05 prior). But its geometry was weak "
        "(matched_iou=0.27 vs anchor's 0.54 in NB48) because K=64 free-form queries can't cover lane space "
        "as densely as 192 geometrically-initialized priors.\n"
        "\n"
        "Meanwhile NB48 (anchor + VFL + full 70K data) hit project geometry record (matched_iou=0.544, "
        "oracle_f1=0.467) but its cls stayed at the dynamic-k matching collapse (gap=0.015).\n"
        "\n"
        "**The two heads have orthogonal strengths.** Exp2TT combines them via the existing `HybridPriorQueryHead`:\n"
        "\n"
        "- **Stage 1**: 192 anchor priors with dynamic-k matching, supervised by `stage1_aux_loss_weight=1.0`. "
        "Inherits NB48's geometry (matched_iou=0.54).\n"
        "- **Stage 2**: K=12 learned queries that cross-attend to stage 1's per-prior features AND the spatial "
        "feature map. Hungarian 1-to-1 matching + VFL. Inherits NB47's clean cls signal.\n"
        "- Inference uses stage 2's cls scores to rank, with stage 2's curves (which started from stage 1's "
        "geometry-rich feature pool) for the geometry.\n"
        "\n"
        "Reference: this is the `HybridPriorQueryHead` from NB22 (Exp2Q). NB22 failed because stage 1 had no "
        "direct supervision back then; we now have `stage1_aux_loss_weight` infrastructure (added for Exp2T) so "
        "stage 1 trains exactly like NB48's anchor head.\n"
        "\n"
        "Reference: combines DETR-style query supervision with anchor-style geometric priors, conceptually "
        "similar to Deformable DETR's reference points or DAB-DETR's anchor queries."
    )
    nb['cells'][1]['source'] = (
        "### Run mode\n"
        "\n"
        "1. `DEBUG_MODE = True` smoke first.\n"
        "2. `DEBUG_MODE = False` for 20 epochs at limit=3000.\n"
        "3. AMP keeps wall-clock ~ 30 minutes for 20 epochs at 3000 samples.\n"
        "4. Independent of all prior NBs."
    )
    nb['cells'][3]['source'] = _SMOKE_TEMPLATE.format(
        config='stage2/configs/exp44_rmt_gca_hybrid_anchor_query_vfl_joint.yaml'
    )
    nb['cells'][4]['source'] = _train_cell(
        config='stage2/configs/exp44_rmt_gca_hybrid_anchor_query_vfl_joint.yaml',
        run_tag='short20', epochs=20, limit_train=3000,
    )
    nb['cells'][5]['source'] = (
        "## What to watch in Exp2TT training\n"
        "\n"
        "Reference NB47 (K=64 query alone): pos-neg gap=0.099, val_lane_f1=0.246, matched_iou=0.27, decoded_f1=0.019.\n"
        "Reference NB48 (anchor + full data): pos-neg gap=0.015, val_lane_f1=0.07, matched_iou=0.544, decoded_f1=0.05.\n"
        "\n"
        "Pass criteria at epoch 20:\n"
        "- **`val/matched_line_iou >= 0.45`** -- inherit NB48-style geometry from stage 1 (with limit=3000, expect "
        "slightly below 0.544; 0.45 is a reasonable lower bound).\n"
        "- **`pos_score - neg_score >= 0.10`** on stage 2 -- inherit NB47-style cls discrimination via Hungarian.\n"
        "- **`val/lane_f1 >= 0.20` and `val/lane_best_f1 >= 0.30`** -- decisive evidence the cls is now ranking "
        "real lanes.\n"
        "- **`val/lane/decoded_f1 >= 0.10`** -- 2x NB48's 0.05 because cls is finally discriminative.\n"
        "- `lane/stage1_aux_total` should DECREASE monotonically -- confirms stage 1 is being supervised.\n"
        "\n"
        "Failure signals:\n"
        "- matched_iou < 0.30: hybrid stage 2 distorts geometry. Set `stage1_aux_loss_weight=2.0`, `aux_stage_loss_weight=0.0`.\n"
        "- pos-neg gap < 0.05: K=12 queries are too few; switch to K=24 in a follow-up.\n"
        "- val_det >> 2.5: hybrid head adds enough capacity that joint conflict resurges. Drop lambda_lane to 0.7."
    )
    out = NOTEBOOKS / 'stage2_notebook_49_exp2tt_hybrid_anchor_query_vfl_joint.ipynb'
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    return out


def write_nb50() -> Path:
    nb = _new_nb_from_template(TEMPLATE)
    nb['cells'][0]['source'] = (
        "# Stage 2 Notebook 50 - Exp2UU K=64 query head + VFL + full 70K dataset\n"
        "\n"
        "**Validate NB47 at scale.** NB47 with K=64 queries + Hungarian + VFL was the first cls-discriminating "
        "experiment (pos-neg gap=0.099, val_lane_f1=0.246, val_lane_best_f1=0.344) but its geometry was capped "
        "at matched_iou=0.27 because 3000 samples isn't enough to ground 64 free-form queries. NB48 showed full "
        "70K data lifts the anchor head's matched_iou from 0.525 to 0.544 -- a similar lift is plausible for "
        "K=64 queries.\n"
        "\n"
        "Single config diff vs NB47:\n"
        "- `train.end_epoch: 20 -> 6` (~7x more total iters from full data)\n"
        "- LIMIT_TRAIN flag REMOVED -> full 70K split\n"
        "- All else identical to NB47 (exp42 yaml).\n"
        "\n"
        "Wall-clock ~ 60-80 minutes on RTX Pro 6000."
    )
    nb['cells'][1]['source'] = (
        "### Run mode\n"
        "\n"
        "1. `DEBUG_MODE = True` smoke.\n"
        "2. `DEBUG_MODE = False` for full-dataset 6-epoch run.\n"
        "3. Wall-clock ~ 60-80 min.\n"
        "4. Independent of all prior NBs."
    )
    nb['cells'][3]['source'] = _SMOKE_TEMPLATE.format(
        config='stage2/configs/exp45_rmt_gca_query64_vfl_full_dataset_joint.yaml'
    )
    nb['cells'][4]['source'] = _train_cell(
        config='stage2/configs/exp45_rmt_gca_query64_vfl_full_dataset_joint.yaml',
        run_tag='full6', epochs=6, limit_train=None, limit_val=2000,
    )
    nb['cells'][5]['source'] = (
        "## What to watch in Exp2UU training\n"
        "\n"
        "Reference NB47 (K=64 + 3K data): val_lane_f1=0.246, val_lane_best_f1=0.344, matched_iou=0.27.\n"
        "\n"
        "Pass criteria at epoch 6 on full 70K:\n"
        "- **`val/matched_line_iou >= 0.40`** -- 1.5x NB47's 0.27, similar lift to what NB48 showed for the anchor "
        "head.\n"
        "- **`val/lane_f1 >= 0.30`** -- 1.2x NB47's 0.246, with 23x more data variety.\n"
        "- **`val/lane_best_f1 >= 0.40`**.\n"
        "- **`val/lane/decoded_f1 >= 0.10`** -- 5x NB47's 0.019, because better geometry now multiplies with the "
        "already-working cls.\n"
        "- `val/lane/decoded_oracle_f1 >= 0.30`.\n"
        "- pos-neg gap should stay >= 0.05 (don't lose the cls breakthrough).\n"
        "\n"
        "Failure signals:\n"
        "- matched_iou < 0.30: geometry didn't scale with data. Switch to Exp2TT's hybrid architecture.\n"
        "- val_det stalls like NB48 (>= 3.0): joint conflict at full data scale; need Exp2VV's lambda_det=2.0 fix."
    )
    out = NOTEBOOKS / 'stage2_notebook_50_exp2uu_query64_vfl_full_dataset_joint.ipynb'
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    return out


def write_nb51() -> Path:
    nb = _new_nb_from_template(TEMPLATE)
    nb['cells'][0]['source'] = (
        "# Stage 2 Notebook 51 - Exp2VV anchor + VFL + full 70K + det rescue (lambda_det=2.0)\n"
        "\n"
        "**Fix the joint-training conflict NB48 exposed.** NB48 (anchor + VFL + full data) hit project geometry "
        "record (matched_iou=0.544) but val_det stalled at 3.16 and val_map50 = 0.0 -- detection completely failed "
        "to fit. The lane gradient at 23x data variety dominated the shared backbone and starved the det branch. "
        "The grad_cos diagnostic from earlier experiments already showed lane and det gradients fight in late "
        "epochs; at full data scale the conflict becomes catastrophic.\n"
        "\n"
        "Diagnosis: Kendall uncertainty weighting (`use_uncertainty: true`) learns task weights from the loss "
        "magnitudes -- but at full data scale lane loss decreases faster than det loss, so the learned weight "
        "shifts AWAY from det, accelerating the imbalance. Fix: switch to fixed weighting and explicitly boost "
        "det.\n"
        "\n"
        "Config diff vs NB48 (Exp2SS):\n"
        "- `lambda_det: 1.0 -> 2.0`\n"
        "- `lambda_lane: 1.0 -> 0.7`\n"
        "- `use_uncertainty: true -> false`\n"
        "- All else identical to NB48 including VFL recipe + full dataset + 6 epochs.\n"
        "\n"
        "Reference: this is the standard 'fixed multi-task weighting' fallback when learnable weights run away "
        "(Sener & Koltun 2018 'Multi-Task Learning as Multi-Objective Optimization' discusses the failure mode)."
    )
    nb['cells'][1]['source'] = (
        "### Run mode\n"
        "\n"
        "1. `DEBUG_MODE = True` smoke.\n"
        "2. `DEBUG_MODE = False` for full-dataset 6-epoch run.\n"
        "3. Wall-clock ~ 60-80 min.\n"
        "4. Independent of all prior NBs."
    )
    nb['cells'][3]['source'] = _SMOKE_TEMPLATE.format(
        config='stage2/configs/exp46_rmt_gca_anchor_vfl_full_data_det_rescue_joint.yaml'
    )
    nb['cells'][4]['source'] = _train_cell(
        config='stage2/configs/exp46_rmt_gca_anchor_vfl_full_data_det_rescue_joint.yaml',
        run_tag='full6', epochs=6, limit_train=None, limit_val=2000,
    )
    nb['cells'][5]['source'] = (
        "## What to watch in Exp2VV training\n"
        "\n"
        "Reference NB48 (anchor + VFL + full data, uncertainty=true, lambda_det=1.0): matched_iou=0.544 "
        "(record), oracle_f1=0.467, decoded_f1=0.050, **val_det=3.16** (broken), **val_map50=0.0**.\n"
        "\n"
        "Pass criteria at epoch 6:\n"
        "- **`val_det <= 2.2`** and **`val/det/map50 >= 0.005`** -- detection actually trains at full data scale.\n"
        "- **`val/matched_line_iou >= 0.50`** -- preserve geometry from NB48 (small acceptable regression from "
        "0.544 because lane gradient is dialed back from lambda=1.0 to 0.7).\n"
        "- **`val/lane/decoded_oracle_f1 >= 0.40`**.\n"
        "- **`val/lane/decoded_f1 >= 0.04`** -- maintain NB48 level.\n"
        "- `train/grad_cosine_epoch_mean` should stay positive across most epochs -- confirms the conflict is "
        "fixed by the static rebalancing.\n"
        "\n"
        "Failure signals:\n"
        "- val_det still >= 3.0: lambda_det=2.0 not enough; raise to 3.0 in a follow-up.\n"
        "- matched_iou drops below 0.40: lambda_lane=0.7 too aggressive; raise to 0.9.\n"
        "- BOTH val_det and matched_iou regress: the joint conflict is structural and we need Exp2WW (PCGrad "
        "gradient surgery)."
    )
    out = NOTEBOOKS / 'stage2_notebook_51_exp2vv_anchor_vfl_full_data_det_rescue_joint.ipynb'
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    return out


if __name__ == '__main__':
    for fn in (write_nb49, write_nb50, write_nb51):
        out = fn()
        print('wrote:', out)
