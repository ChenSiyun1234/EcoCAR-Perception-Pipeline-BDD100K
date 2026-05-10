"""Append exp36 (Exp2LL) and exp37 (Exp2MM) entries to NB08 eval/plot/video cells.

Idempotent: if the entries are already present, nothing changes.

Run from repo root:
    python yolop_vehicle_lane/stage2/scripts/_patch_nb08_exp36_37.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
NB_PATH = REPO / 'yolop_vehicle_lane' / 'stage2' / 'notebooks' / 'stage2_notebook_08_joint_eval_visualization_and_profile.ipynb'


# --- Eval cell 3: append two new EVAL_ITEMS dict literals before the closing
# `]` of the EVAL_ITEMS list. We anchor on the exp35 entry's closing `},`.
EVAL_ENTRY_LL = (
    "    {\n"
    "        'config': 'stage2/configs/exp36_rmt_gca_anchor_iou_regression_joint.yaml',\n"
    "        'candidates': [\n"
    "            '/content/drive/MyDrive/EcoCAR/training_runs/exp36_rmt_gca_anchor_iou_regression_joint_short20.tar',\n"
    "            '/content/drive/MyDrive/EcoCAR/training_runs/exp36_rmt_gca_anchor_iou_regression_joint_debug.tar',\n"
    "            '/content/drive/MyDrive/EcoCAR/training_runs/exp36_rmt_gca_anchor_iou_regression_joint.tar',\n"
    "        ],\n"
    "    },\n"
)
EVAL_ENTRY_MM = (
    "    {\n"
    "        'config': 'stage2/configs/exp37_rmt_gca_anchor_dual_score_joint.yaml',\n"
    "        'candidates': [\n"
    "            '/content/drive/MyDrive/EcoCAR/training_runs/exp37_rmt_gca_anchor_dual_score_joint_short20.tar',\n"
    "            '/content/drive/MyDrive/EcoCAR/training_runs/exp37_rmt_gca_anchor_dual_score_joint_debug.tar',\n"
    "            '/content/drive/MyDrive/EcoCAR/training_runs/exp37_rmt_gca_anchor_dual_score_joint.tar',\n"
    "        ],\n"
    "    },\n"
)

PLOT_ENTRY_LL = (
    "    '/content/drive/MyDrive/EcoCAR/training_runs/exp36_rmt_gca_anchor_iou_regression_joint_short20_metrics.json',\n"
)
PLOT_ENTRY_MM = (
    "    '/content/drive/MyDrive/EcoCAR/training_runs/exp37_rmt_gca_anchor_dual_score_joint_short20_metrics.json',\n"
)

VIDEO_ENTRY_LL = (
    "    ('/content/drive/MyDrive/EcoCAR/training_runs/exp36_rmt_gca_anchor_iou_regression_joint_short20.tar',"
    " 'stage2/configs/exp36_rmt_gca_anchor_iou_regression_joint.yaml', 'video_profile_exp36'),\n"
)
VIDEO_ENTRY_MM = (
    "    ('/content/drive/MyDrive/EcoCAR/training_runs/exp37_rmt_gca_anchor_dual_score_joint_short20.tar',"
    " 'stage2/configs/exp37_rmt_gca_anchor_dual_score_joint.yaml', 'video_profile_exp37'),\n"
)


def _to_text(src) -> str:
    return ''.join(src) if isinstance(src, list) else src


def _to_lines(text: str):
    # Notebook source is canonically a list of lines, each ending in \n except
    # the last. Keep that shape so json diffs stay clean.
    if not text:
        return []
    parts = text.split('\n')
    out = [p + '\n' for p in parts[:-1]]
    if parts[-1] != '':
        out.append(parts[-1])
    return out


def patch_eval_cell(text: str) -> str:
    if "exp36_rmt_gca_anchor_iou_regression_joint" in text:
        return text  # already patched
    anchor = (
        "    {\n"
        "        'config': 'stage2/configs/exp35_rmt_gca_anchor_asl_amp_joint.yaml',\n"
        "        'candidates': [\n"
        "            '/content/drive/MyDrive/EcoCAR/training_runs/exp35_rmt_gca_anchor_asl_amp_joint_short20.tar',\n"
        "            '/content/drive/MyDrive/EcoCAR/training_runs/exp35_rmt_gca_anchor_asl_amp_joint_debug.tar',\n"
        "            '/content/drive/MyDrive/EcoCAR/training_runs/exp35_rmt_gca_anchor_asl_amp_joint.tar',\n"
        "        ],\n"
        "    },\n"
    )
    if anchor not in text:
        raise RuntimeError('Could not find exp35 anchor in NB08 cell 3.')
    return text.replace(anchor, anchor + EVAL_ENTRY_LL + EVAL_ENTRY_MM, 1)


def patch_plot_cell(text: str) -> str:
    if "exp36_rmt_gca_anchor_iou_regression_joint_short20_metrics.json" in text:
        return text
    anchor = (
        "    '/content/drive/MyDrive/EcoCAR/training_runs/"
        "exp35_rmt_gca_anchor_asl_amp_joint_short20_metrics.json',\n"
    )
    if anchor not in text:
        raise RuntimeError('Could not find exp35 metrics anchor in NB08 cell 5.')
    return text.replace(anchor, anchor + PLOT_ENTRY_LL + PLOT_ENTRY_MM, 1)


def patch_video_cell(text: str) -> str:
    if "exp36_rmt_gca_anchor_iou_regression_joint_short20.tar" in text:
        return text
    # Find a stable insertion anchor: the start of PROFILE_CANDIDATES list.
    anchor = "PROFILE_CANDIDATES = [\n"
    if anchor not in text:
        return text  # silently skip if the cell was restructured
    return text.replace(anchor, anchor + VIDEO_ENTRY_MM + VIDEO_ENTRY_LL, 1)


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding='utf-8'))

    cell3_src = _to_text(nb['cells'][3]['source'])
    cell5_src = _to_text(nb['cells'][5]['source'])
    cell7_src = _to_text(nb['cells'][7]['source'])

    new3 = patch_eval_cell(cell3_src)
    new5 = patch_plot_cell(cell5_src)
    new7 = patch_video_cell(cell7_src)

    nb['cells'][3]['source'] = _to_lines(new3)
    nb['cells'][5]['source'] = _to_lines(new5)
    nb['cells'][7]['source'] = _to_lines(new7)

    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    print('NB08 patched in place:', NB_PATH)
    print(' eval changed:', new3 != cell3_src)
    print(' plot changed:', new5 != cell5_src)
    print(' video changed:', new7 != cell7_src)


if __name__ == '__main__':
    main()
