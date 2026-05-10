"""Append exp44 (Exp2TT), exp45 (Exp2UU), exp46 (Exp2VV) entries to NB08.

Idempotent. Run from repo root:
    python yolop_vehicle_lane/stage2/scripts/_patch_nb08_exp44_45_46.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
NB_PATH = REPO / 'yolop_vehicle_lane' / 'stage2' / 'notebooks' / 'stage2_notebook_08_joint_eval_visualization_and_profile.ipynb'

EVAL_ENTRIES = [
    ('exp44_rmt_gca_hybrid_anchor_query_vfl_joint', ['short20', 'debug', '']),
    ('exp45_rmt_gca_query64_vfl_full_dataset_joint', ['full6', 'debug', '']),
    ('exp46_rmt_gca_anchor_vfl_full_data_det_rescue_joint', ['full6', 'debug', '']),
]
PLOT_TAG = {
    'exp44_rmt_gca_hybrid_anchor_query_vfl_joint': 'short20',
    'exp45_rmt_gca_query64_vfl_full_dataset_joint': 'full6',
    'exp46_rmt_gca_anchor_vfl_full_data_det_rescue_joint': 'full6',
}


def _eval_entry(stem: str, suffixes) -> str:
    cands = []
    for s in suffixes:
        suffix = f'_{s}' if s else ''
        cands.append(
            f"            '/content/drive/MyDrive/EcoCAR/training_runs/{stem}{suffix}.tar',\n"
        )
    return (
        f"    {{\n"
        f"        'config': 'stage2/configs/{stem}.yaml',\n"
        f"        'candidates': [\n"
        + ''.join(cands)
        + "        ],\n"
        + "    },\n"
    )


def _plot_entry(stem: str, run_tag: str) -> str:
    return f"    '/content/drive/MyDrive/EcoCAR/training_runs/{stem}_{run_tag}_metrics.json',\n"


def _video_entry(stem: str, run_tag: str) -> str:
    out_tag = stem.split('_', 1)[0]
    return (
        f"    ('/content/drive/MyDrive/EcoCAR/training_runs/{stem}_{run_tag}.tar',"
        f" 'stage2/configs/{stem}.yaml', 'video_profile_{out_tag}'),\n"
    )


def _to_text(src):
    return ''.join(src) if isinstance(src, list) else src


def _to_lines(text: str):
    if not text:
        return []
    parts = text.split('\n')
    out = [p + '\n' for p in parts[:-1]]
    if parts[-1] != '':
        out.append(parts[-1])
    return out


def patch_eval_cell(text: str) -> str:
    new_text = text
    candidates = [
        "    {\n        'config': 'stage2/configs/exp43_rmt_gca_anchor_vfl_full_dataset_joint.yaml',\n",
        "    {\n        'config': 'stage2/configs/exp42_rmt_gca_query64_hungarian_vfl_joint.yaml',\n",
        "    {\n        'config': 'stage2/configs/exp40_rmt_gca_anchor_width1_long30_joint.yaml',\n",
        "    {\n        'config': 'stage2/configs/exp35_rmt_gca_anchor_asl_amp_joint.yaml',\n",
    ]
    anchor = next((c for c in candidates if c in new_text), None)
    if anchor is None:
        raise RuntimeError('No anchor in eval cell.')
    idx = new_text.find(anchor)
    end_idx = new_text.find("    },\n", idx) + len("    },\n")
    blocks = ''
    for stem, suffixes in EVAL_ENTRIES:
        if stem in new_text:
            continue
        blocks += _eval_entry(stem, suffixes)
    if not blocks:
        return text
    return new_text[:end_idx] + blocks + new_text[end_idx:]


def patch_plot_cell(text: str) -> str:
    new_text = text
    candidates = [
        "    '/content/drive/MyDrive/EcoCAR/training_runs/exp43_rmt_gca_anchor_vfl_full_dataset_joint_full6_metrics.json',\n",
        "    '/content/drive/MyDrive/EcoCAR/training_runs/exp42_rmt_gca_query64_hungarian_vfl_joint_short20_metrics.json',\n",
        "    '/content/drive/MyDrive/EcoCAR/training_runs/exp40_rmt_gca_anchor_width1_long30_joint_short30_metrics.json',\n",
        "    '/content/drive/MyDrive/EcoCAR/training_runs/exp35_rmt_gca_anchor_asl_amp_joint_short20_metrics.json',\n",
    ]
    anchor = next((c for c in candidates if c in new_text), None)
    if anchor is None:
        raise RuntimeError('No anchor in plot cell.')
    additions = ''
    for stem, _suf in EVAL_ENTRIES:
        run_tag = PLOT_TAG[stem]
        key = f'{stem}_{run_tag}_metrics.json'
        if key in new_text:
            continue
        additions += _plot_entry(stem, run_tag)
    if not additions:
        return text
    return new_text.replace(anchor, anchor + additions, 1)


def patch_video_cell(text: str) -> str:
    new_text = text
    anchor = "PROFILE_CANDIDATES = [\n"
    if anchor not in new_text:
        return text
    additions = ''
    for stem, _suf in EVAL_ENTRIES:
        run_tag = PLOT_TAG[stem]
        if f'{stem}_{run_tag}.tar' in new_text:
            continue
        additions += _video_entry(stem, run_tag)
    if not additions:
        return text
    return new_text.replace(anchor, anchor + additions, 1)


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding='utf-8'))
    cell3 = _to_text(nb['cells'][3]['source'])
    cell5 = _to_text(nb['cells'][5]['source'])
    cell7 = _to_text(nb['cells'][7]['source'])
    new3 = patch_eval_cell(cell3)
    new5 = patch_plot_cell(cell5)
    new7 = patch_video_cell(cell7)
    nb['cells'][3]['source'] = _to_lines(new3)
    nb['cells'][5]['source'] = _to_lines(new5)
    nb['cells'][7]['source'] = _to_lines(new7)
    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
    print('NB08 patched in place:', NB_PATH)
    print(' eval changed:', new3 != cell3)
    print(' plot changed:', new5 != cell5)
    print(' video changed:', new7 != cell7)


if __name__ == '__main__':
    main()
