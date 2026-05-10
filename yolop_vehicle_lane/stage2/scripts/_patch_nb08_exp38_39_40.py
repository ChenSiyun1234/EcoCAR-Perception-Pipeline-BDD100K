"""Append exp38 (Exp2NN), exp39 (Exp2OO), exp40 (Exp2PP) entries to NB08.

Idempotent: if entries are already present, nothing changes.
Run from repo root:
    python yolop_vehicle_lane/stage2/scripts/_patch_nb08_exp38_39_40.py
"""
from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
NB_PATH = REPO / 'yolop_vehicle_lane' / 'stage2' / 'notebooks' / 'stage2_notebook_08_joint_eval_visualization_and_profile.ipynb'

EVAL_ENTRIES = [
    (
        'exp38_rmt_gca_anchor_mask_consistency_joint',
        ['short20', 'debug', ''],
    ),
    (
        'exp39_rmt_gca_anchor_hungarian_joint',
        ['short20', 'debug', ''],
    ),
    (
        'exp40_rmt_gca_anchor_width1_long30_joint',
        ['short30', 'debug', ''],
    ),
]


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
    out_tag = stem.split('_', 1)[0]  # e.g. 'exp38'
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
    sentinel_anchor = (
        "    {\n"
        "        'config': 'stage2/configs/exp37_rmt_gca_anchor_dual_score_joint.yaml',\n"
    )
    if sentinel_anchor not in new_text:
        # Fall back to exp35 (Exp2KK) anchor if exp37 wasn't yet added by the
        # previous patch script.
        sentinel_anchor = (
            "    {\n"
            "        'config': 'stage2/configs/exp35_rmt_gca_anchor_asl_amp_joint.yaml',\n"
        )
    # Find the END of the sentinel block (next "    },\n" after the anchor)
    idx = new_text.find(sentinel_anchor)
    if idx < 0:
        raise RuntimeError('Could not find anchor in eval cell.')
    end_idx = new_text.find("    },\n", idx)
    if end_idx < 0:
        raise RuntimeError('Could not find end of anchor block.')
    insertion_point = end_idx + len("    },\n")
    new_blocks = ''
    for stem, suffixes in EVAL_ENTRIES:
        if stem in new_text:
            continue
        new_blocks += _eval_entry(stem, suffixes)
    if not new_blocks:
        return text
    return new_text[:insertion_point] + new_blocks + new_text[insertion_point:]


def patch_plot_cell(text: str) -> str:
    new_text = text
    insert_after = (
        "    '/content/drive/MyDrive/EcoCAR/training_runs/"
        "exp37_rmt_gca_anchor_dual_score_joint_short20_metrics.json',\n"
    )
    if insert_after not in new_text:
        insert_after = (
            "    '/content/drive/MyDrive/EcoCAR/training_runs/"
            "exp35_rmt_gca_anchor_asl_amp_joint_short20_metrics.json',\n"
        )
    if insert_after not in new_text:
        raise RuntimeError('Could not find anchor in plot cell.')

    additions = ''
    for stem, _suffixes in EVAL_ENTRIES:
        run_tag = 'short30' if stem.endswith('long30_joint') else 'short20'
        if f'{stem}_{run_tag}_metrics.json' in new_text:
            continue
        additions += _plot_entry(stem, run_tag)
    if not additions:
        return text
    return new_text.replace(insert_after, insert_after + additions, 1)


def patch_video_cell(text: str) -> str:
    new_text = text
    anchor = "PROFILE_CANDIDATES = [\n"
    if anchor not in new_text:
        return text
    additions = ''
    for stem, _suffixes in EVAL_ENTRIES:
        run_tag = 'short30' if stem.endswith('long30_joint') else 'short20'
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
