from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd

KEYS = [
    'val/det_loss',
    'val/lane_loss',
    'val/lane_loss_unscaled',
    'val/lane/unweighted_total',
    'val/lane/cls',
    'val/lane/reg',
    'val/lane/line_iou',
    'val/lane/mask_aux',
    'val/lane_point_mae',
    'val/matched_line_iou',
    'val/lane_exist_acc',
    'val/lane_exist_precision',
    'val/lane_exist_recall',
    'val/lane_exist_f1',
    'val/lane_exist_best_f1',
    'val/lane_exist_best_threshold',
    'val/false_positive_lane_slots',
    'val/false_negative_lane_slots',
    'val/lane/decoded_f1',
    'val/lane/decoded_precision',
    'val/lane/decoded_recall',
    'val/lane/decoded_pred_count',
    'val/lane/decoded_avg_score',
    'val/lane/decoded_oracle_f1',
    'val/lane/decoded_oracle_precision',
    'val/lane/decoded_oracle_recall',
    'train/lane/geometry_scale',
    'train/mtl/lambda_lane_runtime',
]


def load_metrics(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text(encoding='utf-8'))
    if isinstance(data, dict):
        data = data.get('history', data.get('metrics', data))
    df = pd.DataFrame(data)
    if 'epoch' not in df.columns:
        df.insert(0, 'epoch', range(1, len(df) + 1))
    df['experiment'] = path.stem.replace('_metrics', '')
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description='Plot Stage 2 per-epoch metric trends.')
    parser.add_argument('--metrics', nargs='+', required=True)
    parser.add_argument('--out-dir', default='/content/drive/MyDrive/EcoCAR/stage2/trend_plots')
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames: List[pd.DataFrame] = []
    for item in args.metrics:
        p = Path(item)
        if p.exists():
            frames.append(load_metrics(p))
    if not frames:
        raise FileNotFoundError('No metrics JSON files were found.')
    df = pd.concat(frames, ignore_index=True)
    summary_cols = ['experiment', 'epoch'] + [k for k in KEYS if k in df.columns]
    df[summary_cols].to_csv(out_dir / 'stage2_metric_trend_summary.csv', index=False)
    for key in KEYS:
        if key not in df.columns:
            continue
        plt.figure(figsize=(8, 5))
        for exp, sub in df.groupby('experiment'):
            sub = sub.sort_values('epoch')
            plt.plot(sub['epoch'], sub[key], marker='o', label=exp)
        plt.xlabel('epoch')
        plt.ylabel(key)
        plt.title(key)
        plt.legend(fontsize=8)
        plt.tight_layout()
        safe = key.replace('/', '_').replace('@', 'at')
        plt.savefig(out_dir / f'{safe}.png', dpi=160)
        plt.close()
    print(f'[plot] wrote plots and CSV to {out_dir}', flush=True)


if __name__ == '__main__':
    main()
