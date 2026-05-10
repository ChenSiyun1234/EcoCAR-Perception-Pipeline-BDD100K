from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd


KEYS = [
    'val/det_loss',
    'val/det/metric_map50',
    'val/lane_loss_scaled',
    'val/lane/unweighted_total',
    'val/lane/cls',
    'val/lane/reg',
    'val/lane/line_iou',
    'val/lane/mask_aux',
    'val/lane_exist_acc',
    'val/lane_exist_f1',
    'val/lane_exist_best_f1',
    'val/lane_exist_best_threshold',
    'val/lane_point_mae',
    'val/matched_line_iou',
    'val/lane/clrkd_style_f1',
    'val/lane/decoded_f1',
    'val/lane/decoded_precision',
    'val/lane/decoded_recall',
    'val/lane/decoded_pred_count',
    'val/lane/decoded_avg_score',
    'val/lane/decoded_oracle_f1',
    'val/lane/decoded_oracle_precision',
    'val/lane/decoded_oracle_recall',
]


def read_metrics(path: Path) -> List[Dict[str, float]]:
    if path.is_file() and path.suffix == '.json':
        return json.loads(path.read_text())
    if path.is_file() and path.suffix == '.tar':
        with tarfile.open(path, 'r') as tf:
            member = next((m for m in tf.getmembers() if m.name.endswith('metrics.json')), None)
            if member is None:
                return []
            f = tf.extractfile(member)
            if f is None:
                return []
            return json.loads(f.read().decode('utf-8'))
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description='Plot Stage2 experiment trends and write summary CSV.')
    parser.add_argument('--inputs', nargs='+', required=True, help='metrics.json files or experiment .tar files')
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for raw in args.inputs:
        p = Path(raw)
        exp = p.stem.replace('_metrics', '')
        metrics = read_metrics(p)
        for row in metrics:
            flat = {'experiment': exp, 'epoch': row.get('epoch')}
            for key in KEYS:
                if key in row:
                    flat[key] = row[key]
            rows.append(flat)
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit('No metrics rows found.')
    df.to_csv(out_dir / 'stage2_trend_summary.csv', index=False)
    for key in KEYS:
        if key not in df.columns:
            continue
        plt.figure()
        for exp, sub in df.groupby('experiment'):
            sub = sub.sort_values('epoch')
            plt.plot(sub['epoch'], sub[key], marker='o', label=exp)
        plt.xlabel('epoch')
        plt.ylabel(key)
        plt.title(key)
        plt.legend()
        plt.grid(True, alpha=0.3)
        name = key.replace('/', '_').replace('@', 'at') + '.png'
        plt.tight_layout()
        plt.savefig(out_dir / name, dpi=160)
        plt.close()
    print(f'wrote {out_dir}')


if __name__ == '__main__':
    main()
