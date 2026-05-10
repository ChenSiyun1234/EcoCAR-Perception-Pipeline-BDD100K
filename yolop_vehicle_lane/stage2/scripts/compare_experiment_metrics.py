from __future__ import annotations

import argparse
import json
import tarfile
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
from typing import Dict, List


def _load_metrics_from_tar(tar_path: Path) -> List[Dict[str, float]]:
    if not tar_path.exists():
        raise FileNotFoundError(tar_path)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        with tarfile.open(tar_path, 'r') as tar:
            members = [m for m in tar.getmembers() if Path(m.name).name == 'metrics.json']
            if not members:
                raise FileNotFoundError(f'metrics.json not found inside {tar_path}')
            tar.extract(members[0], path=tmp_path)
        metric_path = next(tmp_path.rglob('metrics.json'))
        return json.loads(metric_path.read_text(encoding='utf-8'))


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return 'n/a'
    try:
        return f'{float(value):.{digits}f}'
    except Exception:
        return 'n/a'


def main() -> None:
    parser = argparse.ArgumentParser(description='Compare Stage 2 experiment metrics from result tar files.')
    parser.add_argument('--tars', nargs='+', required=True)
    parser.add_argument('--output-json', default=None)
    parser.add_argument('--plot-dir', default=None, help='Optional directory for trend plots.')
    args = parser.parse_args()

    rows = []
    for item in args.tars:
        tar_path = Path(item)
        if not tar_path.exists():
            print(f'[missing] {tar_path}')
            continue
        metrics = _load_metrics_from_tar(tar_path)
        if not metrics:
            print(f'[empty] {tar_path}')
            continue
        first = metrics[0]
        last = metrics[-1]
        rows.append({
            'experiment': tar_path.stem,
            'epochs': len(metrics),
            'first_val_det_loss': first.get('val/det_loss'),
            'last_val_det_loss': last.get('val/det_loss'),
            'first_val_lane_loss': first.get('val/lane_loss'),
            'last_val_lane_loss': last.get('val/lane_loss'),
            'first_lane_exist_acc': first.get('val/lane_exist_acc'),
            'last_lane_exist_acc': last.get('val/lane_exist_acc'),
            'first_lane_point_mae': first.get('val/lane_point_mae'),
            'last_lane_point_mae': last.get('val/lane_point_mae'),
            'last_lane_exist_precision': last.get('val/lane_exist_precision'),
            'last_lane_exist_recall': last.get('val/lane_exist_recall'),
            'last_lane_exist_f1': last.get('val/lane_exist_f1'),
            'last_matched_line_iou': last.get('val/matched_line_iou'),
            'last_false_positive_lane_slots': last.get('val/false_positive_lane_slots'),
            'last_false_negative_lane_slots': last.get('val/false_negative_lane_slots'),
            'last_grad_cosine': last.get('train/grad_cosine'),
            'last_det_gate_mean': last.get('train/gate/det_mean'),
            'last_lane_gate_mean': last.get('train/gate/lane_mean'),
        })

    headers = [
        'experiment', 'epochs', 'last_val_det_loss', 'last_val_lane_loss',
        'last_lane_exist_acc', 'last_lane_point_mae', 'last_grad_cosine',
        'last_det_gate_mean', 'last_lane_gate_mean', 'last_lane_exist_f1', 'last_matched_line_iou'
    ]
    print('\nStage 2 experiment comparison')
    print('-' * 120)
    print(f'{headers[0]:<34} {headers[1]:>6} {headers[2]:>17} {headers[3]:>18} {headers[4]:>20} {headers[5]:>20} {headers[6]:>17} {headers[7]:>18} {headers[8]:>19}')
    print('-' * 120)
    for r in rows:
        print(
            f'{r["experiment"]:<34} {r["epochs"]:>6} '
            f'{_fmt(r["last_val_det_loss"]):>17} {_fmt(r["last_val_lane_loss"]):>18} '
            f'{_fmt(r["last_lane_exist_acc"]):>20} {_fmt(r["last_lane_point_mae"]):>20} '
            f'{_fmt(r["last_grad_cosine"], 6):>17} {_fmt(r["last_det_gate_mean"]):>18} {_fmt(r["last_lane_gate_mean"]):>19}'
        )
        print(
            f'    lane_f1={_fmt(r.get("last_lane_exist_f1"))} '
            f'precision={_fmt(r.get("last_lane_exist_precision"))} '
            f'recall={_fmt(r.get("last_lane_exist_recall"))} '
            f'matched_line_iou={_fmt(r.get("last_matched_line_iou"))} '
            f'fp_slots={_fmt(r.get("last_false_positive_lane_slots"), 1)} '
            f'fn_slots={_fmt(r.get("last_false_negative_lane_slots"), 1)}'
        )
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(rows, indent=2), encoding='utf-8')
        print(f'Wrote {out}')

    if args.plot_dir:
        plot_dir = Path(args.plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_keys = [
            ('val/det_loss', 'Validation detection loss'),
            ('val/lane_loss', 'Validation lane loss'),
            ('val/lane_point_mae', 'Validation lane point MAE'),
            ('val/lane_exist_f1', 'Validation lane existence F1'),
            ('val/matched_line_iou', 'Validation matched LineIoU'),
        ]
        for key, title in plot_keys:
            plt.figure()
            for item in args.tars:
                tar_path = Path(item)
                if not tar_path.exists():
                    continue
                metrics = _load_metrics_from_tar(tar_path)
                xs = [int(m.get('epoch', i + 1)) for i, m in enumerate(metrics)]
                ys = [m.get(key) for m in metrics]
                xs = [x for x, y in zip(xs, ys) if y is not None]
                ys = [float(y) for y in ys if y is not None]
                if xs and ys:
                    plt.plot(xs, ys, marker='o', label=tar_path.stem)
            plt.title(title)
            plt.xlabel('epoch')
            plt.ylabel(key)
            plt.legend()
            plt.grid(True, alpha=0.3)
            out_png = plot_dir / (key.replace('/', '_') + '.png')
            plt.tight_layout()
            plt.savefig(out_png, dpi=160)
            plt.close()
            print(f'Wrote {out_png}')


if __name__ == '__main__':
    main()
