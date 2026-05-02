import os
import os.path as osp
import pickle as pkl
from pathlib import Path

import numpy as np

from .base_dataset import BaseDataset
from .registry import DATASETS

LIST_FILE = {
    'train': 'list/train_gt.txt',
    'val': 'list/val.txt',
    'test': 'list/test.txt',
}


@DATASETS.register_module
class BDDCurve(BaseDataset):
    def __init__(self, data_root, split, processes=None, cfg=None):
        super().__init__(data_root, split, processes=processes, cfg=cfg)
        self.list_path = osp.join(data_root, LIST_FILE[split])
        self.split = split
        self.eval_distance_px = getattr(cfg, 'eval_lane_distance_px', 20.0)
        self.load_annotations()
        print(f'Number of BDD curve images loaded: {len(self.data_infos)}')
        self.logger.info(f'Number of BDD curve images loaded: {len(self.data_infos)}')

    def load_annotations(self):
        self.logger.info('Loading BDD100K curve annotations...')
        os.makedirs('cache', exist_ok=True)
        cache_key = str(Path(self.data_root).resolve()).replace('/', '_').replace(':', '_')
        cache_path = f'cache/bdd_curve_{cache_key}_{self.split}.pkl'
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as cache_file:
                self.data_infos = pkl.load(cache_file)
            self.max_lanes = max([len(anno['lanes']) for anno in self.data_infos], default=0)
            return

        if not osp.exists(self.list_path):
            raise FileNotFoundError(f'Missing BDDCurve list file: {self.list_path}')

        self.data_infos = []
        with open(self.list_path, 'r', encoding='utf-8') as list_file:
            for raw_line in list_file:
                line = raw_line.strip().split()
                if not line:
                    continue
                self.data_infos.append(self.load_annotation(line))

        self.max_lanes = max([len(anno['lanes']) for anno in self.data_infos], default=0)
        with open(cache_path, 'wb') as cache_file:
            pkl.dump(self.data_infos, cache_file)

    def load_annotation(self, line):
        infos = {}
        img_line = line[0]
        img_line = img_line[1 if img_line[0] == '/' else 0:]
        img_path = osp.join(self.data_root, img_line)
        infos['img_name'] = img_line
        infos['img_path'] = img_path

        if len(line) > 1 and 'train' in self.split:
            mask_line = line[1]
            mask_line = mask_line[1 if mask_line[0] == '/' else 0:]
            infos['mask_path'] = osp.join(self.data_root, mask_line)
        else:
            stem, _ = osp.splitext(img_path)
            candidate_mask = stem.replace('/images/', '/masks/') + '.png'
            infos['mask_path'] = candidate_mask

        if len(line) > 2:
            infos['lane_exist'] = np.array([int(value) for value in line[2:]])

        anno_path = img_path[:-3] + 'lines.txt'
        lanes = []
        if osp.exists(anno_path):
            with open(anno_path, 'r', encoding='utf-8') as anno_file:
                rows = [row.strip() for row in anno_file.readlines() if row.strip()]
            for row in rows:
                values = [float(value) for value in row.split()]
                points = [(values[i], values[i + 1]) for i in range(0, len(values) - 1, 2)
                          if values[i] >= 0 and values[i + 1] >= 0]
                points = list(set(points))
                if len(points) > 2:
                    lanes.append(sorted(points, key=lambda point: point[1]))
        infos['lanes'] = lanes
        return infos

    @staticmethod
    def _resample(points, num=50):
        points = np.asarray(points, dtype=np.float32)
        if len(points) == 0:
            return np.zeros((0, 2), dtype=np.float32)
        if len(points) == 1:
            return np.repeat(points, num, axis=0)
        segment = np.linalg.norm(points[1:] - points[:-1], axis=1)
        cum = np.concatenate([[0.0], np.cumsum(segment)])
        total = cum[-1]
        if total < 1e-6:
            return np.repeat(points[:1], num, axis=0)
        targets = np.linspace(0.0, total, num)
        out = np.zeros((num, 2), dtype=np.float32)
        for i, target in enumerate(targets):
            idx = np.searchsorted(cum, target, side='right') - 1
            idx = int(np.clip(idx, 0, len(points) - 2))
            denom = max(float(cum[idx + 1] - cum[idx]), 1e-6)
            alpha = (target - cum[idx]) / denom
            out[i] = points[idx] * (1.0 - alpha) + points[idx + 1] * alpha
        return out

    @classmethod
    def _lane_distance(cls, pred, gt):
        pred_rs = cls._resample(pred, 50)
        gt_rs = cls._resample(gt, 50)
        if len(pred_rs) == 0 or len(gt_rs) == 0:
            return float('inf')
        return float(np.linalg.norm(pred_rs - gt_rs, axis=1).mean())

    def get_prediction_string(self, pred):
        out = []
        for lane in pred:
            arr = lane.to_array(self.cfg)
            if len(arr) <= 1:
                continue
            lane_str = ' '.join(['{:.3f} {:.3f}'.format(float(x), float(y)) for x, y in arr])
            if lane_str:
                out.append(lane_str)
        return '\n'.join(out)

    def evaluate(self, predictions, output_basedir):
        pred_dir = osp.join(output_basedir, 'bdd_curve_predictions')
        os.makedirs(pred_dir, exist_ok=True)

        tp = 0
        fp = 0
        fn = 0
        distances = []
        for idx, pred in enumerate(predictions):
            info = self.data_infos[idx]
            output_name = info['img_name'].replace('/', '_')[:-3] + 'lines.txt'
            with open(osp.join(pred_dir, output_name), 'w', encoding='utf-8') as out_file:
                out_file.write(self.get_prediction_string(pred))

            pred_lanes = [lane.to_array(self.cfg) for lane in pred]
            pred_lanes = [lane for lane in pred_lanes if len(lane) > 1]
            gt_lanes = [np.asarray(lane, dtype=np.float32) for lane in info['lanes'] if len(lane) > 1]

            if len(pred_lanes) == 0 and len(gt_lanes) == 0:
                continue
            if len(pred_lanes) == 0:
                fn += len(gt_lanes)
                continue
            if len(gt_lanes) == 0:
                fp += len(pred_lanes)
                continue

            pairs = []
            for pred_idx, pred_lane in enumerate(pred_lanes):
                for gt_idx, gt_lane in enumerate(gt_lanes):
                    distance = self._lane_distance(pred_lane, gt_lane)
                    pairs.append((distance, pred_idx, gt_idx))
            pairs.sort(key=lambda item: item[0])
            used_pred = set()
            used_gt = set()
            for distance, pred_idx, gt_idx in pairs:
                if distance > self.eval_distance_px:
                    continue
                if pred_idx in used_pred or gt_idx in used_gt:
                    continue
                used_pred.add(pred_idx)
                used_gt.add(gt_idx)
                tp += 1
                distances.append(distance)
            fp += len(pred_lanes) - len(used_pred)
            fn += len(gt_lanes) - len(used_gt)

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-8)
        mean_distance = float(np.mean(distances)) if distances else 0.0
        result = {
            'F1': f1,
            'precision': precision,
            'recall': recall,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'mean_matched_distance_px': mean_distance,
            'distance_threshold_px': self.eval_distance_px,
        }
        self.logger.info('BDD curve metric: ' + str(result))
        return f1
