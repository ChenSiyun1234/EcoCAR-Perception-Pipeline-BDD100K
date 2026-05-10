from __future__ import annotations

import runpy
import sys

if __name__ == '__main__':
    if '--config' not in sys.argv:
        sys.argv.extend(['--config', '/content/drive/MyDrive/EcoCAR/yolop_vehicle_lane/stage2/configs/exp01_rmt_backbone_neck_joint.yaml'])
    runpy.run_module('stage2.scripts.train_joint_model_experiment', run_name='__main__')
