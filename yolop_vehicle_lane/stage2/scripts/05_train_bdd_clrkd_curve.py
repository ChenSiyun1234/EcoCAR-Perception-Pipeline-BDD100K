from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description='Run CLRKDNet curve-prior lane training on the BDD100K curve dataset.')
    parser.add_argument('--project-root', default='/content/drive/MyDrive/EcoCAR/yolop_vehicle_lane', help='Project root.')
    parser.add_argument('--config', default=None, help='Config file. Defaults to stage2/configs/BDD100K_CLRKD_Curve.py.')
    parser.add_argument('--gpus', nargs='+', default=['0'], help='GPU ids passed to CLRKDNet main.py.')
    parser.add_argument('--work-dirs', default='/content/drive/MyDrive/EcoCAR/training_runs/stage2_clrkd_curve', help='Output directory for checkpoints and logs.')
    parser.add_argument('--resume-from', default=None, help='Checkpoint path to resume from.')
    parser.add_argument('--load-from', default=None, help='Checkpoint path to load from.')
    parser.add_argument('--validate', action='store_true', help='Run validation only.')
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    vendor_root = project_root / 'stage2' / 'vendor' / 'CLRKDNet'
    config = Path(args.config).resolve() if args.config else project_root / 'stage2' / 'configs' / 'BDD100K_CLRKD_Curve.py'

    if not vendor_root.exists():
        raise FileNotFoundError(f'Missing CLRKDNet vendor directory: {vendor_root}')
    if not config.exists():
        raise FileNotFoundError(f'Missing CLRKDNet config: {config}')

    env = os.environ.copy()
    env['PYTHONPATH'] = str(vendor_root) + os.pathsep + env.get('PYTHONPATH', '')

    command = [sys.executable, str(vendor_root / 'main.py'), str(config), '--gpus'] + list(args.gpus) + ['--work_dirs', args.work_dirs]
    if args.resume_from:
        command += ['--resume_from', args.resume_from]
    if args.load_from:
        command += ['--load_from', args.load_from]
    if args.validate:
        command += ['--validate']

    print('Running command:')
    print(' '.join(command))
    subprocess.run(command, cwd=str(vendor_root), env=env, check=True)


if __name__ == '__main__':
    main()
