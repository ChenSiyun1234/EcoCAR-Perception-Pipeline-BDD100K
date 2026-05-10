from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--labels-zip', default='/content/drive/MyDrive/EcoCAR/downloads/bdd100k_labels.zip')
    parser.add_argument('--images-zip', default='/content/drive/MyDrive/EcoCAR/downloads/bdd100k_images_100k.zip')
    parser.add_argument('--output-tar', default='/content/drive/MyDrive/EcoCAR/datasets/bdd100k_clrkd_curve.tar')
    parser.add_argument('--local-root', default='/content/bdd100k_clrkd_curve')
    parser.add_argument('--raw-root', default='/content/bdd100k_raw')
    parser.add_argument('--downloads-root', default='/content/drive/MyDrive/EcoCAR/downloads')
    args = parser.parse_args()
    downloads = Path(args.downloads_root)
    downloads.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).with_name('04_prepare_bdd_curve_labels.py')
    cmd = [
        sys.executable, '-u', str(script),
        '--raw-root', args.raw_root,
        '--downloads-root', str(downloads),
        '--auto-extract',
        '--output-root', args.local_root,
        '--pack-to', args.output_tar,
    ]
    print('labels_zip_hint=', args.labels_zip, flush=True)
    print('images_zip_hint=', args.images_zip, flush=True)
    print(' '.join(cmd), flush=True)
    subprocess.check_call(cmd)


if __name__ == '__main__':
    main()
