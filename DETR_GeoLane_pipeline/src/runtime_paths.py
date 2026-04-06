import os
import shutil
import tarfile
from typing import Optional


def _candidate_dataset_roots(local_base: str, dataset_name: str):
    return [
        local_base,
        os.path.join(local_base, dataset_name),
    ]


def _has_dataset_layout(root: str) -> bool:
    return os.path.isdir(os.path.join(root, 'images', 'train')) and os.path.isdir(os.path.join(root, 'labels', 'train'))


def ensure_local_dataset_from_drive(
    dataset_name: str,
    ecocar_root: str,
    local_base: Optional[str] = None,
    force_reextract: bool = False,
) -> str:
    """Prepare a per-notebook local SSD copy from Drive and return the true dataset root.

    This function is notebook-safe: every notebook/runtime must call it independently.
    It never assumes another notebook's /content state exists.
    """
    if local_base is None:
        local_base = f'/content/{dataset_name}'

    dataset_drive = os.path.join(ecocar_root, 'datasets', dataset_name)
    dataset_tar = os.path.join(ecocar_root, 'datasets', f'{dataset_name}.tar')
    global_paths_cfg = os.path.join(ecocar_root, 'paths_config.yaml')

    if force_reextract and os.path.isdir(local_base):
        shutil.rmtree(local_base, ignore_errors=True)

    os.makedirs(local_base, exist_ok=True)

    for cand in _candidate_dataset_roots(local_base, dataset_name):
        if _has_dataset_layout(cand):
            local_paths_cfg = os.path.join(cand, 'paths_config.yaml')
            if os.path.isfile(local_paths_cfg):
                shutil.copy2(local_paths_cfg, global_paths_cfg)
            return cand

    if os.path.isfile(dataset_tar):
        print(f'Extracting {dataset_tar} into this notebook runtime ...')
        with tarfile.open(dataset_tar, 'r') as tar:
            tar.extractall('/content', filter='data')
        print('Done.')
    elif os.path.isdir(os.path.join(dataset_drive, 'images')):
        print(f'Using Drive dataset directory directly: {dataset_drive}')
        return dataset_drive
    else:
        raise FileNotFoundError(
            f'Dataset not found on Drive. Expected {dataset_tar} or {dataset_drive}'
        )

    for cand in _candidate_dataset_roots(local_base, dataset_name):
        if _has_dataset_layout(cand):
            local_paths_cfg = os.path.join(cand, 'paths_config.yaml')
            if os.path.isfile(local_paths_cfg):
                shutil.copy2(local_paths_cfg, global_paths_cfg)
                print(f'Synced paths_config -> {global_paths_cfg}')
            return cand

    raise FileNotFoundError(
        'Dataset archive extracted, but no valid dataset root was found. '
        f'Checked: {_candidate_dataset_roots(local_base, dataset_name)}'
    )
