"""
Shared Stage2 training helper.

Every Stage2 training notebook loads one YAML and calls
`run_stage2_training(yaml_path)`. The config drives which protection
mechanisms are active (warm-start, distill, gradient balancing), so a
single helper handles all branch variants.

This keeps the 02 / 03 / 04 / 05 notebooks lean — one-cell wrappers
around this helper, differing only in which YAML they point at.
"""

from __future__ import annotations

import math
import os
from typing import Optional

import torch
import torch.nn as nn
from torch import amp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T

from lib.config import cfg
from lib.models import get_net
# DETR-lane stage2 model lives under stage2/lib/models/
from stage2.lib.models.yolopv2_detrlane import get_net_yolopv2_detrlane
from stage2.lib.dataset.bdd_points import BddDatasetPoints
from stage2.lib.core.loss_detrlane import get_loss_detrlane
from stage2.lib.core.distill import DetectionDistillLoss, freeze_as_teacher
from stage2.lib.core.grad_balance import PCGrad, UncertaintyWeighting
from stage2.lib.utils.warm_start import warm_start_from_stage1, print_warm_start_report
from lib.utils.drive_dataset import (
    ensure_local_dataset_from_drive,
    find_raw_bdd_root,
    find_lane_polygon_jsons,
    resolve_bdd_images_100k_dir,
    resolve_bdd_labels_100k_dir,
)


def _build_config(yaml_path: str, repo_root: str, ecocar_root: str):
    cfg.defrost()
    cfg.merge_from_file(yaml_path)

    DATASET_ROOT = ensure_local_dataset_from_drive('bdd100k_vehicle5', ecocar_root)
    RAW_BDD_ROOT = find_raw_bdd_root(ecocar_root)
    lane_jsons = find_lane_polygon_jsons(RAW_BDD_ROOT)
    BDD_IMAGES = resolve_bdd_images_100k_dir(RAW_BDD_ROOT)
    BDD_LABELS = resolve_bdd_labels_100k_dir(RAW_BDD_ROOT)

    cfg.DATASET.ROOT = DATASET_ROOT
    cfg.DATASET.DATAROOT = BDD_IMAGES
    cfg.DATASET.LABELROOT = BDD_LABELS
    cfg.DATASET.LANEROOT = os.path.join(DATASET_ROOT, 'masks')
    cfg.DATASET.LANE_JSON_TRAIN = lane_jsons['train'] or ''
    cfg.DATASET.LANE_JSON_VAL   = lane_jsons['val']   or ''
    cfg.freeze()
    return cfg


def _build_teacher(cfg, ckpt_path: str, device):
    """Load the stage1 teacher model for distillation. Teacher has
    stage1's class count; student has stage2's.
    """
    import copy
    # The teacher is the stage1 YOLOPv2 (2-ch lane, nc=1 vehicle). We
    # use the main factory but override NC to whatever the teacher was
    # trained with.
    teacher_cfg = copy.deepcopy(cfg)
    teacher_cfg.defrost()
    teacher_cfg.MODEL.NC = int(cfg.STAGE2.DISTILL_TEACHER_NC)
    teacher_cfg.MODEL.NAME = 'YOLOPv2'
    teacher_cfg.freeze()
    teacher = get_net(teacher_cfg).to(device)
    state = torch.load(ckpt_path, map_location=device)['state_dict']
    teacher.load_state_dict(state, strict=False)
    return freeze_as_teacher(teacher)


def run_stage2_training(
    yaml_path: str,
    repo_root: str = '/content/drive/MyDrive/EcoCAR/yolop_vehicle_lane',
    ecocar_root: str = '/content/drive/MyDrive/EcoCAR',
    max_epochs: Optional[int] = None,
):
    """End-to-end stage2 training driven entirely by the YAML."""
    cfg_ = _build_config(yaml_path, repo_root, ecocar_root)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Student model ────────────────────────────────────────────────
    student = get_net_yolopv2_detrlane(cfg_).to(device)
    student.gr = 1.0
    student.set_det_only_epochs(int(cfg_.LANE.DET_ONLY_EPOCHS))
    student.names = cfg_.MODEL.VEHICLE_CLASSES

    # ── Warm-start ────────────────────────────────────────────────────
    if bool(getattr(cfg_.STAGE2, 'WARM_START', False)):
        ckpt = str(getattr(cfg_.STAGE2, 'WARM_START_CKPT', '') or '')
        if not ckpt or not os.path.exists(ckpt):
            raise FileNotFoundError(f'WARM_START_CKPT not found: {ckpt}')
        report = warm_start_from_stage1(
            student, ckpt,
            student_nc=int(cfg_.MODEL.NC),
            teacher_nc=int(getattr(cfg_.STAGE2, 'WARM_START_TEACHER_NC', 1)),
            device='cpu',
        )
        print_warm_start_report(report)

    # ── Teacher (distillation) ───────────────────────────────────────
    teacher = None
    distill_loss = None
    if bool(getattr(cfg_.STAGE2, 'DISTILL', False)):
        teacher = _build_teacher(cfg_,
                                 str(cfg_.STAGE2.DISTILL_CKPT),
                                 device)
        distill_loss = DetectionDistillLoss(
            teacher_nc=int(cfg_.STAGE2.DISTILL_TEACHER_NC),
            student_nc=int(cfg_.MODEL.NC),
            box_weight=float(cfg_.STAGE2.DISTILL_BOX_WEIGHT),
            obj_weight=float(cfg_.STAGE2.DISTILL_OBJ_WEIGHT),
            vehicle_cls_weight=float(cfg_.STAGE2.DISTILL_VEHICLE_CLS_WEIGHT),
        ).to(device)

    # ── Datasets + loaders ───────────────────────────────────────────
    transform = T.ToTensor()
    train_ds = BddDatasetPoints(cfg_, is_train=True,  inputsize=640, transform=transform)
    val_ds   = BddDatasetPoints(cfg_, is_train=False, inputsize=(384, 640), transform=transform)
    train_loader = DataLoader(
        train_ds, batch_size=cfg_.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=True, num_workers=cfg_.WORKERS,
        pin_memory=cfg_.PIN_MEMORY, collate_fn=train_ds.collate_fn)
    val_loader = DataLoader(
        val_ds, batch_size=cfg_.TEST.BATCH_SIZE_PER_GPU,
        shuffle=False, num_workers=cfg_.WORKERS,
        pin_memory=cfg_.PIN_MEMORY, collate_fn=val_ds.collate_fn)

    # ── Loss + optimizer + scheduler ─────────────────────────────────
    criterion = get_loss_detrlane(cfg_, device)

    optimizer = torch.optim.SGD(
        student.parameters(), lr=cfg_.TRAIN.LR0,
        momentum=cfg_.TRAIN.MOMENTUM, weight_decay=cfg_.TRAIN.WD,
        nesterov=cfg_.TRAIN.NESTEROV,
    ) if cfg_.TRAIN.OPTIMIZER == 'sgd' else torch.optim.Adam(
        student.parameters(), lr=cfg_.TRAIN.LR0,
        betas=(cfg_.TRAIN.MOMENTUM, 0.999),
    )
    for pg in optimizer.param_groups:
        pg['initial_lr'] = cfg_.TRAIN.LR0

    if bool(getattr(cfg_.TRAIN, 'SGDR', False)):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=int(cfg_.TRAIN.SGDR_T0),
            T_mult=int(cfg_.TRAIN.SGDR_TMULT),
            eta_min=cfg_.TRAIN.LR0 * cfg_.TRAIN.LRF,
        )
    else:
        lf = lambda x: ((1 + math.cos(x * math.pi / cfg_.TRAIN.END_EPOCH)) / 2) * \
                       (1 - cfg_.TRAIN.LRF) + cfg_.TRAIN.LRF
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # ── Gradient balancing ───────────────────────────────────────────
    grad_mode = str(getattr(cfg_.STAGE2, 'GRAD_BALANCE', 'none')).lower()
    pcgrad = PCGrad(optimizer) if grad_mode == 'pcgrad' else None
    n_tasks = 2 + (1 if distill_loss is not None else 0)
    uw = UncertaintyWeighting(n_tasks).to(device) if grad_mode == 'uncertainty' else None
    if uw is not None:
        # add learned log-vars to optimizer so they train with the model
        optimizer.add_param_group({'params': uw.parameters(), 'initial_lr': cfg_.TRAIN.LR0})

    scaler = amp.GradScaler(device.type, enabled=device.type != 'cpu')

    # ── Run ──────────────────────────────────────────────────────────
    os.makedirs(cfg_.DRIVE.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(cfg_.DRIVE.METRICS_DIR, exist_ok=True)
    writer = SummaryWriter(os.path.join(cfg_.DRIVE.METRICS_DIR, 'tb'))

    end_epoch = max_epochs if max_epochs is not None else cfg_.TRAIN.END_EPOCH
    for epoch in range(cfg_.TRAIN.BEGIN_EPOCH, end_epoch):
        student.train()
        student.set_epoch(epoch)
        criterion.set_epoch(epoch)
        for i, (imgs, targets, paths, shapes) in enumerate(train_loader):
            imgs = imgs.to(device, non_blocking=True)
            targets = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                       for k, v in targets.items()}

            with amp.autocast(device_type=device.type, enabled=device.type != 'cpu'):
                out = student(imgs)
                task_loss, info = criterion(out, targets, student)

                losses = [task_loss]
                if distill_loss is not None:
                    with torch.no_grad():
                        t_out = teacher(imgs)
                        t_det = t_out[0][1] if isinstance(t_out[0], tuple) else t_out[0]
                    s_det = out['det_out'][1] if isinstance(out['det_out'], tuple) else out['det_out']
                    kd = distill_loss(s_det, t_det) * float(cfg_.STAGE2.DISTILL_TOTAL_WEIGHT)
                    losses.append(kd)
                    info['distill'] = float(kd.item())

                if uw is not None:
                    total = uw(losses)
                else:
                    total = sum(losses)

            if pcgrad is not None and len(losses) > 1:
                pcgrad.zero_grad()
                pcgrad.pc_backward(losses)
                pcgrad.step()
            else:
                optimizer.zero_grad()
                scaler.scale(total).backward()
                scaler.step(optimizer)
                scaler.update()

            if i % cfg_.PRINT_FREQ == 0:
                print(f'ep{epoch} [{i}/{len(train_loader)}] loss={float(total):.4f} '
                      f'det={info.get("det_total", 0):.3f} lane={info.get("lane_total", 0):.3f} '
                      f'{"kd=" + f"{info.get(chr(100)+chr(105)+chr(115)+chr(116)+chr(105)+chr(108)+chr(108), 0):.3f}" if distill_loss is not None else ""}')
                writer.add_scalar('train/loss', float(total), epoch * len(train_loader) + i)
        scheduler.step()

        # Save each epoch.
        torch.save({
            'epoch': epoch,
            'state_dict': student.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(cfg_.DRIVE.CHECKPOINT_DIR, 'latest.pth'))
        print(f'[{cfg_.DRIVE.CHECKPOINT_DIR}] latest.pth saved @ epoch {epoch}')

    writer.close()
    return cfg_
