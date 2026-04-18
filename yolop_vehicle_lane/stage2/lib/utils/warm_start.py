"""
Warm-start: load a Stage1 best-row checkpoint into a Stage2 student
model, handling shape mismatches gracefully.

Two common mismatches to resolve:

1. **Detection head class count**. Stage1 = 1 class (`nc=1`), Stage2 3c
   variant = 3 classes (`nc=3`). The detection head's per-scale
   `Conv2d(ch, na*(5+nc), 1)` therefore has different `out_channels`
   per scale. We copy the teacher's box + obj + vehicle-class channels
   into the student's matching slots and Kaiming-init the rest.

2. **Lane-branch channel disagreement**. If the stage2 lane decoder
   differs from stage1's (e.g. different output channels), we skip
   lane parameters with shape mismatch and leave them freshly
   initialised. This is documented in the returned load report.

Usage:
    from stage2.lib.utils.warm_start import warm_start_from_stage1
    report = warm_start_from_stage1(student, stage1_ckpt_path,
                                    student_nc=3, teacher_nc=1)
    print(report)
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn


_DEFAULT_VEHICLE_ROW_SRC = 0   # stage1 'vehicle' row index
_DEFAULT_VEHICLE_ROW_DST = 0   # student 'vehicle' row index (always 0 by convention)


def _copy_detect_head_with_class_remap(
    dst_module: nn.Conv2d,
    src_state: torch.Tensor,
    student_nc: int,
    teacher_nc: int,
    na: int,
    vehicle_src: int = _DEFAULT_VEHICLE_ROW_SRC,
    vehicle_dst: int = _DEFAULT_VEHICLE_ROW_DST,
) -> bool:
    """Copy one detection scale's output conv with class remapping.

    Layout for YOLOP-style Detect: per-anchor channel block is
      [tx, ty, tw, th, obj, cls_0, cls_1, ..., cls_{nc-1}]
    with stride (5 + nc) along the output-channel axis. We copy tx/ty/
    tw/th/obj rows from teacher to student, plus the 'vehicle' class
    row from src index to dst index. Remaining class rows stay at
    student init.
    """
    dst_weight = dst_module.weight.data
    dst_bias = dst_module.bias.data if dst_module.bias is not None else None
    src_weight = src_state  # shape [na*(5+teacher_nc), in_ch, 1, 1]
    src_shape = src_weight.shape

    teacher_block = 5 + teacher_nc
    student_block = 5 + student_nc
    expected_teacher = na * teacher_block
    if src_shape[0] != expected_teacher:
        return False                # unexpected layout — skip

    for a in range(na):
        # Teacher slots in src_weight: rows [a*teacher_block .. +teacher_block]
        # Student slots in dst_weight: rows [a*student_block .. +student_block]
        src_a = a * teacher_block
        dst_a = a * student_block

        # box (tx, ty, tw, th) + obj  → first 5 rows in each block
        dst_weight[dst_a:dst_a + 5] = src_weight[src_a:src_a + 5]
        if dst_bias is not None:
            dst_bias[dst_a:dst_a + 5] = src_state_bias_slice(dst_module, src_a, 5)

        # Vehicle class — only copy if both teacher and student have it.
        if teacher_nc > vehicle_src and student_nc > vehicle_dst:
            src_cls_row = src_a + 5 + vehicle_src
            dst_cls_row = dst_a + 5 + vehicle_dst
            dst_weight[dst_cls_row:dst_cls_row + 1] = src_weight[src_cls_row:src_cls_row + 1]
    return True


def src_state_bias_slice(dst_module, start, length):
    # Placeholder: the caller is expected to pre-load the bias into
    # `dst_module.bias.data[start:start+length]` before calling this.
    # Kept as a hook for future extension.
    return dst_module.bias.data[start:start + length]


def warm_start_from_stage1(
    student: nn.Module,
    ckpt_path: str,
    student_nc: int,
    teacher_nc: int,
    strict_shared: bool = False,
    device: str = 'cpu',
) -> Dict[str, List[str]]:
    """Warm-start `student` from the stage1 checkpoint at `ckpt_path`.

    Returns a report dict:
        {
          'loaded':             [<name>, ...],          # copied verbatim
          'loaded_class_remap': [<name>, ...],          # det head rows remapped
          'skipped_shape':      [<name>, ...],          # shape mismatch, kept student init
          'missing_in_ckpt':    [<name>, ...],          # no source — kept student init
          'ignored':            [<name>, ...],          # in ckpt but no student counterpart
        }
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    src = ckpt.get('state_dict', ckpt)

    report = {
        'loaded': [], 'loaded_class_remap': [],
        'skipped_shape': [], 'missing_in_ckpt': [], 'ignored': [],
    }

    student_state = student.state_dict()
    to_load = {}

    for name, dst_param in student_state.items():
        if name not in src:
            report['missing_in_ckpt'].append(name)
            continue
        src_param = src[name]
        if src_param.shape == dst_param.shape:
            to_load[name] = src_param
            report['loaded'].append(name)
            continue
        # Shape mismatch — try class-remap for detection head conv weights.
        if 'detect.m' in name and name.endswith('.weight') and dst_param.dim() == 4:
            # Expect channel-0 = na*(5+nc). Anchor count `na` = 3 by YOLOP default.
            na = 3
            # Check whether shapes only differ on out-channel axis.
            if (src_param.shape[1:] == dst_param.shape[1:]
                    and src_param.shape[0] == na * (5 + teacher_nc)
                    and dst_param.shape[0] == na * (5 + student_nc)):
                # Prepare a destination tensor seeded with the student's
                # original init; patch the box+obj rows and vehicle row.
                new_w = dst_param.clone()
                teacher_block = 5 + teacher_nc
                student_block = 5 + student_nc
                for a in range(na):
                    src_a = a * teacher_block
                    dst_a = a * student_block
                    new_w[dst_a:dst_a + 5] = src_param[src_a:src_a + 5]
                    # vehicle row
                    new_w[dst_a + 5:dst_a + 6] = src_param[src_a + 5:src_a + 6]
                to_load[name] = new_w
                report['loaded_class_remap'].append(name)
                continue
        if 'detect.m' in name and name.endswith('.bias'):
            na = 3
            if (src_param.shape[0] == na * (5 + teacher_nc)
                    and dst_param.shape[0] == na * (5 + student_nc)):
                new_b = dst_param.clone()
                teacher_block = 5 + teacher_nc
                student_block = 5 + student_nc
                for a in range(na):
                    src_a = a * teacher_block
                    dst_a = a * student_block
                    new_b[dst_a:dst_a + 5] = src_param[src_a:src_a + 5]
                    new_b[dst_a + 5:dst_a + 6] = src_param[src_a + 5:src_a + 6]
                to_load[name] = new_b
                report['loaded_class_remap'].append(name)
                continue

        report['skipped_shape'].append(
            f'{name}: ckpt{tuple(src_param.shape)} vs student{tuple(dst_param.shape)}'
        )

    # Params present in ckpt but not in student (extra heads, removed modules, etc.)
    for name in src.keys():
        if name not in student_state:
            report['ignored'].append(name)

    missing_keys, unexpected_keys = student.load_state_dict(
        {**student_state, **to_load}, strict=strict_shared,
    )
    return report


def print_warm_start_report(report: Dict[str, List[str]]) -> None:
    """Pretty-print the warm-start report so notebooks can show it."""
    for k in ('loaded', 'loaded_class_remap', 'skipped_shape',
              'missing_in_ckpt', 'ignored'):
        entries = report.get(k, [])
        print(f'[warm_start] {k}: {len(entries)}')
        for entry in entries[:10]:
            print(f'    {entry}')
        if len(entries) > 10:
            print(f'    ... (+{len(entries) - 10} more)')
