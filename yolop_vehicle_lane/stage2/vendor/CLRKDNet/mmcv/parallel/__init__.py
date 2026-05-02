from __future__ import annotations

import numbers
from typing import Any

import numpy as np
import torch
import torch.nn as nn


class DataContainer:
    def __init__(self, data, stack=False, padding_value=0, cpu_only=False):
        self.data = data
        self.stack = stack
        self.padding_value = padding_value
        self.cpu_only = cpu_only


def _collate_values(values: list[Any]):
    first = values[0]
    if isinstance(first, DataContainer):
        return DataContainer([value.data for value in values], cpu_only=first.cpu_only)
    if torch.is_tensor(first):
        return torch.stack(values, dim=0)
    if isinstance(first, np.ndarray):
        return torch.from_numpy(np.stack(values, axis=0))
    if isinstance(first, numbers.Number):
        return torch.tensor(values)
    if isinstance(first, dict):
        return {key: _collate_values([value[key] for value in values]) for key in first}
    if isinstance(first, (list, tuple)):
        return values
    return values


def collate(batch, samples_per_gpu=1):
    if not batch:
        return batch
    first = batch[0]
    if isinstance(first, dict):
        return {key: _collate_values([item[key] for item in batch]) for key in first}
    return _collate_values(batch)


class MMDataParallel(nn.Module):
    def __init__(self, module, device_ids=None, dim=0, **kwargs):
        super().__init__()
        self.module = module
        self.device_ids = list(device_ids) if device_ids is not None else [0]
        self.dim = dim

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def cuda(self, device=None):
        self.module.cuda(device=device)
        return self

    def train(self, mode=True):
        self.module.train(mode)
        return super().train(mode)

    def eval(self):
        self.module.eval()
        return super().eval()
