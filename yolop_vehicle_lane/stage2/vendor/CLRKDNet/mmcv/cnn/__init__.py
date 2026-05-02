from __future__ import annotations

import torch.nn as nn


class ConvModule(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias='auto', conv_cfg=None, norm_cfg=None,
                 act_cfg=None, inplace=True, **kwargs):
        if bias == 'auto':
            bias = norm_cfg is None
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                            padding=padding, dilation=dilation, groups=groups, bias=bias)]
        if norm_cfg is not None:
            layers.append(nn.BatchNorm2d(out_channels))
        if act_cfg is not None:
            act_type = act_cfg.get('type', 'ReLU') if isinstance(act_cfg, dict) else str(act_cfg)
            if act_type.lower() == 'relu':
                layers.append(nn.ReLU(inplace=inplace))
            elif act_type.lower() in ('silu', 'swish'):
                layers.append(nn.SiLU(inplace=inplace))
            elif act_type.lower() == 'leakyrelu':
                layers.append(nn.LeakyReLU(0.1, inplace=inplace))
        super().__init__(*layers)
