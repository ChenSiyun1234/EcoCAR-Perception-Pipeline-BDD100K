"""Phase-1 model factory.

Two faithful baselines:
- `yolop_baseline.py`   : upstream YOLOP `MCnet_0` with DA head removed, nc=5.
- `yolopv2_baseline.py` : YOLOP MCnet_0 surgically edited toward the
                          YOLOPv2 paper spec: E-ELAN backbone (with
                          groups), SPP kept, deconv lane decoder, 1-ch
                          sigmoid lane output at H/2×W/2.

Stage-2 experimental variants (DETR lane, etc.) live under
`stage2/lib/models/` and are imported on demand from that sub-tree,
not from here.
"""

from .yolop_baseline import get_net as _get_net_yolop
from .yolopv2_baseline import get_net_yolopv2 as _get_net_yolopv2


def get_net(cfg=None, **kwargs):
    name = 'YOLOPv2'
    if cfg is not None:
        name = getattr(getattr(cfg, 'MODEL', object()), 'NAME', 'YOLOPv2') or 'YOLOPv2'
    name = str(name).strip().upper().replace('_', '-')

    if name in ('YOLOP', 'YOLOP-VEHICLE-LANE', 'YOLOP-BASELINE'):
        return _get_net_yolop(cfg, **kwargs)
    if name in ('YOLOPV2', 'YOLOPV2-VEHICLE-LANE', 'YOLOPV2-BASELINE', 'VEHICLELANE'):
        return _get_net_yolopv2(cfg, **kwargs)
    raise ValueError(f"Unknown MODEL.NAME={name!r}. Use 'YOLOP' or 'YOLOPv2'.")


get_net_yolop = _get_net_yolop
get_net_yolopv2 = _get_net_yolopv2
