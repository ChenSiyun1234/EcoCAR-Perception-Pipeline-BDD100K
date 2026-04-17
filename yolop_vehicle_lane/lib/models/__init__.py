"""Model factory with explicit baseline dispatch.

Baselines:
- `yolop_baseline.py`       honest YOLOP-style baseline (DA removed, nc=5).
- `yolopv2_baseline.py`     YOLOPv2-style reconstruction (ELAN + SPPCSPC +
                            transpose-conv lane-seg decoder; INFERRED).
Stage-2 variant (point lane, baseline untouched):
- `yolopv2_detrlane.py`     YOLOPv2 encoder/neck + IDetect + DETR-style
                            LaneSetHead emitting (exist, points, vis, type).

`get_net(cfg)` dispatches on `cfg.MODEL.NAME`:
  - 'YOLOP'          -> yolop_baseline.get_net
  - 'YOLOPv2'        -> yolopv2_baseline.get_net_yolopv2
  - 'YOLOPv2-DETRLane' -> yolopv2_detrlane.get_net_yolopv2_detrlane
"""

from .yolop_baseline import get_net as _get_net_yolop
from .yolopv2_baseline import get_net_yolopv2 as _get_net_yolopv2
from .yolopv2_detrlane import get_net_yolopv2_detrlane as _get_net_yolopv2_detrlane


def get_net(cfg=None, **kwargs):
    name = 'YOLOPv2'
    if cfg is not None:
        name = getattr(getattr(cfg, 'MODEL', object()), 'NAME', 'YOLOPv2') or 'YOLOPv2'
    name = str(name).strip().upper().replace('_', '-')

    if name in ('YOLOP', 'YOLOP-VEHICLE-LANE', 'YOLOP-BASELINE'):
        return _get_net_yolop(cfg, **kwargs)
    if name in ('YOLOPV2', 'YOLOPV2-VEHICLE-LANE', 'YOLOPV2-BASELINE', 'VEHICLELANE'):
        return _get_net_yolopv2(cfg, **kwargs)
    if name in ('YOLOPV2-DETRLANE', 'YOLOPV2DETRLANE', 'YOLOPV2-DETR-LANE'):
        return _get_net_yolopv2_detrlane(cfg, **kwargs)
    raise ValueError(
        f"Unknown MODEL.NAME={name!r}. Use 'YOLOP', 'YOLOPv2', or 'YOLOPv2-DETRLane'."
    )


get_net_yolop = _get_net_yolop
get_net_yolopv2 = _get_net_yolopv2
get_net_yolopv2_detrlane = _get_net_yolopv2_detrlane
