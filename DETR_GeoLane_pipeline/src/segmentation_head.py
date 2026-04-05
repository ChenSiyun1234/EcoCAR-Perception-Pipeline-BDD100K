
import torch
import torch.nn as nn

class LightweightSegmentationHead(nn.Module):
    """A lightweight prototype-mask branch that can be attached to detection queries.
    It is intentionally cheaper than a full DETR-style segmentation decoder.
    """
    def __init__(self, in_channels: int = 256, hidden_dim: int = 128, num_prototypes: int = 32, mask_dim: int = 32):
        super().__init__()
        self.proto_net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, num_prototypes, 1),
        )
        self.coeff_mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, mask_dim),
        )

    def forward(self, feature_map: torch.Tensor, det_queries: torch.Tensor) -> dict:
        protos = self.proto_net(feature_map)
        coeffs = self.coeff_mlp(det_queries)
        masks = torch.einsum('bqc,bchw->bqhw', coeffs, protos)
        return {'seg_prototypes': protos, 'seg_coeffs': coeffs, 'seg_masks': masks}
