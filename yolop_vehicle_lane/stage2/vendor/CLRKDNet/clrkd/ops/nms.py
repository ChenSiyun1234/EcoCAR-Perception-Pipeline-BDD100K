import torch


def nms(boxes, scores, overlap, top_k):
    if boxes.numel() == 0:
        keep = torch.zeros((0,), dtype=torch.long, device=scores.device)
        return keep, 0, None
    order = torch.argsort(scores, descending=True)
    keep = order[:top_k].contiguous()
    return keep, keep.numel(), None
