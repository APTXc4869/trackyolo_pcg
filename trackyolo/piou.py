# PIoU / PIoUv2 loss helper (PyTorch)
# Based on the public reference implementation in fppccc/Powerful-IoU.

from __future__ import annotations
import torch


def bbox_xywh_to_xyxy(xywh: torch.Tensor) -> torch.Tensor:
    x, y, w, h = xywh.unbind(-1)
    w2, h2 = w / 2.0, h / 2.0
    return torch.stack((x - w2, y - h2, x + w2, y + h2), dim=-1)


def piou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    xywh: bool = True,
    PIoU: bool = False,
    PIoU2: bool = False,
    Lambda: float = 1.3,
    eps: float = 1e-7,
) -> torch.Tensor:
    if xywh:
        b1 = bbox_xywh_to_xyxy(box1)
        b2 = bbox_xywh_to_xyxy(box2)
    else:
        b1 = box1
        b2 = box2

    b1_x1, b1_y1, b1_x2, b1_y2 = b1.unbind(-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = b2.unbind(-1)

    w1 = (b1_x2 - b1_x1).clamp(min=eps)
    h1 = (b1_y2 - b1_y1).clamp(min=eps)
    w2 = (b2_x2 - b2_x1).clamp(min=eps)
    h2 = (b2_y2 - b2_y1).clamp(min=eps)

    inter_w = (torch.minimum(b1_x2, b2_x2) - torch.maximum(b1_x1, b2_x1)).clamp(min=0.0)
    inter_h = (torch.minimum(b1_y2, b2_y2) - torch.maximum(b1_y1, b2_y1)).clamp(min=0.0)
    inter = inter_w * inter_h

    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    dw1 = torch.abs(torch.minimum(b1_x2, b1_x1) - torch.minimum(b2_x2, b2_x1))
    dw2 = torch.abs(torch.maximum(b1_x2, b1_x1) - torch.maximum(b2_x2, b2_x1))
    dh1 = torch.abs(torch.minimum(b1_y2, b1_y1) - torch.minimum(b2_y2, b2_y1))
    dh2 = torch.abs(torch.maximum(b1_y2, b1_y1) - torch.maximum(b2_y2, b2_y1))
    P = ((dw1 + dw2) / w2 + (dh1 + dh2) / h2) / 4.0

    L_v1 = 1.0 - iou - torch.exp(-(P ** 2)) + 1.0

    if PIoU:
        return L_v1
    if PIoU2:
        q = torch.exp(-P)
        x = q * Lambda
        u = 3.0 * x * torch.exp(-(x ** 2))
        return u * L_v1
    return iou
