import torch


def box_iou(boxes1, boxes2):
    # boxes in xyxy format
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-6)


def box_ciou(boxes1, boxes2):
    # boxes in xyxy format, returns CIoU loss (not IoU value)
    eps = 1e-7

    # intersection
    lt = torch.max(boxes1[:, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]

    # union
    w1, h1 = boxes1[:, 2] - boxes1[:, 0], boxes1[:, 3] - boxes1[:, 1]
    w2, h2 = boxes2[:, 2] - boxes2[:, 0], boxes2[:, 3] - boxes2[:, 1]
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - inter
    iou = inter / (union + eps)

    # enclosing box
    clt = torch.min(boxes1[:, :2], boxes2[:, :2])
    crb = torch.max(boxes1[:, 2:], boxes2[:, 2:])
    cw, ch = crb[:, 0] - clt[:, 0], crb[:, 1] - clt[:, 1]
    c2 = cw ** 2 + ch ** 2 + eps

    # center distance
    cx1, cy1 = (boxes1[:, 0] + boxes1[:, 2]) / 2, (boxes1[:, 1] + boxes1[:, 3]) / 2
    cx2, cy2 = (boxes2[:, 0] + boxes2[:, 2]) / 2, (boxes2[:, 1] + boxes2[:, 3]) / 2
    rho2 = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2

    # aspect ratio
    v = (4 / 3.14159 ** 2) * (torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps))) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    return 1 - iou + rho2 / c2 + alpha * v


def xywh_to_xyxy(boxes):
    # center x, center y, width, height -> x1, y1, x2, y2
    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return torch.stack([x - w/2, y - h/2, x + w/2, y + h/2], dim=1)


def xyxy_to_xywh(boxes):
    # x1, y1, x2, y2 -> center x, center y, width, height
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    return torch.stack([(x1 + x2)/2, (y1 + y2)/2, x2 - x1, y2 - y1], dim=1)


def nms(boxes, scores, thresh):
    # boxes in xyxy format
    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order.item())
            break
        i = order[0].item()
        keep.append(i)

        ious = box_iou(boxes[i:i+1], boxes[order[1:]])[0]
        mask = ious <= thresh
        order = order[1:][mask]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def encode_boxes(gt_boxes, anchors, grid_size, stride):
    # gt_boxes: (N, 4) in xyxy absolute coords
    # anchors: list of (w, h) tuples
    # returns: (N, 4) encoded as tx, ty, tw, th
    device = gt_boxes.device
    anchors = torch.tensor(anchors, device=device, dtype=torch.float32)

    gt_xywh = xyxy_to_xywh(gt_boxes)
    cx, cy, w, h = gt_xywh[:, 0], gt_xywh[:, 1], gt_xywh[:, 2], gt_xywh[:, 3]

    # grid cell indices
    gx = (cx / stride).long().clamp(0, grid_size - 1)
    gy = (cy / stride).long().clamp(0, grid_size - 1)

    # find best anchor for each gt box
    anchor_wh = anchors.unsqueeze(0)  # (1, 3, 2)
    gt_wh = torch.stack([w, h], dim=1).unsqueeze(1)  # (N, 1, 2)

    # IoU between gt and anchors (w/h only)
    inter_w = torch.min(gt_wh[:, :, 0], anchor_wh[:, :, 0])
    inter_h = torch.min(gt_wh[:, :, 1], anchor_wh[:, :, 1])
    inter = inter_w * inter_h
    anchor_area = anchor_wh[:, :, 0] * anchor_wh[:, :, 1]
    gt_area = gt_wh[:, :, 0] * gt_wh[:, :, 1]
    iou = inter / (anchor_area + gt_area - inter + 1e-6)
    best_anchor = iou.argmax(dim=1)

    # encode offsets
    tx = cx / stride - gx.float()
    ty = cy / stride - gy.float()

    anchor_w = anchors[best_anchor, 0]
    anchor_h = anchors[best_anchor, 1]
    tw = torch.log(w / (anchor_w + 1e-6) + 1e-6)
    th = torch.log(h / (anchor_h + 1e-6) + 1e-6)

    encoded = torch.stack([tx, ty, tw, th], dim=1)
    return encoded, gx, gy, best_anchor


def decode_boxes(pred, anchors, grid_size, stride):
    # pred: (B, num_anchors, H, W, 4) - tx, ty, tw, th
    # returns: (B, num_anchors, H, W, 4) in xyxy absolute coords
    device = pred.device
    B, A, H, W, _ = pred.shape

    anchors = torch.tensor(anchors, device=device, dtype=torch.float32)

    # grid offsets
    gy, gx = torch.meshgrid(torch.arange(H, device=device),
                            torch.arange(W, device=device), indexing='ij')
    gx = gx.view(1, 1, H, W).expand(B, A, H, W).float()
    gy = gy.view(1, 1, H, W).expand(B, A, H, W).float()

    # anchor sizes
    aw = anchors[:, 0].view(1, A, 1, 1).expand(B, A, H, W)
    ah = anchors[:, 1].view(1, A, 1, 1).expand(B, A, H, W)

    # decode
    tx, ty, tw, th = pred[..., 0], pred[..., 1], pred[..., 2], pred[..., 3]

    cx = (torch.sigmoid(tx) + gx) * stride
    cy = (torch.sigmoid(ty) + gy) * stride
    w = aw * torch.exp(tw.clamp(-10, 10))
    h = ah * torch.exp(th.clamp(-10, 10))

    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    return torch.stack([x1, y1, x2, y2], dim=-1)


if __name__ == '__main__':
    # test IoU
    b1 = torch.tensor([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=torch.float)
    b2 = torch.tensor([[5, 5, 15, 15], [0, 0, 10, 10]], dtype=torch.float)
    print("IoU:", box_iou(b1, b2))

    # test NMS
    boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=torch.float)
    scores = torch.tensor([0.9, 0.8, 0.7])
    keep = nms(boxes, scores, 0.5)
    print("NMS keep:", keep)
