import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (ANCHORS, STRIDES, GRID_SIZES, NUM_CLASSES, NUM_ANCHORS,
                    LAMBDA_BOX, LAMBDA_OBJ, LAMBDA_NOOBJ, LAMBDA_CLS)
from utils.boxes import box_iou, box_ciou, xywh_to_xyxy


def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    """Focal loss for handling class imbalance."""
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
    p = torch.sigmoid(pred)
    pt = target * p + (1 - target) * (1 - p)
    focal_weight = (1 - pt) ** gamma
    return alpha * focal_weight * bce


class YOLOLoss(nn.Module):
    def __init__(self, focal=False, label_smoothing=0.0):
        super().__init__()
        self.focal = focal
        self.label_smoothing = label_smoothing
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, predictions, targets):
        device = predictions[0].device
        loss_box = torch.tensor(0.0, device=device)
        loss_obj = torch.tensor(0.0, device=device)
        loss_cls = torch.tensor(0.0, device=device)

        for scale_idx, pred in enumerate(predictions):
            anchors = torch.tensor(ANCHORS[scale_idx], device=device, dtype=torch.float32)
            stride = STRIDES[scale_idx]
            grid_size = GRID_SIZES[scale_idx]

            # build targets for this scale
            obj_mask, noobj_mask, tx, ty, tw, th, tcls, tbox = self.build_targets(
                pred, targets, anchors, stride, grid_size, device
            )

            # box loss (only for positive samples)
            if obj_mask.sum() > 0:
                # get predicted boxes
                pred_box = pred[..., :4]
                px = torch.sigmoid(pred_box[..., 0])
                py = torch.sigmoid(pred_box[..., 1])
                pw = pred_box[..., 2]
                ph = pred_box[..., 3]

                # decode to absolute coords
                B, A, H, W = px.shape
                gy, gx = torch.meshgrid(torch.arange(H, device=device),
                                        torch.arange(W, device=device), indexing='ij')
                gx = gx.view(1, 1, H, W).expand(B, A, H, W).float()
                gy = gy.view(1, 1, H, W).expand(B, A, H, W).float()
                aw = anchors[:, 0].view(1, A, 1, 1).expand(B, A, H, W)
                ah = anchors[:, 1].view(1, A, 1, 1).expand(B, A, H, W)

                pred_cx = (px + gx) * stride
                pred_cy = (py + gy) * stride
                pred_w = aw * torch.exp(pw.clamp(-10, 10))
                pred_h = ah * torch.exp(ph.clamp(-10, 10))

                pred_x1 = pred_cx - pred_w / 2
                pred_y1 = pred_cy - pred_h / 2
                pred_x2 = pred_cx + pred_w / 2
                pred_y2 = pred_cy + pred_h / 2

                pred_xyxy = torch.stack([pred_x1, pred_y1, pred_x2, pred_y2], dim=-1)
                pred_xyxy_pos = pred_xyxy[obj_mask]
                tbox_pos = tbox[obj_mask]

                ciou_loss = box_ciou(pred_xyxy_pos, tbox_pos)
                loss_box += ciou_loss.mean()

            # objectness loss
            pred_obj = pred[..., 4]
            obj_target = obj_mask.float()

            if self.focal:
                loss_pos = focal_loss(pred_obj, obj_target) * obj_mask
                loss_neg = focal_loss(pred_obj, obj_target) * noobj_mask
            else:
                loss_pos = self.bce(pred_obj, obj_target) * obj_mask
                loss_neg = self.bce(pred_obj, obj_target) * noobj_mask

            loss_obj += LAMBDA_OBJ * loss_pos.sum() / (obj_mask.sum() + 1)
            loss_obj += LAMBDA_NOOBJ * loss_neg.sum() / (noobj_mask.sum() + 1)

            # classification loss (only for positive samples)
            if obj_mask.sum() > 0:
                pred_cls = pred[..., 5:]

                # apply label smoothing
                if self.label_smoothing > 0:
                    tcls_smooth = tcls * (1 - self.label_smoothing) + self.label_smoothing / NUM_CLASSES
                else:
                    tcls_smooth = tcls

                if self.focal:
                    cls_loss = focal_loss(pred_cls, tcls_smooth) * obj_mask.unsqueeze(-1)
                else:
                    cls_loss = self.bce(pred_cls, tcls_smooth) * obj_mask.unsqueeze(-1)
                loss_cls += cls_loss.sum() / (obj_mask.sum() + 1)

        total = LAMBDA_BOX * loss_box + loss_obj + LAMBDA_CLS * loss_cls
        return total, loss_box.item(), loss_obj.item(), loss_cls.item()

    def build_targets(self, pred, targets, anchors, stride, grid_size, device):
        B, A, H, W, _ = pred.shape

        obj_mask = torch.zeros(B, A, H, W, dtype=torch.bool, device=device)
        noobj_mask = torch.ones(B, A, H, W, dtype=torch.bool, device=device)
        tx = torch.zeros(B, A, H, W, device=device)
        ty = torch.zeros(B, A, H, W, device=device)
        tw = torch.zeros(B, A, H, W, device=device)
        th = torch.zeros(B, A, H, W, device=device)
        tcls = torch.zeros(B, A, H, W, NUM_CLASSES, device=device)
        tbox = torch.zeros(B, A, H, W, 4, device=device)

        for b in range(B):
            boxes = targets[b]['boxes']  # xyxy format
            labels = targets[b]['labels']

            if len(boxes) == 0:
                continue

            # convert to center format
            cx = (boxes[:, 0] + boxes[:, 2]) / 2
            cy = (boxes[:, 1] + boxes[:, 3]) / 2
            w = boxes[:, 2] - boxes[:, 0]
            h = boxes[:, 3] - boxes[:, 1]

            # grid cell indices
            gi = (cx / stride).long().clamp(0, grid_size - 1)
            gj = (cy / stride).long().clamp(0, grid_size - 1)

            # find best anchor for each gt
            gt_wh = torch.stack([w, h], dim=1)
            anchor_wh = anchors.unsqueeze(0)  # 1, 3, 2
            gt_wh_exp = gt_wh.unsqueeze(1)    # N, 1, 2

            inter_w = torch.min(gt_wh_exp[..., 0], anchor_wh[..., 0])
            inter_h = torch.min(gt_wh_exp[..., 1], anchor_wh[..., 1])
            inter = inter_w * inter_h
            anchor_area = anchor_wh[..., 0] * anchor_wh[..., 1]
            gt_area = gt_wh_exp[..., 0] * gt_wh_exp[..., 1]
            iou = inter / (anchor_area + gt_area - inter + 1e-6)
            best_anchor = iou.argmax(dim=1)

            for t in range(len(boxes)):
                a = best_anchor[t]
                i, j = gi[t], gj[t]

                obj_mask[b, a, j, i] = True
                noobj_mask[b, a, j, i] = False

                # target offsets (not used directly, but kept for reference)
                tx[b, a, j, i] = cx[t] / stride - i.float()
                ty[b, a, j, i] = cy[t] / stride - j.float()
                tw[b, a, j, i] = torch.log(w[t] / anchors[a, 0] + 1e-6)
                th[b, a, j, i] = torch.log(h[t] / anchors[a, 1] + 1e-6)

                # target box in xyxy for CIoU
                tbox[b, a, j, i] = boxes[t]

                # one-hot class
                tcls[b, a, j, i, labels[t]] = 1.0

            # ignore predictions with high IoU but not best match
            pred_boxes = self.get_pred_boxes(pred[b:b+1], anchors, stride)
            pred_boxes = pred_boxes.view(-1, 4)

            for t in range(len(boxes)):
                ious = box_iou(boxes[t:t+1], pred_boxes)[0]
                high_iou = ious > 0.5
                high_iou = high_iou.view(A, H, W)

                # don't penalize high IoU predictions
                noobj_mask[b][high_iou] = False

        return obj_mask, noobj_mask, tx, ty, tw, th, tcls, tbox

    def get_pred_boxes(self, pred, anchors, stride):
        B, A, H, W, _ = pred.shape
        device = pred.device

        px = torch.sigmoid(pred[..., 0])
        py = torch.sigmoid(pred[..., 1])
        pw = pred[..., 2]
        ph = pred[..., 3]

        gy, gx = torch.meshgrid(torch.arange(H, device=device),
                                torch.arange(W, device=device), indexing='ij')
        gx = gx.view(1, 1, H, W).expand(B, A, H, W).float()
        gy = gy.view(1, 1, H, W).expand(B, A, H, W).float()
        aw = anchors[:, 0].view(1, A, 1, 1).expand(B, A, H, W)
        ah = anchors[:, 1].view(1, A, 1, 1).expand(B, A, H, W)

        cx = (px + gx) * stride
        cy = (py + gy) * stride
        w = aw * torch.exp(pw.clamp(-10, 10))
        h = ah * torch.exp(ph.clamp(-10, 10))

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        return torch.stack([x1, y1, x2, y2], dim=-1)


if __name__ == '__main__':
    from models.detector import YOLODetector

    model = YOLODetector()
    loss_fn = YOLOLoss()

    # fake batch
    x = torch.randn(2, 3, 416, 416)
    targets = [
        {'boxes': torch.tensor([[100, 100, 200, 200], [50, 50, 150, 250]], dtype=torch.float32),
         'labels': torch.tensor([0, 5])},
        {'boxes': torch.tensor([[200, 200, 350, 350]], dtype=torch.float32),
         'labels': torch.tensor([10])}
    ]

    outputs = model(x)
    loss, l_box, l_obj, l_cls = loss_fn(outputs, targets)

    print(f"Total loss: {loss.item():.4f}")
    print(f"  Box loss: {l_box:.4f}")
    print(f"  Obj loss: {l_obj:.4f}")
    print(f"  Cls loss: {l_cls:.4f}")
