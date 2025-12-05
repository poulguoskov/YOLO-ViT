import torch
import torch.nn as nn

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ANCHORS, STRIDES, CONF_THRESH, NMS_THRESH, NUM_CLASSES
from utils.boxes import decode_boxes, nms, xywh_to_xyxy

from models.cnn import CNNBackbone, ResNet50Backbone
from models.vit import ViTBackbone, ViTBackbonePretrained, SwinBackbone
from models.neck import FPNNeck, BiFPN, TransformerFPN
from models.head import YOLOHead


class YOLODetector(nn.Module):
    """Full YOLOv3-style detector."""
    def __init__(self, backbone='cnn', neck='fpn', attention=False):
        super().__init__()

        # backbone
        if backbone == 'cnn':
            self.backbone = CNNBackbone()
        elif backbone == 'resnet50':
            self.backbone = ResNet50Backbone()
        elif backbone == 'vit':
            self.backbone = ViTBackbone()
        elif backbone == 'vit_pretrained':
            self.backbone = ViTBackbonePretrained()
        elif backbone == 'swin':
            self.backbone = SwinBackbone()
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        # neck
        in_ch = self.backbone.out_channels
        if neck == 'fpn':
            self.neck = FPNNeck(in_channels=in_ch)
        elif neck == 'bifpn':
            self.neck = BiFPN(in_channels=in_ch, use_attention=attention)
        elif neck == 'transformer':
            self.neck = TransformerFPN(in_channels=in_ch)
        else:
            raise ValueError(f"Unknown neck: {neck}")

        self.head = YOLOHead(in_channels=self.neck.out_channels)

    def forward(self, x):
        features = self.backbone(x)
        features = self.neck(features)
        outputs = self.head(features)
        return outputs

    @torch.no_grad()
    def predict(self, x, conf_thresh=CONF_THRESH, nms_thresh=NMS_THRESH):
        """Run inference and return decoded boxes."""
        outputs = self.forward(x)
        return self.decode_predictions(outputs, conf_thresh, nms_thresh)

    def decode_predictions(self, outputs, conf_thresh=CONF_THRESH, nms_thresh=NMS_THRESH):
        """Decode raw outputs to boxes, scores, labels."""
        batch_size = outputs[0].shape[0]
        results = []

        for b in range(batch_size):
            all_boxes = []
            all_scores = []
            all_labels = []

            for scale_idx, out in enumerate(outputs):
                pred = out[b:b+1]  # 1, A, H, W, 5+C
                anchors = ANCHORS[scale_idx]
                stride = STRIDES[scale_idx]
                grid_size = pred.shape[2]

                # split predictions
                box_pred = pred[..., :4]
                obj_pred = torch.sigmoid(pred[..., 4])
                cls_pred = torch.sigmoid(pred[..., 5:])

                # decode boxes
                boxes = decode_boxes(box_pred, anchors, grid_size, stride)
                boxes = boxes.reshape(-1, 4)

                # compute scores
                obj_pred = obj_pred.reshape(-1)
                cls_pred = cls_pred.reshape(-1, NUM_CLASSES)
                scores, labels = cls_pred.max(dim=1)
                scores = scores * obj_pred

                # filter by confidence
                mask = scores > conf_thresh
                boxes = boxes[mask]
                scores = scores[mask]
                labels = labels[mask]

                all_boxes.append(boxes)
                all_scores.append(scores)
                all_labels.append(labels)

            # concatenate all scales
            all_boxes = torch.cat(all_boxes, dim=0)
            all_scores = torch.cat(all_scores, dim=0)
            all_labels = torch.cat(all_labels, dim=0)

            # per-class NMS
            final_boxes = []
            final_scores = []
            final_labels = []

            for cls in range(NUM_CLASSES):
                cls_mask = all_labels == cls
                if cls_mask.sum() == 0:
                    continue

                cls_boxes = all_boxes[cls_mask]
                cls_scores = all_scores[cls_mask]

                keep = nms(cls_boxes, cls_scores, nms_thresh)
                final_boxes.append(cls_boxes[keep])
                final_scores.append(cls_scores[keep])
                final_labels.append(torch.full((len(keep),), cls, device=all_labels.device))

            if final_boxes:
                final_boxes = torch.cat(final_boxes, dim=0)
                final_scores = torch.cat(final_scores, dim=0)
                final_labels = torch.cat(final_labels, dim=0)
            else:
                dev = outputs[0].device
                final_boxes = torch.empty(0, 4, device=dev)
                final_scores = torch.empty(0, device=dev)
                final_labels = torch.empty(0, dtype=torch.long, device=dev)

            results.append({
                'boxes': final_boxes,
                'scores': final_scores,
                'labels': final_labels
            })

        return results


if __name__ == '__main__':
    model = YOLODetector(backbone='cnn')
    x = torch.randn(2, 3, 416, 416)

    # test forward
    outputs = model(x)
    print("Forward pass outputs:")
    for i, out in enumerate(outputs):
        print(f"  Scale {i}: {out.shape}")

    # test predict
    results = model.predict(x)
    print(f"\nPredictions for batch of {len(results)}:")
    for i, res in enumerate(results):
        print(f"  Image {i}: {len(res['boxes'])} detections")

    # model stats
    params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {params / 1e6:.2f}M")
