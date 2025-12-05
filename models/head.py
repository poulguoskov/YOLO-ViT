import torch
import torch.nn as nn

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import NUM_CLASSES, NUM_ANCHORS


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1):
        super().__init__()
        pad = kernel // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, pad, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DetectionHead(nn.Module):
    """YOLOv3 detection head for one scale."""
    def __init__(self, in_ch, num_anchors=NUM_ANCHORS, num_classes=NUM_CLASSES):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        out_ch = num_anchors * (5 + num_classes)  # 4 box + 1 obj + classes

        self.conv = nn.Sequential(
            ConvBlock(in_ch, in_ch * 2, kernel=3),
        )
        self.pred = nn.Conv2d(in_ch * 2, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        x = self.pred(x)

        B, _, H, W = x.shape
        x = x.view(B, self.num_anchors, 5 + self.num_classes, H, W)
        x = x.permute(0, 1, 3, 4, 2)  # B, A, H, W, 5+C

        return x


class YOLOHead(nn.Module):
    """Multi-scale YOLO detection head."""
    def __init__(self, in_channels=[128, 256, 512], num_anchors=NUM_ANCHORS, num_classes=NUM_CLASSES):
        super().__init__()
        self.heads = nn.ModuleList([
            DetectionHead(ch, num_anchors, num_classes)
            for ch in in_channels
        ])

    def forward(self, features):
        return [head(f) for head, f in zip(self.heads, features)]


if __name__ == '__main__':
    head = YOLOHead(in_channels=[128, 256, 512])

    # fake neck outputs
    p3 = torch.randn(2, 128, 52, 52)
    p4 = torch.randn(2, 256, 26, 26)
    p5 = torch.randn(2, 512, 13, 13)

    outs = head([p3, p4, p5])

    print("YOLO Head outputs:")
    for i, out in enumerate(outs):
        print(f"  Scale {i}: {out.shape}")  # should be B, 3, H, W, 25

    params = sum(p.numel() for p in head.parameters())
    print(f"Parameters: {params / 1e6:.2f}M")
