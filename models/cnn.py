import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1):
        super().__init__()
        pad = kernel // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, pad, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv1 = ConvBlock(ch, ch // 2, kernel=1)
        self.conv2 = ConvBlock(ch // 2, ch, kernel=3)

    def forward(self, x):
        return x + self.conv2(self.conv1(x))


class CNNBackbone(nn.Module):
    """Simple CNN backbone producing 3-scale features for YOLO."""
    def __init__(self):
        super().__init__()
        # input: 416 -> 208 -> 104 -> 52 -> 26 -> 13

        # stem: 416 -> 208
        self.stem = nn.Sequential(
            ConvBlock(3, 32, kernel=3, stride=1),
            ConvBlock(32, 64, kernel=3, stride=2),
        )

        # stage1: 208 -> 104
        self.stage1 = nn.Sequential(
            ConvBlock(64, 128, kernel=3, stride=2),
            ResBlock(128),
            ResBlock(128),
        )

        # stage2: 104 -> 52 (output scale 0: 52x52, 128 channels)
        self.stage2 = nn.Sequential(
            ConvBlock(128, 256, kernel=3, stride=2),
            ResBlock(256),
            ResBlock(256),
            ResBlock(256),
        )

        # stage3: 52 -> 26 (output scale 1: 26x26, 512 channels)
        self.stage3 = nn.Sequential(
            ConvBlock(256, 512, kernel=3, stride=2),
            ResBlock(512),
            ResBlock(512),
            ResBlock(512),
        )

        # stage4: 26 -> 13 (output scale 2: 13x13, 1024 channels)
        self.stage4 = nn.Sequential(
            ConvBlock(512, 1024, kernel=3, stride=2),
            ResBlock(1024),
            ResBlock(1024),
        )

        self.out_channels = [256, 512, 1024]

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)

        c3 = self.stage2(x)   # 52x52, 256ch
        c4 = self.stage3(c3)  # 26x26, 512ch
        c5 = self.stage4(c4)  # 13x13, 1024ch

        return [c3, c4, c5]


class ResNet50Backbone(nn.Module):
    """Pretrained ResNet-50 backbone for object detection."""
    def __init__(self):
        super().__init__()
        import torchvision.models as models

        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # use resnet layers as feature extractor
        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1  # 256ch, stride 4
        self.layer2 = resnet.layer2  # 512ch, stride 8
        self.layer3 = resnet.layer3  # 1024ch, stride 16
        self.layer4 = resnet.layer4  # 2048ch, stride 32

        # project to match expected FPN channels
        self.proj_s = nn.Conv2d(512, 256, kernel_size=1)   # 52x52
        self.proj_m = nn.Conv2d(1024, 512, kernel_size=1)  # 26x26
        self.proj_l = nn.Conv2d(2048, 1024, kernel_size=1) # 13x13

        self.out_channels = [256, 512, 1024]

    def forward(self, x):
        x = self.stem(x)       # /4: 104x104
        x = self.layer1(x)     # /4: 104x104
        c3 = self.layer2(x)    # /8: 52x52, 512ch
        c4 = self.layer3(c3)   # /16: 26x26, 1024ch
        c5 = self.layer4(c4)   # /32: 13x13, 2048ch

        # project channels
        c3 = self.proj_s(c3)   # 256ch
        c4 = self.proj_m(c4)   # 512ch
        c5 = self.proj_l(c5)   # 1024ch

        return [c3, c4, c5]


if __name__ == '__main__':
    model = CNNBackbone()
    x = torch.randn(2, 3, 416, 416)
    outs = model(x)

    print("CNN Backbone outputs:")
    for i, out in enumerate(outs):
        print(f"  Scale {i}: {out.shape}")

    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params / 1e6:.2f}M")

    print("\nResNet50 Backbone:")
    model_r = ResNet50Backbone()
    outs = model_r(x)
    for i, out in enumerate(outs):
        print(f"  Scale {i}: {out.shape}")
    params = sum(p.numel() for p in model_r.parameters())
    print(f"Parameters: {params / 1e6:.2f}M")
