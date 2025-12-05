import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attention import CBAM


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1):
        super().__init__()
        pad = kernel // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, pad, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class FPNBlock(nn.Module):
    """Single FPN block: upsample + concat + conv."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv1 = ConvBlock(in_ch, in_ch // 2, kernel=1)
        self.conv2 = ConvBlock(in_ch // 2 + skip_ch, out_ch, kernel=3)

    def forward(self, x, skip):
        x = self.conv1(x)
        x = F.interpolate(x, size=skip.shape[2:], mode='nearest')
        x = torch.cat([x, skip], dim=1)
        return self.conv2(x)


class FPNNeck(nn.Module):
    """FPN-style neck for YOLOv3."""
    def __init__(self, in_channels=[256, 512, 1024]):
        super().__init__()
        c3_ch, c4_ch, c5_ch = in_channels

        # process c5 (13x13)
        self.p5_conv = nn.Sequential(
            ConvBlock(c5_ch, 512, kernel=1),
            ConvBlock(512, 1024, kernel=3),
            ConvBlock(1024, 512, kernel=1),
        )

        # upsample and merge with c4 (26x26)
        self.p4_fpn = FPNBlock(512, c4_ch, 256)
        self.p4_conv = nn.Sequential(
            ConvBlock(256, 512, kernel=3),
            ConvBlock(512, 256, kernel=1),
        )

        # upsample and merge with c3 (52x52)
        self.p3_fpn = FPNBlock(256, c3_ch, 128)
        self.p3_conv = nn.Sequential(
            ConvBlock(128, 256, kernel=3),
            ConvBlock(256, 128, kernel=1),
        )

        self.out_channels = [128, 256, 512]

    def forward(self, features):
        c3, c4, c5 = features

        # top-down path
        p5 = self.p5_conv(c5)       # 13x13, 512ch

        p4 = self.p4_fpn(p5, c4)    # 26x26
        p4 = self.p4_conv(p4)       # 26x26, 256ch

        p3 = self.p3_fpn(p4, c3)    # 52x52
        p3 = self.p3_conv(p3)       # 52x52, 128ch

        return [p3, p4, p5]


class WeightedFusion(nn.Module):
    """Learnable weighted feature fusion for BiFPN."""
    def __init__(self, n_inputs, eps=1e-4):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(n_inputs))
        self.eps = eps

    def forward(self, inputs):
        # fast normalized fusion
        w = F.relu(self.weights)
        w = w / (w.sum() + self.eps)
        out = sum(wi * inp for wi, inp in zip(w, inputs))
        return out


class BiFPNBlock(nn.Module):
    """Single BiFPN block with top-down and bottom-up paths."""
    def __init__(self, channels, use_attention=True):
        super().__init__()
        # BiFPN uses uniform channels across all scales
        self.ch = channels

        # top-down fusion (p5->p4, p4->p3)
        self.td_fusion_p4 = WeightedFusion(2)
        self.td_conv_p4 = ConvBlock(channels, channels)

        self.td_fusion_p3 = WeightedFusion(2)
        self.td_conv_p3 = ConvBlock(channels, channels)

        # bottom-up fusion (p3->p4, p4->p5)
        self.bu_fusion_p4 = WeightedFusion(3)  # orig_p4, td_p4, bu_p3
        self.bu_conv_p4 = ConvBlock(channels, channels)

        self.bu_fusion_p5 = WeightedFusion(2)  # orig_p5, bu_p4
        self.bu_conv_p5 = ConvBlock(channels, channels)

        # attention
        self.use_attention = use_attention
        if use_attention:
            self.attn_p3 = CBAM(channels)
            self.attn_p4 = CBAM(channels)
            self.attn_p5 = CBAM(channels)

    def forward(self, features):
        p3, p4, p5 = features

        # top-down path
        p5_up = F.interpolate(p5, size=p4.shape[2:], mode='nearest')
        td_p4 = self.td_conv_p4(self.td_fusion_p4([p4, p5_up]))

        td_p4_up = F.interpolate(td_p4, size=p3.shape[2:], mode='nearest')
        td_p3 = self.td_conv_p3(self.td_fusion_p3([p3, td_p4_up]))

        # bottom-up path
        td_p3_down = F.avg_pool2d(td_p3, kernel_size=2, stride=2)
        bu_p4 = self.bu_conv_p4(self.bu_fusion_p4([p4, td_p4, td_p3_down]))

        bu_p4_down = F.avg_pool2d(bu_p4, kernel_size=2, stride=2)
        bu_p5 = self.bu_conv_p5(self.bu_fusion_p5([p5, bu_p4_down]))

        # apply attention
        if self.use_attention:
            td_p3 = self.attn_p3(td_p3)
            bu_p4 = self.attn_p4(bu_p4)
            bu_p5 = self.attn_p5(bu_p5)

        return [td_p3, bu_p4, bu_p5]


class BiFPN(nn.Module):
    """Bidirectional Feature Pyramid Network."""
    def __init__(self, in_channels=[256, 512, 1024], out_channels=[128, 256, 512],
                 n_blocks=2, use_attention=True, bifpn_channels=256):
        super().__init__()
        c3_ch, c4_ch, c5_ch = in_channels

        # project all scales to uniform channels (key for BiFPN)
        self.proj_p3 = ConvBlock(c3_ch, bifpn_channels, kernel=1)
        self.proj_p4 = ConvBlock(c4_ch, bifpn_channels, kernel=1)
        self.proj_p5 = ConvBlock(c5_ch, bifpn_channels, kernel=1)

        # stack BiFPN blocks (all use same channel dim)
        self.blocks = nn.ModuleList([
            BiFPNBlock(bifpn_channels, use_attention=use_attention)
            for _ in range(n_blocks)
        ])

        # project back to expected output channels for head
        self.out_p3 = ConvBlock(bifpn_channels, out_channels[0], kernel=1)
        self.out_p4 = ConvBlock(bifpn_channels, out_channels[1], kernel=1)
        self.out_p5 = ConvBlock(bifpn_channels, out_channels[2], kernel=1)

        self.out_channels = out_channels

    def forward(self, features):
        c3, c4, c5 = features

        # project to uniform channels
        p3 = self.proj_p3(c3)
        p4 = self.proj_p4(c4)
        p5 = self.proj_p5(c5)

        # run through BiFPN blocks
        feats = [p3, p4, p5]
        for block in self.blocks:
            feats = block(feats)

        # project to output channels
        p3, p4, p5 = feats
        p3 = self.out_p3(p3)
        p4 = self.out_p4(p4)
        p5 = self.out_p5(p5)

        return [p3, p4, p5]


class TransformerBlock(nn.Module):
    """Simple transformer block for neck."""
    def __init__(self, dim, n_heads=4, mlp_ratio=2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)

        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim)
        )

    def forward(self, x):
        # x: (B, N, C)
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerFPN(nn.Module):
    """BiFPN with transformer enhancement at each scale."""
    def __init__(self, in_channels=[256, 512, 1024], out_channels=[128, 256, 512],
                 n_blocks=1, n_heads=4):
        super().__init__()

        # base BiFPN
        self.bifpn = BiFPN(in_channels, out_channels, n_blocks=n_blocks, use_attention=True)

        # transformer blocks per scale
        self.trans_p3 = TransformerBlock(out_channels[0], n_heads)
        self.trans_p4 = TransformerBlock(out_channels[1], n_heads)
        self.trans_p5 = TransformerBlock(out_channels[2], n_heads)

        self.out_channels = out_channels

    def forward(self, features):
        # BiFPN first
        p3, p4, p5 = self.bifpn(features)

        # apply transformer to each scale
        B = p3.shape[0]

        # p3: (B, C, H, W) -> (B, HW, C) -> transformer -> (B, C, H, W)
        h3, w3 = p3.shape[2:]
        p3_flat = p3.flatten(2).permute(0, 2, 1)
        p3_flat = self.trans_p3(p3_flat)
        p3 = p3_flat.permute(0, 2, 1).view(B, -1, h3, w3)

        h4, w4 = p4.shape[2:]
        p4_flat = p4.flatten(2).permute(0, 2, 1)
        p4_flat = self.trans_p4(p4_flat)
        p4 = p4_flat.permute(0, 2, 1).view(B, -1, h4, w4)

        h5, w5 = p5.shape[2:]
        p5_flat = p5.flatten(2).permute(0, 2, 1)
        p5_flat = self.trans_p5(p5_flat)
        p5 = p5_flat.permute(0, 2, 1).view(B, -1, h5, w5)

        return [p3, p4, p5]


if __name__ == '__main__':
    # fake backbone outputs
    c3 = torch.randn(2, 256, 52, 52)
    c4 = torch.randn(2, 512, 26, 26)
    c5 = torch.randn(2, 1024, 13, 13)
    features = [c3, c4, c5]

    # test FPN
    print("FPN Neck:")
    fpn = FPNNeck(in_channels=[256, 512, 1024])
    outs = fpn(features)
    for i, out in enumerate(outs):
        print(f"  Scale {i}: {out.shape}")
    print(f"  Params: {sum(p.numel() for p in fpn.parameters()) / 1e6:.2f}M\n")

    # test BiFPN
    print("BiFPN:")
    bifpn = BiFPN(in_channels=[256, 512, 1024], use_attention=True)
    outs = bifpn(features)
    for i, out in enumerate(outs):
        print(f"  Scale {i}: {out.shape}")
    print(f"  Params: {sum(p.numel() for p in bifpn.parameters()) / 1e6:.2f}M\n")

    # test TransformerFPN
    print("TransformerFPN:")
    tfpn = TransformerFPN(in_channels=[256, 512, 1024])
    outs = tfpn(features)
    for i, out in enumerate(outs):
        print(f"  Scale {i}: {out.shape}")
    print(f"  Params: {sum(p.numel() for p in tfpn.parameters()) / 1e6:.2f}M")
