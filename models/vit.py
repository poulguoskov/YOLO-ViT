import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PatchEmbed(nn.Module):
    def __init__(self, img_size=416, patch_size=16, in_ch=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # B, C, H, W
        x = x.flatten(2).transpose(1, 2)  # B, N, C
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=12, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)
        self.drop_path = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class ViTBackbone(nn.Module):
    """ViT backbone with multi-scale output for object detection."""
    def __init__(self, img_size=416, patch_size=16, embed_dim=768, depth=12, num_heads=12, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.grid_size = img_size // patch_size  # 26 for 416/16

        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout=dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        # project to FPN channel dimensions
        self.proj_s = nn.Conv2d(embed_dim, 256, kernel_size=1)   # scale 0: 52x52
        self.proj_m = nn.Conv2d(embed_dim, 512, kernel_size=1)   # scale 1: 26x26
        self.proj_l = nn.Conv2d(embed_dim, 1024, kernel_size=1)  # scale 2: 13x13

        self.out_channels = [256, 512, 1024]

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        # initialize all linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        B = x.shape[0]

        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # extract features at layers 4, 8, 12
        features = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in [3, 7, 11]:  # layers 4, 8, 12 (0-indexed)
                features.append(x)

        # reshape to spatial (remove cls token)
        outs = []
        for feat in features:
            feat = feat[:, 1:, :]  # remove cls token
            feat = feat.transpose(1, 2).reshape(B, self.embed_dim, self.grid_size, self.grid_size)
            outs.append(feat)

        # project and resize to match FPN expected sizes
        # scale 0: 52x52 (upsample 2x)
        c3 = self.proj_s(F.interpolate(outs[0], scale_factor=2, mode='bilinear', align_corners=False))
        # scale 1: 26x26 (keep same)
        c4 = self.proj_m(outs[1])
        # scale 2: 13x13 (downsample 2x)
        c5 = self.proj_l(F.avg_pool2d(outs[2], kernel_size=2, stride=2))

        return [c3, c4, c5]


class SwinBackbone(nn.Module):
    """Swin Transformer backbone with native multi-scale features."""
    def __init__(self, img_size=416, model_name='swin_base_patch4_window7_224', pretrained=True):
        super().__init__()
        import timm

        # load pretrained swin
        self.swin = timm.create_model(model_name, pretrained=pretrained, img_size=img_size,
                                      features_only=True, out_indices=(1, 2, 3))

        # swin stages output channels: [256, 512, 1024] for swin_base
        self.stage_channels = self.swin.feature_info.channels()

        # project to standard FPN channels
        self.proj_s = nn.Conv2d(self.stage_channels[0], 256, kernel_size=1)
        self.proj_m = nn.Conv2d(self.stage_channels[1], 512, kernel_size=1)
        self.proj_l = nn.Conv2d(self.stage_channels[2], 1024, kernel_size=1)

        self.out_channels = [256, 512, 1024]

    def forward(self, x):
        # swin outputs features at multiple scales natively
        features = self.swin(x)  # list of [stage2, stage3, stage4]

        # swin outputs (B, H, W, C), need to convert to (B, C, H, W)
        feats = []
        for f in features:
            if f.dim() == 4 and f.shape[-1] != f.shape[-2]:
                # likely (B, H, W, C) format, permute to (B, C, H, W)
                f = f.permute(0, 3, 1, 2).contiguous()
            feats.append(f)

        # project to FPN channels
        c3 = self.proj_s(feats[0])  # 52x52
        c4 = self.proj_m(feats[1])  # 26x26
        c5 = self.proj_l(feats[2])  # 13x13

        return [c3, c4, c5]


class ViTBackbonePretrained(nn.Module):
    """Pretrained ViT backbone using timm."""
    def __init__(self, img_size=416, model_name='vit_base_patch16_224'):
        super().__init__()
        import timm

        # load pretrained vit
        self.vit = timm.create_model(model_name, pretrained=True, img_size=img_size)
        self.embed_dim = self.vit.embed_dim
        self.patch_size = self.vit.patch_embed.patch_size[0]
        self.grid_size = img_size // self.patch_size

        # remove classification head
        self.vit.head = nn.Identity()

        # projection layers for FPN
        self.proj_s = nn.Conv2d(self.embed_dim, 256, kernel_size=1)
        self.proj_m = nn.Conv2d(self.embed_dim, 512, kernel_size=1)
        self.proj_l = nn.Conv2d(self.embed_dim, 1024, kernel_size=1)

        self.out_channels = [256, 512, 1024]

    def forward(self, x):
        B = x.shape[0]

        # patch embed
        x = self.vit.patch_embed(x)
        cls_token = self.vit.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        x = x + self.vit.pos_embed
        x = self.vit.pos_drop(x)

        # extract features at layers 4, 8, 12
        features = []
        for i, block in enumerate(self.vit.blocks):
            x = block(x)
            if i in [3, 7, 11]:
                features.append(x)

        # reshape to spatial
        outs = []
        for feat in features:
            feat = feat[:, 1:, :]  # remove cls token
            feat = feat.transpose(1, 2).reshape(B, self.embed_dim, self.grid_size, self.grid_size)
            outs.append(feat)

        # project and resize
        c3 = self.proj_s(F.interpolate(outs[0], scale_factor=2, mode='bilinear', align_corners=False))
        c4 = self.proj_m(outs[1])
        c5 = self.proj_l(F.avg_pool2d(outs[2], kernel_size=2, stride=2))

        return [c3, c4, c5]


if __name__ == '__main__':
    x = torch.randn(2, 3, 416, 416)

    print("ViT Backbone (scratch):")
    model = ViTBackbone()
    outs = model(x)
    for i, out in enumerate(outs):
        print(f"  Scale {i}: {out.shape}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M\n")

    print("ViT Backbone (pretrained):")
    model_pt = ViTBackbonePretrained()
    outs = model_pt(x)
    for i, out in enumerate(outs):
        print(f"  Scale {i}: {out.shape}")
    print(f"  Params: {sum(p.numel() for p in model_pt.parameters()) / 1e6:.2f}M\n")

    print("Swin Backbone (pretrained):")
    model_swin = SwinBackbone()
    outs = model_swin(x)
    for i, out in enumerate(outs):
        print(f"  Scale {i}: {out.shape}")
    print(f"  Params: {sum(p.numel() for p in model_swin.parameters()) / 1e6:.2f}M")
