import torch
import numpy as np
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import IMG_SIZE


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model, name='Model'):
    """Print model parameter summary."""
    total = count_parameters(model)
    print(f"\n{name} Summary")
    print("-" * 40)
    print(f"Total parameters: {total / 1e6:.2f}M")

    # breakdown by major components
    if hasattr(model, 'backbone'):
        bb_params = count_parameters(model.backbone)
        print(f"  Backbone: {bb_params / 1e6:.2f}M")
    if hasattr(model, 'neck'):
        neck_params = count_parameters(model.neck)
        print(f"  Neck: {neck_params / 1e6:.2f}M")
    if hasattr(model, 'head'):
        head_params = count_parameters(model.head)
        print(f"  Head: {head_params / 1e6:.2f}M")


def visualize_feature_maps(feature_maps, save_path=None, max_channels=16):
    """Visualize feature maps from different scales."""
    n_scales = len(feature_maps)
    fig, axes = plt.subplots(n_scales, max_channels, figsize=(max_channels * 1.5, n_scales * 1.5))

    for i, fmap in enumerate(feature_maps):
        # fmap shape: (1, C, H, W)
        fmap = fmap[0].detach().cpu()
        n_channels = min(fmap.shape[0], max_channels)

        for j in range(n_channels):
            ax = axes[i, j] if n_scales > 1 else axes[j]
            ax.imshow(fmap[j], cmap='viridis')
            ax.axis('off')
            if j == 0:
                ax.set_ylabel(f'Scale {i+1}', fontsize=8)

    plt.suptitle('Feature Maps (first 16 channels per scale)')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved feature maps to {save_path}")
    plt.close()


def visualize_attention(attn_weights, save_path=None, n_heads=4):
    """Visualize ViT attention maps."""
    # attn_weights: (n_heads, seq_len, seq_len)
    attn = attn_weights.detach().cpu().numpy()
    n_heads_total = attn.shape[0]
    n_show = min(n_heads, n_heads_total)

    fig, axes = plt.subplots(1, n_show, figsize=(n_show * 3, 3))

    for i in range(n_show):
        ax = axes[i] if n_show > 1 else axes
        im = ax.imshow(attn[i], cmap='hot')
        ax.set_title(f'Head {i+1}', fontsize=10)
        ax.axis('off')

    plt.colorbar(im, ax=axes, shrink=0.8)
    plt.suptitle('Attention Maps')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved attention maps to {save_path}")
    plt.close()


def extract_backbone_features(model, img):
    """Extract feature maps from backbone."""
    model.eval()
    with torch.no_grad():
        features = model.backbone(img)
    return features


def extract_vit_attention(model, img):
    """Extract attention weights from ViT backbone."""
    if not hasattr(model.backbone, 'vit'):
        print("Model doesn't have ViT backbone")
        return None

    model.eval()
    vit = model.backbone.vit

    # register hook to capture attention
    attention_maps = []

    def hook_fn(module, input, output):
        # output is tuple (attn_output, attn_weights)
        if isinstance(output, tuple) and len(output) > 1:
            attention_maps.append(output[1])

    # find attention modules
    hooks = []
    for name, module in vit.named_modules():
        if 'attn' in name.lower() and hasattr(module, 'forward'):
            hooks.append(module.register_forward_hook(hook_fn))

    with torch.no_grad():
        _ = model.backbone(img)

    # remove hooks
    for h in hooks:
        h.remove()

    return attention_maps[-1] if attention_maps else None


def compare_backbones_table():
    """Print comparison table for all backbones."""
    from models.detector import YOLODetector

    backbones = ['cnn', 'vit', 'resnet50', 'vit_pretrained']
    print("\nBackbone Comparison")
    print("=" * 60)
    print(f"{'Backbone':<20} {'Params (M)':<12} {'Pretrained':<12}")
    print("-" * 60)

    for bb in backbones:
        try:
            model = YOLODetector(backbone=bb)
            params = count_parameters(model) / 1e6
            pretrained = 'Yes' if 'pretrained' in bb or bb == 'resnet50' else 'No'
            print(f"{bb:<20} {params:<12.2f} {pretrained:<12}")
        except Exception as e:
            print(f"{bb:<20} {'Error':<12} {str(e)[:20]}")

    print("=" * 60)


if __name__ == '__main__':
    compare_backbones_table()
