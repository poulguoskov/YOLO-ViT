import os
import argparse
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from config import VOC_CLASSES, CONF_THRESH, NMS_THRESH, IMG_SIZE
from data.voc import VOCDataset, collate_fn
from models.detector import YOLODetector
from utils.metrics import (evaluate_detections, evaluate_detections_range,
                           compute_confusion_matrix, plot_confusion_matrix,
                           print_eval_results)


def evaluate(model, loader, device, conf_thresh=CONF_THRESH, nms_thresh=NMS_THRESH):
    """Returns mAP, per-class APs, and raw predictions/targets."""
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for imgs, targets in tqdm(loader, desc='Evaluating'):
            imgs = imgs.to(device)

            preds = model.predict(imgs, conf_thresh, nms_thresh)

            # move to cpu
            for p in preds:
                p['boxes'] = p['boxes'].cpu()
                p['scores'] = p['scores'].cpu()
                p['labels'] = p['labels'].cpu()

            all_preds.extend(preds)
            all_targets.extend(targets)

    mAP, aps = evaluate_detections(all_preds, all_targets)
    return mAP, aps, all_preds, all_targets


@torch.no_grad()
def benchmark(model, device, batch_size=1, num_warmup=10, num_runs=100):
    model.eval()
    input_tensor = torch.randn(batch_size, 3, IMG_SIZE, IMG_SIZE).to(device)

    # warmup
    for _ in range(num_warmup):
        _ = model(input_tensor)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    # benchmark
    start = time.time()
    for _ in range(num_runs):
        _ = model(input_tensor)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    total_time = time.time() - start
    avg_time = total_time / num_runs * 1000  # ms
    fps = num_runs * batch_size / total_time

    # memory
    if device.type == 'cuda':
        mem = torch.cuda.max_memory_allocated() / 1e9
    else:
        mem = 0

    # params
    params = sum(p.numel() for p in model.parameters()) / 1e6

    return {
        'params_M': params,
        'fps': fps,
        'ms_per_img': avg_time / batch_size,
        'memory_GB': mem
    }


def visualize_sample(model, dataset, device, idx, save_path=None):
    model.eval()
    img, target = dataset[idx]

    with torch.no_grad():
        preds = model.predict(img.unsqueeze(0).to(device))[0]

    # denormalize image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_vis = img * std + mean
    img_vis = img_vis.permute(1, 2, 0).numpy().clip(0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # ground truth
    axes[0].imshow(img_vis)
    axes[0].set_title('Ground Truth')
    for box, label in zip(target['boxes'], target['labels']):
        x1, y1, x2, y2 = box.tolist()
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                  linewidth=2, edgecolor='green', facecolor='none')
        axes[0].add_patch(rect)
        axes[0].text(x1, y1-5, VOC_CLASSES[label], color='green', fontsize=10,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    axes[0].axis('off')

    # predictions
    axes[1].imshow(img_vis)
    axes[1].set_title('Predictions')
    preds['boxes'] = preds['boxes'].cpu()
    preds['scores'] = preds['scores'].cpu()
    preds['labels'] = preds['labels'].cpu()

    for box, score, label in zip(preds['boxes'], preds['scores'], preds['labels']):
        x1, y1, x2, y2 = box.tolist()
        rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                  linewidth=2, edgecolor='red', facecolor='none')
        axes[1].add_patch(rect)
        txt = f"{VOC_CLASSES[label]} {score:.2f}"
        axes[1].text(x1, y1-5, txt, color='red', fontsize=10,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    axes[1].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./data/VOCdevkit')
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--backbone', type=str, default='cnn',
                        choices=['cnn', 'resnet50', 'vit', 'vit_pretrained', 'swin'])
    parser.add_argument('--neck', type=str, default='fpn',
                        choices=['fpn', 'bifpn', 'transformer'])
    parser.add_argument('--attention', action='store_true', help='use CBAM attention')
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--conf', type=float, default=CONF_THRESH)
    parser.add_argument('--nms', type=float, default=NMS_THRESH)
    parser.add_argument('--visualize', type=int, default=0, help='num samples to visualize')
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--benchmark', action='store_true', help='run speed benchmark')
    parser.add_argument('--map_range', action='store_true', help='compute mAP@0.5:0.95')
    parser.add_argument('--confusion', action='store_true', help='compute confusion matrix')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # dataset - use 2007 test set
    test_ds = VOCDataset(args.data, years='2007', split='test', augment=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                             num_workers=4, collate_fn=collate_fn)
    print(f"Test samples: {len(test_ds)}")

    # model
    model = YOLODetector(backbone=args.backbone, neck=args.neck, attention=args.attention).to(device)
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    print(f"Loaded weights from {args.weights}")
    print(f"Model: backbone={args.backbone}, neck={args.neck}, attention={args.attention}")

    # benchmark
    if args.benchmark:
        print("\nRunning speed benchmark...")
        results = benchmark(model, device, batch_size=1)
        print(f"\nBenchmark Results ({args.backbone}):")
        print(f"  Parameters: {results['params_M']:.2f}M")
        print(f"  FPS: {results['fps']:.1f}")
        print(f"  Latency: {results['ms_per_img']:.2f} ms/img")
        if results['memory_GB'] > 0:
            print(f"  GPU Memory: {results['memory_GB']:.2f} GB")
        print()

    # evaluate
    mAP, aps, all_preds, all_targets = evaluate(model, test_loader, device, args.conf, args.nms)

    # mAP@0.5:0.95
    mAP_range = None
    if args.map_range:
        print("\nComputing mAP@0.5:0.95...")
        mAP_range, _, _ = evaluate_detections_range(all_preds, all_targets)

    print_eval_results(mAP, aps, mAP_range)

    # confusion matrix
    if args.confusion:
        os.makedirs(args.save_dir, exist_ok=True)
        cm = compute_confusion_matrix(all_preds, all_targets, iou_thresh=args.conf)
        cm_path = os.path.join(args.save_dir, f'confusion_matrix_{args.backbone}.png')
        plot_confusion_matrix(cm, save_path=cm_path)

    # visualize
    if args.visualize > 0:
        os.makedirs(args.save_dir, exist_ok=True)
        for i in range(min(args.visualize, len(test_ds))):
            save_path = os.path.join(args.save_dir, f'sample_{i}.png')
            visualize_sample(model, test_ds, device, i, save_path)


if __name__ == '__main__':
    main()
