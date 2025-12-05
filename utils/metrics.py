import torch
import numpy as np
from collections import defaultdict

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import NUM_CLASSES, VOC_CLASSES
from utils.boxes import box_iou


def compute_ap(recall, precision):
    """Compute AP using 11-point interpolation (VOC style)."""
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        prec_at_rec = precision[recall >= t]
        if len(prec_at_rec) > 0:
            ap += prec_at_rec.max() / 11
    return ap


def compute_ap_all_points(recall, precision):
    """Compute AP using all-point interpolation (COCO style)."""
    # prepend/append sentinel values
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # compute precision envelope
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # find where recall changes
    idx = np.where(mrec[1:] != mrec[:-1])[0]

    # sum areas
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return ap


def evaluate_detections(predictions, targets, iou_thresh=0.5):
    """Compute mAP for object detection."""
    # collect all detections and ground truths per class
    all_dets = defaultdict(list)  # class -> list of (img_idx, score, box)
    all_gts = defaultdict(list)   # class -> list of (img_idx, box)
    gt_count = defaultdict(int)

    for img_idx, (pred, gt) in enumerate(zip(predictions, targets)):
        # detections
        for box, score, label in zip(pred['boxes'], pred['scores'], pred['labels']):
            label = int(label)
            all_dets[label].append({
                'img_idx': img_idx,
                'score': float(score),
                'box': box.cpu().numpy() if torch.is_tensor(box) else box
            })

        # ground truths
        for box, label in zip(gt['boxes'], gt['labels']):
            label = int(label)
            all_gts[label].append({
                'img_idx': img_idx,
                'box': box.cpu().numpy() if torch.is_tensor(box) else box,
                'matched': False
            })
            gt_count[label] += 1

    # compute AP per class
    aps = {}
    for cls in range(NUM_CLASSES):
        dets = all_dets[cls]
        gts = all_gts[cls]
        n_gt = gt_count[cls]

        if n_gt == 0:
            continue

        # sort by score descending
        dets = sorted(dets, key=lambda x: x['score'], reverse=True)

        # reset matched flags
        for g in gts:
            g['matched'] = False

        tp = np.zeros(len(dets))
        fp = np.zeros(len(dets))

        for det_idx, det in enumerate(dets):
            img_idx = det['img_idx']
            det_box = det['box']

            # find best matching gt
            best_iou = 0
            best_gt_idx = -1

            for gt_idx, gt in enumerate(gts):
                if gt['img_idx'] != img_idx or gt['matched']:
                    continue

                gt_box = gt['box']
                det_t = torch.from_numpy(np.array([det_box], dtype=np.float32))
                gt_t = torch.from_numpy(np.array([gt_box], dtype=np.float32))
                iou = box_iou(det_t, gt_t)[0, 0].item()

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_thresh and best_gt_idx >= 0:
                tp[det_idx] = 1
                gts[best_gt_idx]['matched'] = True
            else:
                fp[det_idx] = 1

        # cumulative sums
        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)

        recall = tp_cum / n_gt
        precision = tp_cum / (tp_cum + fp_cum + 1e-6)

        ap = compute_ap(recall, precision)
        aps[cls] = ap

    # compute mAP
    if len(aps) > 0:
        mAP = np.mean(list(aps.values()))
    else:
        mAP = 0.0

    return mAP, aps


def evaluate_detections_range(predictions, targets, iou_range=(0.5, 0.95, 0.05)):
    """Compute mAP over a range of IoU thresholds (COCO-style)."""
    start, end, step = iou_range
    thresholds = np.arange(start, end + step, step)

    maps = []
    for thresh in thresholds:
        mAP, _ = evaluate_detections(predictions, targets, iou_thresh=thresh)
        maps.append(mAP)

    return np.mean(maps), maps, thresholds.tolist()


def compute_confusion_matrix(predictions, targets, iou_thresh=0.5):
    """Compute confusion matrix for detections."""
    n = NUM_CLASSES + 1  # +1 for background/missed
    cm = np.zeros((n, n), dtype=np.int32)

    for pred, gt in zip(predictions, targets):
        pred_boxes = pred['boxes'].cpu().numpy() if torch.is_tensor(pred['boxes']) else pred['boxes']
        pred_labels = pred['labels'].cpu().numpy() if torch.is_tensor(pred['labels']) else pred['labels']
        gt_boxes = gt['boxes'].cpu().numpy() if torch.is_tensor(gt['boxes']) else gt['boxes']
        gt_labels = gt['labels'].cpu().numpy() if torch.is_tensor(gt['labels']) else gt['labels']

        gt_matched = [False] * len(gt_boxes)
        pred_matched = [False] * len(pred_boxes)

        # match predictions to ground truths
        if len(pred_boxes) > 0 and len(gt_boxes) > 0:
            pred_t = torch.from_numpy(np.array(pred_boxes, dtype=np.float32))
            gt_t = torch.from_numpy(np.array(gt_boxes, dtype=np.float32))
            ious = box_iou(pred_t, gt_t).numpy()

            # greedy matching by highest IoU
            for _ in range(min(len(pred_boxes), len(gt_boxes))):
                max_iou = 0
                max_i, max_j = -1, -1
                for i in range(len(pred_boxes)):
                    if pred_matched[i]:
                        continue
                    for j in range(len(gt_boxes)):
                        if gt_matched[j]:
                            continue
                        if ious[i, j] > max_iou:
                            max_iou = ious[i, j]
                            max_i, max_j = i, j

                if max_iou >= iou_thresh:
                    gt_matched[max_j] = True
                    pred_matched[max_i] = True
                    gt_cls = int(gt_labels[max_j])
                    pred_cls = int(pred_labels[max_i])
                    cm[gt_cls, pred_cls] += 1
                else:
                    break

        # unmatched ground truths -> missed (background prediction)
        for j, matched in enumerate(gt_matched):
            if not matched:
                gt_cls = int(gt_labels[j])
                cm[gt_cls, NUM_CLASSES] += 1  # predicted as background

        # unmatched predictions -> false positives (background as class)
        for i, matched in enumerate(pred_matched):
            if not matched:
                pred_cls = int(pred_labels[i])
                cm[NUM_CLASSES, pred_cls] += 1  # background predicted as class

    return cm


def plot_confusion_matrix(cm, save_path=None, normalize=True):
    """Plot confusion matrix as heatmap."""
    import matplotlib.pyplot as plt

    classes = VOC_CLASSES + ['background']

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = cm.astype(float) / (row_sums + 1e-6)
    else:
        cm_norm = cm.astype(float)

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_norm, cmap='Blues')

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)

    ax.set_xlabel('Predicted')
    ax.set_ylabel('Ground Truth')
    ax.set_title('Detection Confusion Matrix')

    plt.colorbar(im, ax=ax)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved confusion matrix to {save_path}")
    plt.close()

    return fig


def print_eval_results(mAP, aps, mAP_range=None):
    print(f"\nmAP@0.5: {mAP * 100:.2f}%")
    if mAP_range is not None:
        print(f"mAP@0.5:0.95: {mAP_range * 100:.2f}%")
    print("-" * 30)
    for cls, ap in sorted(aps.items()):
        print(f"  {VOC_CLASSES[cls]:15s}: {ap * 100:.2f}%")


if __name__ == '__main__':
    # test with fake data
    predictions = [
        {'boxes': torch.tensor([[100, 100, 200, 200], [50, 50, 150, 150]]),
         'scores': torch.tensor([0.9, 0.7]),
         'labels': torch.tensor([0, 0])},
        {'boxes': torch.tensor([[200, 200, 300, 300]]),
         'scores': torch.tensor([0.8]),
         'labels': torch.tensor([1])}
    ]

    targets = [
        {'boxes': torch.tensor([[100, 100, 200, 200], [60, 60, 160, 160]]),
         'labels': torch.tensor([0, 0])},
        {'boxes': torch.tensor([[200, 200, 300, 300]]),
         'labels': torch.tensor([1])}
    ]

    mAP, aps = evaluate_detections(predictions, targets)
    print_eval_results(mAP, aps)

    # test mAP range
    print("\nTesting mAP@0.5:0.95...")
    mAP_range, maps, thresholds = evaluate_detections_range(predictions, targets)
    print(f"mAP@0.5:0.95: {mAP_range * 100:.2f}%")
    for t, m in zip(thresholds, maps):
        print(f"  IoU={t:.2f}: {m * 100:.2f}%")

    # test confusion matrix
    print("\nTesting confusion matrix...")
    cm = compute_confusion_matrix(predictions, targets)
    print(f"Confusion matrix shape: {cm.shape}")
    print(cm[:3, :3])  # show first 3x3
