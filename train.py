import os
import argparse
import time
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import BATCH_SIZE, LR, WEIGHT_DECAY, EPOCHS, IMG_SIZE, CONF_THRESH, NMS_THRESH
from data.voc import VOCDataset, collate_fn
from models.detector import YOLODetector
from utils.loss import YOLOLoss
from utils.metrics import evaluate_detections


def train_one_epoch(model, loader, optimizer, loss_fn, device, epoch, scaler=None):
    model.train()
    total_loss = 0
    total_box = 0
    total_obj = 0
    total_cls = 0
    use_amp = scaler is not None

    pbar = tqdm(loader, desc=f'Epoch {epoch}')
    for imgs, targets in pbar:
        imgs = imgs.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs = model(imgs)
            loss, l_box, l_obj, l_cls = loss_fn(outputs, targets)

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        total_box += l_box
        total_obj += l_obj
        total_cls += l_cls

        pbar.set_postfix({
            'loss': f'{loss.item():.3f}',
            'box': f'{l_box:.3f}',
            'obj': f'{l_obj:.3f}',
            'cls': f'{l_cls:.3f}'
        })

    n = len(loader)
    return total_loss / n, total_box / n, total_obj / n, total_cls / n


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []

    for imgs, targets in tqdm(loader, desc='Validating'):
        imgs = imgs.to(device)
        preds = model.predict(imgs, CONF_THRESH, NMS_THRESH)

        for p in preds:
            p['boxes'] = p['boxes'].cpu()
            p['scores'] = p['scores'].cpu()
            p['labels'] = p['labels'].cpu()

        all_preds.extend(preds)
        all_targets.extend(targets)

    mAP, _ = evaluate_detections(all_preds, all_targets)
    return mAP


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./data/VOCdevkit')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--backbone', type=str, default='cnn',
                        choices=['cnn', 'resnet50', 'vit', 'vit_pretrained', 'swin'])
    parser.add_argument('--neck', type=str, default='fpn',
                        choices=['fpn', 'bifpn', 'transformer'])
    parser.add_argument('--attention', action='store_true', help='use CBAM attention in neck')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--amp', action='store_true', help='use mixed precision')
    parser.add_argument('--val_interval', type=int, default=1, help='validate every N epochs')
    parser.add_argument('--patience', type=int, default=0, help='early stopping patience (0=disabled)')
    parser.add_argument('--focal', action='store_true', help='use focal loss')
    parser.add_argument('--label_smoothing', type=float, default=0.0, help='label smoothing factor')
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # dataset with 80/20 split
    full_ds = VOCDataset(args.data, years=['2007', '2012'], split='trainval', augment=False)
    n_total = len(full_ds)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train

    # random split indices with fixed seed for reproducibility
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(n_total, generator=generator).tolist()
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    # create separate datasets
    train_ds = VOCDataset(args.data, years=['2007', '2012'], split='trainval', augment=True)
    val_ds = VOCDataset(args.data, years=['2007', '2012'], split='trainval', augment=False)

    # subset with indices
    train_ds = torch.utils.data.Subset(train_ds, train_indices)
    val_ds = torch.utils.data.Subset(val_ds, val_indices)

    pin_mem = device.type == 'cuda'
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=4, collate_fn=collate_fn, pin_memory=pin_mem)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=4, collate_fn=collate_fn, pin_memory=pin_mem)
    print(f"Training samples: {len(train_ds)}, Validation samples: {len(val_ds)}")

    # model
    model = YOLODetector(backbone=args.backbone, neck=args.neck, attention=args.attention).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model: backbone={args.backbone}, neck={args.neck}, attention={args.attention}")
    print(f"Model parameters: {params / 1e6:.2f}M")

    # resume checkpoint
    start_epoch = 0
    ckpt = None
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        start_epoch = ckpt['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")

    # optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WEIGHT_DECAY)
    if ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = YOLOLoss(focal=args.focal, label_smoothing=args.label_smoothing)
    if args.focal:
        print("Using focal loss")
    if args.label_smoothing > 0:
        print(f"Using label smoothing: {args.label_smoothing}")

    # mixed precision
    scaler = torch.amp.GradScaler('cuda') if args.amp and device.type == 'cuda' else None
    if scaler:
        print("Using mixed precision (AMP)")

    os.makedirs(args.save_dir, exist_ok=True)

    # training history
    history = {'epochs': [], 'loss': [], 'box_loss': [], 'obj_loss': [], 'cls_loss': [],
               'lr': [], 'val_mAP': [], 'time': []}

    # training loop
    best_loss = float('inf')
    best_mAP = 0.0
    patience_counter = 0

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        loss, l_box, l_obj, l_cls = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch, scaler
        )
        scheduler.step()

        t1 = time.time()
        lr = optimizer.param_groups[0]['lr']
        epoch_time = t1 - t0

        # validate
        val_mAP = 0.0
        if args.val_interval > 0 and (epoch + 1) % args.val_interval == 0:
            val_mAP = validate(model, val_loader, device)
            print(f"Epoch {epoch}: loss={loss:.4f} box={l_box:.4f} obj={l_obj:.4f} "
                  f"cls={l_cls:.4f} lr={lr:.6f} mAP={val_mAP*100:.2f}% time={epoch_time:.1f}s")
        else:
            print(f"Epoch {epoch}: loss={loss:.4f} box={l_box:.4f} obj={l_obj:.4f} "
                  f"cls={l_cls:.4f} lr={lr:.6f} time={epoch_time:.1f}s")

        # update history
        history['epochs'].append(epoch)
        history['loss'].append(loss)
        history['box_loss'].append(l_box)
        history['obj_loss'].append(l_obj)
        history['cls_loss'].append(l_cls)
        history['lr'].append(lr)
        history['val_mAP'].append(val_mAP)
        history['time'].append(epoch_time)

        # save history
        with open(os.path.join(args.save_dir, 'history.json'), 'w') as f:
            json.dump(history, f, indent=2)

        # save checkpoint
        model_state = model.state_dict()
        ckpt = {
            'epoch': epoch,
            'model': model_state,
            'optimizer': optimizer.state_dict(),
            'loss': loss,
            'mAP': val_mAP
        }
        torch.save(ckpt, os.path.join(args.save_dir, 'last.pt'))

        # save best by loss
        if loss < best_loss:
            best_loss = loss
            torch.save(ckpt, os.path.join(args.save_dir, 'best.pt'))
            print(f"  Saved best model (loss={loss:.4f})")

        # save best by mAP
        if val_mAP > best_mAP:
            best_mAP = val_mAP
            torch.save(ckpt, os.path.join(args.save_dir, 'best_mAP.pt'))
            print(f"  Saved best mAP model (mAP={val_mAP*100:.2f}%)")
            patience_counter = 0
        elif val_mAP > 0:
            patience_counter += 1

        # early stopping
        if args.patience > 0 and patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    print(f"\nTraining complete. Best loss: {best_loss:.4f}, Best mAP: {best_mAP*100:.2f}%")


if __name__ == '__main__':
    main()
