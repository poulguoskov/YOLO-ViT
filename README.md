# ViT-YOLO Object Detection

A deep learning project exploring Vision Transformers as backbones for YOLO-style object detection. This implementation compares CNN and transformer architectures on Pascal VOC object detection, built from scratch for DTU 02456 Deep Learning (Fall 2024).

> **Note on main.ipynb:** The notebook displays results and visualizations with hardcoded values from the experiments. It doesn't make sense to demonstrate training in a notebook for this project, and running inference would require uploading trained model weights. The notebook serves to present the key findings as plots and tables.

## Setup

This project uses `uv` for Python environment management.

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/poulguoskov/YOLO-ViT.git
cd YOLO-ViT

# Install dependencies
uv sync
```

### Dataset Setup

Download Pascal VOC 2007 and 2012 datasets:

```bash
# Download VOC datasets
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

# Extract to data directory
tar xf VOCtrainval_06-Nov-2007.tar -C data/
tar xf VOCtest_06-Nov-2007.tar -C data/
tar xf VOCtrainval_11-May-2012.tar -C data/
```

Your directory structure should look like:
```
yolo-vit/
├── data/
│   └── VOCdevkit/
│       ├── VOC2007/
│       └── VOC2012/
├── models/
├── utils/
└── ...
```

## Usage

### Training

Basic training command:
```bash
python train.py --backbone <backbone> --epochs 100 --amp
```

#### Training Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--data` | str | `./data/VOCdevkit` | Path to VOC dataset |
| `--epochs` | int | 100 | Number of training epochs |
| `--batch` | int | 32 | Batch size |
| `--lr` | float | 1e-4 | Learning rate |
| `--backbone` | str | `cnn` | Backbone: `cnn`, `resnet50`, `vit`, `vit_pretrained`, `swin` |
| `--neck` | str | `fpn` | Neck: `fpn`, `bifpn`, `transformer` |
| `--attention` | flag | False | Use CBAM attention modules in neck |
| `--amp` | flag | False | Use automatic mixed precision for faster training |
| `--resume` | str | None | Path to checkpoint to resume training |
| `--save_dir` | str | `./checkpoints` | Directory to save checkpoints |
| `--val_interval` | int | 1 | Validate every N epochs |
| `--patience` | int | 0 | Early stopping patience (0 = disabled) |
| `--focal` | flag | False | Use focal loss instead of BCE |
| `--label_smoothing` | float | 0.0 | Label smoothing factor |

#### Examples

**ResNet50 with BiFPN and CBAM:**
```bash
python train.py --backbone resnet50 --neck bifpn --attention --epochs 100 --batch 32 --lr 1e-4 --amp
```

**Swin Transformer (best model):**
```bash
python train.py --backbone swin --neck bifpn --epochs 100 --batch 16 --lr 5e-5 --amp
```

### Evaluation

Basic evaluation command:
```bash
python eval.py --weights <checkpoint_path> --backbone <backbone> [--neck <neck>] [--attention]
```

#### Evaluation Flags

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--data` | str | `./data/VOCdevkit` | Path to VOC dataset |
| `--weights` | str | **Required** | Path to model checkpoint (.pt file) |
| `--backbone` | str | `cnn` | Backbone architecture (must match training) |
| `--neck` | str | `fpn` | Neck architecture (must match training) |
| `--attention` | flag | False | Use CBAM attention (must match training) |
| `--batch` | int | 16 | Batch size for evaluation |
| `--conf` | float | 0.25 | Confidence threshold for detections |
| `--nms` | float | 0.45 | NMS IoU threshold |
| `--benchmark` | flag | False | Run speed benchmark (FPS, latency) |
| `--map_range` | flag | False | Compute mAP@0.5:0.95 (slower) |
| `--confusion` | flag | False | Generate confusion matrix |
| `--visualize` | int | 0 | Number of samples to visualize |
| `--save_dir` | str | `./results` | Directory to save outputs |

#### Examples

**Basic evaluation:**
```bash
python eval.py --weights checkpoints/best_mAP.pt --backbone resnet50
```

**Full evaluation with benchmarks and visualizations:**
```bash
python eval.py --weights checkpoints/best_mAP.pt --backbone swin --neck bifpn \
    --benchmark --map_range --visualize 10 --save_dir results/swin_bifpn
```

## Results

Test set: Pascal VOC 2007 test (4,952 images). Training: VOC 2007+2012 trainval (80/20 train/val split, seed=42). FPS measured on NVIDIA A100 GPU.

| Backbone | Neck | Attention | Params | Val mAP | Test mAP | mAP@.5:.95 | FPS |
|----------|------|-----------|--------|---------|----------|------------|-----|
| Custom CNN | FPN | - | 38M | 60.4% | 58.8% | 30.8% | 185 |
| Custom ViT | FPN | - | 104M | 33.3% | 32.5% | 13.1% | 51 |
| ResNet50 | FPN | - | 42M | 79.2% | 74.1% | 43.5% | 103 |
| ResNet50 | BiFPN | - | 38M | 79.5% | 74.0% | 45.2% | 106 |
| ResNet50 | BiFPN | CBAM | 38M | 80.8% | 74.4% | 44.2% | 112 |
| ViT (pretrained) | FPN | - | 104M | 86.6% | 70.1% | 42.4% | 61 |
| ViT (pretrained) | BiFPN | - | 99M | 87.8% | 71.1% | 43.9% | 56 |
| ViT (pretrained) | BiFPN | CBAM | 99M | 88.4% | 70.5% | 43.6% | 44 |
| Swin | FPN | - | 104M | 87.2% | 81.4% | 51.6% | 52 |
| **Swin** | **BiFPN** | **-** | **100M** | **87.7%** | **81.7%** | **53.3%** | **50** |
| Swin | BiFPN | CBAM | 100M | 87.5% | 80.5% | 51.3% | 44 |

## Implementation Roadmap

- [x] VOC dataset loader with augmentation
- [x] Box operations (IoU, NMS, encoding)
- [x] Custom CNN backbone
- [x] Feature Pyramid Network (FPN)
- [x] YOLO detection head with anchors
- [x] CIoU loss and training pipeline
- [x] mAP evaluation and visualization
- [x] Vision Transformer backbone
- [x] ResNet50 pretrained backbone
- [x] ViT pretrained backbone
- [x] Bidirectional FPN (BiFPN)
- [x] CBAM attention module
- [x] Swin Transformer backbone
- [x] Transformer neck with cross-scale attention
- [x] Mixed precision training
- [x] Full ablation study
- [x] Main results notebook
- [x] Technical report

## Project Structure

```
yolo-vit/
├── config.py              # Global configuration
├── train.py               # Training script
├── eval.py                # Evaluation script
├── main.ipynb             # Main results notebook
├── data/
│   └── voc.py            # VOC dataset loader
├── models/
│   ├── cnn.py            # Custom CNN backbone
│   ├── vit.py            # ViT and Swin backbones
│   ├── neck.py           # FPN, BiFPN, Transformer necks
│   ├── attention.py      # CBAM attention module
│   ├── head.py           # YOLO detection head
│   └── detector.py       # End-to-end detector
├── utils/
│   ├── boxes.py          # Box operations
│   ├── loss.py           # Loss functions
│   └── metrics.py        # Evaluation metrics
├── report/
│   ├── report.tex        # LaTeX report
│   └── references.bib    # Bibliography
└── checkpoints/          # Model weights
```
