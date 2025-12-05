#!/bin/sh
# DTU HPC Job Script Template for ViT-YOLO Training

### -- specify queue --
#BSUB -q gpuv100

### -- job name --
#BSUB -J yolo_train

### -- cores (4 per GPU) --
#BSUB -n 4
#BSUB -R "span[hosts=1]"

### -- GPU --
#BSUB -gpu "num=1:mode=exclusive_process"

### -- walltime (hh:mm) --
#BSUB -W 24:00

### -- system memory --
#BSUB -R "rusage[mem=8GB]"

### -- output files --
#BSUB -o jobs/logs/%J.out
#BSUB -e jobs/logs/%J.err

# Create directories
mkdir -p jobs/logs
mkdir -p checkpoints

# Print job info
echo "Job ID: $LSB_JOBID"
echo "Running on: $(hostname)"
nvidia-smi

# Activate environment and run training
cd ~/yolo-vit
source .venv/bin/activate

python train.py \
    --data /work3/YOUR_ID/yolo-vit/data/VOCdevkit \
    --backbone resnet50 \
    --neck fpn \
    --epochs 100 \
    --batch 32 \
    --lr 1e-4 \
    --amp \
    --save_dir /work3/YOUR_ID/yolo-vit/checkpoints/MODELNAME

echo "Training complete"

# Example configurations:
# ResNet50 + BiFPN + CBAM:
#   --backbone resnet50 --neck bifpn --attention --batch 32 --lr 1e-4
#
# Swin + BiFPN (best model):
#   --backbone swin --neck bifpn --batch 16 --lr 5e-5
#
# ViT (pretrained):
#   --backbone vit_pretrained --neck fpn --batch 32 --lr 1e-4
