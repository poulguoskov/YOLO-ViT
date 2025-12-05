IMG_SIZE = 416

VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
NUM_CLASSES = len(VOC_CLASSES)

# Anchors for 3 scales (width, height) - normalized to 416
# Small objects (52x52 grid)
ANCHORS_S = [(10, 13), (16, 30), (33, 23)]
# Medium objects (26x26 grid)
ANCHORS_M = [(30, 61), (62, 45), (59, 119)]
# Large objects (13x13 grid)
ANCHORS_L = [(116, 90), (156, 198), (373, 326)]

ANCHORS = [ANCHORS_S, ANCHORS_M, ANCHORS_L]
NUM_ANCHORS = 3

# Feature map sizes for input 416x416
STRIDES = [8, 16, 32]
GRID_SIZES = [52, 26, 13]

# Training
BATCH_SIZE = 16
LR = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 100

# Loss weights
LAMBDA_BOX = 5.0
LAMBDA_OBJ = 1.0
LAMBDA_NOOBJ = 0.5
LAMBDA_CLS = 1.0

# NMS
CONF_THRESH = 0.5
NMS_THRESH = 0.45


if __name__ == '__main__':
    print(f"Classes: {NUM_CLASSES}")
    print(f"Anchors per scale: {NUM_ANCHORS}")
    print(f"Strides: {STRIDES}")
