import os
import xml.etree.ElementTree as ET
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import random
import numpy as np

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VOC_CLASSES, IMG_SIZE

# augmentation config
AUG_BRIGHTNESS = 0.3
AUG_CONTRAST = 0.3
AUG_SATURATION = 0.3
AUG_HUE = 0.1
MOSAIC_PROB = 0.5


class VOCDataset(Dataset):
    def __init__(self, root, years=['2007'], split='trainval', augment=True):
        self.root = root
        self.augment = augment
        self.img_size = IMG_SIZE
        self.class_to_idx = {c: i for i, c in enumerate(VOC_CLASSES)}

        # support single year string or list
        if isinstance(years, str):
            years = [years]

        # collect samples from all years
        self.samples = []  # list of (img_dir, ann_dir, img_id)
        for year in years:
            voc_root = os.path.join(root, f'VOC{year}')
            img_dir = os.path.join(voc_root, 'JPEGImages')
            ann_dir = os.path.join(voc_root, 'Annotations')
            split_file = os.path.join(voc_root, 'ImageSets', 'Main', f'{split}.txt')

            with open(split_file, 'r') as f:
                for line in f.readlines():
                    img_id = line.strip()
                    self.samples.append((img_dir, ann_dir, img_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # mosaic augmentation
        if self.augment and random.random() < MOSAIC_PROB:
            img, boxes, labels = self.mosaic(idx)
            boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))
            labels = torch.tensor(labels, dtype=torch.long) if labels else torch.zeros((0,), dtype=torch.long)
        else:
            img_dir, ann_dir, img_id = self.samples[idx]

            # load image
            img_path = os.path.join(img_dir, f'{img_id}.jpg')
            img = Image.open(img_path).convert('RGB')
            orig_w, orig_h = img.size

            # load annotations
            ann_path = os.path.join(ann_dir, f'{img_id}.xml')
            boxes, labels = self.parse_annotation(ann_path)

            # convert to tensor
            boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4))
            labels = torch.tensor(labels, dtype=torch.long) if labels else torch.zeros((0,), dtype=torch.long)

            # resize
            img, boxes = self.resize(img, boxes, orig_w, orig_h)

        # augmentation
        if self.augment:
            img = self.color_jitter(img)
            img, boxes = self.random_flip(img, boxes)

        # to tensor and normalize
        img = F.to_tensor(img)
        img = F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        target = {'boxes': boxes, 'labels': labels}
        return img, target

    def parse_annotation(self, path):
        tree = ET.parse(path)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in self.class_to_idx:
                continue

            difficult = obj.find('difficult')
            if difficult is not None and int(difficult.text) == 1:
                continue

            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(self.class_to_idx[name])

        return boxes, labels

    def random_flip(self, img, boxes):
        if random.random() > 0.5:
            img = F.hflip(img)
            w = img.size[0]
            if len(boxes) > 0:
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        return img, boxes

    def color_jitter(self, img):
        # random brightness
        if random.random() > 0.5:
            factor = 1.0 + random.uniform(-AUG_BRIGHTNESS, AUG_BRIGHTNESS)
            img = F.adjust_brightness(img, factor)
        # random contrast
        if random.random() > 0.5:
            factor = 1.0 + random.uniform(-AUG_CONTRAST, AUG_CONTRAST)
            img = F.adjust_contrast(img, factor)
        # random saturation
        if random.random() > 0.5:
            factor = 1.0 + random.uniform(-AUG_SATURATION, AUG_SATURATION)
            img = F.adjust_saturation(img, factor)
        # random hue
        if random.random() > 0.5:
            factor = random.uniform(-AUG_HUE, AUG_HUE)
            img = F.adjust_hue(img, factor)
        return img

    def load_sample(self, idx):
        img_dir, ann_dir, img_id = self.samples[idx]
        img_path = os.path.join(img_dir, f'{img_id}.jpg')
        img = Image.open(img_path).convert('RGB')
        ann_path = os.path.join(ann_dir, f'{img_id}.xml')
        boxes, labels = self.parse_annotation(ann_path)
        return img, boxes, labels

    def mosaic(self, idx):
        # get 4 random images
        indices = [idx] + [random.randint(0, len(self.samples) - 1) for _ in range(3)]

        # create output image (2x size, then resize)
        s = self.img_size
        out_img = Image.new('RGB', (s * 2, s * 2))
        out_boxes = []
        out_labels = []

        # mosaic center point
        cx = s + random.randint(-s // 4, s // 4)
        cy = s + random.randint(-s // 4, s // 4)

        for i, idx_i in enumerate(indices):
            img, boxes, labels = self.load_sample(idx_i)
            orig_w, orig_h = img.size

            # resize image to fit quadrant
            if i == 0:  # top-left
                x1, y1, x2, y2 = cx - s, cy - s, cx, cy
            elif i == 1:  # top-right
                x1, y1, x2, y2 = cx, cy - s, cx + s, cy
            elif i == 2:  # bottom-left
                x1, y1, x2, y2 = cx - s, cy, cx, cy + s
            else:  # bottom-right
                x1, y1, x2, y2 = cx, cy, cx + s, cy + s

            # resize and paste
            w, h = x2 - x1, y2 - y1
            img_resized = img.resize((w, h), Image.BILINEAR)
            out_img.paste(img_resized, (max(0, x1), max(0, y1)))

            # transform boxes
            scale_x = w / orig_w
            scale_y = h / orig_h
            for box, label in zip(boxes, labels):
                bx1 = box[0] * scale_x + x1
                by1 = box[1] * scale_y + y1
                bx2 = box[2] * scale_x + x1
                by2 = box[3] * scale_y + y1

                # clip to valid region
                bx1 = max(0, min(bx1, s * 2))
                by1 = max(0, min(by1, s * 2))
                bx2 = max(0, min(bx2, s * 2))
                by2 = max(0, min(by2, s * 2))

                # skip tiny boxes
                if bx2 - bx1 > 5 and by2 - by1 > 5:
                    out_boxes.append([bx1, by1, bx2, by2])
                    out_labels.append(label)

        # resize to final size
        out_img = out_img.resize((s, s), Image.BILINEAR)
        scale = 0.5
        out_boxes = [[b[0] * scale, b[1] * scale, b[2] * scale, b[3] * scale] for b in out_boxes]

        return out_img, out_boxes, out_labels

    def resize(self, img, boxes, orig_w, orig_h):
        img = F.resize(img, (self.img_size, self.img_size))

        if len(boxes) > 0:
            scale_x = self.img_size / orig_w
            scale_y = self.img_size / orig_h
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y

        return img, boxes


def collate_fn(batch):
    imgs = []
    targets = []
    for img, target in batch:
        imgs.append(img)
        targets.append(target)
    return torch.stack(imgs, 0), targets


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    data_root = './data/VOCdevkit'

    if os.path.exists(data_root):
        # test single year
        ds07 = VOCDataset(data_root, years='2007', split='trainval')
        print(f"VOC 2007 trainval: {len(ds07)} images")

        # test combined (if 2012 exists)
        voc12_path = os.path.join(data_root, 'VOC2012')
        if os.path.exists(voc12_path):
            ds_combined = VOCDataset(data_root, years=['2007', '2012'], split='trainval')
            print(f"VOC 07+12 trainval: {len(ds_combined)} images")

        # test set
        test_path = os.path.join(data_root, 'VOC2007', 'ImageSets', 'Main', 'test.txt')
        if os.path.exists(test_path):
            ds_test = VOCDataset(data_root, years='2007', split='test', augment=False)
            print(f"VOC 2007 test: {len(ds_test)} images")

        # visualize sample
        img, target = ds07[10]
        print(f"\nSample - shape: {img.shape}, boxes: {target['boxes'].shape}")

        fig, ax = plt.subplots(1)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_vis = img * std + mean
        img_vis = img_vis.permute(1, 2, 0).numpy().clip(0, 1)
        ax.imshow(img_vis)

        for box, label in zip(target['boxes'], target['labels']):
            x1, y1, x2, y2 = box.tolist()
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                      linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1-5, VOC_CLASSES[label], color='red', fontsize=8)

        plt.savefig('sample.png')
        print("Saved sample.png")
    else:
        print(f"Data not found at {data_root}")
        print("\nDownload VOC 2007 + 2012:")
        print("  mkdir -p data && cd data")
        print("  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar")
        print("  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar")
        print("  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar")
        print("  tar -xf VOCtrainval_06-Nov-2007.tar")
        print("  tar -xf VOCtest_06-Nov-2007.tar")
        print("  tar -xf VOCtrainval_11-May-2012.tar")
