"""
Penn-Fudan Pedestrian Dataset Loader
-------------------------------------
Download: https://www.cis.upenn.edu/~jshi/ped_html/
Expected directory structure:
    PennFudanPed/
        PNGImages/   ← .png images
        PedMasks/    ← mask images (pixel value = instance id)
"""

import os
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as F


class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs  = sorted(os.listdir(os.path.join(root, "PNGImages")))
        self.masks = sorted(os.listdir(os.path.join(root, "PedMasks")))

    def __getitem__(self, idx):
        img_path  = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks",  self.masks[idx])

        # Load & resize to 512x512
        img  = Image.open(img_path).convert("RGB").resize((512, 512))
        mask = Image.open(mask_path).resize((512, 512), Image.NEAREST)

        mask = torch.as_tensor(np.array(mask), dtype=torch.uint8)

        # Each non-zero pixel value is a unique instance
        obj_ids = torch.unique(mask)
        obj_ids = obj_ids[obj_ids != 0]          # remove background
        num_objs = len(obj_ids)

        # Binary mask per instance
        masks = (mask == obj_ids[:, None, None])  # (N, H, W)

        # Bounding box from mask
        boxes = []
        valid = []
        for i, m in enumerate(masks):
            pos = torch.where(m)
            if len(pos[0]) == 0:
                valid.append(False)
                continue
            xmin = pos[1].min().item()
            xmax = pos[1].max().item()
            ymin = pos[0].min().item()
            ymax = pos[0].max().item()
            if xmax <= xmin or ymax <= ymin:
                valid.append(False)
                continue
            boxes.append([xmin, ymin, xmax, ymax])
            valid.append(True)

        valid = torch.tensor(valid)
        masks = masks[valid]

        if len(boxes) == 0:
            boxes  = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
            area   = torch.zeros(0, dtype=torch.float32)
        else:
            boxes  = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.ones(len(boxes), dtype=torch.int64)  # class 1 = person
            area   = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {
            "boxes":    boxes,
            "labels":   labels,
            "masks":    masks,
            "image_id": torch.tensor([idx]),
            "area":     area,
            "iscrowd":  torch.zeros(len(boxes), dtype=torch.int64),
        }

        img = F.to_tensor(img)

        if self.transforms:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def collate_fn(batch):
    return tuple(zip(*batch))
