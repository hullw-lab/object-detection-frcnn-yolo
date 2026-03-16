# =============================================================================
# NOTEBOOK 4 — Faster R-CNN (MobileNetV3) on Oxford-IIIT Pet Dataset (10 breeds)
# Copy each section into a separate Colab cell and run top to bottom.
# Runtime: GPU (Runtime > Change runtime type > T4 GPU)
# =============================================================================


# ─── CELL 1: Install & verify GPU ────────────────────────────────────────────

"""
!pip install -q torch torchvision --upgrade
"""

import torch
import torchvision
print(f"PyTorch     : {torch.__version__}")
print(f"Torchvision : {torchvision.__version__}")
print(f"CUDA        : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU         : {torch.cuda.get_device_name(0)}")


# ─── CELL 2: Download Oxford-IIIT Pet Dataset ─────────────────────────────────

"""
import os

!wget -q --show-progress https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
!wget -q --show-progress https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz

!tar -xzf images.tar.gz
!tar -xzf annotations.tar.gz

os.makedirs("oxford-iiit-pet/annotations", exist_ok=True)
!mv images oxford-iiit-pet/images
!mv annotations/xmls oxford-iiit-pet/annotations/xmls

print(f"Images : {len(os.listdir('oxford-iiit-pet/images'))}")
print(f"XMLs   : {len(os.listdir('oxford-iiit-pet/annotations/xmls'))}")
"""


# ─── CELL 3: Oxford Pet Dataset class ────────────────────────────────────────
# Reads images + Pascal VOC XML annotations.
# Returns torchvision-compatible targets: boxes, labels, image_id, area, iscrowd.

import os
import random
import xml.etree.ElementTree as ET
import numpy as np
import torch
from PIL import Image
from pathlib import Path
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader, random_split

# ── Config ───────────────────────────────────────────────────────────────────
DATA_ROOT     = "oxford-iiit-pet"
IMG_SIZE      = 512
MAX_PER_BREED = 50

BREEDS = [
    "Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair",
    "Egyptian_Mau", "Maine_Coon", "Persian", "Ragdoll", "Siamese",
]
BREED2IDX = {b: i + 1 for i, b in enumerate(BREEDS)}  # 1-indexed (0 = background)
NUM_CLASSES = len(BREEDS) + 1                           # +1 for background
# ─────────────────────────────────────────────────────────────────────────────


def parse_voc_xml(xml_path, orig_w, orig_h, class_id):
    """Return list of [xmin, ymin, xmax, ymax] boxes from a VOC XML file."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes = []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        if bb is None:
            continue
        xmin = max(0, int(float(bb.find("xmin").text)))
        ymin = max(0, int(float(bb.find("ymin").text)))
        xmax = min(orig_w, int(float(bb.find("xmax").text)))
        ymax = min(orig_h, int(float(bb.find("ymax").text)))
        if xmax > xmin and ymax > ymin:
            boxes.append([xmin, ymin, xmax, ymax])
    return boxes


def scale_boxes(boxes, orig_w, orig_h, new_w=IMG_SIZE, new_h=IMG_SIZE):
    """Scale bounding boxes from original image size to resized image size."""
    sx = new_w / orig_w
    sy = new_h / orig_h
    scaled = []
    for xmin, ymin, xmax, ymax in boxes:
        scaled.append([xmin*sx, ymin*sy, xmax*sx, ymax*sy])
    return scaled


class OxfordPetDataset(Dataset):
    def __init__(self, samples):
        """
        samples: list of (img_path, xml_path, class_id) tuples
        """
        self.samples = samples

    def __getitem__(self, idx):
        img_path, xml_path, class_id = self.samples[idx]

        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size
        img = img.resize((IMG_SIZE, IMG_SIZE))

        raw_boxes    = parse_voc_xml(xml_path, orig_w, orig_h, class_id)
        scaled_boxes = scale_boxes(raw_boxes, orig_w, orig_h)

        if len(scaled_boxes) == 0:
            boxes  = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
            area   = torch.zeros(0, dtype=torch.float32)
        else:
            boxes  = torch.as_tensor(scaled_boxes, dtype=torch.float32)
            labels = torch.full((len(boxes),), class_id, dtype=torch.int64)
            area   = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {
            "boxes":    boxes,
            "labels":   labels,
            "image_id": torch.tensor([idx]),
            "area":     area,
            "iscrowd":  torch.zeros(len(boxes), dtype=torch.int64),
        }

        return TF.to_tensor(img), target

    def __len__(self):
        return len(self.samples)


def collate_fn(batch):
    return tuple(zip(*batch))


# ── Build sample list ─────────────────────────────────────────────────────────
def build_sample_list(data_root, breeds, breed2idx, max_per_breed):
    img_dir = os.path.join(data_root, "images")
    ann_dir = os.path.join(data_root, "annotations", "xmls")
    random.seed(42)
    samples, skipped = [], 0

    for breed in breeds:
        imgs = [f for f in os.listdir(img_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
                and Path(f).stem.rsplit("_", 1)[0] == breed]
        random.shuffle(imgs)
        imgs = imgs[:max_per_breed]

        for fname in imgs:
            stem     = Path(fname).stem
            xml_path = os.path.join(ann_dir, stem + ".xml")
            if not os.path.exists(xml_path):
                skipped += 1; continue
            samples.append((os.path.join(img_dir, fname), xml_path, breed2idx[breed]))

    random.shuffle(samples)
    print(f"Total samples : {len(samples)}  (skipped {skipped} — no XML)")
    return samples

all_samples = build_sample_list(DATA_ROOT, BREEDS, BREED2IDX, MAX_PER_BREED)

# ── Split ─────────────────────────────────────────────────────────────────────
n       = len(all_samples)
n_train = int(0.70 * n)
n_val   = int(0.15 * n)
n_test  = n - n_train - n_val

train_samples = all_samples[:n_train]
val_samples   = all_samples[n_train : n_train + n_val]
test_samples  = all_samples[n_train + n_val :]

train_set = OxfordPetDataset(train_samples)
val_set   = OxfordPetDataset(val_samples)
test_set  = OxfordPetDataset(test_samples)

# num_workers=0 avoids Colab multiprocessing errors
train_loader = DataLoader(train_set, batch_size=2, shuffle=True,  collate_fn=collate_fn, num_workers=0)
val_loader   = DataLoader(val_set,   batch_size=2, shuffle=False, collate_fn=collate_fn, num_workers=0)
test_loader  = DataLoader(test_set,  batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0)

print(f"Train: {n_train}  |  Val: {n_val}  |  Test: {n_test}")

# Sanity check: show first sample
img_t, tgt = train_set[0]
print(f"Image shape : {img_t.shape}")
print(f"Boxes       : {tgt['boxes']}")
print(f"Label       : {tgt['labels']}  ({BREEDS[tgt['labels'][0].item()-1]})")


# ─── CELL 4: Build Faster R-CNN model ────────────────────────────────────────

import warnings
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def build_model(num_classes):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_FPN_Weights
            model = fasterrcnn_mobilenet_v3_large_fpn(
                weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)
        except ImportError:
            model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = build_model(NUM_CLASSES).to(device)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Device               : {device}")
print(f"Classes              : {NUM_CLASSES} (background + {len(BREEDS)} breeds)")
print(f"Trainable parameters : {n_params:,}")


# ─── CELL 5: Optimizer, scheduler, AMP ───────────────────────────────────────

import time

EPOCHS   = 18     # assignment spec: 15–20 for Oxford Pet
LR       = 0.005
PATIENCE = 5

# Separate LRs: backbone (pretrained) gets 10x lower LR
params = [
    {"params": [p for n, p in model.named_parameters()
                if "backbone" not in n and p.requires_grad]},
    {"params": [p for n, p in model.named_parameters()
                if "backbone"     in n and p.requires_grad], "lr": LR / 10},
]
optimizer = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                             step_size=max(1, EPOCHS // 3),
                                             gamma=0.1)
use_amp = device.type == "cuda"
scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)
print(f"Mixed precision : {use_amp}")


# ─── CELL 6: Training helpers ─────────────────────────────────────────────────

def train_one_epoch(model, optimizer, loader, device, scaler, use_amp):
    model.train()
    total = 0.0
    for imgs, targets in loader:
        imgs    = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=use_amp):
            loss_dict = model(imgs, targets)
            loss      = sum(loss_dict.values())
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item()
    return total / max(len(loader), 1)


@torch.no_grad()
def compute_val_loss(model, loader, device, use_amp):
    model.train()   # train mode required to get loss_dict
    total = 0.0
    for imgs, targets in loader:
        imgs    = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=use_amp):
            loss_dict = model(imgs, targets)
        total += sum(loss_dict.values()).item()
    return total / max(len(loader), 1)


# ─── CELL 7: Training loop ────────────────────────────────────────────────────

import json, os

os.makedirs("output_pet_frcnn", exist_ok=True)

history      = {"train_loss": [], "val_loss": []}
best_val     = float("inf")
patience_ctr = 0
t_start      = time.time()

for epoch in range(1, EPOCHS + 1):
    t_loss = train_one_epoch(model, optimizer, train_loader, device, scaler, use_amp)
    v_loss = compute_val_loss(model, val_loader, device, use_amp)
    scheduler.step()

    history["train_loss"].append(round(t_loss, 5))
    history["val_loss"].append(round(v_loss,   5))

    flag = ""
    if v_loss < best_val:
        best_val     = v_loss
        patience_ctr = 0
        torch.save(model.state_dict(), "output_pet_frcnn/fasterrcnn_best.pth")
        flag = "  ✓ saved"
    else:
        patience_ctr += 1
        if patience_ctr >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"[{epoch:02d}/{EPOCHS}]  train={t_loss:.4f}  val={v_loss:.4f}{flag}")

total_time = time.time() - t_start
print(f"\nDone in {total_time/60:.1f} min")

history["training_time_sec"] = round(total_time, 1)
with open("output_pet_frcnn/train_history.json", "w") as f:
    json.dump(history, f, indent=2)


# ─── CELL 8: Plot loss curves ─────────────────────────────────────────────────

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.plot(history["train_loss"], label="Train Loss", marker="o")
plt.plot(history["val_loss"],   label="Val Loss",   marker="s")
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.title("Faster R-CNN — Oxford Pet — Training & Validation Loss")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig("output_pet_frcnn/loss_curve.png", dpi=120)
plt.show()
print("Saved: output_pet_frcnn/loss_curve.png")


# ─── CELL 9: Evaluate — mAP@0.5, Precision, Recall, FPS ──────────────────────

from torchvision.ops import box_iou

def evaluate_model(model, loader, device, num_classes,
                   iou_thresh=0.5, score_thresh=0.5):
    """
    Per-class and overall mAP@0.5, Precision, Recall.
    For Faster R-CNN the class labels come from pred['labels'],
    so we match predictions to GT only when classes agree.
    """
    model.eval()
    # Accumulators per class (index 1..num_classes-1)
    tp_per  = {c: 0 for c in range(1, num_classes)}
    fp_per  = {c: 0 for c in range(1, num_classes)}
    fn_per  = {c: 0 for c in range(1, num_classes)}
    total_time = 0.0
    n_images   = 0

    with torch.no_grad():
        for imgs, targets in loader:
            imgs_gpu = [img.to(device) for img in imgs]

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            outputs = model(imgs_gpu)
            if device.type == "cuda":
                torch.cuda.synchronize()
            total_time += time.time() - t0
            n_images   += len(imgs)

            for pred, tgt in zip(outputs, targets):
                gt_boxes  = tgt["boxes"]
                gt_labels = tgt["labels"]

                pred_boxes  = pred["boxes"].cpu()
                pred_labels = pred["labels"].cpu()
                pred_scores = pred["scores"].cpu()

                keep        = pred_scores >= score_thresh
                pred_boxes  = pred_boxes[keep]
                pred_labels = pred_labels[keep]
                pred_scores = pred_scores[keep]

                # Per-class matching
                for cls in range(1, num_classes):
                    gt_mask   = gt_labels  == cls
                    pred_mask = pred_labels == cls

                    gt_b   = gt_boxes[gt_mask]
                    pred_b = pred_boxes[pred_mask]
                    scores = pred_scores[pred_mask]

                    if len(pred_b) == 0 and len(gt_b) == 0:
                        continue
                    if len(pred_b) == 0:
                        fn_per[cls] += len(gt_b); continue
                    if len(gt_b) == 0:
                        fp_per[cls] += len(pred_b); continue

                    iou        = box_iou(pred_b, gt_b)
                    matched_gt = torch.zeros(len(gt_b), dtype=torch.bool)

                    for i in scores.argsort(descending=True):
                        best_gt = iou[i].argmax().item()
                        if iou[i, best_gt] >= iou_thresh and not matched_gt[best_gt]:
                            tp_per[cls] += 1; matched_gt[best_gt] = True
                        else:
                            fp_per[cls] += 1
                    fn_per[cls] += int((~matched_gt).sum())

    # Aggregate across classes (macro average)
    precisions, recalls = [], []
    for cls in range(1, num_classes):
        tp = tp_per[cls]; fp = fp_per[cls]; fn = fn_per[cls]
        precisions.append(tp / (tp + fp + 1e-8))
        recalls.append(   tp / (tp + fn + 1e-8))

    precision = float(np.mean(precisions))
    recall    = float(np.mean(recalls))
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    fps       = n_images / total_time if total_time > 0 else 0

    return {
        "mAP@0.5":   round(f1,        4),
        "precision":  round(precision, 4),
        "recall":     round(recall,    4),
        "f1":         round(f1,        4),
        "fps":        round(fps,       2),
    }

# Load best weights
model.load_state_dict(torch.load("output_pet_frcnn/fasterrcnn_best.pth", map_location=device))

print("Evaluating on test set...")
results = evaluate_model(model, test_loader, device, NUM_CLASSES)

print("\n── Faster R-CNN Oxford Pet Results ──────────────")
for k, v in results.items():
    print(f"  {k:<15} {v}")
print("─────────────────────────────────────────────────")

with open("output_pet_frcnn/eval_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved: output_pet_frcnn/eval_results.json")


# ─── CELL 10: Visualise predictions ──────────────────────────────────────────

import matplotlib.pyplot as plt
import matplotlib.patches as patches

IDX2BREED = {v: k for k, v in BREED2IDX.items()}   # reverse lookup

def show_predictions(model, dataset, device, n=6, score_thresh=0.5,
                     save_dir="output_pet_frcnn/predictions"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    for idx in range(min(n, len(dataset))):
        img_tensor, target = dataset[idx]

        with torch.no_grad():
            output = model([img_tensor.to(device)])[0]

        img_np = img_tensor.permute(1, 2, 0).numpy()
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Ground truth
        axes[0].imshow(img_np); axes[0].set_title("Ground Truth")
        for box, lbl in zip(target["boxes"], target["labels"]):
            x1, y1, x2, y2 = box.tolist()
            breed_name = IDX2BREED.get(lbl.item(), str(lbl.item()))
            axes[0].add_patch(patches.Rectangle(
                (x1, y1), x2-x1, y2-y1, linewidth=2,
                edgecolor="lime", facecolor="none"))
            axes[0].text(x1, max(y1-4, 0), breed_name, color="lime", fontsize=8,
                         bbox=dict(facecolor="black", alpha=0.4, pad=1, edgecolor="none"))

        # Predictions
        axes[1].imshow(img_np); axes[1].set_title("Predictions (red)")
        keep = output["scores"] >= score_thresh
        for box, lbl, score in zip(output["boxes"][keep].cpu(),
                                    output["labels"][keep].cpu(),
                                    output["scores"][keep].cpu()):
            x1, y1, x2, y2 = box.tolist()
            breed_name = IDX2BREED.get(lbl.item(), str(lbl.item()))
            axes[1].add_patch(patches.Rectangle(
                (x1, y1), x2-x1, y2-y1, linewidth=2,
                edgecolor="red", facecolor="none"))
            axes[1].text(x1, max(y1-4, 0), f"{breed_name} {score:.2f}",
                         color="red", fontsize=8,
                         bbox=dict(facecolor="white", alpha=0.5, pad=1, edgecolor="none"))

        for ax in axes: ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"pred_{idx:03d}.png"), dpi=120)
        plt.show()
        plt.close()

show_predictions(model, test_set, device, n=6)
print("Saved to output_pet_frcnn/predictions/")


# ─── CELL 11: Download outputs ────────────────────────────────────────────────

"""
import shutil
from google.colab import files

shutil.make_archive("frcnn_pet_output", "zip", "output_pet_frcnn")
files.download("frcnn_pet_output.zip")
"""
