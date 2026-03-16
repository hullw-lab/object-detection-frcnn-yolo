# =============================================================================
# NOTEBOOK 1 — Faster R-CNN on Penn-Fudan Pedestrian Dataset
# Copy each section into a separate Colab cell and run top to bottom.
# Runtime: GPU (Runtime > Change runtime type > T4 GPU)
# =============================================================================


# ─── CELL 1: Install & verify GPU ────────────────────────────────────────────
# (paste into Colab cell, then run)

"""
!pip install -q torch torchvision --upgrade
"""

import torch
print(f"PyTorch : {torch.__version__}")
print(f"CUDA    : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU     : {torch.cuda.get_device_name(0)}")


# ─── CELL 2: Download & extract Penn-Fudan dataset ───────────────────────────

"""
import os

url = "https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip"

# Download
!wget -q --show-progress "$url" -O PennFudanPed.zip

# Extract
!unzip -q PennFudanPed.zip

# Verify
imgs  = os.listdir("PennFudanPed/PNGImages")
masks = os.listdir("PennFudanPed/PedMasks")
print(f"Images: {len(imgs)},  Masks: {len(masks)}")
"""


# ─── CELL 3: Dataset class ────────────────────────────────────────────────────

import os
import numpy as np
import torch
from PIL import Image
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader, random_split, Subset

class PennFudanDataset(Dataset):
    def __init__(self, root: str):
        self.root  = root
        self.imgs  = sorted([f for f in os.listdir(os.path.join(root, "PNGImages"))
                             if not f.startswith(".")])
        self.masks = sorted([f for f in os.listdir(os.path.join(root, "PedMasks"))
                             if not f.startswith(".")])

    def __getitem__(self, idx):
        img  = Image.open(os.path.join(self.root, "PNGImages", self.imgs[idx])).convert("RGB")
        mask = Image.open(os.path.join(self.root, "PedMasks",  self.masks[idx]))

        # Resize to 512×512
        img  = img.resize((512, 512))
        mask = mask.resize((512, 512), Image.NEAREST)

        mask    = torch.as_tensor(np.array(mask), dtype=torch.uint8)
        obj_ids = torch.unique(mask)
        obj_ids = obj_ids[obj_ids != 0]            # drop background (0)
        masks   = (mask == obj_ids[:, None, None]) # (N, H, W) bool

        boxes, valid = [], []
        for m in masks:
            pos = torch.where(m)
            if len(pos[0]) == 0:
                valid.append(False); continue
            xmin, xmax = pos[1].min().item(), pos[1].max().item()
            ymin, ymax = pos[0].min().item(), pos[0].max().item()
            if xmax <= xmin or ymax <= ymin:
                valid.append(False); continue
            boxes.append([xmin, ymin, xmax, ymax])
            valid.append(True)

        valid = torch.tensor(valid)
        masks = masks[valid]

        if len(boxes) == 0:
            boxes  = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0,      dtype=torch.int64)
            area   = torch.zeros(0,      dtype=torch.float32)
        else:
            boxes  = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.ones(len(boxes), dtype=torch.int64)   # class 1 = person
            area   = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        target = {
            "boxes":    boxes,
            "labels":   labels,
            "masks":    masks,
            "image_id": torch.tensor([idx]),
            "area":     area,
            "iscrowd":  torch.zeros(len(boxes), dtype=torch.int64),
        }

        return TF.to_tensor(img), target

    def __len__(self):
        return len(self.imgs)

def collate_fn(batch):
    return tuple(zip(*batch))

# Quick sanity check
dataset = PennFudanDataset("PennFudanPed")
img, target = dataset[0]
print(f"Dataset size : {len(dataset)}")
print(f"Image shape  : {img.shape}")
print(f"Boxes        : {target['boxes']}")


# ─── CELL 4: Split dataset ────────────────────────────────────────────────────

torch.manual_seed(42)
n       = len(dataset)
n_train = int(0.70 * n)
n_val   = int(0.15 * n)
n_test  = n - n_train - n_val

train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test])

# num_workers=0 avoids Colab multiprocessing issues
train_loader = DataLoader(train_set, batch_size=2, shuffle=True,  collate_fn=collate_fn, num_workers=0)
val_loader   = DataLoader(val_set,   batch_size=2, shuffle=False, collate_fn=collate_fn, num_workers=0)
test_loader  = DataLoader(test_set,  batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0)

print(f"Train: {n_train}  |  Val: {n_val}  |  Test: {n_test}")


# ─── CELL 5: Build model ──────────────────────────────────────────────────────

import warnings
import torchvision
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def build_model(num_classes: int = 2):
    # Suppress the pretrained deprecation warning in newer torchvision
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Use weights= API if available, fall back to pretrained=True
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
model  = build_model(num_classes=2).to(device)
print(f"Model on: {device}")
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {n_params:,}")


# ─── CELL 6: Optimizer & scheduler ───────────────────────────────────────────

EPOCHS      = 12
LR          = 0.005
PATIENCE    = 5

# Lower LR for backbone (already pretrained), higher for the new head
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

# Mixed precision — works on Colab T4/A100
use_amp = device.type == "cuda"
scaler  = torch.cuda.amp.GradScaler(enabled=use_amp)

print(f"Mixed precision: {use_amp}")


# ─── CELL 7: Training helpers ─────────────────────────────────────────────────

import time

def train_one_epoch(model, optimizer, loader, device, scaler, use_amp):
    model.train()
    total_loss = 0.0
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

        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


@torch.no_grad()
def compute_val_loss(model, loader, device, use_amp):
    model.train()   # keep train mode so loss_dict is returned
    total = 0.0
    for imgs, targets in loader:
        imgs    = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=use_amp):
            loss_dict = model(imgs, targets)
        total += sum(loss_dict.values()).item()
    return total / max(len(loader), 1)


# ─── CELL 8: Training loop ────────────────────────────────────────────────────

import json, os

os.makedirs("output", exist_ok=True)

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
        torch.save(model.state_dict(), "output/fasterrcnn_best.pth")
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
with open("output/train_history.json", "w") as f:
    json.dump(history, f, indent=2)
# Save test split so evaluate cell can reuse it
torch.save({"test_indices": list(test_set.indices)}, "output/test_split.pth")


# ─── CELL 9: Plot training curves ─────────────────────────────────────────────

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))
plt.plot(history["train_loss"], label="Train Loss", marker="o")
plt.plot(history["val_loss"],   label="Val Loss",   marker="s")
plt.xlabel("Epoch"); plt.ylabel("Loss")
plt.title("Faster R-CNN — Training & Validation Loss")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig("output/loss_curve.png", dpi=120)
plt.show()
print("Saved: output/loss_curve.png")


# ─── CELL 10: Evaluation — mAP@0.5, Precision, Recall, FPS ──────────────────

from torchvision.ops import box_iou

def evaluate_model(model, loader, device, iou_thresh=0.5, score_thresh=0.5):
    model.eval()
    all_tp, all_fp, all_fn = [], [], []
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
                gt_boxes   = tgt["boxes"]
                pred_boxes = pred["boxes"].cpu()
                scores     = pred["scores"].cpu()

                keep       = scores >= score_thresh
                pred_boxes = pred_boxes[keep]
                scores     = scores[keep]

                if len(pred_boxes) == 0 and len(gt_boxes) == 0:
                    continue
                if len(pred_boxes) == 0:
                    all_fn.append(len(gt_boxes));  continue
                if len(gt_boxes)   == 0:
                    all_fp.append(len(pred_boxes)); continue

                iou        = box_iou(pred_boxes, gt_boxes)
                matched_gt = torch.zeros(len(gt_boxes), dtype=torch.bool)
                tp = fp    = 0

                for i in scores.argsort(descending=True):
                    best_gt = iou[i].argmax().item()
                    if iou[i, best_gt] >= iou_thresh and not matched_gt[best_gt]:
                        tp += 1; matched_gt[best_gt] = True
                    else:
                        fp += 1

                all_tp.append(tp)
                all_fp.append(fp)
                all_fn.append(int((~matched_gt).sum()))

    tp  = sum(all_tp);  fp = sum(all_fp);  fn = sum(all_fn)
    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    fps       = n_images / total_time if total_time > 0 else 0

    return {
        "mAP@0.5":  round(f1,        4),   # single-class: F1 = AP@0.5
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "fps":       round(fps,       2),
        "TP": tp, "FP": fp, "FN": fn,
    }

# Load best weights before evaluating
model.load_state_dict(torch.load("output/fasterrcnn_best.pth", map_location=device))

print("Evaluating on test set...")
results = evaluate_model(model, test_loader, device)

print("\n── Test Results ─────────────────────────────────")
for k, v in results.items():
    print(f"  {k:<15} {v}")
print("─────────────────────────────────────────────────")

with open("output/eval_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved: output/eval_results.json")


# ─── CELL 11: Visualise predictions ──────────────────────────────────────────

import matplotlib.patches as patches

def show_predictions(model, dataset, device, n=6, score_thresh=0.5, save_dir="output/predictions"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    for idx in range(min(n, len(dataset))):
        img_tensor, target = dataset[idx]

        with torch.no_grad():
            output = model([img_tensor.to(device)])[0]

        img_np = img_tensor.permute(1, 2, 0).numpy()

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Ground truth
        axes[0].imshow(img_np); axes[0].set_title("Ground Truth", fontsize=13)
        for box in target["boxes"]:
            x1, y1, x2, y2 = box.tolist()
            axes[0].add_patch(patches.Rectangle(
                (x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor="lime", facecolor="none"))

        # Predictions
        axes[1].imshow(img_np); axes[1].set_title("Predictions (red)", fontsize=13)
        keep = output["scores"] >= score_thresh
        for box, score in zip(output["boxes"][keep].cpu(), output["scores"][keep].cpu()):
            x1, y1, x2, y2 = box.tolist()
            axes[1].add_patch(patches.Rectangle(
                (x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor="red", facecolor="none"))
            axes[1].text(x1, max(y1-4, 0), f"{score:.2f}", color="red", fontsize=9,
                         bbox=dict(facecolor="white", alpha=0.5, pad=1, edgecolor="none"))

        for ax in axes: ax.axis("off")
        plt.tight_layout()
        path = os.path.join(save_dir, f"pred_{idx:03d}.png")
        plt.savefig(path, dpi=120)
        plt.show()   # displays inline in Colab
        plt.close()

show_predictions(model, test_set, device, n=6)
print("Prediction images saved to output/predictions/")


# ─── CELL 12: Download all outputs from Colab ─────────────────────────────────

"""
# Run this cell to zip and download your outputs

import shutil
shutil.make_archive("assignment2_output", "zip", "output")

from google.colab import files
files.download("assignment2_output.zip")
"""
