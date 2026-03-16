"""
Train Faster R-CNN (MobileNetV3 backbone) on Penn-Fudan Pedestrian Dataset
---------------------------------------------------------------------------
Usage:
    python train.py --data_path ./PennFudanPed --epochs 12 --batch_size 2

Requirements:
    pip install torch torchvision
"""

import argparse
import time
import os
import json
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, random_split

from dataset import PennFudanDataset, collate_fn


# ──────────────────────────────────────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────────────────────────────────────

def build_model(num_classes: int = 2):
    """
    Load pretrained Faster R-CNN (MobileNetV3-Large + FPN backbone).
    Replace the classification head for our number of classes.
    num_classes = 1 (person) + 1 (background) = 2
    """
    model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# ──────────────────────────────────────────────────────────────────────────────
# Training helpers
# ──────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, optimizer, loader, device, scaler=None):
    model.train()
    total_loss = 0.0
    for imgs, targets in loader:
        imgs    = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        if scaler:
            with torch.cuda.amp.autocast():
                loss_dict = model(imgs, targets)
                loss = sum(loss_dict.values())
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(imgs, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def val_loss(model, loader, device):
    """Compute validation loss (model stays in train mode for loss computation)."""
    model.train()
    total = 0.0
    for imgs, targets in loader:
        imgs    = [img.to(device) for img in imgs]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(imgs, targets)
        total += sum(loss_dict.values()).item()
    return total / len(loader)


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    dataset = PennFudanDataset(args.data_path)
    n       = len(dataset)
    n_train = int(0.70 * n)
    n_val   = int(0.15 * n)
    n_test  = n - n_train - n_val

    torch.manual_seed(42)
    train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test])
    print(f"Split → train: {n_train}, val: {n_val}, test: {n_test}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True,  collate_fn=collate_fn, num_workers=2)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn, num_workers=2)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn, num_workers=2)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(num_classes=2).to(device)

    # Fine-tune: lower LR for backbone, higher for head
    params = [
        {"params": [p for n, p in model.named_parameters()
                    if "backbone" not in n and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters()
                    if "backbone" in n and p.requires_grad], "lr": args.lr / 10},
    ]
    optimizer = torch.optim.SGD(params, lr=args.lr,
                                momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                 step_size=max(1, args.epochs // 3),
                                                 gamma=0.1)

    # Mixed precision (optional, speeds up on modern GPUs)
    scaler = torch.cuda.amp.GradScaler() if (device.type == "cuda" and args.amp) else None

    # ── Training loop ─────────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    history       = {"train_loss": [], "val_loss": []}
    best_val      = float("inf")
    patience_ctr  = 0
    training_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t_loss = train_one_epoch(model, optimizer, train_loader, device, scaler)
        v_loss = val_loss(model, val_loader, device)
        scheduler.step()

        history["train_loss"].append(round(t_loss, 5))
        history["val_loss"].append(round(v_loss, 5))

        print(f"Epoch [{epoch:02d}/{args.epochs}] "
              f"train_loss={t_loss:.4f}  val_loss={v_loss:.4f}")

        # Save best model
        if v_loss < best_val:
            best_val     = v_loss
            patience_ctr = 0
            torch.save(model.state_dict(),
                       os.path.join(args.out_dir, "fasterrcnn_best.pth"))
            print(f"  ✓ Best model saved (val_loss={best_val:.4f})")
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"  Early stopping triggered at epoch {epoch}")
                break

    total_time = time.time() - training_start
    print(f"\nTraining complete in {total_time/60:.1f} min")

    # Save history
    history["training_time_sec"] = round(total_time, 1)
    with open(os.path.join(args.out_dir, "train_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Save test_loader for evaluation
    torch.save({"test_indices": test_set.indices},
               os.path.join(args.out_dir, "test_split.pth"))

    return model, test_loader, device


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",  default="./PennFudanPed")
    parser.add_argument("--out_dir",    default="./output")
    parser.add_argument("--epochs",     type=int,   default=12)
    parser.add_argument("--batch_size", type=int,   default=2)
    parser.add_argument("--lr",         type=float, default=0.005)
    parser.add_argument("--patience",   type=int,   default=5,
                        help="Early stopping patience (epochs)")
    parser.add_argument("--amp",        action="store_true",
                        help="Enable mixed precision training (GPU only)")
    args = parser.parse_args()
    main(args)
