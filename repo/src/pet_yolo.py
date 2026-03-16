"""
Oxford-IIIT Pet Dataset — Subset Preparation + YOLOv8n Training
----------------------------------------------------------------
Download dataset: https://www.robots.ox.ac.uk/~vgg/data/pets/
Expected structure after extracting:
    oxford-iiit-pet/
        images/        ← .jpg images
        annotations/
            xmls/      ← Pascal VOC .xml bounding boxes

Usage:
    # 1. Prepare data
    python pet_yolo.py --mode prepare --data_root ./oxford-iiit-pet

    # 2. Train
    python pet_yolo.py --mode train --epochs 18

    # 3. Evaluate
    python pet_yolo.py --mode eval
"""

import argparse
import json
import os
import random
import shutil
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────

BREEDS = [
    "Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair",
    "Egyptian_Mau", "Maine_Coon", "Persian", "Ragdoll", "Siamese",
]
BREED2IDX = {b: i for i, b in enumerate(BREEDS)}
MAX_PER_BREED = 50       # cap images per breed → ~500 total
IMG_SIZE = 512
OUT_DIR  = "pet_yolo"


# ──────────────────────────────────────────────────────────────────────────────
# Data preparation
# ──────────────────────────────────────────────────────────────────────────────

def convert_voc_to_yolo(xml_path: str, orig_w: int, orig_h: int, class_id: int):
    """Parse Pascal VOC XML and return YOLO-format annotation lines."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    lines = []
    for obj in root.findall("object"):
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue
        xmin = int(float(bndbox.find("xmin").text))
        ymin = int(float(bndbox.find("ymin").text))
        xmax = int(float(bndbox.find("xmax").text))
        ymax = int(float(bndbox.find("ymax").text))

        # Clamp to image bounds
        xmin, xmax = max(0, xmin), min(orig_w, xmax)
        ymin, ymax = max(0, ymin), min(orig_h, ymax)

        cx = ((xmin + xmax) / 2) / orig_w
        cy = ((ymin + ymax) / 2) / orig_h
        w  = (xmax - xmin) / orig_w
        h  = (ymax - ymin) / orig_h

        if w > 0 and h > 0:
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def prepare_dataset(data_root: str):
    img_dir = os.path.join(data_root, "images")
    ann_dir = os.path.join(data_root, "annotations", "xmls")

    # Create output dirs
    for split in ["train", "val", "test"]:
        os.makedirs(f"{OUT_DIR}/images/{split}", exist_ok=True)
        os.makedirs(f"{OUT_DIR}/labels/{split}", exist_ok=True)

    # Collect samples per breed
    random.seed(42)
    all_samples = []
    skipped = 0

    for breed in BREEDS:
        imgs = [f for f in os.listdir(img_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
                and os.path.splitext(f)[0].rsplit("_", 1)[0] == breed]
        random.shuffle(imgs)
        imgs = imgs[:MAX_PER_BREED]

        for fname in imgs:
            stem     = Path(fname).stem
            xml_path = os.path.join(ann_dir, stem + ".xml")
            img_path = os.path.join(img_dir, fname)
            if not os.path.exists(xml_path):
                skipped += 1
                continue
            all_samples.append((fname, breed, img_path, xml_path))

    random.shuffle(all_samples)
    n       = len(all_samples)
    n_train = int(0.70 * n)
    n_val   = int(0.15 * n)

    splits = {
        "train": all_samples[:n_train],
        "val":   all_samples[n_train : n_train + n_val],
        "test":  all_samples[n_train + n_val :],
    }

    print(f"Total samples: {n}  (skipped {skipped} without XML)")
    for split, samples in splits.items():
        print(f"  {split}: {len(samples)}")

    # Copy + convert
    for split, samples in splits.items():
        for fname, breed, img_path, xml_path in samples:
            stem = Path(fname).stem
            # Resize image
            img = Image.open(img_path).convert("RGB")
            orig_w, orig_h = img.size
            img = img.resize((IMG_SIZE, IMG_SIZE))
            out_img = os.path.join(OUT_DIR, "images", split, stem + ".jpg")
            img.save(out_img, quality=95)

            # Convert annotation
            lines = convert_voc_to_yolo(xml_path, orig_w, orig_h, BREED2IDX[breed])
            out_lbl = os.path.join(OUT_DIR, "labels", split, stem + ".txt")
            with open(out_lbl, "w") as f:
                f.write("\n".join(lines))

    # Write YOLO data.yaml
    yaml_content = f"""path: {os.path.abspath(OUT_DIR)}
train: images/train
val:   images/val
test:  images/test

nc: {len(BREEDS)}
names: {BREEDS}
"""
    with open(os.path.join(OUT_DIR, "data.yaml"), "w") as f:
        f.write(yaml_content)

    print(f"\nDataset ready at ./{OUT_DIR}/")
    print(f"YAML written to ./{OUT_DIR}/data.yaml")


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train_yolo(epochs: int = 18, batch: int = 8, patience: int = 5):
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Install ultralytics: pip install ultralytics")

    data_yaml = os.path.join(OUT_DIR, "data.yaml")
    assert os.path.exists(data_yaml), \
        f"data.yaml not found. Run with --mode prepare first."

    model = YOLO("yolov8n.pt")   # auto-downloads nano model

    start = time.time()
    results = model.train(
        data       = data_yaml,
        epochs     = epochs,
        imgsz      = IMG_SIZE,
        batch      = batch,
        patience   = patience,
        device     = 0 if __import__("torch").cuda.is_available() else "cpu",
        project    = "runs/pet",
        name       = "yolov8n",
        pretrained = True,
        verbose    = True,
    )
    elapsed = time.time() - start
    print(f"\nTraining complete in {elapsed/60:.1f} min")

    # Save timing
    with open("runs/pet/yolov8n/training_time.json", "w") as f:
        json.dump({"training_time_sec": round(elapsed, 1)}, f)

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_yolo(weights: str = "runs/pet/yolov8n/weights/best.pt"):
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("Install ultralytics: pip install ultralytics")

    data_yaml = os.path.join(OUT_DIR, "data.yaml")
    model   = YOLO(weights)
    metrics = model.val(data=data_yaml, imgsz=IMG_SIZE, split="test")

    results = {
        "mAP@0.5":   round(float(metrics.box.map50), 4),
        "mAP@0.5:0.95": round(float(metrics.box.map),  4),
        "precision":  round(float(metrics.box.p),    4),
        "recall":     round(float(metrics.box.r),    4),
    }

    # Inference speed
    import torch, time
    model_eval = YOLO(weights)
    dummy_imgs = [os.path.join(OUT_DIR, "images", "test",
                               f) for f in os.listdir(
                               os.path.join(OUT_DIR, "images", "test"))[:20]]
    t0 = time.time()
    model_eval.predict(dummy_imgs, imgsz=IMG_SIZE, verbose=False)
    fps = len(dummy_imgs) / (time.time() - t0)
    results["fps"] = round(fps, 2)

    print("\n── YOLOv8n Results ──────────────────────────────")
    for k, v in results.items():
        print(f"  {k:<20} {v}")
    print("─────────────────────────────────────────────────\n")

    with open("runs/pet/yolov8n/eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Results saved to runs/pet/yolov8n/eval_results.json")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",      choices=["prepare", "train", "eval"],
                        required=True)
    parser.add_argument("--data_root", default="./oxford-iiit-pet",
                        help="Root dir of Oxford Pet dataset (for prepare mode)")
    parser.add_argument("--epochs",    type=int,   default=18)
    parser.add_argument("--batch",     type=int,   default=8)
    parser.add_argument("--patience",  type=int,   default=5)
    parser.add_argument("--weights",   default="runs/pet/yolov8n/weights/best.pt")
    args = parser.parse_args()

    if args.mode == "prepare":
        prepare_dataset(args.data_root)
    elif args.mode == "train":
        train_yolo(args.epochs, args.batch, args.patience)
    elif args.mode == "eval":
        evaluate_yolo(args.weights)
