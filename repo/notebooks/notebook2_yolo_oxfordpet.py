# =============================================================================
# NOTEBOOK 2 — YOLOv8n on Oxford-IIIT Pet Dataset (10-breed subset)
# Copy each section into a separate Colab cell and run top to bottom.
# Runtime: GPU (Runtime > Change runtime type > T4 GPU)
# =============================================================================


# ─── CELL 1: Install dependencies ────────────────────────────────────────────

"""
!pip install -q ultralytics
"""

import torch
from ultralytics import YOLO
print(f"PyTorch    : {torch.__version__}")
print(f"CUDA       : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU        : {torch.cuda.get_device_name(0)}")


# ─── CELL 2: Download Oxford-IIIT Pet Dataset ─────────────────────────────────

"""
import os

# Download images + annotations (about 800 MB total)
!wget -q --show-progress https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
!wget -q --show-progress https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz

# Extract
!tar -xzf images.tar.gz
!tar -xzf annotations.tar.gz

# Move into expected layout
os.makedirs("oxford-iiit-pet/annotations", exist_ok=True)
!mv images oxford-iiit-pet/images
!mv annotations/xmls oxford-iiit-pet/annotations/xmls

# Verify
imgs = os.listdir("oxford-iiit-pet/images")
xmls = os.listdir("oxford-iiit-pet/annotations/xmls")
print(f"Images: {len(imgs)},  XMLs: {len(xmls)}")
"""


# ─── CELL 3: Prepare 10-breed YOLO-format subset ─────────────────────────────

import os
import json
import random
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image

# ── Config ──────────────────────────────────────────────────────
DATA_ROOT     = "oxford-iiit-pet"
OUT_DIR       = "pet_yolo"
IMG_SIZE      = 512
MAX_PER_BREED = 50   # ~500 images total across 10 breeds

BREEDS = [
    "Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair",
    "Egyptian_Mau", "Maine_Coon", "Persian", "Ragdoll", "Siamese",
]
BREED2IDX = {b: i for i, b in enumerate(BREEDS)}
# ────────────────────────────────────────────────────────────────

def convert_voc_to_yolo(xml_path, orig_w, orig_h, class_id):
    """Convert a Pascal VOC XML bbox to YOLO format lines."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    lines = []
    for obj in root.findall("object"):
        bb = obj.find("bndbox")
        if bb is None:
            continue
        xmin = int(float(bb.find("xmin").text))
        ymin = int(float(bb.find("ymin").text))
        xmax = int(float(bb.find("xmax").text))
        ymax = int(float(bb.find("ymax").text))
        # Clamp to image bounds
        xmin = max(0, min(xmin, orig_w))
        xmax = max(0, min(xmax, orig_w))
        ymin = max(0, min(ymin, orig_h))
        ymax = max(0, min(ymax, orig_h))
        w = xmax - xmin; h = ymax - ymin
        if w <= 0 or h <= 0:
            continue
        cx = ((xmin + xmax) / 2) / orig_w
        cy = ((ymin + ymax) / 2) / orig_h
        lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w/orig_w:.6f} {h/orig_h:.6f}")
    return lines


def prepare_pet_dataset():
    img_dir = os.path.join(DATA_ROOT, "images")
    ann_dir = os.path.join(DATA_ROOT, "annotations", "xmls")

    for split in ["train", "val", "test"]:
        os.makedirs(f"{OUT_DIR}/images/{split}", exist_ok=True)
        os.makedirs(f"{OUT_DIR}/labels/{split}", exist_ok=True)

    random.seed(42)
    all_samples, skipped = [], 0

    for breed in BREEDS:
        # Oxford filenames start with breed name, e.g. "Abyssinian_1.jpg"
        imgs = [f for f in os.listdir(img_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
                and Path(f).stem.rsplit("_", 1)[0] == breed]
        random.shuffle(imgs)
        imgs = imgs[:MAX_PER_BREED]

        for fname in imgs:
            stem     = Path(fname).stem
            xml_path = os.path.join(ann_dir, stem + ".xml")
            if not os.path.exists(xml_path):
                skipped += 1; continue
            all_samples.append((fname, breed,
                                 os.path.join(img_dir, fname), xml_path))

    random.shuffle(all_samples)
    n       = len(all_samples)
    n_train = int(0.70 * n)
    n_val   = int(0.15 * n)

    splits = {
        "train": all_samples[:n_train],
        "val":   all_samples[n_train:n_train + n_val],
        "test":  all_samples[n_train + n_val:],
    }

    print(f"Total samples : {n}  (skipped {skipped} — no XML)")
    for s, samp in splits.items():
        print(f"  {s:<6}: {len(samp)}")

    for split, samples in splits.items():
        for fname, breed, img_path, xml_path in samples:
            stem  = Path(fname).stem
            img   = Image.open(img_path).convert("RGB")
            orig_w, orig_h = img.size
            img   = img.resize((IMG_SIZE, IMG_SIZE))
            img.save(os.path.join(OUT_DIR, "images", split, stem + ".jpg"), quality=95)

            lines = convert_voc_to_yolo(xml_path, orig_w, orig_h, BREED2IDX[breed])
            with open(os.path.join(OUT_DIR, "labels", split, stem + ".txt"), "w") as f:
                f.write("\n".join(lines))

    # Write data.yaml with absolute path (required by Ultralytics in Colab)
    abs_path = os.path.abspath(OUT_DIR)
    yaml = (
        f"path: {abs_path}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"test:  images/test\n\n"
        f"nc: {len(BREEDS)}\n"
        f"names: {BREEDS}\n"
    )
    with open(os.path.join(OUT_DIR, "data.yaml"), "w") as f:
        f.write(yaml)

    print(f"\nDataset ready → {abs_path}")
    print(f"data.yaml    → {abs_path}/data.yaml")


prepare_pet_dataset()


# ─── CELL 4: Train YOLOv8n ────────────────────────────────────────────────────

import os, time, json, glob
from ultralytics import YOLO

DATA_YAML = os.path.join(os.path.abspath(OUT_DIR), "data.yaml")

model = YOLO("yolov8n.pt")   # downloads ~6 MB nano model automatically

t_start = time.time()

train_results = model.train(
    data      = DATA_YAML,
    epochs    = 18,
    imgsz     = IMG_SIZE,
    batch     = 8,            # safe for T4 8 GB
    patience  = 5,            # early stopping
    device    = 0 if torch.cuda.is_available() else "cpu",
    project   = "runs/pet",
    name      = "yolov8n",
    exist_ok  = True,         # prevents duplicate run folders in Colab
    pretrained= True,
    verbose   = True,
)

elapsed = time.time() - t_start
print(f"\nTraining complete in {elapsed/60:.1f} min")

# ── Auto-detect the actual weights path ──────────────────────────────────────
# Ultralytics may create runs/pet/yolov8n, runs/pet/yolov8n2, etc.
# The safest way is to read the save_dir directly from the results object,
# or fall back to searching for the most recently modified best.pt.

def find_best_weights():
    # Method 1: read from train results object (Ultralytics >= 8.0.20)
    try:
        save_dir = str(train_results.save_dir)
        candidate = os.path.join(save_dir, "weights", "best.pt")
        if os.path.exists(candidate):
            return candidate
    except Exception:
        pass

    # Method 2: glob for all best.pt files and pick the newest
    candidates = glob.glob("runs/pet/**/weights/best.pt", recursive=True)
    if candidates:
        return max(candidates, key=os.path.getmtime)

    raise FileNotFoundError(
        "Could not find best.pt. Check the runs/pet/ folder manually and set WEIGHTS below.")

WEIGHTS = find_best_weights()
print(f"\n✓ Best weights found at: {WEIGHTS}")

# Save timing alongside the weights
run_dir = os.path.dirname(os.path.dirname(WEIGHTS))   # e.g. runs/pet/yolov8n
with open(os.path.join(run_dir, "training_time.json"), "w") as f:
    json.dump({"training_time_sec": round(elapsed, 1)}, f)


# ─── CELL 5: Evaluate YOLOv8n ─────────────────────────────────────────────────

import os, json, time, glob
from ultralytics import YOLO

DATA_YAML = os.path.join(os.path.abspath(OUT_DIR), "data.yaml")

# WEIGHTS is set automatically by Cell 4.
# If you are re-running this cell in a new session, set it manually:
#   WEIGHTS = "runs/pet/yolov8n/weights/best.pt"   ← adjust folder name if needed
# Or let this block find it automatically:
if "WEIGHTS" not in dir() or not os.path.exists(WEIGHTS):
    candidates = glob.glob("runs/pet/**/weights/best.pt", recursive=True)
    if not candidates:
        raise FileNotFoundError(
            "No best.pt found. Make sure Cell 4 (training) completed successfully.\n"
            "Check what folders exist with:  !find runs/ -name best.pt")
    WEIGHTS = max(candidates, key=os.path.getmtime)
    print(f"Auto-detected weights: {WEIGHTS}")
else:
    print(f"Using weights: {WEIGHTS}")

run_dir = os.path.dirname(os.path.dirname(WEIGHTS))   # e.g. runs/pet/yolov8n

model   = YOLO(WEIGHTS)
metrics = model.val(data=DATA_YAML, imgsz=IMG_SIZE, split="test", verbose=False)

import numpy as np

def to_scalar(v):
    """Handle both scalar and per-class array — return mean as float."""
    arr = np.array(v).flatten()
    return round(float(arr.mean()), 4)

results = {
    "mAP@0.5":      to_scalar(metrics.box.map50),
    "mAP@0.5:0.95": to_scalar(metrics.box.map),
    "precision":    to_scalar(metrics.box.p),
    "recall":       to_scalar(metrics.box.r),
}

# Measure inference FPS on a small batch of test images
test_imgs = [
    os.path.join(OUT_DIR, "images", "test", f)
    for f in os.listdir(os.path.join(OUT_DIR, "images", "test"))[:20]
]
t0  = time.time()
model.predict(test_imgs, imgsz=IMG_SIZE, verbose=False)
fps = len(test_imgs) / (time.time() - t0)
results["fps"] = round(fps, 2)

print("\n── YOLOv8n Test Results ──────────────────────────")
for k, v in results.items():
    print(f"  {k:<20} {v}")
print("──────────────────────────────────────────────────")

with open(os.path.join(run_dir, "eval_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {run_dir}/eval_results.json")


# ─── CELL 6: Visualise YOLOv8n predictions ───────────────────────────────────

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os, glob
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

# Reuse WEIGHTS from Cell 5, or re-detect if needed
if "WEIGHTS" not in dir() or not os.path.exists(WEIGHTS):
    candidates = glob.glob("runs/pet/**/weights/best.pt", recursive=True)
    WEIGHTS = max(candidates, key=os.path.getmtime)
    print(f"Auto-detected weights: {WEIGHTS}")

run_dir   = os.path.dirname(os.path.dirname(WEIGHTS))
model_viz = YOLO(WEIGHTS)

test_img_dir = os.path.join(OUT_DIR, "images", "test")
test_imgs    = sorted(os.listdir(test_img_dir))[:6]

os.makedirs(os.path.join(run_dir, "predictions"), exist_ok=True)

for fname in test_imgs:
    img_path = os.path.join(test_img_dir, fname)
    preds    = model_viz.predict(img_path, imgsz=IMG_SIZE, verbose=False)[0]

    img = Image.open(img_path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Ground truth
    lbl_path = os.path.join(OUT_DIR, "labels", "test", Path(fname).stem + ".txt")
    axes[0].imshow(img); axes[0].set_title("Ground Truth")
    if os.path.exists(lbl_path):
        w, h = img.size
        with open(lbl_path) as f:
            for line in f:
                cls, cx, cy, bw, bh = map(float, line.split())
                x1 = (cx - bw/2) * w; y1 = (cy - bh/2) * h
                axes[0].add_patch(patches.Rectangle(
                    (x1, y1), bw*w, bh*h,
                    linewidth=2, edgecolor="lime", facecolor="none"))
                axes[0].text(x1, y1-4, BREEDS[int(cls)], color="lime", fontsize=8,
                             bbox=dict(facecolor="black", alpha=0.4, pad=1, edgecolor="none"))

    # Predictions
    axes[1].imshow(img); axes[1].set_title("Predictions (red)")
    if preds.boxes is not None and len(preds.boxes):
        for box in preds.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
            score = float(box.conf[0])
            cls   = int(box.cls[0])
            axes[1].add_patch(patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=2, edgecolor="red", facecolor="none"))
            axes[1].text(x1, max(y1-4, 0), f"{BREEDS[cls]} {score:.2f}",
                         color="red", fontsize=8,
                         bbox=dict(facecolor="white", alpha=0.5, pad=1, edgecolor="none"))

    for ax in axes: ax.axis("off")
    plt.tight_layout()
    save_path = os.path.join(run_dir, "predictions", Path(fname).stem + ".png")
    plt.savefig(save_path, dpi=120)
    plt.show()
    plt.close()

print("Done. Prediction images saved to runs/pet/yolov8n/predictions/")


# ─── CELL 7: Print final comparison table ─────────────────────────────────────

"""
NOTE: Run this cell after completing BOTH notebooks.
Paste your Faster R-CNN results manually if running in separate sessions.
"""

import json, os, glob

def load(path):
    return json.load(open(path)) if os.path.exists(path) else {}

frcnn = load("output/eval_results.json")
hist  = load("output/train_history.json")

# Auto-find YOLO results from whichever run folder was created
yolo_result_candidates = glob.glob("runs/pet/**/eval_results.json", recursive=True)
yolo_time_candidates   = glob.glob("runs/pet/**/training_time.json", recursive=True)
yolo = load(max(yolo_result_candidates, key=os.path.getmtime)) if yolo_result_candidates else {}
yt   = load(max(yolo_time_candidates,   key=os.path.getmtime)) if yolo_time_candidates   else {}

def pct(v):
    return f"{float(v)*100:.1f}%" if v and isinstance(v, (int, float)) and v <= 1 else (str(v) if v else "—")

def mins(s):
    return f"{float(s)/60:.1f} min" if s else "—"

rows = [
    ("Dataset",          "Penn-Fudan",                      "Oxford Pet (10 breeds)"),
    ("Model",            "Faster R-CNN (MobileNetV3+FPN)",  "YOLOv8n"),
    ("mAP@0.5",          pct(frcnn.get("mAP@0.5")),         pct(yolo.get("mAP@0.5"))),
    ("Precision",        pct(frcnn.get("precision")),        pct(yolo.get("precision"))),
    ("Recall",           pct(frcnn.get("recall")),           pct(yolo.get("recall"))),
    ("F1 / mAP",         pct(frcnn.get("f1")),               pct(yolo.get("mAP@0.5:0.95"))),
    ("Training Time",    mins(hist.get("training_time_sec")),mins(yt.get("training_time_sec"))),
    ("Inference (FPS)",  str(frcnn.get("fps", "—")),         str(yolo.get("fps", "—"))),
]

w = [22, 34, 26]
div = "+" + "+".join("-"*(x+2) for x in w) + "+"
def row_str(cells):
    return "| " + " | ".join(str(c).ljust(w[i]) for i, c in enumerate(cells)) + " |"

print(div)
print(row_str(["Metric", "Faster R-CNN", "YOLOv8n"]))
print(div)
for r in rows:
    print(row_str(r))
print(div)


# ─── CELL 8: Download outputs from Colab ──────────────────────────────────────

"""
import shutil
from google.colab import files

shutil.make_archive("assignment2_yolo_output", "zip", "runs/pet/yolov8n")
files.download("assignment2_yolo_output.zip")
"""
