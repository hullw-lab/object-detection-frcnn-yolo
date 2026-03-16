
# NOTEBOOK 3 — YOLOv8n on Penn-Fudan Pedestrian Dataset
# Runtime: GPU 



import torch
from ultralytics import YOLO
print(f"PyTorch : {torch.__version__}")
print(f"CUDA    : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU     : {torch.cuda.get_device_name(0)}")


#  Download Penn-Fudan dataset

"""
!wget -q --show-progress https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip -O PennFudanPed.zip
!unzip -q PennFudanPed.zip

import os
imgs  = os.listdir("PennFudanPed/PNGImages")
masks = os.listdir("PennFudanPed/PedMasks")
print(f"Images: {len(imgs)},  Masks: {len(masks)}")
"""


# Convert Penn-Fudan masks to YOLO bbox format 
# Penn-Fudan has segmentation masks, not XML annotations.
# We derive tight bounding boxes from each mask instance and write YOLO .txt files.

import os
import random
import numpy as np
from PIL import Image
from pathlib import Path

DATA_ROOT  = "PennFudanPed"
OUT_DIR    = "pennfudan_yolo"
IMG_SIZE   = 512
CLASS_ID   = 0          # single class: person

def mask_to_yolo_boxes(mask_path, img_w, img_h):
    """Read a Penn-Fudan mask and return YOLO-format lines (one per person instance)."""
    mask    = np.array(Image.open(mask_path))
    obj_ids = np.unique(mask)
    obj_ids = obj_ids[obj_ids != 0]     # 0 = background

    lines = []
    for obj_id in obj_ids:
        ys, xs = np.where(mask == obj_id)
        if len(xs) == 0:
            continue
        xmin, xmax = int(xs.min()), int(xs.max())
        ymin, ymax = int(ys.min()), int(ys.max())
        w = xmax - xmin
        h = ymax - ymin
        if w <= 0 or h <= 0:
            continue
        cx = ((xmin + xmax) / 2) / img_w
        cy = ((ymin + ymax) / 2) / img_h
        lines.append(f"{CLASS_ID} {cx:.6f} {cy:.6f} {w/img_w:.6f} {h/img_h:.6f}")
    return lines


def prepare_pennfudan_yolo():
    img_dir  = os.path.join(DATA_ROOT, "PNGImages")
    mask_dir = os.path.join(DATA_ROOT, "PedMasks")

    for split in ["train", "val", "test"]:
        os.makedirs(f"{OUT_DIR}/images/{split}", exist_ok=True)
        os.makedirs(f"{OUT_DIR}/labels/{split}", exist_ok=True)

    # Pair images with masks (sorted so they match)
    img_files  = sorted([f for f in os.listdir(img_dir)  if not f.startswith(".")])
    mask_files = sorted([f for f in os.listdir(mask_dir) if not f.startswith(".")])
    assert len(img_files) == len(mask_files), "Image/mask count mismatch!"

    pairs = list(zip(img_files, mask_files))
    random.seed(42)
    random.shuffle(pairs)

    n       = len(pairs)
    n_train = int(0.70 * n)
    n_val   = int(0.15 * n)

    splits = {
        "train": pairs[:n_train],
        "val":   pairs[n_train : n_train + n_val],
        "test":  pairs[n_train + n_val :],
    }

    print(f"Total: {n} images")
    for s, p in splits.items():
        print(f"  {s:<6}: {len(p)}")

    skipped = 0
    for split, pairs_split in splits.items():
        for img_fname, mask_fname in pairs_split:
            stem      = Path(img_fname).stem
            img_path  = os.path.join(img_dir,  img_fname)
            mask_path = os.path.join(mask_dir, mask_fname)

            # Open original image to get dimensions for annotation scaling
            img_orig = Image.open(img_path).convert("RGB")
            orig_w, orig_h = img_orig.size

            # Extract YOLO boxes from mask (using original dimensions)
            lines = mask_to_yolo_boxes(mask_path, orig_w, orig_h)
            if not lines:
                skipped += 1
                continue

            # Resize image to 512×512 and save
            img_orig.resize((IMG_SIZE, IMG_SIZE)).save(
                os.path.join(OUT_DIR, "images", split, stem + ".jpg"), quality=95)

            # Save YOLO label
            with open(os.path.join(OUT_DIR, "labels", split, stem + ".txt"), "w") as f:
                f.write("\n".join(lines))

    # Write data.yaml with absolute path (required in Colab)
    abs_path = os.path.abspath(OUT_DIR)
    yaml = (
        f"path: {abs_path}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"test:  images/test\n\n"
        f"nc: 1\n"
        f"names: ['person']\n"
    )
    with open(os.path.join(OUT_DIR, "data.yaml"), "w") as f:
        f.write(yaml)

    print(f"\nSkipped {skipped} images with no valid boxes")
    print(f"Dataset ready  → {abs_path}")
    print(f"data.yaml      → {abs_path}/data.yaml")


prepare_pennfudan_yolo()


#  Verify a few converted labels 

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

def show_yolo_label(img_path, lbl_path, title=""):
    img = Image.open(img_path)
    w, h = img.size
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(img)
    ax.set_title(title or Path(img_path).name)
    if os.path.exists(lbl_path):
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5: continue
                _, cx, cy, bw, bh = map(float, parts)
                x1 = (cx - bw/2) * w; y1 = (cy - bh/2) * h
                ax.add_patch(mpatches.Rectangle(
                    (x1, y1), bw*w, bh*h,
                    linewidth=2, edgecolor="lime", facecolor="none"))
    ax.axis("off"); plt.tight_layout(); plt.show()

# Show first 3 training images with their boxes
train_imgs = sorted(os.listdir(f"{OUT_DIR}/images/train"))[:3]
for fname in train_imgs:
    img_path = f"{OUT_DIR}/images/train/{fname}"
    lbl_path = f"{OUT_DIR}/labels/train/{Path(fname).stem}.txt"
    show_yolo_label(img_path, lbl_path)


#Train YOLOv8n

import os, time, json, glob
from ultralytics import YOLO

DATA_YAML = os.path.join(os.path.abspath(OUT_DIR), "data.yaml")

model = YOLO("yolov8n.pt")   # downloads ~6 MB nano weights automatically

t_start = time.time()

train_results = model.train(
    data       = DATA_YAML,
    epochs     = 15,         # assignment spec: 10–15 for Penn-Fudan
    imgsz      = IMG_SIZE,
    batch      = 8,          # safe for T4 8 GB
    patience   = 5,
    device     = 0 if torch.cuda.is_available() else "cpu",
    project    = "runs/pennfudan",
    name       = "yolov8n",
    exist_ok   = True,
    pretrained = True,
    verbose    = True,
)

elapsed = time.time() - t_start
print(f"\nTraining complete in {elapsed/60:.1f} min")

# Auto-detect actual weights path 
def find_best_weights(project="runs/pennfudan"):
    # Method 1: read from result object (most reliable)
    try:
        candidate = os.path.join(str(train_results.save_dir), "weights", "best.pt")
        if os.path.exists(candidate):
            return candidate
    except Exception:
        pass
    # Method 2: glob fallback
    candidates = glob.glob(f"{project}/**/weights/best.pt", recursive=True)
    if candidates:
        return max(candidates, key=os.path.getmtime)
    raise FileNotFoundError(f"No best.pt found under {project}/. Run !find {project}/ -name best.pt to debug.")

WEIGHTS = find_best_weights()
print(f"✓ Weights: {WEIGHTS}")

run_dir = os.path.dirname(os.path.dirname(WEIGHTS))
with open(os.path.join(run_dir, "training_time.json"), "w") as f:
    json.dump({"training_time_sec": round(elapsed, 1)}, f)


# Evaluate

import os, json, time, glob, numpy as np
from ultralytics import YOLO

DATA_YAML = os.path.join(os.path.abspath(OUT_DIR), "data.yaml")

# Re-detect weights if running cell standalone in a new session
if "WEIGHTS" not in dir() or not os.path.exists(WEIGHTS):
    candidates = glob.glob("runs/pennfudan/**/weights/best.pt", recursive=True)
    if not candidates:
        raise FileNotFoundError("No best.pt found. Run !find runs/pennfudan -name best.pt")
    WEIGHTS = max(candidates, key=os.path.getmtime)
    print(f"Auto-detected: {WEIGHTS}")
else:
    print(f"Using weights: {WEIGHTS}")

run_dir = os.path.dirname(os.path.dirname(WEIGHTS))

model   = YOLO(WEIGHTS)
metrics = model.val(data=DATA_YAML, imgsz=IMG_SIZE, split="test", verbose=False)

def to_scalar(v):
    return round(float(np.array(v).flatten().mean()), 4)

results = {
    "mAP@0.5":      to_scalar(metrics.box.map50),
    "mAP@0.5:0.95": to_scalar(metrics.box.map),
    "precision":    to_scalar(metrics.box.p),
    "recall":       to_scalar(metrics.box.r),
}

# Inference FPS
test_imgs = [
    os.path.join(OUT_DIR, "images", "test", f)
    for f in os.listdir(os.path.join(OUT_DIR, "images", "test"))[:20]
]
t0  = time.time()
model.predict(test_imgs, imgsz=IMG_SIZE, verbose=False)
results["fps"] = round(len(test_imgs) / (time.time() - t0), 2)

print("\n── YOLOv8n Penn-Fudan Test Results ──────────────")
for k, v in results.items():
    print(f"  {k:<20} {v}")
print("─────────────────────────────────────────────────")

with open(os.path.join(run_dir, "eval_results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: {run_dir}/eval_results.json")


#  Visualise predictions 

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os, glob
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

if "WEIGHTS" not in dir() or not os.path.exists(WEIGHTS):
    candidates = glob.glob("runs/pennfudan/**/weights/best.pt", recursive=True)
    WEIGHTS = max(candidates, key=os.path.getmtime)

run_dir   = os.path.dirname(os.path.dirname(WEIGHTS))
model_viz = YOLO(WEIGHTS)
os.makedirs(os.path.join(run_dir, "predictions"), exist_ok=True)

test_imgs = sorted(os.listdir(os.path.join(OUT_DIR, "images", "test")))[:6]

for fname in test_imgs:
    img_path = os.path.join(OUT_DIR, "images", "test", fname)
    lbl_path = os.path.join(OUT_DIR, "labels", "test", Path(fname).stem + ".txt")
    preds    = model_viz.predict(img_path, imgsz=IMG_SIZE, verbose=False)[0]

    img = Image.open(img_path)
    w, h = img.size
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Ground truth
    axes[0].imshow(img); axes[0].set_title("Ground Truth")
    if os.path.exists(lbl_path):
        with open(lbl_path) as f:
            for line in f:
                _, cx, cy, bw, bh = map(float, line.split())
                x1 = (cx - bw/2)*w; y1 = (cy - bh/2)*h
                axes[0].add_patch(patches.Rectangle(
                    (x1, y1), bw*w, bh*h, linewidth=2,
                    edgecolor="lime", facecolor="none"))
                axes[0].text(x1, max(y1-4, 0), "person", color="lime", fontsize=8,
                             bbox=dict(facecolor="black", alpha=0.4, pad=1, edgecolor="none"))

    # Predictions
    axes[1].imshow(img); axes[1].set_title("Predictions (red)")
    if preds.boxes is not None and len(preds.boxes):
        for box in preds.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
            score = float(box.conf[0])
            axes[1].add_patch(patches.Rectangle(
                (x1, y1), x2-x1, y2-y1, linewidth=2,
                edgecolor="red", facecolor="none"))
            axes[1].text(x1, max(y1-4, 0), f"person {score:.2f}", color="red", fontsize=8,
                         bbox=dict(facecolor="white", alpha=0.5, pad=1, edgecolor="none"))

    for ax in axes: ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "predictions", Path(fname).stem + ".png"), dpi=120)
    plt.show()
    plt.close()

print(f"Saved to {run_dir}/predictions/")

