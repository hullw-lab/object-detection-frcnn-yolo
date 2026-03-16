# Object Detection: Faster R-CNN vs YOLOv8n

**Assignment 2 — Object Detection and Recognition**  
Comparison of two detection architectures across two datasets under an 8 GB GPU constraint.

---

## Results Summary

| Dataset | Model | mAP@0.5 | Precision | Recall | FPS |
|---|---|---|---|---|---|
| Penn-Fudan | Faster R-CNN | **90.0%** | 84.4% | **96.4%** | 29.3 |
| Penn-Fudan | YOLOv8n | 79.3% | **94.2%** | 72.1% | 97.8 |
| Oxford Pet | Faster R-CNN | 46.0% | 62.1% | 36.5% | 35.9 |
| Oxford Pet | YOLOv8n | **63.8%** | 63.2% | 60.6% | **100.3** |

**Key finding:** Model choice is task-dependent. Faster R-CNN wins on single-class pedestrian detection (high recall). YOLOv8n wins on multi-class fine-grained breed detection and is ~3.4× faster across both tasks.

---

## Repository Structure

```
├── notebooks/
│   ├── notebook1_fasterrcnn_pennfudan.py     # Faster R-CNN on Penn-Fudan
│   ├── notebook2_yolo_oxfordpet.py           # YOLOv8n on Oxford Pet
│   ├── notebook3_yolo_pennfudan.py           # YOLOv8n on Penn-Fudan
│   └── notebook4_fasterrcnn_oxfordpet.py     # Faster R-CNN on Oxford Pet
├── src/
│   ├── dataset.py       # Penn-Fudan dataset loader (PyTorch Dataset class)
│   ├── train.py         # Faster R-CNN training script (CLI)
│   ├── evaluate.py      # Evaluation: mAP, precision, recall, FPS
│   ├── pet_yolo.py      # Oxford Pet subset prep + YOLOv8n training
│   └── compare.py       # Generate comparison table from eval_results.json
├── results/
│   └── results_summary.json    # All four experiment metrics
└── README.md
```

---

## Datasets

### Penn-Fudan Pedestrian Dataset
- ~170 images, 1 class (person), instance segmentation masks
- Masks converted to bounding boxes programmatically
- Split: 119 train / 26 val / 25 test

```bash
wget https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip
unzip PennFudanPed.zip
```

### Oxford-IIIT Pet Dataset (10-breed subset)
- 10 breeds: Abyssinian, Bengal, Birman, Bombay, British Shorthair,
  Egyptian Mau, Maine Coon, Persian, Ragdoll, Siamese
- ~50 images per breed (~500 total), Pascal VOC XML annotations
- Split: ~350 train / ~75 val / ~75 test

```bash
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
tar -xf images.tar.gz && tar -xf annotations.tar.gz
```

---

## Running on Google Colab

Each notebook in `notebooks/` is structured as a plain Python file with clearly marked cell boundaries (`# ─── CELL N ───`). To run:

1. Open [Google Colab](https://colab.research.google.com)
2. Set runtime to **GPU** (Runtime → Change runtime type → T4 GPU)
3. Create a new notebook
4. Copy each `# ─── CELL N ───` block into a separate cell
5. Run cells top to bottom

### Notebook order
| Notebook | Dataset | Model | Epochs | mAP@0.5 |
|---|---|---|---|---|
| notebook1 | Penn-Fudan | Faster R-CNN | 10 | 90.0% |
| notebook2 | Oxford Pet | YOLOv8n | 18 | 63.8% |
| notebook3 | Penn-Fudan | YOLOv8n | 7 | 79.3% |
| notebook4 | Oxford Pet | Faster R-CNN | 12 | 46.0% |

---

## Running Locally (src/)

### Train Faster R-CNN on Penn-Fudan
```bash
pip install torch torchvision

python src/train.py \
  --data_path ./PennFudanPed \
  --epochs 12 \
  --batch_size 2
```

### Evaluate
```bash
python src/evaluate.py \
  --data_path ./PennFudanPed \
  --weights ./checkpoints/best_model.pth
```

### Compare results
```bash
# After running both models, place their eval_results.json files in results/
python src/compare.py \
  --frcnn results/frcnn_eval.json \
  --yolo  results/yolo_eval.json
```

---

## Requirements

```
torch>=2.0
torchvision>=0.15
ultralytics>=8.0
Pillow
numpy
matplotlib
```

Install:
```bash
pip install torch torchvision ultralytics Pillow numpy matplotlib
```

---

## GPU Memory Notes (8 GB constraint)

| Setting | Faster R-CNN | YOLOv8n |
|---|---|---|
| Batch size | 2 | 8 |
| Image size | 512×512 | 512×512 |
| Mixed precision | Yes (GradScaler) | Yes (built-in AMP) |
| Backbone | MobileNetV3-Large | CSPDarknet nano |

---

## Architecture Notes

**Faster R-CNN** uses a two-stage pipeline: a Region Proposal Network (RPN) first generates candidate regions, which are then classified and refined. This produces high recall (finds most objects) but slower inference (~30 FPS).

**YOLOv8n** predicts all boxes and classes in a single forward pass using a decoupled head. ~3.4× faster than Faster R-CNN, better at multi-class tasks with limited data per class.
