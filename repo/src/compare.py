"""
Generate Final Comparison Table
---------------------------------
Reads eval_results.json from both models and prints a clean
side-by-side comparison table. Also exports comparison.json.

Usage:
    python compare.py \
        --frcnn_results ./output/eval_results.json \
        --frcnn_history ./output/train_history.json \
        --yolo_results  ./runs/pet/yolov8n/eval_results.json \
        --yolo_time     ./runs/pet/yolov8n/training_time.json
"""

import argparse
import json
import os


def load_json(path):
    if not os.path.exists(path):
        print(f"  [WARNING] File not found: {path}")
        return {}
    with open(path) as f:
        return json.load(f)


def fmt(val, pct=True):
    if val == "—" or val is None:
        return "—"
    if pct and isinstance(val, float) and val <= 1.0:
        return f"{val*100:.1f}%"
    if isinstance(val, float):
        return f"{val:.2f}"
    return str(val)


def main(args):
    frcnn  = load_json(args.frcnn_results)
    hist   = load_json(args.frcnn_history)
    yolo   = load_json(args.yolo_results)
    yt     = load_json(args.yolo_time)

    frcnn_time = hist.get("training_time_sec", "—")
    yolo_time  = yt.get("training_time_sec", "—")

    frcnn_time_min = f"{frcnn_time/60:.1f} min" if isinstance(frcnn_time, (int, float)) else "—"
    yolo_time_min  = f"{yolo_time/60:.1f} min"  if isinstance(yolo_time,  (int, float)) else "—"

    rows = [
        ("Dataset",        "Penn-Fudan",                   "Oxford Pet (subset)"),
        ("Model",          "Faster R-CNN (MobileNetV3)",   "YOLOv8n"),
        ("Backbone",       "MobileNetV3-Large + FPN",      "CSPDarknet (nano)"),
        ("mAP@0.5",        fmt(frcnn.get("mAP@0.5")),      fmt(yolo.get("mAP@0.5"))),
        ("Precision",      fmt(frcnn.get("precision")),     fmt(yolo.get("precision"))),
        ("Recall",         fmt(frcnn.get("recall")),        fmt(yolo.get("recall"))),
        ("F1 Score",       fmt(frcnn.get("f1")),            "—"),
        ("Training Time",  frcnn_time_min,                  yolo_time_min),
        ("Inf. Speed (fps)", str(frcnn.get("fps", "—")),   str(yolo.get("fps", "—"))),
    ]

    col_w = [22, 30, 30]

    def divider():
        return "+" + "+".join("-" * (w + 2) for w in col_w) + "+"

    def row_str(cells):
        return "| " + " | ".join(str(c).ljust(w) for c, w in zip(cells, col_w)) + " |"

    print("\n" + divider())
    print(row_str(["Metric", "Faster R-CNN", "YOLOv8n"]))
    print(divider())
    for r in rows:
        print(row_str(r))
    print(divider())
    print()

    # Save comparison JSON
    comparison = {
        "faster_rcnn": {
            "dataset":       "Penn-Fudan Pedestrian",
            "mAP@0.5":       frcnn.get("mAP@0.5"),
            "precision":     frcnn.get("precision"),
            "recall":        frcnn.get("recall"),
            "f1":            frcnn.get("f1"),
            "training_time": frcnn_time_min,
            "fps":           frcnn.get("fps"),
        },
        "yolov8n": {
            "dataset":       "Oxford Pet (10 breeds)",
            "mAP@0.5":       yolo.get("mAP@0.5"),
            "precision":     yolo.get("precision"),
            "recall":        yolo.get("recall"),
            "training_time": yolo_time_min,
            "fps":           yolo.get("fps"),
        },
    }

    out_path = "comparison.json"
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"Comparison saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frcnn_results", default="./output/eval_results.json")
    parser.add_argument("--frcnn_history", default="./output/train_history.json")
    parser.add_argument("--yolo_results",  default="./runs/pet/yolov8n/eval_results.json")
    parser.add_argument("--yolo_time",     default="./runs/pet/yolov8n/training_time.json")
    args = parser.parse_args()
    main(args)
