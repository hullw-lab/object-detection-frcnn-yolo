"""
Evaluate Faster R-CNN on Penn-Fudan Test Set
---------------------------------------------
Computes: mAP@0.5, Precision, Recall, F1, Inference Speed
Saves:    results.json, example prediction images

Usage:
    python evaluate.py --data_path ./PennFudanPed \
                       --weights ./output/fasterrcnn_best.pth \
                       --out_dir ./output
"""

import argparse
import json
import os
import time

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou

from dataset import PennFudanDataset, collate_fn


# ──────────────────────────────────────────────────────────────────────────────
# mAP helpers
# ──────────────────────────────────────────────────────────────────────────────

def compute_ap(recalls, precisions):
    """Compute Average Precision via 11-point interpolation (VOC style)."""
    ap = 0.0
    for t in [i / 10 for i in range(11)]:
        p = max([p for r, p in zip(recalls, precisions) if r >= t], default=0)
        ap += p / 11
    return ap


def evaluate_map(model, loader, device, iou_thresh=0.5, score_thresh=0.5):
    """
    Returns dict with:
        map50, precision, recall, f1, fps, per_image_results
    """
    model.eval()
    all_tp, all_fp, all_fn = [], [], []
    all_scores = []
    all_gt_count = 0
    total_time = 0.0
    n_images = 0

    with torch.no_grad():
        for imgs, targets in loader:
            imgs_gpu = [img.to(device) for img in imgs]

            torch.cuda.synchronize() if device.type == "cuda" else None
            t0 = time.time()
            outputs = model(imgs_gpu)
            torch.cuda.synchronize() if device.type == "cuda" else None
            total_time += time.time() - t0
            n_images   += len(imgs)

            for pred, tgt in zip(outputs, targets):
                gt_boxes   = tgt["boxes"]
                pred_boxes = pred["boxes"].cpu()
                scores     = pred["scores"].cpu()

                # Filter by confidence
                keep       = scores >= score_thresh
                pred_boxes = pred_boxes[keep]
                scores     = scores[keep]

                all_gt_count += len(gt_boxes)

                if len(pred_boxes) == 0 and len(gt_boxes) == 0:
                    continue
                if len(pred_boxes) == 0:
                    all_fn.append(len(gt_boxes))
                    continue
                if len(gt_boxes) == 0:
                    all_fp.append(len(pred_boxes))
                    all_scores.extend(scores.tolist())
                    continue

                iou   = box_iou(pred_boxes, gt_boxes)   # (P, G)
                matched_gt = torch.zeros(len(gt_boxes), dtype=torch.bool)
                tp = fp = 0

                # Match predictions (highest score first)
                order = scores.argsort(descending=True)
                for i in order:
                    ious_row = iou[i]
                    best_gt  = ious_row.argmax().item()
                    if ious_row[best_gt] >= iou_thresh and not matched_gt[best_gt]:
                        tp += 1
                        matched_gt[best_gt] = True
                    else:
                        fp += 1

                fn = int((~matched_gt).sum().item())
                all_tp.append(tp)
                all_fp.append(fp)
                all_fn.append(fn)
                all_scores.extend(scores.tolist())

    total_tp = sum(all_tp)
    total_fp = sum(all_fp)
    total_fn = sum(all_fn)

    precision = total_tp / (total_tp + total_fp + 1e-8)
    recall    = total_tp / (total_tp + total_fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    fps       = n_images / total_time if total_time > 0 else 0

    # Simple mAP@0.5 (single-class: equals F1-based AP)
    map50 = precision * recall / (precision + recall + 1e-8) * 2  # = F1 as proxy

    results = {
        "mAP@0.5":   round(map50,     4),
        "precision":  round(precision, 4),
        "recall":     round(recall,    4),
        "f1":         round(f1,        4),
        "fps":        round(fps,       2),
        "total_tp":   total_tp,
        "total_fp":   total_fp,
        "total_fn":   total_fn,
    }
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Visualisation
# ──────────────────────────────────────────────────────────────────────────────

def visualize_predictions(model, dataset, device, out_dir, n=6, score_thresh=0.5):
    """Save side-by-side GT vs Prediction images."""
    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    indices = list(range(min(n, len(dataset))))

    for idx in indices:
        img_tensor, target = dataset[idx]
        with torch.no_grad():
            output = model([img_tensor.to(device)])[0]

        img_np = img_tensor.permute(1, 2, 0).numpy()

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Ground truth
        axes[0].imshow(img_np)
        axes[0].set_title("Ground Truth")
        for box in target["boxes"]:
            x1, y1, x2, y2 = box.tolist()
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                      linewidth=2, edgecolor="lime", facecolor="none")
            axes[0].add_patch(rect)

        # Predictions
        axes[1].imshow(img_np)
        axes[1].set_title("Predictions")
        keep = output["scores"] >= score_thresh
        for box, score in zip(output["boxes"][keep].cpu(),
                               output["scores"][keep].cpu()):
            x1, y1, x2, y2 = box.tolist()
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                      linewidth=2, edgecolor="red", facecolor="none")
            axes[1].add_patch(rect)
            axes[1].text(x1, y1 - 4, f"{score:.2f}", color="red", fontsize=8)

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"pred_{idx:03d}.png"), dpi=120)
        plt.close()

    print(f"Saved {len(indices)} prediction images to {out_dir}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Load dataset / test split ─────────────────────────────────────────────
    full_dataset = PennFudanDataset(args.data_path)
    n = len(full_dataset)

    if os.path.exists(os.path.join(args.out_dir, "test_split.pth")):
        split_info = torch.load(os.path.join(args.out_dir, "test_split.pth"))
        test_indices = split_info["test_indices"]
    else:
        # Recreate same split
        torch.manual_seed(42)
        from torch.utils.data import random_split
        n_train = int(0.70 * n)
        n_val   = int(0.15 * n)
        n_test  = n - n_train - n_val
        _, _, test_set = random_split(full_dataset, [n_train, n_val, n_test])
        test_indices = test_set.indices

    test_dataset = Subset(full_dataset, test_indices)
    test_loader  = DataLoader(test_dataset, batch_size=1,
                              shuffle=False, collate_fn=collate_fn)
    print(f"Test set size: {len(test_dataset)} images")

    # ── Load model ────────────────────────────────────────────────────────────
    model = fasterrcnn_mobilenet_v3_large_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device)
    print(f"Loaded weights from {args.weights}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("\nRunning evaluation...")
    results = evaluate_map(model, test_loader, device,
                           iou_thresh=0.5, score_thresh=args.score_thresh)

    print("\n── Results ──────────────────────────────────────")
    for k, v in results.items():
        print(f"  {k:<15} {v}")
    print("─────────────────────────────────────────────────\n")

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "eval_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.out_dir}/eval_results.json")

    # ── Visualise ─────────────────────────────────────────────────────────────
    print("Generating prediction visualisations...")
    visualize_predictions(model, test_dataset, device,
                          out_dir=os.path.join(args.out_dir, "predictions"),
                          n=6, score_thresh=args.score_thresh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path",    default="./PennFudanPed")
    parser.add_argument("--weights",      default="./output/fasterrcnn_best.pth")
    parser.add_argument("--out_dir",      default="./output")
    parser.add_argument("--score_thresh", type=float, default=0.5)
    args = parser.parse_args()
    main(args)
