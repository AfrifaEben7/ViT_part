"""
infer.py — Run inference with the fine-tuned SAM2 macrophage model.

Loads sam2_macrophage_best.pt, runs on all test images using a positive
point prompt sampled from the GT mask interior, saves side-by-side
visualizations to ./inference_results/.

Usage:
    python3 infer.py [--n_images N]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for HPC
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch

from pycocotools import mask as coco_mask
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights",    default="./sam2_macrophage_best.pt")
    p.add_argument("--base_ckpt",  default="./checkpoints/sam2_hiera_tiny.pt")
    p.add_argument("--model_cfg",  default="sam2_hiera_t.yaml")
    p.add_argument("--test_dir",   default="./microphage-4/test")
    p.add_argument("--out_dir",    default="./inference_results")
    p.add_argument("--n_images",   type=int, default=5,
                   help="Number of test images to run (0 = all)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Mask decoding helper (same as training)
# ---------------------------------------------------------------------------
def decode_rle(segmentation):
    counts = segmentation["counts"]
    if isinstance(counts, str):
        counts = counts.encode()
    rle = {"counts": counts, "size": segmentation["size"]}
    return coco_mask.decode(rle).astype(np.uint8)


def load_gt_mask(json_path: Path, orig_h: int, orig_w: int) -> np.ndarray:
    with open(json_path) as f:
        data = json.load(f)
    annotations = data.get("annotations", [])
    merged = np.zeros((orig_h, orig_w), dtype=np.uint8)
    for ann in annotations:
        m = decode_rle(ann["segmentation"])
        merged = np.maximum(merged, m)
    return merged


# ---------------------------------------------------------------------------
# Point prompt: random positive from GT mask interior
# ---------------------------------------------------------------------------
def sample_point_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        h, w = mask.shape
        return np.array([[w // 2, h // 2]]), np.array([0])
    idx = np.random.randint(len(xs))
    return np.array([[xs[idx], ys[idx]]]), np.array([1])


# ---------------------------------------------------------------------------
# Dice score (for quick quantitative check)
# ---------------------------------------------------------------------------
def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt   = gt.astype(bool)
    inter = (pred & gt).sum()
    denom = pred.sum() + gt.sum()
    return (2 * inter / denom) if denom > 0 else 1.0


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def save_figure(image_rgb, gt_mask, pred_mask, point_xy, label,
                dice, iou_score, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Dice: {dice:.3f}   SAM2 IoU score: {iou_score:.3f}",
                 fontsize=13, fontweight="bold")

    # Original image + point prompt
    axes[0].imshow(image_rgb)
    marker = "g*" if label[0] == 1 else "r*"
    axes[0].plot(point_xy[0, 0], point_xy[0, 1], marker, markersize=14,
                 markeredgecolor="white", markeredgewidth=1.5)
    axes[0].set_title("Input + Point Prompt", fontsize=11)
    axes[0].axis("off")

    # Ground truth
    axes[1].imshow(image_rgb)
    gt_overlay = np.zeros((*gt_mask.shape, 4), dtype=np.float32)
    gt_overlay[gt_mask > 0] = [0.0, 1.0, 0.0, 0.45]   # green
    axes[1].imshow(gt_overlay)
    axes[1].set_title("Ground Truth", fontsize=11)
    axes[1].axis("off")

    # Predicted mask
    axes[2].imshow(image_rgb)
    pred_overlay = np.zeros((*pred_mask.shape, 4), dtype=np.float32)
    pred_overlay[pred_mask > 0] = [1.0, 0.4, 0.0, 0.45]  # orange
    axes[2].imshow(pred_overlay)
    tp_mask = (pred_mask > 0) & (gt_mask > 0)
    fp_mask = (pred_mask > 0) & (gt_mask == 0)
    fn_mask = (pred_mask == 0) & (gt_mask > 0)
    tp_ov = np.zeros((*pred_mask.shape, 4), dtype=np.float32)
    fp_ov = np.zeros((*pred_mask.shape, 4), dtype=np.float32)
    fn_ov = np.zeros((*pred_mask.shape, 4), dtype=np.float32)
    tp_ov[tp_mask] = [0.0, 1.0, 0.0, 0.5]   # green  = TP
    fp_ov[fp_mask] = [1.0, 0.0, 0.0, 0.5]   # red    = FP
    fn_ov[fn_mask] = [0.0, 0.0, 1.0, 0.5]   # blue   = FN
    axes[2].imshow(tp_ov)
    axes[2].imshow(fp_ov)
    axes[2].imshow(fn_ov)
    legend = [
        mpatches.Patch(color="green",  label="TP"),
        mpatches.Patch(color="red",    label="FP"),
        mpatches.Patch(color="blue",   label="FN"),
    ]
    axes[2].legend(handles=legend, loc="upper right", fontsize=9)
    axes[2].set_title("Prediction (TP/FP/FN)", fontsize=11)
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    np.random.seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")

    # ---- Load model -------------------------------------------------------
    print(f"[INFO] Loading base model from {args.base_ckpt}")
    model = build_sam2(args.model_cfg, args.base_ckpt, device=device)

    print(f"[INFO] Loading fine-tuned weights from {args.weights}")
    state_dict = torch.load(args.weights, map_location=device)
    # Handle both plain state_dict and checkpoint dict
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    model.load_state_dict(state_dict)
    model.eval()

    predictor = SAM2ImagePredictor(model)

    # ---- Collect test images ----------------------------------------------
    test_dir = Path(args.test_dir)
    exts = {".jpg", ".jpeg", ".png"}
    images = sorted([p for p in test_dir.iterdir() if p.suffix.lower() in exts])
    if args.n_images > 0:
        images = images[:args.n_images]
    print(f"[INFO] Running inference on {len(images)} images")

    # ---- Output directory -------------------------------------------------
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Inference loop ---------------------------------------------------
    dice_scores = []

    for img_path in images:
        json_path = img_path.with_suffix(".json")
        if not json_path.exists():
            print(f"  [SKIP] No JSON for {img_path.name}")
            continue

        # Load image (keep original size for display; SAM2 handles resizing)
        image_bgr = cv2.imread(str(img_path))
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image_rgb.shape[:2]

        # Ground truth mask at original resolution
        gt_mask = load_gt_mask(json_path, orig_h, orig_w)

        # Sample positive point from GT mask
        point_coords, point_labels = sample_point_from_mask(gt_mask)

        # SAM2 inference
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            predictor.set_image(image_rgb)
            masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=False,
            )

        pred_mask = (masks[0] > 0).astype(np.uint8)   # (H, W) binary
        iou_score = float(scores[0])
        dice = dice_score(pred_mask, gt_mask)
        dice_scores.append(dice)

        print(f"  {img_path.name[:50]:<50}  Dice={dice:.3f}  IoU_score={iou_score:.3f}")

        # Save figure
        save_path = out_dir / (img_path.stem + "_result.png")
        save_figure(image_rgb, gt_mask, pred_mask,
                    point_coords, point_labels,
                    dice, iou_score, save_path)

    # ---- Summary ----------------------------------------------------------
    if dice_scores:
        print(f"\n{'='*50}")
        print(f"  Images evaluated : {len(dice_scores)}")
        print(f"  Mean Dice        : {np.mean(dice_scores):.3f}")
        print(f"  Median Dice      : {np.median(dice_scores):.3f}")
        print(f"  Min / Max Dice   : {np.min(dice_scores):.3f} / {np.max(dice_scores):.3f}")
        print(f"  Results saved to : {out_dir.resolve()}")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()
