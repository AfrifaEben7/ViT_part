"""
pseudo_label.py — Generate pseudo-labeled training data from a microscopy video.

Pipeline:
  1. Extract all frames from the video → video_frames/
  2. Auto-detect cells on frame 0 via 488nm fluorescence thresholding
     (bright blobs on dark background) → point prompts
  3. Run SAM2VideoPredictor to track all detected cells through every frame
  4. Save each frame + merged COCO-RLE mask JSON → cell_camb_dataset/{split}/

The output directory structure matches our MacrophageDataset format:
    cell_camb_dataset/
      train/   (80% of frames)
        00000.jpg + 00000.json
        ...
      valid/   (20% of frames)
        ...

Usage:
    python3 pseudo_label.py [--video PATH] [--out_dir PATH] [--max_cells N]
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from pycocotools import mask as coco_mask
from scipy import ndimage

try:
    from sam2.build_sam import build_sam2_video_predictor
except ImportError:
    print("[ERROR] sam2 not found. Activate sam2_env first.", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video",      default="./Cell_CamB_ch0_stack0149_488nm_1121941msec_0004809588msecAbs_decon.mp4")
    p.add_argument("--checkpoint", default="./sam2_macrophage_best.pt",
                   help="Fine-tuned weights for initialisation (or base checkpoint)")
    p.add_argument("--base_ckpt",  default="./checkpoints/sam2_hiera_tiny.pt")
    p.add_argument("--model_cfg",  default="sam2_hiera_t.yaml")
    p.add_argument("--out_dir",    default="./cell_camb_dataset")
    p.add_argument("--frames_dir", default="./video_frames",
                   help="Temp directory to store extracted frames")
    p.add_argument("--max_cells",  type=int, default=20,
                   help="Maximum number of cells to track")
    p.add_argument("--min_cell_px", type=int, default=200,
                   help="Minimum blob area in pixels to be considered a cell")
    p.add_argument("--val_frac",   type=float, default=0.2,
                   help="Fraction of frames to put in valid/ split")
    p.add_argument("--score_thresh", type=float, default=0.5,
                   help="SAM2 mask score threshold for keeping a prediction")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Step 1: Extract frames
# ---------------------------------------------------------------------------
def extract_frames(video_path: str, frames_dir: Path):
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Check if already extracted
    existing = sorted(frames_dir.glob("*.jpg"))
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if len(existing) == total:
        print(f"[INFO] Frames already extracted ({total} frames) — skipping.")
        cap.release()
        return total

    print(f"[INFO] Extracting {total} frames to {frames_dir} ...")
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(str(frames_dir / f"{idx:05d}.jpg"), frame)
        idx += 1
    cap.release()
    print(f"[INFO] Extracted {idx} frames.")
    return idx


# ---------------------------------------------------------------------------
# Step 2: Auto-detect cells in frame 0
# ---------------------------------------------------------------------------
def detect_cells_frame0(frames_dir: Path, min_area: int, max_cells: int):
    """
    488nm fluorescence: cells are bright blobs on a dark background.
    Returns list of (x, y) centroid coordinates in pixel space.
    """
    frame0 = cv2.imread(str(frames_dir / "00000.jpg"))
    gray   = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)

    # Normalise to [0,255] uint8 (handles 16-bit or unusual dynamic range)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Otsu threshold to separate bright cells from background
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological cleanup: remove noise, fill holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = ndimage.binary_fill_holes(binary).astype(np.uint8) * 255

    # Connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

    # Filter by area; sort by area descending (largest = most prominent cells)
    cells = []
    for label_id in range(1, num_labels):   # skip background (0)
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= min_area:
            cx, cy = centroids[label_id]
            cells.append((area, cx, cy))

    cells.sort(reverse=True)   # largest first
    cells = cells[:max_cells]

    points = [(int(cx), int(cy)) for _, cx, cy in cells]
    print(f"[INFO] Detected {len(points)} cells in frame 0 "
          f"(filtered from {num_labels - 1} blobs, min_area={min_area}px)")

    # Save a debug image
    debug = frame0.copy()
    for i, (x, y) in enumerate(points):
        cv2.circle(debug, (x, y), 8, (0, 255, 0), -1)
        cv2.putText(debug, str(i+1), (x+10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 255), 1)
    cv2.imwrite(str(frames_dir.parent / "frame0_detections.jpg"), debug)
    print(f"[INFO] Detection debug image → frame0_detections.jpg")

    return points


# ---------------------------------------------------------------------------
# Step 3: SAM2 video tracking
# ---------------------------------------------------------------------------
def run_video_tracking(frames_dir: Path, cell_points: list,
                       base_ckpt: str, model_cfg: str, weights: str,
                       score_thresh: float, device: str):
    """
    Returns dict: {frame_idx: merged_binary_mask (H, W) uint8}
    """
    print(f"\n[INFO] Loading SAM2 video predictor ...")
    predictor = build_sam2_video_predictor(model_cfg, base_ckpt, device=device)

    # Load fine-tuned weights into the predictor's model
    if Path(weights).exists():
        print(f"[INFO] Loading fine-tuned weights: {weights}")
        state = torch.load(weights, map_location=device)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        predictor.load_state_dict(state, strict=False)
    else:
        print(f"[WARN] Fine-tuned weights not found, using base model.")

    frame_masks = {}

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        # Initialise video state
        print(f"[INFO] Initialising video state from {frames_dir} ...")
        inference_state = predictor.init_state(video_path=str(frames_dir))
        predictor.reset_state(inference_state)

        # Register each detected cell as a separate tracked object on frame 0
        print(f"[INFO] Adding {len(cell_points)} point prompts on frame 0 ...")
        for obj_id, (px, py) in enumerate(cell_points, start=1):
            _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=obj_id,
                points=np.array([[px, py]], dtype=np.float32),
                labels=np.array([1], dtype=np.int32),
            )

        # Propagate through all frames
        print(f"[INFO] Propagating masks through video ...")
        n_frames = len(sorted(frames_dir.glob("*.jpg")))

        for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(inference_state):
            # Merge all object masks into one binary mask
            h, w = mask_logits.shape[-2], mask_logits.shape[-1]
            merged = np.zeros((h, w), dtype=np.uint8)

            for obj_id, logit in zip(obj_ids, mask_logits):
                score = torch.sigmoid(logit).max().item()
                if score >= score_thresh:
                    binary = (logit.squeeze() > 0).cpu().numpy().astype(np.uint8)
                    merged = np.maximum(merged, binary)

            frame_masks[frame_idx] = merged

            if (frame_idx + 1) % 50 == 0 or frame_idx == n_frames - 1:
                covered = merged.sum()
                print(f"  Frame {frame_idx+1:>4}/{n_frames}  "
                      f"mask_pixels={covered:>7}")

    print(f"[INFO] Tracking complete: {len(frame_masks)} frames processed.")
    return frame_masks


# ---------------------------------------------------------------------------
# Step 4: Save pseudo-labeled dataset
# ---------------------------------------------------------------------------
def mask_to_coco_rle(mask: np.ndarray) -> dict:
    """Convert binary (H,W) uint8 mask to COCO compressed-RLE dict."""
    rle = coco_mask.encode(np.asfortranarray(mask))
    rle["counts"] = rle["counts"].decode("utf-8")   # make JSON-serialisable
    return rle


def save_dataset(frames_dir: Path, frame_masks: dict,
                 out_dir: Path, val_frac: float):
    frame_indices = sorted(frame_masks.keys())
    n_total = len(frame_indices)
    n_val   = max(1, int(n_total * val_frac))
    n_train = n_total - n_val

    # Deterministic split: last val_frac% → valid
    train_indices = frame_indices[:n_train]
    valid_indices = frame_indices[n_train:]

    print(f"\n[INFO] Saving dataset: {n_train} train / {n_val} valid")

    for split, indices in [("train", train_indices), ("valid", valid_indices)]:
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        for frame_idx in indices:
            mask   = frame_masks[frame_idx]
            h, w   = mask.shape
            stem   = f"{frame_idx:05d}"

            # Copy frame image
            src_img  = frames_dir / f"{stem}.jpg"
            dest_img = split_dir  / f"{stem}.jpg"
            shutil.copy2(src_img, dest_img)

            # Build annotation JSON (same format as Roboflow SAM2 export)
            rle = mask_to_coco_rle(mask)
            ann_data = {
                "image": {
                    "file_name": f"{stem}.jpg",
                    "height": h,
                    "width":  w,
                },
                "annotations": [
                    {
                        "id": 0,
                        "segmentation": rle,
                    }
                ] if mask.sum() > 0 else []   # skip empty masks
            }

            json_path = split_dir / f"{stem}.json"
            with open(json_path, "w") as f:
                json.dump(ann_data, f)

    print(f"[INFO] Dataset saved to: {out_dir.resolve()}")
    print(f"       train/ : {n_train} pairs")
    print(f"       valid/ : {n_val}   pairs")

    # Quick coverage report
    all_masks = list(frame_masks.values())
    coverage  = [m.sum() / m.size * 100 for m in all_masks]
    print(f"\n[INFO] Mask coverage stats:")
    print(f"       Mean  : {np.mean(coverage):.1f}%")
    print(f"       Min   : {np.min(coverage):.1f}%")
    print(f"       Max   : {np.max(coverage):.1f}%")
    empty = sum(1 for m in all_masks if m.sum() == 0)
    if empty:
        print(f"       [WARN] {empty} frames have empty masks — "
              "consider lowering --score_thresh")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"[ERROR] Video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    frames_dir = Path(args.frames_dir)
    out_dir    = Path(args.out_dir)

    # 1. Extract frames
    extract_frames(str(video_path), frames_dir)

    # 2. Auto-detect cells in frame 0
    cell_points = detect_cells_frame0(frames_dir, args.min_cell_px, args.max_cells)
    if not cell_points:
        print("[ERROR] No cells detected in frame 0. "
              "Try lowering --min_cell_px.", file=sys.stderr)
        sys.exit(1)

    # 3. SAM2 video tracking
    frame_masks = run_video_tracking(
        frames_dir, cell_points,
        args.base_ckpt, args.model_cfg, args.checkpoint,
        args.score_thresh, device,
    )

    # 4. Save pseudo-labeled dataset
    save_dataset(frames_dir, frame_masks, out_dir, args.val_frac)

    print("\n[DONE] Pseudo-labeling complete.")
    print(f"       Next step: retrain with --data_root {out_dir.resolve()}")
    print(f"       Or merge with existing dataset and retrain.")


if __name__ == "__main__":
    main()
