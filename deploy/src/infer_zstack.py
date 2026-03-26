"""
infer_zstack.py — Segment cells in a 3-D fluorescence z-stack.

Pipeline:
  1. Load multi-page TIFF z-stack (uint16 grayscale)
  2. Compute max-Z projection → single 2-D image
  3. Detect cells (merge blobs, centre-first, edge exclusion)
  4. SAM2 image predictor: bounding box + n_pos interior points + n_neg background
  5. Save result image(s) to out_dir/

Usage:
    python3 infer_zstack.py [--tif PATH] [--checkpoint PATH]
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import ndimage

try:
    import tifffile
except ImportError:
    print("[ERROR] tifffile not installed. Run: pip install tifffile", file=sys.stderr)
    sys.exit(1)

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    print("[ERROR] sam2 not found. Activate sam2_env first.", file=sys.stderr)
    sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tif",          default="./videos/cell_CamB_ch0_stack0000_488nm_0000000msec_0002581605msecAbs_decon.tif")
    p.add_argument("--checkpoint",   default="./weights/small/sam2_macrophage_small_best.pt")
    p.add_argument("--base_ckpt",    default="./checkpoints/sam2_hiera_small.pt")
    p.add_argument("--model_cfg",    default="sam2_hiera_s.yaml")
    p.add_argument("--out_dir",      default="./outputs/zstack")
    p.add_argument("--projection",   default="max", choices=["max", "mean", "sum"],
                   help="Z-projection method")
    p.add_argument("--max_cells",    type=int,   default=1)
    p.add_argument("--min_cell_px",  type=int,   default=200)
    p.add_argument("--merge_px",     type=int,   default=25)
    p.add_argument("--edge_margin",  type=float, default=0.15)
    p.add_argument("--n_pos",        type=int,   default=4)
    p.add_argument("--n_neg",        type=int,   default=2)
    p.add_argument("--score_thresh", type=float, default=0.5)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Z-projection
# ---------------------------------------------------------------------------
def z_project(tif_path: str, method: str = "max"):
    with tifffile.TiffFile(tif_path) as tif:
        stack = tif.asarray().astype(np.float32)   # (Z, H, W)

    if method == "max":
        proj = stack.max(axis=0)
    elif method == "mean":
        proj = stack.mean(axis=0)
    else:  # sum
        proj = stack.sum(axis=0)

    # Normalise to uint8
    vmin, vmax = proj.min(), proj.max()
    if vmax == vmin:
        vmax = vmin + 1
    proj8 = ((proj - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    bgr   = cv2.cvtColor(proj8, cv2.COLOR_GRAY2BGR)

    print(f"[INFO] Z-stack: {stack.shape[0]} slices → {method}-projection "
          f"({stack.shape[2]}×{stack.shape[1]}, raw range {int(stack.min())}–{int(stack.max())})")
    return bgr


# ---------------------------------------------------------------------------
# Cell detection (same logic as infer_video.py)
# ---------------------------------------------------------------------------
def detect_cells(img_bgr, min_area, max_cells, merge_px, edge_margin):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k5, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k5, iterations=2)
    binary = ndimage.binary_fill_holes(binary).astype(np.uint8) * 255

    s  = merge_px * 2 + 1
    mk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s, s))
    dilated = cv2.dilate(binary, mk)

    H, W = img_bgr.shape[:2]
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated)

    cells = []
    for lid in range(1, num_labels):
        area = stats[lid, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        x, y  = stats[lid, cv2.CC_STAT_LEFT], stats[lid, cv2.CC_STAT_TOP]
        bw, bh = stats[lid, cv2.CC_STAT_WIDTH], stats[lid, cv2.CC_STAT_HEIGHT]
        cx, cy = int(centroids[lid][0]), int(centroids[lid][1])
        x1 = max(0,   x  + merge_px);  y1 = max(0,   y  + merge_px)
        x2 = min(W-1, x  + bw - merge_px); y2 = min(H-1, y  + bh - merge_px)
        if x2 <= x1 or y2 <= y1:
            x1, y1, x2, y2 = x, y, x+bw, y+bh
        cells.append({
            "area":      area,
            "bbox":      (x1, y1, x2, y2),
            "centroid":  (cx, cy),
            "orig_mask": (binary > 0) & (labels == lid),
        })

    # Edge exclusion + sort by distance from centre
    mx, my = W * edge_margin, H * edge_margin
    cells = [c for c in cells
             if mx <= c["centroid"][0] <= W - mx
             and my <= c["centroid"][1] <= H - my]
    cx_img, cy_img = W / 2.0, H / 2.0
    cells.sort(key=lambda c: (c["centroid"][0]-cx_img)**2 + (c["centroid"][1]-cy_img)**2)
    cells = cells[:max_cells]

    print(f"[INFO] Detected {len(cells)} cell(s)")
    return cells


# ---------------------------------------------------------------------------
# Point sampling
# ---------------------------------------------------------------------------
def sample_positive_points(cell, n_pos):
    cx, cy = cell["centroid"]
    pts    = [(cx, cy)]
    if n_pos <= 1:
        return pts
    mask   = cell["orig_mask"]
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return pts
    for angle in np.linspace(0, 2 * np.pi, n_pos - 1, endpoint=False):
        dx, dy = np.cos(angle), np.sin(angle)
        proj   = (xs - cx) * dx + (ys - cy) * dy
        in_dir = proj > 0
        if not np.any(in_dir):
            pts.append((cx, cy))
            continue
        target = proj[in_dir].max() * 0.75
        best   = np.argmin(np.abs(proj[in_dir] - target))
        pts.append((int(xs[in_dir][best]), int(ys[in_dir][best])))
    return pts


def sample_negative_points(cells, img_shape, n_neg, safety_px=40):
    H, W = img_shape[:2]
    forbidden = np.zeros((H, W), dtype=np.uint8)
    for cell in cells:
        m  = cell["orig_mask"].astype(np.uint8)
        s  = safety_px * 2 + 1
        mk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s, s))
        forbidden = np.maximum(forbidden, cv2.dilate(m, mk))
    bg_ys, bg_xs = np.where(forbidden == 0)
    if len(bg_xs) < n_neg:
        return []
    step = max(1, len(bg_xs) // (n_neg * 20))
    cands = list(zip(bg_xs[::step], bg_ys[::step]))
    idxs  = np.linspace(0, len(cands)-1, n_neg, dtype=int)
    return [cands[i] for i in idxs]


# ---------------------------------------------------------------------------
# Save result
# ---------------------------------------------------------------------------
def save_result(img_bgr, cells, pred_masks, out_dir: Path, stem: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    vis = img_bgr.copy()
    overlay = vis.copy()

    COLOURS = [(0,255,0),(0,128,255),(255,0,0),(0,255,255),(255,0,255)]

    for i, (cell, mask) in enumerate(zip(cells, pred_masks)):
        colour      = COLOURS[i % len(COLOURS)]
        x1,y1,x2,y2 = cell["bbox"]
        H, W        = img_bgr.shape[:2]
        if mask.shape != (H, W):
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

        overlay[mask > 0] = colour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, colour, 2)
        cv2.rectangle(vis, (x1, y1), (x2, y2), colour, 2)

        # Draw prompt points
        for px, py in cell.get("pos_pts", []):
            cv2.circle(vis, (px, py), 6, (0, 255, 0), -1)
            cv2.circle(vis, (px, py), 6, (255,255,255), 1)
        for px, py in cell.get("neg_pts", []):
            cv2.circle(vis, (px, py), 6, (0, 0, 255), -1)
            cv2.circle(vis, (px, py), 6, (255,255,255), 1)

    cv2.addWeighted(overlay, 0.4, vis, 0.6, 0, vis)

    # Side-by-side: original | result
    combined = np.hstack([img_bgr, vis])
    cv2.putText(combined, "Max-Z Projection", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.putText(combined, "SAM2 Segmentation", (img_bgr.shape[1]+10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    out_path = out_dir / f"{stem}_result.jpg"
    cv2.imwrite(str(out_path), combined)
    print(f"[INFO] Saved: {out_path}")

    # Also save projection alone
    proj_path = out_dir / f"{stem}_projection.jpg"
    cv2.imwrite(str(proj_path), img_bgr)
    print(f"[INFO] Saved projection: {proj_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")

    tif_path = Path(args.tif)
    if not tif_path.exists():
        print(f"[ERROR] File not found: {tif_path}", file=sys.stderr)
        sys.exit(1)

    # 1. Z-projection
    img_bgr = z_project(str(tif_path), args.projection)

    # 2. Detect cells
    cells = detect_cells(img_bgr, args.min_cell_px, args.max_cells,
                         args.merge_px, args.edge_margin)
    if not cells:
        print("[ERROR] No cells detected. Try lowering --min_cell_px.", file=sys.stderr)
        sys.exit(1)

    # 3. Build prompt points
    neg_pts = sample_negative_points(cells, img_bgr.shape, args.n_neg)
    for cell in cells:
        cell["pos_pts"] = sample_positive_points(cell, args.n_pos)
        cell["neg_pts"] = neg_pts

    # 4. Load SAM2
    print(f"[INFO] Loading SAM2 ({args.model_cfg}) ...")
    model = build_sam2(args.model_cfg, args.base_ckpt, device=device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    model_sd   = model.state_dict()
    compatible = {k: v for k, v in ckpt.items()
                  if k in model_sd and model_sd[k].shape == v.shape}
    model.load_state_dict(compatible, strict=False)
    print(f"[INFO] Loaded {len(compatible)}/{len(ckpt)} layers")
    model.eval()
    predictor = SAM2ImagePredictor(model)

    # 5. Inference
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pred_masks = []

    with torch.inference_mode(), \
         torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        predictor.set_image(img_rgb)

        for cell in cells:
            pos_pts = cell["pos_pts"]
            neg_pts = cell["neg_pts"]
            x1,y1,x2,y2 = cell["bbox"]

            pts    = np.array(pos_pts + neg_pts, dtype=np.float32)
            labels = np.array([1]*len(pos_pts) + [0]*len(neg_pts), dtype=np.int32)
            box    = np.array([x1, y1, x2, y2], dtype=np.float32)

            masks, scores, _ = predictor.predict(
                point_coords=pts,
                point_labels=labels,
                box=box[None],
                multimask_output=False,
            )
            score = float(scores[0])
            print(f"[INFO] Cell at ({cell['centroid'][0]},{cell['centroid'][1]}) "
                  f"→ IoU score: {score:.3f}")
            if score >= args.score_thresh:
                pred_masks.append((masks[0] > 0).astype(np.uint8))
            else:
                print(f"  [WARN] Score below threshold ({score:.3f} < {args.score_thresh}) — skipping")
                pred_masks.append(np.zeros(img_rgb.shape[:2], dtype=np.uint8))

    # 6. Save
    stem = tif_path.stem
    save_result(img_bgr, cells, pred_masks, Path(args.out_dir), stem)
    print("\n[DONE]")


if __name__ == "__main__":
    main()
