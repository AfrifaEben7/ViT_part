"""
infer_jetson.py — Lightweight SAM2-tiny inference for Jetson Orin Nano.

Supports:
  - Single image  (--mode image  --input path/to/image.jpg)
  - Video / TIF   (--mode video  --input path/to/video.mp4)

Prompting: auto-detects the largest centred object via Otsu + morphology,
then prompts with 4 positive (centre + 3 inner-edge) + 2 negative points
and a bounding box.  Results are saved as annotated images / a video.

Usage:
    python3 infer_jetson.py --mode image --input cell.jpg
    python3 infer_jetson.py --mode video --input cell.mp4 --out output.mp4
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import ndimage

try:
    from sam2.build_sam import build_sam2, build_sam2_video_predictor
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    print("[ERROR] sam2 not installed. Run deploy/setup_jetson.sh first.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",        choices=["image", "video"], default="image")
    p.add_argument("--input",       required=True)
    p.add_argument("--out",         default="./jetson_output")
    p.add_argument("--weights",     default="./weights/tiny/sam2_macrophage_jetson_best.pt",
                   help="Fine-tuned weights (falls back to base if not found)")
    p.add_argument("--base_ckpt",   default="./checkpoints/sam2_hiera_tiny.pt")
    p.add_argument("--model_cfg",   default="sam2_hiera_t_512.yaml",
                   help="Use sam2_hiera_t.yaml if trained at 1024")
    p.add_argument("--max_cells",   type=int,   default=1)
    p.add_argument("--min_cell_px", type=int,   default=150)
    p.add_argument("--merge_px",    type=int,   default=20)
    p.add_argument("--score_thresh",type=float, default=0.5)
    p.add_argument("--alpha",       type=float, default=0.45)
    p.add_argument("--fps",         type=float, default=10.0,
                   help="Output video FPS")
    return p.parse_args()


COLOURS = [
    (0, 255, 0), (0, 128, 255), (255, 0, 0), (0, 255, 255),
    (255, 0, 255),(255, 255, 0),(0, 200, 100),(200, 0, 100),
]


# ---------------------------------------------------------------------------
# Shared: load model weights
# ---------------------------------------------------------------------------
def _load_weights(model, weights_path: str, device: str):
    wp = Path(weights_path)
    if not wp.exists():
        print(f"[WARN] Weights not found at {wp} — using base model.")
        return
    ckpt = torch.load(str(wp), map_location=device)
    if "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    elif "model" in ckpt:
        ckpt = ckpt["model"]
    model_sd   = model.state_dict()
    compatible = {k: v for k, v in ckpt.items()
                  if k in model_sd and model_sd[k].shape == v.shape}
    model.load_state_dict(compatible, strict=False)
    print(f"[INFO] Loaded {len(compatible)}/{len(ckpt)} fine-tuned layers.")


# ---------------------------------------------------------------------------
# Cell detection
# ---------------------------------------------------------------------------
def detect_cells(img_bgr: np.ndarray, min_area: int, max_cells: int,
                 merge_px: int):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=2)
    binary = ndimage.binary_fill_holes(binary).astype(np.uint8) * 255

    # Merge nearby blobs
    s  = merge_px * 2 + 1
    mk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s, s))
    dilated = cv2.dilate(binary, mk)

    H, W = img_bgr.shape[:2]
    cx_img, cy_img = W / 2, H / 2
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated)

    cells = []
    for lid in range(1, n):
        area = stats[lid, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        x  = stats[lid, cv2.CC_STAT_LEFT]
        y  = stats[lid, cv2.CC_STAT_TOP]
        bw = stats[lid, cv2.CC_STAT_WIDTH]
        bh = stats[lid, cv2.CC_STAT_HEIGHT]
        cx = int(centroids[lid][0])
        cy = int(centroids[lid][1])
        x1 = max(0,   x  + merge_px); y1 = max(0,   y  + merge_px)
        x2 = min(W-1, x  + bw - merge_px); y2 = min(H-1, y + bh - merge_px)
        if x2 <= x1 or y2 <= y1:
            x1, y1, x2, y2 = x, y, x+bw, y+bh
        # Score: area × proximity to image centre
        dist  = ((cx - cx_img)**2 + (cy - cy_img)**2) ** 0.5
        score = area / (1 + dist)
        orig_mask = ((binary > 0) & (labels == lid))
        cells.append({"score": score, "bbox": (x1,y1,x2,y2),
                      "centroid": (cx,cy), "orig_mask": orig_mask})

    cells.sort(key=lambda c: c["score"], reverse=True)
    return cells[:max_cells]


def _sample_positive_points(cell, n_pos=4):
    cx, cy = cell["centroid"]
    pts = [(cx, cy)]
    if n_pos <= 1:
        return pts
    mask = cell["orig_mask"]
    ys, xs = np.where(mask)
    if not len(xs):
        return pts
    for angle in np.linspace(0, 2*np.pi, n_pos-1, endpoint=False):
        dx, dy = np.cos(angle), np.sin(angle)
        proj = (xs - cx)*dx + (ys - cy)*dy
        fwd  = proj > 0
        if not np.any(fwd):
            pts.append((cx, cy)); continue
        target = proj[fwd].max() * 0.75
        best   = np.argmin(np.abs(proj[fwd] - target))
        pts.append((int(xs[fwd][best]), int(ys[fwd][best])))
    return pts


def _sample_negative_points(cells, img_shape, n_neg=2, safety=40):
    H, W = img_shape[:2]
    forbidden = np.zeros((H, W), dtype=np.uint8)
    for c in cells:
        m  = c["orig_mask"].astype(np.uint8)
        mk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (safety*2+1,)*2)
        forbidden = np.maximum(forbidden, cv2.dilate(m, mk))
    bg_ys, bg_xs = np.where(forbidden == 0)
    if len(bg_xs) < n_neg:
        return []
    step = max(1, len(bg_xs) // (n_neg * 20))
    cands = list(zip(bg_xs[::step], bg_ys[::step]))
    idxs  = np.linspace(0, len(cands)-1, n_neg, dtype=int)
    return [cands[i] for i in idxs]


def build_prompts(cells, img_shape, n_pos=4, n_neg=2):
    """Returns list of (points_np, labels_np, box_np) per cell."""
    neg_pts = _sample_negative_points(cells, img_shape, n_neg)
    prompts = []
    for cell in cells:
        pos = _sample_positive_points(cell, n_pos)
        pts    = np.array(pos + neg_pts, dtype=np.float32)
        labels = np.array([1]*len(pos) + [0]*len(neg_pts), dtype=np.int32)
        x1,y1,x2,y2 = cell["bbox"]
        box    = np.array([x1,y1,x2,y2], dtype=np.float32)
        prompts.append((pts, labels, box))
    return prompts


# ---------------------------------------------------------------------------
# Image mode
# ---------------------------------------------------------------------------
def run_image(args, device):
    img_bgr = cv2.imread(args.input)
    if img_bgr is None:
        print(f"[ERROR] Cannot read: {args.input}"); sys.exit(1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    print("[INFO] Building SAM2 image predictor ...")
    model = build_sam2(args.model_cfg, args.base_ckpt, device=device)
    _load_weights(model, args.weights, device)
    predictor = SAM2ImagePredictor(model)

    cells = detect_cells(img_bgr, args.min_cell_px, args.max_cells, args.merge_px)
    if not cells:
        print("[WARN] No cells detected. Lower --min_cell_px.")
    prompts = build_prompts(cells, img_bgr.shape)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = img_bgr.copy()
    overlay = img_bgr.copy()

    with torch.inference_mode(), \
         torch.autocast(device_type="cuda", dtype=torch.float16):
        predictor.set_image(img_rgb)
        for i, (pts, labels, box) in enumerate(prompts):
            masks, scores, _ = predictor.predict(
                point_coords=pts, point_labels=labels,
                box=box, multimask_output=False)
            score = float(scores[0])
            if score < args.score_thresh:
                continue
            mask   = (masks[0] > 0).astype(np.uint8)
            colour = COLOURS[i % len(COLOURS)]
            overlay[mask > 0] = colour
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result, contours, -1, colour, 2)
            print(f"  Cell {i+1}: IoU={score:.3f}")

    cv2.addWeighted(overlay, args.alpha, result, 1-args.alpha, 0, result)
    stem   = Path(args.input).stem
    outpath = out_dir / f"{stem}_result.jpg"
    cv2.imwrite(str(outpath), result)
    print(f"[DONE] Saved: {outpath}")


# ---------------------------------------------------------------------------
# Video / TIF mode
# ---------------------------------------------------------------------------
def extract_frames(input_path: str, frames_dir: Path):
    frames_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(input_path).suffix.lower()

    if ext in (".tif", ".tiff"):
        import tifffile
        with tifffile.TiffFile(input_path) as tif:
            stack = tif.asarray()
        if stack.ndim == 2:
            stack = stack[np.newaxis]
        n = stack.shape[0]
        lo, hi = stack.min(), stack.max()
        for i, page in enumerate(stack):
            norm = ((page - lo) / max(hi - lo, 1) * 255).astype(np.uint8)
            bgr  = cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(str(frames_dir / f"{i:05d}.jpg"), bgr)
        return n, 10.0
    else:
        cap   = cv2.VideoCapture(input_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps   = cap.get(cv2.CAP_PROP_FPS) or 10.0
        existing = sorted(frames_dir.glob("*.jpg"))
        if len(existing) == total:
            cap.release(); return total, fps
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            cv2.imwrite(str(frames_dir / f"{idx:05d}.jpg"), frame)
            idx += 1
        cap.release()
        return total, fps


def run_video(args, device):
    out_dir    = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"

    print("[INFO] Extracting frames ...")
    n_frames, fps = extract_frames(args.input, frames_dir)
    frame_files   = sorted(frames_dir.glob("*.jpg"))
    sample        = cv2.imread(str(frame_files[0]))
    H, W          = sample.shape[:2]
    print(f"[INFO] {n_frames} frames, {W}x{H} @ {fps:.1f}fps")

    print("[INFO] Building SAM2 video predictor ...")
    predictor = build_sam2_video_predictor(args.model_cfg, args.base_ckpt, device=device)
    _load_weights(predictor, args.weights, device)

    # Detect cells in frame 0
    cells = detect_cells(sample, args.min_cell_px, args.max_cells, args.merge_px)
    if not cells:
        print("[WARN] No cells detected in frame 0."); sys.exit(1)
    neg_pts = _sample_negative_points(cells, sample.shape, n_neg=2)

    all_masks = {}
    with torch.inference_mode(), \
         torch.autocast(device_type="cuda", dtype=torch.float16):

        state = predictor.init_state(video_path=str(frames_dir))

        # Forward pass
        predictor.reset_state(state)
        for obj_id, cell in enumerate(cells, 1):
            pos  = _sample_positive_points(cell, n_pos=4)
            pts  = np.array(pos + neg_pts, dtype=np.float32)
            lbls = np.array([1]*len(pos) + [0]*len(neg_pts), dtype=np.int32)
            x1,y1,x2,y2 = cell["bbox"]
            predictor.add_new_points_or_box(
                state, frame_idx=0, obj_id=obj_id,
                points=pts, labels=lbls,
                box=np.array([x1,y1,x2,y2], dtype=np.float32))

        fwd = {}
        for fi, obj_ids, logits in predictor.propagate_in_video(state):
            fwd[fi] = {}
            for oid, logit in zip(obj_ids, logits):
                if torch.sigmoid(logit).max().item() >= args.score_thresh:
                    fwd[fi][int(oid)] = (logit.squeeze() > 0).cpu().numpy().astype(np.uint8)

        # Backward pass
        predictor.reset_state(state)
        for obj_id, cell in enumerate(cells, 1):
            pos  = _sample_positive_points(cell, n_pos=4)
            pts  = np.array(pos + neg_pts, dtype=np.float32)
            lbls = np.array([1]*len(pos) + [0]*len(neg_pts), dtype=np.int32)
            x1,y1,x2,y2 = cell["bbox"]
            predictor.add_new_points_or_box(
                state, frame_idx=0, obj_id=obj_id,
                points=pts, labels=lbls,
                box=np.array([x1,y1,x2,y2], dtype=np.float32))

        bwd = {}
        for fi, obj_ids, logits in predictor.propagate_in_video(
                state, start_frame_idx=n_frames-1, reverse=True):
            bwd[fi] = {}
            for oid, logit in zip(obj_ids, logits):
                if torch.sigmoid(logit).max().item() >= args.score_thresh:
                    bwd[fi][int(oid)] = (logit.squeeze() > 0).cpu().numpy().astype(np.uint8)

    # Merge forward + backward
    for fi in range(n_frames):
        f, b   = fwd.get(fi, {}), bwd.get(fi, {})
        merged = {}
        for oid in set(list(f) + list(b)):
            if oid in f and oid in b:
                merged[oid] = np.maximum(f[oid], b[oid])
            elif oid in f: merged[oid] = f[oid]
            else:          merged[oid] = b[oid]
        all_masks[fi] = merged

    # Render
    out_video = str(out_dir / (Path(args.input).stem + "_tracked.mp4"))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_video, fourcc, fps, (W, H))

    for fi, fpath in enumerate(frame_files):
        frame   = cv2.imread(str(fpath))
        overlay = frame.copy()
        for oid, mask in all_masks.get(fi, {}).items():
            colour = COLOURS[(oid-1) % len(COLOURS)]
            if mask.shape != (H, W):
                mask = cv2.resize(mask, (W,H), interpolation=cv2.INTER_NEAREST)
            overlay[mask > 0] = colour
            cntrs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, cntrs, -1, colour, 2)
        cv2.addWeighted(overlay, args.alpha, frame, 1-args.alpha, 0, frame)
        n_cells = len(all_masks.get(fi, {}))
        cv2.putText(frame, f"Frame {fi+1}/{n_frames}  Cells: {n_cells}",
                    (10,28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        writer.write(frame)

    writer.release()
    mb = Path(out_video).stat().st_size / 1024**2
    print(f"[DONE] Saved: {out_video}  ({mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'}")

    if args.mode == "image":
        run_image(args, device)
    else:
        run_video(args, device)


if __name__ == "__main__":
    main()
