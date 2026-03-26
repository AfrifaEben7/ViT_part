"""
infer_video.py — Run fine-tuned SAM2 on a microscopy video.

Prompting strategy:
  - 3 anchor frames spread across the video (best cell visibility)
  - Nearby blobs merged via dilation → one object per cell (no split-cell issue)
  - Bounding box + positive centroid + negative background points per object
  - Bidirectional propagation (forward + backward), masks merged by union

Usage:
    python3 infer_video.py [--video PATH] [--checkpoint PATH]
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import ndimage

try:
    from sam2.build_sam import build_sam2_video_predictor
except ImportError:
    print("[ERROR] sam2 not found. Activate sam2_env first.", file=sys.stderr)
    sys.exit(1)


CELL_COLOURS = [
    (0, 255, 0),   (0, 128, 255), (255, 0, 0),   (0, 255, 255),
    (255, 0, 255), (255, 255, 0), (0, 200, 100),  (200, 0, 100),
    (100, 200, 0), (0, 100, 200), (150, 50, 255), (255, 150, 50),
    (50, 255, 150),(255, 50, 150),(150, 255, 50),  (50, 150, 255),
    (200, 200, 0), (0, 200, 200), (200, 0, 200),  (128, 128, 0),
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video",         default="./Cell_CamB_ch0_stack0149_488nm_1121941msec_0004809588msecAbs_decon.mp4")
    p.add_argument("--checkpoint",    default="./sam2_macrophage_best.pt")
    p.add_argument("--base_ckpt",     default="./checkpoints/sam2_hiera_tiny.pt")
    p.add_argument("--model_cfg",     default="sam2_hiera_t.yaml")
    p.add_argument("--frames_dir",    default="./video_frames")
    p.add_argument("--out_video",     default="./output_video.mp4")
    p.add_argument("--n_anchors",     type=int,   default=3)
    p.add_argument("--anchor_window", type=int,   default=0,
                   help="Search window around each anchor position (0=auto)")
    p.add_argument("--max_cells",     type=int,   default=1)
    p.add_argument("--edge_margin",   type=float, default=0.15,
                   help="Fraction of frame to exclude near each edge (0-0.4)")
    p.add_argument("--min_cell_px",   type=int,   default=200)
    p.add_argument("--merge_px",      type=int,   default=25,
                   help="Dilation radius (px) to merge nearby blobs into one cell")
    p.add_argument("--n_pos",         type=int,   default=3,
                   help="Positive points sampled from inside the cell mask")
    p.add_argument("--n_neg",         type=int,   default=2,
                   help="Negative background points per object")
    p.add_argument("--score_thresh",  type=float, default=0.5)
    p.add_argument("--alpha",         type=float, default=0.45)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------
def _extract_tif(video_path: str, frames_dir: Path, fps: float = 10.0):
    """Extract frames from a multi-page TIFF stack (uint16 grayscale)."""
    import tifffile
    with tifffile.TiffFile(video_path) as tif:
        pages = tif.pages
        total = len(pages)
        p0    = pages[0].asarray()
        h, w  = p0.shape[:2]

        existing = sorted(frames_dir.glob("*.jpg"))
        if len(existing) == total:
            print(f"[INFO] Frames already extracted ({total} frames) — skipping.")
            return total, fps, w, h

        print(f"[INFO] Extracting {total} TIF frames ({w}x{h}) ...")
        # Global normalisation across the whole stack for consistent brightness
        stack = tif.asarray()                          # (N, H, W) uint16
        vmin, vmax = stack.min(), stack.max()
        if vmax == vmin:
            vmax = vmin + 1

        for idx, page in enumerate(pages):
            gray = page.asarray().astype(np.float32)
            gray = ((gray - vmin) / (vmax - vmin) * 255).clip(0, 255).astype(np.uint8)
            bgr  = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(str(frames_dir / f"{idx:05d}.jpg"), bgr)

        print(f"[INFO] Done (normalised {vmin}–{vmax} → 0–255).")
        return total, fps, w, h


def extract_frames(video_path: str, frames_dir: Path, tif_fps: float = 10.0):
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Handle multi-page TIFF stacks
    if Path(video_path).suffix.lower() in (".tif", ".tiff"):
        return _extract_tif(video_path, frames_dir, tif_fps)

    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    existing = sorted(frames_dir.glob("*.jpg"))
    if len(existing) == total:
        print(f"[INFO] Frames already extracted ({total} frames) — skipping.")
        cap.release()
        return total, fps, w, h

    print(f"[INFO] Extracting {total} frames ({w}x{h} @ {fps:.1f}fps) ...")
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(str(frames_dir / f"{idx:05d}.jpg"), frame)
        idx += 1
    cap.release()
    print(f"[INFO] Done.")
    return total, fps, w, h


# ---------------------------------------------------------------------------
# Cell detection: merge nearby blobs → bounding boxes
# ---------------------------------------------------------------------------
def detect_cells_in_frame(img_bgr: np.ndarray, min_area: int,
                           max_cells: int, merge_px: int,
                           edge_margin: float = 0.15):
    """
    Returns list of dicts: {bbox:(x1,y1,x2,y2), centroid:(cx,cy), dilated_mask}

    Key step: dilate the thresholded image by merge_px before running connected
    components.  This merges nearby bright fragments (e.g. two bright lobes of
    the same macrophage) into a single object.  The bounding box is then
    shrunk back by merge_px to approximate the true cell boundary.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  k5, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k5, iterations=2)
    binary = ndimage.binary_fill_holes(binary).astype(np.uint8) * 255

    # Merge nearby fragments
    s = merge_px * 2 + 1
    mk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s, s))
    dilated = cv2.dilate(binary, mk)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(dilated)

    H, W = img_bgr.shape[:2]
    cells = []
    for lid in range(1, num_labels):
        area = stats[lid, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        x  = stats[lid, cv2.CC_STAT_LEFT]
        y  = stats[lid, cv2.CC_STAT_TOP]
        bw = stats[lid, cv2.CC_STAT_WIDTH]
        bh = stats[lid, cv2.CC_STAT_HEIGHT]
        cx = int(centroids[lid][0])
        cy = int(centroids[lid][1])
        # Shrink box back by merge_px to undo dilation padding
        x1 = max(0,   x  + merge_px)
        y1 = max(0,   y  + merge_px)
        x2 = min(W-1, x  + bw - merge_px)
        y2 = min(H-1, y  + bh - merge_px)
        if x2 <= x1 or y2 <= y1:        # fallback: keep full dilated box
            x1, y1, x2, y2 = x, y, x+bw, y+bh
        cells.append({
            "area":       area,
            "bbox":       (x1, y1, x2, y2),
            "centroid":   (cx, cy),
            "label_mask": labels == lid,              # dilated region (H,W)
            "orig_mask":  (binary > 0) & (labels == lid),  # undilated interior
        })

    # Reject cells too close to any edge
    margin_x = W * edge_margin
    margin_y = H * edge_margin
    cells = [c for c in cells
             if margin_x <= c["centroid"][0] <= W - margin_x
             and margin_y <= c["centroid"][1] <= H - margin_y]

    # Sort by distance from image centre — closest first
    cx_img, cy_img = W / 2.0, H / 2.0
    cells.sort(key=lambda c: (c["centroid"][0] - cx_img) ** 2
                              + (c["centroid"][1] - cy_img) ** 2)
    return cells[:max_cells]


def _cell_score(img_bgr: np.ndarray, min_area: int, merge_px: int,
                edge_margin: float) -> int:
    """
    Score a frame by the area of the cell closest to the image centre,
    ignoring anything near the edges.
    """
    cells = detect_cells_in_frame(img_bgr, min_area, 1, merge_px, edge_margin)
    if not cells:
        return 0
    return cells[0]["area"]


# ---------------------------------------------------------------------------
# Negative point sampling
# ---------------------------------------------------------------------------
def sample_negative_points(cells: list, img_shape: tuple,
                            n_neg: int, safety_px: int = 40):
    """
    Sample n_neg points from the image background, well away from all cells.
    Returns list of (x, y).
    """
    H, W = img_shape[:2]
    # Mark all cell regions (dilated) + a safety margin as "forbidden"
    forbidden = np.zeros((H, W), dtype=np.uint8)
    for cell in cells:
        mask = cell["label_mask"].astype(np.uint8)
        if mask.shape != (H, W):
            continue
        s  = safety_px * 2 + 1
        mk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s, s))
        forbidden = np.maximum(forbidden, cv2.dilate(mask, mk))

    bg_ys, bg_xs = np.where(forbidden == 0)
    if len(bg_xs) < n_neg:
        return []

    # Spread evenly across background
    step       = max(1, len(bg_xs) // (n_neg * 20))
    candidates = list(zip(bg_xs[::step], bg_ys[::step]))
    if len(candidates) < n_neg:
        return candidates
    idxs = np.linspace(0, len(candidates) - 1, n_neg, dtype=int)
    return [candidates[i] for i in idxs]


# ---------------------------------------------------------------------------
# Find anchor frames
# ---------------------------------------------------------------------------
def find_anchor_frames(frames_dir: Path, n_anchors: int, window: int,
                       min_area: int, max_cells: int, merge_px: int,
                       edge_margin: float = 0.15):
    frame_files = sorted(frames_dir.glob("*.jpg"))
    n = len(frame_files)

    if window <= 0:
        window = max(5, n // (n_anchors * 6))

    targets = [int(n * (i + 1) / (n_anchors + 1)) for i in range(n_anchors)]

    anchors = []
    for target in targets:
        lo = max(0, target - window)
        hi = min(n - 1, target + window)

        best_idx   = target
        best_score = -1
        for fi in range(lo, hi + 1):
            img = cv2.imread(str(frame_files[fi]))
            sc  = _cell_score(img, min_area, merge_px, edge_margin)
            if sc > best_score:
                best_score = sc
                best_idx   = fi

        img   = cv2.imread(str(frame_files[best_idx]))
        cells = detect_cells_in_frame(img, min_area, max_cells, merge_px, edge_margin)
        print(f"[INFO] Anchor frame {best_idx:>5}  (target {target:>5}, "
              f"window ±{window})  → {len(cells)} cell(s)  score={best_score}")
        anchors.append((best_idx, cells, img.shape))

    return anchors


# ---------------------------------------------------------------------------
# Debug image: all anchor frames side by side
# ---------------------------------------------------------------------------
def save_anchor_debug(frames_dir: Path, anchors: list, out_path: Path):
    imgs = []
    frame_files = sorted(frames_dir.glob("*.jpg"))
    for frame_idx, cells, _ in anchors:
        img = cv2.imread(str(frame_files[frame_idx]))
        for i, cell in enumerate(cells):
            colour      = CELL_COLOURS[i % len(CELL_COLOURS)]
            x1,y1,x2,y2 = cell["bbox"]
            cx, cy       = cell["centroid"]
            cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
            cv2.circle(img, (cx, cy), 6, colour, -1)
            cv2.putText(img, f"#{i+1}", (x1, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)
        cv2.putText(img, f"Anchor f{frame_idx}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        imgs.append(img)

    h_ref   = imgs[0].shape[0]
    resized = [cv2.resize(im, (int(im.shape[1] * h_ref / im.shape[0]), h_ref))
               for im in imgs]
    cv2.imwrite(str(out_path), np.hstack(resized))
    print(f"[INFO] Anchor debug → {out_path.name}")


# ---------------------------------------------------------------------------
# SAM2 prompting helpers
# ---------------------------------------------------------------------------
def _sample_positive_points(cell: dict, n_pos: int):
    """
    Returns n_pos points: 1 at the centroid + (n_pos-1) points near the inner
    edge of the cell, placed at evenly-spaced angles ~75% of the way from the
    centroid to the cell boundary. This covers the centre and the shape without
    hitting the extreme edge pixels.
    """
    cx, cy = cell["centroid"]
    pts    = [(cx, cy)]
    if n_pos <= 1:
        return pts

    mask = cell["orig_mask"]
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return pts

    n_rays = n_pos - 1
    angles = np.linspace(0, 2 * np.pi, n_rays, endpoint=False)

    for angle in angles:
        dx, dy = np.cos(angle), np.sin(angle)
        # Projection of each mask pixel onto this ray
        proj = (xs - cx) * dx + (ys - cy) * dy
        in_dir = proj > 0
        if not np.any(in_dir):
            pts.append((cx, cy))
            continue
        # Target: 75% of the way to the farthest pixel in this direction
        target = proj[in_dir].max() * 0.75
        best   = np.argmin(np.abs(proj[in_dir] - target))
        pts.append((int(xs[in_dir][best]), int(ys[in_dir][best])))

    return pts


def _add_prompts(predictor, state, anchors, n_pos: int, n_neg: int):
    for frame_idx, cells, img_shape in anchors:
        neg_pts = sample_negative_points(cells, img_shape, n_neg)

        for obj_id, cell in enumerate(cells, start=1):
            x1, y1, x2, y2 = cell["bbox"]

            pos_pts = _sample_positive_points(cell, n_pos)

            pts    = np.array(pos_pts + neg_pts, dtype=np.float32)
            labels = np.array([1] * len(pos_pts) + [0] * len(neg_pts),
                               dtype=np.int32)
            box    = np.array([x1, y1, x2, y2], dtype=np.float32)

            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=pts,
                labels=labels,
                box=box,
            )


def _run_pass(predictor, state, anchors, n_frames, score_thresh,
              reverse=False, label="fwd"):
    start  = anchors[0][0] if not reverse else anchors[-1][0]
    masks  = {}
    for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(
            state, start_frame_idx=start, reverse=reverse):
        masks[frame_idx] = {}
        for obj_id, logit in zip(obj_ids, mask_logits):
            score = torch.sigmoid(logit).max().item()
            if score >= score_thresh:
                binary = (logit.squeeze() > 0).cpu().numpy().astype(np.uint8)
                masks[frame_idx][int(obj_id)] = binary

        if (frame_idx + 1) % 50 == 0 or frame_idx == n_frames - 1 or frame_idx == 0:
            print(f"  [{label}] Frame {frame_idx+1:>4}/{n_frames}  "
                  f"cells tracked: {len(masks[frame_idx])}")
    return masks


# ---------------------------------------------------------------------------
# SAM2 video tracking — bidirectional
# ---------------------------------------------------------------------------
def track(frames_dir: Path, anchors: list, base_ckpt: str, model_cfg: str,
          weights: str, score_thresh: float, n_pos: int, n_neg: int, device: str):

    print(f"\n[INFO] Building SAM2 video predictor ({model_cfg}) ...")
    predictor = build_sam2_video_predictor(model_cfg, base_ckpt, device=device)

    if Path(weights).exists():
        print(f"[INFO] Loading fine-tuned weights: {weights}")
        ckpt = torch.load(weights, map_location=device)
        if "model_state_dict" in ckpt:
            ckpt = ckpt["model_state_dict"]
        elif "model" in ckpt:
            ckpt = ckpt["model"]
        # Filter out keys whose shape doesn't match the current model
        # (e.g. tiny→small image encoder dim mismatch); prompt encoder +
        # mask decoder use fixed 256-dim so they load fine across variants.
        model_sd  = predictor.state_dict()
        compatible = {k: v for k, v in ckpt.items()
                      if k in model_sd and model_sd[k].shape == v.shape}
        skipped    = len(ckpt) - len(compatible)
        predictor.load_state_dict(compatible, strict=False)
        print(f"[INFO] Loaded {len(compatible)}/{len(ckpt)} layers "
              f"({skipped} skipped — shape mismatch)")
    else:
        print(f"[WARN] Weights not found — using base SAM2.")

    n_frames = len(sorted(frames_dir.glob("*.jpg")))

    with torch.inference_mode(), \
         torch.autocast(device_type="cuda", dtype=torch.bfloat16):

        inf_state = predictor.init_state(video_path=str(frames_dir))

        # ── Forward pass ──────────────────────────────────────────────────
        print(f"\n[INFO] Forward pass ({n_frames} frames) ...")
        predictor.reset_state(inf_state)
        _add_prompts(predictor, inf_state, anchors, n_pos, n_neg)
        fwd = _run_pass(predictor, inf_state, anchors, n_frames,
                        score_thresh, reverse=False, label="fwd")

        # ── Backward pass ─────────────────────────────────────────────────
        print(f"\n[INFO] Backward pass ({n_frames} frames) ...")
        predictor.reset_state(inf_state)
        _add_prompts(predictor, inf_state, anchors, n_pos, n_neg)
        bwd = _run_pass(predictor, inf_state, anchors, n_frames,
                        score_thresh, reverse=True, label="bwd")

    # ── Merge by union ────────────────────────────────────────────────────
    all_masks = {}
    for fi in range(n_frames):
        f, b   = fwd.get(fi, {}), bwd.get(fi, {})
        merged = {}
        for oid in set(list(f.keys()) + list(b.keys())):
            if oid in f and oid in b:
                merged[oid] = np.maximum(f[oid], b[oid])
            elif oid in f:
                merged[oid] = f[oid]
            else:
                merged[oid] = b[oid]
        all_masks[fi] = merged

    return all_masks


# ---------------------------------------------------------------------------
# Render output video
# ---------------------------------------------------------------------------
def render_video(frames_dir: Path, all_masks: dict,
                 out_path: str, fps: float, alpha: float):
    frame_files = sorted(frames_dir.glob("*.jpg"))
    n           = len(frame_files)
    sample      = cv2.imread(str(frame_files[0]))
    h, w        = sample.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    print(f"\n[INFO] Rendering output video: {out_path}")

    for frame_idx, fpath in enumerate(frame_files):
        frame   = cv2.imread(str(fpath))
        overlay = frame.copy()

        for obj_id, mask in all_masks.get(frame_idx, {}).items():
            colour = CELL_COLOURS[(obj_id - 1) % len(CELL_COLOURS)]
            if mask.shape != (h, w):
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            overlay[mask > 0] = colour
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, contours, -1, colour, 2)

        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        n_cells = len(all_masks.get(frame_idx, {}))
        cv2.putText(frame, f"Frame {frame_idx+1}/{n}  Cells: {n_cells}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        writer.write(frame)

    writer.release()
    size_mb = Path(out_path).stat().st_size / 1024**2
    print(f"[INFO] Saved: {out_path}  ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args   = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}")

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"[ERROR] Video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    frames_dir = Path(args.frames_dir)
    out_dir    = Path(args.out_video).parent

    # 1. Extract frames
    n_frames, fps, w, h = extract_frames(str(video_path), frames_dir)
    print(f"[INFO] Video: {n_frames} frames, {w}x{h} @ {fps:.1f}fps")

    # 2. Find anchor frames with best cell visibility
    print(f"\n[INFO] Selecting {args.n_anchors} anchor frames "
          f"(merge_px={args.merge_px}) ...")
    anchors = find_anchor_frames(frames_dir, args.n_anchors, args.anchor_window,
                                 args.min_cell_px, args.max_cells, args.merge_px,
                                 args.edge_margin)

    if not any(cells for _, cells, _ in anchors):
        print("[ERROR] No cells detected. Lower --min_cell_px or --merge_px.",
              file=sys.stderr)
        sys.exit(1)

    save_anchor_debug(frames_dir, anchors, out_dir / "anchor_detections.jpg")

    # 3. Bidirectional tracking
    all_masks = track(frames_dir, anchors,
                      args.base_ckpt, args.model_cfg, args.checkpoint,
                      args.score_thresh, args.n_pos, args.n_neg, device)

    # 4. Render
    render_video(frames_dir, all_masks, args.out_video, fps, args.alpha)

    print("\n[DONE] Download results:")
    print(f"  scp Discovery:~/ViT_Partision/{Path(args.out_video).name} ./")
    print(f"  scp Discovery:~/ViT_Partision/anchor_detections.jpg ./")


if __name__ == "__main__":
    main()
