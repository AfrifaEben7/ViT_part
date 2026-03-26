"""
train.py — Fine-tune SAM2 (sam2_hiera_tiny) for macrophage segmentation.

Constraints:
  - Image encoder frozen; only prompt encoder + mask decoder are trained.
  - Model runs in bfloat16 (float16 fallback on older GPUs).
  - Input resolution: 1024 x 1024.
  - Loss: MONAI DiceLoss + FocalLoss.
  - Optimizer: AdamW, lr=1e-5.
  - Output: sam2_macrophage_finetuned.pt

Usage:
    python3 train.py [--data_root PATH] [--epochs N] ...
"""

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

import albumentations as A
import cv2
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

from monai.losses import DiceLoss, FocalLoss
from pycocotools import mask as coco_mask

# SAM2 — installed via `pip install -e .` from the cloned repo.
# If Hydra config resolution fails, we chdir to the repo root before loading.
try:
    from sam2.build_sam import build_sam2
except ImportError:
    print("[ERROR] sam2 not found. Run setup_env.sh first.", file=sys.stderr)
    sys.exit(1)

# ---------------------------------------------------------------------------
# ImageNet normalisation constants (SAM2 uses these)
# ---------------------------------------------------------------------------
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ===========================================================================
# Dataset
# ===========================================================================

class MacrophageDataset(Dataset):
    """
    Reads the Roboflow SAM2-format export, which stores images and COCO-RLE
    annotations as sibling files in a flat directory:

        {root}/{split}/{stem}.jpg   — RGB image
        {root}/{split}/{stem}.json  — COCO annotation with RLE masks

    The JSON structure (per Roboflow SAM2 export):
        {
          "image": {"height": H, "width": W, ...},
          "annotations": [
            {"segmentation": {"counts": "<RLE>", "size": [H, W]}}, ...
          ]
        }

    All annotation masks are merged into a single binary mask.

    Returns per sample:
        image_tensor  : (3, H, W)  float32, ImageNet-normalised
        mask_tensor   : (1, H, W)  float32, binary {0, 1}
        point_coords  : (1, 1, 2)  float32, one positive (x, y) from mask interior
        point_labels  : (1, 1)     int32,   1 = foreground; 0 = fallback (empty mask)
    """

    def __init__(self, root_dir: str, split: str = "train", image_size: int = 1024,
                 n_points: int = 1, augment: bool = False):
        self.image_size = image_size
        self.n_points   = n_points
        split_dir = Path(root_dir) / split

        # Augmentation pipeline — geometric ops applied to image+mask jointly;
        # photometric ops applied to image only (albumentations handles this).
        if augment:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                                   rotate_limit=15,
                                   border_mode=cv2.BORDER_CONSTANT, p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2,
                                           contrast_limit=0.2, p=0.5),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20,
                                     val_shift_limit=10, p=0.3),
                A.GaussianBlur(blur_limit=(3, 7), p=0.3),
                A.GaussNoise(p=0.2),
            ])
        else:
            self.transform = None

        if not split_dir.exists():
            raise FileNotFoundError(f"Split directory not found: {split_dir}")

        exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
        all_images = sorted([p for p in split_dir.iterdir() if p.suffix in exts])

        self.samples = []
        skipped = 0
        for img_path in all_images:
            json_path = img_path.with_suffix(".json")
            if not json_path.exists():
                skipped += 1
                continue
            self.samples.append((img_path, json_path))

        if skipped:
            print(f"[WARN] {split}: skipped {skipped} images with no matching JSON.")
        print(f"[INFO] {split}: {len(self.samples)} image-annotation pairs loaded.")

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid samples found in {split_dir}. "
                "Check that .json annotation files exist alongside the images."
            )

    @staticmethod
    def _decode_rle_to_mask(annotation: dict) -> np.ndarray:
        """Decode a single COCO compressed-RLE annotation to a binary uint8 mask."""
        seg = annotation["segmentation"]
        counts = seg["counts"]
        # pycocotools expects bytes for compressed RLE
        if isinstance(counts, str):
            counts = counts.encode()
        rle = {"counts": counts, "size": seg["size"]}
        return coco_mask.decode(rle).astype(np.uint8)  # (H, W) uint8

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, json_path = self.samples[idx]

        # ---- Load image --------------------------------------------------
        image = cv2.imread(str(img_path))
        if image is None:
            raise IOError(f"Failed to read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = image.shape[:2]

        # ---- Build merged binary mask from COCO-RLE annotations ----------
        with open(json_path) as f:
            ann_data = json.load(f)

        annotations = ann_data.get("annotations", [])
        if annotations:
            merged = np.zeros((orig_h, orig_w), dtype=np.uint8)
            for ann in annotations:
                m = self._decode_rle_to_mask(ann)
                merged = np.maximum(merged, m)
        else:
            merged = np.zeros((orig_h, orig_w), dtype=np.uint8)

        # ---- Augment image + mask jointly (before resize) ----------------
        if self.transform is not None:
            out    = self.transform(image=image, mask=merged)
            image  = out["image"]
            merged = out["mask"]

        # ---- Resize to model input size ----------------------------------
        image  = cv2.resize(image,  (self.image_size, self.image_size),
                            interpolation=cv2.INTER_LINEAR)
        merged = cv2.resize(merged, (self.image_size, self.image_size),
                            interpolation=cv2.INTER_NEAREST)

        # ---- Normalise and convert to tensors ----------------------------
        image = image.astype(np.float32) / 255.0
        image = (image - IMAGENET_MEAN) / IMAGENET_STD
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)  # (3,H,W)
        mask_tensor  = torch.from_numpy(merged.astype(np.float32)).unsqueeze(0)  # (1,H,W)

        # ---- Sample point prompts ----------------------------------------
        # torch.where returns (rows, cols) → swap to (x=col, y=row) for SAM2
        fg_ys, fg_xs = torch.where(mask_tensor[0] > 0.5)

        if fg_xs.numel() > 0:
            n = min(self.n_points, fg_xs.numel())
            picks = random.sample(range(fg_xs.numel()), n)
            pts    = [[float(fg_xs[i]), float(fg_ys[i])] for i in picks]
            labels = [1] * n
        else:
            pts    = [[float(self.image_size // 2), float(self.image_size // 2)]]
            labels = [0]

        point_coords = torch.tensor([pts],    dtype=torch.float32)  # (1, N, 2)
        point_labels = torch.tensor([labels], dtype=torch.int32)    # (1, N)

        return image_tensor, mask_tensor, point_coords, point_labels


# ===========================================================================
# Argument parser
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune SAM2 hiera_tiny for macrophage segmentation.")
    p.add_argument("--data_root",   default="./microphage-4",
                   help="Path to Roboflow dataset root")
    p.add_argument("--checkpoint",  default="./checkpoints/sam2_hiera_tiny.pt",
                   help="Path to SAM2 hiera_tiny checkpoint")
    p.add_argument("--model_cfg",   default="sam2_hiera_t.yaml",
                   help="SAM2 model config filename (looked up relative to sam2 package)")
    p.add_argument("--epochs",        type=int,   default=30)
    p.add_argument("--batch_size",    type=int,   default=2,
                   help="Per-GPU batch size. Reduce to 1 if OOM.")
    p.add_argument("--lr",            type=float, default=1e-5,
                   help="Peak learning rate after warmup")
    p.add_argument("--lr_min",        type=float, default=1e-6,
                   help="Minimum LR at end of cosine decay")
    p.add_argument("--warmup_steps",  type=int,   default=50,
                   help="Linear LR warmup steps from lr_min to lr")
    p.add_argument("--n_points",      type=int,   default=3,
                   help="Number of positive point prompts sampled per image")
    p.add_argument("--resume",        default="./sam2_macrophage_best.pt",
                   help="Resume fine-tuning from these weights ('' to disable)")
    p.add_argument("--num_workers",   type=int,   default=4)
    p.add_argument("--image_size",    type=int,   default=1024,
                   help="Input resolution (use 512 for Jetson to save memory)")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--output",        default="sam2_macrophage_v2_finetuned.pt",
                   help="Path for saved fine-tuned weights")
    return p.parse_args()


# ===========================================================================
# Helpers
# ===========================================================================

def load_model(model_cfg: str, checkpoint: str, device: str, dtype: torch.dtype):
    """
    Build SAM2 model.

    Hydra's pkg://sam2 loader only sees files registered in the package
    manifest at install time — custom configs (e.g. sam2_hiera_t_512.yaml)
    are not picked up.  We re-initialise Hydra with initialize_config_dir
    pointing at the actual configs folder on disk, which finds any .yaml
    in that directory.
    """
    import sam2 as _sam2_pkg
    from hydra import initialize_config_dir, compose
    from hydra.core.global_hydra import GlobalHydra

    checkpoint = str(Path(checkpoint).resolve())

    # Locate the installed sam2 package's configs/sam2/ directory
    sam2_pkg_dir  = Path(_sam2_pkg.__file__).parent
    configs_dir   = sam2_pkg_dir / "configs" / "sam2"

    # Strip .yaml suffix if present — Hydra config names don't include it
    cfg_name = model_cfg.removesuffix(".yaml")

    # Re-initialise Hydra with a filesystem path so custom configs are visible
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(configs_dir), version_base="1.2"):
        cfg = compose(config_name=cfg_name)

    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    OmegaConf.resolve(cfg)
    sam2_model = instantiate(cfg.model, _recursive_=True)
    state_dict = torch.load(checkpoint, map_location=device)
    # SAM2 checkpoints are sometimes wrapped
    if "model" in state_dict:
        state_dict = state_dict["model"]
    sam2_model.load_state_dict(state_dict, strict=False)
    sam2_model = sam2_model.to(device)
    sam2_model.eval()

    # Do NOT cast to bfloat16 here — SAM2's position encoding buffers have
    # internal float32 casts that break under a whole-model dtype conversion.
    # Use torch.autocast in the forward pass instead.

    return sam2_model


def count_params(model):
    total   = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


# ===========================================================================
# Training forward pass
# ===========================================================================

def forward_pass(sam2_model, images, gt_masks, point_coords, point_labels,
                 dtype, device, dice_loss_fn, focal_loss_fn):
    """
    Full SAM2 forward pass for one batch with gradient tracking only over
    the prompt encoder and mask decoder.

    Returns:
        loss        : scalar tensor (with grad)
        pred_masks  : (B, 1, 1024, 1024) float, raw logits upsampled
    """
    B = images.shape[0]

    # Use autocast for bfloat16/float16 mixed precision.
    # We do NOT cast the whole model to bfloat16 because SAM2's position
    # encoding has internal .to(torch.float) calls that break under a
    # whole-model dtype conversion. autocast handles it correctly.
    with torch.autocast(device_type="cuda", dtype=dtype):

        # 1. Image encoding — NO GRAD (encoder is frozen)
        with torch.no_grad():
            backbone_out = sam2_model.forward_image(images)
            # _prepare_backbone_features returns (backbone_out, vision_feats,
            # vision_pos_embeds, feat_sizes); use the 4th value for shapes.
            _, vision_feats, _, feat_sizes = sam2_model._prepare_backbone_features(backbone_out)

        # 2. Reshape (H*W, B, C) → (B, C, H, W) per feature level
        feats = []
        for feat, (H, W) in zip(vision_feats, feat_sizes):
            C = feat.shape[2]
            feats.append(feat.permute(1, 2, 0).reshape(B, C, H, W))

        # 3. Prompt encoding — gradients flow here
        coords = point_coords.squeeze(1)   # (B, 1, 2)
        labels = point_labels.squeeze(1)   # (B, 1)

        sparse_emb, dense_emb = sam2_model.sam_prompt_encoder(
            points=(coords, labels),
            boxes=None,
            masks=None,
        )

        # 4. Mask decoding — gradients flow here
        low_res_masks, iou_preds, _, _ = sam2_model.sam_mask_decoder(
            image_embeddings=feats[-1],
            image_pe=sam2_model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
            repeat_image=False,
            high_res_features=feats[:-1],
        )

        # 5. Upsample to input image size
        img_size = images.shape[-1]  # works for 512 and 1024
        pred_masks = F.interpolate(
            low_res_masks.float(),
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False,
        )

        # 6. Loss
        loss = dice_loss_fn(pred_masks, gt_masks) + focal_loss_fn(pred_masks, gt_masks)

    return loss, pred_masks


# ===========================================================================
# Main
# ===========================================================================

def main():
    args = parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Device
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for SAM2 fine-tuning. "
            "Submit via Slurm on a GPU node."
        )
    device = "cuda"

    # Dtype — prefer bfloat16 (wider dynamic range), fall back to float16
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"[INFO] Device: {torch.cuda.get_device_name(0)}")
    print(f"[INFO] Training dtype: {dtype}")

    # Resolve paths
    data_root  = Path(args.data_root).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    output     = Path(args.output).resolve()

    if not data_root.exists():
        # Hint: show what's in the parent directory
        parent = data_root.parent
        print(f"[ERROR] data_root not found: {data_root}", file=sys.stderr)
        if parent.exists():
            candidates = [p.name for p in parent.iterdir() if p.is_dir()]
            print(f"        Directories in {parent}: {candidates}", file=sys.stderr)
        sys.exit(1)

    if not checkpoint.exists():
        print(f"[ERROR] Checkpoint not found: {checkpoint}", file=sys.stderr)
        print("        Run setup_env.sh to download it.", file=sys.stderr)
        sys.exit(1)

    # ---- Datasets & DataLoaders -------------------------------------------
    print(f"\n[INFO] Loading dataset from: {data_root}  (n_points={args.n_points})")
    train_dataset = MacrophageDataset(str(data_root), split="train", image_size=args.image_size,
                                      n_points=args.n_points, augment=True)
    val_dataset   = MacrophageDataset(str(data_root), split="valid", image_size=args.image_size,
                                      n_points=1, augment=False)  # no aug for consistent val

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # ---- Model setup -------------------------------------------------------
    print(f"\n[INFO] Loading SAM2 model: {checkpoint}")
    sam2_model = load_model(args.model_cfg, str(checkpoint), device, dtype)
    sam2_model.train()

    # Freeze image encoder entirely
    for param in sam2_model.image_encoder.parameters():
        param.requires_grad = False
    sam2_model.image_encoder.eval()

    total, trainable = count_params(sam2_model)
    print(f"[INFO] Total params:     {total:,}")
    print(f"[INFO] Trainable params: {trainable:,}  "
          f"({100.0 * trainable / total:.1f}%)  — prompt encoder + mask decoder only")

    # ---- Resume from previous fine-tuned weights --------------------------
    if args.resume and Path(args.resume).exists():
        print(f"[INFO] Resuming from: {args.resume}")
        state = torch.load(args.resume, map_location=device)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        sam2_model.load_state_dict(state)
    elif args.resume:
        print(f"[WARN] --resume path not found ({args.resume}), starting from base checkpoint.")

    # ---- Optimizer ---------------------------------------------------------
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, sam2_model.parameters()),
        lr=args.lr_min,        # start at lr_min; warmup brings it up to args.lr
        weight_decay=1e-4,
    )

    # ---- LR schedule: linear warmup → cosine decay to lr_min -------------
    total_steps   = args.epochs * len(train_loader)
    warmup_steps  = args.warmup_steps

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            # Linear warmup: lr_min → lr
            return (args.lr_min + (args.lr - args.lr_min) * step / warmup_steps) / args.lr
        # Cosine decay: lr → lr_min
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cos_val  = 0.5 * (1.0 + math.cos(math.pi * progress))
        return (args.lr_min + (args.lr - args.lr_min) * cos_val) / args.lr

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    global_step = 0

    # ---- Loss functions ----------------------------------------------------
    dice_loss_fn  = DiceLoss(sigmoid=True, reduction="mean")
    focal_loss_fn = FocalLoss(reduction="mean")

    # ---- Training loop -----------------------------------------------------
    print(f"\n[INFO] Starting training: {args.epochs} epochs, "
          f"batch_size={args.batch_size}, lr={args.lr_min}→{args.lr}→{args.lr_min} "
          f"(warmup={warmup_steps} steps, cosine decay), n_points={args.n_points}\n")

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # Keep image encoder in eval mode every epoch (defensive)
        sam2_model.sam_prompt_encoder.train()
        sam2_model.sam_mask_decoder.train()
        sam2_model.image_encoder.eval()

        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, (images, gt_masks, point_coords, point_labels) in enumerate(train_loader):
            images       = images.to(device)                          # float32
            gt_masks     = gt_masks.to(device, dtype=torch.float32)
            point_coords = point_coords.to(device, dtype=torch.float32)
            point_labels = point_labels.to(device, dtype=torch.int32)

            loss, _ = forward_pass(
                sam2_model, images, gt_masks, point_coords, point_labels,
                dtype, device, dice_loss_fn, focal_loss_fn,
            )

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(
                filter(lambda p: p.requires_grad, sam2_model.parameters()),
                max_norm=1.0,
            )
            optimizer.step()
            scheduler.step()
            global_step += 1

            epoch_loss += loss.item()

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                elapsed = time.time() - t0
                avg = epoch_loss / (batch_idx + 1)
                cur_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"  Epoch [{epoch:>3}/{args.epochs}] "
                    f"Step [{batch_idx+1:>4}/{len(train_loader)}]  "
                    f"Loss: {loss.item():.4f}  Avg: {avg:.4f}  "
                    f"LR: {cur_lr:.2e}  Time: {elapsed:.1f}s"
                )

        train_loss = epoch_loss / len(train_loader)

        # ---- Validation ---------------------------------------------------
        sam2_model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, gt_masks, point_coords, point_labels in val_loader:
                images       = images.to(device)
                gt_masks     = gt_masks.to(device, dtype=torch.float32)
                point_coords = point_coords.to(device, dtype=torch.float32)
                point_labels = point_labels.to(device, dtype=torch.int32)

                loss, _ = forward_pass(
                    sam2_model, images, gt_masks, point_coords, point_labels,
                    dtype, device, dice_loss_fn, focal_loss_fn,
                )
                val_loss += loss.item()
        val_loss /= max(len(val_loader), 1)

        cur_lr = optimizer.param_groups[0]["lr"]
        print(
            f"\nEpoch {epoch:>3}/{args.epochs} Summary — "
            f"Train Loss: {train_loss:.4f}  Val Loss: {val_loss:.4f}  "
            f"LR: {cur_lr:.2e}\n"
        )

        # ---- Periodic checkpoint save ------------------------------------
        if epoch % 5 == 0 or epoch == args.epochs:
            ckpt_path = output.parent / f"checkpoint_epoch{epoch:03d}.pt"
            torch.save(
                {
                    "epoch":                epoch,
                    "model_state_dict":     sam2_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss":           train_loss,
                    "val_loss":             val_loss,
                },
                ckpt_path,
            )
            print(f"[CKPT] Saved: {ckpt_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = output.parent / (output.stem.replace("finetuned", "best") + ".pt")
            torch.save(sam2_model.state_dict(), best_path)
            print(f"[BEST] New best val loss {val_loss:.4f} — saved: {best_path}")

        # Resume train mode for next epoch
        sam2_model.train()
        sam2_model.image_encoder.eval()

    # ---- Final save (weights only — clean for inference) ------------------
    torch.save(sam2_model.state_dict(), output)
    print(f"\n[DONE] Final weights saved to: {output}")
    print(f"       Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
