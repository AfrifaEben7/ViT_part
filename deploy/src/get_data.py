"""
get_data.py — Download the macrophage segmentation dataset from Roboflow.

Usage:
    python3 get_data.py

Run from ~/ViT_Partision/ with sam2_env activated.
"""

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Roboflow download (exact configuration — do not modify)
# ---------------------------------------------------------------------------
from roboflow import Roboflow

rf = Roboflow(api_key="eCmKTW2qAhPqgU8vXej8")
project = rf.workspace("ebenezer-klxtj").project("microphage-quwej")
version = project.version(4)
dataset = version.download("sam2")

# ---------------------------------------------------------------------------
# Post-download verification
# ---------------------------------------------------------------------------
cwd = Path.cwd()

# Roboflow SDK names the folder {project-slug}-{version}; glob to be safe.
# The actual name may be shortened (e.g. "microphage-4" not "microphage-quwej-4").
matches = sorted(
    p for p in cwd.iterdir()
    if p.is_dir() and "microphage" in p.name.lower()
)
if not matches:
    print(
        "\n[ERROR] Could not find downloaded dataset directory. "
        "Contents of current directory:",
        file=sys.stderr,
    )
    for p in sorted(cwd.iterdir()):
        print(f"  {p.name}", file=sys.stderr)
    sys.exit(1)

dataset_path = matches[0]
print(f"\nDataset directory: {dataset_path.resolve()}")

if len(matches) > 1:
    print(f"[WARNING] Multiple matching directories found: {[m.name for m in matches]}")
    print(f"          Using: {dataset_path.name}")

assert dataset_path.exists(), f"Dataset path does not exist: {dataset_path}"

# ---------------------------------------------------------------------------
# Check expected subdirectory structure
# Roboflow SAM2 format: flat {split}/{stem}.jpg + {stem}.json pairs
# ---------------------------------------------------------------------------
print("\nChecking dataset structure...")

exts = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}
all_ok = True

for split in ("train", "valid", "test"):
    split_dir = dataset_path / split
    if not split_dir.exists():
        if split == "test":
            continue   # test split is optional
        print(f"  [WARN] {split}/ not found")
        all_ok = False
        continue

    images = [f for f in split_dir.iterdir() if f.suffix in exts]
    jsons  = [f for f in split_dir.iterdir() if f.suffix == ".json"]
    paired = sum(1 for img in images if img.with_suffix(".json").exists())
    print(f"  [OK]   {split}/   {len(images)} images, {len(jsons)} JSONs, "
          f"{paired} paired")

    if split == "train" and paired == 0:
        print("         [ERROR] No image-JSON pairs found in train/")
        all_ok = False

if not all_ok:
    print(
        "\n[ERROR] Dataset structure is missing required splits.",
        file=sys.stderr,
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Count training images
# ---------------------------------------------------------------------------
train_dir = dataset_path / "train"
if train_dir.exists():
    train_images = [f for f in train_dir.iterdir() if f.suffix in exts]
    print(f"\nTraining images found: {len(train_images)}")
    if len(train_images) == 0:
        print("[WARNING] No training images found — check the download.")

# ---------------------------------------------------------------------------
# Print final path for use in submit.slurm / train.py
# ---------------------------------------------------------------------------
print(f"\n[DONE] Dataset ready at:")
print(f"       {dataset_path.resolve()}")
print(f"\nPass this path to train.py with:")
print(f"  --data_root {dataset_path.resolve()}")
