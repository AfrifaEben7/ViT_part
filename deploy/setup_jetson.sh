#!/usr/bin/env bash
# =============================================================================
# setup_jetson.sh — One-time bootstrap for SAM2 macrophage on Jetson Orin Nano
#
# Tested on: JetPack 6.x  (Ubuntu 22.04, Python 3.10, CUDA 12.2, Ampere GPU)
#
# Usage:
#   chmod +x setup_jetson.sh && ./setup_jetson.sh
# =============================================================================
set -e

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"   # ViT_Partision root
ENV_DIR="$REPO_DIR/sam2_env_jetson"
SAM2_REPO="$REPO_DIR/segment-anything-2"
CKPT_DIR="$REPO_DIR/checkpoints"

echo "=== [1/6] Creating Python venv ==="
python3 -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"

echo "=== [2/6] Installing PyTorch for JetPack 6.x (CUDA 12.2) ==="
# NVIDIA Jetson wheel — ARM64, Python 3.10, CUDA 12.2
# If this URL breaks, find the latest at:
#   https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
pip install --upgrade pip wheel
pip install \
  "https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl" \
  --no-cache-dir

# torchvision — build from source (no ARM wheel available)
echo "=== [2b/6] Building torchvision from source (takes ~10 min) ==="
pip install --no-cache-dir \
  "https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl" \
  || {
    echo "Pre-built wheel not found — compiling from source..."
    git clone --depth 1 --branch v0.18.0 https://github.com/pytorch/vision /tmp/torchvision
    cd /tmp/torchvision && pip install -e . && cd -
  }

echo "=== [3/6] Installing Python dependencies ==="
pip install --no-cache-dir \
  "numpy<2" \
  hydra-core>=1.3.2 \
  "monai>=1.3" \
  pycocotools \
  opencv-python-headless \
  albumentations \
  tifffile \
  scipy \
  roboflow \
  matplotlib

echo "=== [4/6] Installing SAM2 ==="
if [ ! -d "$SAM2_REPO" ]; then
  git clone https://github.com/facebookresearch/segment-anything-2.git "$SAM2_REPO"
fi
cd "$SAM2_REPO"
# Use non-editable install: SAM2's setup.py backend lacks the build_editable
# hook required by pip 26+ (PEP 660), so -e fails. Non-editable is fine for deployment.
pip install ".[demo]" --no-build-isolation
cd "$REPO_DIR"

# SAM2's [demo] extras may pull in a newer torch (e.g. cu130) that is
# incompatible with the Jetson driver (CUDA 12.x).  Re-pin the Jetson wheel.
echo "=== [4b/6] Re-pinning Jetson PyTorch (CUDA 12.2) ==="
JETSON_TORCH="https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl"
pip install "$JETSON_TORCH" --no-cache-dir --force-reinstall \
  --no-deps   # keep SAM2's other deps intact; only swap torch itself
# Also re-pin torchvision — SAM2 [demo] upgrades it to match torch 2.11.
JETSON_TV="https://github.com/ultralytics/assets/releases/download/v0.0.0/torchvision-0.18.0a0+6043bc2-cp310-cp310-linux_aarch64.whl"
pip install "$JETSON_TV" --no-cache-dir --force-reinstall --no-deps

# torch 2.5 (Jetson) links against libcusparseLt.so.0 which lives inside the
# nvidia-cusparselt PyPI package but is NOT auto-added to LD_LIBRARY_PATH.
# Patch the venv activate script so every `source activate` sets it correctly.
echo "=== [4c/6] Patching venv activate for libcusparseLt ==="
ACTIVATE="$ENV_DIR/bin/activate"
grep -q "cusparselt" "$ACTIVATE" || cat >> "$ACTIVATE" << 'ACTIVATE_PATCH'

# --- cusparseLt fix for Jetson torch 2.5 (added by setup_jetson.sh) ---
_CUSPARSELT_DIR="$(dirname "${BASH_SOURCE[0]}")/../lib/python3.10/site-packages/nvidia/cusparselt/lib"
if [ -d "$_CUSPARSELT_DIR" ]; then
    export LD_LIBRARY_PATH="$(realpath "$_CUSPARSELT_DIR"):${LD_LIBRARY_PATH:-}"
fi
ACTIVATE_PATCH

echo "=== [5/6] Downloading SAM2-tiny checkpoint ==="
mkdir -p "$CKPT_DIR"
if [ ! -f "$CKPT_DIR/sam2_hiera_tiny.pt" ]; then
  wget -q --show-progress \
    -O "$CKPT_DIR/sam2_hiera_tiny.pt" \
    https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt
fi

echo "=== [6/6] Sanity check ==="
python3 - <<'PY'
import torch, sam2
print("PyTorch :", torch.__version__)
print("CUDA    :", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NOT FOUND")
print("SAM2    : OK")
PY

echo ""
echo "=== Setup complete ==="
echo "Activate with: source $ENV_DIR/bin/activate"
