#!/usr/bin/env bash
# =============================================================================
# train_jetson.sh — Fine-tune SAM2-tiny on Jetson Orin Nano
#
# Memory budget (8 GB unified):
#   OS + idle  : ~2 GB
#   Model      : ~0.6 GB (float16)
#   Activations: ~2.5 GB (512×512, batch=1)
#   Gradients  : ~0.3 GB (prompt enc + mask dec only)
#   Safe margin: ~2.6 GB free
#
# With 1024×1024 the activation footprint doubles (~5 GB) → high OOM risk.
# Use 512×512 (--image_size 512 + 512-config) unless you have 8 GB free.
# =============================================================================
set -e

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO_DIR/sam2_env_jetson/bin/activate"
cd "$REPO_DIR"

# Copy the 512 config where SAM2/Hydra can find it.
# SAM2 is installed as a package, so Hydra resolves pkg://sam2 to the
# site-packages directory — copy there, plus the cloned repo for fallback.
CFG_SRC="$REPO_DIR/deploy/configs/sam2_hiera_t_512.yaml"
SAM2_PKG=$(source "$REPO_DIR/sam2_env_jetson/bin/activate" && \
           python3 -c "import sam2, os; print(os.path.dirname(sam2.__file__))" 2>/dev/null)
if [ -n "$SAM2_PKG" ]; then
    mkdir -p "$SAM2_PKG/configs/sam2"
    cp "$CFG_SRC" "$SAM2_PKG/configs/sam2/sam2_hiera_t_512.yaml"
    echo "[INFO] Config copied to installed package: $SAM2_PKG/configs/sam2/"
fi
# Also copy to cloned repo (fallback path used by load_model)
CFG_DST="$REPO_DIR/segment-anything-2/sam2/configs/sam2/sam2_hiera_t_512.yaml"
mkdir -p "$(dirname "$CFG_DST")"
cp "$CFG_SRC" "$CFG_DST"


echo "=== [pre] Checking dataset ==="
DATA_DIR="$REPO_DIR/microphage-4"
if [ ! -d "$DATA_DIR/train" ]; then
    echo "[INFO] Dataset not found — downloading from Roboflow..."
    python3 "$REPO_DIR/deploy/src/get_data.py"
else
    n=$(ls "$DATA_DIR/train"/*.jpg 2>/dev/null | wc -l)
    echo "[INFO] Dataset found: $n training images"
fi

echo ""
echo "=== Fine-tuning SAM2-tiny @ 512×512 on Jetson ==="
python3 "$REPO_DIR/deploy/src/train.py" \
  --data_root    "$REPO_DIR/microphage-4" \
  --checkpoint   "$REPO_DIR/checkpoints/sam2_hiera_tiny.pt" \
  --model_cfg    sam2_hiera_t_512.yaml \
  --image_size   512 \
  --epochs       20 \
  --batch_size   1 \
  --lr           1e-5 \
  --lr_min       1e-6 \
  --warmup_steps 30 \
  --n_points     3 \
  --num_workers  2 \
  --output       "$REPO_DIR/weights/tiny/sam2_macrophage_jetson_best.pt"

echo ""
echo "=== Done. Weights saved to weights/tiny/sam2_macrophage_jetson_best.pt ==="
