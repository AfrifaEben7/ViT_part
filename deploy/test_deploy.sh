#!/usr/bin/env bash
# =============================================================================
# test_deploy.sh — Quick end-to-end smoke test for the Jetson deployment.
#
# Runs inference on 3 sample macrophage images from sample_data/ and reports
# the IoU score for each.  Verifies the whole stack works before deploying
# to a real microscopy video.
#
# Usage:
#   chmod +x test_deploy.sh && ./test_deploy.sh
# =============================================================================
set -e

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DEPLOY_DIR="$REPO_DIR/deploy"
SAMPLE_DIR="$DEPLOY_DIR/sample_data"
OUT_DIR="$REPO_DIR/outputs/test_jetson"

# Use jetson venv if present, fall back to main venv
if [ -d "$REPO_DIR/sam2_env_jetson" ]; then
    source "$REPO_DIR/sam2_env_jetson/bin/activate"
else
    source "$REPO_DIR/sam2_env/bin/activate"
fi

cd "$REPO_DIR"

# Copy 512 config where Hydra can find it
cp "$DEPLOY_DIR/configs/sam2_hiera_t_512.yaml" \
   "$REPO_DIR/segment-anything-2/sam2/configs/sam2/sam2_hiera_t_512.yaml" 2>/dev/null || true

mkdir -p "$OUT_DIR"

echo "================================================"
echo " SAM2-tiny Jetson Deployment Test"
echo "================================================"
echo ""

# Pick best available weights: jetson > tiny_v1 > base
WEIGHTS=""
for w in \
    "$REPO_DIR/weights/tiny/sam2_macrophage_jetson_best.pt" \
    "$REPO_DIR/weights/tiny/sam2_macrophage_best.pt"; do
    if [ -f "$w" ]; then WEIGHTS="$w"; break; fi
done

if [ -z "$WEIGHTS" ]; then
    echo "[ERROR] No fine-tuned weights found. Run train_jetson.sh first."
    exit 1
fi

# Pick config: 512 if weights were trained at 512, else 1024
if echo "$WEIGHTS" | grep -q "jetson"; then
    CFG="sam2_hiera_t_512.yaml"
else
    CFG="sam2_hiera_t.yaml"
fi

echo "Weights : $WEIGHTS"
echo "Config  : $CFG"
echo ""

PASS=0; FAIL=0

for img in "$SAMPLE_DIR"/*.jpg; do
    echo "--- $(basename $img) ---"
    python3 "$DEPLOY_DIR/infer_jetson.py" \
        --mode       image \
        --input      "$img" \
        --out        "$OUT_DIR" \
        --weights    "$WEIGHTS" \
        --base_ckpt  "$REPO_DIR/checkpoints/sam2_hiera_tiny.pt" \
        --model_cfg  "$CFG" \
        --max_cells  1 \
        --min_cell_px 100 \
    && PASS=$((PASS+1)) || FAIL=$((FAIL+1))
    echo ""
done

echo "================================================"
echo " Results: $PASS passed  /  $FAIL failed"
echo " Output images: $OUT_DIR"
echo "================================================"

if [ $FAIL -gt 0 ]; then exit 1; fi
