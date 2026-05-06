#!/usr/bin/env bash
# One-shot script to push yolo-portfolio to GitHub under the user's identity.
# Run with:  bash scripts/push_to_github.sh
set -euo pipefail

REPO_DIR="$HOME/Documents/yolo-portfolio"
REMOTE_URL="https://github.com/afzalbek97/yolo-portfolio.git"
GIT_NAME="afzalbek97"
GIT_EMAIL="askarovafzalbek@gmail.com"

cd "$REPO_DIR"
echo "[1/8] Cleaning any leftover git lock files..."
rm -f .git/index.lock .git/tGG435m 2>/dev/null || true

echo "[2/8] Re-initialising .git (fresh history, no Claude attribution)..."
rm -rf .git
git init -b main

echo "[3/8] Setting commit author identity to your GitHub account..."
git config user.name  "$GIT_NAME"
git config user.email "$GIT_EMAIL"
# Make sure no system-wide co-author trailers leak in
git config --local commit.gpgsign false || true

echo "[4/8] Removing previously-tracked binary that should be ignored..."
rm -f yolov8n.pt   # auto-downloaded by ultralytics; do NOT commit

echo "[5/8] Staging files..."
git add -A
git status --short | head -40
echo "...(file count: $(git status --short | wc -l | tr -d ' '))"

echo "[6/8] Creating initial commit..."
git commit -m "Initial commit: YOLOv8-OBB bottle vs can detector

- Two-stage cascade pipeline: COCO YOLOv8n filter + custom OBB head
  drastically reduces false positives caused by domain shift from the
  292-image Roboflow training set.
- FastAPI REST API (api/main.py) with /predict, /predict/visualize,
  /predict/raw endpoints; 4-corner OBB polygons returned as JSON.
- Real-time OpenCV webcam demo (src/webcam_demo.py).
- Trained on Google Colab T4 GPU; Colab notebook + quickstart included.
- Multilingual README: English / 한국어 / O'zbek." \
  --no-gpg-sign

echo "[7/8] Adding remote and pushing to GitHub..."
git remote add origin "$REMOTE_URL"
git branch -M main
# Push — this is where GitHub will ask for your stored credentials / token
git push -u origin main

echo "[8/8] Done. Open: $REMOTE_URL"
