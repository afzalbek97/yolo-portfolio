#!/usr/bin/env bash
# One-step setup for the FastAPI environment on macOS (Apple Silicon / Intel)
# Usage: bash setup_mac.sh

set -e

echo "=========================================="
echo "  YOLO Portfolio — macOS Setup"
echo "=========================================="
echo ""

cd "$(dirname "$0")"
echo "Project directory: $(pwd)"
echo ""

echo "Checking Python..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found. Install from https://www.python.org"
    exit 1
fi
PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python: $PY_VER"
echo ""

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    echo "OK: .venv created"
else
    echo ".venv already exists, skipping creation"
fi
echo ""

echo "Activating venv..."
source .venv/bin/activate
echo "OK: $(which python)"
echo ""

echo "Installing dependencies (this may take a few minutes)..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
echo "OK: all packages installed"
echo ""

echo "Verifying installation..."
python -c "import torch; print(f'  torch: {torch.__version__}')"
python -c "import ultralytics; print(f'  ultralytics: {ultralytics.__version__}')"
python -c "import fastapi; print(f'  fastapi: {fastapi.__version__}')"
if [ -f "models/best.pt" ]; then
    SIZE=$(ls -lh models/best.pt | awk '{print $5}')
    echo "  models/best.pt: $SIZE — ready"
else
    echo "  WARNING: models/best.pt not found. Train the model in Colab first."
fi
echo ""

echo "=========================================="
echo "  Setup complete. Start the server with:"
echo "=========================================="
echo ""
echo "  source .venv/bin/activate"
echo "  uvicorn api.main:app --host 0.0.0.0 --port 8000"
echo ""
echo "  Then open: http://localhost:8000/docs"
echo ""
