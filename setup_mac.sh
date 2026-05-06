#!/usr/bin/env bash
# Mac M1 da FastAPI muhitini avtomatik sozlash
# Ishlatish: bash setup_mac.sh

set -e  # bironta xato bo'lsa to'xta

echo "=========================================="
echo "  YOLO Portfolio — Mac sozlash skripti"
echo "=========================================="
echo ""

# 1. Loyiha papkasi
cd "$(dirname "$0")"
echo "Loyiha: $(pwd)"
echo ""

# 2. Python versiyasi
echo "Python tekshirilmoqda..."
if ! command -v python3 &> /dev/null; then
    echo "XATO: python3 topilmadi. https://www.python.org dan o'rnating."
    exit 1
fi
PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "Python: $PY_VER"
echo ""

# 3. venv yaratish
if [ ! -d ".venv" ]; then
    echo "Virtual environment yaratilmoqda..."
    python3 -m venv .venv
    echo "OK: .venv yaratildi"
else
    echo ".venv allaqachon mavjud, qayta yaratilmaydi"
fi
echo ""

# 4. Faollashtirish
echo "Venv faollashtirilmoqda..."
source .venv/bin/activate
echo "OK: venv faol — $(which python)"
echo ""

# 5. Paketlar
echo "Paketlar o'rnatilmoqda... (5-10 daqiqa, sabr bilan kuting)"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
echo "OK: barcha paketlar o'rnatildi"
echo ""

# 6. Tasdiqlash
echo "Tekshirish..."
python -c "import torch; print(f'  torch: {torch.__version__}')"
python -c "import ultralytics; print(f'  ultralytics: {ultralytics.__version__}')"
python -c "import fastapi; print(f'  fastapi: {fastapi.__version__}')"
if [ -f "models/best.pt" ]; then
    SIZE=$(ls -lh models/best.pt | awk '{print $5}')
    echo "  best.pt: $SIZE — tayyor"
else
    echo "  XATO: models/best.pt topilmadi"
fi
echo ""

# 7. Tugashi va ishga tushirish ko'rsatmasi
echo "=========================================="
echo "  TAYYOR! Endi serverni ishga tushiring:"
echo "=========================================="
echo ""
echo "  source .venv/bin/activate"
echo "  uvicorn api.main:app --host 0.0.0.0 --port 8000"
echo ""
echo "Brauzerda: http://localhost:8000/docs"
echo ""
