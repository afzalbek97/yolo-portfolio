# Training on Google Colab: Step-by-Step Guide

This is the recommended way to train YOLOv8-OBB on a T4 GPU without any local GPU setup.
The dataset is small (~110 MB), so we upload the zip directly instead of using the Roboflow API.

**Estimated time:** 25–40 minutes (upload + training + download).

---

## Step 1 — Open Colab (30 seconds)

1. Go to `https://colab.research.google.com` in your browser.
2. Sign in with your Google account.
3. Click **File → New notebook**.

---

## Step 2 — Enable T4 GPU (30 seconds — IMPORTANT)

1. In the top menu: **Runtime → Change runtime type**.
2. Under **Hardware accelerator**, select **T4 GPU**.
3. Click **Save**.

Verify the GPU is active — run this in the first cell (Shift+Enter):

```python
!nvidia-smi
```

If you see `Tesla T4`, you are ready. If you see an error, repeat Step 2.

---

## Step 3 — Upload the dataset (5–10 minutes depending on connection)

Create a new cell and run:

```python
from google.colab import files
uploaded = files.upload()  # opens a file dialog
```

Select `dataset.zip` from your local machine (the file is in `data/` inside this repo after you prepare it).

Then unzip:

```python
!unzip -q dataset.zip -d /content/
!ls /content/processed/
```

Expected output: `train  val  test`

---

## Step 4 — Install Ultralytics (1 minute)

```python
!pip install -q "ultralytics>=8.3.0"
import ultralytics
ultralytics.checks()
```

This installs YOLO and confirms the GPU/CUDA setup.

---

## Step 5 — Create data.yaml (10 seconds)

```python
yaml_content = """path: /content/processed
train: train/images
val: val/images
test: test/images
names:
  0: bottle
  1: can
nc: 2
"""
with open('/content/processed/data.yaml', 'w') as f:
    f.write(yaml_content)
print(yaml_content)
```

---

## Step 6 — Sanity check: visualise OBB annotations (1 minute)

```python
import cv2, glob, random
import numpy as np
import matplotlib.pyplot as plt

CLASS_NAMES = {0: 'bottle', 1: 'can'}
COLORS = {0: (0, 200, 0), 1: (0, 100, 255)}

def draw_obb(img, label_path):
    h, w = img.shape[:2]
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:
                continue
            cls = int(parts[0])
            coords = list(map(float, parts[1:9]))
            pts = np.array(
                [[coords[i] * w, coords[i + 1] * h] for i in range(0, 8, 2)],
                dtype=np.int32,
            )
            cv2.polylines(img, [pts], True, COLORS[cls], 3)
            cv2.putText(img, CLASS_NAMES[cls], tuple(pts[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[cls], 2)
    return img

random.seed(0)
imgs = random.sample(glob.glob('/content/processed/train/images/*.jpg'), 6)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for ax, img_path in zip(axes.flat, imgs):
    lbl = img_path.replace('/images/', '/labels/').replace('.jpg', '.txt')
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img = draw_obb(img, lbl)
    ax.imshow(img)
    ax.axis('off')
plt.tight_layout()
plt.show()
```

Green and orange polygons should appear over bottles and cans. If they are missing or misaligned, check that the dataset was extracted correctly.

---

## Step 7 — Training (15–25 minutes on T4)

```python
from ultralytics import YOLO

model = YOLO('yolov8n-obb.pt')  # COCO-pretrained nano OBB

results = model.train(
    data='/content/processed/data.yaml',
    epochs=80,
    imgsz=640,
    batch=16,
    optimizer='AdamW',
    lr0=0.001,
    cos_lr=True,
    patience=20,        # early stopping: stop if no improvement for 20 epochs
    mosaic=1.0,
    mixup=0.15,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    degrees=10.0, translate=0.1, scale=0.5,
    project='/content/runs',
    name='bottle_can_obb_v1',
    exist_ok=True,
    plots=True,
)
```

**What you will see:** `box_loss`, `cls_loss`, `mAP50`, `mAP50-95` printed after each epoch.
mAP50 typically reaches 0.6+ by epoch 10–20 and plateaus around epoch 50–70.

**If you get an out-of-memory error:** reduce `batch=16` to `batch=8`.

---

## Step 8 — Evaluate on the test set (1–2 minutes)

```python
best = '/content/runs/bottle_can_obb_v1/weights/best.pt'
model = YOLO(best)
metrics = model.val(data='/content/processed/data.yaml', split='test', plots=True)
print('mAP@0.5      :', round(metrics.box.map50, 3))
print('mAP@0.5:0.95 :', round(metrics.box.map, 3))
print('Per-class mAP@0.5 (bottle, can):', [round(v, 3) for v in metrics.box.maps])
```

Record these numbers for the README Results section.

---

## Step 9 — Save sample predictions (1 minute)

```python
results = model.predict(
    source='/content/processed/test/images',
    save=True,
    conf=0.25,
    project='/content/preds',
    name='test',
    exist_ok=True,
)
print('Saved to: /content/preds/test')
```

---

## Step 10 — Download everything (1–2 minutes)

```python
import shutil
from google.colab import files

shutil.make_archive('/content/training_results', 'zip', '/content/runs/bottle_can_obb_v1')
shutil.make_archive('/content/predictions', 'zip', '/content/preds/test')

files.download('/content/runs/bottle_can_obb_v1/weights/best.pt')
files.download('/content/training_results.zip')
files.download('/content/predictions.zip')
```

Three download dialogs will open. Click **OK** for each. Files land in your `~/Downloads` folder:
- `best.pt` — trained model weights (~6 MB)
- `training_results.zip` — confusion matrix, PR curves, `results.png`, etc.
- `predictions.zip` — test-set sample predictions

---

## Step 11 — Move files into the repo

```bash
cd path/to/yolo-portfolio
mv ~/Downloads/best.pt models/
mkdir -p results/training results/sample_predictions
unzip -o ~/Downloads/training_results.zip -d /tmp/train_results/
cp -r /tmp/train_results/content/runs/bottle_can_obb_v1/* results/training/
unzip -o ~/Downloads/predictions.zip -d /tmp/preds/
cp -r /tmp/preds/* results/sample_predictions/
```

---

## Step 12 — Run the FastAPI server locally

```bash
cd path/to/yolo-portfolio
bash setup_mac.sh         # first time only: creates .venv and installs deps
source .venv/bin/activate
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Open in browser: `http://localhost:8000/docs`

---

## Troubleshooting

| Error | Fix |
|---|---|
| `nvidia-smi: command not found` | GPU not enabled — repeat Step 2 |
| `CUDA out of memory` | Change `batch=16` to `batch=8` |
| `data.yaml not found` | Check path in Step 5: must be `/content/processed/data.yaml` |
| Training very slow | Running on CPU, not GPU — re-check Runtime settings |
| `ConnectionError` on `files.download` | Browser popup blocker — disable it or save to Google Drive instead |
