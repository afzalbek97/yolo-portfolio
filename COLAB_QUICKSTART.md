# Colab da Training: Bosqichma-bosqich qo'llanma

Bu — sizning M1 Mac da YOLOv8-OBB ni o'qitishning eng oson yo'li. Driverdan foydalanmaymiz — to'g'ridan-to'g'ri zip yuklaymiz, chunki dataset kichik (~110MB).

**Kutilayotgan vaqt:** 25-40 daqiqa (yuklash + training + saqlash).

---

## Qadam 1 — Colab ni ochish (30 soniya)

1. Brauzeringizda `https://colab.research.google.com` ni oching.
2. Google akkauntingiz bilan kiring.
3. **File → New notebook** bosing.

---

## Qadam 2 — GPU ni yoqish (30 soniya, JUDA MUHIM)

1. Yuqori menyuda **Runtime → Change runtime type**.
2. **Hardware accelerator** ostida **T4 GPU** ni tanlang.
3. **Save** bosing.

GPU borligini tekshirish — birinchi cell ga yozing va ishga tushiring (Shift+Enter):

```python
!nvidia-smi
```

Agar `Tesla T4` ko'rsatsa — tayyor. Agar "command not found" yoki xato bo'lsa — GPU yoqilmagan, qadamni qaytaring.

---

## Qadam 3 — Datasetni yuklash (5-10 daqiqa, internetingizga qarab)

Yangi cell yarating va shuni yozing:

```python
from google.colab import files
uploaded = files.upload()  # dialog ochiladi
```

Shift+Enter bilan ishga tushiring. Ochilgan dialogdan **`dataset.zip`** ni tanlang. Joylashuvi:

> `/Users/askarovafzalbek/Documents/yolo-portfolio/data/dataset.zip`

Yuklash progress ko'rinadi. Tugagach — quyidagi cell ni qo'shing:

```python
!unzip -q dataset.zip -d /content/
!ls /content/processed/
```

Natija quyidagicha bo'lishi kerak: `train  val  test`

---

## Qadam 4 — Ultralytics o'rnatish (1 daqiqa)

```python
!pip install -q ultralytics==8.2.0
import ultralytics
ultralytics.checks()
```

Bu YOLO ni o'rnatadi va GPU/CUDA ni tasdiqlaydi.

---

## Qadam 5 — data.yaml ni yaratish (10 soniya)

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

## Qadam 6 — Sanity check: dataset to'g'ri ekanligini tekshirish (1 daqiqa)

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
            if len(parts) < 9: continue
            cls = int(parts[0])
            coords = list(map(float, parts[1:9]))
            pts = np.array([[coords[i]*w, coords[i+1]*h] for i in range(0,8,2)], dtype=np.int32)
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
    ax.imshow(img); ax.axis('off')
plt.tight_layout(); plt.show()
```

Yashil va apelsin polygonlar to'g'ri butilkalar/banka ustida ko'rinmasa — to'xtang va menga aytib bering.

---

## Qadam 7 — TRAINING (15-25 daqiqa, T4 da)

Bu eng muhim cell. Ishga tushiring va kuting:

```python
from ultralytics import YOLO

model = YOLO('yolov8n-obb.pt')  # COCO pretrained nano OBB

results = model.train(
    data='/content/processed/data.yaml',
    epochs=80,
    imgsz=640,
    batch=16,
    optimizer='AdamW',
    lr0=0.001,
    cos_lr=True,
    patience=20,         # 20 epoch yaxshilanmasa to'xtaydi
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

**Kutish davomida nima ko'rasiz:**
- Har bir epoch oxirida — `box_loss`, `cls_loss`, `mAP50`, `mAP50-95` raqamlari
- 10-20 epoch dan keyin mAP50 odatda 0.6+ ga chiqadi
- 50+ epoch dan keyin yaxshilanish sekinlashadi (EarlyStopping ishlaydi bo'lishi mumkin)

**Agar xato chiqsa (out of memory):** `batch=16` ni `batch=8` qiling.

---

## Qadam 8 — Test setda baholash (1-2 daqiqa)

```python
best = '/content/runs/bottle_can_obb_v1/weights/best.pt'
model = YOLO(best)
metrics = model.val(data='/content/processed/data.yaml', split='test', plots=True)
print('mAP@0.5      :', metrics.box.map50)
print('mAP@0.5:0.95 :', metrics.box.map)
print('Per-class mAP@0.5 (bottle, can):', metrics.box.maps)
```

Bu raqamlarni README ning "Results" qismiga yozasiz.

---

## Qadam 9 — Sample prediction rasmlari (1 daqiqa)

```python
results = model.predict(
    source='/content/processed/test/images',
    save=True,
    conf=0.25,
    project='/content/preds',
    name='test',
    exist_ok=True,
)
print('Saqlandi: /content/preds/test')
```

---

## Qadam 10 — Hammasini yuklab olish (1-2 daqiqa)

```python
# Trained weights va training natijalarini zip qilamiz
!zip -qr /content/training_results.zip /content/runs/bottle_can_obb_v1
!zip -qr /content/predictions.zip /content/preds/test

from google.colab import files
files.download('/content/runs/bottle_can_obb_v1/weights/best.pt')
files.download('/content/training_results.zip')
files.download('/content/predictions.zip')
```

3 ta dialog ochiladi — har birida **OK** bosing. Fayllar Mac ning `Downloads` ga tushadi:
- `best.pt` — trained model weights (~6MB)
- `training_results.zip` — confusion matrix, PR curves, results.png va h.k.
- `predictions.zip` — test setdagi sample predictions

---

## Qadam 11 — Mac da fayllarni o'z joyiga qo'yish

Terminalda:

```bash
cd ~/Documents/yolo-portfolio
mv ~/Downloads/best.pt models/
mkdir -p results/training
unzip -o ~/Downloads/training_results.zip -d /tmp/
cp -r /tmp/content/runs/bottle_can_obb_v1/* results/training/
mkdir -p results/sample_predictions
unzip -o ~/Downloads/predictions.zip -d /tmp/
cp -r /tmp/content/preds/test/* results/sample_predictions/
```

Yoki menga ayting — men sizga ko'chirish uchun yordam beraman.

---

## Qadam 12 — FastAPI ni mahalliy ishga tushirish

```bash
cd ~/Documents/yolo-portfolio
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Brauzerda ochish: `http://localhost:8000/docs`

---

## Muammolar bo'lsa

| Xato | Yechim |
|---|---|
| `nvidia-smi: command not found` | GPU yoqilmagan — Qadam 2 ni qaytaring |
| `CUDA out of memory` | `batch=16` ni `batch=8` ga o'zgartiring |
| `data.yaml not found` | Qadam 5 da yo'l noto'g'ri — `/content/processed/data.yaml` bo'lishi kerak |
| Training juda sekin | GPU emas, CPU ishlamoqda — Runtime ni qayta tekshiring |
| `ConnectionError` files.download da | Brauzerda popup blocker — yoqing yoki Drive ga saqlang |
