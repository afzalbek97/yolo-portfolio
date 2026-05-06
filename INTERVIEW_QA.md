# Interview Q&A — Bottle vs Can YOLOv8-OBB Portfolio

Bu hujjat — sizning portfolio loyihangiz va ish e'lonidagi (AI 서버 개발자) talablar bo'yicha **kutilayotgan savollar** va **tayyor javoblar**. Har bir javob *mantiq* + *aniq raqam/misol* + *tajribadan o'rganganim* tarkibida.

---

## A. Loyihangiz haqida (eng ko'p so'raladi)

### 1. Loyihangiz haqida 1 daqiqada gapirib bering

> "Roboflow dan eksport qilingan oriented bounding box (OBB) datasetda butilka va banka aniqlovchi YOLOv8-OBB modelni o'qitdim. End-to-end ishladim: data exploration, training, evaluation va FastAPI orqali REST API sifatida deploy qildim. Modelni Google Colab T4 GPU da o'qitdim, inference ni Mac M1 da MPS bilan ishlatdim. Asosiy texnik qiyinchilik — datasetda klass imbalance (5:1 bottle:can) bo'lgani edi; transfer learning va aggressive augmentation bilan kompensatsiya qildim."

### 2. Nega aynan YOLOv8 (YOLOv5/YOLOv9/Faster R-CNN emas)?

- **YOLOv8 — bir bosqichli detector**, real-time inference uchun mos (REST API javob vaqti < 50ms M1 da)
- **OBB variant qo'llab-quvvatlanadi** — bizning datasetda labellar OBB formatida
- Ultralytics paketida **clean Python API** bor, fine-tuning bir necha qator kod
- COCO pretrained weights mavjud — kichik datasetga **transfer learning**
- Faster R-CNN — aniqlik bir oz yuqori, lekin **2-3x sekin**, REST API uchun ortiqcha
- YOLOv9 — yangiroq, lekin OBB variant Ultralytics da hali stabil emas

### 3. Nega yolov8n (nano), `s` yoki `m` emas?

- **Kichik dataset (292 train rasm)** — kattaroq model overfit qiladi
- **T4 GPU resurslari** — nano batch=16 bilan 80 epoch ~30 minutda tugaydi
- **Inference Mac M1** — nano CPU/MPS da ham real-time ishlaydi
- **Trade-off:** mAP biroz pastroq, lekin generalization yaxshi va deploy oson

### 4. Nega 80 epoch, 100 emas?

- `patience=20` bilan **EarlyStopping** o'rnatilgan — agar 20 epoch davomida val mAP yaxshilanmasa, training to'xtaydi va eng yaxshi weights saqlanadi
- 80 — yetarli yuqori chegara, lekin overfitting boshlanguncha to'xtaydi
- Validation curve bo'yicha aniq epoch ko'rinadi (results.png)

### 5. Augmentation parametrlarini qanday tanladingiz?

- `mosaic=1.0` — har bir iteratsiyada 4 ta rasmdan kompozitsiya, kichik dataset uchun **eng samarali augmentation**
- `mixup=0.15` — 15% probability bilan ikki rasmni aralashtirish, regularizatsiya
- `hsv_h/s/v=0.015/0.7/0.4` — yoritish/rang sharoitlari turlichaligini taqlid qilish (haqiqiy hayotda bottle/can har xil yorug'likda turadi)
- `degrees=10` — kamtar rotatsiya, OBB modeli aslida burchakni o'rganadi, lekin training paytida ham qisman aylantirib o'rgansak yaxshi
- Hammasi Ultralytics defaults dan biroz kuchaytirilgan, kichik dataset uchun

### 6. Datasetdagi qaysi muammoni topdingiz va qanday hal qildingiz?

- **Train labels bo'sh edi** — Roboflow eksportida 1324 ta train rasm bor, lekin labels papkasi bo'sh. Faqat valid (302) va test (117) labelli edi
- **Yechim:** valid+test (419) ni birlashtirdim, fixed seed (42) bilan 70/15/15 ga qayta bo'ldim → 292 train, 61 val, 63 test
- **Trade-off:** kichikroq train set, lekin aniq evaluation mumkin. Kompensatsiya — transfer learning + heavy augmentation
- **README da ochiq yozdim** — bu professional yondashuv: muammoni yashirmaydi, qanday hal qilganini aytadi

### 7. Klass imbalance bilan qanday ishladingiz?

- **Aniqladim:** bottle:can = 5.05:1 (460 vs 91 annotation), test setda atigi 9 ta can bor
- **Eslatdim README ga** — gen interpretatsiya uchun muhim
- **Per-class mAP** ni alohida kuzatdim — overall mAP yolg'on yuqori bo'lishi mumkin, can class quyi
- **Augmentation** can class ga ko'proq foyda keltiradi (mosaic ko'paytiradi)
- **Yaxshilash imkoniyatlari (kelajak ish):** weighted sampling, focal loss tuning, ko'proq can rasm yig'ish

---

## B. YOLO va detection asoslari

### 8. mAP nima va qanday hisoblanadi?

- **mAP = mean Average Precision** — barcha klasslar bo'yicha o'rtacha AP
- **AP** = precision-recall curve ostidagi maydon (har bir klass uchun alohida)
- **mAP@0.5** — IoU threshold 0.5 da hisoblanadi (klassik)
- **mAP@0.5:0.95** — 0.5 dan 0.95 gacha 10 ta IoU thresholdda hisoblanib, o'rtacha olinadi (COCO standarti, qattiqroq)
- Bizning loyihada ikkalasi ham hisoblanadi

### 9. IoU nima?

- **IoU = Intersection over Union** — predicted bbox va ground truth bbox kesishmasini ularning birlashmasiga nisbati
- Formula: `IoU = (A ∩ B) / (A ∪ B)`
- Detection to'g'ri deb hisoblanishi uchun IoU ≥ threshold (odatda 0.5) bo'lishi kerak
- OBB uchun standard rotated polygon IoU ishlatiladi (cv2.rotatedRectangleIntersection)

### 10. NMS (Non-Maximum Suppression) nima?

- Bir obyekt uchun ko'pincha bir nechta overlapping bbox chiqadi (turli confidence bilan)
- NMS algoritmi:
  1. Eng yuqori confidence li bbox ni saqla
  2. Boshqa bbox lar bilan IoU > threshold bo'lsa, ularni o'chir
  3. Qayta takrorla
- YOLOv8 da `iou=0.7` default
- OBB versiyasi uchun rotated NMS ishlatiladi

### 11. Anchor boxes haqida nima bilasiz? YOLOv8 da bormi?

- Eski YOLO versiyalarida (v3, v5) anchor boxes — oldindan tanlangan default size va aspect ratio bo'yicha bbox shabloni
- **YOLOv8 — anchor-free detector** — to'g'ridan-to'g'ri bbox koordinatalarini predict qiladi (markaz + offset)
- Bu eski YOLO ga qaraganda **soddaroq, kamroq hyperparameter**, kichik datasetlarga moslashuvchanroq

### 12. Confusion matrix da nimani ko'rasiz?

- **Diagonal** — to'g'ri klassifikatsiyalar
- **Off-diagonal** — model qaysi klasslarni adashtiradi
- "background" qatori/ustuni — false positive/negative
- Bizning loyihada `bottle ↔ can` adashishi ko'rsatkichi muhim. Agar can ko'p marta "background" deb aniqlansa — recall past, modelga ko'proq can sample yoki augmentation kerak

### 13. Precision va Recall farqi?

- **Precision = TP / (TP + FP)** — model "bu bottle" deganda, qancha to'g'ri?
- **Recall = TP / (TP + FN)** — haqiqatda bottle bor bo'lsa, model qancha topa oldi?
- **F1 = 2·P·R / (P+R)** — ikkalasining harmonic mean i
- Trade-off: confidence threshold ni oshirsangiz precision yuqori, recall pasayadi

### 14. OBB va standard bbox farqi?

- **Standard bbox (XYXY/XYWH)**: 4 ta raqam, axis-aligned to'rtburchak
- **OBB (Oriented Bounding Box)**: 5 ta raqam (cx, cy, w, h, angle) yoki 8 ta (4 burchak nuqtasi)
- **Foydasi:** burilgan obyektlarni aniqroq qoplaydi — havo fotosuratlari, hujjat sahifalari, yotgan butilka. IoU ham aniqroq hisoblanadi
- **Loss function:** OBB da rotated IoU yoki Gaussian-based loss ishlatiladi
- YOLOv8-OBB ProbIoU loss ishlatadi

---

## C. PyTorch va deep learning asoslari

### 15. Transfer learning nima va nega ishlatdingiz?

- **Transfer learning** = pretrained model (boshqa, kattaroq datasetda o'qitilgan) ni boshlang'ich nuqta sifatida olish va o'z datasetingizda fine-tune qilish
- Sabab: 292 ta rasm scratch dan model o'qitishga **kam**, lekin COCO da 100k+ rasmda backbone yaxshi feature ni allaqachon o'rgangan
- Asosan **convolutional backbone** (Darknet/CSPDarkNet) feature extraction qiladi, **detection head** ni qayta o'qitamiz
- Natijada: 80 epoch da yuqori mAP, scratch da 500+ epoch kerak bo'lardi

### 16. Optimizer tanlovi: nega AdamW?

- **SGD with momentum** — katta datasetlarda yaxshi, generalization yuqori
- **Adam/AdamW** — kichik dataset va noise li gradients da barqarorroq, tezroq yaqinlashadi
- **AdamW** = Adam + decoupled weight decay (regularizatsiya to'g'ri ishlaydi)
- Bizda 292 ta rasm — AdamW logiq tanlov

### 17. Cosine learning rate scheduler nima beradi?

- LR ni epoch boshida `lr0` dan, oxirida `lr0 * lrf` (~0.01·lr0) gacha cosine egri chiziq bo'yicha pasaytiradi
- Boshida tez o'rganadi, oxirida fine-tune
- Constant LR ga qaraganda final mAP odatda 1-2% yuqori

### 18. Overfitting belgilarini qanday aniqladingiz?

- **Train loss pasayyapti, val loss ko'tarilyapti** — klassik overfit signali
- **Val mAP plateau** bo'lib qolyapti yoki tushyapti
- Tools: `results.png` (training curves), `confusion_matrix.png`
- Hal qilish: ko'proq augmentation, dropout, EarlyStopping, kichikroq model

### 19. Batch size ni qanday tanladingiz?

- T4 GPU 16GB xotira — yolov8n-obb + imgsz=640 bilan batch=16 sig'adi (~10GB band)
- Kichik batch (1-4): noisy gradients, lekin generalization yaxshi (kichik dataset uchun ham mos)
- Katta batch (32+): tezroq, lekin xotira kerak va kichik datasetda overfit xavfi
- 16 — balansli tanlov

### 20. Loss function da nima bor?

YOLOv8 multitask loss:
- **Box loss** (CIoU yoki ProbIoU OBB uchun) — bbox koordinatalari aniqligi
- **Cls loss** (BCE) — klass predict qilish
- **Dfl** (Distribution Focal Loss) — bbox regression aniqlik

Hammasi vazni bilan summa qilinadi.

---

## D. REST API, deployment, GPU

### 21. FastAPI ni qanday qurdingiz va nega FastAPI?

- **FastAPI** — Python da eng tez REST framework (Starlette ustida), avtomatik OpenAPI/Swagger docs, type hints orqali validatsiya
- Endpoints: `/health`, `/predict` (JSON), `/predict/visualize` (PNG)
- `startup` event da model bir marta yuklanadi (har request da emas)
- **multipart/form-data** orqali rasm yuklanadi
- Pydantic schemalar bilan response strukturasi tasdiqlanadi

### 22. Productionga deploy qilish uchun nima qo'shar edingiz?

- **Docker container** — `python:3.11-slim` base, requirements + model bundled
- **Gunicorn + Uvicorn workers** — production WSGI/ASGI
- **Async** endpoint logic, Celery/Redis queue agar inference batch bo'lsa
- **Authentication** — API key yoki OAuth
- **Rate limiting** — `slowapi` yoki nginx
- **Logging + monitoring** — structured logs, Prometheus metrics, request latency
- **Model versioning** — `models/v1/best.pt`, header `X-Model-Version` orqali
- **GPU server** — vLLM emas, lekin Triton Inference Server yoki TorchServe — YOLO uchun yetarli

### 23. GPU server tajribangiz haqida (e'londa shu so'ralgan)

- Bu loyihada — **Google Colab T4 GPU** ni training uchun ishlatdim, Drive bilan ulab fayl uzatish, nvidia-smi bilan monitoring
- Mac M1 — **MPS backend** (Apple Silicon Metal) PyTorch da `device='mps'` orqali, lekin OBB to'liq qo'llab-quvvatlanmaydi, shuning uchun inference CPU da ishladim
- Production scenario uchun bilamanlar: NVIDIA driver/CUDA o'rnatish, `nvidia-smi`, multi-GPU training (`device=0,1`), batch parallelism, mixed precision (`amp=True`)
- vLLM/Ollama — LLM serving uchun, YOLO emas. LLM kontekst da PagedAttention bilan batchli serving optimizatsiya qiladi

### 24. ONNX export va inference optimizatsiyasi

- `model.export(format='onnx')` — Ultralytics bir qatorda eksport qiladi
- **Foyda:** PyTorch dependency olinadi, inference 1.5-3x tezroq, har xil platformda (C++, mobile)
- Production da: ONNX Runtime yoki TensorRT (NVIDIA) bilan inference
- Mac M1 da CoreML format ham mumkin

### 25. RAG va LLM bilan tanishmisiz? (e'londa 'RAG system' so'ralgan)

- **RAG = Retrieval-Augmented Generation** — LLM ga kontekst yuklash usuli:
  1. **Indexing**: hujjatlarni chunklarga bo'lib, embedding (vector) ga aylantirib, vector DB (FAISS, Chroma, Pinecone) ga saqlash
  2. **Retrieval**: foydalanuvchi savoliga embedding olib, k ta eng yaqin chunkni topish (cosine similarity)
  3. **Generation**: chunklarni context ga qo'yib LLM ga prompt yuborish
- **Foydasi:** LLM ning bilim cheklovini engib o'tish, fresh ma'lumot bilan ishlash, hallucination ni kamaytirish
- **Tools:** LangChain (chain orchestration), LlamaIndex (data layer), VLLM (serving), Ollama (local LLM)
- Bu portfolio CV ga qaratilgan, lekin RAG/LLM mini-project ni qo'shish kelajak ish (README ham aytdim)

---

## E. Yumshoq savollar (HR / kalon-kabin)

### 26. Nega bizning kompaniyaga kelmoqchisiz?
> Konkret kompaniya nomini, mahsulotini, missiyasini eslatib o'ting. Ishxonaning ish e'lonida AI serving, RAG va Computer Vision bo'yicha 4 ta katta yo'nalish bor — bu mening qiziqishlarim bilan to'g'ri keladi: men CV (YOLO portfolio) qilganman, RAG/LLM tomonida o'qilyapman, REST API tajribam bor.

### 27. 5 yildan keyin o'zingizni qaerda ko'rasiz?
> ML/AI engineer dan senior level ga o'sib, end-to-end ML system (data → model → deploy → monitor) qura oladigan inson bo'lish. AI server tomonida ko'proq experience yig'ish, ehtimol, MLOps yoki research engineer yo'nalishida ixtisoslashish.

### 28. Sizdan jamoa nima kutadi?
> Tezda fideyback qabul qilish, dokumentatsiya yozish (loyihada README va Q&A tayyorladim), muammoni yashirmasdan ochiq aytish (datasetdagi label muammosini qanday hal qilganim — misol).

### 29. Sizning kuchli/kuchsiz tomoningiz?
- **Kuchli:** O'zim mustaqil o'rganishni, muammoni decompose qilishni, dokumentatsiya yozishni yaxshi ko'raman. Loyihada — ML xayoldan emas, **kichik real natija** topshirishga harakat qilaman
- **Kuchsiz:** Production-grade system tajribam yo'q (CI/CD, monitoring, large-scale GPU). Lekin men allaqachon Docker, FastAPI, ONNX o'rganib, kichik loyihalarda qo'llashga o'tdim.

### 30. Maoshingizni ayting? (eng oxirida)
> Avval ish va jamoa haqida ko'proq bilishni istardim. Hududdagi neonqilingan junior AI engineer maoshlar diapazonida (Korea da 약 3,000–4,000만 won/yil neonqilingan), aniq raqamni siz tomondan eshitsam yaxshi bo'lardi.

---

## F. Texnik live-coding savollarga tayyorgarlik

### Eng ehtimoli kelishi mumkin bo'lganlari:

1. **"YOLO output ni JSON ga konvertatsiya qiladigan funksiya yozing"** — bizda allaqachon `api/main.py` da bor, ishlash logikasini eslab oling
2. **"OBB labelni axis-aligned bbox ga o'tkazing"** — `min_x = min(x1,x2,x3,x4)`, `max_x = max(...)` va h.k.
3. **"NMS ni o'zingiz yozing"** — confidence bo'yicha sort, IoU > thr larni o'chirib chiqing
4. **"Confusion matrix dan precision/recall hisoblang"** — TP, FP, FN ni qator/ustun summalaridan toping
5. **"Class imbalance ni neon balanslash usullari"** — oversampling, undersampling, class weights, focal loss

### Tayyor pseudocode (NMS):
```python
def nms(boxes, scores, iou_thr=0.5):
    idx = scores.argsort()[::-1]  # konfidens bo'yicha kamayuvchi
    keep = []
    while len(idx) > 0:
        i = idx[0]
        keep.append(i)
        if len(idx) == 1: break
        ious = compute_iou(boxes[i], boxes[idx[1:]])
        idx = idx[1:][ious < iou_thr]
    return keep
```

---

## G. Salbiy savollarga tayyorgarlik

### "Bu loyiha juda kichkina, real production kabi emas-ku"
> Roziman — bu portfolio loyiha, vaqt cheklovida (6 soat) end-to-end ish jarayonini ko'rsatadi: data muammolarini topish, transfer learning, evaluation, REST API. Production scale uchun nima qo'shilishi kerakini README ning "future work" qismida sanab o'tdim — Docker, monitoring, ONNX, CI/CD, ko'p data.

### "Nega siz LLM/RAG loyihasi qilmadingiz, e'londa shuni so'rashgan?"
> CV / YOLO ham e'londa "preferred skill" qatorida. Vaqt cheklovida real loyiha qilishni tanladim — RAG/LangChain bo'yicha o'rganishim davom etmoqda, allaqachon LangChain bilan kichik chatbot tutorial qilganman, va ishga kirsam — bu yo'nalishda chuqurroq tezda tayyorlanaman.

### "Class imbalance ni nega weighted sampling bilan hal qilmagansiz?"
> Vaqt budjeti cheklangani uchun augmentation va per-class metric tracking bilan boshlandim. Weighted sampling — keyingi qadam (README da ham bor). Production model uchun albatta sinab ko'rar edim, A/B test qilib qarar qilardim.

---

**Eslatma:** har bir javobni *qisqa* (30-60 soniya) gapiring. Suhbatchini bo'g'maslik kerak. Agar texnik chuqurlikka kirish kerak bo'lsa, **interviewer o'zi so'raydi**.
