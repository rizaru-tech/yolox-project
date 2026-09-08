# YOLOX-S — Workshop Tools

Implementasi ini melakukan fine-tuning YOLOX-S untuk tiga kelas alat pada dataset
lokal: `Wrenches`, `pliers`, dan `screwdriver`. Dataset memakai anotasi COCO, tetapi
folder gambarnya bernama `train`, `val`, dan `test` (bukan `train2017`/`val2017`).

Gunakan folder ini sebagai sumber kanonis:

```text
workshop_tools_yolox/
├── workshop_yolox_s.py             # experiment YOLOX-S
├── validate_dataset.py              # preflight integritas dan kualitas label
├── build_training_annotations.py    # membersihkan artefak bbox train
├── predict.py                       # inference dengan label lokal + JSON
└── workshop_tools/
    ├── train/                        # 360 gambar
    ├── val/                          # 20 gambar
    ├── test/                         # 10 gambar
    └── annotations/
        ├── instances_train.json      # anotasi asli: 483 bbox
        ├── instances_train.yolox.json # anotasi train bersih: 471 bbox
        ├── instances_val.json        # 22 bbox
        └── instances_test.json       # 10 bbox
```

Experiment memakai `instances_train.yolox.json`. Anotasi asli tidak ditimpa. Sebanyak
12 bbox train yang hanya berukuran 0,5–2 px pada salah satu sisi dihapus dari salinan
training; semuanya merupakan kotak tambahan yang nyaris berupa titik/garis pada gambar
yang masih memiliki bbox objek normal. Validation dan test tidak dibersihkan atau
diubah.

## 1. Validasi data

Jalankan sebelum training:

```powershell
python workshop_tools_yolox/validate_dataset.py
python workshop_tools_yolox/validate_dataset.py --train-ann instances_train.yolox.json
```

Perintah pertama mengaudit anotasi asli; perintah kedua memastikan anotasi yang benar-
benar dipakai training. Pemeriksaan mencakup JSON, gambar hilang/rusak, dimensi, bbox,
category mapping, file duplikat byte-identik, dan nama sumber lintas split.

Untuk membuat ulang anotasi training bersih:

```powershell
python workshop_tools_yolox/build_training_annotations.py
```

## 2. Siapkan environment YOLOX resmi

Disarankan memakai environment Python 3.10 terpisah. Pilih build PyTorch yang cocok
dengan GPU/driver dari https://pytorch.org/get-started/locally/, lalu clone dan install
YOLOX resmi secara editable:

```powershell
git clone https://github.com/Megvii-BaseDetection/YOLOX.git
cd YOLOX
py -3.10 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel

# Pasang torch dan torchvision sesuai perintah dari selector PyTorch terlebih dahulu.
python -m pip install -r requirements.txt
python -m pip install --no-build-isolation -v -e .
python -m pip install -r "C:/path/workshop_tools_yolox/requirements-workshop.txt"
```

Catat versi yang berhasil agar eksperimen dapat direproduksi:

```powershell
git rev-parse HEAD
python -m pip freeze > environment-lock.txt
```

Download checkpoint pretrained `yolox_s.pth` dari model zoo repositori resmi. Fine-
tuning dari bobot COCO direkomendasikan oleh dokumentasi YOLOX untuk custom dataset.

## 3. Training baseline

Jalankan dari root repositori YOLOX. Contoh satu GPU, total batch 8:

```powershell
$env:WORKSHOP_EVAL_SPLIT = "val"
python tools/train.py `
  -f "C:/path/workshop_tools_yolox/workshop_yolox_s.py" `
  -d 1 -b 8 --fp16 `
  -c "C:/path/checkpoints/yolox_s.pth"
```

Konfigurasi mempertahankan baseline resmi: 300 epoch, EMA, multi-scale, Mosaic, MixUp,
warmup 5 epoch, dan 15 epoch terakhir tanpa augmentasi kuat. Ini sengaja dijadikan
baseline terlebih dahulu. Dataset train asal Roboflow memang sudah memiliki augmentasi
offline; bila kurva validation menunjukkan overfit atau augmentasi gabungan terlalu
agresif, ubah satu faktor per eksperimen dan bandingkan AP validation.

Pedoman praktis:

- Batch rekomendasi awal adalah 8 per GPU; turunkan ke 4 atau 2 jika CUDA OOM. YOLOX
  menghitung learning rate dari total batch.
- Gunakan `--fp16` hanya pada CUDA. Jangan gunakan FP16 untuk CPU.
- Gunakan `--cache ram` hanya bila RAM cukup; caching bukan syarat training.
- Jangan memilih epoch atau threshold dari test set.
- Untuk smoke test pipeline, tambahkan override `max_epoch 1 data_num_workers 0`.

Checkpoint biasanya tersimpan di:

```text
YOLOX_outputs/workshop_yolox_s/best_ckpt.pth
```

## 4. Evaluasi

Pilih checkpoint berdasarkan validation:

```powershell
$env:WORKSHOP_EVAL_SPLIT = "val"
python tools/eval.py `
  -f "C:/path/workshop_tools_yolox/workshop_yolox_s.py" `
  -c "YOLOX_outputs/workshop_yolox_s/best_ckpt.pth" `
  -d 1 -b 8 --conf 0.001 --fp16 --fuse
```

Setelah model dan semua keputusan final terkunci, evaluasi test satu kali:

```powershell
$env:WORKSHOP_EVAL_SPLIT = "test"
python tools/eval.py `
  -f "C:/path/workshop_tools_yolox/workshop_yolox_s.py" `
  -c "YOLOX_outputs/workshop_yolox_s/best_ckpt.pth" `
  -d 1 -b 8 --conf 0.001 --fp16 --fuse
$env:WORKSHOP_EVAL_SPLIT = "val"
```

Jangan tambahkan `--test` atau `--testdev`; file test lokal memiliki ground truth dan
harus melalui evaluator biasa.

## 5. Inference gambar

Demo resmi YOLOX memakai nama kelas COCO secara default. Gunakan script lokal agar
hasil menampilkan tiga label workshop yang benar:

```powershell
python "C:/path/workshop_tools_yolox/predict.py" `
  "C:/path/images" `
  --checkpoint "YOLOX_outputs/workshop_yolox_s/best_ckpt.pth" `
  --device cuda --fp16 --fuse `
  --conf 0.25 --nms 0.45 `
  --output "C:/path/predictions"
```

Output berisi gambar beranotasi serta `predictions.json` dengan label, confidence, bbox
`xyxy`, dan waktu inference. Untuk CPU gunakan `--device cpu` tanpa `--fp16`.

## Kondisi dan batasan data

- Split sumber dipertahankan: train 360, val 20, test 10; tidak ada resplit.
- Semua gambar berukuran 640×640 hasil resize stretch dari Roboflow.
- Tidak ditemukan gambar byte-identik atau nama sumber yang bocor antar-split.
- Distribusi bbox train relatif seimbang: 135 Wrenches, 162 pliers, 174 screwdriver
  setelah 12 artefak Wrenches dibuang.
- Tujuh bbox train mencakup setidaknya 99% gambar dan perlu inspeksi manual lanjutan.
- Tidak ada negative/background-only image. Tambahkan contoh latar deployment tanpa
  ketiga alat untuk mengukur dan menekan false positive.
- Test hanya 10 gambar, jadi AP test memiliki variance tinggi dan belum layak dianggap
  estimasi performa produksi.
- Dataset berlisensi CC BY 4.0; pertahankan atribusi dari `README.dataset.txt`.

## Referensi

- Implementasi resmi: https://github.com/Megvii-BaseDetection/YOLOX
- Custom training: https://github.com/Megvii-BaseDetection/YOLOX/blob/main/docs/train_custom_data.md
- Paper YOLOX: https://arxiv.org/abs/2107.08430
- Dataset: https://universe.roboflow.com/yj-ebtzo/workshop-tools-3/dataset/2
