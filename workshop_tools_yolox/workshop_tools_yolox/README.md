# Workshop Tools — persiapan data YOLOX

Paket ini berisi script, konfigurasi YOLOX-S, dan HASIL persiapan ZIP yang diberikan Rizal.
Dataset khusus peralatan ini tidak menggunakan nama folder atau anotasi `2017`.
Istilah COCO di sini hanya merujuk format anotasi JSON, bukan asal dataset COCO 2017.

## Kelas dan hasil

| Indeks output model | Category ID COCO lokal | Nama | Arti |
|---|---|---|---|
| 0 | 1 | Wrenches | Kunci pas |
| 1 | 2 | pliers | Tang |
| 2 | 3 | screwdriver | Obeng |

Kategori induk `Warehouse-cbFv-B57h` (ID 0) tidak digunakan oleh anotasi mana pun dan
dihapus dari daftar categories. Nama serta urutan ketiga kelas dipertahankan.

| Split | Gambar | Bounding box |
|---|---:|---:|
| train | 360 | 483 |
| val | 20 | 22 |
| test | 10 | 10 |

Split asli dipertahankan. `valid` hanya diganti nama menjadi `val`.
Tidak ada pemindahan gambar test ke training, pembagian ulang, resize, atau augmentasi baru.

## Isi paket

- `prepare_workshop.py`: membaca ZIP Roboflow dan mempersiapkan data.
- `workshop_yolox_s.py`: experiment YOLOX-S untuk struktur folder ini.
- `workshop_tools/`: dataset yang sudah dipersiapkan; dapat langsung dipakai dengan experiment tersebut.
- `workshop_tools/train/`: 360 gambar.
- `workshop_tools/val/`: 20 gambar.
- `workshop_tools/test/`: 10 gambar.
- `workshop_tools/annotations/instances_train.json`.
- `workshop_tools/annotations/instances_val.json`.
- `workshop_tools/annotations/instances_test.json`.
- `workshop_tools/labels.json`: urutan label untuk tampilan aplikasi.
- `workshop_tools/class_mapping.json`: pemetaan kategori dan indeks model.
- `workshop_tools/preparation_report.json`: statistik dan hasil pemeriksaan.
- README asal dataset dipertahankan di folder dataset untuk atribusi/lisensi.

## Menjalankan script sendiri

Jika memakai dataset yang sudah tersedia di paket, tahap ini boleh dilewati.
Script membutuhkan Python 3.10+ dan Pillow, tanpa GPU dan tanpa akses jaringan ke COCO.
Pillow diperlukan untuk memeriksa gambar, bukan untuk mengubah gambar.

```bash
python -m pip install Pillow
python prepare_workshop.py "D:/Downloads/Workshop tools 3.v2i.coco.zip" --output "D:/datasets/workshop_tools"
```

Pada Windows, boleh menggunakan `py` sebagai pengganti `python`.
Jika folder tujuan sudah ada, script berhenti tanpa menimpa isinya. Pilih folder
output baru. Persiapan yang gagal dibersihkan; file ZIP sumber tidak diubah.
Script menerima format ZIP Roboflow dengan `train`, `valid`, dan `test`, masing-masing
memiliki `_annotations.coco.json` dan gambar yang dirujuk.

Pemeriksaan mencakup: ID unik dalam setiap JSON, referensi gambar/kategori, keberadaan
gambar, ukuran gambar, bounding box dalam batas gambar, dan duplikat identik antar-split.
ID gambar boleh berulang di JSON split lain karena tiap JSON berdiri sendiri.

## Memakai di YOLOX

Prasyarat: clone/install YOLOX resmi dan PyTorch yang kompatibel dengan GPU kamu.
Ikuti https://github.com/Megvii-BaseDetection/YOLOX dan kunci versi yang berhasil.
Experiment ini mengikuti API YOLOX yang menyediakan `get_dataset` dan `get_eval_dataset`,
serta COCODataset dengan opsi cache/cache_type. Fork atau versi lama dapat memerlukan penyesuaian.

Konfigurasi memakai `name='train'`, `name='val'`, dan `name='test'` secara eksplisit.
Ini yang membuat nama folder tanpa tahun bekerja. Jumlah kelas dibaca dari labels.json.
Jangan menggunakan file experiment paket botol/cangkir/kursi sebelumnya.

Secara default, data dicari di folder `workshop_tools` di sebelah file experiment.
Jangan memisahkan kedua lokasi itu kecuali mengatur WORKSHOP_DATA_DIR:

PowerShell:

```powershell
$env:WORKSHOP_DATA_DIR = "D:/datasets/workshop_tools"
```

Bash/WSL:

```bash
export WORKSHOP_DATA_DIR=/path/to/workshop_tools
```

Contoh training dari folder repositori YOLOX, satu GPU NVIDIA, batch 4:

```bash
python tools/train.py -f /path/to/workshop_tools_yolox/workshop_yolox_s.py -d 1 -b 4 -c /path/to/yolox_s.pth
```

Ganti semua `/path/to/...` dengan lokasi sebenarnya. Bobot pretrained YOLOX-S
diunduh terpisah dari tabel model repositori resmi; bobot tidak ada di paket ini.
30 epoch adalah pengaturan awal percobaan, bukan jaminan hasil terbaik.

## Evaluasi test yang memiliki anotasi

Data testing sudah ada di `workshop_tools/test` dan
`workshop_tools/annotations/instances_test.json`. Untuk menggunakannya, pilih split
test lewat variabel berikut lalu jalankan evaluator standar, TANPA flag testdev/--test:

PowerShell:

```powershell
$env:WORKSHOP_EVAL_SPLIT = "test"
python tools/eval.py -f "D:/project/workshop_tools_yolox/workshop_yolox_s.py" -c "YOLOX_outputs/workshop_yolox_s/best_ckpt.pth" -d 1 -b 4
$env:WORKSHOP_EVAL_SPLIT = "val"
```

Bash/WSL:

```bash
WORKSHOP_EVAL_SPLIT=test python tools/eval.py -f /path/to/workshop_tools_yolox/workshop_yolox_s.py -c YOLOX_outputs/workshop_yolox_s/best_ckpt.pth -d 1 -b 4
```

Evaluator menghitung hasil terhadap ground truth test lokal. Pilih model/threshold
dengan validation; gunakan test setelah pilihan tersebut ditetapkan.
Konfigurasi mencegah training ketika WORKSHOP_EVAL_SPLIT masih bernilai test.

## Status verifikasi dan batasan

Script benar-benar sudah dijalankan pada ZIP unggahan: 390 gambar dan 515 bounding box
berhasil disiapkan. Data gambar tetap byte-identik. Semua JSON hasil memiliki tepat
tiga kategori yang sama. Sintaks script dan experiment sudah diperiksa.
Training/inference YOLOX GPU belum dijalankan pada lingkungan ini.
Ketepatan semantik label dan near-duplicate visual belum diperiksa satu per satu.
Dataset ini tidak memiliki gambar negatif; tambahkan gambar tanpa objek target untuk
menguji false positive. Test 10 gambar hanya cukup untuk pemeriksaan awal pipeline.

Sumber dataset: https://universe.roboflow.com/yj-ebtzo/workshop-tools-3/dataset/2
Sumber format/loader: https://github.com/Megvii-BaseDetection/YOLOX
